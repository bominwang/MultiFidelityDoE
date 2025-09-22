import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from typing import List
import matplotlib.pyplot as plt


def standard_sampling(dim: int, num_samples: int, seed: int = None):
    # 保准拉丁超立方采样
    lhs_sampler = LatinHypercube(d=dim, optimization='random-cd', seed=seed)
    lhs_set = lhs_sampler.random(n=num_samples)
    return lhs_set


def subset_sampling(original_samples: np.ndarray, num_subset: int, method: str = 'random'):
    # 从样本中选择子集
    size, dim = original_samples.shape
    if method == 'random':
        indices = np.random.choice(size, num_subset, replace=False)
        samples = original_samples[indices, :]
    elif method == 'cluster':
        metric = "euclidean"
        k_means = KMedoids(n_clusters=num_subset, metric=metric, init="k-medoids++")
        k_means.fit(original_samples)
        indices = np.asarray(k_means.medoid_indices_, dtype=int)
        samples = original_samples[indices, :]
    elif method == 'maxmin':
        indices = np.empty(num_subset, dtype=int)
        indices[0] = np.random.randint(0, size)
        for i in range(1, num_subset):
            remaining_indices = [j for j in range(size) if j not in indices[:i]]
            distances = cdist(original_samples[remaining_indices], original_samples[indices[:i]], metric='euclidean')
            min_distances = np.min(distances, axis=1)
            max_min_distance_idx = np.argmax(min_distances)
            indices[i] = remaining_indices[max_min_distance_idx]
        samples = original_samples[indices, :]
    else:
        raise ValueError(f"Invalid subset selection method: {method}")
    return samples


def expand_sampling(lhs_set: np.ndarray, add_num_samples: int):
    # 扩展拉丁超立方采样方法
    lhs_sampler = LatinHypercube(d=lhs_set.shape[1], optimization='random-cd')
    new_samples = lhs_sampler.random(n=add_num_samples)
    return np.vstack([lhs_set, new_samples])


def separate_samples(original_samples: np.ndarray, augmented_samples: np.ndarray, tol: float = 1e-12):
    # 样本分离
    diff = np.abs(augmented_samples[:, None, :] - original_samples[None, :, :])
    index = np.any(np.all(diff < tol, axis=2), axis=1)
    samples = augmented_samples[~index]
    return samples


class Sampler:
    """
    Unified Multi-Fidelity Latin Hypercube Sampler (Nested or Non-Nested)
    """

    def __init__(self, dim: int, fidelity_level: int, low_bounds: np.ndarray, high_bounds: np.ndarray,
                 mode: str = 'nested', subset_method: str = 'random'):
        """
        Initialize the sampler.

        Parameters:
        - dim: Number of dimensions
        - fidelity_level: Number of fidelity levels
        - low_bounds: Lower bounds for each dimension
        - high_bounds: Upper bounds for each dimension
        - mode: 'nested' or 'non-nested' sampling
        - subset_method: Subset selection method for nested sampling ('random', 'cluster', 'maxmin')
        - seed: Random seed for reproducibility
        """
        self.dim = dim
        self.fidelity_level = fidelity_level
        self.low_bounds = np.array(low_bounds)
        self.high_bounds = np.array(high_bounds)
        self.mode = mode.lower()
        self.subset_method = subset_method.lower()
        if self.mode not in ['nested', 'non-nested']:
            raise ValueError("Mode must be 'nested' or 'non-nested'")
        if self.mode == 'nested' and self.subset_method not in ['random', 'cluster', 'maxmin']:
            raise ValueError("Subset method must be 'random', 'cluster', or 'maxmin' for nested sampling")

    def __sampling__(self, num_samples: List[int]):
        samples = []
        for i in range(self.fidelity_level):
            if i == 0:
                samples.append(standard_sampling(dim=self.dim, num_samples=num_samples[i]))
            else:
                if self.mode == 'nested':
                    samples.append(subset_sampling(original_samples=samples[i - 1], num_subset=num_samples[i],
                                                   method=self.subset_method))
                else:
                    # Non-nested: expand and separate
                    augmented = expand_sampling(lhs_set=samples[i - 1], add_num_samples=num_samples[i])
                    samples.append(separate_samples(original_samples=samples[i - 1], augmented_samples=augmented))
        return samples

    def __mapping__(self, samples: np.ndarray):
        # Map samples from [0,1] to the specified bounds
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] * (self.high_bounds[i] - self.low_bounds[i])) + self.low_bounds[i]
        return samples

    def __separating__(self, samples_set):
        samples = {}
        for i in range(self.fidelity_level):
            key = f"fidelity_{i}"
            samples[key] = self.__mapping__(samples=samples_set[i])
        return samples

    def __visualize__(self, samples: dict):
        if self.dim != 2:
            print("Visualization is only supported for 2D data (dim=2)")
            return
        plt.figure(figsize=(8, 6), facecolor='white')
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        markers = ['o', 's', '^', 'D', '*']
        sizes = np.linspace(50, 100, self.fidelity_level)

        for i in range(self.fidelity_level):
            key = f"fidelity_{i}"
            data = samples[key]
            plt.scatter(data[:, 0], data[:, 1], c=colors[i % len(colors)],
                        marker=markers[i % len(markers)], s=sizes[i],
                        alpha=0.7, edgecolors='white', linewidth=1.0, label=f"Fidelity {i} ({len(data)})")
            if self.mode == 'nested' and i > 0:
                key_prev = f"fidelity_{i - 1}"
                for point in samples[key]:
                    distances = cdist([point], samples[key_prev])
                    closest_idx = np.argmin(distances)
                    closest = samples[key_prev][closest_idx]
                    plt.plot([point[0], closest[0]], [point[1], closest[1]],
                             c='gray', alpha=0.3, linestyle='--')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"{'Nested' if self.mode == 'nested' else 'Non-Nested'} LHS Sampling")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.tight_layout()
        plt.show()

    def __call__(self, num_samples: List[int], verbose: bool = False):
        samples_set = self.__sampling__(num_samples=num_samples)
        samples = self.__separating__(samples_set=samples_set)
        if verbose and self.dim == 2:
            self.__visualize__(samples)
        return samples
