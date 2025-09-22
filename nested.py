import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def standard_sampling(dim: int, num_samples: int):
    # 标准 LHS
    lhs_sampler = LatinHypercube(d=dim, optimization='random-cd')
    lhs_set = lhs_sampler.random(n=num_samples)
    return lhs_set


def subset_sampling(original_samples: np.ndarray, num_subset: int, method: str = 'random'):
    """
    从样本中选择子集
    """
    size, dim = original_samples.shape
    if method == 'random':
        # 使用随机选择
        indices = np.random.choice(size, num_subset, replace=False)
        samples = original_samples[indices, :]
    elif method == 'cluster':
        # 使用 K-Means++ 聚类选择 num_subset 个中心点
        metric = "euclidean"
        k_means = KMedoids(n_clusters=num_subset, metric=metric, init="k-medoids++")
        k_means.fit(original_samples)
        indices = np.asarray(k_means.medoid_indices_, dtype=int)
        samples = original_samples[indices, :]
    elif method == 'maxmin':
        indices = np.empty(num_subset, dtype=int)
        indices[0] = np.random.randint(0, size)
        for i in range(1, num_subset):
            # 计算剩余点到已选择点的最小距离
            remaining_indices = [j for j in range(size) if j not in indices[:i]]
            distances = cdist(original_samples[remaining_indices], original_samples[indices[:i]], metric='euclidean')
            min_distances = np.min(distances, axis=1)
            # 选择最小距离最大的点
            max_min_distance_idx = np.argmax(min_distances)
            indices[i] = remaining_indices[max_min_distance_idx]
        samples = original_samples[indices, :]
    else:
        raise ValueError(f"Invalid subset selection method: {method}")
    return samples


class Sampler:
    """
    嵌套多可信度 LHS 采样器
    """

    def __init__(self, dim: int, fidelity_level: int, low_bounds: np.ndarray, high_bounds: np.ndarray,
                 method: str = 'random'):
        self.dim = dim
        self.fidelity_level = fidelity_level
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.method = method

    def __sampling__(self, num_samples: List[int]):
        samples = []
        for i in range(self.fidelity_level):
            if i == 0:
                samples.append(standard_sampling(dim=self.dim, num_samples=num_samples[i]))
            else:
                samples.append(subset_sampling(original_samples=samples[i-1], num_subset=num_samples[i],
                                               method=self.method))
        return samples

    def __mapping__(self, samples: np.ndarray):
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
            print("可视化仅支持二维数据 (dim=2)")
            return
        plt.figure(figsize=(10, 8))
        cmap = cm.get_cmap('viridis', self.fidelity_level)
        markers = ['o', 's', '^', 'D', '*', 'v', 'p', 'h']
        sizes = np.linspace(50, 150, self.fidelity_level)

        for i in range(self.fidelity_level):
            key = f"fidelity_{i}"
            data = samples[key]
            # 绘制点：低可信度透明度低，高可信度加边框
            plt.scatter(data[:, 0], data[:, 1], c=[cmap(i)],
                        marker=markers[i % len(markers)],
                        s=sizes[i], alpha=0.5 + 0.3 * (i / self.fidelity_level),
                        edgecolors='black', linewidth=0.5, label=key)

        # 添加嵌套关系连线
        for i in range(1, self.fidelity_level):
            key_curr = f"fidelity_{i}"
            key_prev = f"fidelity_{i - 1}"
            for point in samples[key_curr]:
                distances = cdist([point], samples[key_prev])
                closest_idx = np.argmin(distances)
                closest = samples[key_prev][closest_idx]
                plt.plot([point[0], closest[0]], [point[1], closest[1]],
                         c='gray', alpha=0.3, linestyle='--')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def __call__(self, num_samples: List[int], verbose: bool = False):
        samples_set = self.__sampling__(num_samples=num_samples)
        samples = self.__separating__(samples_set)
        if verbose:
            if self.dim == 2:
                self.__visualize__(samples)
        return samples
