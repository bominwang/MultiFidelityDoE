import numpy as np
from scipy.stats.qmc import LatinHypercube
from expandLHS import ExpandLHS
from typing import List
import matplotlib.pyplot as plt


def standard_sampling(dim: int, num_samples: int):
    # 标准 LHS
    lhs_sampler = LatinHypercube(d=dim, optimization='random-cd')
    lhs_set = lhs_sampler.random(n=num_samples)
    return lhs_set


def expand_sampling(lhs_set: np.ndarray, add_num_samples: int):
    # 扩展 LHS
    lhs_sampler = ExpandLHS(samples=lhs_set)
    lhs_set = lhs_sampler(M=add_num_samples, optimize='discrepancy')
    return lhs_set


def separate_samples(original_samples: np.ndarray, augmented_samples: np.ndarray, tol: float = 1e-12):
    diff = np.abs(augmented_samples[:, None, :] - original_samples[None, :, :])
    index = np.any(np.all(diff < tol, axis=2), axis=1)
    samples = augmented_samples[~index]
    return samples


class Sampler:
    """
    非嵌套多可信度 LHS 采样器
    """

    def __init__(self, dim: int, fidelity_level: int, low_bounds: np.ndarray, high_bounds: np.ndarray, seed: int = 0):
        self.seed = seed
        self.dim = dim
        self.fidelity_level = fidelity_level
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds

    def __sampling__(self, num_samples: List[int]):
        samples = []
        for i in range(self.fidelity_level):
            if i == 0:
                samples.append(standard_sampling(dim=self.dim, num_samples=num_samples[i]))
            else:
                samples.append(expand_sampling(lhs_set=samples[i - 1], add_num_samples=num_samples[i]))
        return samples

    def __mapping__(self, samples: np.ndarray):
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] * (self.high_bounds[i] - self.low_bounds[i])) + self.low_bounds[i]
        return samples

    def __separating__(self, samples_set):
        samples = {}
        for i in range(self.fidelity_level):
            key = f"fidelity_{i}"
            if i == 0:
                samples[key] = self.__mapping__(samples=samples_set[i])
            else:
                samples[key] = self.__mapping__(samples=separate_samples(original_samples=samples_set[i - 1],
                                                                         augmented_samples=samples_set[i]))
        return samples

    def __visualize__(self, samples: dict):
        if self.dim != 2:
            print("可视化仅支持二维数据 (dim=2)")
            return
        plt.figure(figsize=(8, 6))
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i in range(self.fidelity_level):
            key = f"fidelity_{i}"
            data = samples[key]
            plt.scatter(data[:, 0], data[:, 1], c=colors[i % len(colors)], label=key, alpha=0.6, s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()  # 显示图片

    def __call__(self, num_samples: List[int], verbose: bool = False):
        samples_set = self.__sampling__(num_samples=num_samples)
        samples = self.__separating__(samples_set=samples_set)
        if verbose:
            if self.dim == 2:
                self.__visualize__(samples)
        return samples
