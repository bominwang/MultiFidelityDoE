# Unified Multi-Fidelity Latin Hypercube Sampler

> 一个用于**单/多保真**（Nested/Non-Nested）实验设计的轻量采样工具，基于 `SciPy` 的 LHS、`scikit-learn-extra` 的 KMedoids 聚类与若干几何启发式策略，支持可视化与自定义边界映射。

---

## ✨ 功能特性

* **标准 LHS**：基于 `scipy.stats.qmc.LatinHypercube`，支持随机坐标下降优化（`optimization='random-cd'`）。
* **多保真策略**

  * **嵌套（nested）**：高保真样本为低保真样本的**子集**（支持 `random` / `cluster` / `maxmin` 三种子集选择）。
  * **非嵌套（non-nested）**：逐层**扩展**并**去重**，各层相互独立。
* **多方法子集选取**

  * `random`：随机子集。
  * `cluster`：K-Medoids 聚类中心作为代表点（更均匀）。
  * `maxmin`：Max-Min 距离贪心，增强空间分散性。
* **边界映射**：将 $0,1$^d 样本线性映射到给定的物理区间。
* **二维可视化**：直观展示不同保真层，并可在嵌套模式下显示“父-子”连线。

---

## 🧩 依赖环境

* Python 3.8+
* `numpy`
* `scipy`（含 `qmc` 与 `spatial.distance`）
* `matplotlib`
* `scikit-learn-extra`（用于 `KMedoids`，仅在 `subset_method='cluster'` 时需要）

安装示例：

```bash
pip install numpy scipy matplotlib scikit-learn-extra
```

> ⚠️ 如果你不使用 `cluster` 模式，可不安装 `scikit-learn-extra`。

---

## 🚀 快速上手

```python
import numpy as np
from sampler import Sampler  # 假设你的文件名为 sampler.py

dim = 2
fidelity_level = 3
low_bounds  = np.array([0.0, -1.0])
high_bounds = np.array([1.0,  1.0])

# 选择模式：'nested' 或 'non-nested'
sampler = Sampler(
    dim=dim,
    fidelity_level=fidelity_level,
    low_bounds=low_bounds,
    high_bounds=high_bounds,
    mode='nested',
    subset_method='maxmin',  # 'random' / 'cluster' / 'maxmin'
)

# 每一层的样本量（从低到高保真）
num_samples = [60, 20, 8]

# 生成样本；若 dim==2 且 verbose=True，会自动弹出可视化图
samples = sampler(num_samples=num_samples, verbose=True)

# samples 是字典：
# {
#   'fidelity_0': ndarray(#60, 2),
#   'fidelity_1': ndarray(#20, 2),
#   'fidelity_2': ndarray(#8,  2),
# }
```

---

## 🧭 使用示例

### 1）嵌套（Nested）采样 + 三种子集策略

```python
# random 子集
sampler_random = Sampler(dim, 3, low_bounds, high_bounds, mode='nested', subset_method='random')
nested_random = sampler_random([60, 20, 8], verbose=False)

# cluster 子集（需要 scikit-learn-extra）
sampler_cluster = Sampler(dim, 3, low_bounds, high_bounds, mode='nested', subset_method='cluster')
nested_cluster = sampler_cluster([60, 20, 8], verbose=False)

# maxmin 子集（空间分散性更强）
sampler_maxmin = Sampler(dim, 3, low_bounds, high_bounds, mode='nested', subset_method='maxmin')
nested_maxmin = sampler_maxmin([60, 20, 8], verbose=True)
```

### 2）非嵌套（Non-Nested）采样

```python
sampler_non_nested = Sampler(dim, 3, low_bounds, high_bounds, mode='non-nested')
non_nested = sampler_non_nested([40, 40, 40], verbose=True)
```

---

## 🧪 独立函数（Functional API）

如需在类外直接使用模块化函数：

```python
lhs = standard_sampling(dim=3, num_samples=100, seed=42)

subset = subset_sampling(original_samples=lhs, num_subset=20, method='maxmin')

lhs_expanded = expand_sampling(lhs_set=lhs, add_num_samples=50)

# 将新扩展的样本中与原样本“重复/近似重复”的点剔除
new_only = separate_samples(original_samples=lhs, augmented_samples=lhs_expanded, tol=1e-12)
```

---

## 🧱 类与方法说明

### `Sampler` 类

* **初始化**

  * `dim (int)`：维度 d
  * `fidelity_level (int)`：保真层数 L
  * `low_bounds (np.ndarray)`：每一维下界（长度为 d）
  * `high_bounds (np.ndarray)`：每一维上界（长度为 d）
  * `mode (str)`：`'nested'` 或 `'non-nested'`
  * `subset_method (str)`：嵌套模式下的子集方法：`'random' | 'cluster' | 'maxmin'`

* **调用**

  * `__call__(num_samples: List[int], verbose: bool=False) -> dict`

    * `num_samples`：长度为 `fidelity_level` 的列表，表示各层样本数（从低到高保真
