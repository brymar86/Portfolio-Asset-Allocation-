# Hierarchical Risk Parity (HRP) Portfolio Optimizer

This document describes the **Hierarchical Risk Parity (HRP)** portfolio construction method as developed by Marcos Lopez de Prado. HRP uses hierarchical clustering to construct diversified portfolios without inverting the covariance matrix, making it more stable than traditional mean-variance optimization.

**IMPORTANT ATTRIBUTION**: This implementation is based on the research of Marcos Lopez de Prado. The original algorithm and mathematical foundations are from his 2016 paper.

**References**:
- De Prado, M. L. (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59-69.
- DOI: https://doi.org/10.3905/jpm.2016.42.4.059
- De Prado's book: "Advances in Financial Machine Learning" (2018)

---

## 1) Overview and Motivation

### 1.1 The Problem with Mean-Variance Optimization

Traditional mean-variance optimization (MVO) has several limitations:

1. **Matrix Inversion Instability**: Requires inverting the covariance matrix, which is numerically unstable when:
   - The number of assets approaches the number of observations
   - Assets are highly correlated (near-singular matrix)
   - Estimation errors are present

2. **Estimation Error Sensitivity**: Small errors in covariance estimates lead to large errors in optimal weights

3. **Poor Out-of-Sample Performance**: Optimized portfolios often perform poorly on new data due to overfitting

### 1.2 HRP's Solution

HRP addresses these issues by:

1. **Avoiding Matrix Inversion**: Uses hierarchical clustering and recursive allocation instead
2. **Leveraging Correlation Structure**: Uses clustering to identify asset relationships
3. **Robust Allocation**: Allocates risk (not returns) recursively down a clustering tree
4. **Mathematically Sound**: Based on proper distance metrics and topological structure

---

## 2) Mathematical Foundation

### 2.1 Notation

- $n$: Number of assets
- $T$: Number of time periods (observations)
- $\mathbf{R} \in \mathbb{R}^{T \times n}$: Returns matrix (rows = time periods, columns = assets)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$: Covariance matrix
- $\boldsymbol{\rho} \in \mathbb{R}^{n \times n}$: Correlation matrix
- $\mathbf{w} \in \mathbb{R}^n$: Portfolio weights vector
- $\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$: Portfolio variance

### 2.2 Correlation to Distance Conversion

HRP converts the correlation matrix to a distance matrix for clustering:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

where $\rho_{ij}$ is the correlation between assets $i$ and $j$.

**Properties**:
- $d_{ij} = 0$ when $\rho_{ij} = 1$ (perfectly correlated)
- $d_{ij} = \sqrt{2}$ when $\rho_{ij} = 0$ (uncorrelated)
- $d_{ij} = 2$ when $\rho_{ij} = -1$ (perfectly anti-correlated)

This creates a **proper metric space** that satisfies the triangle inequality, unlike correlation coefficients.

---

## 3) The HRP Algorithm: Four Steps

### Step 1: Convert Correlation to Distance Matrix

Convert correlation matrix to distance matrix:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

**Implementation**: `distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))`

### Step 2: Build Hierarchical Clustering Tree

Build a hierarchical clustering tree using the distance matrix. Common linkage methods:
- **Ward** (default): Minimizes within-cluster variance
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance between clusters
- **Average**: Average distance between clusters

The clustering tree captures the structure of asset relationships.

### Step 3: Quasi-Diagonalize the Covariance Matrix

Reorder the covariance matrix based on the clustering tree structure:

$$\boldsymbol{\Sigma}_{\text{quasi}} = \boldsymbol{\Sigma}[order, order]$$

where `order` is the order of assets from the dendrogram (tree leaves).

This creates a **quasi-diagonal** structure where similar assets (according to the tree) are placed near each other.

### Step 4: Recursively Allocate Risk

Recursively split the portfolio and allocate weights using inverse variance weighting:

For a split into left and right sub-portfolios with variances $\sigma_L^2$ and $\sigma_R^2$:

$$\alpha = \frac{\sigma_R^2}{\sigma_L^2 + \sigma_R^2}$$

Then combine weights:

$$\mathbf{w} = [(1-\alpha)\mathbf{w}_L, \alpha\mathbf{w}_R]$$

This ensures **risk parity** between sub-portfolios: lower variance sub-portfolios receive higher weights.

---

## 4) Detailed Algorithm Description

### 4.1 Recursive Allocation Formula

The core HRP allocation formula is:

$$\alpha = \frac{\sigma_R^2}{\sigma_L^2 + \sigma_R^2}$$

where:
- $\sigma_L^2 = \mathbf{w}_L^T \boldsymbol{\Sigma}_L \mathbf{w}_L$ (variance of left sub-portfolio)
- $\sigma_R^2 = \mathbf{w}_R^T \boldsymbol{\Sigma}_R \mathbf{w}_R$ (variance of right sub-portfolio)

**Key Insight**: This is **inverse variance weighting**. Lower variance sub-portfolios get higher weights, ensuring risk parity.

### 4.2 Finding the Split Point

For each node in the tree, find the optimal split point by minimizing:

$$\min_{i} \left\{ \frac{i}{n} \sigma_L^2(i) + \frac{n-i}{n} \sigma_R^2(i) \right\}$$

where:
- $i$ is the split index
- $\sigma_L^2(i)$ and $\sigma_R^2(i)$ are variances of left and right sub-portfolios with equal weights

### 4.3 Base Case

When a sub-portfolio contains only one asset, return weight = 1.0 (base case of recursion).

---

## 5) Covariance Matrix Denoising (Optional)

HRP supports optional covariance matrix denoising using Random Matrix Theory to remove noise from the eigenvalue spectrum.

### 5.1 Enabling Denoising

```python
from portfolios.portfolio_src import HierarchicalRiskParity

hrp = HierarchicalRiskParity(
    denoise=True,  # Enable denoising
    denoising_method="constant_residual"  # or "targeted_shrinkage", "eigenvalue_clipping"
)
```

### 5.2 Denoising Process

When denoising is enabled:

1. **Compute Matrices**: Both covariance and correlation matrices computed from returns
2. **Extract Volatilities**: Standard deviations extracted from covariance matrix diagonal
3. **Denoise Correlation**: Correlation matrix is denoised using the selected method (recommended: MP assumes isotropic noise)
4. **Rescale to Covariance**: Denoised correlation rescaled to covariance: $\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R}_{\text{denoised}} \mathbf{D}$
5. **Optimization**: Proceeds with both denoised covariance and correlation matrices

**Why Denoise Correlation?** Marcenko-Pastur theory assumes isotropic noise ($\sigma^2\mathbf{I}$), which correlation matrices better satisfy. This approach is theoretically sound and recommended for production use.

**When to Use Denoising**:
- High-dimensional portfolios (many assets relative to observations)
- Noisy return data
- Unstable optimization results
- When $T/N < 3$ (few observations per asset)

See [Denoising.md](Denoising.md) for detailed documentation on the denoising methodology.

---

## 6) Implementation Details

### 6.1 API Usage

```python
from portfolios.portfolio_src import HierarchicalRiskParity
import pandas as pd

# Prepare returns DataFrame
returns_df = pd.DataFrame(...)  # Rows = time periods, Columns = assets

# Initialize HRP optimizer
hrp = HierarchicalRiskParity(
    linkage_method='ward',  # Clustering linkage method
    denoise=False,  # Optional: enable covariance denoising
    denoising_method="constant_residual"  # If denoising enabled
)

# Fit the optimizer
hrp.fit(returns_df)

# Get portfolio weights
weights = hrp.predict()

# Get portfolio performance
expected_return, volatility, sharpe = hrp.portfolio_performance()
```

### 6.2 Parameters

- **linkage_method** (str): Linkage method for hierarchical clustering. Options: 'ward' (default), 'single', 'complete', 'average'
- **denoise** (bool): If True, apply covariance matrix denoising. Default: False
- **denoising_method** (str): Denoising method to use when denoise=True. Options: 'constant_residual' (default), 'targeted_shrinkage', 'eigenvalue_clipping'

### 6.3 Attributes (After fit())

- **weights_** (np.ndarray): Portfolio weights for each asset
- **cov_matrix_** (pd.DataFrame): Covariance matrix (possibly denoised)
- **corr_matrix_** (pd.DataFrame): Correlation matrix
- **linkage_matrix_** (np.ndarray): Linkage matrix from hierarchical clustering
- **tree_order_** (np.ndarray): Order of assets after quasi-diagonalization
- **cov_quasi_diag_** (pd.DataFrame): Quasi-diagonalized covariance matrix

---

## 7) Comparison with Other Methods

### 7.1 HRP vs Mean-Variance Optimization (MVO)

| Aspect | MVO | HRP |
|--------|-----|-----|
| Matrix Inversion | Required | Avoided |
| Numerical Stability | Unstable | Stable |
| Out-of-Sample Performance | Often poor | Better |
| Return Estimates | Required | Not used |
| Risk Focus | Mean-variance trade-off | Pure risk parity |

**HRP Advantages**:
- More stable (no matrix inversion)
- Better out-of-sample performance
- Doesn't require return estimates
- Robust to estimation errors

### 7.2 HRP vs Risk Parity

| Aspect | Risk Parity | HRP |
|--------|-------------|-----|
| Allocation Method | Direct optimization | Hierarchical clustering |
| Clustering | None | Uses correlation structure |
| Computational Complexity | Optimization problem | Tree traversal |
| Robustness | Good | Excellent |

**HRP Advantages**:
- Leverages correlation structure via clustering
- More intuitive (hierarchical structure)
- Better diversification across asset groups

### 7.3 HRP vs Return-Enhanced HRP (RE-HRP)

HRP and RE-HRP share the same clustering structure, but differ in allocation:

| Aspect | HRP | RE-HRP |
|--------|-----|--------|
| Clustering | Correlation distance | Same (preserved) |
| Allocation Metric | Inverse variance | Information Ratio (default), Sharpe, or Sortino |
| Return Information | Not used | Used |
| Focus | Pure risk parity | Risk-adjusted return maximization |

**When to Use HRP**: When you want pure risk parity without return information.

**When to Use RE-HRP**: When you want to incorporate return information while preserving HRP's clustering structure.

---

## 8) Code-to-Math Mapping

| Mathematical Concept | Code Implementation |
|---------------------|---------------------|
| Correlation to distance: $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ | `distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))` |
| Hierarchical clustering | `linkage(condensed_distances, method=self.linkage_method)` |
| Tree order (dendrogram leaves) | `leaves_list(self.linkage_matrix_)` |
| Quasi-diagonalization | `cov_matrix.iloc[order, :].iloc[:, order]` |
| Portfolio variance: $\sigma^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$ | `var = weights.T @ cov_matrix @ weights` |
| Allocation: $\alpha = \frac{\sigma_R^2}{\sigma_L^2 + \sigma_R^2}$ | `alpha = var_right / (var_left + var_right)` |
| Weight combination: $w = [(1-\alpha)w_L, \alpha w_R]$ | `weights = np.concatenate([(1-alpha)*w_left, alpha*w_right])` |

---

## 9) Advantages and Limitations

### 9.1 Advantages

1. **No Matrix Inversion**: Avoids numerical instability
2. **Robust to Estimation Errors**: Less sensitive to covariance estimation errors
3. **Better Out-of-Sample Performance**: More stable than MVO
4. **No Return Estimates Required**: Pure risk-based approach
5. **Leverages Correlation Structure**: Uses clustering to capture asset relationships
6. **Computationally Efficient**: Tree traversal is fast
7. **Intuitive**: Hierarchical structure is easy to visualize

### 9.2 Limitations

1. **Ignores Return Information**: Doesn't incorporate expected returns (see RE-HRP for extension)
2. **Clustering Dependency**: Results depend on correlation structure
3. **No Explicit Risk Target**: Doesn't target specific risk levels
4. **Linkage Method Sensitivity**: Different linkage methods can produce different results

---

## 10) Best Practices

### 10.1 When to Use HRP

- **Recommended**: When you want robust, risk-parity-based portfolios
- **Recommended**: When return estimates are unreliable or unavailable
- **Recommended**: For high-dimensional portfolios with limited data
- **Recommended**: When matrix inversion is problematic

### 10.2 Parameter Selection

- **linkage_method**: Use 'ward' (default) for most cases. Try others if clustering seems suboptimal.
- **denoise**: Enable when $T/N < 3$ or experiencing instability
- **denoising_method**: Use 'constant_residual' (default) unless you have specific requirements

### 10.3 Validation

- Compare results with and without denoising
- Visualize the dendrogram to understand clustering structure
- Check portfolio weights for reasonableness
- Monitor out-of-sample performance

---

## 11) Conclusion

Hierarchical Risk Parity is a robust portfolio construction method that avoids the numerical instability of mean-variance optimization while leveraging the correlation structure of assets through hierarchical clustering. By allocating risk recursively down a clustering tree using inverse variance weighting, HRP provides stable, diversified portfolios that perform well out-of-sample.

The optional covariance matrix denoising feature further enhances robustness by removing noise from the eigenvalue spectrum, making HRP particularly effective for high-dimensional portfolios with limited historical data.

HRP is recommended for investors seeking robust, risk-parity-based portfolios without relying on return forecasts, and serves as the foundation for extensions like Return-Enhanced HRP that incorporate return information while preserving the clustering structure.


