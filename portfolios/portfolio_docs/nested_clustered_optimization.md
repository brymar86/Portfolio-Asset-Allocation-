# Nested Clustered Optimization (NCO) Portfolio Optimizer

This document describes the **Nested Clustered Optimization (NCO)** portfolio construction method with **LaTeX formatted for typical Jupyter/Markdown renderers** using `$...$` (inline) and `$$...$$` (display).

**Core Philosophy**: NCO addresses the fundamental instability of inverting high-dimensional covariance matrices. Even with accurate return forecasts, the numerical instability and estimation error in covariance matrix inversion destroys any potential benefit. NCO preserves statistical information by using proper metric spaces (topological structure) rather than working directly with correlation matrices, which lack topological properties. By clustering assets using distance metrics and reducing dimensionality from $n$ assets to $k$ clusters (where $k \ll n$), NCO dramatically improves out-of-sample stability.

**Notation**: $n$ assets, $k$ clusters ($k \ll n$), covariance matrix $\boldsymbol{\Sigma}$, correlation matrix $\boldsymbol{\rho}$, cluster assignments $C_i$ with asset indices $\mathcal{I}_i$.

---

## 3) The NCO Algorithm: Four Steps

NCO performs optimization in a nested, hierarchical fashion:

1. **Cluster Assets**: Group assets into $k$ clusters using hierarchical clustering on correlation-based distance metrics
2. **Optimize Within Clusters**: For each cluster $C_i$, compute optimal weights $\mathbf{w}_i$ for assets within that cluster
3. **Optimize Between Clusters**: Build a reduced $k \times k$ covariance matrix at the cluster level, then optimize cluster weights
4. **Combine Weights**: Multiply cluster weights by within-cluster weights to get final asset weights: $w_{\text{final},j} = w_C[i] \cdot w_{i,j}$

---

## 4) Step 1: Clustering Assets

### 4.1 Distance Metric and Topological Structure

We convert the correlation matrix $\boldsymbol{\rho}$ into a distance matrix for hierarchical clustering:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

where $\rho_{ij}$ is the correlation between assets $i$ and $j$.

**Why this matters**: Correlation matrices are non-topological subspaces—they don't preserve geometric structure. The distance metric $d_{ij}$ creates a proper metric space that:
- Satisfies the triangle inequality (unlike correlation)
- Preserves statistical information in a topological structure
- Enables meaningful clustering that respects asset relationships

**Properties**:
- $d_{ij} = 0$ when $\rho_{ij} = 1$ (perfectly correlated)
- $d_{ij} = \sqrt{2}$ when $\rho_{ij} = 0$ (uncorrelated)
- $d_{ij} = 2$ when $\rho_{ij} = -1$ (perfectly anti-correlated)

**Implementation**: `distance_matrix = np.sqrt(2 * (1 - self.corr_matrix_.values))` (line ~144)

### 4.2 Hierarchical Clustering

Use hierarchical clustering (e.g., Ward linkage) on the distance matrix to create clusters. The clustering preserves the topological structure of asset relationships.

**Automatic cluster selection**: If $k$ is not specified, use:
$$k = \max(2, \lfloor\sqrt{n}\rfloor)$$

This ensures $k \ll n$ while maintaining diversification. After clustering, each asset $j$ is assigned to a cluster: $\text{cluster}(j) \in \{1, 2, \ldots, k\}$.

---

## 5) Step 2: Within-Cluster Optimization

For each cluster $C_i$ with asset indices $\mathcal{I}_i$:

1. Extract the sub-covariance matrix: $\boldsymbol{\Sigma}_i \in \mathbb{R}^{|\mathcal{I}_i| \times |\mathcal{I}_i|}$
2. Optimize weights $\mathbf{w}_i$ for assets in cluster $i$

**Optimization methods**:

- **Risk Parity** (recommended): Equalize risk contributions within the cluster. This avoids reliance on return forecasts and focuses purely on risk structure.
- **Mean-Variance**: Maximize Sharpe ratio or minimize variance. Note: even with good return forecasts, this step operates on smaller covariance matrices (within clusters), reducing numerical instability.

**Edge cases**: Single asset → weight of 1.0; optimization failure → fall back to equal weights.

---

## 6) Step 3: Between-Cluster Optimization

After optimizing within clusters, treat each cluster as a "super-asset" and optimize allocation between clusters using a reduced $k \times k$ problem instead of the full $n \times n$ problem.

### 6.1 Cluster-Level Covariance Matrix

Build reduced covariance matrix $\boldsymbol{\Sigma}_C$ where:

$$\Sigma_C[i,j] = \mathbf{w}_i^\top \boldsymbol{\Sigma}_{ij} \mathbf{w}_j$$

where $\boldsymbol{\Sigma}_{ij}$ is the submatrix of $\boldsymbol{\Sigma}$ containing covariances between assets in clusters $i$ and $j$, and $\mathbf{w}_i, \mathbf{w}_j$ are within-cluster weights.

**Intuition**: The covariance between two clusters is the weighted average of cross-cluster asset covariances. Since we've already optimized within clusters, we preserve that information while dramatically reducing problem size.

### 6.2 Cluster-Level Optimization

For cluster-level expected returns: $\mu_C[i] = \mathbf{w}_i^\top \boldsymbol{\mu}_i$ (if using mean-variance).

Optimize cluster weights $\mathbf{w}_C \in \mathbb{R}^k$ using mean-variance optimization:

$$\min_{\mathbf{w}_C} \mathbf{w}_C^\top \boldsymbol{\Sigma}_C \mathbf{w}_C \quad \text{or} \quad \max_{\mathbf{w}_C} \frac{\boldsymbol{\mu}_C^\top \mathbf{w}_C}{\sqrt{\mathbf{w}_C^\top \boldsymbol{\Sigma}_C \mathbf{w}_C}}$$

subject to $\mathbf{1}^\top \mathbf{w}_C = 1$, $\mathbf{w}_C \ge 0$.

**Key advantage**: Reduced from $n$ variables to $k$ variables where $k \ll n$, making the optimization both faster and more numerically stable.

---

## 7) Step 4: Combining Weights

Final asset weights are computed by multiplying cluster weights by within-cluster weights:

For asset $j$ in cluster $C_i$:
$$w_{\text{final},j} = w_C[i] \cdot w_{i,j}$$

**Constraint satisfaction**: Since both cluster weights and within-cluster weights sum to 1, final weights automatically sum to 1:
$$\sum_{j=1}^{n} w_{\text{final},j} = \sum_{i=1}^{k} w_C[i] \sum_{j \in \mathcal{I}_i} w_{i,j} = 1$$

**Implementation**: `final_weights[indices] = cluster_weight * within_weights` (line ~448)

---

## 8) The Complete Optimization Problem

The full NCO optimization can be written as a nested problem:

$$\min_{\mathbf{w}_C, \{\mathbf{w}_i\}_{i=1}^{k}} \mathbf{w}_C^\top \boldsymbol{\Sigma}_C(\{\mathbf{w}_i\}) \mathbf{w}_C$$

subject to:
- $\mathbf{1}^\top \mathbf{w}_C = 1$, $\mathbf{w}_C \ge 0$
- For each cluster $i$: $\mathbf{1}^\top \mathbf{w}_i = 1$, $\mathbf{w}_i \ge 0$
- $\boldsymbol{\Sigma}_C[i,j] = \mathbf{w}_i^\top \boldsymbol{\Sigma}_{ij} \mathbf{w}_j$ (coupling constraint)

In practice, we solve sequentially: first optimize $\{\mathbf{w}_i\}$ within each cluster, then optimize $\mathbf{w}_C$ given the within-cluster weights. This is computationally efficient and empirically performs well.

---

## 9) Numerical Optimization Details

### 9.1 Optimization Methods

We use SLSQP for both within-cluster and between-cluster optimization because it:
- Supports equality constraints (sum of weights = 1)
- Supports bound constraints (long-only: $w_i \ge 0$)
- Handles smooth nonlinear objectives
- Works well for small to medium problems

### 9.2 Algorithm Outline

1. Compute $\boldsymbol{\Sigma}$ and $\boldsymbol{\rho}$ from returns
2. Convert correlation to distance: $d_{ij} = \sqrt{2(1-\rho_{ij})}$
3. Perform hierarchical clustering to assign assets to $k$ clusters
4. For each cluster, optimize weights (risk parity or mean-variance)
5. Build cluster covariance matrix $\boldsymbol{\Sigma}_C$
6. Optimize cluster weights $\mathbf{w}_C$
7. Combine: $\mathbf{w}_{\text{final}} = \mathbf{W}_{\text{within}} \mathbf{w}_C$
8. Validate weights and handle edge cases

### 9.3 Numerical Edge Cases

- **Near-singular covariance matrices**: Consider shrinkage $\boldsymbol{\Sigma} + \epsilon \mathbf{I}$
- **Single-asset clusters**: Return weight of 1.0
- **Optimization failures**: Fall back to equal weights
- **Boundary solutions**: Redistribute small weights to prevent corner solutions

---

## 10) Validation Checks

Given final weights $\hat{\mathbf{w}}$:

1. **Constraints**: $\sum_j \hat{w}_j \approx 1$, $\hat{w}_j \ge 0$ for all $j$
2. **Cluster structure**: All assets in exactly one cluster; cluster weights and within-cluster weights each sum to 1
3. **Portfolio metrics**: Compute $\hat{\sigma}_p = \sqrt{\hat{\mathbf{w}}^\top \boldsymbol{\Sigma} \hat{\mathbf{w}}}$ and verify reasonable values
4. **Stability**: Perturb inputs slightly; verify weights don't change dramatically; check for NaNs/infinities

---

## 13) In Practice: Code Implementation Walkthrough

### 13.1 Code-to-Math Mapping

| Mathematical Concept | Code Location | Implementation |
|---------------------|---------------|----------------|
| $d_{ij} = \sqrt{2(1-\rho_{ij})}$ | Line ~144 | `distance_matrix = np.sqrt(2 * (1 - self.corr_matrix_.values))` |
| Hierarchical clustering | Lines ~148-162 | `linkage()` and `fcluster()` from scipy |
| $k = \max(2, \lfloor\sqrt{n}\rfloor)$ | Line ~153 | `max(2, int(np.sqrt(len(self.asset_names_))))` |
| Within-cluster covariance $\boldsymbol{\Sigma}_i$ | Line ~179 | `cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]` |
| Within-cluster optimization | Lines ~182-187 | `_risk_parity_optimize()` or `_mean_variance_optimize()` |
| $\Sigma_C[i,j] = \mathbf{w}_i^\top \boldsymbol{\Sigma}_{ij} \mathbf{w}_j$ | Line ~376 | `cluster_cov[i-1, j-1] = weights_i.T @ cov_ij @ weights_j` |
| $\mu_C[i] = \mathbf{w}_i^\top \boldsymbol{\mu}_i$ | Line ~367 | `cluster_returns[i-1] = expected_returns[indices_i].T @ weights_i` |
| Cluster weight optimization | Lines ~384-412 | Mean-variance optimization with SLSQP |
| $w_{\text{final},j} = w_C[i] \cdot w_{i,j}$ | Line ~448 | `final_weights[indices] = cluster_weight * within_weights` |

### 13.2 Configuration Options

- **`n_clusters`**: Auto-selects $k = \max(2, \lfloor\sqrt{n}\rfloor)$ if `None`
- **`within_cluster_method`**: `'risk_parity'` (recommended, avoids return forecasts) or `'mean_variance'`
- **`linkage_method`**: `'ward'` (default, minimizes within-cluster variance) or other options
- **`target_return`**: Optional constraint for mean-variance optimization

---

## 14) Why This Implementation is Correct

### 14.1 Mathematical Correctness

1. **Distance metric**: $d_{ij} = \sqrt{2(1-\rho_{ij})}$ creates a valid metric space preserving topological structure
2. **Cluster covariance**: $\Sigma_C[i,j] = \mathbf{w}_i^\top \boldsymbol{\Sigma}_{ij} \mathbf{w}_j$ correctly computes portfolio covariance
3. **Weight combination**: Multiplicative combination preserves sum-to-one constraint
4. **Constraint satisfaction**: All optimizations enforce proper constraints

### 14.2 Numerical Robustness

- Edge case handling (single assets, empty clusters, failures)
- Boundary solution prevention (redistribute small weights)
- Appropriate optimization tolerances
- Post-processing validation

**Key Insight**: The code directly implements De Prado's algorithm, translating mathematical theory step-by-step with proper handling of numerical issues.

---

## 15) References

- De Prado, M. L. (2016). *Building Diversified Portfolios that Outperform Out of Sample.* The Journal of Portfolio Management, 42(4), 59-69. DOI: https://doi.org/10.3905/jpm.2016.42.4.059

- De Prado, M. L. (2018). *Advances in Financial Machine Learning.* John Wiley & Sons.

**Additional Resources**: Open-source implementations (`emoen/Machine-Learning-for-Asset-Managers`, `skfolio`, `emialb34i/beyond-markowitz`), related methods (Hierarchical Risk Parity, Risk Parity)

---

## 16) Practical Considerations

### 16.1 Choosing the Number of Clusters

**Default heuristic**: $k = \max(2, \lfloor\sqrt{n}\rfloor)$ ensures $k \ll n$ while maintaining diversification. For example: $n = 100$ → $k = 10$; $n = 25$ → $k = 5$.

**Domain knowledge**: If assets naturally form groups (sectors, regions), choose $k$ to match that structure.

### 16.2 Choosing Within-Cluster Method

**Risk Parity** (recommended): Purely risk-based, avoids return forecast dependencies, more robust to estimation error.

**Mean-Variance**: Can incorporate return forecasts, but operates on smaller matrices reducing instability. Even with good forecasts, remember that covariance matrix inversion instability is the primary concern.

### 16.3 Performance Considerations

- **Clustering**: $O(n^2 \log n)$ one-time cost
- **Within-cluster**: $O(\sum_i |C_i|^3)$ where $|C_i|$ is cluster size
- **Between-cluster**: $O(k^3)$ where $k \ll n$
- **Overall**: Much faster and more stable than $O(n^3)$ full optimization

### 16.4 Stability and Rebalancing

NCO's stability reduces sensitivity to input changes. Cluster assignments may evolve over time; consider rebalancing frequency based on cluster stability and transaction costs. The topological structure preservation makes cluster changes more meaningful than noise.

---

**End of Document**
