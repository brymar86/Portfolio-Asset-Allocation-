# Return-Enhanced Hierarchical Risk Parity (RE-HRP) Portfolio Optimizer

This document describes the **Return-Enhanced Hierarchical Risk Parity (RE-HRP)** portfolio construction method, which extends De Prado's Hierarchical Risk Parity (HRP) algorithm by incorporating return information through risk-adjusted return maximization (default: Information Ratio, with Sharpe Ratio and Sortino Ratio as options).

**IMPORTANT ATTRIBUTION**: This implementation extends the research of Marcos Lopez de Prado. The original HRP algorithm and mathematical foundations are from his 2016 paper. This extension preserves HRP's clustering structure but allocates based on risk-adjusted returns (Information Ratio by default, Sharpe Ratio and Sortino Ratio as options).

---

## 1) Overview and Motivation

RE-HRP preserves HRP's mathematically sound clustering structure (based on correlation distance metrics) while allocating capital based on risk-adjusted returns rather than pure risk parity. The key insight is that HRP's clustering is based on correlation structure, which is independent of returns, so we can preserve this structure while incorporating return information in the allocation step.

**Key Differences from HRP:**

| Aspect | HRP | RE-HRP |
|--------|-----|--------|
| Clustering | Correlation distance | Same (preserved) |
| Allocation Metric | Inverse variance | Information Ratio (default), Sharpe Ratio, or Sortino Ratio |
| Allocation Formula | $\alpha = \frac{\sigma^2_R}{\sigma^2_L + \sigma^2_R}$ | $\alpha = \frac{IR_R}{IR_L + IR_R}$ (default) or $\alpha = \frac{SR_R}{SR_L + SR_R}$ |
| Return Information | Not used | Used (expected returns, benchmark comparison) |
| Risk Measure | Total volatility | Tracking error (Information Ratio), Total volatility (Sharpe), or Downside volatility (Sortino) |

---

## 2) Mathematical Foundation

### 2.1 Clustering Structure (Preserved from HRP)

RE-HRP uses the same correlation-based distance metric as HRP:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

where $\rho_{ij}$ is the correlation coefficient between assets $i$ and $j$. This creates a proper metric space and is mathematically sound. The hierarchical clustering tree is built using this distance matrix, preserving the same structure as HRP.

**Why preserve clustering?** The correlation structure captures the underlying relationships between assets, which is independent of their expected returns. This clustering provides a robust foundation for portfolio construction.

### 2.2 Information Ratio (Default)

The Information Ratio is the default risk-adjusted return measure used in RE-HRP:

$$IR = \frac{E[R_{portfolio}] - E[R_{benchmark}]}{\sigma_{excess}}$$

where:
- $E[R_{portfolio}]$ is the expected portfolio return (annualized)
- $E[R_{benchmark}]$ is the expected benchmark return (annualized)
- $\sigma_{excess}$ is the tracking error (annualized standard deviation of excess returns)
- Excess returns: $R_{excess} = R_{portfolio} - R_{benchmark}$

**Benchmark Definition:**
- Default: Equal-weighted portfolio of all assets
- Custom: User can provide custom benchmark returns

**Why Information Ratio as Default?**
1. **Favors Return Clusters**: Compares excess return vs benchmark, not absolute risk - naturally favors clusters with better return characteristics
2. **Avoids Low-Volatility Bias**: Unlike Sharpe Ratio, Information Ratio doesn't over-weight low-volatility assets with modest returns (like TLT bonds)
3. **Cluster Context**: Within HRP's clustering structure, Information Ratio identifies which clusters generate better excess returns relative to the benchmark
4. **Covariance Information**: Leverages the rich information in covariance matrices to identify return clusters while preserving clustering structure
5. **Real-World Performance**: Leads to better portfolio returns by favoring return clusters rather than just low-risk assets

**Mathematical Properties:**
- Information Ratio measures risk-adjusted excess return relative to a benchmark
- Higher Information Ratios indicate better risk-adjusted performance vs benchmark
- Tracking error captures the volatility of excess returns (how much portfolio deviates from benchmark)
- Formula: $\alpha = \frac{IR_{right}}{IR_{left} + IR_{right}}$ favors clusters with higher Information Ratios

### 2.2.1 Information Ratio Thresholds and Constraints

**Problem: Low-Return Asset Over-weighting**

Information Ratio can favor low-return, low-volatility assets (like TLT bonds) because:
- Low-volatility assets have tiny tracking errors (denominator)
- Small positive excess return ÷ tiny tracking error = high IR
- Example: TLT with 0.1% excess return and 0.05% tracking error gives IR = 2.0, even though absolute return is low

**Solution: Constraint Framework**

The implementation uses a three-layer constraint system to prevent low-return asset over-weighting:

1. **Minimum Return Threshold** (`min_return_threshold`):
   - **Single-asset clusters**: Individual asset return must ≥ threshold
   - **Multi-asset clusters**: Cluster return ≥ threshold AND ≤50% of cluster weight in assets below threshold
   - **Default**: `None` (uses benchmark return as threshold)
   - **Custom**: Absolute minimum return (e.g., 0.05 for 5% annualized)
   - **Mathematical constraint**: $R_{asset} \geq \tau$ for single assets, $R_{cluster} \geq \tau \land \sum_{i: R_i < \tau} w_i \leq 0.5$ for clusters

2. **Tracking Error Floor** (`min_tracking_error`):
   - Minimum tracking error floor (default: 1% annualized)
   - Prevents tiny denominators from inflating IR
   - **Mathematical constraint**: $\sigma_{excess} = \max(\sigma_{excess,calculated}, \min\_tracking\_error)$

3. **Allocation Logic** (4 cases):
   - **Both meet threshold**: Use Information Ratio allocation normally
   - **Only left meets**: Favor left cluster (ratio_left = 1.0, ratio_right = 0.001 if single asset else 0.1)
   - **Only right meets**: Favor right cluster (ratio_left = 0.001 if single asset else 0.1, ratio_right = 1.0)
   - **Neither meets**: Return-based allocation, with single assets below threshold heavily penalized (ratio = 0.001)

**Mathematical Formulation:**

For a split into left and right sub-portfolios:
- Threshold: $\tau = \min\_return\_threshold$ if provided, else $E[R_{benchmark}]$
- Single-asset check: $R_{asset} \geq \tau$ (for single-asset clusters)
- Multi-asset check: $R_{cluster} \geq \tau \land \sum_{i: R_i < \tau} w_i \leq 0.5$ (for multi-asset clusters)
- Allocation: Based on threshold satisfaction, with Information Ratio used when both meet threshold

**Constraint Properties:**
- **Completeness**: Always produces valid allocation (fallback to return-based when thresholds not met)
- **Consistency**: Single assets below threshold get minimal weight (ratio = 0.001) regardless of comparison
- **Preservation**: Information Ratio benefits preserved for return clusters that meet threshold

### 2.3 Sharpe Ratio (Alternative Option)

The Sharpe Ratio is available as an alternative allocation metric:

$$SR = \frac{E[R] - r_f}{\sigma_{total}}$$

where:
- $E[R]$ is the expected return (annualized)
- $r_f$ is the risk-free rate (annualized)
- $\sigma_{total}$ is the total volatility (annualized standard deviation)

**When to Use Sharpe Ratio?**
- When you want absolute risk-adjusted return (not relative to benchmark)
- When you prefer a more traditional metric
- For comparison with other portfolio optimization methods

**Trade-offs:**
- May favor low-volatility assets even with modest returns (similar to Information Ratio without constraints)
- Less effective at identifying return clusters within the hierarchical structure
- Information Ratio with constraints is generally recommended for most use cases

### 2.4 Sortino Ratio (Alternative Option)

The Sortino Ratio is available as an alternative allocation metric:

$$SR_{Sortino} = \frac{E[R] - r_f}{\sigma_{downside}}$$

where:
- $\sigma_{downside}$ is the downside deviation (annualized)

**Downside Deviation:**

$$\sigma_{downside} = \sqrt{E[\min(0, R - target)^2]}$$

where $target$ is typically 0 or the risk-free rate. This only penalizes returns below the target.

**When to Use Sortino Ratio?**
- When you want to focus specifically on downside risk
- When upside volatility is explicitly desirable and should not be penalized
- For investors with asymmetric risk preferences

**Trade-offs:**
- Sortino may be less stable for small sub-portfolios in recursive allocation
- Downside deviation can be noisy with limited data
- Information Ratio with constraints is generally recommended for most use cases

### 2.5 Allocation Formula

In HRP, the allocation between left and right sub-portfolios uses inverse variance weighting:

$$\alpha_{HRP} = \frac{\sigma^2_{right}}{\sigma^2_{left} + \sigma^2_{right}}$$

In RE-HRP, we replace this with risk-adjusted return ratio weighting:

$$\alpha_{RE-HRP} = \frac{IR_{right}}{IR_{left} + IR_{right}}$$

where $IR_{left}$ and $IR_{right}$ are the Information Ratios (default), Sharpe Ratios, or Sortino Ratios of the left and right sub-portfolios, respectively.

**Information Ratio Allocation (Default):**
- Compares each sub-portfolio's excess return vs benchmark
- Favors clusters with higher Information Ratios (better risk-adjusted excess returns)
- Formula: $\alpha = \frac{IR_{right}}{IR_{left} + IR_{right}}$ where $IR = \frac{E[R] - E[R_{benchmark}]}{\sigma_{excess}}$

**Mathematical Justification:**
- Both formulas normalize to sum to 1 (ensuring weights sum to 1)
- Information Ratio provides a proper metric that compares excess return vs benchmark
- Higher Information Ratio sub-portfolios get higher weight via $\alpha$: $w_{final} = [(1-\alpha) \mathbf{w}_{left}, \alpha \mathbf{w}_{right}]$ where $\alpha = \frac{IR_{right}}{IR_{left} + IR_{right}}$
- The recursive structure is preserved (mathematically valid)
- Information Ratio naturally identifies return clusters within the hierarchical structure

---

## 3) Algorithm Description

RE-HRP follows the same steps as HRP, with modifications only in the recursive allocation step:

### Step 1: Convert Correlation to Distance Matrix

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

This creates a proper metric space where highly correlated assets are close together.

### Step 2: Build Hierarchical Clustering Tree

Use hierarchical clustering (e.g., Ward linkage) to build a tree structure from the distance matrix. This tree captures the hierarchical relationships between assets.

### Step 3: Quasi-Diagonalize Covariance Matrix

Reorder the covariance matrix according to the tree structure, placing similar assets (according to the clustering tree) near each other. This creates a quasi-diagonal structure that makes recursive allocation more effective.

### Step 4: Compute Expected Returns

Calculate expected returns for each asset:

$$\mu_i = E[R_i] = \frac{1}{T}\sum_{t=1}^T R_{i,t}$$

where $T$ is the number of time periods.

### Step 5: Set Up Benchmark Returns

For Information Ratio allocation (default):
- If no custom benchmark provided: Use equal-weighted portfolio of all assets
- Calculate benchmark returns: $R_{benchmark} = \frac{1}{n}\sum_{i=1}^n R_i$ (for each time period)
- Store benchmark returns for recursive allocation

### Step 6: Recursive Allocation with Risk-Adjusted Return Ratio

This is where RE-HRP differs from HRP. For each split in the tree:

1. **Split the portfolio** into left and right sub-portfolios
2. **Recursively compute weights** for each sub-portfolio
3. **Calculate portfolio returns** for each sub-portfolio:
   - $R_{left} = \mathbf{w}_{left}^T \mathbf{R}_{left}$ (for each time period)
   - $R_{right} = \mathbf{w}_{right}^T \mathbf{R}_{right}$ (for each time period)
4. **Check return thresholds** (for Information Ratio only):
   - Calculate portfolio returns: $R_{left} = E[\mathbf{w}_{left}^T \mathbf{R}_{left}]$, $R_{right} = E[\mathbf{w}_{right}^T \mathbf{R}_{right}]$
   - Determine threshold: $\tau = \min\_return\_threshold$ if provided, else $E[R_{benchmark}]$
   - Check if returns meet threshold: $R_{left} \geq \tau$, $R_{right} \geq \tau$
5. **Calculate risk-adjusted return ratios** (Information Ratio by default, Sharpe Ratio, or Sortino Ratio if specified):
   - **If both meet threshold (Information Ratio)**: 
     - $IR_{left} = \frac{E[R_{left}] - E[R_{benchmark}]}{\max(\sigma_{excess,left}, \min\_tracking\_error)}$
     - $IR_{right} = \frac{E[R_{right}] - E[R_{benchmark}]}{\max(\sigma_{excess,right}, \min\_tracking\_error)}$
     - where $\sigma_{excess} = \sigma(R_{portfolio} - R_{benchmark})$ (tracking error with floor applied)
   - **If only one meets threshold**: Allocate more to that one (0.9/0.1 split)
   - **If neither meets threshold**: Fall back to return-based allocation (favor higher return)
   - **Sharpe Ratio**: $SR_{left} = \frac{E[R_{left}] - r_f}{\sigma_{total,left}}$, $SR_{right} = \frac{E[R_{right}] - r_f}{\sigma_{total,right}}$
   - **Sortino Ratio**: $SR_{left} = \frac{E[R_{left}] - r_f}{\sigma_{downside,left}}$, $SR_{right} = \frac{E[R_{right}] - r_f}{\sigma_{downside,right}}$
6. **Allocate weights**:
   - $\alpha = \frac{IR_{right}}{IR_{left} + IR_{right}}$ (default, if thresholds met) or $\alpha = \frac{SR_{right}}{SR_{left} + SR_{right}}$
   - **Weight combination**: $w_{final} = [(1-\alpha) \mathbf{w}_{left}, \alpha \mathbf{w}_{right}]$
   - **Note**: Higher $\alpha$ (higher right ratio) → right sub-portfolio gets more weight, preserving the risk-adjusted return maximization objective

**Edge Case Handling:**
- Both ratios ≤ 0: use equal weights ($\alpha = 0.5$)
- One ratio ≤ 0: allocate all to positive one (clamped to $[0.1, 0.9]$ for diversification)
- Zero volatility/deviation: treated as perfect (high ratio)
- Single asset: return weight of 1.0 (base case)

### Step 7: Reorder Weights

Reorder weights back to original asset order (from quasi-diagonal order).

---

## 4) Covariance Matrix Denoising (Optional)

RE-HRP supports optional covariance matrix denoising using Random Matrix Theory to remove noise from the eigenvalue spectrum before optimization.

### 4.1 Enabling Denoising

```python
from portfolios.portfolio_src import ReturnEnhancedHRP

re_hrp = ReturnEnhancedHRP(
    denoise=True,  # Enable denoising
    denoising_method="constant_residual"  # or "targeted_shrinkage", "eigenvalue_clipping"
)
```

### 4.2 Denoising Process

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

The denoising process removes random noise eigenvalues identified using the Marcenko-Pastur distribution bounds, leading to more robust covariance estimates and better portfolio stability.

See [Denoising.md](Denoising.md) for detailed documentation on the denoising methodology, including the three available methods (constant residual, targeted shrinkage, eigenvalue clipping) and mathematical foundations.

### 4.3 Denoising Methods

Three denoising methods are available:

- **constant_residual** (default/standard): Replace random eigenvalues with their average value. Preserves approximate trace and is the standard approach.
- **targeted_shrinkage**: Shrink eigenvalues toward the mean eigenvalue. More conservative denoising.
- **eigenvalue_clipping**: Set random eigenvalues to the Marcenko-Pastur upper bound. More aggressive denoising.

---

## 5) Implementation Details

### 5.1 Information Ratio Calculation (Default)

The implementation calculates Information Ratio as follows:

```python
# Calculate excess returns
excess_returns = portfolio_returns - benchmark_returns

# Calculate expected excess return (per period)
expected_excess_period = np.mean(excess_returns)

# Annualize expected excess return
expected_excess_annual = expected_excess_period * periods_per_year

# Calculate tracking error (standard deviation of excess returns)
tracking_error_period = np.std(excess_returns, ddof=1)

# Annualize tracking error
tracking_error_annual = tracking_error_period * np.sqrt(periods_per_year)

# Apply tracking error floor to prevent tiny denominators from inflating IR
tracking_error_annual = max(tracking_error_annual, min_tracking_error)

# Calculate Information Ratio
information_ratio = expected_excess_annual / tracking_error_annual
```

**Edge Cases:**
- Zero tracking error: return large positive value if excess return > 0, large negative otherwise
- Negative excess return: return negative Information Ratio (acceptable, will get lower weight)
- Tiny tracking error: Floor applied to prevent IR inflation for low-volatility assets

**Benchmark Setup:**
- Default: Equal-weighted portfolio of all assets
- Custom: User can provide custom benchmark returns via `benchmark_returns` parameter

### 5.2 Sharpe Ratio Calculation (Alternative)

The implementation calculates Sharpe Ratio as follows:

```python
# Calculate expected return (per period)
expected_return_period = np.mean(portfolio_returns)

# Annualize expected return
expected_return_annual = expected_return_period * periods_per_year

# Calculate total volatility (standard deviation)
volatility_period = np.std(portfolio_returns, ddof=1)

# Annualize volatility
volatility_annual = volatility_period * np.sqrt(periods_per_year)

# Calculate Sharpe Ratio
sharpe_ratio = (expected_return_annual - risk_free_rate) / volatility_annual
```

**Edge Cases:**
- Zero volatility: return large positive value if return > risk-free rate, large negative otherwise
- Negative expected return: return negative Sharpe (acceptable, will get lower weight)

### 5.3 Sortino Ratio Calculation (Alternative)

The implementation calculates Sortino Ratio as follows:

```python
# Calculate expected return (per period)
expected_return_period = np.mean(portfolio_returns)

# Annualize expected return
expected_return_annual = expected_return_period * periods_per_year

# Calculate downside deviation
downside_returns = np.minimum(0, portfolio_returns - target_return)
downside_variance = np.mean(downside_returns ** 2)
downside_deviation_period = np.sqrt(downside_variance)

# Annualize downside deviation
downside_deviation_annual = downside_deviation_period * np.sqrt(periods_per_year)

# Calculate Sortino Ratio
sortino_ratio = (expected_return_annual - risk_free_rate) / downside_deviation_annual
```

**Edge Cases:**
- Zero downside deviation: return large positive value (perfect downside protection)
- Negative expected return: return negative Sortino (acceptable, will get lower weight)

### 4.4 Recursive Allocation with Constraints

The recursive allocation function implements the following algorithm:

1. **Base case**: Single asset → return weight of 1.0
2. **Split**: Find optimal split point (same as HRP)
3. **Recurse**: Compute weights for left and right sub-portfolios
4. **Constraint checking** (for Information Ratio only):
   - **Single-asset clusters**: Check individual asset return $R_{asset} \geq \tau$
   - **Multi-asset clusters**: Check cluster return $R_{cluster} \geq \tau$ AND $\sum_{i: R_i < \tau} w_i \leq 0.5$
   - **Threshold**: $\tau = \min\_return\_threshold$ if provided, else $E[R_{benchmark}]$
5. **Calculate risk-adjusted ratios**: Information Ratio (default), Sharpe Ratio, or Sortino Ratio for each sub-portfolio
   - **If Information Ratio**: Apply tracking error floor $\sigma_{excess} = \max(\sigma_{excess}, \min\_tracking\_error)$
   - **Allocation cases**:
     - Both meet threshold: Use Information Ratio allocation
     - Only one meets: Favor that cluster, heavily penalize single assets below threshold (ratio = 0.001)
     - Neither meets: Return-based allocation, heavily penalize single assets below threshold
6. **Allocate**: Use risk-adjusted return ratio weighting: $\alpha = \frac{ratio_{right}}{ratio_{left} + ratio_{right}}$
7. **Combine weights**: $w_{final} = [(1-\alpha) \mathbf{w}_{left}, \alpha \mathbf{w}_{right}]$
8. **Logging**: If `verbose=True`, print threshold checks and allocation decisions at each recursive step

**Constraint Enforcement:**
- Single assets below threshold receive ratio = 0.001 (minimal weight)
- Multi-asset clusters with >50% weight in low-return assets are treated as not meeting threshold
- Always produces valid allocation (fallback to return-based when thresholds not met)

### 4.5 Parameters

- `linkage_method` (str): Linkage method for hierarchical clustering (default: 'ward')
- `risk_free_rate` (float): Annual risk-free rate for risk-adjusted ratio calculation (default: 0.02)
- `target_return` (float): Target return for downside deviation (only used with Sortino Ratio, default: 0.0)
- `allocation_metric` (str): Risk-adjusted return metric to use ('information_ratio' default, 'sharpe', or 'sortino')
- `benchmark_returns` (pd.Series, optional): Custom benchmark returns for Information Ratio. If None, uses equal-weighted portfolio (default: None)
- `verbose` (bool): If True, prints detailed Information Ratio calculations during allocation (default: False)
- `min_return_threshold` (float, optional): Minimum return threshold for Information Ratio allocation. If None (default), uses benchmark return as threshold. If float, absolute minimum return (e.g., 0.05 for 5% annualized). Prevents low-return assets from being over-weighted. Defaults to None.
- `min_tracking_error` (float): Minimum tracking error floor to prevent tiny denominators from inflating Information Ratio. Defaults to 0.01 (1% annualized).
- `denoise` (bool, optional): If True, apply covariance matrix denoising before optimization. Denoising removes random noise from the eigenvalue spectrum using Random Matrix Theory. Defaults to False.
- `denoising_method` (str, optional): Denoising method to use when denoise=True. Options: 'constant_residual' (default/standard), 'targeted_shrinkage', or 'eigenvalue_clipping'. Defaults to 'constant_residual'. See [Denoising.md](Denoising.md) for details on each method.

---

## 6) Mathematical Verification

The implementation must verify:

1. **Clustering Preservation**: RE-HRP uses identical clustering to HRP (same correlation distance, same tree structure)
2. **Allocation Formula**: $\alpha = \frac{IR_{right}}{IR_{left} + IR_{right}}$ is correctly implemented (default, when thresholds met)
3. **Weight Combination**: $w_{final} = [(1-\alpha) \mathbf{w}_{left}, \alpha \mathbf{w}_{right}]$ correctly implements higher ratio → higher weight
4. **Information Ratio Calculation**: Information Ratio (default), Sharpe Ratio, or Sortino Ratio are calculated correctly
5. **Benchmark Setup**: Equal-weighted benchmark is correctly calculated and used
6. **Return Threshold Constraints**: 
   - Single-asset clusters: Individual return $R_{asset} \geq \tau$ checked
   - Multi-asset clusters: Cluster return $R_{cluster} \geq \tau$ AND $\sum_{i: R_i < \tau} w_i \leq 0.5$ checked
7. **Tracking Error Floor**: Minimum tracking error floor $\sigma_{excess} = \max(\sigma_{excess}, \min\_tracking\_error)$ correctly applied
8. **Allocation Logic**: Four cases (both meet, only left, only right, neither) correctly implemented with single-asset penalties
9. **Weight Constraints**: $\sum w_i = 1$, $w_i \geq 0$ (same as HRP)
10. **Edge Cases**: Handles negative ratios, zero tracking error/volatility/deviation, single assets, threshold violations

---

## 7) Comparison with Other Methods

### 6.1 RE-HRP vs HRP

- **Similarities**: Same clustering structure, same quasi-diagonalization, same recursive framework
- **Differences**: RE-HRP uses Information Ratio (default), Sharpe Ratio, or Sortino Ratio for allocation, HRP uses inverse variance
- **When to use RE-HRP**: When you want to incorporate return information and favor return clusters while preserving HRP's robust clustering structure
- **When to use HRP**: When you want pure risk parity without return information
- **Information Ratio Advantage**: Favors return clusters vs benchmark, avoids over-weighting low-volatility assets with modest returns

### 6.2 RE-HRP vs NCO

- **NCO**: Uses clustering to form groups, then optimizes within and between groups
- **RE-HRP**: Uses clustering for structure, allocates based on risk-adjusted return ratio (Information Ratio default, Sharpe/Sortino options)
- **RE-HRP advantage**: Simpler allocation (no optimization needed), preserves HRP's structure, favors return clusters via Information Ratio
- **NCO advantage**: More flexible optimization within clusters

### 6.3 RE-HRP vs Mean-Variance Optimization

- **MVO**: Requires covariance matrix inversion, sensitive to estimation errors
- **RE-HRP**: Avoids matrix inversion, more robust to estimation errors
- **RE-HRP advantage**: More stable out-of-sample, preserves clustering structure
- **MVO advantage**: Can target specific return/risk levels

---

## 8) Code-to-Math Mapping

| Mathematical Concept | Code Implementation |
|---------------------|---------------------|
| Correlation distance: $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ | `distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))` |
| Hierarchical clustering | `linkage(condensed_distances, method=self.linkage_method)` |
| Quasi-diagonalization | `cov_matrix.iloc[order, :].iloc[:, order]` |
| Expected returns: $\mu_i = E[R_i]$ | `expected_returns = returns_df.mean().values` |
| Portfolio returns: $R_p = \mathbf{w}^T \mathbf{R}$ | `portfolio_returns = returns_array @ weights` |
| Benchmark returns: $R_{benchmark} = \frac{1}{n}\sum_{i=1}^n R_i$ | `benchmark_returns = returns_df.mean(axis=1)` (equal-weighted) |
| Excess returns: $R_{excess} = R_p - R_{benchmark}$ | `excess_returns = portfolio_returns - benchmark_returns` |
| Tracking error: $\sigma_{excess} = \sqrt{E[(R_{excess} - E[R_{excess}])^2]}$ | `tracking_error = np.std(excess_returns, ddof=1)` |
| Information Ratio: $IR = \frac{E[R_p] - E[R_{benchmark}]}{\sigma_{excess}}$ | `ir = (expected_excess_annual) / tracking_error_annual` |
| Total volatility: $\sigma_{total} = \sqrt{E[(R - E[R])^2]}$ | `volatility = np.std(returns, ddof=1)` |
| Sharpe Ratio: $SR = \frac{E[R] - r_f}{\sigma_{total}}$ | `sharpe = (expected_return - risk_free_rate) / volatility` |
| Downside deviation: $\sigma_{downside} = \sqrt{E[\min(0, R - target)^2]}$ | `downside_returns = np.minimum(0, returns - target); np.sqrt(np.mean(downside_returns**2))` |
| Sortino Ratio: $SR = \frac{E[R] - r_f}{\sigma_{downside}}$ | `sortino = (expected_return - risk_free_rate) / downside_deviation` |
| Allocation: $\alpha = \frac{IR_R}{IR_L + IR_R}$ (default) | `alpha = ir_right / (ir_left + ir_right)` |
| Weight combination: $w = [(1-\alpha) w_L, \alpha w_R]$ | `weights = np.concatenate([(1-alpha) * weights_left, alpha * weights_right])` |

---

## 9) References

1. **De Prado, M. L. (2016)**. Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59-69.
   - DOI: https://doi.org/10.3905/jpm.2016.42.4.059
   - Original HRP algorithm and mathematical foundations

2. **Sharpe, W. F. (1966)**. Mutual fund performance. *Journal of Business*, 39(1), 119-138.
   - Sharpe Ratio definition and justification

3. **Sortino, F. A., & Price, L. N. (1994)**. Performance measurement in a downside risk framework. *The Journal of Investing*, 3(3), 59-64.
   - Sortino Ratio definition and justification

4. **De Prado, M. L. (2018)**. Advances in Financial Machine Learning. *Wiley*.
   - In-depth discussion of HRP and hierarchical clustering in finance

---

## 10) Usage Example

```python
from portfolios.portfolio_src import ReturnEnhancedHRP
import pandas as pd

# Load returns data
returns_df = pd.DataFrame(...)  # Shape: (n_periods, n_assets)

# Initialize RE-HRP optimizer (default: Information Ratio)
re_hrp = ReturnEnhancedHRP(
    linkage_method='ward',
    risk_free_rate=0.02,
    allocation_metric='information_ratio',  # Default
    denoise=False,  # Optional: enable covariance matrix denoising
    denoising_method='constant_residual',  # If denoise=True
    verbose=True  # Optional: show detailed Information Ratio calculations
)

# Or use Sharpe Ratio:
# re_hrp = ReturnEnhancedHRP(
#     linkage_method='ward',
#     risk_free_rate=0.02,
#     allocation_metric='sharpe'
# )

# Or use Sortino Ratio:
# re_hrp = ReturnEnhancedHRP(
#     linkage_method='ward',
#     risk_free_rate=0.02,
#     target_return=0.0,  # Only needed for Sortino
#     allocation_metric='sortino'
# )

# Or use custom benchmark for Information Ratio:
# custom_benchmark = pd.Series(...)  # Custom benchmark returns
# re_hrp = ReturnEnhancedHRP(
#     linkage_method='ward',
#     allocation_metric='information_ratio',
#     benchmark_returns=custom_benchmark
# )

# Fit the optimizer
re_hrp.fit(returns_df)
# Output will show:
# ================================================================================
# RE-HRP: Information Ratio Allocation
# ================================================================================
# Benchmark: Equal-weighted portfolio of N assets
# Allocation Metric: Information Ratio
# Formula: α = IR_right / (IR_left + IR_right)
# Information Ratio = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error
# ================================================================================

# Get portfolio weights
weights = re_hrp.predict()

# Get performance metrics
expected_return, volatility, sharpe_ratio = re_hrp.portfolio_performance()

# Plot dendrogram
re_hrp.plot_dendrogram()
```

---

## 11) Key Advantages

1. **Preserves HRP's Robustness**: Same clustering structure, avoids matrix inversion
2. **Favors Return Clusters**: Uses Information Ratio (default) to identify and favor clusters with better excess returns vs benchmark
3. **Avoids Low-Volatility Bias**: Information Ratio doesn't over-weight low-volatility assets with modest returns (unlike Sharpe Ratio)
4. **Leverages Covariance Information**: Uses rich information in covariance matrices to identify return clusters within clustering structure
5. **Clear User Communication**: Comprehensive logging shows benchmark definition, Information Ratio calculations, and allocation decisions
6. **Flexibility**: Options to use Sharpe Ratio or Sortino Ratio for different use cases
7. **Mathematically Sound**: Proper risk-adjusted return metric, preserves recursive structure
8. **Out-of-Sample Stability**: More stable than mean-variance optimization
9. **Real-World Performance**: Leads to better portfolio returns by favoring return clusters rather than just low-risk assets

---

## 12) Limitations and Considerations

1. **Return Estimation**: Requires reliable expected return estimates (same as any return-based method)
2. **Benchmark Choice**: Information Ratio (default) depends on benchmark definition - equal-weighted is default but may not suit all use cases
3. **Computational Cost**: Slightly more expensive than HRP (needs to compute portfolio returns and benchmark returns for Information Ratio)
4. **Parameter Sensitivity**: Benchmark choice (for Information Ratio), risk-free rate (for Sharpe/Sortino), and target return (for Sortino) affect results
5. **Clustering Dependency**: Still depends on correlation structure (same as HRP)
6. **Metric Choice**: Information Ratio (default) is recommended for most cases to favor return clusters; Sharpe/Sortino available as alternatives

---

## 13) Conclusion

Return-Enhanced HRP extends De Prado's Hierarchical Risk Parity by incorporating return information through risk-adjusted return maximization (Information Ratio by default, Sharpe Ratio and Sortino Ratio as options). It preserves HRP's mathematically sound clustering structure while allocating capital based on risk-adjusted returns. The default use of Information Ratio naturally favors return clusters within the hierarchical structure, avoiding the low-volatility bias of Sharpe Ratio while leveraging the rich information in covariance matrices. This makes RE-HRP a powerful tool for investors who want to favor return characteristics within HRP's robust clustering framework, leading to better portfolio returns while maintaining HRP's stability and mathematical soundness.

