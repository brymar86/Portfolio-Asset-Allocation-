# Covariance Matrix Denoising using Random Matrix Theory

This document describes the **covariance matrix denoising** methodology based on Marcos Lopez de Prado's work using Random Matrix Theory. Denoising removes random noise from the eigenvalue spectrum of covariance matrices, leading to more robust portfolio optimization.

**IMPORTANT ATTRIBUTION**: This implementation is based on the research of Marcos Lopez de Prado. The denoising methodology using Random Matrix Theory and Marcenko-Pastur distribution is from his work on robust covariance matrix estimation.

**References**:
- De Prado, M. L. (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59-69. DOI: https://doi.org/10.3905/jpm.2016.42.4.059
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley. (Chapter on covariance matrix denoising)
- Marcenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457-483.

---

## 1) Notation and Setup

Let there be $n$ assets and $T$ observations (time periods).

- **Covariance matrix** (symmetric positive semi-definite):  
  $$\mathbf{C} \in \mathbb{R}^{n \times n}$$

- **Eigenvalue decomposition**:  
  $$\mathbf{C} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$$

  where:
  - $\mathbf{Q} \in \mathbb{R}^{n \times n}$: Orthogonal matrix of eigenvectors (each column is an eigenvector)
  - $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$: Diagonal matrix of eigenvalues
  - Eigenvalues ordered: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$

- **Q ratio** (observations to assets):  
  $$Q = \frac{T}{N}$$

- **Marcenko-Pastur bounds**:  
  $$\lambda_{\min} = \sigma^2(1 - \sqrt{1/Q})^2 \quad \text{(when } Q \geq 1 \text{, else 0)}$$  
  $$\lambda_{\max} = \sigma^2(1 + \sqrt{1/Q})^2$$

  where $\sigma^2$ is estimated from the eigenvalue distribution (typically $\sigma^2 = \text{mean}(\{\lambda_i\})$).
  
  **Note**: The canonical MP formula uses $q = N/T$, giving $\lambda_{\pm} = \sigma^2(1 \pm \sqrt{q})^2$. When using $Q = T/N$, we use $\lambda_{\pm} = \sigma^2(1 \pm \sqrt{1/Q})^2$, which is equivalent. When $Q < 1$ (rank-deficient case: $N > T$), we have $\lambda_{\min} = 0$.

- **Denoised correlation matrix**:  
  $$\mathbf{R}_{\text{denoised}} = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T$$

- **Rescaled to covariance** (if volatilities provided):  
  $$\boldsymbol{\Sigma}_{\text{denoised}} = \mathbf{D} \mathbf{R}_{\text{denoised}} \mathbf{D}$$  
  where $\mathbf{D} = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_n)$ contains standard deviations.

Implementation mapping (exactly what you compute in code):

- `eigenvalues, eigenvectors = eigh(corr_matrix)` corresponds to $\mathbf{R} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$
- `Q = num_observations / num_assets` corresponds to $Q = T/N$
- `lambda_max = sigma_sq * (1 + np.sqrt(1/Q))**2` corresponds to $\lambda_{\max} = \sigma^2(1 + \sqrt{1/Q})^2$
- `corr_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T` corresponds to $\mathbf{R}_{\text{denoised}} = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T$
- `cov_denoised = D @ corr_denoised @ D` corresponds to $\boldsymbol{\Sigma}_{\text{denoised}} = \mathbf{D} \mathbf{R}_{\text{denoised}} \mathbf{D}$ (rescaling step)

---

## 2) Overview and Motivation

### 2.1 The Problem: Noise in Covariance Matrices

Empirical covariance matrices estimated from financial returns contain both:
- **Signal**: True relationships between assets (genuine correlations and covariances)
- **Noise**: Random estimation errors (sample-specific noise from finite observations)

When the number of observations $T$ is not much larger than the number of assets $n$, the covariance matrix is particularly noisy. This noise manifests as:
- Eigenvalues that represent random fluctuations rather than true signal
- Overestimation of extreme correlations
- Instability in portfolio optimization
- Poor out-of-sample performance

### 2.2 The Solution: Random Matrix Theory

Random Matrix Theory (RMT) provides a mathematical framework to identify which eigenvalues represent signal vs. noise. The **Marcenko-Pastur distribution** describes the distribution of eigenvalues for random covariance matrices, allowing us to:
1. Identify random (noise) eigenvalues using theoretical bounds
2. Filter or adjust these noise eigenvalues
3. Preserve signal eigenvalues that contain genuine information

**Key Insight**: Eigenvalues within the Marcenko-Pastur bounds are likely noise and should be filtered. Eigenvalues outside these bounds contain signal and should be preserved.

---

## 3) Mathematical Foundation: Marcenko-Pastur Distribution

### 3.1 Theoretical Background

The Marcenko-Pastur distribution describes the limiting distribution of eigenvalues for sample covariance matrices of random data. For a $T \times n$ matrix of independent random vectors with variance $\sigma^2$, as $T, n \to \infty$ with $Q = T/n$ fixed, the eigenvalue distribution converges to the Marcenko-Pastur law.

### 3.2 Marcenko-Pastur Bounds

For a covariance matrix computed from $T$ observations of $n$ assets, define:

$$Q = \frac{T}{n}$$

The **Marcenko-Pastur bounds** define the range of random eigenvalues:

$$\lambda_{\min} = \sigma^2(1 - \sqrt{Q})^2 \quad \text{(when } Q \leq 1 \text{, else 0)}$$

$$\lambda_{\max} = \sigma^2(1 + \sqrt{Q})^2$$

where $\sigma^2$ is typically estimated as:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} \lambda_i = \text{mean}(\{\lambda_i\})$$

**Classification**:
- **Random eigenvalues** (noise): $\lambda_i \in [\lambda_{\min}, \lambda_{\max}]$
- **Signal eigenvalues**: $\lambda_i < \lambda_{\min}$ or $\lambda_i > \lambda_{\max}$

### 3.3 Interpretation

- **Small Q ($Q < 1$)**: Few observations per asset → wide bounds → more eigenvalues classified as noise
- **Large Q ($Q > 1$)**: Many observations per asset → narrow bounds → fewer eigenvalues classified as noise
- **Eigenvalues outside bounds**: Represent genuine structure (signal) in the covariance matrix

---

## 4) Denoising Methodology

### 4.1 General Denoising Process

The denoising process follows these steps:

1. **Matrix Preparation**:  
   - **Recommended**: Use correlation matrix $\mathbf{R}$ (standardized returns)
   - MP theory assumes isotropic noise ($\sigma^2\mathbf{I}$), which correlation matrices better satisfy
   - For correlation matrices, $\sigma^2 \approx 1$ (trace = $n$, so mean eigenvalue $\approx 1$)

2. **Eigenvalue Decomposition**:  
   $$\mathbf{R} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$$

3. **Calculate MP Bounds**:  
   Compute $\lambda_{\min}$ and $\lambda_{\max}$ using $Q = T/n$ and $\sigma^2 = \text{mean}(\{\lambda_i\})$

4. **Identify Random Eigenvalues**:  
   Find eigenvalues within MP bounds: $\{\lambda_r | \lambda_{\min} \leq \lambda_r \leq \lambda_{\max}\}$

5. **Apply Denoising Method**:  
   Modify random eigenvalues using chosen method (see Section 5)

6. **Reconstruct Matrix**:  
   $$\mathbf{R}_{\text{denoised}} = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T$$

7. **Rescale to Covariance (Optional)**:  
   If covariance matrix is needed, rescale using volatilities:  
   $$\boldsymbol{\Sigma}_{\text{denoised}} = \mathbf{D} \mathbf{R}_{\text{denoised}} \mathbf{D}$$  
   where $\mathbf{D} = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_n)$ contains the standard deviations.

### 4.2 Correlation vs Covariance Denoising

**Why Denoise Correlation Matrices?**

Marcenko-Pastur theory assumes isotropic noise, meaning the underlying true covariance is proportional to the identity matrix ($\sigma^2\mathbf{I}$). This assumption is better satisfied by correlation matrices:

- **Correlation matrices**: Standardized, so diagonal elements are exactly 1.0, and $\sigma^2 \approx 1$
- **Covariance matrices**: Have heterogeneous volatilities, violating the isotropic assumption
- **Practical benefit**: MP bounds are more accurate for correlation matrices, leading to better noise/signal classification

**Implementation Approach**:
1. Compute correlation matrix from returns
2. Denoise correlation matrix using MP bounds
3. Rescale to covariance: $\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R}_{\text{denoised}} \mathbf{D}$ if needed

This approach is theoretically sound and recommended for production use.

### 4.3 Implementation Details

The algorithm ensures:
- Eigenvalues are sorted in descending order
- Signal eigenvalues (outside MP bounds) remain unchanged
- Random eigenvalues (within MP bounds) are modified according to the chosen method
- Matrix is reconstructed maintaining orthogonality of eigenvectors
- For correlation matrices: diagonal remains exactly 1.0, values clipped to $[-1, 1]$ range

---

## 5) Denoising Methods

Three denoising methods are implemented, each with different characteristics:

### 5.1 Constant Residual Eigenvalue (Standard/Default)

**Mathematical Formulation**:

For random eigenvalues $\{\lambda_r\}$, replace each with their mean:

$$\lambda_{\text{denoised},r} = \frac{1}{|\{\lambda_r\}|} \sum_{\lambda \in \{\lambda_r\}} \lambda = \text{mean}(\{\lambda_r\})$$

Signal eigenvalues remain unchanged:

$$\lambda_{\text{denoised},s} = \lambda_{\text{original},s} \quad \text{for signal eigenvalues}$$

**Properties**:
- **Trace Preservation**: Exactly preserves trace of matrix (see Section 6.3 for mathematical proof)
- **Conservative Denoising**: Balanced approach between noise removal and structure preservation
- **Default in Implementation**: Set as the default method in this implementation

**When to Use**: Recommended default choice for most applications. Provides good balance between noise removal and signal preservation while maintaining mathematical properties (trace preservation).

**Implementation**:
```python
# Identify random eigenvalues
random_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
random_eigenvalues = eigenvalues[random_mask]

# Replace with mean
if len(random_eigenvalues) > 0:
    mean_random = np.mean(random_eigenvalues)
    eigenvalues_denoised[random_mask] = mean_random
```

### 5.2 Targeted Shrinkage

**Mathematical Formulation**:

Calculate global mean eigenvalue:

$$\lambda_{\text{mean}} = \frac{1}{n}\sum_{i=1}^{n} \lambda_i$$

For random eigenvalues, apply shrinkage:

$$\lambda_{\text{denoised},r} = \lambda_{\text{original},r} + \alpha \cdot (\lambda_{\text{mean}} - \lambda_{\text{original},r})$$

where $\alpha = 0.5$ is the shrinkage factor (default).

Signal eigenvalues remain unchanged:

$$\lambda_{\text{denoised},s} = \lambda_{\text{original},s} \quad \text{for signal eigenvalues}$$

**Properties**:
- **Gradual Modification**: Eigenvalues move partially toward mean (shrinkage factor $\alpha = 0.5$)
- **Conservative**: Preserves more of the original eigenvalue structure
- **Relative Ordering**: Maintains relative ordering better than constant residual

**When to Use**: When you want more conservative denoising that preserves more of the original eigenvalue structure.

**Implementation**:
```python
# Calculate global mean
mean_eigenvalue = np.mean(eigenvalues)

# Shrink random eigenvalues toward mean
shrinkage_factor = 0.5
eigenvalues_denoised[random_mask] = (
    eigenvalues[random_mask] + 
    shrinkage_factor * (mean_eigenvalue - eigenvalues[random_mask])
)
```

### 5.3 Eigenvalue Clipping

**Mathematical Formulation**:

For random eigenvalues, set all to the MP upper bound:

$$\lambda_{\text{denoised},r} = \lambda_{\max} \quad \text{for all random eigenvalues}$$

Signal eigenvalues remain unchanged:

$$\lambda_{\text{denoised},s} = \lambda_{\text{original},s} \quad \text{for signal eigenvalues}$$

**Properties**:
- **Aggressive Denoising**: Sets all noise eigenvalues to a single threshold value
- **Simplified Spectrum**: Creates a cleaner eigenvalue spectrum
- **Maximum Noise Removal**: Most aggressive of the three methods

**When to Use**: When you want aggressive noise removal and are confident in the signal identification.

**Implementation**:
```python
# Clip random eigenvalues to lambda_max
eigenvalues_denoised[random_mask] = lambda_max
```

---

## 6) Mathematical Properties

### 6.1 Symmetry Preservation

**Property**: The denoised covariance matrix $\mathbf{C}_{\text{denoised}}$ is symmetric.

**Mathematical Proof**:

Since the original covariance matrix $\mathbf{C}$ is symmetric, its eigenvalue decomposition is:

$$\mathbf{C} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$$

where $\mathbf{Q}$ is orthogonal ($\mathbf{Q}^T = \mathbf{Q}^{-1}$).

After denoising, we have:

$$\mathbf{C}_{\text{denoised}} = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T$$

Taking the transpose:

$$\mathbf{C}_{\text{denoised}}^T = (\mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T)^T = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}}^T \mathbf{Q}^T = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T = \mathbf{C}_{\text{denoised}}$$

since $\boldsymbol{\Lambda}_{\text{denoised}}$ is diagonal (and therefore symmetric).

**Why This Matters**:
- Covariance matrices must be symmetric by definition
- Symmetry is required for correlation matrix computation
- Essential for distance matrix calculation in HRP clustering
- Required for valid covariance structure in portfolio optimization

**Implementation Verification**:
```python
# After reconstruction, enforce symmetry (handles numerical precision)
cov_denoised_array = (cov_denoised_array + cov_denoised_array.T) / 2
```

### 6.2 Positive Semi-Definiteness Preservation

**Property**: The denoised covariance matrix $\mathbf{C}_{\text{denoised}}$ is positive semi-definite (all eigenvalues $\geq 0$).

**Mathematical Proof**:

For a matrix to be positive semi-definite, all eigenvalues must be non-negative:

$$\lambda_i(\mathbf{C}_{\text{denoised}}) \geq 0 \quad \forall i$$

This is guaranteed because:

1. **Original eigenvalues are non-negative**: Covariance matrices are positive semi-definite, so $\lambda_i(\mathbf{C}) \geq 0$ for all $i$

2. **Random eigenvalues are replaced with non-negative values**:
   - **Constant residual**: $\lambda_{\text{denoised}} = \text{mean}(\{\lambda_r\}) \geq 0$ (mean of non-negative numbers)
   - **Targeted shrinkage**: $\lambda_{\text{denoised}} = \lambda_r + \alpha(\lambda_{\text{mean}} - \lambda_r)$ where $\lambda_{\text{mean}} \geq 0$ and $\lambda_r \geq 0$, so $\lambda_{\text{denoised}} \geq 0$
   - **Eigenvalue clipping**: $\lambda_{\text{denoised}} = \lambda_{\max} \geq 0$ (MP bound is non-negative)

3. **Signal eigenvalues remain unchanged**: $\lambda_{\text{denoised}} = \lambda_{\text{original}} \geq 0$

Therefore, all eigenvalues of $\mathbf{C}_{\text{denoised}}$ are non-negative, making it positive semi-definite.

**Why This Matters**:
- **Portfolio Variance Non-Negativity**: Portfolio variance $\sigma_p^2 = \mathbf{w}^T \mathbf{C} \mathbf{w} \geq 0$ requires positive semi-definite covariance matrix
- **Optimization Stability**: Many optimization algorithms require positive semi-definite matrices
- **Cholesky Decomposition**: Positive semi-definiteness enables matrix decompositions used in optimization
- **Valid Covariance Structure**: Essential property for any covariance matrix

**Implementation Verification**:
```python
# Verify all eigenvalues >= 0 (within numerical tolerance)
eigenvalues = np.linalg.eigvalsh(cov_denoised.values)
assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
```

### 6.3 Trace Preservation (Constant Residual Method)

**Property**: For the constant residual method, the trace is **exactly** preserved: $\text{Tr}(\mathbf{C}_{\text{denoised}}) = \text{Tr}(\mathbf{C}_{\text{original}})$

**Mathematical Background**:

The trace of a matrix equals the sum of its eigenvalues (or equivalently, the sum of its diagonal elements):

$$\text{Tr}(\mathbf{C}) = \sum_{i=1}^{n} C_{ii} = \sum_{i=1}^{n} \lambda_i$$

For constant residual denoising:
- Random eigenvalues $\{\lambda_r\}$ are replaced with their mean: $\lambda_{\text{denoised},r} = \text{mean}(\{\lambda_r\})$
- Signal eigenvalues remain unchanged: $\lambda_{\text{denoised},s} = \lambda_{\text{original},s}$

**Mathematical Proof**:

Let $|\{\lambda_r\}| = k$ be the number of random eigenvalues. The sum of denoised random eigenvalues is:

$$\sum_{\lambda_r \in \{\lambda_r\}} \lambda_{\text{denoised},r} = k \cdot \text{mean}(\{\lambda_r\}) = k \cdot \frac{1}{k}\sum_{\lambda_r \in \{\lambda_r\}} \lambda_r = \sum_{\lambda_r \in \{\lambda_r\}} \lambda_r$$

Therefore:

$$\text{Tr}(\mathbf{C}_{\text{denoised}}) = \sum_{\lambda_s} \lambda_{\text{denoised},s} + \sum_{\lambda_r} \lambda_{\text{denoised},r} = \sum_{\lambda_s} \lambda_{\text{original},s} + \sum_{\lambda_r} \lambda_{\text{original},r} = \text{Tr}(\mathbf{C}_{\text{original}})$$

**Mathematical Interpretation: What Trace Preservation Means**

**Critical Distinction - Trace Preservation ≠ Information Preservation**:

Trace preservation does **NOT** mean that informational content is preserved. The denoising process **explicitly changes** the information content by:
- Removing noise from eigenvalues
- Modifying the eigenvalue distribution
- Changing the covariance structure (matrix elements)

However, trace preservation ensures a specific mathematical constraint: the **total variance budget** (sum of eigenvalues) remains constant. This is a conservative property that prevents the matrix from being dramatically rescaled, while still allowing noise removal.

**What Trace Preservation Actually Preserves**:

1. **Total Variance Budget**: The trace $\text{Tr}(\mathbf{C}) = \sum \lambda_i$ represents the total variance "budget" in the system. For a covariance matrix, this can be interpreted as the sum of individual asset variances (diagonal elements) or equivalently, the sum of eigenvalues.

2. **Scale Preservation**: By preserving trace, we ensure that the overall "magnitude" or "scale" of the covariance matrix is unchanged. The denoised matrix has the same total variance budget as the original.

3. **What Changes vs. What Stays Constant**:
   - **What changes**: The **distribution** of variance across eigenvalues (noise eigenvalues are modified)
   - **What stays constant**: The **total sum** of variance (trace)

4. **Geometric Interpretation**: In the eigenvalue space, we're redistributing variance from noise eigenvalues to their mean, while keeping the total sum constant. Think of it like redistributing weight across multiple objects on a scale - the total weight (trace) stays the same, but the distribution changes.

5. **Why This Matters Mathematically**:
   - **Prevents Dramatic Rescaling**: Without trace preservation, denoising could accidentally scale the entire matrix up or down
   - **Conservative Modification**: Ensures the matrix "scale" remains interpretable and comparable to the original
   - **Physical Interpretation**: In portfolio context, total variance represents total risk in the system - preserving this maintains interpretability

**Comparison with Other Methods**:
- **Constant Residual**: Trace preserved exactly (most conservative in terms of scale)
- **Targeted Shrinkage**: Trace may change (depends on shrinkage target)
- **Eigenvalue Clipping**: Trace typically increases (noise eigenvalues set to $\lambda_{\max}$, which is often larger than their mean)

**Why This Matters**:
- **Matrix Scale Preservation**: Trace represents the "total variance" or "total risk" in the system
- **Conservative Denoising**: Preserving trace ensures denoising doesn't dramatically change the overall scale or magnitude
- **Interpretability**: Maintains intuitive meaning of the covariance matrix scale (total variance budget unchanged)
- **Mathematical Rigor**: Provides a clear mathematical constraint that distinguishes constant residual from other methods

**Implementation Verification**:
```python
original_trace = np.trace(cov_matrix.values)
denoised_trace = np.trace(cov_denoised.values)
# Trace should be exactly preserved (within numerical precision)
trace_diff = abs(original_trace - denoised_trace) / abs(original_trace)
assert trace_diff < 1e-10  # Should be exactly equal (within floating-point precision)
```

**Note on Numerical Precision**: In practice, floating-point arithmetic may introduce tiny numerical errors (typically $< 10^{-10}$), but mathematically the trace is exactly preserved. This is a fundamental mathematical property of the constant residual method, not an approximation.

### 6.4 Eigenvalue Modification Verification

Each denoising method modifies eigenvalues according to its specification. This section verifies that the modifications are correctly implemented.

#### 6.4.1 Constant Residual: Random Eigenvalues Replaced with Mean

**Verification**:
- Identify random eigenvalues: $\{\lambda_r | \lambda_{\min} \leq \lambda_r \leq \lambda_{\max}\}$
- Calculate expected value: $\lambda_{\text{expected}} = \text{mean}(\{\lambda_r\})$
- Verify: $\lambda_{\text{denoised},r} = \lambda_{\text{expected}}$ for all random eigenvalues (within numerical precision)
- Verify: $\lambda_{\text{denoised},s} = \lambda_{\text{original},s}$ for all signal eigenvalues

#### 6.4.2 Targeted Shrinkage: Eigenvalues Shrink Toward Mean

**Verification**:
- Calculate global mean: $\lambda_{\text{mean}} = (1/n)\sum_i \lambda_i$
- For each random eigenvalue $\lambda_r$:
  - Expected: $\lambda_{\text{denoised},r} = \lambda_r + \alpha(\lambda_{\text{mean}} - \lambda_r)$ where $\alpha = 0.5$
  - Verify: $\lambda_{\text{denoised},r}$ lies between $\lambda_r$ and $\lambda_{\text{mean}}$
  - Verify direction: If $\lambda_r > \lambda_{\text{mean}}$, then $\lambda_{\text{denoised},r} < \lambda_r$ (shrinks downward)
  - Verify direction: If $\lambda_r < \lambda_{\text{mean}}$, then $\lambda_{\text{denoised},r} > \lambda_r$ (shrinks upward)
- Verify: $\lambda_{\text{denoised},s} = \lambda_{\text{original},s}$ for all signal eigenvalues

#### 6.4.3 Eigenvalue Clipping: Random Eigenvalues Set to Threshold

**Verification**:
- Calculate MP upper bound: $\lambda_{\max} = \sigma^2(1 + \sqrt{Q})^2$
- Verify: $\lambda_{\text{denoised},r} = \lambda_{\max}$ for all random eigenvalues (within numerical precision)
- Verify: $\lambda_{\text{denoised},s} = \lambda_{\text{original},s}$ for all signal eigenvalues

---

## 7) Code-to-Math Mapping

| Mathematical Concept | Code Implementation |
|---------------------|---------------------|
| Eigenvalue decomposition: $\mathbf{R} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$ | `eigenvalues, eigenvectors = eigh(corr_matrix)` |
| Q ratio: $Q = T/n$ | `Q = num_observations / num_assets` |
| Mean eigenvalue: $\sigma^2 = \text{mean}(\{\lambda_i\})$ | `sigma_sq = np.mean(eigenvalues)` |
| MP upper bound: $\lambda_{\max} = \sigma^2(1 + \sqrt{1/Q})^2$ | `lambda_max = sigma_sq * (1 + np.sqrt(1/Q))**2` |
| MP lower bound: $\lambda_{\min} = \sigma^2(1 - \sqrt{1/Q})^2$ | `lambda_min = sigma_sq * (1 - np.sqrt(1/Q))**2` (if $Q \geq 1$, else 0) |
| Random eigenvalue identification | `random_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)` |
| Constant residual: $\lambda_{\text{denoised}} = \text{mean}(\{\lambda_r\})$ | `eigenvalues_denoised[random_mask] = np.mean(eigenvalues[random_mask])` |
| Targeted shrinkage: $\lambda_{\text{denoised}} = \lambda + \alpha(\lambda_{\text{mean}} - \lambda)$ | `eigenvalues_denoised[random_mask] = eigenvalues[random_mask] + shrinkage_factor * (mean_eigenvalue - eigenvalues[random_mask])` |
| Eigenvalue clipping: $\lambda_{\text{denoised}} = \lambda_{\max}$ | `eigenvalues_denoised[random_mask] = lambda_max` |
| Matrix reconstruction: $\mathbf{R}_{\text{denoised}} = \mathbf{Q} \boldsymbol{\Lambda}_{\text{denoised}} \mathbf{Q}^T$ | `corr_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T` |
| Correlation clipping | `corr_denoised = np.clip(corr_denoised, -1.0, 1.0)` |
| Diagonal enforcement (correlation) | `np.fill_diagonal(corr_denoised, 1.0)` |
| Rescale to covariance: $\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R} \mathbf{D}$ | `cov_denoised = D @ corr_denoised @ D` where `D = np.diag(volatilities)` |
| Symmetry enforcement | `matrix_denoised = (matrix_denoised + matrix_denoised.T) / 2` |
| Trace calculation: $\text{Tr}(\mathbf{C}) = \sum \lambda_i$ | `trace = np.trace(cov_matrix)` or `trace = np.sum(eigenvalues)` |

---

## 8) Edge Cases and Numerical Considerations

### 8.1 Small Sample Size ($T \ll n$)

**Scenario**: Few observations relative to assets (e.g., $T/n < 1$)

**Impact**:
- MP bounds become wide: $\lambda_{\max} - \lambda_{\min}$ increases
- Most eigenvalues may be classified as random (noise)
- Denoising becomes more aggressive

**Recommendation**: Use denoising when $T/n < 3$ (few observations per asset). Denoising is particularly important in this regime.

### 8.2 Large Sample Size ($T \gg n$)

**Scenario**: Many observations relative to assets (e.g., $T/n > 10$)

**Impact**:
- MP bounds become narrow: $\lambda_{\max} - \lambda_{\min}$ decreases
- Most eigenvalues may be classified as signal
- Denoising has minimal effect (most eigenvalues preserved)

**Note**: Denoising is still safe to use (it will preserve most eigenvalues as signal). The computational cost is minimal, so it can be applied by default.

### 8.3 Identity Matrix

**Scenario**: All eigenvalues equal 1 (identity matrix)

**Impact**:
- All eigenvalues may fall within MP bounds (depending on $Q$ ratio)
- Denoising will modify eigenvalues but preserve matrix structure
- Result remains a valid covariance matrix

**Verification**: Even with this degenerate case, symmetry and positive semi-definiteness are maintained.

### 8.4 Numerical Precision

**Considerations**:
- Eigenvalue decomposition uses `scipy.linalg.eigh` for numerical stability (specialized for symmetric matrices)
- Symmetry is enforced after reconstruction: $\mathbf{C} = (\mathbf{C} + \mathbf{C}^T) / 2$
- Small numerical errors in eigenvalues are acceptable ($< 10^{-10}$)
- Correlation values are clipped to $[-1, 1]$ when computed from denoised covariance

---

## 9) Integration with Portfolio Optimizers

### 9.1 Integration Point

Denoising is applied in the portfolio optimization workflow as follows:

1. **Compute Covariance Matrix**: Standard covariance matrix computed from returns
2. **Apply Denoising** (if enabled): Covariance matrix is denoised using selected method
3. **Compute Correlation Matrix**: Computed from (possibly denoised) covariance matrix
4. **Optimization**: Portfolio optimization proceeds using denoised covariance matrix

**Important**: Denoising is applied **once** to the full covariance matrix before optimization begins. It is **not** applied recursively during tree traversal in HRP.

### 9.2 Correlation Matrix Computation

When denoising is enabled, the correlation matrix is computed from the denoised covariance matrix:

$$\rho_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}$$

This ensures consistency between the denoised covariance and correlation matrices.

**Implementation**:
```python
# Compute correlation from denoised covariance matrix
std_devs = np.sqrt(np.diag(cov_matrix_denoised.values))
corr_array = cov_matrix_denoised.values / np.outer(std_devs, std_devs)
# Clip to [-1, 1] for numerical precision
corr_array = np.clip(corr_array, -1.0, 1.0)
np.fill_diagonal(corr_array, 1.0)  # Diagonal should be exactly 1.0
```

### 9.3 Usage Example

```python
from portfolios.portfolio_src import HierarchicalRiskParity, ReturnEnhancedHRP

# HRP with denoising
hrp = HierarchicalRiskParity(
    denoise=True,
    denoising_method="constant_residual"  # default
)
hrp.fit(returns_df)

# RE-HRP with denoising
re_hrp = ReturnEnhancedHRP(
    denoise=True,
    denoising_method="constant_residual"  # default
)
re_hrp.fit(returns_df)
```

---

## 10) Best Practices

### 10.1 When to Use Denoising

**Recommended**:
- When $T/n < 3$ (few observations per asset)
- For high-dimensional portfolios (20+ assets)
- When experiencing unstable optimization results
- When covariance matrix has high condition number

**Optional**:
- For well-sampled, low-dimensional portfolios ($T/n > 10$, $n < 10$)

### 10.2 Method Selection

- **Default**: Use `constant_residual` (standard approach recommended by de Prado)
- **Conservative**: Use `targeted_shrinkage` when you want to preserve more structure
- **Aggressive**: Use `eigenvalue_clipping` when you want maximum noise removal

### 10.3 Validation

- Always verify that denoised matrices are symmetric and positive semi-definite
- Compare portfolio weights with and without denoising
- Monitor out-of-sample performance improvements
- Check that denoising doesn't remove too much signal (compare eigenvalue distributions)

---

## 11) References

1. **De Prado, M. L. (2016)**. Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59-69.
   - DOI: https://doi.org/10.3905/jpm.2016.42.4.059
   - Original HRP algorithm and covariance matrix denoising methodology

2. **De Prado, M. L. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Chapter on covariance matrix denoising
   - Detailed discussion of Random Matrix Theory applications in finance

3. **Marcenko, V. A., & Pastur, L. A. (1967)**. Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457-483.
   - Original derivation of the Marcenko-Pastur distribution
   - Theoretical foundation for eigenvalue bounds

4. **Bouchaud, J. P., & Potters, M. (2009)**. *Theory of Financial Risk and Derivative Pricing: From Statistical Physics to Risk Management*. Cambridge University Press.
   - Applications of Random Matrix Theory in finance
   - Eigenvalue filtering techniques

---

## 12) Conclusion

Covariance matrix denoising using Random Matrix Theory is a mathematically rigorous technique for improving the robustness of portfolio optimization. By identifying and filtering random noise eigenvalues using the Marcenko-Pastur distribution, denoising leads to:

- More stable portfolio weights
- Better out-of-sample performance
- Reduced sensitivity to estimation errors
- More robust optimization results

The implementation preserves fundamental mathematical properties (symmetry, positive semi-definiteness, trace preservation) while providing three methods (constant residual, targeted shrinkage, eigenvalue clipping) with the constant residual method recommended as the standard approach. Denoising is seamlessly integrated into HRP and RE-HRP optimizers, making it easy to apply in practice while maintaining mathematical rigor and theoretical foundations.
