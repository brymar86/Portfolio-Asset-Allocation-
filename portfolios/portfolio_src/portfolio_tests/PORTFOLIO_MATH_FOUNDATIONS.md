# Portfolio Optimization: Mathematical Foundations

This document provides a concise mathematical reference for the portfolio optimization classes implemented in this project. Use this as a quick reference when reviewing tests or understanding the underlying mathematics.

## Table of Contents

1. [Base Concepts](#base-concepts)
2. [Hierarchical Risk Parity (HRP)](#hierarchical-risk-parity-hrp)
3. [Risk Parity (Equal Risk Contribution)](#risk-parity-equal-risk-contribution)
4. [Nested Clustered Optimization (NCO)](#nested-clustered-optimization-nco)

---

## Base Concepts

### Covariance Matrix

The covariance matrix **Σ** captures how asset returns move together:

$$\Sigma_{ij} = \text{Cov}(R_i, R_j) = \mathbb{E}[(R_i - \mu_i)(R_j - \mu_j)]$$

**Properties:**
- **Symmetry**: $\Sigma_{ij} = \Sigma_{ji}$
- **Positive Semi-Definite**: $\mathbf{w}^T \Sigma \mathbf{w} \geq 0$ for all $\mathbf{w}$
- **Diagonal Elements**: $\Sigma_{ii} = \text{Var}(R_i) = \sigma_i^2$

### Correlation Matrix

The correlation matrix **ρ** normalizes covariance by standard deviations:

$$\rho_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$$

**Properties:**
- **Bounded**: $-1 \leq \rho_{ij} \leq 1$
- **Unit Diagonal**: $\rho_{ii} = 1$ (asset perfectly correlated with itself)
- **Symmetric**: $\rho_{ij} = \rho_{ji}$

### Portfolio Return and Variance

For a portfolio with weights $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$:

**Expected Return:**
$$\mu_p = \mathbf{w}^T \boldsymbol{\mu} = \sum_{i=1}^{n} w_i \mu_i$$

**Portfolio Variance:**
$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w} = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \Sigma_{ij}$$

**Portfolio Volatility (Standard Deviation):**
$$\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$$

### Sharpe Ratio

The Sharpe ratio measures risk-adjusted return:

$$\text{SR} = \frac{\mu_p - r_f}{\sigma_p}$$

where $r_f$ is the risk-free rate.

### Weight Constraints

All portfolio optimizers enforce:
- **Fully Invested**: $\sum_{i=1}^{n} w_i = 1$
- **No Short Selling**: $w_i \geq 0$ for all $i$

---

## Hierarchical Risk Parity (HRP)

HRP addresses the instability of mean-variance optimization by using hierarchical clustering to construct portfolios without inverting the covariance matrix.

**Key Difference from NCO:** HRP does **NOT** perform mean-variance optimization. Instead, it uses hierarchical clustering to group similar assets, then recursively allocates weights inversely proportional to variance. It's a completely different approach that avoids optimization entirely.

### Algorithm Overview

1. **Distance Matrix Construction**
2. **Hierarchical Clustering**
3. **Quasi-Diagonalization**
4. **Recursive Bisection**

### Step 1: Distance Matrix

Convert correlation matrix to distance matrix:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

**Properties:**
- $d_{ij} = 0$ when $\rho_{ij} = 1$ (perfect correlation → zero distance)
- $d_{ij} = \sqrt{2}$ when $\rho_{ij} = 0$ (no correlation)
- $d_{ij} = 2$ when $\rho_{ij} = -1$ (perfect negative correlation)

### Step 2: Hierarchical Clustering

Build a dendrogram (tree structure) using linkage algorithms (e.g., Ward's method). The tree groups similar assets together based on correlation distance.

**Example:** With 4 assets, the clustering might produce:
- First, merge the two most similar assets (e.g., AAPL and MSFT)
- Then, merge the next most similar pair (e.g., GOOGL and TSLA)
- Finally, merge the two groups together

This creates a tree showing how assets are related. The tree structure determines the order for quasi-diagonalization.

### Step 3: Quasi-Diagonalization

Reorder the covariance matrix according to the hierarchical tree structure. This creates an approximate block-diagonal structure where similar assets are placed near each other.

**Example:** If clustering groups [AAPL, MSFT] and [GOOGL, TSLA], we reorder the matrix so these groups are together:

```
Original order: [AAPL, GOOGL, MSFT, TSLA]
Reordered:      [AAPL, MSFT, GOOGL, TSLA]  ← similar assets together
```

The reordered matrix looks like:
```
        AAPL  MSFT  GOOGL  TSLA
AAPL   [high  high   low    low ]  ← AAPL-MSFT block
MSFT   [high  high   low    low ]
GOOGL  [low   low   high   high]  ← GOOGL-TSLA block
TSLA   [low   low   high   high]
```

**Why this matters:** When we split the portfolio, we want to split along these natural groupings. The quasi-diagonal structure makes it clear where to split.

**Key Property:** Reordering preserves:
- Trace: $\text{tr}(\Sigma_{\text{reordered}}) = \text{tr}(\Sigma_{\text{original}})$
- Determinant: $\det(\Sigma_{\text{reordered}}) = \det(\Sigma_{\text{original}})$
- Eigenvalues (similarity transformation)

### Step 4: Recursive Bisection

Recursively split the portfolio and allocate weights inversely proportional to variance. **This is NOT optimization** - it's a deterministic recursive algorithm that splits the portfolio and allocates weights based on variance.

**Recursive Formula:**

For a split at index $k$, dividing into left sub-portfolio (indices $0$ to $k-1$) and right sub-portfolio (indices $k$ to $n-1$):

1. Compute sub-portfolio variances (using equal weights within each sub-portfolio):
   $$\sigma_L^2 = \mathbf{w}_L^T \Sigma_L \mathbf{w}_L, \quad \sigma_R^2 = \mathbf{w}_R^T \Sigma_R \mathbf{w}_R$$

2. Allocate weights inversely proportional to variance:
   $$\alpha = \frac{\sigma_R^2}{\sigma_L^2 + \sigma_R^2}$$
   $$\mathbf{w} = [\alpha \mathbf{w}_L, (1-\alpha) \mathbf{w}_R]$$

**Base Case:** For a single asset, $w = [1.0]$

**Key Insight:** The lower-variance sub-portfolio gets MORE weight. This ensures risk parity between sub-portfolios.

**Mathematical Proof (Weights Sum to 1):**

By induction:
- **Base case**: $n=1$, $w = [1.0]$ → sum = 1 ✓
- **Inductive step**: If $\sum w_L = 1$ and $\sum w_R = 1$, then:
  $$\sum w = \alpha \sum w_L + (1-\alpha) \sum w_R = \alpha \cdot 1 + (1-\alpha) \cdot 1 = 1$$

### Complete Example: HRP with 4 Assets

Let's walk through HRP with a concrete example: 4 stocks (AAPL, MSFT, GOOGL, TSLA).

**Step 1: Distance Matrix**

Suppose we have correlation matrix:
- AAPL and MSFT: high correlation (0.8)
- GOOGL and TSLA: high correlation (0.7)
- Cross-correlations: low (0.2-0.3)

Distance matrix: $d_{ij} = \sqrt{2(1 - \rho_{ij})}$
- AAPL-MSFT: $d = \sqrt{2(1-0.8)} = 0.63$ (close)
- GOOGL-TSLA: $d = \sqrt{2(1-0.7)} = 0.77$ (close)
- AAPL-GOOGL: $d = \sqrt{2(1-0.2)} = 1.26$ (far)

**Step 2: Hierarchical Clustering**

The clustering tree groups assets:
- First merge: AAPL + MSFT (they're closest)
- Second merge: GOOGL + TSLA (they're closest)
- Final merge: (AAPL, MSFT) + (GOOGL, TSLA)

**Tree order** (from dendrogram): [AAPL, MSFT, GOOGL, TSLA]

**Step 3: Quasi-Diagonalization**

Reorder covariance matrix to match tree order:
```
Original order:        Quasi-diagonal order:
[AAPL, MSFT, GOOGL, TSLA] → [AAPL, MSFT, GOOGL, TSLA]
```

The reordered matrix has high values near diagonal (similar assets together):
```
        AAPL  MSFT  GOOGL  TSLA
AAPL   [high  high   low    low ]
MSFT   [high  high   low    low ]
GOOGL  [low   low   high   high]
TSLA   [low   low   high   high]
```

**Step 4: Recursive Bisection (This is where weights come from!)**

**Level 1: Split entire portfolio**
- Try all split points, find best split (minimizes combined variance)
- Best split: After MSFT (split into [AAPL, MSFT] vs [GOOGL, TSLA])

**Level 2a: Left sub-portfolio [AAPL, MSFT]**
- Base case: 2 assets
- Compute variances with equal weights:
  - $\sigma_{AAPL}^2 = 0.04$ (4% volatility)
  - $\sigma_{MSFT}^2 = 0.03$ (3% volatility)
  - With equal weights: $\sigma_L^2 = 0.025$ (portfolio variance)
- Recursively split:
  - Try split after AAPL: [AAPL] vs [MSFT]
  - [AAPL] variance: 0.04
  - [MSFT] variance: 0.03
  - $\alpha = 0.03 / (0.04 + 0.03) = 0.43$
  - Weights: $w_{AAPL} = 0.43$, $w_{MSFT} = 0.57$

**Level 2b: Right sub-portfolio [GOOGL, TSLA]**
- Similar process:
  - $\sigma_{GOOGL}^2 = 0.05$
  - $\sigma_{TSLA}^2 = 0.08$
  - $\alpha = 0.08 / (0.05 + 0.08) = 0.62$
  - Weights: $w_{GOOGL} = 0.38$, $w_{TSLA} = 0.62$

**Level 1: Combine sub-portfolios**
- Left sub-portfolio variance: $\sigma_L^2 = 0.025$
- Right sub-portfolio variance: $\sigma_R^2 = 0.045$
- $\alpha = 0.045 / (0.025 + 0.045) = 0.64$
- Final weights:
  - $w_{AAPL} = 0.64 \times 0.43 = 0.28$
  - $w_{MSFT} = 0.64 \times 0.57 = 0.36$
  - $w_{GOOGL} = 0.36 \times 0.38 = 0.14$
  - $w_{TSLA} = 0.36 \times 0.62 = 0.22$
- Check: $0.28 + 0.36 + 0.14 + 0.22 = 1.0$ ✓

**Why This Works Without Optimization:**
- We're not maximizing or minimizing anything
- We're just recursively splitting and allocating based on variance
- Lower variance → higher weight (inverse relationship)
- This naturally balances risk across the portfolio

**Visual Representation of Recursive Bisection:**

```
Level 0: [AAPL, MSFT, GOOGL, TSLA] (all assets)
         ↓ Split at best point
         ├─ Level 1a: [AAPL, MSFT]     (variance: 0.025)
         │            ↓ Split
         │            ├─ [AAPL] → weight: 0.43
         │            └─ [MSFT] → weight: 0.57
         │
         └─ Level 1b: [GOOGL, TSLA]   (variance: 0.045)
                      ↓ Split
                      ├─ [GOOGL] → weight: 0.38
                      └─ [TSLA] → weight: 0.62

Combine: α = 0.045/(0.025+0.045) = 0.64
         Left gets 64%, Right gets 36%
         
Final:   AAPL:  0.64 × 0.43 = 0.28
         MSFT:  0.64 × 0.57 = 0.36
         GOOGL: 0.36 × 0.38 = 0.14
         TSLA:  0.36 × 0.62 = 0.22
```

**Key Insight:** The algorithm finds weights by:
1. **Splitting** the portfolio into two groups (based on clustering)
2. **Recursively** finding weights within each group
3. **Combining** the groups with weights inversely proportional to their variance
4. **No optimization needed** - it's just a deterministic recursive algorithm!

### Advantages of HRP

- **No Matrix Inversion**: Avoids numerical instability from inverting ill-conditioned covariance matrices
- **Robust to Estimation Errors**: Hierarchical structure provides stability
- **Intuitive Clustering**: Groups similar assets together

---

## Risk Parity (Equal Risk Contribution)

### What Risk Parity Does (Simple Explanation)

Risk Parity solves for **individual asset weights** (you can apply it to individual stocks, bonds, or asset classes - it works at any level).

**The Goal:** Make each asset contribute **equally to portfolio risk**, not equal capital allocation.

**Simple Example:**
- **Equal Weighting (50/50)**: 50% stocks, 50% bonds → Stocks contribute ~80% of risk, bonds contribute ~20% of risk
- **Risk Parity**: Adjust weights so stocks and bonds each contribute 50% of risk
  - If stocks are 4x more volatile than bonds, you might get: 20% stocks, 80% bonds
  - Or with leverage: 50% stocks, 200% bonds (leveraged)

**You're right about the leverage!** Many Risk Parity funds do leverage bonds to match stock risk without having an all-bond portfolio. This is a common real-world application.

### The Algorithm

Risk Parity allocates portfolio weights such that each **individual asset** contributes equally to portfolio risk. This is different from equal weighting (where each asset gets the same capital allocation) - instead, each asset contributes the same amount of **risk** to the portfolio.

### What Risk Parity Does

Risk Parity solves for **individual asset weights** (not asset class weights, though you can apply it at any level). The algorithm finds weights such that:

**Each asset contributes the same amount of risk to the portfolio.**

### Risk Contribution Formula

The risk contribution of asset $i$ is:

$$\text{RC}_i = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p} = w_i \cdot \frac{\partial \sigma_p}{\partial w_i}$$

where:
- $(\Sigma \mathbf{w})_i = \sum_{j=1}^{n} \Sigma_{ij} w_j$ is the marginal contribution (how much portfolio risk changes per unit change in weight $i$)
- $\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$ is portfolio volatility

**Intuition:** Risk contribution = (weight of asset) × (marginal risk of asset). Higher volatility assets need lower weights to contribute the same risk as lower volatility assets.

### Key Property: Risk Contributions Sum to Portfolio Volatility

$$\sum_{i=1}^{n} \text{RC}_i = \sum_{i=1}^{n} w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p} = \frac{\mathbf{w}^T \Sigma \mathbf{w}}{\sigma_p} = \frac{\sigma_p^2}{\sigma_p} = \sigma_p$$

### Optimization Problem

Risk Parity solves:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} \left(\text{RC}_i - \frac{\sigma_p}{n}\right)^2$$

**Subject to:**
- $\sum_{i=1}^{n} w_i = 1$ (fully invested)
- $w_i \geq 0$ for all $i$ (no short selling)

### Equal Risk Contribution Property

At the optimal solution:

$$\text{RC}_i = \text{RC}_j \quad \forall i, j$$

Since $\sum_{i=1}^{n} \text{RC}_i = \sigma_p$, this implies:

$$\text{RC}_i = \frac{\sigma_p}{n} \quad \forall i$$

### Inverse Volatility Relationship

For Risk Parity, weights are approximately inversely proportional to volatility:

$$w_i \approx \frac{\text{constant}}{\sigma_i}$$

**Intuition:** Higher volatility assets get lower weights to achieve equal risk contribution.

**Real-World Example (Stocks vs Bonds):**
- Stocks might have 20% annual volatility
- Bonds might have 5% annual volatility
- To get equal risk contribution, you'd need roughly 4x more bond weight than stock weight
- So instead of 50/50, you might get 20% stocks, 80% bonds (or use leverage to get 50% stocks, 200% bonds)
- This is why Risk Parity portfolios often use leverage on bonds - to match the risk contribution of stocks without having an all-bond portfolio

### Advantages of Risk Parity

- **No Return Estimates**: Only requires covariance matrix (more robust than mean-variance which needs expected returns)
- **Equal Risk**: Each asset contributes equally to portfolio risk (not equal capital allocation)
- **Stable**: Less sensitive to estimation errors than mean-variance
- **Diversification**: Naturally allocates more to lower-risk assets, creating better diversification

### Common Applications

- **60/40 Portfolio Alternative**: Instead of 60% stocks / 40% bonds by capital, use Risk Parity to get equal risk contribution
- **Multi-Asset Portfolios**: Apply to stocks, bonds, commodities, etc. to balance risk across asset classes
- **Leveraged Strategies**: Often combined with leverage on low-volatility assets (like bonds) to match risk of high-volatility assets (like stocks)

---

## Nested Clustered Optimization (NCO)

NCO reduces the dimensionality of portfolio optimization by:
1. **Clustering assets into groups** (using hierarchical clustering)
2. **Optimizing within each cluster** (using mean-variance or risk parity)
3. **Optimizing between clusters** (using mean-variance on cluster-level portfolios)

**Key Difference from HRP:** NCO **DOES** perform mean-variance optimization (or risk parity) - first within clusters, then between clusters. This is the method that clusters first, then does optimization. HRP does NOT do optimization - it uses recursive bisection based on variance.

### Algorithm Overview

1. **Clustering Step**
2. **Within-Cluster Optimization**
3. **Between-Cluster Optimization**
4. **Weight Combination**

### Step 1: Clustering

Convert correlation to distance and perform hierarchical clustering:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

Assign assets to $k$ clusters using fcluster algorithm.

### Step 2: Within-Cluster Optimization

For each cluster $c$ with asset set $\mathcal{C}_c$:

**Extract sub-matrices:**
- Sub-covariance: $\Sigma_c = [\Sigma_{ij}]$ for $i, j \in \mathcal{C}_c$
- Sub-expected returns: $\boldsymbol{\mu}_c = [\mu_i]$ for $i \in \mathcal{C}_c$

**Optimize using either:**
- **Mean-Variance**: Maximize Sharpe ratio or minimize variance for target return
- **Risk Parity**: Equalize risk contributions within cluster

**Result:** Within-cluster weights $\mathbf{w}_c$ for each cluster $c$

**Constraint:** $\sum_{i \in \mathcal{C}_c} w_{c,i} = 1$ (each cluster forms valid sub-portfolio)

### Step 3: Between-Cluster Optimization

Build reduced covariance matrix at cluster level:

**Cluster-Level Covariance:**
$$\Sigma_{\text{cluster},ij} = \mathbf{w}_i^T \Sigma_{ij} \mathbf{w}_j$$

where:
- $\Sigma_{ij}$ is the covariance matrix between assets in cluster $i$ and cluster $j$
- $\mathbf{w}_i, \mathbf{w}_j$ are the within-cluster weight vectors

**Cluster-Level Expected Returns:**
$$\mu_{\text{cluster},i} = \mathbf{w}_i^T \boldsymbol{\mu}_i$$

**Optimize cluster weights** $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \ldots, \alpha_k]^T$:
- Maximize Sharpe ratio, or
- Minimize variance for target return

**Constraint:** $\sum_{c=1}^{k} \alpha_c = 1$

### Step 4: Weight Combination

For asset $i$ in cluster $c$:

$$w_{\text{final},i} = \alpha_c \cdot w_{c,i}$$

### Mathematical Proof (Final Weights Sum to 1)

$$\sum_{i=1}^{n} w_{\text{final},i} = \sum_{c=1}^{k} \sum_{i \in \mathcal{C}_c} \alpha_c \cdot w_{c,i}$$

$$= \sum_{c=1}^{k} \alpha_c \cdot \left(\sum_{i \in \mathcal{C}_c} w_{c,i}\right)$$

$$= \sum_{c=1}^{k} \alpha_c \cdot 1 = \sum_{c=1}^{k} \alpha_c = 1$$

### Advantages of NCO

- **Dimensionality Reduction**: Optimize \(k\) clusters instead of \(n\) assets
- **Stability**: Avoids high-dimensional optimization problems
- **Flexibility**: Can use different methods within vs. between clusters
- **Captures Structure**: Uses both intra-cluster and inter-cluster relationships

---

## Common Properties Across All Methods

### Weight Constraints

All methods enforce:
$$\sum_{i=1}^{n} w_i = 1, \quad w_i \geq 0 \quad \forall i$$

### Portfolio Performance Metrics

All methods compute:
- **Expected Return**: $\mu_p = \mathbf{w}^T \boldsymbol{\mu}$
- **Volatility**: $\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$
- **Sharpe Ratio**: $\text{SR} = \frac{\mu_p - r_f}{\sigma_p}$

### Annualization

For daily returns (252 trading days per year):
- **Return**: $\mu_{p,\text{annual}} = \mu_p \times 252$
- **Volatility**: $\sigma_{p,\text{annual}} = \sigma_p \times \sqrt{252}$

---

## References

- De Prado, M. L. (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59-69.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *The Journal of Portfolio Management*, 36(4), 60-70.

---

## Quick Reference: Key Formulas

| Concept | Formula |
|--------|---------|
| Portfolio Return | $\mu_p = \mathbf{w}^T \boldsymbol{\mu}$ |
| Portfolio Variance | $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$ |
| Sharpe Ratio | $\text{SR} = \frac{\mu_p - r_f}{\sigma_p}$ |
| HRP Distance | $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ |
| HRP Split Weight | $\alpha = \frac{\sigma_R^2}{\sigma_L^2 + \sigma_R^2}$ |
| Risk Contribution | $\text{RC}_i = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$ |
| NCO Cluster Covariance | $\Sigma_{\text{cluster},ij} = \mathbf{w}_i^T \Sigma_{ij} \mathbf{w}_j$ |
| NCO Final Weight | $w_{\text{final},i} = \alpha_c \cdot w_{c,i}$ |

