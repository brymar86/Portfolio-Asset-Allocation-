# Risk Parity (Equal Risk Contribution) Portfolio Optimizer

This document describes the **Risk Parity** (also called **Equal Risk Contribution, ERC**) portfolio construction method with **LaTeX formatted for typical Jupyter/Markdown renderers** using `$...$` (inline) and `$$...$$` (display).

Risk Parity chooses portfolio weights so that **each asset contributes the same amount of risk** to total portfolio risk. Unlike mean–variance optimization, Risk Parity does **not** require expected return estimates, which can make it more robust when return forecasts are noisy.

---

## 1) Notation and setup

Let there be $n$ assets.

- Weight vector:  
  $$\mathbf{w}=
  \begin{bmatrix}
  w_1 \\ \vdots \\ w_n
  \end{bmatrix}$$

- Covariance matrix of asset returns (symmetric positive semidefinite):  
  $$\boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}$$

- Portfolio variance and volatility:  
  $$\sigma_p^2(\mathbf{w})=\mathbf{w}^\top \boldsymbol{\Sigma}\mathbf{w},\qquad
  \sigma_p(\mathbf{w})=\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma}\mathbf{w}}$$

Implementation mapping (exactly what you compute in code):

- `port_var = w.T @ Sigma @ w` corresponds to $\,\sigma_p^2(\mathbf{w})=\mathbf{w}^\top \boldsymbol{\Sigma}\mathbf{w}$
- `port_vol = sqrt(port_var)` corresponds to $\,\sigma_p(\mathbf{w})=\sqrt{\sigma_p^2(\mathbf{w})}$

---

## 2) What does “risk contribution” mean?

Risk Parity is defined in terms of each asset’s **contribution** to total portfolio risk. Two related objects matter:

### 2.1 Marginal Contribution to Risk (MRC)

The marginal contribution to risk of asset $i$ is the partial derivative of portfolio volatility with respect to its weight:

$$\mathrm{MRC}_i(\mathbf{w})=\frac{\partial \sigma_p(\mathbf{w})}{\partial w_i}$$

Start from:

$$\sigma_p(\mathbf{w})=\left(\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\right)^{1/2}$$

The gradient is:

$$\nabla_{\mathbf{w}}\sigma_p(\mathbf{w})
=
\frac{\boldsymbol{\Sigma}\mathbf{w}}{\sigma_p(\mathbf{w})}$$

So for asset $i$:

$$\mathrm{MRC}_i(\mathbf{w})=\frac{\left(\boldsymbol{\Sigma}\mathbf{w}\right)_i}{\sigma_p(\mathbf{w})}$$

### 2.2 Total Risk Contribution (RC)

Define the **total** contribution of asset $i$ to portfolio volatility as:

$$\mathrm{RC}_i(\mathbf{w})=w_i\cdot \mathrm{MRC}_i(\mathbf{w})
=
w_i\cdot \frac{\left(\boldsymbol{\Sigma}\mathbf{w}\right)_i}{\sigma_p(\mathbf{w})}$$

A key identity (useful as a sanity check in code):

$$\sum_{i=1}^{n}\mathrm{RC}_i(\mathbf{w})=\sigma_p(\mathbf{w})$$

> If your code computes `RC = w * (Sigma @ w) / port_vol`, then `RC.sum()` should be extremely close to `port_vol` (assuming `port_vol > 0`).

---

## 3) The Risk Parity / ERC condition

### 3.1 Equal Risk Contribution (ERC)

ERC requires:

$$\mathrm{RC}_1(\mathbf{w})=\mathrm{RC}_2(\mathbf{w})=\cdots=\mathrm{RC}_n(\mathbf{w})$$

Since the contributions sum to $\sigma_p(\mathbf{w})$, the equal target is:

$$\mathrm{RC}_i(\mathbf{w})=\frac{1}{n}\sigma_p(\mathbf{w})\qquad \forall i$$

### 3.2 General “risk budgets” (optional but common)

A generalization assigns each asset a **risk budget** $b_i$ such that:

$$b_i\ge 0,\qquad \sum_{i=1}^{n} b_i = 1$$

and then targets:

$$\mathrm{RC}_i(\mathbf{w})=b_i\,\sigma_p(\mathbf{w})$$

ERC is the special case $b_i=1/n$.

### 3.3 Equivalent condition without square roots (variance domain)

Because $\mathrm{RC}_i(\mathbf{w})=\dfrac{w_i(\Sigma w)_i}{\sigma_p}$, the budgeted condition
$\mathrm{RC}_i=b_i\sigma_p$ is equivalent to:

$$w_i\left(\boldsymbol{\Sigma}\mathbf{w}\right)_i
=
b_i\left(\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\right)$$

This version is often helpful for debugging and for avoiding numerical issues near $\sigma_p\approx 0$.

---

## 4) The constrained optimization problem (typical long-only implementation)

In practice, we solve ERC as a constrained nonlinear optimization.

### 4.1 Constraints (long-only, fully invested, max weight)

A common institutional long-only setup:

- Fully invested:  
  $$\sum_{i=1}^{n} w_i = 1$$

- Long-only:  
  $$w_i \ge 0\qquad \forall i$$

- Concentration cap (example):  
  $$w_i \le w_{\max}\qquad \forall i$$

Example: $w_{\max}=0.50$.

### 4.2 Objective: match risk contributions to targets

Define target risk contributions:

$$\mathrm{RC}_i^\star(\mathbf{w})=b_i\,\sigma_p(\mathbf{w})$$

and minimize squared deviations:

$$\min_{\mathbf{w}}
\;\;
\sum_{i=1}^{n}\left(\mathrm{RC}_i(\mathbf{w})-\mathrm{RC}_i^\star(\mathbf{w})\right)^2$$

For ERC budgets $b_i=1/n$:

$$\min_{\mathbf{w}}
\;\;
\sum_{i=1}^{n}\left(\mathrm{RC}_i(\mathbf{w})-\frac{1}{n}\sigma_p(\mathbf{w})\right)^2$$

> This objective is smooth (when $\sigma_p>0$) but generally **non-convex**. SLSQP tends to work well in practice given good initialization and reasonable bounds.

---

## 5) Leverage: what it means, how it changes the math, and why it’s tricky

Most “naive” implementations assume **no leverage**: $\sum_i w_i = 1$ and $w_i\ge 0$. If you add leverage, be explicit about which leverage definition you mean.

### 5.1 Case A — Levered long-only exposure (borrow cash, scale risky weights)

You allow total risky exposure $L\ge 1$ (financed by borrowing cash or using derivatives):

- Exposure constraint:  
  $$\sum_{i=1}^{n} w_i = L,\qquad L\ge 1$$
- Still long-only:  
  $$w_i \ge 0$$

**Scaling property (important):** if you scale a portfolio by a positive constant $c$:

$$\mathbf{w}' = c\,\mathbf{w}$$

then:

$$\sigma_p(\mathbf{w}') = c\,\sigma_p(\mathbf{w})$$

and risk contributions scale the same way:

$$\mathrm{RC}_i(\mathbf{w}') = c\,\mathrm{RC}_i(\mathbf{w})$$

So if $\mathbf{w}$ satisfies ERC under $\sum w_i=1$, then $L\mathbf{w}$ satisfies ERC under $\sum w_i=L$.

**What changes in practice:**
- The **risk level** increases linearly with $L$.
- Weight caps can bind differently when $\sum w_i=L>1$.
- You may want a **volatility target** constraint, e.g.:

$$\sigma_p(\mathbf{w}) \le \sigma_{\text{target}}$$

or a penalty term to make “risk targeting” explicit.

### 5.2 Case B — Gross leverage with long/short

If you allow short positions, leverage is often defined by **gross exposure**:

$$\|\mathbf{w}\|_1=\sum_{i=1}^{n}|w_i| = L_{\text{gross}}$$

This changes interpretation:
- Some $w_i$ are negative, and “contribution” can be negative depending on convention.
- Many practitioners switch to variance contributions or sign-adjusted definitions.
- You need additional constraints (borrow limits, margin, turnover, etc.).

If your goal is a clean institutional RP baseline, Case A is the usual first step.

### 5.3 Practical risk management considerations with leverage

Even though ERC is scale-invariant, leverage introduces real-world risks:

- financing cost / carry,
- margin and forced deleveraging risk,
- regime shifts (correlations rise in stress),
- tail risk amplification.

A common “clean extension” is:
1. Solve ERC under $\sum w_i=1$
2. Scale to hit a volatility target:

$$\mathbf{w}_{\text{scaled}}=\frac{\sigma_{\text{target}}}{\sigma_p(\mathbf{w})}\mathbf{w}$$

3. Cap the scale factor (max leverage) and enforce per-asset caps.

---

## 6) Numerical optimization details (how to make scripts trustworthy)

### 6.1 Why SLSQP is a reasonable choice

SLSQP supports:
- equality constraints like $\sum_i w_i=1$ (or $L$),
- bound constraints like $0\le w_i \le w_{\max}$,
- smooth nonlinear objectives.

### 6.2 Typical algorithm outline

1. Inputs: $\Sigma$, bounds, budgets $\mathbf{b}$ (optional), tolerance
2. Initialization: equal weights or inverse-vol
3. Compute $\sigma_p$, $\Sigma\mathbf{w}$, and $\mathrm{RC}$
4. Optimize with constraints
5. Post-checks:
   - feasibility: sums and bounds
   - ERC quality: dispersion of $\mathrm{RC}$
   - stability: no NaNs/inf, $\sigma_p>0$
6. Fallback: retry with different initializations if needed

### 6.3 Numerical edge cases

- Near-singular $\Sigma$ can cause instability  
  → consider shrinkage or $\Sigma+\epsilon I$
- If $\sigma_p\approx 0$, $\mathrm{RC}$ divides by near-zero  
  → clamp $\sigma_p \leftarrow \max(\sigma_p,\epsilon)$ in code
- Bounds too tight may make ERC infeasible  
  → optimizer returns closest feasible solution

---

## 7) Validation checks (recommended)

Given final weights $\hat{\mathbf{w}}$:

1. Constraints:
   - $\sum_i \hat{w}_i \approx 1$ (or $L$)
   - $0\le \hat{w}_i \le w_{\max}$

2. Risk contributions:
   - compute $\hat{\sigma}_p$ and $\widehat{\mathrm{RC}}_i$
   - verify $\sum_i \widehat{\mathrm{RC}}_i \approx \hat{\sigma}_p$
   - verify near-equality: $\widehat{\mathrm{RC}}_i \approx \hat{\sigma}_p/n$ (or $b_i\hat{\sigma}_p$)

3. Sensitivity:
   - perturb $\Sigma$ slightly and confirm weights/RC behave reasonably

---

## 8) Reference

- Maillard, S., Roncalli, T., & Teiletche, J. (2010). *The Properties of Equally Weighted Risk Contribution Portfolios.* The Journal of Portfolio Management, 36(4), 60–70.

---

## 10) In Practice: Code Implementation Walkthrough

### 10.7 Code-to-Math Mapping Summary

| Mathematical Concept | Code Location | Implementation |
|---------------------|---------------|----------------|
| $\boldsymbol{\Sigma}$ | Line ~230 | `self.cov_matrix_ = self._compute_covariance(returns_df)` |
| $\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma}\mathbf{w}$ | Line ~262 | `portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)` |
| $(\boldsymbol{\Sigma}\mathbf{w})_i$ | Line ~268 | `marginal_contrib = cov_matrix @ weights` |
| $\mathrm{RC}_i = w_i \cdot (\boldsymbol{\Sigma}\mathbf{w})_i / \sigma_p$ | Line ~269 | `risk_contributions = weights * marginal_contrib / portfolio_vol` |
| $\sum_i w_i = 1$ | Line ~248 | `{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}` |
| $w_i \ge 0$ | Line ~250 | `bounds = tuple((0, 1) for _ in range(n_assets))` |
| $\min \sum_i (\mathrm{RC}_i - \sigma_p/n)^2$ | Line ~338-342 | `target_rc = portfolio_vol / n_assets; return np.sum((risk_contributions - target_rc)**2)` |

### 10.8 Short Positions and Leverage: Extending Beyond Long-Only

**Current Implementation:** The base implementation enforces long-only constraints ($w_i \ge 0$), which is appropriate for most institutional investors. However, allowing short positions can improve diversification and risk-adjusted returns, especially when assets have negative correlations.

**Short Positions (Long/Short Portfolio):**

To allow short positions, we need to modify the constraints:

**Mathematical Changes:**
- Remove long-only constraint: Change bounds from $w_i \ge 0$ to $w_i \in \mathbb{R}$ (or $w_i \ge -w_{\text{short}}$ with a short limit)
- Gross exposure constraint: $\sum_{i=1}^{n} |w_i| = L_{\text{gross}}$ where $L_{\text{gross}} \ge 1$
- Net exposure constraint: $\sum_{i=1}^{n} w_i = L_{\text{net}}$ (typically $L_{\text{net}} = 1$ for market-neutral, or $L_{\text{net}} > 1$ for leveraged long)
- Margin requirements: $\sum_{i: w_i < 0} |w_i| \le M$ (maximum short exposure)

**Risk Contribution with Shorts:**

When $w_i < 0$, the risk contribution formula remains the same:
$$\mathrm{RC}_i = w_i \cdot \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p}$$

However, negative weights can lead to negative risk contributions, which means the asset **reduces** portfolio risk. This is mathematically valid and occurs when:
- Asset has negative correlation with the portfolio
- Short position acts as a hedge

**Key Identity Still Holds:**
$$\sum_{i=1}^{n} \mathrm{RC}_i = \sigma_p$$

Even with short positions, risk contributions still sum to portfolio volatility.

**Leverage (Long-Only with Borrowing):**

For leveraged long-only portfolios (Case A from Section 5.1):

**Mathematical Changes:**
- Exposure constraint: $\sum_{i=1}^{n} w_i = L$ where $L > 1$ (e.g., $L = 1.5$ for 1.5x leverage)
- Still long-only: $w_i \ge 0$ for all $i$
- Risk contributions scale: $\mathrm{RC}_i = w_i \cdot (\boldsymbol{\Sigma}\mathbf{w})_i / \sigma_p$ where $\sigma_p$ now reflects leveraged exposure

**Scaling Property:**
If $\mathbf{w}$ satisfies ERC under $\sum w_i = 1$, then $L\mathbf{w}$ satisfies ERC under $\sum w_i = L$, and:
$$\sigma_p(L\mathbf{w}) = L \cdot \sigma_p(\mathbf{w})$$
$$\mathrm{RC}_i(L\mathbf{w}) = L \cdot \mathrm{RC}_i(\mathbf{w})$$

**Implementation Considerations:**

1. **Bounds Modification:**
   ```python
   # Long-only (current):
   bounds = tuple((0, 1) for _ in range(n_assets))
   
   # Long/short (with short limit):
   max_short = 0.5  # Maximum 50% short per asset
   bounds = tuple((-max_short, 1) for _ in range(n_assets))
   ```

2. **Constraint Modification:**
   ```python
   # Long-only fully invested:
   constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
   
   # Leveraged long-only:
   leverage = 1.5
   constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - leverage}]
   
   # Long/short market-neutral:
   constraints = [
       {'type': 'eq', 'fun': lambda w: np.sum(w)},  # Net exposure = 0
       {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - gross_exposure}  # Gross exposure
   ]
   ```

3. **Risk Contribution Calculation:**
   The formula remains unchanged, but negative weights produce negative contributions:
   ```python
   # Works for both positive and negative weights
   risk_contributions = weights * marginal_contrib / portfolio_vol
   ```

4. **Objective Function:**
   For long/short, we may want to target **absolute** risk contributions:
   ```python
   # Option 1: Target absolute RC (standard)
   target_rc = portfolio_vol / n_assets
   diff = risk_contributions - target_rc
   
   # Option 2: Target absolute value of RC (for long/short)
   target_rc_abs = portfolio_vol / n_assets
   diff = np.abs(risk_contributions) - target_rc_abs
   ```

**Practical Considerations:**

1. **Margin Requirements:** Short positions require margin, typically 50% of short value
2. **Regulatory Constraints:** Many institutional investors cannot short
3. **Transaction Costs:** Shorting often has higher costs (borrowing fees, bid-ask spreads)
4. **Risk Management:** Short positions can lead to unlimited losses if not properly hedged
5. **Correlation Breakdown:** In stress periods, correlations can spike, reducing diversification benefits

**When to Use Shorts:**

- **Market-Neutral Strategies:** Long/short portfolios with $\sum w_i = 0$ can target pure alpha
- **Hedging:** Short positions can hedge specific risks
- **Diversification:** When assets have negative correlations, shorts can improve risk-adjusted returns
- **Volatility Targeting:** Shorts can help achieve specific volatility targets

**When to Avoid Shorts:**

- **Regulatory Restrictions:** Many funds cannot short
- **High Transaction Costs:** Shorting costs may outweigh benefits
- **Limited Liquidity:** Some assets are difficult/expensive to short
- **Risk Management Complexity:** Requires more sophisticated risk systems

**Future Implementation:**

The current implementation focuses on long-only portfolios for simplicity and regulatory compliance. To add short support:

1. Add `allow_shorts: bool = False` parameter to `__init__`
2. Add `max_short: float = 0.5` parameter for maximum short per asset
3. Add `leverage: float = 1.0` parameter for leveraged long-only
4. Modify bounds and constraints based on these parameters
5. Update post-processing to handle negative weights appropriately
6. Add tests to verify risk contributions still sum correctly with shorts

This extension would make the optimizer more flexible while maintaining mathematical correctness.

### 10.9 Why This Implementation is Correct

**Mathematical Correctness:**
1. **Risk contributions sum to portfolio volatility**: The code computes $\mathrm{RC}_i$ exactly as defined, so $\sum_i \mathrm{RC}_i = \sigma_p$ by mathematical identity
2. **Constraints are satisfied**: The optimizer enforces constraints, and post-processing ensures they remain satisfied
3. **Objective matches theory**: The objective function directly implements the mathematical formula

**Numerical Robustness:**
1. **Handles edge cases**: Near-zero volatility, optimization failures, corner solutions
2. **Fallback strategies**: Multiple initialization strategies ensure we always get valid weights
3. **Post-processing**: Ensures practical diversification even if optimization produces extreme weights

**Verification Through Tests:**
The test suite in `tests/test_portfolio_optimizers.py` verifies:
- Weights sum to 1.0 (constraint satisfaction)
- Weights are non-negative (long-only constraint)
- Risk contributions are calculated correctly
- Portfolio performance metrics are valid

**Key Insight:** The code is a direct translation of the mathematical theory. Each mathematical operation has a corresponding code operation, making it easy to verify correctness and debug issues.
