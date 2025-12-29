# Financial Glossary

This glossary defines key financial and portfolio optimization terms used throughout the portfolio construction research notebooks.

---

## Portfolio Performance Metrics

### **Return (Annualized)**
- **Definition**: The average annual return of a portfolio, typically expressed as a percentage.
- **Calculation**: Daily returns are annualized by multiplying by 252 (trading days per year).
- **Interpretation**: Higher values indicate better performance, but must be considered alongside risk metrics.
- **Example**: A 10% annualized return means the portfolio would grow by 10% over one year on average.

### **Volatility (Annualized)**
- **Definition**: A measure of the dispersion of returns, representing the standard deviation of portfolio returns.
- **Calculation**: Daily volatility is annualized by multiplying by √252 (square root of trading days).
- **Interpretation**: Lower volatility indicates more stable, predictable returns. Higher volatility means greater price swings.
- **Risk Context**: Volatility is the primary risk metric - portfolios with higher volatility have greater uncertainty in returns.

### **Sharpe Ratio**
- **Definition**: A risk-adjusted return metric that measures excess return per unit of risk (volatility).
- **Formula**: `Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility`
- **Interpretation**: 
  - Higher values indicate better risk-adjusted returns
  - Values > 1.0 are generally considered good
  - Values > 2.0 are considered excellent
  - Negative values indicate the portfolio underperformed the risk-free rate
- **Use Case**: Compares portfolios on a risk-adjusted basis, helping identify which portfolios deliver better returns for the risk taken.

### **Information Ratio**
- **Definition**: Measures the risk-adjusted return of a portfolio relative to a benchmark (typically equal-weighted).
- **Formula**: `Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error`
- **Tracking Error**: The standard deviation of the difference between portfolio and benchmark returns.
- **Interpretation**:
  - Positive values indicate the portfolio outperformed the benchmark on a risk-adjusted basis
  - Higher values indicate better active management
  - Negative values indicate underperformance relative to the benchmark
- **Use Case**: Evaluates how well a portfolio strategy performs compared to a simple benchmark.

### **Max Drawdown**
- **Definition**: The largest peak-to-trough decline in portfolio value over a given time period.
- **Calculation**: Maximum percentage decline from any previous peak to the subsequent lowest point.
- **Interpretation**:
  - Lower (less negative) values indicate better downside protection
  - Represents the worst-case loss an investor would have experienced
  - Critical for understanding maximum potential loss
- **Example**: A -20% max drawdown means the portfolio fell 20% from its highest point to its lowest point.

### **Return/Risk Ratio**
- **Definition**: A simple risk-adjusted return measure calculated as return divided by volatility.
- **Formula**: `Return/Risk Ratio = Annualized Return / Annualized Volatility`
- **Interpretation**: Higher values indicate better return per unit of risk taken.
- **Note**: This is a simplified version of the Sharpe Ratio (without risk-free rate adjustment).

---

## Portfolio Optimization Methods

### **HRP (Hierarchical Risk Parity)**
- **Definition**: A portfolio construction method that uses hierarchical clustering to allocate risk equally across assets.
- **Key Features**:
  - Does not require return estimates (only uses covariance matrix)
  - More stable than traditional mean-variance optimization
  - Handles different asset classes with varying risk characteristics well
- **Use Case**: Ideal for multi-asset portfolios where return estimation is difficult or unreliable.

### **Risk Parity**
- **Definition**: A portfolio construction method that equalizes risk contribution from each asset.
- **Key Features**:
  - Allocates more capital to low-volatility assets and less to high-volatility assets
  - Achieves equal risk contribution rather than equal capital allocation
  - Avoids return estimation errors
- **Use Case**: Effective for portfolios with assets having vastly different volatilities.

### **NCO (Nested Clustered Optimization)**
- **Definition**: A portfolio construction method that first clusters assets, then optimizes within and across clusters.
- **Key Features**:
  - Combines clustering with optimization
  - More stable for large portfolios
  - Handles sector/asset class structure naturally
- **Use Case**: Suitable for large portfolios where assets can be naturally grouped (e.g., by sector or asset class).

### **RE-HRP (Return-Enhanced Hierarchical Risk Parity)**
- **Definition**: An extension of HRP that incorporates return information into the allocation process.
- **Key Features**:
  - Uses information ratio or Sharpe ratio to allocate between clusters
  - Balances risk parity principles with return enhancement
  - Can favor clusters/assets with better risk-adjusted returns
- **Use Case**: When you have confidence in return estimates and want to tilt toward better-performing assets.

### **Mean-Variance Optimization**
- **Definition**: The classic portfolio optimization framework that maximizes return for a given level of risk (or minimizes risk for a given return).
- **Key Features**:
  - Requires both expected returns and covariance matrix
  - Can produce extreme weights (concentration risk)
  - Sensitive to input estimation errors
- **Variants**:
  - **Max Sharpe**: Maximizes the Sharpe ratio (risk-adjusted return)
  - **Min Variance**: Minimizes portfolio volatility (lowest risk)

---

## Portfolio Modifications

### **Leveraged Portfolios (1.5x)**
- **Definition**: Portfolios where all weights are scaled by a leverage factor (e.g., 1.5x).
- **Effect**: 
  - Increases both expected returns and volatility proportionally
  - Maintains the same risk-adjusted characteristics (Sharpe ratio unchanged)
  - Increases maximum potential loss (larger drawdowns)
- **Use Case**: When investors want to amplify returns and are willing to accept higher risk.

### **Denoised Portfolios**
- **Definition**: Portfolios constructed using denoised covariance matrices to filter out estimation noise.
- **Method**: Uses Random Matrix Theory with the `constant_residual` approach to separate signal from noise in correlation matrices.
- **Benefits**:
  - More stable portfolio weights
  - Better out-of-sample performance
  - Reduces impact of estimation errors in covariance matrix
- **Use Case**: Particularly important for large portfolios where covariance matrix estimation becomes noisy.

---

## Additional Terms

### **Covariance Matrix**
- **Definition**: A matrix showing how asset returns move together (correlations) and their individual volatilities.
- **Importance**: Central input for most portfolio optimization methods.
- **Challenge**: With many assets, estimation becomes noisy and unreliable.

### **Correlation Matrix**
- **Definition**: A normalized version of the covariance matrix showing only how assets move together (ranging from -1 to +1).
- **Use**: Often denoised first, then rescaled back to covariance using individual asset volatilities.

### **Equal-Weighted Portfolio**
- **Definition**: A simple portfolio where each asset receives an equal allocation (1/N for N assets).
- **Use**: Often used as a benchmark to compare optimized portfolios against.

### **Tracking Error**
- **Definition**: The standard deviation of the difference between portfolio returns and benchmark returns.
- **Use**: Measures how closely a portfolio follows its benchmark. Lower tracking error means more similar performance.

### **Risk-Free Rate**
- **Definition**: The return on a risk-free investment (typically government bonds).
- **Use**: Used in Sharpe ratio calculations to measure excess return above the risk-free rate.
- **Typical Value**: Often assumed to be 2% annually (0.02).

---

## Reading the Analysis Tables

When interpreting the comprehensive risk-return analysis tables:

1. **Compare Sharpe Ratios**: Higher is better - shows which portfolios deliver better risk-adjusted returns.

2. **Check Information Ratios**: Positive values indicate outperformance vs. equal-weighted benchmark.

3. **Consider Volatility**: Lower volatility means less risk, but may come with lower returns.

4. **Review Max Drawdown**: Understand the worst-case loss scenario for each portfolio.

5. **Evaluate Return/Risk Ratio**: Simple measure of efficiency - how much return per unit of risk.

6. **Compare Variants**: 
   - Regular vs. Leveraged: Leveraged increases both return and risk proportionally
   - Regular vs. Denoised: Denoised may have more stable weights and better out-of-sample performance

---

## References

- **Modern Portfolio Theory**: Markowitz (1952)
- **Hierarchical Risk Parity**: Lopez de Prado (2016)
- **Covariance Matrix Denoising**: Random Matrix Theory applications
- **Risk Parity**: Qian (2005)

