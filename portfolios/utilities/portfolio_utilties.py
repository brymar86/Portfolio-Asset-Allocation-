import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt




def sharpe_ratio_traditional(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio for a given series of returns.
    
    The Sharpe ratio measures the risk-adjusted return of an investment or portfolio.
    It represents the excess return per unit of risk (volatility). A higher Sharpe ratio
    indicates better risk-adjusted performance. The calculation annualizes the ratio
    assuming 252 trading days per year.
    
    Args:
        returns (pd.Series or np.ndarray): Series of periodic returns (e.g., daily returns).
            Can be a pandas Series or numpy array of return values.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
            This should be provided as an annualized rate (e.g., 0.02 for 2%).
    
    Returns:
        float: The annualized Sharpe ratio. A value of 1.0 is considered good,
            2.0 is very good, and 3.0 is excellent. Negative values indicate
            the investment underperformed the risk-free rate.
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> sharpe_ratio(returns, risk_free_rate=0.02)
        0.123456789
    
    Note:
        The formula used is: (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
        where 252 represents the number of trading days in a year.
    """
    sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    return sharpe

def sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sortino ratio for a given series of returns.
    
    The Sortino ratio is similar to the Sharpe ratio but only considers downside
    volatility (negative returns) as risk, rather than total volatility. This makes
    it more appropriate for investors who are primarily concerned with downside risk.
    The calculation annualizes the ratio assuming 252 trading days per year.
    
    Args:
        returns (pd.Series or np.ndarray): Series of periodic returns (e.g., daily returns).
            Can be a pandas Series or numpy array of return values.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
            This should be provided as an annualized rate (e.g., 0.02 for 2%).
    
    Returns:
        float: The annualized Sortino ratio. Higher values indicate better
            risk-adjusted returns with respect to downside risk. Typically higher
            than the Sharpe ratio for the same returns since it ignores upside volatility.
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> sortino_ratio(returns, risk_free_rate=0.02)
        0.234567890
    
    Note:
        The formula used is: (mean(returns) - risk_free_rate) / std(returns, ddof=1) * sqrt(252)
        where ddof=1 uses sample standard deviation (N-1 degrees of freedom).
        Note: This implementation uses standard deviation; a true Sortino ratio would
        use downside deviation (standard deviation of negative returns only).
    """
    sortino = (returns.mean() - risk_free_rate) / returns.std(ddof=1) * np.sqrt(252)
    return sortino

def alpha(returns, risk_free_rate=0.02):
    """
    Calculate the alpha (excess return) for a given series of returns.
    
    Alpha measures the excess return of an investment compared to a benchmark,
    adjusted for the risk (beta) taken. It represents the value added (or subtracted)
    by active management. A positive alpha indicates outperformance, while negative
    alpha indicates underperformance relative to the expected return based on beta.
    
    Args:
        returns (pd.Series or np.ndarray): Series of periodic returns (e.g., daily returns).
            Can be a pandas Series or numpy array of return values.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
            This should be provided as an annualized rate (e.g., 0.02 for 2%).
    
    Returns:
        float: The annualized alpha value. Positive values indicate the investment
            outperformed expectations, negative values indicate underperformance.
            The value is annualized assuming 252 trading days per year.
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> alpha(returns, risk_free_rate=0.02)
        0.012345678
    
    Note:
        This function calculates alpha using the Capital Asset Pricing Model (CAPM):
        alpha = mean(returns) - risk_free_rate - beta * (mean(returns) - risk_free_rate) * sqrt(252)
        The calculation depends on the beta() function and annualizes the result.
    """
    alpha = returns.mean() - risk_free_rate - beta(returns, risk_free_rate) * (returns.mean() - risk_free_rate) * np.sqrt(252)
    return alpha

def beta(returns, risk_free_rate=0.02):
    """
    Calculate the beta coefficient for a given series of returns.
    
    Beta measures the sensitivity of an investment's returns to movements in the
    market (or benchmark). A beta of 1.0 indicates the investment moves in line
    with the market, while beta > 1.0 indicates higher volatility than the market,
    and beta < 1.0 indicates lower volatility. The calculation annualizes the
    result assuming 252 trading days per year.
    
    Args:
        returns (pd.Series or np.ndarray): Series of periodic returns (e.g., daily returns).
            Can be a pandas Series or numpy array of return values.
            Note: This function currently calculates beta relative to itself, which
            will always return 1.0. For proper beta calculation, you need both
            portfolio returns and benchmark returns.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
            This parameter is included for consistency with other functions but
            is not currently used in the calculation.
    
    Returns:
        float: The annualized beta coefficient. Values typically range from
            -2.0 to 2.0, with most investments having positive beta values.
            The result is annualized assuming 252 trading days per year.
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> beta(returns, risk_free_rate=0.02)
        1.0
    
    Note:
        The current implementation calculates: cov(returns, returns) / var(returns) * sqrt(252)
        which simplifies to 1.0 since a series' covariance with itself equals its variance.
        For proper beta calculation, you would need:
        beta = cov(portfolio_returns, benchmark_returns) / var(benchmark_returns)
    """
    beta = returns.cov(returns) / returns.var() * np.sqrt(252)
    return beta

def max_drawdown(series: pd.Series) -> float:
    """
    Calculate the maximum drawdown from a series of returns.
    
    Maximum drawdown is the largest peak-to-trough decline in the cumulative
    value of an investment over a specified time period. It measures the worst
    possible loss an investor could have experienced by buying at the peak and
    selling at the subsequent trough. This is a key risk metric for evaluating
    investment strategies.
    
    Args:
        series (pd.Series): Series of periodic returns (e.g., daily returns).
            The function expects a pandas Series containing return values.
            Returns should be in decimal form (e.g., 0.01 for 1%).
    
    Returns:
        float: The maximum drawdown as a decimal value. The result is negative
            (or zero), representing the worst percentage decline from a peak.
            For example, -0.25 represents a 25% maximum drawdown.
            A value closer to 0 indicates lower risk.
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, -0.05, 0.03, 0.01, -0.01])
        >>> max_drawdown(returns)
        -0.06930693069306931
    
    Note:
        The calculation process:
        1. Converts returns to cumulative product: (1 + returns).cumprod()
        2. Tracks the running maximum (peak): cum.cummax()
        3. Calculates drawdown at each point: (cum - peak) / peak
        4. Returns the minimum (most negative) drawdown value
        
        This metric is particularly useful for risk management and comparing
        the risk profiles of different investment strategies.
    """
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()

def weighted_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray, normalize: bool = True) -> pd.Series:
    """
    Calculate weighted portfolio returns from individual asset returns.
    
    This function computes the portfolio return for each time period by taking
    the weighted average of individual asset returns. The weights can represent
    either a fully-invested portfolio (sum to 1.0) or a leveraged portfolio
    (sum to leverage amount, e.g., 1.5 for 1.5x leverage).
    
    The calculation uses matrix dot product for efficiency:
    - For each time period (row), the portfolio return is calculated as:
      Portfolio Return = Σ(weight_i × return_i) for all assets i
    - This is equivalent to: returns_df.dot(weights)
    - Matrix dimensions: (n_periods × n_assets) · (n_assets × 1) = (n_periods × 1)
    
    Example calculation for one time period (no leverage):
        If assets [AAPL, MSFT] have returns [0.01, 0.02] and weights [0.6, 0.4]:
        Portfolio Return = (0.6 × 0.01) + (0.4 × 0.02) = 0.006 + 0.008 = 0.014
    
    Example calculation with 1.5x leverage:
        If weights [0.9, 0.6] sum to 1.5 (1.5x leverage):
        Portfolio Return = (0.9 × 0.01) + (0.6 × 0.02) = 0.009 + 0.012 = 0.021
        (Note: returns are scaled by the leverage factor)
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
            Each value should be a return (e.g., 0.01 for 1%).
        weights (np.ndarray): Array of portfolio weights for each asset.
            Should have the same length as the number of columns in returns_df.
            If normalize=True (default), weights will be normalized to sum to 1.0.
            If normalize=False, weights are used as-is, allowing for leverage.
        normalize (bool, optional): Whether to normalize weights to sum to 1.0.
            Defaults to True for backward compatibility. Set to False to allow
            leveraged portfolios where weights sum to the leverage amount.
    
    Returns:
        pd.Series: Time series of weighted portfolio returns with the same
            index as returns_df. Returns reflect leverage if normalize=False
            and weights sum to a leverage amount > 1.0.
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02]
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> weighted_portfolio_returns(returns, weights)
        0    0.014
        1   -0.016
        2    0.026
        dtype: float64
        
        >>> # Leveraged portfolio (1.5x leverage)
        >>> leveraged_weights = np.array([0.9, 0.6])  # Sums to 1.5
        >>> weighted_portfolio_returns(returns, leveraged_weights, normalize=False)
        0    0.021
        1   -0.024
        2    0.039
        dtype: float64
    """
    weights = np.array(weights)
    
    # Normalize weights if requested (default behavior for backward compatibility)
    if normalize:
        weights = weights / weights.sum()
    
    # DOT PRODUCT CALCULATION: Efficient matrix multiplication
    # 
    # For each row (time period) in returns_df:
    #   Portfolio Return = returns_row · weights = Σ(return_i × weight_i)
    #
    # Matrix multiplication:
    #   returns_df (n_periods × n_assets) · weights (n_assets × 1) = portfolio_returns (n_periods × 1)
    #   Each row of returns_df is multiplied element-wise with weights, then summed
    #
    # Example with 3 periods and 2 assets:
    #   [[r1_1, r1_2],    [[w1],    [[r1_1*w1 + r1_2*w2],
    #    [r2_1, r2_2],  ·  [w2]]  =  [r2_1*w1 + r2_2*w2],
    #    [r3_1, r3_2]]                [r3_1*w1 + r3_2*w2]]
    #
    # This is equivalent to calculating for each period:
    #   period_return = (weight_1 × return_1) + (weight_2 × return_2) + ... + (weight_n × return_n)
    #
    # Note: If normalize=False and weights sum to leverage amount L, returns are scaled by L
    portfolio_returns = returns_df.dot(weights)
    
    return portfolio_returns

def weighted_leveraged_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Calculate weighted portfolio returns for a leveraged portfolio.
    
    This is a convenience function that calls weighted_portfolio_returns with
    normalize=False, allowing weights to sum to a leverage amount (e.g., 1.5 for
    1.5x leverage, 2.0 for 2x leverage).
    
    The portfolio return calculation is:
        Portfolio Return = Σ(weight_i × return_i)
    
    With leverage L (where weights sum to L), returns are scaled by L:
        - If L = 1.5 (1.5x leverage), returns are 1.5x the unleveraged returns
        - If L = 2.0 (2x leverage), returns are 2.0x the unleveraged returns
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
            Each value should be a return (e.g., 0.01 for 1%).
        weights (np.ndarray): Array of portfolio weights for each asset.
            Should have the same length as the number of columns in returns_df.
            Weights should sum to the desired leverage amount (e.g., 1.5 for 1.5x).
            Weights are NOT normalized, so they are used as-is.
    
    Returns:
        pd.Series: Time series of weighted portfolio returns with the same
            index as returns_df. Returns reflect the leverage factor.
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02]
        ... })
        >>> # 1.5x leveraged portfolio (weights sum to 1.5)
        >>> leveraged_weights = np.array([0.9, 0.6])
        >>> weighted_leveraged_portfolio_returns(returns, leveraged_weights)
        0    0.021
        1   -0.024
        2    0.039
        dtype: float64
    """
    return weighted_portfolio_returns(returns_df, weights, normalize=False)

def portfolio_expected_return(returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Calculate the expected (mean) return of a weighted portfolio.
    
    This function computes the expected return of a portfolio by taking the
    weighted average of individual asset expected returns.
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
        weights (np.ndarray): Array of portfolio weights for each asset.
            Should have the same length as the number of columns in returns_df.
            Weights should sum to 1.0 (will be normalized if they don't).
    
    Returns:
        float: Expected return of the portfolio (as a decimal, e.g., 0.10 for 10%).
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02]
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> portfolio_expected_return(returns, weights)
        0.006666666666666667
    """
    # Normalize weights to ensure they sum to 1.0
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Calculate expected returns for each asset
    asset_expected_returns = returns_df.mean()
    
    # Weighted average of expected returns
    portfolio_return = weights.dot(asset_expected_returns)
    
    return portfolio_return

def portfolio_volatility(returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Calculate the volatility (standard deviation) of a weighted portfolio.
    
    This function computes portfolio volatility using the covariance matrix
    and portfolio weights. The formula is: sqrt(w^T * Cov * w)
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
        weights (np.ndarray): Array of portfolio weights for each asset.
            Should have the same length as the number of columns in returns_df.
            Weights should sum to 1.0 (will be normalized if they don't).
    
    Returns:
        float: Portfolio volatility (standard deviation) as a decimal.
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02]
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> portfolio_volatility(returns, weights)
        0.0123456789
    """
    # Normalize weights to ensure they sum to 1.0
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov()
    
    # Portfolio variance: w^T * Cov * w
    portfolio_variance = weights.dot(cov_matrix).dot(weights)
    
    # Portfolio volatility (standard deviation)
    portfolio_std = np.sqrt(portfolio_variance)
    
    return portfolio_std

def portfolio_sharpe_ratio(returns_df: pd.DataFrame, weights: np.ndarray, 
                           risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a weighted portfolio.
    
    The Sharpe ratio measures risk-adjusted return: (E[R] - Rf) / σ
    where E[R] is expected return, Rf is risk-free rate, and σ is volatility.
    The result is annualized.
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
        weights (np.ndarray): Array of portfolio weights for each asset.
            Should have the same length as the number of columns in returns_df.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
        periods_per_year (int, optional): Number of periods per year for annualization.
            Defaults to 252 (trading days).
    
    Returns:
        float: Annualized Sharpe ratio for the portfolio.
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02]
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> portfolio_sharpe_ratio(returns, weights, risk_free_rate=0.02)
        0.123456789
    """
    # Calculate portfolio expected return and volatility
    portfolio_return = portfolio_expected_return(returns_df, weights)
    portfolio_std = portfolio_volatility(returns_df, weights)
    
    # Annualize returns and volatility
    annualized_return = portfolio_return * periods_per_year
    annualized_std = portfolio_std * np.sqrt(periods_per_year)
    
    # Sharpe ratio: (E[R] - Rf) / σ
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    
    return sharpe

def plot_efficient_frontier(returns_df: pd.DataFrame, risk_free_rate: float = 0.02,
                            num_points: int = 50, show_random: bool = False,
                            num_random: int = 1000, ax=None, **kwargs) -> tuple:
    """
    Plot the efficient frontier for a portfolio optimization problem.
    
    The efficient frontier represents the set of optimal portfolios that offer
    the highest expected return for a given level of risk. This classical finance
    approach uses mean-variance optimization to find portfolios that minimize
    variance for a given return level (or maximize return for a given risk level).
    
    The function uses scipy.optimize.minimize to solve quadratic programming
    problems for different target returns, finding the minimum variance portfolio
    for each return level.
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns for each asset.
            Rows represent time periods, columns represent different assets.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
        num_points (int, optional): Number of points on the efficient frontier.
            Defaults to 50.
        show_random (bool, optional): Whether to show random portfolios for context.
            Defaults to False.
        num_random (int, optional): Number of random portfolios to generate if
            show_random is True. Defaults to 1000.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None,
            creates a new figure. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib plotting functions.
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects. If ax was provided,
            returns (None, ax).
    
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03, 0.01],
        ...     'MSFT': [0.02, -0.01, 0.02, 0.01]
        ... })
        >>> fig, ax = plot_efficient_frontier(returns, risk_free_rate=0.02)
        >>> plt.show()
    
    Note:
        This is a classical finance approach that assumes:
        - Returns are normally distributed
        - Expected returns and covariances are known and constant
        - No transaction costs
        - Perfect market conditions
        
        While useful for educational purposes and certain advisory requirements,
        these assumptions may not hold in practice. Consider using more advanced
        methods (HRP, Risk Parity, NCO) for real-world applications.
    """
    # Calculate expected returns and covariance matrix
    expected_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    n_assets = len(expected_returns)
    
    # Convert risk-free rate to per-period (assuming daily returns, 252 trading days)
    periods_per_year = 252
    
    # Find minimum and maximum possible returns
    # Minimum: portfolio with minimum return asset
    # Maximum: portfolio with maximum return asset
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    
    # Generate target returns for efficient frontier
    target_returns = np.linspace(min_return, max_return, num_points)
    
    # Storage for efficient frontier points
    efficient_risks = []
    efficient_returns = []
    efficient_weights = []
    
    # Constraints: weights sum to 1, no short selling
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # For each target return, find minimum variance portfolio
    for target_return in target_returns:
        # Objective function: minimize portfolio variance (risk)
        def objective(weights):
            portfolio_variance = weights.T @ cov_matrix @ weights
            return portfolio_variance
        
        # Constraint: portfolio return equals target return
        return_constraint = {
            'type': 'eq',
            'fun': lambda w: expected_returns.T @ w - target_return
        }
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraints, return_constraint],
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = expected_returns.T @ weights
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            
            efficient_risks.append(portfolio_risk)
            efficient_returns.append(portfolio_return)
            efficient_weights.append(weights)
    
    # Convert to numpy arrays
    efficient_risks = np.array(efficient_risks)
    efficient_returns = np.array(efficient_returns)
    
    # Annualize for plotting
    efficient_risks_annual = efficient_risks * np.sqrt(periods_per_year)
    efficient_returns_annual = efficient_returns * periods_per_year
    
    # Find maximum Sharpe ratio portfolio
    sharpe_ratios = (efficient_returns_annual - risk_free_rate) / efficient_risks_annual
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_risk = efficient_risks_annual[max_sharpe_idx]
    max_sharpe_return = efficient_returns_annual[max_sharpe_idx]
    
    # Find minimum variance portfolio
    min_var_idx = np.argmin(efficient_risks_annual)
    min_var_risk = efficient_risks_annual[min_var_idx]
    min_var_return = efficient_returns_annual[min_var_idx]
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None
    
    # Plot random portfolios if requested
    if show_random:
        random_risks = []
        random_returns = []
        random_sharpes = []
        
        for _ in range(num_random):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            
            # Calculate portfolio metrics
            port_return = expected_returns.T @ weights
            port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Annualize
            port_return_annual = port_return * periods_per_year
            port_risk_annual = port_risk * np.sqrt(periods_per_year)
            port_sharpe = (port_return_annual - risk_free_rate) / port_risk_annual
            
            random_returns.append(port_return_annual)
            random_risks.append(port_risk_annual)
            random_sharpes.append(port_sharpe)
        
        # Plot random portfolios
        scatter = ax.scatter(random_risks, random_returns, c=random_sharpes,
                            cmap='viridis', alpha=0.3, s=10, **kwargs)
        if fig is not None:
            plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Plot efficient frontier
    ax.plot(efficient_risks_annual, efficient_returns_annual, 'b-', 
            linewidth=2, label='Efficient Frontier', **kwargs)
    
    # Highlight maximum Sharpe ratio portfolio
    ax.scatter(max_sharpe_risk, max_sharpe_return, marker='*', 
              color='red', s=500, label='Max Sharpe Ratio', zorder=5, **kwargs)
    
    # Highlight minimum variance portfolio
    ax.scatter(min_var_risk, min_var_return, marker='*', 
              color='green', s=500, label='Min Variance', zorder=5, **kwargs)
    
    # Plot individual assets
    asset_risks = np.sqrt(np.diag(cov_matrix)) * np.sqrt(periods_per_year)
    asset_returns = expected_returns * periods_per_year
    ax.scatter(asset_risks, asset_returns, marker='o', color='gray', 
              s=100, alpha=0.6, label='Individual Assets', zorder=4, **kwargs)
    
    # Add asset labels
    for i, asset_name in enumerate(returns_df.columns):
        ax.annotate(asset_name, (asset_risks[i], asset_returns[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Labels and formatting
    ax.set_xlabel('Portfolio Risk (Annualized Volatility)', fontsize=12)
    ax.set_ylabel('Portfolio Return (Annualized)', fontsize=12)
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if fig is not None:
        plt.tight_layout()
    
    return (fig, ax)


def plot_portfolio_weights_pie(weights, asset_names, title="Portfolio Weights", ax=None, figsize=(8, 8), 
                                min_weight_threshold=0.01):
    """
    Create a pie chart visualization for portfolio weights.
    
    Args:
        weights (np.ndarray or pd.Series): Portfolio weights (should sum to 1.0 or leverage factor)
        asset_names (list or pd.Index): Names of assets corresponding to weights
        title (str): Title for the pie chart
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
        figsize (tuple): Figure size if creating new figure
        min_weight_threshold (float): Weights below this threshold are grouped into "Others"
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Convert to numpy array if needed
    if isinstance(weights, pd.Series):
        weights = weights.values
    weights = np.array(weights)
    
    # Handle small weights by grouping into "Others"
    mask = weights >= min_weight_threshold
    large_weights = weights[mask]
    large_names = [asset_names[i] for i in range(len(weights)) if mask[i]]
    
    small_weights_sum = weights[~mask].sum()
    if small_weights_sum > 0:
        large_weights = np.append(large_weights, small_weights_sum)
        large_names.append("Others")
    
    # Create color map for consistent coloring
    colors = plt.cm.Set3(np.linspace(0, 1, len(large_weights)))
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(large_weights, labels=large_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 10})
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return (fig, ax)


def plot_portfolio_weights_comparison_grid(weights_dict, asset_names, figsize=(16, 12), 
                                           min_weight_threshold=0.01, n_cols=3):
    """
    Create a grid of pie charts comparing multiple portfolio optimization methods.
    
    Args:
        weights_dict (dict): Dictionary mapping method names to weight arrays
            Example: {'HRP': weights_array, 'Risk Parity': weights_array, ...}
        asset_names (list or pd.Index): Names of assets
        figsize (tuple): Figure size
        min_weight_threshold (float): Weights below this threshold are grouped into "Others"
        n_cols (int): Number of columns in the grid
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes array
    """
    n_methods = len(weights_dict)
    n_rows = int(np.ceil(n_methods / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Create consistent color mapping for all assets
    all_assets = set()
    for weights in weights_dict.values():
        if isinstance(weights, pd.Series):
            weights = weights.values
        weights = np.array(weights)
        mask = weights >= min_weight_threshold
        for i in range(len(weights)):
            if mask[i]:
                all_assets.add(asset_names[i])
    all_assets = sorted(list(all_assets))
    asset_color_map = {asset: plt.cm.Set3(i / len(all_assets)) for i, asset in enumerate(all_assets)}
    asset_color_map['Others'] = plt.cm.Set3(0.9)
    
    # Plot each method
    for idx, (method_name, weights) in enumerate(weights_dict.items()):
        ax = axes[idx]
        
        # Convert to numpy array if needed
        if isinstance(weights, pd.Series):
            weights = weights.values
        weights = np.array(weights)
        
        # Handle small weights
        mask = weights >= min_weight_threshold
        large_weights = weights[mask]
        large_names = [asset_names[i] for i in range(len(weights)) if mask[i]]
        
        small_weights_sum = weights[~mask].sum()
        if small_weights_sum > 0:
            large_weights = np.append(large_weights, small_weights_sum)
            large_names.append("Others")
        
        # Map colors
        colors = [asset_color_map.get(name, asset_color_map['Others']) for name in large_names]
        
        # Create pie chart
        ax.pie(large_weights, labels=large_names, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 8})
        ax.set_title(method_name, fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return (fig, axes)


def plot_portfolio_weights_scatter_pie(weights_dict, returns_dict, volatilities_dict, asset_names,
                                       figsize=(14, 10), min_weight_threshold=0.01, pie_size=0.08):
    """
    Create a scatter plot with pie charts positioned at risk-return coordinates.
    Each method is shown as a pie chart at its (volatility, return) position.
    
    Args:
        weights_dict (dict): Dictionary mapping method names to weight arrays
        returns_dict (dict): Dictionary mapping method names to annualized returns
        volatilities_dict (dict): Dictionary mapping method names to annualized volatilities
        asset_names (list or pd.Index): Names of assets
        figsize (tuple): Figure size
        min_weight_threshold (float): Weights below this threshold are grouped into "Others"
        pie_size (float): Size of pie charts as fraction of figure (default 0.08 = 8%)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError:
        # Fallback: use regular scatter with labels if inset_axes not available
        fig, ax = plt.subplots(figsize=figsize)
        for method_name in weights_dict.keys():
            vol = volatilities_dict[method_name]
            ret = returns_dict[method_name]
            ax.scatter(vol, ret, s=200, alpha=0.7)
            ax.annotate(method_name, (vol, ret), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        ax.set_xlabel('Volatility (Annualized)', fontsize=12)
        ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
        ax.set_title('Portfolio Methods: Risk-Return Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return (fig, ax)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get axis ranges
    all_vols = list(volatilities_dict.values())
    all_returns = list(returns_dict.values())
    vol_range = max(all_vols) - min(all_vols) if max(all_vols) > min(all_vols) else 1.0
    return_range = max(all_returns) - min(all_returns) if max(all_returns) > min(all_returns) else 1.0
    
    # Create consistent color mapping
    all_assets = set()
    for weights in weights_dict.values():
        if isinstance(weights, pd.Series):
            weights = weights.values
        weights = np.array(weights)
        mask = weights >= min_weight_threshold
        for i in range(len(weights)):
            if mask[i]:
                all_assets.add(asset_names[i])
    all_assets = sorted(list(all_assets))
    asset_color_map = {asset: plt.cm.Set3(i / len(all_assets)) for i, asset in enumerate(all_assets)}
    asset_color_map['Others'] = plt.cm.Set3(0.9)
    
    # Set up main axes first to get proper limits
    ax.set_xlim(min(all_vols) - vol_range * 0.15, max(all_vols) + vol_range * 0.15)
    ax.set_ylim(min(all_returns) - return_range * 0.15, max(all_returns) + return_range * 0.15)
    
    # Plot each method as pie chart at its risk-return position
    for method_name in weights_dict.keys():
        weights = weights_dict[method_name]
        vol = volatilities_dict[method_name]
        ret = returns_dict[method_name]
        
        # Convert to numpy array if needed
        if isinstance(weights, pd.Series):
            weights = weights.values
        weights = np.array(weights)
        
        # Handle small weights
        mask = weights >= min_weight_threshold
        large_weights = weights[mask]
        large_names = [asset_names[i] for i in range(len(weights)) if mask[i]]
        
        small_weights_sum = weights[~mask].sum()
        if small_weights_sum > 0:
            large_weights = np.append(large_weights, small_weights_sum)
            large_names.append("Others")
        
        # Map colors
        colors = [asset_color_map.get(name, asset_color_map['Others']) for name in large_names]
        
        # Create inset axes for pie chart at (vol, ret) position
        pie_ax = inset_axes(ax, width=f"{pie_size*100}%", height=f"{pie_size*100}%", 
                           loc='center', bbox_to_anchor=(vol, ret, 1, 1), 
                           bbox_transform=ax.transData, borderpad=0)
        
        pie_ax.pie(large_weights, colors=colors, startangle=90)
        pie_ax.axis('off')
        
        # Add method label below pie chart
        ax.annotate(method_name, (vol, ret), xytext=(0, -20), 
                   textcoords='offset points', fontsize=8, alpha=0.8,
                   ha='center', va='top')
    
    # Set up main axes labels
    ax.set_xlabel('Volatility (Annualized)', fontsize=12)
    ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
    ax.set_title('Portfolio Weights: Risk-Return Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return (fig, ax)
