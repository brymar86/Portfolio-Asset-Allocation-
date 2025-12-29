"""
Return-Enhanced Hierarchical Risk Parity (RE-HRP) Portfolio Optimizer.

This module implements a modification of De Prado's Hierarchical Risk Parity (HRP)
algorithm that incorporates return information through risk-adjusted return maximization
(default: Information Ratio). RE-HRP preserves HRP's mathematically sound clustering structure
(based on correlation distance metrics) while allocating capital based on risk-adjusted
returns rather than pure risk parity.

**IMPORTANT ATTRIBUTION**: This implementation extends the research of Marcos Lopez de Prado.
The original HRP algorithm and mathematical foundations are from his 2016 paper. This
extension preserves HRP's clustering structure but allocates based on risk-adjusted returns
(Information Ratio by default, with Sharpe Ratio and Sortino Ratio as options).

References:
    De Prado, M. L. (2016). Building Diversified Portfolios that Outperform
    Out of Sample. The Journal of Portfolio Management, 42(4), 59-69.
    
    DOI: https://doi.org/10.3905/jpm.2016.42.4.059
    
    Sharpe, W. F. (1966). Mutual fund performance. Journal of Business, 39(1), 119-138.
    
    Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside
    risk framework. The Journal of Investing, 3(3), 59-64.
    
    Key Differences from HRP:
    - Clustering: Same correlation-based distance metric (preserved)
    - Allocation: Uses Information Ratio (default), Sharpe Ratio, or Sortino Ratio instead of inverse variance
    - Formula: α = IR_right / (IR_left + IR_right) instead of α = σ²_right / (σ²_left + σ²_right)
      where IR = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import Optional, Literal
from .base_optimizer import BasePortfolioOptimizer


class ReturnEnhancedHRP(BasePortfolioOptimizer):
    """
    Return-Enhanced Hierarchical Risk Parity (RE-HRP) Portfolio Optimizer.
    
    RE-HRP extends HRP by incorporating return information through risk-adjusted return
    maximization (default: Information Ratio). It preserves HRP's clustering structure
    (correlation-based distance) but allocates capital based on risk-adjusted returns
    rather than pure risk parity.
    
    The algorithm:
    1. Builds hierarchical clustering tree from correlation matrix (same as HRP)
    2. Quasi-diagonalizes the covariance matrix (same as HRP)
    3. Sets up benchmark returns (equal-weighted portfolio by default)
    4. Recursively allocates weights using Information Ratio (default), Sharpe Ratio,
       or Sortino Ratio instead of inverse variance
    
    Information Ratio (Default):
    - Compares sub-portfolios against an equal-weighted benchmark
    - Formula: IR = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error
    - Favors clusters that generate excess returns relative to benchmark
    - Avoids over-weighting low-volatility assets with modest returns
    
    Attributes:
        linkage_matrix_ (np.ndarray): Linkage matrix from hierarchical clustering.
        tree_order_ (np.ndarray): Order of assets after quasi-diagonalization.
        cov_quasi_diag_ (pd.DataFrame): Quasi-diagonalized covariance matrix.
        expected_returns_ (np.ndarray): Expected returns for each asset.
        returns_array_ (np.ndarray): Historical returns array (needed for risk-adjusted ratio calculation).
        benchmark_returns_ (np.ndarray): Benchmark returns array (equal-weighted by default).
    """
    
    def __init__(self, linkage_method: str = 'ward', risk_free_rate: float = 0.02,
                 target_return: float = 0.0, allocation_metric: str = 'information_ratio',
                 benchmark_returns: Optional[pd.Series] = None, verbose: bool = False,
                 min_return_threshold: Optional[float] = None, min_tracking_error: float = 0.01,
                 denoise: bool = False,
                 denoising_method: Literal["constant_residual", "targeted_shrinkage", "eigenvalue_clipping"] = "constant_residual"):
        """
        Initialize RE-HRP optimizer.
        
        Args:
            linkage_method (str, optional): Linkage method for hierarchical clustering.
                Options: 'ward', 'single', 'complete', 'average'. Defaults to 'ward'.
            risk_free_rate (float, optional): Annual risk-free rate for risk-adjusted ratio calculation.
                Defaults to 0.02 (2%).
            target_return (float, optional): Target return for downside deviation calculation
                (only used with Sortino Ratio). Defaults to 0.0.
            allocation_metric (str, optional): Risk-adjusted return metric to use for allocation.
                Options: 'information_ratio' (default), 'sharpe', or 'sortino'. Defaults to 'information_ratio'.
                - 'information_ratio': Uses Information Ratio vs equal-weighted benchmark (favors return clusters)
                - 'sharpe': Uses total volatility (more stable, aligns with HRP's variance approach)
                - 'sortino': Uses downside volatility only (focuses on downside risk)
            benchmark_returns (pd.Series, optional): Benchmark returns for Information Ratio calculation.
                If None, uses equal-weighted portfolio of all assets. Defaults to None.
            verbose (bool, optional): If True, prints detailed Information Ratio calculations during allocation.
                Defaults to False.
            min_return_threshold (float, optional): Minimum return threshold for Information Ratio allocation.
                If None (default), uses benchmark return as threshold. If float, absolute minimum return
                (e.g., 0.05 for 5% annualized). Prevents low-return assets from being over-weighted.
                Defaults to None.
            min_tracking_error (float, optional): Minimum tracking error floor to prevent tiny denominators
                from inflating Information Ratio. Defaults to 0.01 (1% annualized).
            denoise (bool, optional): If True, apply covariance matrix denoising before
                optimization. Denoising removes random noise from the eigenvalue spectrum
                using Random Matrix Theory. Defaults to False.
            denoising_method (str, optional): Denoising method to use when denoise=True.
                Options: 'constant_residual' (default/standard), 'targeted_shrinkage',
                or 'eigenvalue_clipping'. Defaults to 'constant_residual'.
        """
        super().__init__()
        if allocation_metric not in ['information_ratio', 'sharpe', 'sortino']:
            raise ValueError("allocation_metric must be 'information_ratio', 'sharpe', or 'sortino'")
        self.linkage_method = linkage_method
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.allocation_metric = allocation_metric
        self.benchmark_returns = benchmark_returns
        self.verbose = verbose
        self.min_return_threshold = min_return_threshold
        self.min_tracking_error = min_tracking_error
        self.denoise = denoise
        self.denoising_method = denoising_method
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.tree_order_: Optional[np.ndarray] = None
        self.cov_quasi_diag_: Optional[pd.DataFrame] = None
        self.expected_returns_: Optional[np.ndarray] = None
        self.returns_array_: Optional[np.ndarray] = None
        self.benchmark_returns_: Optional[np.ndarray] = None
    
    def fit(self, returns_df: pd.DataFrame) -> 'ReturnEnhancedHRP':
        """
        Fit the RE-HRP optimizer on historical returns.
        
        This method:
        1. Computes correlation and covariance matrices
        2. Converts correlation to distance matrix
        3. Builds hierarchical clustering tree
        4. Quasi-diagonalizes the covariance matrix
        5. Sets up benchmark returns (equal-weighted if not provided)
        6. Computes RE-HRP weights through recursive allocation using Information Ratio
           (default), Sharpe Ratio, or Sortino Ratio
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
                Each value should be a return (e.g., 0.01 for 1%).
        
        Returns:
            ReturnEnhancedHRP: Returns self for method chaining.
        """
        # Validate and store returns
        returns_df = self._validate_returns(returns_df)
        self.returns_df = returns_df.copy()
        self.asset_names_ = list(returns_df.columns)
        
        # Compute expected returns and store returns array
        self.expected_returns_ = returns_df.mean().values
        self.returns_array_ = returns_df.values
        
        # Set up benchmark returns for Information Ratio (if selected)
        if self.allocation_metric == 'information_ratio':
            if self.benchmark_returns is None:
                # Default: equal-weighted portfolio of all assets
                equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                self.benchmark_returns_ = returns_df.values @ equal_weights
            else:
                # Use provided benchmark returns
                if len(self.benchmark_returns) != len(returns_df):
                    raise ValueError("benchmark_returns must have same length as returns_df")
                self.benchmark_returns_ = self.benchmark_returns.values if isinstance(self.benchmark_returns, pd.Series) else np.array(self.benchmark_returns)
            
            # Print Information Ratio allocation header with warning
            print("=" * 80)
            print("RE-HRP: Information Ratio Allocation")
            print("=" * 80)
            if self.benchmark_returns is None:
                print(f"Benchmark: Equal-weighted portfolio of {len(returns_df.columns)} assets")
            else:
                print(f"Benchmark: Custom benchmark returns (length: {len(self.benchmark_returns_)})")
            print(f"Allocation Metric: Information Ratio")
            print(f"Formula: α = IR_right / (IR_left + IR_right)")
            print(f"Information Ratio = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error")
            print("=" * 80)
            print("WARNING: Information Ratio vs equal-weighted benchmark may favor low-volatility")
            print("assets if the benchmark performs well. Consider using 'sharpe' instead.")
            print("=" * 80)
        elif self.allocation_metric == 'sharpe':
            # Print Sharpe Ratio allocation header
            print("=" * 80)
            print("RE-HRP: Sharpe Ratio Allocation (Recommended)")
            print("=" * 80)
            print(f"Allocation Metric: Sharpe Ratio")
            print(f"Formula: α = SR_right / (SR_left + SR_right)")
            print(f"Sharpe Ratio = (E[R] - r_f) / σ_total")
            print(f"Risk-free rate: {self.risk_free_rate:.2%}")
            print("=" * 80)
        
        # Compute covariance and correlation matrices
        self.cov_matrix_ = self._compute_covariance(returns_df)
        self.corr_matrix_ = self._compute_correlation(returns_df)
        
        # Apply denoising if requested
        if self.denoise:
            from portfolios.utilities.denoising import CovarianceDenoiser
            denoiser = CovarianceDenoiser()
            num_observations = len(returns_df)
            
            # Extract volatilities (standard deviations) from covariance matrix
            volatilities = pd.Series(
                np.sqrt(np.diag(self.cov_matrix_.values)),
                index=self.cov_matrix_.index
            )
            
            # Denoise correlation matrix (recommended approach: MP assumes isotropic noise)
            corr_denoised = denoiser.denoise(
                self.corr_matrix_,
                method=self.denoising_method,
                num_observations=num_observations,
                matrix_type="correlation"
            )
            
            # Rescale denoised correlation to covariance: Σ = D R̂ D
            self.cov_matrix_ = denoiser._rescale_to_covariance(corr_denoised, volatilities)
            self.corr_matrix_ = corr_denoised
        
        # Step 1: Convert correlation matrix to distance matrix
        # Distance: d = sqrt(2 * (1 - corr))
        # This ensures that highly correlated assets are close together
        distance_matrix = np.sqrt(2 * (1 - self.corr_matrix_.values))
        
        # Step 2: Build hierarchical clustering tree
        # Convert distance matrix to condensed form for linkage
        condensed_distances = squareform(distance_matrix, checks=False)
        self.linkage_matrix_ = linkage(condensed_distances, method=self.linkage_method)
        
        # Step 3: Get the order of assets from the dendrogram
        # This order will be used to quasi-diagonalize the covariance matrix
        self.tree_order_ = leaves_list(self.linkage_matrix_)
        
        # Step 4: Quasi-diagonalize the covariance matrix
        self.cov_quasi_diag_ = self._quasi_diagonalize(self.cov_matrix_, self.tree_order_)
        
        # Reorder returns array to match quasi-diagonal order
        returns_quasi_order = self.returns_array_[:, self.tree_order_]
        expected_returns_quasi_order = self.expected_returns_[self.tree_order_]
        
        # Step 5: Recursively allocate weights using risk-adjusted return metric
        # (Information Ratio by default, Sharpe Ratio, or Sortino Ratio if specified)
        # Weights are computed in quasi-diagonal order
        weights_quasi_order = self._recursive_allocation(
            self.cov_quasi_diag_.values,
            returns_quasi_order,
            expected_returns_quasi_order
        )
        
        # Reorder weights back to original asset order
        original_order = np.argsort(self.tree_order_)
        self.weights_ = weights_quasi_order[original_order]
        
        # Validate and normalize weights
        self.weights_ = self._validate_weights(self.weights_)
        
        # Print allocation summary
        if self.verbose or self.allocation_metric == 'information_ratio':
            print("\n" + "=" * 80)
            print("RE-HRP Allocation Summary")
            print("=" * 80)
            if self.allocation_metric == 'information_ratio':
                print(f"Allocation complete. Portfolio weights favor clusters with higher Information Ratios.")
            elif self.allocation_metric == 'sharpe':
                print(f"Allocation complete. Portfolio weights favor clusters with higher Sharpe Ratios.")
            else:
                print(f"Allocation complete. Portfolio weights favor clusters with higher Sortino Ratios.")
            print(f"Top weighted assets:")
            # Show top 5 assets by weight
            weights_dict = dict(zip(self.asset_names_, self.weights_))
            sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (asset, weight) in enumerate(sorted_weights[:5], 1):
                print(f"  {i}. {asset}: {weight:.4f} ({weight*100:.2f}%)")
            print("=" * 80)
        
        return self
    
    def _quasi_diagonalize(self, cov_matrix: pd.DataFrame, order: np.ndarray) -> pd.DataFrame:
        """
        Quasi-diagonalize the covariance matrix based on hierarchical tree order.
        
        This reorders the covariance matrix so that similar assets (according to
        the clustering tree) are placed near each other, creating a quasi-diagonal
        structure that makes recursive allocation more effective.
        
        Args:
            cov_matrix (pd.DataFrame): Original covariance matrix.
            order (np.ndarray): Order of assets from hierarchical clustering.
        
        Returns:
            pd.DataFrame: Quasi-diagonalized covariance matrix.
        """
        # Reorder rows and columns according to tree order
        cov_reordered = cov_matrix.iloc[order, :].iloc[:, order]
        return cov_reordered
    
    def _compute_sharpe_ratio(self, portfolio_returns: np.ndarray,
                              risk_free_rate: float,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio for a portfolio return series.
        
        Sharpe Ratio = (E[R] - r_f) / σ_total
        
        where:
        - E[R] is the expected return (annualized)
        - r_f is the risk-free rate (annualized)
        - σ_total is the total volatility (annualized standard deviation)
        
        Args:
            portfolio_returns (np.ndarray): Array of portfolio returns (per period).
            risk_free_rate (float): Annual risk-free rate.
            periods_per_year (int, optional): Number of periods per year for annualization.
                Defaults to 252 (trading days).
        
        Returns:
            float: Annualized Sharpe Ratio. Returns 0.0 if volatility is zero.
        """
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Calculate expected return (per period)
        expected_return_period = np.mean(portfolio_returns)
        
        # Annualize expected return
        expected_return_annual = expected_return_period * periods_per_year
        
        # Calculate total volatility (standard deviation)
        volatility_period = np.std(portfolio_returns, ddof=1)
        
        # Annualize volatility
        volatility_annual = volatility_period * np.sqrt(periods_per_year)
        
        # Handle edge case: zero volatility
        if volatility_annual < 1e-10:
            if expected_return_annual > risk_free_rate:
                return 1e6  # Very high Sharpe (perfect risk-free return)
            elif expected_return_annual < risk_free_rate:
                return -1e6  # Very low Sharpe (negative excess return)
            else:
                return 0.0  # Exactly risk-free return
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (expected_return_annual - risk_free_rate) / volatility_annual
        
        return sharpe_ratio
    
    def _compute_sortino_ratio(self, portfolio_returns: np.ndarray,
                               risk_free_rate: float, target_return: float,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio for a portfolio return series.
        
        Sortino Ratio = (E[R] - r_f) / σ_downside
        
        where:
        - E[R] is the expected return (annualized)
        - r_f is the risk-free rate (annualized)
        - σ_downside is the downside deviation (annualized)
        
        Downside deviation: σ_downside = sqrt(E[min(0, R - target)²])
        
        Args:
            portfolio_returns (np.ndarray): Array of portfolio returns (per period).
            risk_free_rate (float): Annual risk-free rate.
            target_return (float): Target return for downside deviation (per period).
            periods_per_year (int, optional): Number of periods per year for annualization.
                Defaults to 252 (trading days).
        
        Returns:
            float: Annualized Sortino Ratio. Returns large positive value if downside
                deviation is zero (perfect downside protection).
        """
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Calculate expected return (per period)
        expected_return_period = np.mean(portfolio_returns)
        
        # Annualize expected return
        expected_return_annual = expected_return_period * periods_per_year
        
        # Calculate downside deviation
        # Only penalize returns below target
        downside_returns = np.minimum(0, portfolio_returns - target_return)
        downside_variance = np.mean(downside_returns ** 2)
        downside_deviation_period = np.sqrt(downside_variance)
        
        # Annualize downside deviation
        downside_deviation_annual = downside_deviation_period * np.sqrt(periods_per_year)
        
        # Handle edge case: zero downside deviation (perfect downside protection)
        if downside_deviation_annual < 1e-10:
            # Return large positive value to favor this portfolio
            if expected_return_annual > risk_free_rate:
                return 1e6  # Very high Sortino (perfect downside protection)
            else:
                return -1e6  # Very low Sortino (negative excess return)
        
        # Calculate Sortino Ratio
        sortino_ratio = (expected_return_annual - risk_free_rate) / downside_deviation_annual
        
        return sortino_ratio
    
    def _compute_information_ratio(self, portfolio_returns: np.ndarray,
                                   benchmark_returns: np.ndarray,
                                   periods_per_year: int = 252) -> float:
        """
        Calculate Information Ratio for a portfolio return series vs benchmark.
        
        Information Ratio = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error
        
        where:
        - E[R_portfolio] is the expected portfolio return (annualized)
        - E[R_benchmark] is the expected benchmark return (annualized)
        - Tracking_Error = σ(excess_returns) * sqrt(252) (annualized), with minimum floor applied
        - excess_returns = portfolio_returns - benchmark_returns
        
        The Information Ratio measures risk-adjusted excess return relative to a benchmark.
        Higher Information Ratios indicate better risk-adjusted performance vs the benchmark.
        
        Note: Tracking error floor (min_tracking_error) is applied to prevent tiny denominators
        from inflating IR for low-volatility assets.
        
        Args:
            portfolio_returns (np.ndarray): Array of portfolio returns (per period).
            benchmark_returns (np.ndarray): Array of benchmark returns (per period).
                Must have same length as portfolio_returns.
            periods_per_year (int, optional): Number of periods per year for annualization.
                Defaults to 252 (trading days).
        
        Returns:
            float: Annualized Information Ratio. Returns 0.0 if tracking error is zero.
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("portfolio_returns and benchmark_returns must have same length")
        
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
        # This prevents low-volatility assets (like TLT) from getting excessive weights
        tracking_error_annual = max(tracking_error_annual, self.min_tracking_error)
        
        # Handle edge case: zero tracking error (shouldn't happen after floor, but keep for safety)
        if tracking_error_annual < 1e-10:
            if expected_excess_annual > 0:
                return 1e6  # Very high IR (perfect excess return with no tracking error)
            elif expected_excess_annual < 0:
                return -1e6  # Very low IR (negative excess return with no tracking error)
            else:
                return 0.0  # Exactly matches benchmark
        
        # Calculate Information Ratio
        information_ratio = expected_excess_annual / tracking_error_annual
        
        return information_ratio
    
    def _recursive_allocation(self, cov_matrix: np.ndarray,
                            returns_array: np.ndarray,
                            expected_returns: np.ndarray) -> np.ndarray:
        """
        Recursively allocate weights down the hierarchical tree using risk-adjusted return metric.
        
        This is the core RE-HRP algorithm. It recursively splits the portfolio
        and allocates weights proportionally to Information Ratio (default), Sharpe Ratio,
        or Sortino Ratio, favoring sub-portfolios with better risk-adjusted returns.
        
        For Information Ratio allocation, minimum return threshold is checked to prevent
        low-return assets from being over-weighted. If threshold not met, falls back to
        return-based allocation.
        
        Key Difference from HRP:
        - HRP: α = σ²_right / (σ²_left + σ²_right) (inverse variance weighting)
        - RE-HRP: α = IR_right / (IR_left + IR_right) (Information Ratio weighting, default)
          or α = SR_right / (SR_left + SR_right) (Sharpe/Sortino Ratio weighting)
          where IR is Information Ratio (default), SR is Sharpe Ratio, or Sortino Ratio
        
        Edge Case Handling:
        - Single asset: returns weight of 1.0 (base case)
        - Both risk-adjusted ratios ≤ 0: use equal weights (0.5)
        - One risk-adjusted ratio ≤ 0: allocate all to positive one (clamped to [0.1, 0.9])
        - Zero volatility/deviation/tracking error: treated as perfect (high ratio)
        - Return threshold not met: falls back to return-based allocation
        
        Args:
            cov_matrix (np.ndarray): Quasi-diagonalized covariance matrix.
            returns_array (np.ndarray): Returns array in quasi-diagonal order.
                Shape: (n_periods, n_assets)
            expected_returns (np.ndarray): Expected returns in quasi-diagonal order.
                Shape: (n_assets,)
        
        Returns:
            np.ndarray: Portfolio weights in quasi-diagonal order.
        """
        n = len(cov_matrix)
        
        if n == 1:
            # Base case: single asset gets full weight
            return np.array([1.0])
        
        # Split the covariance matrix into two sub-matrices
        # Find the split point that minimizes the variance of the split
        split_idx = self._find_split_point(cov_matrix)
        
        # Split into left and right sub-matrices
        cov_left = cov_matrix[:split_idx, :split_idx]
        cov_right = cov_matrix[split_idx:, split_idx:]
        
        # Split returns and expected returns
        returns_left = returns_array[:, :split_idx]
        returns_right = returns_array[:, split_idx:]
        expected_returns_left = expected_returns[:split_idx]
        expected_returns_right = expected_returns[split_idx:]
        
        # Recursively allocate weights for left and right sub-portfolios
        weights_left = self._recursive_allocation(
            cov_left, returns_left, expected_returns_left
        )
        weights_right = self._recursive_allocation(
            cov_right, returns_right, expected_returns_right
        )
        
        # Calculate portfolio returns for each sub-portfolio
        # R_portfolio = w^T @ R_assets (for each time period)
        portfolio_returns_left = returns_left @ weights_left
        portfolio_returns_right = returns_right @ weights_right
        
        # Calculate risk-adjusted ratios for each sub-portfolio
        if self.allocation_metric == 'information_ratio':
            # Use Information Ratio (default) - compare vs benchmark
            portfolio_return_left = portfolio_returns_left.mean() * 252
            portfolio_return_right = portfolio_returns_right.mean() * 252
            benchmark_return = self.benchmark_returns_.mean() * 252
            threshold = self.min_return_threshold if self.min_return_threshold is not None else benchmark_return
            
            # Check if clusters/assets meet return threshold
            left_is_single = len(expected_returns_left) == 1
            right_is_single = len(expected_returns_right) == 1
            
            # Determine if left/right meet threshold (check individual assets for single-asset clusters)
            left_meets = (expected_returns_left[0] * 252 >= threshold) if left_is_single else (
                portfolio_return_left >= threshold and 
                weights_left[(expected_returns_left * 252 < threshold)].sum() <= 0.5
            )
            right_meets = (expected_returns_right[0] * 252 >= threshold) if right_is_single else (
                portfolio_return_right >= threshold and 
                weights_right[(expected_returns_right * 252 < threshold)].sum() <= 0.5
            )
            
            # Determine allocation ratios based on threshold checks (4 cases)
            if left_meets and right_meets:
                # Both meet: Use Information Ratio
                ratio_left = self._compute_information_ratio(portfolio_returns_left, self.benchmark_returns_)
                ratio_right = self._compute_information_ratio(portfolio_returns_right, self.benchmark_returns_)
            elif left_meets and not right_meets:
                # Only left meets: Favor left, heavily penalize single assets below threshold
                ratio_left = 1.0
                ratio_right = 0.001 if right_is_single else 0.1
            elif right_meets and not left_meets:
                # Only right meets: Favor right, heavily penalize single assets below threshold
                ratio_left = 0.001 if left_is_single else 0.1
                ratio_right = 1.0
            else:
                # Neither meets: Return-based allocation, heavily penalize single assets
                if left_is_single:
                    ratio_left = 0.001
                    ratio_right = portfolio_return_right if portfolio_return_right > 0 else 1.0
                elif right_is_single:
                    ratio_left = portfolio_return_left if portfolio_return_left > 0 else 1.0
                    ratio_right = 0.001
                else:
                    # Both clusters: return-based
                    total_return = portfolio_return_left + portfolio_return_right
                    ratio_left = portfolio_return_left if total_return > 0 else 0.5
                    ratio_right = portfolio_return_right if total_return > 0 else 0.5
            
            # Verbose logging
            if self.verbose:
                left_desc = f"single asset ({expected_returns_left[0]*252:.4%})" if left_is_single else f"cluster ({portfolio_return_left:.4%})"
                right_desc = f"single asset ({expected_returns_right[0]*252:.4%})" if right_is_single else f"cluster ({portfolio_return_right:.4%})"
                status = "Both meet" if (left_meets and right_meets) else \
                         "Left meets" if left_meets else \
                         "Right meets" if right_meets else "Neither meets"
                print(f"  Threshold check: {status} (Left {left_desc}, Right {right_desc}, threshold={threshold:.4%})")
        elif self.allocation_metric == 'sharpe':
            # Use Sharpe Ratio
            ratio_left = self._compute_sharpe_ratio(
                portfolio_returns_left,
                self.risk_free_rate
            )
            ratio_right = self._compute_sharpe_ratio(
                portfolio_returns_right,
                self.risk_free_rate
            )
        else:
            # Use Sortino Ratio
            target_return_period = self.target_return / 252 if self.target_return != 0 else 0.0
            ratio_left = self._compute_sortino_ratio(
                portfolio_returns_left,
                self.risk_free_rate,
                target_return_period
            )
            ratio_right = self._compute_sortino_ratio(
                portfolio_returns_right,
                self.risk_free_rate,
                target_return_period
            )
        
        # Allocate based on risk-adjusted return ratio
        # α = ratio_right / (ratio_left + ratio_right)
        total_ratio = ratio_left + ratio_right
        
        if self.verbose:
            metric_name = 'IR' if self.allocation_metric == 'information_ratio' else ('SR' if self.allocation_metric == 'sharpe' else 'Sortino')
            print(f"  Cluster split: {split_idx} assets left, {n-split_idx} assets right")
            print(f"  {metric_name}_left: {ratio_left:.4f}, {metric_name}_right: {ratio_right:.4f}")
        
        if total_ratio <= 0:
            # Both ratios are negative or zero, use equal weights
            alpha = 0.5
        elif ratio_left <= 0:
            # Only right has positive ratio, allocate more to right
            # Clamp to [0.1, 0.9] for diversification
            alpha = 0.9
        elif ratio_right <= 0:
            # Only left has positive ratio, allocate more to left
            # Clamp to [0.1, 0.9] for diversification
            alpha = 0.1
        else:
            # Both have positive ratios, allocate proportionally
            alpha = ratio_right / total_ratio
            
            # Clamp alpha to avoid extreme values (numerical stability and diversification)
            alpha = np.clip(alpha, 0.1, 0.9)
        
        if self.verbose and self.allocation_metric == 'information_ratio':
            print(f"  Allocation α: {alpha:.4f} ({(1-alpha)*100:.1f}% left, {alpha*100:.1f}% right)")
        
        # Combine weights
        # Note: alpha = ratio_right / (ratio_left + ratio_right)
        # Higher ratio_right → higher alpha → we want right to get more weight
        # So we use (1-alpha) for left and alpha for right (opposite of HRP's inverse variance)
        weights = np.concatenate([
            (1 - alpha) * weights_left,
            alpha * weights_right
        ])
        
        return weights
    
    def _find_split_point(self, cov_matrix: np.ndarray) -> int:
        """
        Find the optimal split point in the quasi-diagonalized covariance matrix.
        
        The split point is chosen to minimize the variance of the resulting
        sub-portfolios. This is done by trying all possible split points and
        selecting the one that minimizes the combined variance.
        
        This method is identical to HRP's implementation, preserving the clustering structure.
        
        Edge Case Handling:
        - Two assets: returns split index of 1 (trivial case)
        - Numerical precision: ensures variances are non-negative (clipped to 1e-12 minimum)
        
        Args:
            cov_matrix (np.ndarray): Quasi-diagonalized covariance matrix.
        
        Returns:
            int: Optimal split index (between 1 and n-1).
        """
        n = len(cov_matrix)
        
        # Handle edge case: 2 assets
        if n == 2:
            return 1
        
        min_variance = np.inf
        best_split = n // 2  # Default to middle
        
        # Try all possible split points
        for i in range(1, n):
            # Split into left and right
            cov_left = cov_matrix[:i, :i]
            cov_right = cov_matrix[i:, i:]
            
            # Calculate variance of each sub-portfolio with equal weights
            weights_left = np.ones(i) / i
            weights_right = np.ones(n - i) / (n - i)
            
            var_left = weights_left.T @ cov_left @ weights_left
            var_right = weights_right.T @ cov_right @ weights_right
            
            # Handle numerical precision: ensure variances are non-negative
            var_left = max(var_left, 1e-12)
            var_right = max(var_right, 1e-12)
            
            # Combined variance (weighted by sub-portfolio sizes)
            combined_var = (i / n) * var_left + ((n - i) / n) * var_right
            
            if combined_var < min_variance:
                min_variance = combined_var
                best_split = i
        
        return best_split
    
    def predict(self) -> np.ndarray:
        """
        Generate portfolio weights after fitting.
        
        Returns:
            np.ndarray: Array of portfolio weights for each asset in original order.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.weights_ is None:
            raise ValueError("Must call fit() before predict()")
        
        # Weights are already in original order after fit()
        return self.weights_
    
    def plot_dendrogram(self, ax=None, **kwargs):
        """
        Plot the hierarchical clustering dendrogram.
        
        This visualization shows how assets are clustered together in the
        hierarchical tree structure used by RE-HRP (same as HRP).
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None,
                creates a new figure. Defaults to None.
            **kwargs: Additional arguments passed to dendrogram().
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call fit() before plot_dendrogram()")
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = None
        
        # Create labels from asset names
        labels = [self.asset_names_[i] for i in self.tree_order_] if self.asset_names_ else None
        
        dendrogram(
            self.linkage_matrix_,
            labels=labels,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=10,
            **kwargs
        )
        
        ax.set_title('Hierarchical Clustering Tree (RE-HRP)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        if fig is not None:
            plt.tight_layout()
        
        return (fig, ax) if fig is not None else (None, ax)

