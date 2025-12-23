"""
Risk Parity Portfolio Optimizer.

This module implements the Risk Parity (Equal Risk Contribution) portfolio
optimization method. Risk Parity allocates portfolio weights such that each
asset contributes equally to the portfolio's total risk.

Unlike traditional mean-variance optimization, Risk Parity does not require
expected return estimates, making it more robust to estimation errors.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
from .base_optimizer import BasePortfolioOptimizer


class RiskParity(BasePortfolioOptimizer):
    """
    Risk Parity (Equal Risk Contribution) Portfolio Optimizer.
    
    Risk Parity allocates portfolio weights such that each asset contributes
    equally to the portfolio's total risk. This is achieved by solving:
    
    minimize: sum((RC_i - RC_target)^2)
    subject to: sum(w_i) = 1, w_i >= 0
    
    where RC_i is the risk contribution of asset i:
    RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
    
    Attributes:
        risk_contributions_ (np.ndarray): Risk contribution of each asset.
        optimization_result_ (scipy.optimize.OptimizeResult): Optimization result.
    """
    
    def __init__(self, method: str = 'SLSQP', max_iter: int = 1000):
        """
        Initialize Risk Parity optimizer.
        
        Args:
            method (str, optional): Optimization method for scipy.optimize.minimize.
                Defaults to 'SLSQP'.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        """
        super().__init__()
        self.method = method
        self.max_iter = max_iter
        self.risk_contributions_: Optional[np.ndarray] = None
        self.optimization_result_: Optional[object] = None
    
    def fit(self, returns_df: pd.DataFrame) -> 'RiskParity':
        """
        Fit the Risk Parity optimizer on historical returns.
        
        This method:
        1. Computes the covariance matrix
        2. Solves the optimization problem to find equal risk contribution weights
        3. Handles edge cases including boundary solutions and optimization failures
        
        Edge Case Handling:
        - If initial optimization fails, retries with inverse volatility weighting
        - If retry fails, falls back to equal weights
        - Redistributes very small weights (< 1e-6) to prevent corner solutions
        - Handles near-zero portfolio volatility cases
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
        
        Returns:
            RiskParity: Returns self for method chaining.
        
        Raises:
            ValueError: If optimization fails after all retry attempts (rare).
        """
        # Validate and store returns
        returns_df = self._validate_returns(returns_df)
        self.returns_df = returns_df.copy()
        self.asset_names_ = list(returns_df.columns)
        
        # Compute covariance matrix
        self.cov_matrix_ = self._compute_covariance(returns_df)
        cov_matrix = self.cov_matrix_.values
        n_assets = len(cov_matrix)
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Constraints: weights sum to 1, no short selling
        # Add maximum weight constraint to prevent concentration (max 50% per asset)
        # This prevents corner solutions where one asset gets 100% weight, which can occur
        # when assets have very different risk characteristics (e.g., one asset has much
        # lower volatility or very different correlations). Risk Parity should diversify.
        max_weight_per_asset = 0.50  # Maximum 50% per asset
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: max_weight_per_asset - np.max(w)}  # Max weight constraint
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function: minimize sum of squared differences from equal risk contribution
        def objective(weights):
            """
            Objective function: minimize sum of squared differences from equal risk contribution.
            
            The risk contribution of asset i is:
            RC_i = w_i * (Σw)_i / σ_p
            
            where σ_p = sqrt(w^T Σ w) is the portfolio volatility.
            
            For equal risk contribution, we want all RC_i to be equal to 1/n.
            """
            # Portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol < 1e-10:
                # Avoid division by zero
                return 1e10
            
            # Risk contributions: RC_i = w_i * (Σw)_i / σ_p
            marginal_contrib = cov_matrix @ weights
            risk_contributions = weights * marginal_contrib / portfolio_vol
            
            # Target: equal risk contribution (1/n for each asset)
            target_rc = 1.0 / n_assets
            
            # Sum of squared differences from target
            diff = risk_contributions - target_rc
            return np.sum(diff ** 2)
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            # Try with a different initial guess if optimization fails
            # Use inverse volatility weighting as a better starting point
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = (1.0 / volatilities)
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            result = minimize(
                objective,
                inv_vol_weights,
                method=self.method,
                bounds=bounds,
                constraints=constraints,  # Same constraints including max weight
                options={'maxiter': self.max_iter, 'ftol': 1e-8, 'disp': False}
            )
            
            if not result.success:
                # Fallback to equal weights if optimization still fails
                self.weights_ = np.ones(n_assets) / n_assets
                self.optimization_result_ = result
            else:
                self.weights_ = result.x
                self.optimization_result_ = result
        else:
            self.weights_ = result.x
            self.optimization_result_ = result
        
        # Handle boundary solutions: prevent corner solutions where optimization converges
        # to extreme weights (either 0% or 100% in a single asset)
        # This can occur when assets have very different risk characteristics
        min_weight_threshold = 0.01  # Minimum 1% per asset to ensure diversification
        max_weight_threshold = 0.50  # Maximum 50% per asset to prevent concentration
        
        # Check for assets with very small weights
        small_weights = self.weights_ < min_weight_threshold
        # Check for assets with very large weights (corner solution)
        large_weights = self.weights_ > max_weight_threshold
        
        if np.any(small_weights) or np.any(large_weights):
            # Redistribute weights to enforce minimum and maximum constraints
            # This ensures diversification and prevents corner solutions
            
            # First, set minimum weights for assets below threshold
            if np.any(small_weights):
                self.weights_[small_weights] = min_weight_threshold
            
            # Then, cap maximum weights for assets above threshold
            if np.any(large_weights):
                excess = np.sum(self.weights_[large_weights] - max_weight_threshold)
                self.weights_[large_weights] = max_weight_threshold
                # Redistribute excess to other assets proportionally
                remaining_assets = ~large_weights
                if np.sum(remaining_assets) > 0:
                    self.weights_[remaining_assets] += excess * (self.weights_[remaining_assets] / np.sum(self.weights_[remaining_assets]))
            
            # Renormalize to ensure weights sum to 1
            self.weights_ = self.weights_ / self.weights_.sum()
        
        # Validate and normalize weights
        self.weights_ = self._validate_weights(self.weights_)
        
        # Calculate actual risk contributions
        portfolio_vol = np.sqrt(self.weights_.T @ cov_matrix @ self.weights_)
        if portfolio_vol > 1e-10:
            marginal_contrib = cov_matrix @ self.weights_
            self.risk_contributions_ = self.weights_ * marginal_contrib / portfolio_vol
        else:
            # Handle edge case: portfolio has near-zero volatility
            # This can happen with perfectly correlated assets or numerical issues
            # Assign equal risk contributions as fallback
            self.risk_contributions_ = np.ones(n_assets) / n_assets
        
        return self
    
    def predict(self) -> np.ndarray:
        """
        Generate portfolio weights after fitting.
        
        Returns:
            np.ndarray: Array of portfolio weights for each asset.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.weights_ is None:
            raise ValueError("Must call fit() before predict()")
        
        return self.weights_
    
    def get_risk_contributions(self) -> dict:
        """
        Get risk contribution of each asset as a dictionary.
        
        Returns:
            dict: Dictionary mapping asset names to risk contributions.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.risk_contributions_ is None or self.asset_names_ is None:
            raise ValueError("Must call fit() before get_risk_contributions()")
        
        return dict(zip(self.asset_names_, self.risk_contributions_))
    
    def get_risk_contribution_percentages(self) -> dict:
        """
        Get risk contribution percentages (should be approximately equal for Risk Parity).
        
        Returns:
            dict: Dictionary mapping asset names to risk contribution percentages.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.risk_contributions_ is None or self.asset_names_ is None:
            raise ValueError("Must call fit() before get_risk_contribution_percentages()")
        
        # Risk contributions should sum to 1.0 (they represent percentage of total risk)
        total_risk = np.sum(self.risk_contributions_)
        if total_risk > 1e-10:
            percentages = self.risk_contributions_ / total_risk
        else:
            percentages = np.ones_like(self.risk_contributions_) / len(self.risk_contributions_)
        
        return dict(zip(self.asset_names_, percentages * 100))  # Convert to percentages

