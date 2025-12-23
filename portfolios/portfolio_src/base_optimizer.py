"""
Base class for portfolio optimizers.

This module provides an abstract base class that defines the common interface
for all portfolio optimization methods, including HRP, Risk Parity, and NCO.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Portfolio utility function implementations matching portfolio_utilties.py
# These are inlined here to avoid dependency on statsmodels (which is imported
# in portfolio_utilties.py but not used in these functions).
# This ensures production-quality consistency with the utility functions
# while avoiding unnecessary dependencies.
def _portfolio_expected_return(returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate expected return - matches portfolio_utilties.py implementation."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    asset_expected_returns = returns_df.mean()
    return weights.dot(asset_expected_returns)

def _portfolio_volatility(returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate portfolio volatility - matches portfolio_utilties.py implementation."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    cov_matrix = returns_df.cov()
    portfolio_variance = weights.dot(cov_matrix).dot(weights)
    return np.sqrt(portfolio_variance)

def _portfolio_sharpe_ratio(returns_df: pd.DataFrame, weights: np.ndarray,
                           risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio - matches portfolio_utilties.py implementation."""
    portfolio_return = _portfolio_expected_return(returns_df, weights)
    portfolio_std = _portfolio_volatility(returns_df, weights)
    annualized_return = portfolio_return * periods_per_year
    annualized_std = portfolio_std * np.sqrt(periods_per_year)
    return (annualized_return - risk_free_rate) / annualized_std


class BasePortfolioOptimizer(ABC):
    """
    Abstract base class for portfolio optimizers.
    
    This class defines the common interface that all portfolio optimization
    methods must implement. It provides shared utilities for covariance/correlation
    computation and weight validation.
    
    All portfolio optimizers should inherit from this class and implement:
    - fit(): Train/fit the optimizer on return data
    - predict(): Generate portfolio weights
    
    Attributes:
        returns_df (pd.DataFrame): Historical returns data used for fitting.
        weights_ (np.ndarray): Optimized portfolio weights (set after fit).
        cov_matrix_ (pd.DataFrame): Covariance matrix computed from returns.
        corr_matrix_ (pd.DataFrame): Correlation matrix computed from returns.
        asset_names_ (list): List of asset names from returns_df columns.
    """
    
    def __init__(self):
        """Initialize the base optimizer."""
        self.returns_df: Optional[pd.DataFrame] = None
        self.weights_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[pd.DataFrame] = None
        self.corr_matrix_: Optional[pd.DataFrame] = None
        self.asset_names_: Optional[list] = None
    
    @abstractmethod
    def fit(self, returns_df: pd.DataFrame) -> 'BasePortfolioOptimizer':
        """
        Fit the optimizer on historical returns data.
        
        This method should compute the necessary statistics (covariance,
        correlation, etc.) and perform the optimization to determine portfolio weights.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
                Each value should be a return (e.g., 0.01 for 1%).
        
        Returns:
            BasePortfolioOptimizer: Returns self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Generate portfolio weights after fitting.
        
        Returns:
            np.ndarray: Array of portfolio weights for each asset.
                Weights should sum to approximately 1.0.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        pass
    
    def _validate_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean returns DataFrame.
        
        Args:
            returns_df (pd.DataFrame): Returns DataFrame to validate.
        
        Returns:
            pd.DataFrame: Cleaned returns DataFrame.
        
        Raises:
            ValueError: If returns_df is invalid.
        """
        if not isinstance(returns_df, pd.DataFrame):
            raise ValueError("returns_df must be a pandas DataFrame")
        
        if returns_df.empty:
            raise ValueError("returns_df cannot be empty")
        
        if returns_df.shape[1] < 2:
            raise ValueError("At least 2 assets are required for portfolio optimization")
        
        # Remove any rows with all NaN values
        returns_df = returns_df.dropna(how='all')
        
        if returns_df.empty:
            raise ValueError("No valid data remaining after removing NaN rows")
        
        return returns_df
    
    def _compute_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute covariance matrix from returns.
        
        Args:
            returns_df (pd.DataFrame): Returns DataFrame.
        
        Returns:
            pd.DataFrame: Covariance matrix.
        """
        return returns_df.cov()
    
    def _compute_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix from returns.
        
        Args:
            returns_df (pd.DataFrame): Returns DataFrame.
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        return returns_df.corr()
    
    def _validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Validate and normalize portfolio weights.
        
        This method handles edge cases including:
        - Numerical precision issues (very small negative values are clipped to zero)
        - Zero or near-zero weight sums (fallback to equal weights)
        - Weight normalization to ensure sum equals 1.0
        
        Args:
            weights (np.ndarray): Portfolio weights to validate.
            tolerance (float, optional): Tolerance for weight sum check. Defaults to 1e-6.
                Values between -tolerance and 0 are treated as numerical errors and clipped to 0.
        
        Returns:
            np.ndarray: Normalized weights that sum to 1.0.
        
        Raises:
            ValueError: If weights are invalid (e.g., significantly negative, wrong shape).
        """
        weights = np.array(weights)
        
        if len(weights.shape) != 1:
            raise ValueError("Weights must be a 1D array")
        
        # Check for significantly negative weights (beyond numerical precision)
        # Values < -tolerance indicate actual negative weights, not numerical errors
        if np.any(weights < -tolerance):
            raise ValueError("Weights cannot be negative (short selling not allowed by default)")
        
        # Handle numerical precision: clip very small negative values to zero
        # (values between -tolerance and 0 are treated as numerical errors from optimization)
        weights = np.maximum(weights, 0.0)
        
        # Handle case where all weights are zero or very small
        if np.sum(weights) < tolerance:
            # Fallback to equal weights if all weights are essentially zero
            weights = np.ones_like(weights) / len(weights)
        elif np.abs(weights.sum() - 1.0) > tolerance:
            # Normalize weights to sum to 1.0
            weights = weights / weights.sum()
        
        return weights
    
    def portfolio_performance(self, periods_per_year: int = 252, 
                             risk_free_rate: float = 0.02) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics after optimization.
        
        This method uses the utility functions from portfolio_utilties.py for
        consistency and production-quality calculations. The utility functions
        handle weight normalization and provide robust implementations.
        
        Args:
            periods_per_year (int, optional): Number of periods per year for annualization.
                Defaults to 252 (trading days).
            risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
        
        Returns:
            tuple: (expected_return, volatility, sharpe_ratio)
                - expected_return: Annualized expected return
                - volatility: Annualized volatility (standard deviation)
                - sharpe_ratio: Annualized Sharpe ratio
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.weights_ is None or self.returns_df is None:
            raise ValueError("Must call fit() before portfolio_performance()")
        
        # Use utility functions for consistency and production quality
        # These functions handle weight normalization and provide robust calculations
        # Note: portfolio_sharpe_ratio already handles annualization internally
        # Call functions directly (they're module-level functions, not methods)
        expected_return = _portfolio_expected_return(self.returns_df, self.weights_)
        volatility = _portfolio_volatility(self.returns_df, self.weights_)
        sharpe_ratio = _portfolio_sharpe_ratio(
            self.returns_df, 
            self.weights_, 
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )
        
        # Annualize expected return and volatility for consistency with return format
        # (utility functions return per-period values, we annualize for the API)
        annualized_return = expected_return * periods_per_year
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        return (annualized_return, annualized_volatility, sharpe_ratio)
    
    def get_weights_dict(self) -> dict:
        """
        Get portfolio weights as a dictionary with asset names.
        
        Returns:
            dict: Dictionary mapping asset names to weights.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.weights_ is None or self.asset_names_ is None:
            raise ValueError("Must call fit() before get_weights_dict()")
        
        return dict(zip(self.asset_names_, self.weights_))
    
    def __repr__(self) -> str:
        """String representation of the optimizer."""
        class_name = self.__class__.__name__
        if self.weights_ is None:
            return f"{class_name}(not fitted)"
        else:
            return f"{class_name}(fitted, n_assets={len(self.weights_)})"

