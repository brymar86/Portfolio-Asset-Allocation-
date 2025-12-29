"""
Unit tests for BasePortfolioOptimizer.

This module tests the base class functionality that is shared across all
portfolio optimizers, including covariance/correlation computation and
weight validation.

Mathematical Foundations:
- Covariance matrix: Σ_ij = E[(R_i - μ_i)(R_j - μ_j)] where R_i is return of asset i
- Correlation matrix: ρ_ij = Σ_ij / (σ_i * σ_j) where σ_i is std dev of asset i
- Portfolio variance: σ_p² = w^T Σ w where w is weight vector, Σ is covariance matrix
- Portfolio return: μ_p = w^T μ where μ is expected return vector
"""

import pytest
import numpy as np
import pandas as pd
from ..hierarchical_risk_parity import HierarchicalRiskParity


class TestBaseOptimizer:
    """
    Test suite for BasePortfolioOptimizer.
    
    These tests verify the mathematical correctness of shared utilities
    used by all portfolio optimizers.
    """
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL']
        )
        return returns
    
    def test_covariance_computation(self, sample_returns):
        """
        Test covariance matrix computation.
        
        Mathematical Verification:
        The covariance matrix Σ should satisfy:
        1. Symmetry: Σ_ij = Σ_ji (covariance is symmetric)
        2. Positive semi-definiteness: w^T Σ w ≥ 0 for all w (variance is non-negative)
        3. Diagonal elements are variances: Σ_ii = Var(R_i) = E[(R_i - μ_i)²]
        4. Relationship to correlation: ρ_ij = Σ_ij / (σ_i * σ_j)
        
        This test verifies that the computed covariance matrix has these properties.
        """
        optimizer = HierarchicalRiskParity()
        cov_matrix = optimizer._compute_covariance(sample_returns)
        
        # Check symmetry: Σ^T = Σ
        assert np.allclose(cov_matrix.values, cov_matrix.values.T), \
            "Covariance matrix must be symmetric"
        
        # Check positive semi-definiteness: all eigenvalues ≥ 0
        eigenvalues = np.linalg.eigvals(cov_matrix.values)
        assert np.all(eigenvalues >= -1e-10), \
            "Covariance matrix must be positive semi-definite (all eigenvalues ≥ 0)"
        
        # Check diagonal elements are variances
        for i, asset in enumerate(sample_returns.columns):
            expected_variance = sample_returns[asset].var()
            actual_variance = cov_matrix.iloc[i, i]
            assert np.isclose(expected_variance, actual_variance), \
                f"Diagonal element {i} should equal variance of {asset}"
    
    def test_correlation_computation(self, sample_returns):
        """
        Test correlation matrix computation.
        
        Mathematical Verification:
        The correlation matrix ρ should satisfy:
        1. Symmetry: ρ_ij = ρ_ji
        2. Unit diagonal: ρ_ii = 1 (asset perfectly correlated with itself)
        3. Bounded off-diagonal: -1 ≤ ρ_ij ≤ 1 (correlation coefficient bounds)
        4. Relationship to covariance: ρ_ij = Σ_ij / (σ_i * σ_j)
        
        This test verifies these mathematical properties.
        """
        optimizer = HierarchicalRiskParity()
        corr_matrix = optimizer._compute_correlation(sample_returns)
        
        # Check symmetry
        assert np.allclose(corr_matrix.values, corr_matrix.values.T), \
            "Correlation matrix must be symmetric"
        
        # Check unit diagonal
        diagonal = np.diag(corr_matrix.values)
        assert np.allclose(diagonal, 1.0), \
            "Correlation matrix diagonal must be 1.0 (perfect self-correlation)"
        
        # Check bounds: -1 ≤ ρ_ij ≤ 1
        assert np.all(corr_matrix.values >= -1.0) and np.all(corr_matrix.values <= 1.0), \
            "All correlation coefficients must be in [-1, 1]"
        
        # Verify relationship to covariance
        cov_matrix = optimizer._compute_covariance(sample_returns)
        for i in range(len(sample_returns.columns)):
            for j in range(len(sample_returns.columns)):
                std_i = np.sqrt(cov_matrix.iloc[i, i])
                std_j = np.sqrt(cov_matrix.iloc[j, j])
                expected_corr = cov_matrix.iloc[i, j] / (std_i * std_j)
                actual_corr = corr_matrix.iloc[i, j]
                assert np.isclose(expected_corr, actual_corr), \
                    f"Correlation should equal cov/(σ_i * σ_j) for assets {i}, {j}"
    
    def test_weight_validation(self, sample_returns):
        """
        Test weight validation and normalization.
        
        Mathematical Verification:
        Portfolio weights must satisfy:
        1. Sum constraint: Σ_i w_i = 1 (fully invested)
        2. Non-negativity: w_i ≥ 0 (no short selling in long-only portfolios)
        3. Normalization: If weights don't sum to 1, they should be normalized
        
        This test verifies that the validation function enforces these constraints.
        """
        optimizer = HierarchicalRiskParity()
        
        # Test valid weights (sum to 1, all non-negative)
        valid_weights = np.array([0.4, 0.3, 0.3])
        normalized = optimizer._validate_weights(valid_weights)
        assert np.isclose(normalized.sum(), 1.0), \
            "Valid weights should sum to 1.0"
        assert np.all(normalized >= 0), \
            "All weights must be non-negative"
        
        # Test normalization (weights don't sum to 1)
        unnormalized = np.array([0.5, 0.3, 0.1])  # Sums to 0.9
        normalized = optimizer._validate_weights(unnormalized)
        assert np.isclose(normalized.sum(), 1.0), \
            "Weights should be normalized to sum to 1.0"
        assert np.allclose(normalized, unnormalized / unnormalized.sum()), \
            "Normalization should divide by sum"
        
        # Test negative weights (should raise error)
        negative_weights = np.array([0.5, -0.2, 0.7])
        with pytest.raises(ValueError, match="cannot be negative"):
            optimizer._validate_weights(negative_weights)
        
        # Test 2D array (should raise error)
        two_d_weights = np.array([[0.5, 0.3, 0.2]])
        with pytest.raises(ValueError, match="must be a 1D array"):
            optimizer._validate_weights(two_d_weights)
    
    def test_portfolio_performance_calculation(self, sample_returns):
        """
        Test portfolio performance metrics calculation.
        
        Mathematical Verification:
        Portfolio performance metrics are calculated as:
        1. Expected return: μ_p = w^T μ (weighted average of asset returns)
        2. Portfolio variance: σ_p² = w^T Σ w (quadratic form)
        3. Portfolio volatility: σ_p = sqrt(w^T Σ w)
        4. Sharpe ratio: SR = (μ_p - r_f) / σ_p (risk-adjusted return)
        
        Annualization:
        - Return: μ_p_annual = μ_p * periods_per_year
        - Volatility: σ_p_annual = σ_p * sqrt(periods_per_year)
        
        This test verifies these calculations are mathematically correct.
        """
        optimizer = HierarchicalRiskParity()
        optimizer.fit(sample_returns)
        weights = optimizer.predict()
        
        # Calculate expected metrics manually
        expected_returns = sample_returns.mean().values
        cov_matrix = optimizer.cov_matrix_.values
        
        # Manual calculation
        manual_return = weights.dot(expected_returns)
        manual_variance = weights.T @ cov_matrix @ weights
        manual_vol = np.sqrt(manual_variance)
        
        # Get optimizer results
        annual_return, annual_vol, sharpe = optimizer.portfolio_performance(
            periods_per_year=252,
            risk_free_rate=0.02
        )
        
        # Verify annualization
        assert np.isclose(annual_return, manual_return * 252), \
            "Annualized return should equal daily return * 252"
        assert np.isclose(annual_vol, manual_vol * np.sqrt(252)), \
            "Annualized volatility should equal daily volatility * sqrt(252)"
        
        # Verify Sharpe ratio calculation
        annual_manual_return = manual_return * 252
        annual_manual_vol = manual_vol * np.sqrt(252)
        manual_sharpe = (annual_manual_return - 0.02) / annual_manual_vol
        assert np.isclose(sharpe, manual_sharpe), \
            "Sharpe ratio should equal (return - risk_free) / volatility"
    
    def test_returns_validation(self):
        """
        Test returns DataFrame validation.
        
        Mathematical Verification:
        Returns data must satisfy:
        1. Non-empty: Must have data to compute statistics
        2. Minimum assets: Need at least 2 assets for portfolio optimization
        3. Valid structure: Must be pandas DataFrame
        
        This test verifies error handling for invalid inputs.
        """
        optimizer = HierarchicalRiskParity()
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            optimizer._validate_returns(empty_df)
        
        # Test single asset
        single_asset = pd.DataFrame({'AAPL': np.random.randn(100) * 0.02})
        with pytest.raises(ValueError, match="At least 2 assets"):
            optimizer._validate_returns(single_asset)
        
        # Test non-DataFrame input
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            optimizer._validate_returns(np.array([[1, 2], [3, 4]]))
        
        # Test valid DataFrame
        valid_df = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.02,
            'MSFT': np.random.randn(100) * 0.02
        })
        validated = optimizer._validate_returns(valid_df)
        assert isinstance(validated, pd.DataFrame), \
            "Valid DataFrame should pass validation"

