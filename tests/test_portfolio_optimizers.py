"""
Unit tests for portfolio optimizers.

This module contains comprehensive tests for:
- Efficient frontier plotting function
- Hierarchical Risk Parity (HRP)
- Risk Parity
- Nested Clustered Optimization (NCO)
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolios.utilities.portfolio_utilties import plot_efficient_frontier
from portfolios.portfolio_src import (
    HierarchicalRiskParity,
    RiskParity,
    NestedClusteredOptimization
)


class TestEfficientFrontier(unittest.TestCase):
    """Test efficient frontier plotting function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns_df = pd.DataFrame(
            np.random.randn(100, 3) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL']
        )
    
    def test_efficient_frontier_basic(self):
        """Test basic efficient frontier computation."""
        fig, ax = plot_efficient_frontier(self.returns_df, risk_free_rate=0.02)
        
        # Check that figure and axes are created
        self.assertIsNotNone(ax)
        # Check that axes has data plotted
        self.assertTrue(len(ax.lines) > 0 or len(ax.collections) > 0)
    
    def test_efficient_frontier_with_random(self):
        """Test efficient frontier with random portfolios."""
        fig, ax = plot_efficient_frontier(
            self.returns_df,
            risk_free_rate=0.02,
            show_random=True,
            num_random=100
        )
        self.assertIsNotNone(ax)
    
    def test_efficient_frontier_custom_ax(self):
        """Test efficient frontier with custom axes."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig_result, ax_result = plot_efficient_frontier(
            self.returns_df,
            ax=ax
        )
        self.assertIsNone(fig_result)  # Should return None when ax provided
        self.assertEqual(ax, ax_result)


class TestHierarchicalRiskParity(unittest.TestCase):
    """Test Hierarchical Risk Parity optimizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns_df = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
    
    def test_hrp_fit_predict(self):
        """Test HRP fit and predict methods."""
        hrp = HierarchicalRiskParity()
        hrp.fit(self.returns_df)
        weights = hrp.predict()
        
        # Check weights sum to 1
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        
        # Check weights are non-negative
        self.assertTrue(np.all(weights >= 0))
        
        # Check weights have correct length
        self.assertEqual(len(weights), len(self.returns_df.columns))
    
    def test_hrp_weights_dict(self):
        """Test getting weights as dictionary."""
        hrp = HierarchicalRiskParity()
        hrp.fit(self.returns_df)
        weights_dict = hrp.get_weights_dict()
        
        self.assertEqual(len(weights_dict), len(self.returns_df.columns))
        self.assertAlmostEqual(sum(weights_dict.values()), 1.0, places=5)
    
    def test_hrp_portfolio_performance(self):
        """Test portfolio performance calculation."""
        hrp = HierarchicalRiskParity()
        hrp.fit(self.returns_df)
        
        return_val, vol, sharpe = hrp.portfolio_performance()
        
        self.assertIsInstance(return_val, float)
        self.assertIsInstance(vol, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(vol, 0)  # Volatility should be positive
    
    def test_hrp_dendrogram(self):
        """Test dendrogram plotting."""
        hrp = HierarchicalRiskParity()
        hrp.fit(self.returns_df)
        
        fig, ax = hrp.plot_dendrogram()
        self.assertIsNotNone(ax)
    
    def test_hrp_two_assets(self):
        """Test HRP with minimum number of assets (2)."""
        returns_2 = self.returns_df.iloc[:, :2]
        hrp = HierarchicalRiskParity()
        hrp.fit(returns_2)
        weights = hrp.predict()
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertEqual(len(weights), 2)


class TestRiskParity(unittest.TestCase):
    """Test Risk Parity optimizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns_df = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
    
    def test_risk_parity_fit_predict(self):
        """Test Risk Parity fit and predict methods."""
        rp = RiskParity()
        rp.fit(self.returns_df)
        weights = rp.predict()
        
        # Check weights sum to 1
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        
        # Check weights are non-negative
        self.assertTrue(np.all(weights >= 0))
        
        # Check weights have correct length
        self.assertEqual(len(weights), len(self.returns_df.columns))
    
    def test_risk_parity_risk_contributions(self):
        """Test risk contribution calculation."""
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        risk_contribs = rp.get_risk_contributions()
        percentages = rp.get_risk_contribution_percentages()
        
        # Risk contributions should sum to approximately 1 (portfolio volatility)
        total_risk = sum(risk_contribs.values())
        self.assertGreater(total_risk, 0)
        
        # Percentages should sum to 100
        total_pct = sum(percentages.values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)
    
    def test_risk_parity_risk_contribution_identity(self):
        """
        Test the key mathematical identity: sum(RC_i) = σ_p.
        
        This verifies that risk contributions sum to portfolio volatility,
        which is a fundamental property of risk contributions.
        """
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        # Get portfolio volatility
        _, portfolio_vol, _ = rp.portfolio_performance()
        
        # Get risk contributions
        risk_contribs = rp.get_risk_contributions()
        total_risk_contrib = sum(risk_contribs.values())
        
        # Key identity: sum(RC_i) should equal σ_p
        # Allow small numerical tolerance (1e-6) for floating point precision
        self.assertAlmostEqual(total_risk_contrib, portfolio_vol, places=5,
                               msg="Risk contributions should sum to portfolio volatility")
    
    def test_risk_parity_equal_risk_contribution(self):
        """
        Test that risk contributions are approximately equal (ERC property).
        
        For Risk Parity, each asset should contribute approximately equal risk.
        We test this using absolute risk contributions (in volatility units),
        which should each equal σ_p/n.
        """
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        # Get portfolio volatility
        _, portfolio_vol, _ = rp.portfolio_performance()
        
        # Get absolute risk contributions (in volatility units)
        risk_contribs = rp.get_risk_contributions()
        rc_values = np.array(list(risk_contribs.values()))
        
        # For n assets, each should contribute approximately σ_p/n
        n_assets = len(rc_values)
        expected_rc = portfolio_vol / n_assets
        
        # Check that all risk contributions are close to expected value
        # Allow 5% deviation (reasonable for optimization tolerance)
        max_deviation = 0.05 * expected_rc  # 5% of expected value
        
        for asset, rc_abs in risk_contribs.items():
            deviation = abs(rc_abs - expected_rc)
            self.assertLess(deviation, max_deviation,
                           msg=f"Asset {asset} risk contribution {rc_abs:.6f} deviates "
                               f"too much from expected {expected_rc:.6f} (σ_p/n)")
    
    def test_risk_parity_diversification_constraints(self):
        """
        Test that diversification constraints are satisfied.
        
        Verifies:
        1. Minimum weight threshold (1% per asset)
        2. Maximum weight threshold (50% per asset)
        """
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        weights = rp.predict()
        
        # Check minimum weight constraint (1% per asset)
        min_weight = 0.01
        self.assertTrue(np.all(weights >= min_weight - 1e-6),  # Allow small numerical error
                       msg="All weights should be at least 1% (diversification constraint)")
        
        # Check maximum weight constraint (50% per asset)
        max_weight = 0.50
        self.assertTrue(np.all(weights <= max_weight + 1e-6),  # Allow small numerical error
                       msg="All weights should be at most 50% (concentration limit)")
    
    def test_risk_parity_inverse_volatility_relationship(self):
        """
        Test that Risk Parity weights are approximately inversely proportional to volatility.
        
        For Risk Parity, we expect: w_i ≈ constant / σ_i
        This means higher volatility assets get lower weights.
        """
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        weights = rp.predict()
        
        # Compute asset volatilities
        cov_matrix = rp.cov_matrix_.values
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Check inverse relationship: w_i * σ_i should be approximately constant
        # (for true Risk Parity, w_i * σ_i ≈ constant for all i)
        weighted_vols = weights * volatilities
        
        # Coefficient of variation should be small (indicating near-constant)
        cv = np.std(weighted_vols) / np.mean(weighted_vols)
        
        # Allow up to 20% coefficient of variation (accounts for correlations)
        self.assertLess(cv, 0.20,
                        msg="Risk Parity weights should be approximately inversely "
                            "proportional to volatility (w_i * σ_i ≈ constant)")
    
    def test_risk_parity_portfolio_performance(self):
        """Test portfolio performance calculation."""
        rp = RiskParity()
        rp.fit(self.returns_df)
        
        return_val, vol, sharpe = rp.portfolio_performance()
        
        self.assertIsInstance(return_val, float)
        self.assertIsInstance(vol, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(vol, 0)
    
    def test_risk_parity_two_assets(self):
        """Test Risk Parity with minimum number of assets (2)."""
        returns_2 = self.returns_df.iloc[:, :2]
        rp = RiskParity()
        rp.fit(returns_2)
        weights = rp.predict()
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertEqual(len(weights), 2)


class TestNestedClusteredOptimization(unittest.TestCase):
    """Test Nested Clustered Optimization optimizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns_df = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
    
    def test_nco_fit_predict(self):
        """Test NCO fit and predict methods."""
        nco = NestedClusteredOptimization()
        nco.fit(self.returns_df)
        weights = nco.predict()
        
        # Check weights sum to 1
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        
        # Check weights are non-negative
        self.assertTrue(np.all(weights >= 0))
        
        # Check weights have correct length
        self.assertEqual(len(weights), len(self.returns_df.columns))
    
    def test_nco_cluster_assignments(self):
        """Test cluster assignment functionality."""
        nco = NestedClusteredOptimization(n_clusters=3)
        nco.fit(self.returns_df)
        
        assignments = nco.get_cluster_assignments()
        
        # All assets should be assigned to a cluster
        self.assertEqual(len(assignments), len(self.returns_df.columns))
        
        # Cluster IDs should be between 1 and n_clusters
        cluster_ids = set(assignments.values())
        self.assertTrue(all(1 <= cid <= 3 for cid in cluster_ids))
    
    def test_nco_within_cluster_methods(self):
        """Test NCO with different within-cluster methods."""
        # Test with risk parity
        nco_rp = NestedClusteredOptimization(
            n_clusters=3,
            within_cluster_method='risk_parity'
        )
        nco_rp.fit(self.returns_df)
        weights_rp = nco_rp.predict()
        self.assertAlmostEqual(weights_rp.sum(), 1.0, places=5)
        
        # Test with mean variance
        nco_mv = NestedClusteredOptimization(
            n_clusters=3,
            within_cluster_method='mean_variance'
        )
        nco_mv.fit(self.returns_df)
        weights_mv = nco_mv.predict()
        self.assertAlmostEqual(weights_mv.sum(), 1.0, places=5)
    
    def test_nco_portfolio_performance(self):
        """Test portfolio performance calculation."""
        nco = NestedClusteredOptimization()
        nco.fit(self.returns_df)
        
        return_val, vol, sharpe = nco.portfolio_performance()
        
        self.assertIsInstance(return_val, float)
        self.assertIsInstance(vol, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(vol, 0)
    
    def test_nco_two_assets(self):
        """Test NCO with minimum number of assets (2)."""
        returns_2 = self.returns_df.iloc[:, :2]
        nco = NestedClusteredOptimization(n_clusters=2)
        nco.fit(returns_2)
        weights = nco.predict()
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertEqual(len(weights), 2)


class TestOptimizerComparison(unittest.TestCase):
    """Test comparing different optimizers on the same data."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns_df = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
    
    def test_all_optimizers_same_data(self):
        """Test that all optimizers work on the same data."""
        hrp = HierarchicalRiskParity()
        rp = RiskParity()
        nco = NestedClusteredOptimization()
        
        hrp.fit(self.returns_df)
        rp.fit(self.returns_df)
        nco.fit(self.returns_df)
        
        weights_hrp = hrp.predict()
        weights_rp = rp.predict()
        weights_nco = nco.predict()
        
        # All should produce valid weights
        for weights in [weights_hrp, weights_rp, weights_nco]:
            self.assertAlmostEqual(weights.sum(), 1.0, places=5)
            self.assertTrue(np.all(weights >= 0))
            self.assertEqual(len(weights), len(self.returns_df.columns))
    
    def test_performance_comparison(self):
        """Test comparing performance metrics across optimizers."""
        hrp = HierarchicalRiskParity()
        rp = RiskParity()
        nco = NestedClusteredOptimization()
        
        hrp.fit(self.returns_df)
        rp.fit(self.returns_df)
        nco.fit(self.returns_df)
        
        # All should produce valid performance metrics
        for optimizer in [hrp, rp, nco]:
            ret, vol, sharpe = optimizer.portfolio_performance()
            self.assertIsInstance(ret, float)
            self.assertIsInstance(vol, float)
            self.assertIsInstance(sharpe, float)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        empty_df = pd.DataFrame()
        
        hrp = HierarchicalRiskParity()
        with self.assertRaises(ValueError):
            hrp.fit(empty_df)
    
    def test_single_asset(self):
        """Test error handling for single asset."""
        single_df = pd.DataFrame({'AAPL': np.random.randn(100) * 0.02})
        
        hrp = HierarchicalRiskParity()
        with self.assertRaises(ValueError):
            hrp.fit(single_df)
    
    def test_predict_before_fit(self):
        """Test error when predict is called before fit."""
        hrp = HierarchicalRiskParity()
        with self.assertRaises(ValueError):
            hrp.predict()
    
    def test_performance_before_fit(self):
        """Test error when portfolio_performance is called before fit."""
        hrp = HierarchicalRiskParity()
        with self.assertRaises(ValueError):
            hrp.portfolio_performance()


if __name__ == '__main__':
    unittest.main()



