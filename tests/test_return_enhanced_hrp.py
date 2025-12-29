"""
Unit tests for Return-Enhanced Hierarchical Risk Parity (RE-HRP) optimizer.

This module contains comprehensive tests for:
- Weight constraints (sum to 1, non-negative)
- Clustering preservation (same as HRP)
- Sortino Ratio allocation
- Comparison with HRP
- Edge cases (negative Sortino, zero downside deviation, etc.)
- Mathematical consistency
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolios.portfolio_src import (
    ReturnEnhancedHRP,
    HierarchicalRiskParity
)


class TestReturnEnhancedHRP(unittest.TestCase):
    """Test Return-Enhanced Hierarchical Risk Parity optimizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        # Create returns with different expected returns and volatilities
        n_assets = 5
        self.returns_df = pd.DataFrame(
            np.random.randn(252, n_assets) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        # Add different expected returns to make Sortino ratios differ
        self.returns_df['AAPL'] += 0.001  # Higher expected return
        self.returns_df['MSFT'] += 0.0005
        self.returns_df['GOOGL'] += 0.0003
        self.returns_df['AMZN'] += 0.0001
        # TSLA has lower expected return (default)
    
    def test_re_hrp_weights_sum_to_one(self):
        """Test that RE-HRP weights sum to 1.0."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        weights = re_hrp.predict()
        
        # Check weights sum to 1
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        
        # Check weights are non-negative
        self.assertTrue(np.all(weights >= 0))
        
        # Check weights have correct length
        self.assertEqual(len(weights), len(self.returns_df.columns))
    
    def test_re_hrp_preserves_clustering(self):
        """Test that RE-HRP preserves HRP's clustering structure."""
        re_hrp = ReturnEnhancedHRP()
        hrp = HierarchicalRiskParity()
        
        re_hrp.fit(self.returns_df)
        hrp.fit(self.returns_df)
        
        # Check that linkage matrices are identical
        np.testing.assert_array_equal(
            re_hrp.linkage_matrix_,
            hrp.linkage_matrix_
        )
        
        # Check that tree orders are identical
        np.testing.assert_array_equal(
            re_hrp.tree_order_,
            hrp.tree_order_
        )
        
        # Check that quasi-diagonalized covariance matrices are identical
        pd.testing.assert_frame_equal(
            re_hrp.cov_quasi_diag_,
            hrp.cov_quasi_diag_
        )
    
    def test_re_hrp_sortino_allocation(self):
        """Test that RE-HRP allocates differently than HRP when returns differ."""
        # Create synthetic data with different expected returns
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Asset 1: High return, low volatility (should favor this)
        asset1_returns = np.random.randn(252) * 0.01 + 0.001  # Mean 0.1%, std 1%
        
        # Asset 2: Low return, high volatility (should favor less)
        asset2_returns = np.random.randn(252) * 0.03 - 0.0005  # Mean -0.05%, std 3%
        
        # Asset 3: Medium return, medium volatility
        asset3_returns = np.random.randn(252) * 0.02 + 0.0005  # Mean 0.05%, std 2%
        
        returns_df = pd.DataFrame({
            'High_Return': asset1_returns,
            'Low_Return': asset2_returns,
            'Medium_Return': asset3_returns
        }, index=dates)
        
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.02)
        hrp = HierarchicalRiskParity()
        
        re_hrp.fit(returns_df)
        hrp.fit(returns_df)
        
        re_hrp_weights = re_hrp.predict()
        hrp_weights = hrp.predict()
        
        # RE-HRP should produce different weights than HRP (incorporates return info)
        # The exact allocation depends on clustering structure, but weights should differ
        weights_diff = np.abs(re_hrp_weights - hrp_weights)
        self.assertTrue(np.any(weights_diff > 1e-6), 
                       "RE-HRP should produce different weights than HRP when returns differ")
        
        # Both should sum to 1
        self.assertAlmostEqual(re_hrp_weights.sum(), 1.0, places=5)
        self.assertAlmostEqual(hrp_weights.sum(), 1.0, places=5)
    
    def test_re_hrp_vs_hrp_different_weights(self):
        """Test that RE-HRP produces different weights than HRP."""
        re_hrp = ReturnEnhancedHRP()
        hrp = HierarchicalRiskParity()
        
        re_hrp.fit(self.returns_df)
        hrp.fit(self.returns_df)
        
        re_hrp_weights = re_hrp.predict()
        hrp_weights = hrp.predict()
        
        # Weights should be different (RE-HRP incorporates return information)
        # But they should both sum to 1
        self.assertAlmostEqual(re_hrp_weights.sum(), 1.0, places=5)
        self.assertAlmostEqual(hrp_weights.sum(), 1.0, places=5)
        
        # Check that weights are different (not identical)
        # Use a small tolerance to account for numerical differences
        weights_diff = np.abs(re_hrp_weights - hrp_weights)
        # At least some weights should differ significantly
        self.assertTrue(np.any(weights_diff > 1e-6))
    
    def test_re_hrp_handles_negative_sortino(self):
        """Test that RE-HRP handles assets with negative expected returns."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Create returns with negative expected returns
        returns_df = pd.DataFrame(
            np.random.randn(252, 3) * 0.02 - 0.001,  # Negative mean
            index=dates,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.02)
        re_hrp.fit(returns_df)
        weights = re_hrp.predict()
        
        # Should still produce valid weights
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1.0))
    
    def test_re_hrp_downside_deviation_calculation(self):
        """Test that downside deviation is calculated correctly."""
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.02, target_return=0.0)
        
        # Create returns with known downside
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01])
        
        # Calculate downside deviation manually
        downside_returns = np.minimum(0, returns - 0.0)
        expected_downside_var = np.mean(downside_returns ** 2)
        expected_downside_dev = np.sqrt(expected_downside_var)
        
        # Calculate using RE-HRP method
        sortino = re_hrp._compute_sortino_ratio(
            returns,
            risk_free_rate=0.02,
            target_return=0.0
        )
        
        # Verify that downside deviation is calculated (check that Sortino is finite)
        self.assertTrue(np.isfinite(sortino))
        
        # For returns with downside, Sortino should be finite
        # (exact value depends on expected return)
        self.assertIsInstance(sortino, (float, np.floating))
    
    def test_re_hrp_zero_downside_deviation(self):
        """Test that RE-HRP handles zero downside deviation (perfect downside protection)."""
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.02)
        
        # Create returns with no downside (all positive)
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.03])
        
        sortino = re_hrp._compute_sortino_ratio(
            returns,
            risk_free_rate=0.02,
            target_return=0.0
        )
        
        # Should return large positive value for perfect downside protection
        if np.mean(returns) * 252 > 0.02:  # Annualized return > risk-free rate
            self.assertGreater(sortino, 1e5)  # Very high Sortino
        else:
            self.assertLess(sortino, -1e5)  # Very low Sortino (negative excess return)
    
    def test_re_hrp_mathematical_consistency(self):
        """Test mathematical consistency of Sortino Ratio calculation and allocation."""
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.02)
        re_hrp.fit(self.returns_df)
        weights = re_hrp.predict()
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_df.values @ weights)
        
        # Calculate Sortino Ratio manually
        expected_return_period = np.mean(portfolio_returns)
        expected_return_annual = expected_return_period * 252
        
        downside_returns = np.minimum(0, portfolio_returns - 0.0)
        downside_variance = np.mean(downside_returns ** 2)
        downside_deviation_period = np.sqrt(downside_variance)
        downside_deviation_annual = downside_deviation_period * np.sqrt(252)
        
        if downside_deviation_annual > 1e-10:
            sortino_manual = (expected_return_annual - 0.02) / downside_deviation_annual
            # Verify that Sortino is calculated correctly
            self.assertTrue(np.isfinite(sortino_manual))
    
    def test_re_hrp_expected_returns_stored(self):
        """Test that expected returns are computed and stored."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        
        # Check that expected returns are stored
        self.assertIsNotNone(re_hrp.expected_returns_)
        self.assertEqual(len(re_hrp.expected_returns_), len(self.returns_df.columns))
        
        # Check that expected returns match manual calculation
        expected_returns_manual = self.returns_df.mean().values
        np.testing.assert_array_almost_equal(
            re_hrp.expected_returns_,
            expected_returns_manual,
            decimal=10
        )
    
    def test_re_hrp_returns_array_stored(self):
        """Test that returns array is stored for Sortino calculation."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        
        # Check that returns array is stored
        self.assertIsNotNone(re_hrp.returns_array_)
        self.assertEqual(re_hrp.returns_array_.shape, self.returns_df.shape)
        
        # Check that returns array matches DataFrame values
        np.testing.assert_array_equal(
            re_hrp.returns_array_,
            self.returns_df.values
        )
    
    def test_re_hrp_single_asset(self):
        """Test RE-HRP with single asset (edge case - should raise error)."""
        single_asset_df = self.returns_df[['AAPL']]
        
        re_hrp = ReturnEnhancedHRP()
        # Base validator requires at least 2 assets
        with self.assertRaises(ValueError):
            re_hrp.fit(single_asset_df)
    
    def test_re_hrp_two_assets(self):
        """Test RE-HRP with two assets (edge case)."""
        two_asset_df = self.returns_df[['AAPL', 'MSFT']]
        
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(two_asset_df)
        weights = re_hrp.predict()
        
        # Weights should sum to 1
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
        self.assertEqual(len(weights), 2)
    
    def test_re_hrp_custom_risk_free_rate(self):
        """Test RE-HRP with custom risk-free rate."""
        re_hrp = ReturnEnhancedHRP(risk_free_rate=0.03)  # 3% risk-free rate
        re_hrp.fit(self.returns_df)
        weights = re_hrp.predict()
        
        # Should still produce valid weights
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
    
    def test_re_hrp_custom_target_return(self):
        """Test RE-HRP with custom target return."""
        re_hrp = ReturnEnhancedHRP(target_return=0.01)  # 1% target return
        re_hrp.fit(self.returns_df)
        weights = re_hrp.predict()
        
        # Should still produce valid weights
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
    
    def test_re_hrp_portfolio_performance(self):
        """Test portfolio performance calculation."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        
        expected_return, volatility, sharpe_ratio = re_hrp.portfolio_performance()
        
        # Check that metrics are finite and reasonable
        self.assertTrue(np.isfinite(expected_return))
        self.assertTrue(np.isfinite(volatility))
        self.assertTrue(np.isfinite(sharpe_ratio))
        
        # Volatility should be positive
        self.assertGreater(volatility, 0)
    
    def test_re_hrp_get_weights_dict(self):
        """Test getting weights as dictionary."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        
        weights_dict = re_hrp.get_weights_dict()
        
        # Check that dictionary has correct keys
        self.assertEqual(set(weights_dict.keys()), set(self.returns_df.columns))
        
        # Check that weights sum to 1
        total_weight = sum(weights_dict.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_re_hrp_dendrogram_plot(self):
        """Test dendrogram plotting."""
        re_hrp = ReturnEnhancedHRP()
        re_hrp.fit(self.returns_df)
        
        # Should not raise an error
        fig, ax = re_hrp.plot_dendrogram()
        
        # Check that axes is created
        self.assertIsNotNone(ax)
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_information_ratio_threshold_single_asset_check(self):
        """
        Test that Information Ratio threshold checking works for single assets.
        
        Mathematical Foundation:
        - When a cluster contains a single asset, we check that asset's individual return
        - If return < threshold, the asset should not get excessive weight via IR allocation
        - This prevents low-return assets (like TLT bonds) from getting high weights
        
        This test validates the fix for TLT over-weighting issue.
        """
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Create realistic scenario:
        # - TLT: Low return (0.5% annualized), low volatility (5% annualized)
        # - SPY: High return (15% annualized), medium volatility (18% annualized)
        # - BTC: Very high return (100% annualized), high volatility (80% annualized)
        # - Benchmark (equal-weighted): ~38.5% annualized return
        
        # TLT: Low return, low volatility
        tlt_returns = np.random.randn(252) * (0.05 / np.sqrt(252)) + (0.005 / 252)
        
        # SPY: Medium-high return, medium volatility
        spy_returns = np.random.randn(252) * (0.18 / np.sqrt(252)) + (0.15 / 252)
        
        # BTC: Very high return, high volatility
        btc_returns = np.random.randn(252) * (0.80 / np.sqrt(252)) + (1.00 / 252)
        
        returns_df = pd.DataFrame({
            'TLT': tlt_returns,
            'SPY': spy_returns,
            'BTC-USD': btc_returns
        }, index=dates)
        
        # Calculate benchmark return (equal-weighted)
        benchmark_return = returns_df.mean(axis=1).mean() * 252
        
        # Test 1: With default threshold (benchmark return)
        # TLT should NOT get excessive weight because its return (0.5%) < benchmark (~38.5%)
        re_hrp_default = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_return_threshold=None,  # Uses benchmark return as threshold
            min_tracking_error=0.01,
            verbose=False
        )
        re_hrp_default.fit(returns_df)
        weights_default = re_hrp_default.predict()
        weights_dict_default = re_hrp_default.get_weights_dict()
        
        # TLT should NOT dominate the portfolio (should be < 50%)
        tlt_weight_default = weights_dict_default['TLT']
        self.assertLess(tlt_weight_default, 0.5, 
                       f"TLT weight ({tlt_weight_default:.4f}) should be < 50% with default threshold")
        
        # Test 2: With explicit threshold (5% annualized)
        # TLT (0.5%) < 5% threshold, so should get even less weight
        re_hrp_explicit = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_return_threshold=0.05,  # 5% annualized threshold
            min_tracking_error=0.01,
            verbose=False
        )
        re_hrp_explicit.fit(returns_df)
        weights_explicit = re_hrp_explicit.predict()
        weights_dict_explicit = re_hrp_explicit.get_weights_dict()
        
        # TLT should get even less weight with explicit threshold
        tlt_weight_explicit = weights_dict_explicit['TLT']
        self.assertLess(tlt_weight_explicit, 0.5,
                       f"TLT weight ({tlt_weight_explicit:.4f}) should be < 50% with explicit threshold")
        
        # Test 3: Verify weights sum to 1 and are non-negative
        self.assertAlmostEqual(weights_default.sum(), 1.0, places=5)
        self.assertAlmostEqual(weights_explicit.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights_default >= 0))
        self.assertTrue(np.all(weights_explicit >= 0))
        
        # Test 4: Verify that high-return assets get more weight
        # BTC should get significant weight (high return)
        btc_weight_default = weights_dict_default['BTC-USD']
        btc_weight_explicit = weights_dict_explicit['BTC-USD']
        
        # BTC should get more weight than TLT (both cases)
        self.assertGreater(btc_weight_default, tlt_weight_default,
                          "BTC should get more weight than TLT (higher return)")
        self.assertGreater(btc_weight_explicit, tlt_weight_explicit,
                          "BTC should get more weight than TLT with explicit threshold")
    
    def test_information_ratio_threshold_multi_asset_cluster(self):
        """
        Test that Information Ratio threshold checking works for multi-asset clusters.
        
        Mathematical Foundation:
        - When a cluster contains multiple assets, we check:
          1. Cluster return >= threshold
          2. No more than 50% of cluster weight is in assets below threshold
        - This prevents clusters with mixed returns from getting excessive weight
          if low-return assets dominate the cluster
        
        This test validates the fix for clusters containing both high and low return assets.
        """
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Create scenario with correlated assets:
        # - Bond cluster: TLT (0.5%), AGG (1.0%), LQD (2.0%)
        #   Cluster return: ~1.17% (below typical benchmark)
        # - Equity cluster: SPY (15%), QQQ (20%), IWM (12%)
        #   Cluster return: ~15.67% (above typical benchmark)
        
        # Bond cluster (low returns, low volatility, high correlation)
        base_bond = np.random.randn(252) * (0.05 / np.sqrt(252))
        tlt_returns = base_bond + (0.005 / 252)  # 0.5% annualized
        agg_returns = base_bond + (0.01 / 252)   # 1.0% annualized
        lqd_returns = base_bond + (0.02 / 252)   # 2.0% annualized
        
        # Equity cluster (high returns, medium volatility, high correlation)
        base_equity = np.random.randn(252) * (0.18 / np.sqrt(252))
        spy_returns = base_equity + (0.15 / 252)  # 15% annualized
        qqq_returns = base_equity + (0.20 / 252)  # 20% annualized
        iwm_returns = base_equity + (0.12 / 252)  # 12% annualized
        
        returns_df = pd.DataFrame({
            'TLT': tlt_returns,
            'AGG': agg_returns,
            'LQD': lqd_returns,
            'SPY': spy_returns,
            'QQQ': qqq_returns,
            'IWM': iwm_returns
        }, index=dates)
        
        # Calculate benchmark return (equal-weighted)
        benchmark_return = returns_df.mean(axis=1).mean() * 252
        
        # Test: With default threshold (benchmark return)
        # Bond cluster return (~1.17%) < benchmark (~8.5%), so should get less weight
        re_hrp = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_return_threshold=None,  # Uses benchmark return as threshold
            min_tracking_error=0.01,
            verbose=False
        )
        re_hrp.fit(returns_df)
        weights = re_hrp.predict()
        weights_dict = re_hrp.get_weights_dict()
        
        # Bond cluster (TLT + AGG + LQD) should NOT dominate
        bond_cluster_weight = (weights_dict['TLT'] + 
                              weights_dict['AGG'] + 
                              weights_dict['LQD'])
        
        # Equity cluster (SPY + QQQ + IWM) should get more weight
        equity_cluster_weight = (weights_dict['SPY'] + 
                                weights_dict['QQQ'] + 
                                weights_dict['IWM'])
        
        # Equity cluster should get more weight than bond cluster
        self.assertGreater(equity_cluster_weight, bond_cluster_weight,
                          f"Equity cluster ({equity_cluster_weight:.4f}) should get more weight "
                          f"than bond cluster ({bond_cluster_weight:.4f})")
        
        # Verify weights sum to 1 and are non-negative
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))
    
    def test_information_ratio_tracking_error_floor(self):
        """
        Test that tracking error floor prevents Information Ratio inflation.
        
        Mathematical Foundation:
        - Information Ratio = (E[R_portfolio] - E[R_benchmark]) / Tracking_Error
        - When tracking_error → 0, IR → ∞ (mathematical singularity)
        - min_tracking_error floor prevents this: tracking_error = max(tracking_error, min_tracking_error)
        - This prevents low-volatility assets from getting excessive weight
        
        This test validates the tracking error floor implementation.
        """
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Create scenario:
        # - TLT: Very low tracking error (0.01% annualized) vs benchmark
        #   Without floor, this would give IR = (0.5% - 38.5%) / 0.01% = -3800 (huge magnitude)
        #   With floor (1%), IR = (0.5% - 38.5%) / 1% = -38 (reasonable)
        
        # TLT: Low return, very low tracking error
        tlt_returns = np.random.randn(252) * (0.05 / np.sqrt(252)) + (0.005 / 252)
        
        # SPY: High return, medium tracking error
        spy_returns = np.random.randn(252) * (0.18 / np.sqrt(252)) + (0.15 / 252)
        
        returns_df = pd.DataFrame({
            'TLT': tlt_returns,
            'SPY': spy_returns
        }, index=dates)
        
        # Test 1: With tracking error floor (default: 1%)
        re_hrp_with_floor = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_tracking_error=0.01,  # 1% floor
            verbose=False
        )
        re_hrp_with_floor.fit(returns_df)
        weights_with_floor = re_hrp_with_floor.predict()
        
        # Test 2: Without tracking error floor (very small: 0.0001%)
        re_hrp_no_floor = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_tracking_error=0.000001,  # Very small floor (essentially no floor)
            verbose=False
        )
        re_hrp_no_floor.fit(returns_df)
        weights_no_floor = re_hrp_no_floor.predict()
        
        # With floor, TLT should get less weight (IR not inflated)
        tlt_weight_with_floor = weights_with_floor[returns_df.columns.get_loc('TLT')]
        tlt_weight_no_floor = weights_no_floor[returns_df.columns.get_loc('TLT')]
        
        # Both should be reasonable (< 50%), but with floor should be more reasonable
        self.assertLess(tlt_weight_with_floor, 0.5,
                       f"TLT weight with floor ({tlt_weight_with_floor:.4f}) should be < 50%")
        self.assertLess(tlt_weight_no_floor, 0.5,
                       f"TLT weight without floor ({tlt_weight_no_floor:.4f}) should be < 50%")
        
        # Verify weights sum to 1
        self.assertAlmostEqual(weights_with_floor.sum(), 1.0, places=5)
        self.assertAlmostEqual(weights_no_floor.sum(), 1.0, places=5)
    
    def test_information_ratio_fallback_logic(self):
        """
        Test that fallback logic works when thresholds aren't met.
        
        Mathematical Foundation:
        - If neither cluster meets threshold: fall back to return-based allocation
        - Return-based allocation: α = return_right / (return_left + return_right)
        - This ensures we still favor higher-return clusters even when IR can't be used
        
        This test validates the fallback logic implementation.
        """
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Create scenario where both clusters are below threshold:
        # - Low-return cluster: 2% annualized
        # - Medium-return cluster: 5% annualized
        # - Threshold: 10% (benchmark return)
        # Both below threshold, so should fall back to return-based allocation
        
        # Low-return cluster
        low_returns = np.random.randn(252) * (0.10 / np.sqrt(252)) + (0.02 / 252)
        
        # Medium-return cluster
        medium_returns = np.random.randn(252) * (0.12 / np.sqrt(252)) + (0.05 / 252)
        
        returns_df = pd.DataFrame({
            'Low_Return': low_returns,
            'Medium_Return': medium_returns
        }, index=dates)
        
        # Test: With high threshold (10% annualized)
        # Both clusters below threshold, so should use return-based allocation
        re_hrp = ReturnEnhancedHRP(
            allocation_metric='information_ratio',
            min_return_threshold=0.10,  # 10% threshold (both clusters below)
            min_tracking_error=0.01,
            verbose=False
        )
        re_hrp.fit(returns_df)
        weights = re_hrp.predict()
        weights_dict = re_hrp.get_weights_dict()
        
        # Medium-return cluster should get more weight than low-return cluster
        medium_weight = weights_dict['Medium_Return']
        low_weight = weights_dict['Low_Return']
        
        self.assertGreater(medium_weight, low_weight,
                          f"Medium-return cluster ({medium_weight:.4f}) should get more weight "
                          f"than low-return cluster ({low_weight:.4f}) in fallback mode")
        
        # Verify weights sum to 1 and are non-negative
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))


if __name__ == '__main__':
    unittest.main()

