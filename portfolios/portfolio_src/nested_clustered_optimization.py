"""
Nested Clustered Optimization (NCO) Portfolio Optimizer.

This module implements the Nested Clustered Optimization algorithm as described
by Marcos Lopez de Prado. NCO addresses the "Markowitz curse" by:
1. Clustering assets into groups
2. Optimizing within each cluster
3. Optimizing between clusters using a reduced problem

This approach reduces the dimensionality and instability of traditional
mean-variance optimization.

**IMPORTANT ATTRIBUTION**: This implementation is based on the research of
Marcos Lopez de Prado. The original algorithm and mathematical foundations are
from his 2016 paper. This code provides a production-ready implementation and
demonstration of his work.

References:
    De Prado, M. L. (2016). Building Diversified Portfolios that Outperform
      Out of Sample. The Journal of Portfolio Management, 42(4), 59-69.
    
    DOI: https://doi.org/10.3905/jpm.2016.42.4.059
    
    This paper introduces NCO as a method to address the "Markowitz curse" -
    the instability and estimation errors in high-dimensional mean-variance
    optimization. The algorithm:
    1. Clusters assets using hierarchical clustering
    2. Optimizes within each cluster (using mean-variance or risk parity)
    3. Optimizes between clusters using a reduced covariance matrix
    4. Combines cluster weights with within-cluster weights
    
    This reduces the optimization problem from n assets to k clusters, where
    k << n, significantly improving stability and out-of-sample performance.
    
    Additional references:
    - Open-source implementations: emoen/Machine-Learning-for-Asset-Managers,
      skfolio, emialb34i/beyond-markowitz
    - De Prado's book: "Advances in Financial Machine Learning" (2018)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from typing import Optional, Literal
from .base_optimizer import BasePortfolioOptimizer


class NestedClusteredOptimization(BasePortfolioOptimizer):
    """
    Nested Clustered Optimization (NCO) Portfolio Optimizer.
    
    NCO performs a three-step optimization:
    1. Cluster assets into groups using hierarchical clustering
    2. Optimize within each cluster (mean-variance or risk parity)
    3. Optimize between clusters using a reduced covariance matrix
    
    This reduces the dimensionality of the optimization problem and improves
    out-of-sample stability.
    
    Attributes:
        n_clusters_ (int): Number of clusters used.
        cluster_labels_ (np.ndarray): Cluster assignment for each asset.
        within_cluster_weights_ (dict): Weights within each cluster.
        cluster_weights_ (np.ndarray): Weights for each cluster.
        cluster_cov_matrix_ (pd.DataFrame): Reduced covariance matrix at cluster level.
    """
    
    def __init__(self, n_clusters: Optional[int] = None, 
                 within_cluster_method: Literal['mean_variance', 'risk_parity'] = 'mean_variance',
                 linkage_method: str = 'ward',
                 target_return: Optional[float] = None):
        """
        Initialize NCO optimizer.
        
        Args:
            n_clusters (int, optional): Number of clusters. If None, automatically
                determined based on data. Defaults to None.
            within_cluster_method (str, optional): Method for within-cluster optimization.
                Options: 'mean_variance' or 'risk_parity'. Defaults to 'mean_variance'.
            linkage_method (str, optional): Linkage method for hierarchical clustering.
                Defaults to 'ward'.
            target_return (float, optional): Target return for mean-variance optimization.
                If None, maximizes Sharpe ratio. Defaults to None.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.within_cluster_method = within_cluster_method
        self.linkage_method = linkage_method
        self.target_return = target_return
        
        self.n_clusters_: Optional[int] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self.within_cluster_weights_: Optional[dict] = None
        self.cluster_weights_: Optional[np.ndarray] = None
        self.cluster_cov_matrix_: Optional[pd.DataFrame] = None
    
    def fit(self, returns_df: pd.DataFrame) -> 'NestedClusteredOptimization':
        """
        Fit the NCO optimizer on historical returns.
        
        This method performs the three-step NCO process:
        1. Cluster assets
        2. Optimize within clusters
        3. Optimize between clusters
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
        
        Returns:
            NestedClusteredOptimization: Returns self for method chaining.
        """
        # Validate and store returns
        returns_df = self._validate_returns(returns_df)
        self.returns_df = returns_df.copy()
        self.asset_names_ = list(returns_df.columns)
        
        # Compute covariance and correlation matrices
        self.cov_matrix_ = self._compute_covariance(returns_df)
        self.corr_matrix_ = self._compute_correlation(returns_df)
        
        # Step 1: Cluster assets
        self._cluster_assets()
        
        # Step 2: Optimize within each cluster
        self._optimize_within_clusters()
        
        # Step 3: Optimize between clusters
        self._optimize_between_clusters()
        
        # Combine cluster weights with within-cluster weights to get final weights
        self.weights_ = self._combine_weights()
        
        # Validate and normalize weights
        self.weights_ = self._validate_weights(self.weights_)
        
        return self
    
    def _cluster_assets(self):
        """Cluster assets using hierarchical clustering."""
        # Convert correlation to distance matrix
        distance_matrix = np.sqrt(2 * (1 - self.corr_matrix_.values))
        
        # Convert to condensed form for linkage
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
        
        # Determine number of clusters
        if self.n_clusters is None:
            # Use a heuristic: sqrt of number of assets, but at least 2
            self.n_clusters_ = max(2, int(np.sqrt(len(self.asset_names_))))
        else:
            self.n_clusters_ = min(self.n_clusters, len(self.asset_names_))
        
        # Get cluster labels
        self.cluster_labels_ = fcluster(
            linkage_matrix,
            self.n_clusters_,
            criterion='maxclust'
        )
    
    def _optimize_within_clusters(self):
        """Optimize portfolio weights within each cluster."""
        self.within_cluster_weights_ = {}
        cov_matrix = self.cov_matrix_.values
        expected_returns = self.returns_df.mean().values
        
        for cluster_id in range(1, self.n_clusters_ + 1):
            # Get assets in this cluster
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Subset covariance matrix and expected returns for this cluster
            cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
            cluster_returns = expected_returns[cluster_indices]
            
            if self.within_cluster_method == 'risk_parity':
                # Use risk parity within cluster
                weights = self._risk_parity_optimize(cluster_cov)
            else:
                # Use mean-variance optimization within cluster
                weights = self._mean_variance_optimize(cluster_cov, cluster_returns)
            
            # Store weights for this cluster
            self.within_cluster_weights_[cluster_id] = {
                'indices': cluster_indices,
                'weights': weights
            }
    
    def _risk_parity_optimize(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize using risk parity for a given covariance matrix.
        
        Edge Case Handling:
        - Single asset: returns weight of 1.0
        - Boundary solutions: redistributes weights < 1e-6 to prevent corner solutions
        - Optimization failure: falls back to equal weights
        
        Args:
            cov_matrix (np.ndarray): Covariance matrix for the cluster.
        
        Returns:
            np.ndarray: Optimized weights for assets in the cluster.
        """
        n = len(cov_matrix)
        
        # Handle edge case: single asset
        if n == 1:
            return np.array([1.0])
        
        initial_weights = np.ones(n) / n
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0, 1) for _ in range(n))
        
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            if portfolio_vol < 1e-10:
                return 1e10
            marginal_contrib = cov_matrix @ weights
            risk_contributions = weights * marginal_contrib / portfolio_vol
            target_rc = 1.0 / n
            diff = risk_contributions - target_rc
            return np.sum(diff ** 2)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success:
            weights = result.x
            # Handle boundary solutions: ensure minimum weight threshold
            min_weight_threshold = 1e-6
            small_weights = weights < min_weight_threshold
            
            if np.any(small_weights) and np.sum(~small_weights) > 0:
                # Redistribute small weights
                total_small = np.sum(weights[small_weights])
                weights[small_weights] = min_weight_threshold
                remaining_assets = ~small_weights
                if np.sum(remaining_assets) > 0:
                    excess = total_small - np.sum(weights[small_weights])
                    weights[remaining_assets] += excess * (weights[remaining_assets] / np.sum(weights[remaining_assets]))
                # Renormalize
                weights = weights / weights.sum()
            
            return weights
        else:
            # Fallback to equal weights
            return np.ones(n) / n
    
    def _mean_variance_optimize(self, cov_matrix: np.ndarray, 
                                expected_returns: np.ndarray) -> np.ndarray:
        """
        Optimize using mean-variance for a given covariance matrix and returns.
        
        Edge Case Handling:
        - Single asset: returns weight of 1.0
        - Boundary solutions: redistributes weights < 1e-6 to prevent corner solutions
        - Optimization failure: falls back to equal weights
        - Near-zero volatility: handles division by zero in Sharpe ratio calculation
        
        Args:
            cov_matrix (np.ndarray): Covariance matrix for the cluster.
            expected_returns (np.ndarray): Expected returns for assets in the cluster.
        
        Returns:
            np.ndarray: Optimized weights for assets in the cluster.
        """
        n = len(cov_matrix)
        
        # Handle edge case: single asset
        if n == 1:
            return np.array([1.0])
        
        initial_weights = np.ones(n) / n
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0, 1) for _ in range(n))
        
        if self.target_return is not None:
            # Minimize variance for target return
            target = self.target_return / 252  # Convert to daily if needed
            constraints.append({
                'type': 'eq',
                'fun': lambda w: expected_returns.T @ w - target
            })
            objective = lambda w: w.T @ cov_matrix @ w
        else:
            # Maximize Sharpe ratio (minimize negative Sharpe)
            def objective(weights):
                portfolio_return = expected_returns.T @ weights
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                if portfolio_vol < 1e-10:
                    return 1e10
                # Negative Sharpe (to minimize)
                return -portfolio_return / portfolio_vol
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success:
            weights = result.x
            # Handle boundary solutions: ensure minimum weight threshold
            min_weight_threshold = 1e-6
            small_weights = weights < min_weight_threshold
            
            if np.any(small_weights) and np.sum(~small_weights) > 0:
                # Redistribute small weights
                total_small = np.sum(weights[small_weights])
                weights[small_weights] = min_weight_threshold
                remaining_assets = ~small_weights
                if np.sum(remaining_assets) > 0:
                    excess = total_small - np.sum(weights[small_weights])
                    weights[remaining_assets] += excess * (weights[remaining_assets] / np.sum(weights[remaining_assets]))
                # Renormalize
                weights = weights / weights.sum()
            
            return weights
        else:
            # Fallback to equal weights
            return np.ones(n) / n
    
    def _optimize_between_clusters(self):
        """
        Optimize weights between clusters using reduced covariance matrix.
        
        This method:
        1. Builds a reduced covariance matrix at the cluster level
        2. Computes cluster-level expected returns
        3. Optimizes cluster weights using mean-variance optimization
        
        Edge Case Handling:
        - Boundary solutions: redistributes cluster weights < 1e-6 to prevent corner solutions
        - Optimization failure: falls back to equal cluster weights
        - Near-zero volatility: handles division by zero in Sharpe ratio calculation
        """
        # Build reduced covariance matrix at cluster level
        cluster_cov = np.zeros((self.n_clusters_, self.n_clusters_))
        cluster_returns = np.zeros(self.n_clusters_)
        
        cov_matrix = self.cov_matrix_.values
        expected_returns = self.returns_df.mean().values
        
        for i in range(1, self.n_clusters_ + 1):
            cluster_i = self.within_cluster_weights_[i]
            indices_i = cluster_i['indices']
            weights_i = cluster_i['weights']
            
            # Cluster return: weighted average of asset returns
            cluster_returns[i - 1] = expected_returns[indices_i].T @ weights_i
            
            for j in range(1, self.n_clusters_ + 1):
                cluster_j = self.within_cluster_weights_[j]
                indices_j = cluster_j['indices']
                weights_j = cluster_j['weights']
                
                # Cluster covariance: weights_i^T * Cov_ij * weights_j
                cov_ij = cov_matrix[np.ix_(indices_i, indices_j)]
                cluster_cov[i - 1, j - 1] = weights_i.T @ cov_ij @ weights_j
        
        self.cluster_cov_matrix_ = pd.DataFrame(
            cluster_cov,
            index=range(1, self.n_clusters_ + 1),
            columns=range(1, self.n_clusters_ + 1)
        )
        
        # Optimize cluster weights using mean-variance
        initial_weights = np.ones(self.n_clusters_) / self.n_clusters_
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0, 1) for _ in range(self.n_clusters_))
        
        if self.target_return is not None:
            target = self.target_return / 252
            constraints.append({
                'type': 'eq',
                'fun': lambda w: cluster_returns.T @ w - target
            })
            objective = lambda w: w.T @ cluster_cov @ w
        else:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = cluster_returns.T @ weights
                portfolio_vol = np.sqrt(weights.T @ cluster_cov @ weights)
                if portfolio_vol < 1e-10:
                    return 1e10
                return -portfolio_return / portfolio_vol
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success:
            cluster_weights = result.x
            # Handle boundary solutions: ensure minimum weight threshold
            min_weight_threshold = 1e-6
            small_weights = cluster_weights < min_weight_threshold
            
            if np.any(small_weights) and np.sum(~small_weights) > 0:
                # Redistribute small weights
                total_small = np.sum(cluster_weights[small_weights])
                cluster_weights[small_weights] = min_weight_threshold
                remaining_clusters = ~small_weights
                if np.sum(remaining_clusters) > 0:
                    excess = total_small - np.sum(cluster_weights[small_weights])
                    cluster_weights[remaining_clusters] += excess * (cluster_weights[remaining_clusters] / np.sum(cluster_weights[remaining_clusters]))
                # Renormalize
                cluster_weights = cluster_weights / cluster_weights.sum()
            
            self.cluster_weights_ = cluster_weights
        else:
            # Fallback to equal weights
            self.cluster_weights_ = np.ones(self.n_clusters_) / self.n_clusters_
    
    def _combine_weights(self) -> np.ndarray:
        """Combine cluster weights with within-cluster weights."""
        n_assets = len(self.asset_names_)
        final_weights = np.zeros(n_assets)
        
        for cluster_id in range(1, self.n_clusters_ + 1):
            cluster_info = self.within_cluster_weights_[cluster_id]
            cluster_weight = self.cluster_weights_[cluster_id - 1]
            within_weights = cluster_info['weights']
            indices = cluster_info['indices']
            
            # Final weight = cluster_weight * within_cluster_weight
            final_weights[indices] = cluster_weight * within_weights
        
        return final_weights
    
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
    
    def get_cluster_assignments(self) -> dict:
        """
        Get cluster assignments for each asset.
        
        Returns:
            dict: Dictionary mapping asset names to cluster IDs.
        
        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.cluster_labels_ is None or self.asset_names_ is None:
            raise ValueError("Must call fit() before get_cluster_assignments()")
        
        return dict(zip(self.asset_names_, self.cluster_labels_))

