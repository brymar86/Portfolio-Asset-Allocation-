"""
Hierarchical Risk Parity (HRP) Portfolio Optimizer.

This module implements the Hierarchical Risk Parity algorithm as described by
Marcos Lopez de Prado. HRP uses hierarchical clustering to construct diversified
portfolios without inverting the covariance matrix, making it more stable than
traditional mean-variance optimization.

**IMPORTANT ATTRIBUTION**: This implementation is based on the research of
Marcos Lopez de Prado. The original algorithm and mathematical foundations are
from his 2016 paper. This code provides a production-ready implementation and
demonstration of his work.

References:
    De Prado, M. L. (2016). Building Diversified Portfolios that Outperform
      Out of Sample. The Journal of Portfolio Management, 42(4), 59-69.
    
    DOI: https://doi.org/10.3905/jpm.2016.42.4.059
    
    This paper introduces HRP as a method to address the instability of
    mean-variance optimization by using hierarchical clustering to avoid
    covariance matrix inversion. The algorithm:
    1. Converts correlation matrix to distance matrix
    2. Builds hierarchical clustering tree
    3. Quasi-diagonalizes the covariance matrix
    4. Recursively allocates risk down the tree
    
    Additional references:
    - Open-source implementations: PyPortfolioOpt, tschm/pyhrp, andreybabynin/HRP
    - De Prado's book: "Advances in Financial Machine Learning" (2018)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import Optional, Literal
from .base_optimizer import BasePortfolioOptimizer


class HierarchicalRiskParity(BasePortfolioOptimizer):
    """
    Hierarchical Risk Parity (HRP) Portfolio Optimizer.
    
    HRP addresses the instability of mean-variance optimization by:
    1. Building a hierarchical clustering tree from the correlation matrix
    2. Quasi-diagonalizing the covariance matrix based on the tree structure
    3. Recursively allocating risk down the tree
    
    This approach avoids matrix inversion and is more robust to estimation errors.
    
    Attributes:
        linkage_matrix_ (np.ndarray): Linkage matrix from hierarchical clustering.
        tree_order_ (np.ndarray): Order of assets after quasi-diagonalization.
        cov_quasi_diag_ (pd.DataFrame): Quasi-diagonalized covariance matrix.
    """
    
    def __init__(self, linkage_method: str = 'ward', distance_metric: str = 'euclidean',
                 denoise: bool = False,
                 denoising_method: Literal["constant_residual", "targeted_shrinkage", "eigenvalue_clipping"] = "constant_residual"):
        """
        Initialize HRP optimizer.
        
        Args:
            linkage_method (str, optional): Linkage method for hierarchical clustering.
                Options: 'ward', 'single', 'complete', 'average'. Defaults to 'ward'.
            distance_metric (str, optional): Distance metric for clustering.
                Defaults to 'euclidean'.
            denoise (bool, optional): If True, apply covariance matrix denoising before
                optimization. Denoising removes random noise from the eigenvalue spectrum
                using Random Matrix Theory. Defaults to False.
            denoising_method (str, optional): Denoising method to use when denoise=True.
                Options: 'constant_residual' (default/standard), 'targeted_shrinkage',
                or 'eigenvalue_clipping'. Defaults to 'constant_residual'.
        """
        super().__init__()
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.denoise = denoise
        self.denoising_method = denoising_method
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.tree_order_: Optional[np.ndarray] = None
        self.cov_quasi_diag_: Optional[pd.DataFrame] = None
    
    def fit(self, returns_df: pd.DataFrame) -> 'HierarchicalRiskParity':
        """
        Fit the HRP optimizer on historical returns.
        
        This method:
        1. Computes correlation and covariance matrices
        2. Converts correlation to distance matrix
        3. Builds hierarchical clustering tree
        4. Quasi-diagonalizes the covariance matrix
        5. Computes HRP weights through recursive allocation
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
        
        Returns:
            HierarchicalRiskParity: Returns self for method chaining.
        """
        # Validate and store returns
        returns_df = self._validate_returns(returns_df)
        self.returns_df = returns_df.copy()
        self.asset_names_ = list(returns_df.columns)
        
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
        
        # Step 5: Recursively allocate risk down the tree
        # Weights are computed in quasi-diagonal order
        weights_quasi_order = self._recursive_allocation(
            self.cov_quasi_diag_.values
        )
        
        # Reorder weights back to original asset order
        original_order = np.argsort(self.tree_order_)
        self.weights_ = weights_quasi_order[original_order]
        
        # Validate and normalize weights
        self.weights_ = self._validate_weights(self.weights_)
        
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
    
    def _recursive_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Recursively allocate risk down the hierarchical tree.
        
        This is the core HRP algorithm. It recursively splits the portfolio
        and allocates weights inversely proportional to variance, ensuring
        risk parity between sub-portfolios.
        
        Edge Case Handling:
        - Single asset: returns weight of 1.0 (base case)
        - Near-zero variance: uses equal weights (alpha = 0.5) when total variance < 1e-10
        - Numerical stability: clamps alpha to [1e-6, 1-1e-6] to avoid extreme values
        
        Args:
            cov_matrix (np.ndarray): Quasi-diagonalized covariance matrix.
        
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
        
        # Recursively allocate weights for left and right sub-portfolios
        weights_left = self._recursive_allocation(cov_left)
        weights_right = self._recursive_allocation(cov_right)
        
        # Calculate variance of each sub-portfolio
        var_left = weights_left.T @ cov_left @ weights_left
        var_right = weights_right.T @ cov_right @ weights_right
        
        # Handle edge case: if variances are too small or equal, use equal weights
        total_var = var_left + var_right
        if total_var < 1e-10:
            # Both sub-portfolios have near-zero variance, use equal weights
            alpha = 0.5
        else:
            # Allocate risk inversely proportional to variance
            # Lower variance sub-portfolio gets higher weight
            # This ensures risk parity between sub-portfolios
            # Use inverse variance weighting: alpha = var_right / (var_left + var_right)
            alpha = var_right / total_var
            
            # Clamp alpha to avoid extreme values (numerical stability)
            alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
        
        # Combine weights
        weights = np.concatenate([
            alpha * weights_left,
            (1 - alpha) * weights_right
        ])
        
        return weights
    
    def _find_split_point(self, cov_matrix: np.ndarray) -> int:
        """
        Find the optimal split point in the quasi-diagonalized covariance matrix.
        
        The split point is chosen to minimize the variance of the resulting
        sub-portfolios. This is done by trying all possible split points and
        selecting the one that minimizes the combined variance.
        
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
        hierarchical tree structure used by HRP.
        
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
        
        ax.set_title('Hierarchical Clustering Tree (HRP)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        if fig is not None:
            plt.tight_layout()
        
        return (fig, ax) if fig is not None else (None, ax)

