"""
Covariance Matrix Denoising using Random Matrix Theory.

This module implements Marcos Lopez de Prado's covariance matrix denoising methodology
to filter out random noise from eigenvalue spectra. The denoising process identifies
eigenvalues that fall within the Marcenko-Pastur distribution range (random noise) and
applies one of three denoising methods: constant residual eigenvalue, targeted shrinkage,
or eigenvalue clipping.

**IMPORTANT ATTRIBUTION**: This implementation is based on the research of Marcos Lopez
de Prado. The denoising methodology using Random Matrix Theory and Marcenko-Pastur
distribution is from his work on robust covariance matrix estimation.

References:
    De Prado, M. L. (2016). Building Diversified Portfolios that Outperform
    Out of Sample. The Journal of Portfolio Management, 42(4), 59-69.
    
    DOI: https://doi.org/10.3905/jpm.2016.42.4.059
    
    De Prado, M. L. (2018). Advances in Financial Machine Learning. Wiley.
    (Chapter on covariance matrix denoising)
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional
from scipy.linalg import eigh


DenoisingMethod = Literal["constant_residual", "targeted_shrinkage", "eigenvalue_clipping"]


class CovarianceDenoiser:
    """
    Covariance and correlation matrix denoiser using Random Matrix Theory.
    
    This class implements de Prado's covariance matrix denoising methodology to remove
    random noise from eigenvalue spectra. The denoising process:
    
    1. Performs eigenvalue decomposition: R = Q × Λ × Q^T (for correlation matrices)
    2. Identifies random eigenvalues using Marcenko-Pastur distribution bounds
    3. Applies the selected denoising method to random eigenvalues
    4. Reconstructs the denoised matrix: R_denoised = Q × Λ_denoised × Q^T
    5. Optionally rescales to covariance: Σ = D R̂ D (if volatilities provided)
    
    **Recommended Usage**: Denoise correlation matrices (matrix_type="correlation").
    MP theory assumes isotropic noise (σ²I), which correlation matrices better satisfy.
    After denoising, correlation matrices can be rescaled to covariance using volatilities.
    
    The three available methods are:
    
    - **constant_residual** (standard/default): Replace random eigenvalues with their
      average value. This preserves the trace of the matrix and is the standard approach.
    
    - **targeted_shrinkage**: Shrink eigenvalues toward the mean eigenvalue using a
      shrinkage factor. More conservative than constant residual.
    
    - **eigenvalue_clipping**: Set eigenvalues below the Marcenko-Pastur upper bound
      to the threshold value. More aggressive denoising.
    
    Attributes:
        None (stateless class)
    """
    
    def __init__(self):
        """Initialize the covariance denoiser."""
        pass
    
    def denoise(self, matrix: pd.DataFrame, 
                method: DenoisingMethod = "constant_residual",
                num_observations: Optional[int] = None,
                matrix_type: Literal["correlation", "covariance"] = "correlation",
                volatilities: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Denoise a correlation or covariance matrix using Random Matrix Theory.
        
        This method performs eigenvalue decomposition, identifies random eigenvalues
        using Marcenko-Pastur distribution bounds, applies the selected denoising
        method, and reconstructs the denoised matrix.
        
        **Recommended Usage**: Denoise correlation matrices (matrix_type="correlation").
        MP theory assumes isotropic noise (σ²I), which correlation matrices better satisfy.
        If volatilities are provided, the denoised correlation is rescaled to covariance.
        
        Args:
            matrix (pd.DataFrame): Correlation or covariance matrix to denoise.
                Must be symmetric and positive semi-definite.
            method (DenoisingMethod, optional): Denoising method to apply.
                Options: 'constant_residual' (default/standard), 'targeted_shrinkage',
                or 'eigenvalue_clipping'. Defaults to 'constant_residual'.
            num_observations (int, optional): Number of observations (time periods)
                used to compute the matrix. Required for Marcenko-Pastur bounds.
                Defaults to None, which will raise an error.
            matrix_type (str, optional): Type of input matrix.
                - 'correlation' (default, recommended): Input is a correlation matrix.
                  MP assumes isotropic noise, which correlation matrices better satisfy.
                - 'covariance': Input is a covariance matrix. For backward compatibility.
                  Internally converts to correlation, denoises, then converts back.
            volatilities (pd.Series, optional): Standard deviations (volatilities) for each asset.
                Required if matrix_type="correlation" and you want covariance matrix returned.
                Should have index matching matrix index/columns.
                If provided, returns rescaled covariance: Σ = D R̂ D where D = diag(volatilities).
        
        Returns:
            pd.DataFrame: Denoised matrix (correlation or covariance) with same index and columns
                as the input matrix.
                - If matrix_type="correlation" and volatilities=None: returns denoised correlation
                - If matrix_type="correlation" and volatilities provided: returns denoised covariance
                - If matrix_type="covariance": returns denoised covariance (backward compatibility)
        
        Raises:
            ValueError: If matrix is not square or symmetric, or if
                num_observations is None (required for Marcenko-Pastur bounds).
            ValueError: If method is not one of the valid options.
            ValueError: If matrix_type="correlation" and volatilities provided but don't match matrix dimensions.
        """
        if method not in ["constant_residual", "targeted_shrinkage", "eigenvalue_clipping"]:
            raise ValueError(
                "method must be 'constant_residual', 'targeted_shrinkage', or "
                "'eigenvalue_clipping'"
            )
        
        if matrix_type not in ["correlation", "covariance"]:
            raise ValueError("matrix_type must be 'correlation' or 'covariance'")
        
        if num_observations is None:
            raise ValueError(
                "num_observations must be provided to calculate Marcenko-Pastur bounds. "
                "This should be the number of time periods (observations) used to "
                "compute the matrix."
            )
        
        # Convert to numpy array for eigenvalue computation
        matrix_array = matrix.values
        n_assets = matrix_array.shape[0]
        
        # Validate matrix properties
        if matrix_array.shape[0] != matrix_array.shape[1]:
            raise ValueError("Matrix must be square")
        
        if not np.allclose(matrix_array, matrix_array.T):
            raise ValueError("Matrix must be symmetric")
        
        # Handle covariance matrix input (backward compatibility)
        # Convert to correlation, denoise, then convert back
        if matrix_type == "covariance":
            # Extract volatilities from covariance matrix
            volatilities_array = np.sqrt(np.diag(matrix_array))
            D = np.diag(volatilities_array)
            D_inv = np.diag(1.0 / volatilities_array)
            
            # Convert covariance to correlation: R = D^(-1) Σ D^(-1)
            corr_array = D_inv @ matrix_array @ D_inv
            
            # Ensure diagonal is exactly 1.0 and clip to [-1, 1]
            np.fill_diagonal(corr_array, 1.0)
            corr_array = np.clip(corr_array, -1.0, 1.0)
            
            # Denoise correlation matrix
            corr_denoised = self._denoise_correlation_matrix(
                corr_array, method, num_observations, n_assets, matrix.index, matrix.columns
            )
            
            # Convert back to covariance: Σ = D R̂ D
            cov_denoised_array = D @ corr_denoised.values @ D
            
            # Ensure symmetry
            cov_denoised_array = (cov_denoised_array + cov_denoised_array.T) / 2
            
            result = pd.DataFrame(
                cov_denoised_array,
                index=matrix.index,
                columns=matrix.columns
            )
            
            return result
        
        # Handle correlation matrix input (recommended path)
        # Validate correlation matrix properties
        if not np.allclose(np.diag(matrix_array), 1.0, atol=1e-10):
            raise ValueError(
                "Correlation matrix diagonal must be 1.0. "
                "Found diagonal values that deviate significantly from 1.0."
            )
        
        # Clip correlation values to [-1, 1] range
        matrix_array = np.clip(matrix_array, -1.0, 1.0)
        np.fill_diagonal(matrix_array, 1.0)
        
        # Denoise correlation matrix
        corr_denoised = self._denoise_correlation_matrix(
            matrix_array, method, num_observations, n_assets, matrix.index, matrix.columns
        )
        
        # If volatilities provided, rescale to covariance
        if volatilities is not None:
            if len(volatilities) != n_assets:
                raise ValueError(
                    f"volatilities length ({len(volatilities)}) must match matrix "
                    f"dimensions ({n_assets})"
                )
            
            # Ensure volatilities are aligned with matrix index
            if isinstance(volatilities, pd.Series):
                if not volatilities.index.equals(matrix.index):
                    # Try to reindex
                    volatilities = volatilities.reindex(matrix.index)
                    if volatilities.isna().any():
                        raise ValueError(
                            "volatilities index must match matrix index/columns"
                        )
                vols_array = volatilities.values
            else:
                vols_array = np.asarray(volatilities)
            
            # Rescale to covariance: Σ = D R̂ D where D = diag(volatilities)
            D = np.diag(vols_array)
            cov_denoised_array = D @ corr_denoised.values @ D
            
            # Ensure symmetry
            cov_denoised_array = (cov_denoised_array + cov_denoised_array.T) / 2
            
            result = pd.DataFrame(
                cov_denoised_array,
                index=matrix.index,
                columns=matrix.columns
            )
            
            return result
        
        # Return denoised correlation matrix
        return corr_denoised
    
    def _denoise_correlation_matrix(self, corr_array: np.ndarray,
                                    method: DenoisingMethod,
                                    num_observations: int,
                                    num_assets: int,
                                    index: pd.Index,
                                    columns: pd.Index) -> pd.DataFrame:
        """
        Internal method to denoise a correlation matrix.
        
        This is the core denoising logic that operates on correlation matrices.
        MP theory assumes isotropic noise (σ²I), which correlation matrices satisfy.
        
        Args:
            corr_array (np.ndarray): Correlation matrix (numpy array).
            method (DenoisingMethod): Denoising method to apply.
            num_observations (int): Number of observations.
            num_assets (int): Number of assets.
            index (pd.Index): Index for output DataFrame.
            columns (pd.Index): Columns for output DataFrame.
        
        Returns:
            pd.DataFrame: Denoised correlation matrix.
        """
        # Perform eigenvalue decomposition
        # Use eigh for symmetric matrices (more stable and efficient)
        eigenvalues, eigenvectors = eigh(corr_array)
        
        # Sort eigenvalues in descending order (eigh returns in ascending order)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Calculate Marcenko-Pastur distribution bounds
        lambda_max, lambda_min = self._marcenko_pastur_bounds(
            eigenvalues, num_observations, num_assets
        )
        
        # Identify random eigenvalues (those within MP bounds)
        random_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
        signal_mask = ~random_mask
        
        # Apply denoising method
        eigenvalues_denoised = eigenvalues.copy()
        
        if method == "constant_residual":
            eigenvalues_denoised = self._constant_residual_method(
                eigenvalues, random_mask, signal_mask
            )
        elif method == "targeted_shrinkage":
            eigenvalues_denoised = self._targeted_shrinkage_method(
                eigenvalues, random_mask, signal_mask
            )
        elif method == "eigenvalue_clipping":
            eigenvalues_denoised = self._eigenvalue_clipping_method(
                eigenvalues, random_mask, signal_mask, lambda_max
            )
        
        # Reconstruct correlation matrix: R_denoised = Q × Λ_denoised × Q^T
        lambda_matrix = np.diag(eigenvalues_denoised)
        corr_denoised_array = eigenvectors @ lambda_matrix @ eigenvectors.T
        
        # Ensure symmetry (numerical precision)
        corr_denoised_array = (corr_denoised_array + corr_denoised_array.T) / 2
        
        # Clip to [-1, 1] and ensure diagonal is 1.0
        corr_denoised_array = np.clip(corr_denoised_array, -1.0, 1.0)
        np.fill_diagonal(corr_denoised_array, 1.0)
        
        # Convert back to DataFrame
        corr_denoised = pd.DataFrame(
            corr_denoised_array,
            index=index,
            columns=columns
        )
        
        return corr_denoised
    
    def _rescale_to_covariance(self, corr_matrix: pd.DataFrame,
                               volatilities: pd.Series) -> pd.DataFrame:
        """
        Rescale a denoised correlation matrix to covariance matrix.
        
        Formula: Σ = D R̂ D where D = diag(σ₁, σ₂, ..., σₙ) contains the volatilities.
        
        Args:
            corr_matrix (pd.DataFrame): Denoised correlation matrix.
            volatilities (pd.Series): Standard deviations (volatilities) for each asset.
                Must have index matching corr_matrix index/columns.
        
        Returns:
            pd.DataFrame: Rescaled covariance matrix.
        
        Raises:
            ValueError: If volatilities don't match matrix dimensions.
        """
        if len(volatilities) != corr_matrix.shape[0]:
            raise ValueError(
                f"volatilities length ({len(volatilities)}) must match matrix "
                f"dimensions ({corr_matrix.shape[0]})"
            )
        
        # Ensure volatilities are aligned
        if not volatilities.index.equals(corr_matrix.index):
            volatilities = volatilities.reindex(corr_matrix.index)
            if volatilities.isna().any():
                raise ValueError(
                    "volatilities index must match correlation matrix index/columns"
                )
        
        # Extract as arrays
        corr_array = corr_matrix.values
        vols_array = volatilities.values
        
        # Rescale: Σ = D R̂ D
        D = np.diag(vols_array)
        cov_array = D @ corr_array @ D
        
        # Ensure symmetry
        cov_array = (cov_array + cov_array.T) / 2
        
        return pd.DataFrame(
            cov_array,
            index=corr_matrix.index,
            columns=corr_matrix.columns
        )
    
    def _marcenko_pastur_bounds(self, eigenvalues: np.ndarray,
                                num_observations: int,
                                num_assets: int) -> tuple:
        """
        Calculate Marcenko-Pastur distribution bounds.
        
        For a covariance matrix computed from T observations of N assets,
        the Marcenko-Pastur distribution describes the distribution of random
        eigenvalues. Random eigenvalues lie within [λ_min, λ_max]:
        
        - λ_min = σ²(1 - √(1/Q))² (when Q ≥ 1, else 0)
        - λ_max = σ²(1 + √(1/Q))²
        
        where Q = T/N (observations/assets ratio) and σ² is typically estimated
        as the mean of the eigenvalues.
        
        Note: The canonical MP formula uses q = N/T, giving λ± = σ²(1 ± √q)².
        When using Q = T/N, we use λ± = σ²(1 ± √(1/Q))², which is equivalent.
        
        In practice, we use the mean of all eigenvalues as σ². For correlation
        matrices, σ² ≈ 1 (since trace = N and mean eigenvalue ≈ 1).
        
        Args:
            eigenvalues (np.ndarray): Eigenvalues in descending order.
            num_observations (int): Number of observations (T).
            num_assets (int): Number of assets (N).
        
        Returns:
            tuple[float, float]: (lambda_max, lambda_min) bounds.
        """
        Q = num_observations / num_assets  # Q = T/N
        
        # Estimate σ² from eigenvalue mean
        # For correlation matrices, σ² ≈ 1 (trace = N, so mean ≈ 1)
        # For covariance matrices, σ² is the mean eigenvalue
        sigma_sq = np.mean(eigenvalues)
        
        # Calculate bounds using corrected formula: λ± = σ²(1 ± √(1/Q))²
        # When Q < 1 (N > T), we have rank-deficient case and λ_min = 0
        if Q >= 1:
            lambda_max = sigma_sq * (1 + np.sqrt(1/Q)) ** 2
            lambda_min = sigma_sq * (1 - np.sqrt(1/Q)) ** 2
        else:
            # Rank-deficient case: N > T, so λ_min = 0
            lambda_max = sigma_sq * (1 + np.sqrt(1/Q)) ** 2
            lambda_min = 0.0
        
        return lambda_max, lambda_min
    
    def _constant_residual_method(self, eigenvalues: np.ndarray,
                                  random_mask: np.ndarray,
                                  signal_mask: np.ndarray) -> np.ndarray:
        """
        Constant residual eigenvalue method (standard approach).
        
        Replace random eigenvalues with their average value, preserving signal
        eigenvalues. This method preserves the trace of the matrix.
        
        Args:
            eigenvalues (np.ndarray): Original eigenvalues.
            random_mask (np.ndarray): Boolean mask for random eigenvalues.
            signal_mask (np.ndarray): Boolean mask for signal eigenvalues.
        
        Returns:
            np.ndarray: Denoised eigenvalues.
        """
        eigenvalues_denoised = eigenvalues.copy()
        
        # Calculate mean of random eigenvalues
        random_eigenvalues = eigenvalues[random_mask]
        
        if len(random_eigenvalues) > 0:
            mean_random = np.mean(random_eigenvalues)
            # Replace random eigenvalues with their mean
            eigenvalues_denoised[random_mask] = mean_random
        
        # Signal eigenvalues remain unchanged (already preserved by copy)
        
        return eigenvalues_denoised
    
    def _targeted_shrinkage_method(self, eigenvalues: np.ndarray,
                                   random_mask: np.ndarray,
                                   signal_mask: np.ndarray) -> np.ndarray:
        """
        Targeted shrinkage method.
        
        Shrink eigenvalues toward the mean eigenvalue using a shrinkage factor.
        This is more conservative than constant residual method.
        
        Args:
            eigenvalues (np.ndarray): Original eigenvalues.
            random_mask (np.ndarray): Boolean mask for random eigenvalues.
            signal_mask (np.ndarray): Boolean mask for signal eigenvalues.
        
        Returns:
            np.ndarray: Denoised eigenvalues.
        """
        eigenvalues_denoised = eigenvalues.copy()
        
        # Calculate mean eigenvalue (global mean)
        mean_eigenvalue = np.mean(eigenvalues)
        
        # Shrink random eigenvalues toward mean
        # Typical shrinkage factor: shrink by some fraction (e.g., 0.5 means halfway to mean)
        shrinkage_factor = 0.5  # Can be made configurable if needed
        
        random_eigenvalues = eigenvalues[random_mask]
        if len(random_eigenvalues) > 0:
            # Shrink: λ_new = λ_old + α * (λ_mean - λ_old) where α is shrinkage factor
            eigenvalues_denoised[random_mask] = (
                eigenvalues[random_mask] + 
                shrinkage_factor * (mean_eigenvalue - eigenvalues[random_mask])
            )
        
        # Signal eigenvalues remain unchanged
        
        return eigenvalues_denoised
    
    def _eigenvalue_clipping_method(self, eigenvalues: np.ndarray,
                                    random_mask: np.ndarray,
                                    signal_mask: np.ndarray,
                                    lambda_max: float) -> np.ndarray:
        """
        Eigenvalue clipping method.
        
        Set eigenvalues below the Marcenko-Pastur upper bound to the threshold
        value. This is more aggressive denoising than constant residual.
        
        Args:
            eigenvalues (np.ndarray): Original eigenvalues.
            random_mask (np.ndarray): Boolean mask for random eigenvalues.
            signal_mask (np.ndarray): Boolean mask for signal eigenvalues.
            lambda_max (float): Marcenko-Pastur upper bound.
        
        Returns:
            np.ndarray: Denoised eigenvalues.
        """
        eigenvalues_denoised = eigenvalues.copy()
        
        # Clip random eigenvalues to lambda_max
        eigenvalues_denoised[random_mask] = lambda_max
        
        # Signal eigenvalues remain unchanged
        
        return eigenvalues_denoised

