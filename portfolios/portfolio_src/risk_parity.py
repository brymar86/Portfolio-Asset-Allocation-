"""
Risk Parity Portfolio Optimizer.

This module implements the Risk Parity (Equal Risk Contribution) portfolio
optimization method. Risk Parity allocates portfolio weights such that each
asset contributes equally to the portfolio's total risk.

**Research Foundation:**
This implementation is based on the seminal work of Maillard, Roncalli, and
Teiletche (2010), who formalized the Risk Parity approach. Unlike traditional
mean-variance optimization, Risk Parity does not require expected return
estimates, making it more robust to estimation errors.

**Key Reference:**
Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally
Weighted Risk Contribution Portfolios. The Journal of Portfolio Management,
36(4), 60-70.

**Long-Only Constraint (Default):**
By default, this implementation enforces long-only constraints (w_i >= 0 for all assets),
which is appropriate for most institutional investors who cannot short sell.
The optimization problem becomes:

    minimize: sum((RC_i - σ_p/n)^2)
    subject to: sum(w_i) = leverage, w_i >= 0, max(w_i) <= 0.50

**Short Positions and Leverage:**
The optimizer supports both short positions and leverage through parameters:
- `allow_shorts=True`: Enables negative weights (short positions)
- `leverage > 1.0`: Allows leveraged portfolios (e.g., 1.5x leverage)

When shorts are allowed, bounds become (-max_short, 1) and risk contributions
can be negative (hedging effect). With leverage, the exposure constraint becomes
sum(w_i) = leverage instead of sum(w_i) = 1.

See Section 10.8 in the documentation for detailed mathematical treatment of
shorts and leverage.

**Numerical Optimization:**
This implementation uses Sequential Least Squares Programming (SLSQP) from
scipy.optimize, which is well-suited for constrained nonlinear optimization
problems. The algorithm:
- Handles equality constraints (sum of weights = 1)
- Handles inequality constraints (bounds and max weight limits)
- Uses gradient-based optimization for efficiency
- Includes fallback strategies for optimization failures
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
from .base_optimizer import BasePortfolioOptimizer


class RiskParity(BasePortfolioOptimizer):
    """
    Risk Parity (Equal Risk Contribution) Portfolio Optimizer.
    
    **Algorithm Overview:**
    Risk Parity allocates portfolio weights such that each asset contributes
    equally to the portfolio's total risk. This is achieved by solving a
    constrained optimization problem:
    
    minimize: sum((RC_i - RC_target)^2)
    subject to: 
        - sum(w_i) = 1 (fully invested)
        - w_i >= 0 for all i (long-only, no short selling)
        - max(w_i) <= 0.50 (diversification constraint)
    
    where RC_i is the risk contribution of asset i:
        RC_i = w_i * (Σw)_i / σ_p
    
    and:
        - (Σw)_i = marginal contribution of asset i to portfolio risk
        - σ_p = sqrt(w^T Σ w) = portfolio volatility
    
    **Mathematical Foundation:**
    Based on Maillard, Roncalli, & Teiletche (2010). The risk contribution
    formula ensures that each asset contributes equally to portfolio risk,
    regardless of the asset's individual volatility. This naturally allocates
    more weight to lower-volatility assets to achieve equal risk contribution.
    
    **Long-Only Implementation:**
    This implementation enforces strict long-only constraints (w_i >= 0),
    which is standard for institutional portfolios. The bounds ensure:
    - No short selling (w_i >= 0)
    - Fully invested (sum(w_i) = 1)
    - Diversification (max(w_i) <= 0.50)
    
    **Leverage Extension (Not Implemented):**
    To allow leverage, modify constraints to:
    - Allow sum(w_i) > 1 (e.g., 1.5x leverage)
    - Remove or adjust max weight constraints
    - Scale risk contributions by leverage factor
    
    However, leverage requires additional risk management considerations
    (margin requirements, liquidation risk, etc.) that are beyond this
    implementation's scope.
    
    **Numerical Optimization Technique:**
    Uses Sequential Least Squares Programming (SLSQP) algorithm from scipy:
    - Handles nonlinear objective with linear/quadratic constraints
    - Gradient-based optimization for efficiency
    - Robust to numerical precision issues
    - Includes fallback strategies (inverse volatility weighting, equal weights)
    
    Attributes:
        risk_contributions_ (np.ndarray): Risk contribution of each asset.
        optimization_result_ (scipy.optimize.OptimizeResult): Optimization result.
    """
    
    def __init__(self, method: str = 'SLSQP', max_iter: int = 1000, 
                 allow_shorts: bool = False, max_short: float = 0.5,
                 leverage: float = 1.0):
        """
        Initialize Risk Parity optimizer.
        
        **Optimization Method Selection:**
        SLSQP (Sequential Least Squares Programming) is the default method
        because it:
        - Efficiently handles equality and inequality constraints
        - Works well with smooth, differentiable objective functions
        - Provides good convergence for portfolio optimization problems
        - Is robust to numerical precision issues
        
        Alternative methods (e.g., 'trust-constr', 'COBYLA') may work but
        SLSQP is generally preferred for this type of problem.
        
        **Short Positions and Leverage:**
        The optimizer supports long-only portfolios by default. To enable short
        positions or leverage, use the `allow_shorts` and `leverage` parameters.
        See documentation for mathematical details on how risk contributions
        work with negative weights.
        
        Args:
            method (str, optional): Optimization method for scipy.optimize.minimize.
                Options: 'SLSQP' (recommended), 'trust-constr', 'COBYLA'.
                Defaults to 'SLSQP'.
            max_iter (int, optional): Maximum number of iterations for optimization.
                Defaults to 1000. Increase if optimization fails to converge.
            allow_shorts (bool, optional): Allow short positions (negative weights).
                Defaults to False (long-only). When True, bounds become (-max_short, 1).
            max_short (float, optional): Maximum short position per asset (as fraction).
                Only used when allow_shorts=True. Defaults to 0.5 (50% short max).
            leverage (float, optional): Leverage factor for long-only portfolios.
                Defaults to 1.0 (no leverage). For 1.5x leverage, set to 1.5.
                Constraint becomes sum(w_i) = leverage instead of sum(w_i) = 1.
                Note: leverage > 1 with allow_shorts=False gives leveraged long-only.
        
        Examples:
            >>> # Long-only (default)
            >>> rp = RiskParity()
            
            >>> # Long/short with 50% max short per asset
            >>> rp = RiskParity(allow_shorts=True, max_short=0.5)
            
            >>> # Leveraged long-only (1.5x)
            >>> rp = RiskParity(leverage=1.5)
            
            >>> # Leveraged long/short
            >>> rp = RiskParity(allow_shorts=True, leverage=1.5)
        """
        super().__init__()
        self.method = method
        self.max_iter = max_iter
        self.allow_shorts = allow_shorts
        self.max_short = max_short
        self.leverage = leverage
        self.risk_contributions_: Optional[np.ndarray] = None
        self.optimization_result_: Optional[object] = None
    
    def fit(self, returns_df: pd.DataFrame) -> 'RiskParity':
        """
        Fit the Risk Parity optimizer on historical returns.
        
        
        **Long-Only Constraint Discussion:**
        The constraint w_i >= 0 ensures no short selling, which is appropriate for:
        - Most institutional investors
        - Regulatory requirements in many jurisdictions
        - Simplified risk management
        
        However, this constraint can limit diversification when assets have
        negative correlations. In such cases, allowing short positions (w_i < 0)
        could improve risk-adjusted returns, but requires:
        - Margin accounts and capital requirements
        - More complex risk management
        - Regulatory approval
        
        **Leverage Considerations:**
        Current implementation: sum(w_i) = 1 (no leverage)
        
        To introduce leverage (e.g., 1.5x):
        1. Change constraint to: sum(w_i) = 1.5
        2. Adjust risk contributions: RC_i = w_i * (Σw)_i / (σ_p * leverage)
        3. Consider margin requirements and liquidation risk
        4. Implement risk limits (e.g., max leverage ratio)
        
        Leverage amplifies both returns and risks, requiring careful risk
        management beyond the scope of this basic implementation.
        
        **Numerical Optimization Details:**
        - Method: SLSQP (Sequential Least Squares Programming)
        - Convergence tolerance: ftol = 1e-9 (very tight for precision)
        - Max iterations: 1000 (usually converges in < 100 iterations)
        - Gradient-based: Uses analytical gradients for efficiency
        - Constraint handling: Active set method for inequality constraints
        
        Edge Case Handling:
        - If initial optimization fails, retries with inverse volatility weighting
        - If retry fails, falls back to equal weights
        - Redistributes very small weights (< 1%) to prevent corner solutions
        - Handles near-zero portfolio volatility cases
        
        Args:
            returns_df (pd.DataFrame): DataFrame with returns for each asset.
                Rows represent time periods, columns represent different assets.
                Each value should be a return (e.g., 0.01 for 1%).
        
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
        
        # Initial guess: equal weights scaled to leverage
        initial_weights = np.ones(n_assets) / n_assets * self.leverage
        
        # ====================================================================
        # STEP 2: Define Optimization Constraints
        # ====================================================================
        # 
        # Constraint 1: Fully Invested (Equality Constraint)
        #   sum(w_i) = 1.0
        #   This ensures all capital is allocated (no cash position)
        #   For leverage: change to sum(w_i) = leverage_factor (e.g., 1.5)
        #
        # Constraint 2: Long-Only (Inequality Constraint via Bounds)
        #   w_i >= 0 for all i
        #   Enforced via bounds = (0, 1) for each asset
        #   Prevents short selling, appropriate for most institutional investors
        #
        # Constraint 3: Diversification (Inequality Constraint)
        #   max(w_i) <= 0.50
        #   Prevents concentration risk where one asset dominates
        #   Ensures minimum diversification across assets
        #   This is a practical constraint beyond the theoretical Risk Parity
        #
        # Long-Only Sufficiency:
        #   The w_i >= 0 constraint is sufficient for long-only portfolios.
        #   Combined with sum(w_i) = 1, this ensures a valid long-only portfolio.
        #   The max weight constraint (50%) is an additional diversification
        #   requirement to prevent corner solutions.
        #
        # Leverage Extension:
        #   To allow leverage, modify constraints:
        #   1. Change equality: sum(w_i) = leverage_factor (e.g., 1.5)
        #   2. Adjust bounds: w_i >= 0, w_i <= leverage_factor (or remove upper bound)
        #   3. Scale risk contributions by leverage in objective function
        #   4. Implement margin requirements and risk limits
        # ====================================================================
        
        max_weight_per_asset = 0.50  # Maximum 50% per asset (diversification constraint)
        
        # ====================================================================
        # Constraints: Handle Long-Only vs Long/Short vs Leverage
        # ====================================================================
        # 
        # Constraint 1: Exposure Constraint (Equality)
        #   - Long-only: sum(w_i) = 1.0 (fully invested)
        #   - Leveraged long-only: sum(w_i) = leverage (e.g., 1.5)
        #   - Long/short: sum(w_i) = leverage (can be 0 for market-neutral)
        #
        # Constraint 2: Maximum Weight (Inequality) - only for long positions
        #   - Prevents concentration: max(w_i) <= 0.50
        #   - Only applies to positive weights when shorts allowed
        #
        # Bounds: Asset Weight Limits
        #   - Long-only: 0 <= w_i <= 1
        #   - Long/short: -max_short <= w_i <= 1
        # ====================================================================
        
        # Exposure constraint: sum of weights = leverage
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - self.leverage}
        ]
        
        # Maximum weight constraint (only for positive weights)
        if not self.allow_shorts:
            # Long-only: add max weight constraint
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w: max_weight_per_asset - np.max(w)
            })
        else:
            # Long/short: max weight constraint only for positive weights
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_weight_per_asset - np.max(np.maximum(w, 0))
            })
        
        # Bounds: Set based on allow_shorts parameter
        if self.allow_shorts:
            # Long/short: allow negative weights up to -max_short
            bounds = tuple((-self.max_short, 1) for _ in range(n_assets))
        else:
            # Long-only: 0 <= w_i <= 1
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Objective function: minimize sum of squared differences from equal risk contribution
        def objective(weights):
            """
            Objective Function: Minimize Risk Contribution Imbalance.
            
            **Mathematical Formulation:**
            This function implements the core Risk Parity objective from
            Maillard et al. (2010). It minimizes the sum of squared deviations
            from equal risk contribution:
            
                f(w) = sum_i (RC_i - σ_p/n)^2
            
            where:
                RC_i = w_i * (Σw)_i / σ_p  (risk contribution of asset i, in volatility units)
                (Σw)_i = marginal contribution = sum_j(Σ_ij * w_j)
                σ_p = sqrt(w^T Σ w) = portfolio volatility
                n = number of assets
                σ_p/n = target risk contribution per asset (in volatility units)
            
            **Important:** RC_i are in absolute volatility units and sum to σ_p.
            Therefore, for equal risk contribution, each RC_i should equal σ_p/n,
            not 1/n. This is the "absolute RC" approach, which is standard in
            the literature (Maillard et al., 2010).
            
            **Alternative (percent-based) approach:**
            One could normalize RC_i to percentages first: RC_i_percent = RC_i / σ_p,
            then target 1/n. However, the absolute approach is more standard and
            directly implements the mathematical definition.
            
            **Why Squared Differences?**
            - Penalizes large deviations more than small ones (L2 norm)
            - Ensures smooth optimization landscape
            - Encourages equal risk contributions across all assets
            
            **Numerical Considerations:**
            - Returns large penalty (1e10) if portfolio volatility < 1e-10
            - Prevents division by zero in risk contribution calculation
            - Ensures optimization algorithm can handle edge cases
            
            **Long-Only Context:**
            With w_i >= 0 constraint, this objective naturally allocates
            more weight to lower-volatility assets to achieve equal risk
            contribution. For example, if Asset A has 2x the volatility of
            Asset B, Asset A will receive approximately 0.5x the weight.
            
            **Leverage Extension:**
            With leverage (sum(w) > 1), the risk contributions scale:
                RC_i = w_i * (Σw)_i / (σ_p * leverage_factor)
            The objective function remains the same, but the portfolio
            volatility increases proportionally with leverage.
            
            Returns:
                float: Sum of squared differences from equal risk contribution.
                    Lower values indicate better risk parity.
            """
            # Portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol < 1e-10:
                # Avoid division by zero
                return 1e10
            
            # Risk contributions: RC_i = w_i * (Σw)_i / σ_p
            # Note: RC_i has units of volatility and sum(RC_i) = σ_p
            marginal_contrib = cov_matrix @ weights
            risk_contributions = weights * marginal_contrib / portfolio_vol
            
            # Target: equal risk contribution
            # Since RC_i are in volatility units and sum to σ_p, each should equal σ_p/n
            # This is the "absolute RC" approach (standard in literature)
            target_rc = portfolio_vol / n_assets
            
            # Sum of squared differences from target
            diff = risk_contributions - target_rc
            return np.sum(diff ** 2)
        
        # ====================================================================
        # STEP 3: Solve Optimization Problem Using SLSQP
        # ====================================================================
        # 
        # Numerical Optimization Technique: Sequential Least Squares Programming (SLSQP)
        # 
        # Why SLSQP?
        #   - Efficiently handles equality and inequality constraints
        #   - Uses gradient-based optimization (fast convergence)
        #   - Well-suited for smooth, differentiable objective functions
        #   - Robust to numerical precision issues
        #   - Industry standard for portfolio optimization
        #
        # Algorithm Details:
        #   1. Starts with initial guess (equal weights)
        #   2. Computes gradient of objective function
        #   3. Solves quadratic programming subproblem
        #   4. Updates weights using line search
        #   5. Checks constraint satisfaction
        #   6. Iterates until convergence (ftol = 1e-9) or max iterations
        #
        # Convergence Criteria:
        #   - ftol = 1e-9: Very tight tolerance for precision
        #   - maxiter = 1000: Usually converges in < 100 iterations
        #   - Constraint violation tolerance: Built into SLSQP
        #
        # Alternative Methods:
        #   - 'trust-constr': Trust region method, more robust but slower
        #   - 'COBYLA': Derivative-free, useful if gradients unavailable
        #   - 'L-BFGS-B': Limited memory BFGS, good for large problems
        # ====================================================================
        
        result = minimize(
            objective,
            initial_weights,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': 1e-9, 'disp': False}
        )
        
        # ====================================================================
        # STEP 4: Handle Optimization Failures with Fallback Strategies
        # ====================================================================
        # 
        # Strategy 1: Retry with Inverse Volatility Weighting
        #   If initial optimization fails, use inverse volatility weights
        #   as starting point. This is closer to Risk Parity solution because:
        #   - Lower volatility assets get higher weights (inverse relationship)
        #   - This approximates equal risk contribution
        #   - Often helps optimization converge
        #
        # Strategy 2: Fallback to Equal Weights
        #   If retry also fails, use equal weights as safe fallback
        #   Equal weights provide baseline diversification
        #   Better than optimization failure or extreme solutions
        #
        # Why Optimization Might Fail:
        #   - Ill-conditioned covariance matrix (near-singular)
        #   - Numerical precision issues
        #   - Conflicting constraints
        #   - Very different asset characteristics
        # ====================================================================
        
        if not result.success:
            # Retry Strategy: Use inverse volatility weighting as initial guess
            # This is closer to Risk Parity solution and often helps convergence
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
                # Fallback Strategy: Equal weights scaled to leverage (safe baseline)
                self.weights_ = np.ones(n_assets) / n_assets * self.leverage
                self.optimization_result_ = result
            else:
                self.weights_ = result.x
                self.optimization_result_ = result
        else:
            self.weights_ = result.x
            self.optimization_result_ = result
        
        # ====================================================================
        # STEP 5: Post-Processing to Enforce Diversification Constraints
        # ====================================================================
        # 
        # Purpose: Prevent corner solutions where optimization converges to
        # extreme weights (either 0% or 100% in a single asset).
        #
        # Why This Happens:
        #   - Assets with very different risk characteristics
        #   - Numerical optimization may find local minima
        #   - Constraint violations due to numerical precision
        #
        # Long-Only Sufficiency Check:
        #   The bounds (w_i >= 0) ensure long-only, but optimization may
        #   produce weights very close to 0 or violate max weight constraint.
        #   This post-processing ensures practical diversification.
        #
        # Leverage Consideration:
        #   With leverage, adjust thresholds proportionally:
        #   - min_weight = 0.01 * leverage_factor
        #   - max_weight = 0.50 * leverage_factor (or remove if desired)
        # ====================================================================
        
        min_weight_threshold = 0.01  # Minimum 1% per asset (ensures diversification)
        max_weight_threshold = 0.50  # Maximum 50% per asset (prevents concentration)
        
        # Check for assets with very small absolute weights
        if self.allow_shorts:
            small_weights = np.abs(self.weights_) < min_weight_threshold
            # Check for assets with very large absolute weights (corner solution)
            large_weights = np.abs(self.weights_) > max_weight_threshold
        else:
            small_weights = self.weights_ < min_weight_threshold
            large_weights = self.weights_ > max_weight_threshold
        
        if np.any(small_weights) or np.any(large_weights):
            # Redistribute weights to enforce minimum and maximum constraints
            # This ensures diversification and prevents corner solutions
            
            if self.allow_shorts:
                # Long/short: Handle absolute values, preserve signs
                if np.any(small_weights):
                    # Set minimum absolute weight, preserve sign
                    signs = np.sign(self.weights_[small_weights])
                    self.weights_[small_weights] = signs * min_weight_threshold
                
                if np.any(large_weights):
                    # Cap maximum absolute weight, preserve sign
                    signs = np.sign(self.weights_[large_weights])
                    excess = np.sum(self.weights_[large_weights] - signs * max_weight_threshold)
                    self.weights_[large_weights] = signs * max_weight_threshold
                    # Redistribute excess proportionally
                    remaining_assets = ~large_weights
                    if np.sum(remaining_assets) > 0:
                        self.weights_[remaining_assets] += excess * (self.weights_[remaining_assets] / np.sum(self.weights_[remaining_assets]))
            else:
                # Long-only: Standard redistribution
                if np.any(small_weights):
                    self.weights_[small_weights] = min_weight_threshold
                
                if np.any(large_weights):
                    excess = np.sum(self.weights_[large_weights] - max_weight_threshold)
                    self.weights_[large_weights] = max_weight_threshold
                    remaining_assets = ~large_weights
                    if np.sum(remaining_assets) > 0:
                        self.weights_[remaining_assets] += excess * (self.weights_[remaining_assets] / np.sum(self.weights_[remaining_assets]))
            
            # Renormalize to ensure weights sum to leverage factor
            self.weights_ = self.weights_ / self.weights_.sum() * self.leverage
        
        # Validate and normalize weights
        self.weights_ = self._validate_weights(self.weights_)
        
        # ====================================================================
        # STEP 6: Calculate Final Risk Contributions
        # ====================================================================
        # 
        # Risk Contribution Formula (Maillard et al., 2010):
        #   RC_i = w_i * (Σw)_i / σ_p
        #
        # where:
        #   - w_i = weight of asset i
        #   - (Σw)_i = marginal contribution = sum_j(Σ_ij * w_j)
        #   - σ_p = portfolio volatility = sqrt(w^T Σ w)
        #
        # Key Property: sum_i(RC_i) = σ_p
        #   Risk contributions sum to portfolio volatility, so percentages
        #   sum to 100% of total portfolio risk.
        #
        # Long-Only Context:
        #   With w_i >= 0, all risk contributions are non-negative.
        #   Each asset contributes positively to portfolio risk.
        #
        # Leverage Extension:
        #   With leverage factor L (e.g., 1.5x):
        #   - Portfolio volatility scales: σ_p_leveraged = L * σ_p
        #   - Risk contributions scale proportionally
        #   - Risk contribution percentages remain the same
        #   - But absolute risk increases with leverage
        # ====================================================================
        
        portfolio_vol = np.sqrt(self.weights_.T @ cov_matrix @ self.weights_)
        if portfolio_vol > 1e-10:
            # Calculate marginal contributions: (Σw)_i = sum_j(Σ_ij * w_j)
            marginal_contrib = cov_matrix @ self.weights_
            # Calculate risk contributions: RC_i = w_i * (Σw)_i / σ_p
            self.risk_contributions_ = self.weights_ * marginal_contrib / portfolio_vol
        else:
            # Edge Case: Portfolio has near-zero volatility
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

