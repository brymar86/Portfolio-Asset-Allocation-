"""
Portfolio Optimization Module.

This module provides implementations of advanced portfolio optimization methods
based on the work of Marcos Lopez de Prado, including:

- Hierarchical Risk Parity (HRP)
- Risk Parity (Equal Risk Contribution)
- Nested Clustered Optimization (NCO)

All optimizers inherit from BasePortfolioOptimizer and provide a consistent
interface: fit() and predict() methods.
"""

from .base_optimizer import BasePortfolioOptimizer
from .hierarchical_risk_parity import HierarchicalRiskParity
from .risk_parity import RiskParity
from .nested_clustered_optimization import NestedClusteredOptimization

__all__ = [
    'BasePortfolioOptimizer',
    'HierarchicalRiskParity',
    'RiskParity',
    'NestedClusteredOptimization',
]

__version__ = '0.1.0'


