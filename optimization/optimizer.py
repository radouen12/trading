"""
Portfolio Optimizer - Phase 3
Advanced portfolio optimization using modern portfolio theory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from datetime import datetime, timedelta
import yfinance as yf
import logging

class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        
        # Optimization constraints with validation
        self.min_weight = 0.01  # 1% minimum allocation
        self.max_weight = 0.25  # 25% maximum allocation
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Validate constraints
        if not self._validate_constraints():
            raise ValueError("Invalid optimization constraints")
    
    def _validate_constraints(self):
        """Validate optimization constraints"""
        try:
            # Check weight bounds
            if not (0 <= self.min_weight <= 1):
                self.logger.error(f"Invalid min_weight: {self.min_weight}")
                return False
            
            if not (0 <= self.max_weight <= 1):
                self.logger.error(f"Invalid max_weight: {self.max_weight}")
                return False
            
            if self.min_weight >= self.max_weight:
                self.logger.error(f"min_weight ({self.min_weight}) >= max_weight ({self.max_weight})")
                return False
            
            # Check if constraints can sum to 1.0
            min_possible_sum = self.min_weight * 2  # At least 2 assets required
            max_possible_sum = self.max_weight * 10  # Reasonable upper bound
            
            if min_possible_sum > 1.0:
                self.logger.error(f"Minimum weights cannot sum to 1.0: {min_possible_sum}")
                return False
            
            if max_possible_sum < 1.0:
                self.logger.error(f"Maximum weights cannot reach 1.0: {max_possible_sum}")
                return False
            
            # Validate risk-free rate
            if not (-0.1 <= self.risk_free_rate <= 0.5):  # -10% to 50% annual
                self.logger.error(f"Invalid risk_free_rate: {self.risk_free_rate}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating constraints: {e}")
            return False
    
    def _validate_weights(self, weights, symbols):
        """Validate portfolio weights"""
        try:
            if not isinstance(weights, dict):
                self.logger.error(f"Weights must be a dictionary, got {type(weights)}")
                return False
            
            # Check if all symbols are represented
            if set(weights.keys()) != set(symbols):
                missing = set(symbols) - set(weights.keys())
                extra = set(weights.keys()) - set(symbols)
                self.logger.error(f"Weight mismatch - Missing: {missing}, Extra: {extra}")
                return False
            
            # Validate individual weights
            for symbol, weight in weights.items():
                if not isinstance(weight, (int, float)):
                    self.logger.error(f"Invalid weight type for {symbol}: {type(weight)}")
                    return False
                
                if not (0 <= weight <= 1):
                    self.logger.error(f"Weight out of bounds for {symbol}: {weight}")
                    return False
                
                if weight < self.min_weight - 1e-6:  # Allow small numerical errors
                    self.logger.error(f"Weight below minimum for {symbol}: {weight} < {self.min_weight}")
                    return False
                
                if weight > self.max_weight + 1e-6:  # Allow small numerical errors
                    self.logger.error(f"Weight above maximum for {symbol}: {weight} > {self.max_weight}")
                    return False
            
            # Check if weights sum to approximately 1.0
            total_weight = sum(weights.values())
            if not (0.99 <= total_weight <= 1.01):  # Allow 1% tolerance
                self.logger.error(f"Weights sum to {total_weight}, should be ~1.0")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating weights: {e}")
            return False
    
    def optimize_portfolio(self, symbols: List[str], target_return: Optional[float] = None, 
                          method: str = 'sharpe') -> Dict:
        """
        Optimize portfolio allocation for given symbols
        
        Args:
            symbols: List of stock symbols
            target_return: Target annual return (optional)
            method: Optimization method ('sharpe', 'min_vol', 'max_return', 'efficient_frontier')
            
        Returns:
            Dict with optimized weights and portfolio metrics
        """
        
        try:
            # Get historical data
            price_data = self._get_price_data(symbols)
            if price_data.empty:
                return {"error": "No price data available"}
            
            if len(symbols) < 2:
                return {"error": "Portfolio optimization requires at least 2 symbols"}
            
            # Calculate returns and covariance
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 30:  # Need minimum data points
                return {"error": "Insufficient historical data for optimization (need at least 30 days)"}
            
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Check for singular covariance matrix
            try:
                np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                return {"error": "Covariance matrix is singular (assets are perfectly correlated)"}
            
            # Optimize based on method
            if method == 'sharpe':
                result = self._optimize_sharpe_ratio(mean_returns, cov_matrix)
            elif method == 'min_vol':
                result = self._optimize_min_volatility(mean_returns, cov_matrix)
            elif method == 'max_return':
                result = self._optimize_max_return(mean_returns, cov_matrix)
            elif method == 'target_return' and target_return:
                result = self._optimize_target_return(mean_returns, cov_matrix, target_return)
            elif method == 'efficient_frontier':
                result = self._calculate_efficient_frontier(mean_returns, cov_matrix)
            else:
                result = self._optimize_sharpe_ratio(mean_returns, cov_matrix)
            
            # Add portfolio metrics
            if 'weights' in result:
                # Validate weights before calculating metrics
                if not self._validate_weights(result['weights'], symbols):
                    return {"error": "Invalid portfolio weights generated"}
                
                metrics = self._calculate_portfolio_metrics(result['weights'], mean_returns, cov_matrix)
                result.update(metrics)
            
            # Add allocation recommendations
            result['allocation_recommendations'] = self._generate_allocation_recommendations(
                result.get('weights', {}), symbols
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {"error": str(e)}
    
    def _get_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get historical price data for symbols"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 50)  # Extra buffer
        
        try:
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if len(symbols) == 1:
                # Single symbol case
                return pd.DataFrame({symbols[0]: data['Adj Close']})
            else:
                # Multiple symbols case
                return data['Adj Close'].dropna()
                
        except Exception as e:
            self.logger.error(f"Failed to get price data: {e}")
            return pd.DataFrame()

# Placeholder methods for optimization algorithms - these would need full implementation
    def _optimize_sharpe_ratio(self, mean_returns, cov_matrix):
        """Placeholder for Sharpe ratio optimization"""
        return {"error": "Sharpe ratio optimization not fully implemented"}
    
    def _optimize_min_volatility(self, mean_returns, cov_matrix):
        """Placeholder for minimum volatility optimization"""
        return {"error": "Min volatility optimization not fully implemented"}
    
    def _optimize_max_return(self, mean_returns, cov_matrix):
        """Placeholder for maximum return optimization"""
        return {"error": "Max return optimization not fully implemented"}
    
    def _optimize_target_return(self, mean_returns, cov_matrix, target_return):
        """Placeholder for target return optimization"""
        return {"error": "Target return optimization not fully implemented"}
    
    def _calculate_efficient_frontier(self, mean_returns, cov_matrix):
        """Placeholder for efficient frontier calculation"""
        return {"error": "Efficient frontier calculation not fully implemented"}
    
    def _calculate_portfolio_metrics(self, weights, mean_returns, cov_matrix):
        """Placeholder for portfolio metrics calculation"""
        return {"metrics": "Portfolio metrics calculation not fully implemented"}
    
    def _generate_allocation_recommendations(self, weights, symbols):
        """Placeholder for allocation recommendations"""
        return {"recommendations": "Allocation recommendations not fully implemented"}
