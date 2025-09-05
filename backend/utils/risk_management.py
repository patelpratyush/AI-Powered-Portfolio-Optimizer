#!/usr/bin/env python3
"""
Advanced Risk Management System
Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from scipy.stats import norm, t
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('portfolio_optimizer.risk_management')

@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    var_95: float
    var_99: float
    var_95_percent: float
    var_99_percent: float
    confidence_level: float
    time_horizon: int
    method: str
    portfolio_value: float

@dataclass
class CVaRResult:
    """Conditional Value at Risk calculation result"""
    cvar_95: float
    cvar_99: float
    cvar_95_percent: float
    cvar_99_percent: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    tail_expectation: float
    method: str

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_results: VaRResult
    cvar_results: CVaRResult
    volatility: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    beta: float
    tracking_error: float
    information_ratio: float
    correlation_with_market: float
    downside_deviation: float
    upside_capture: float
    downside_capture: float

@dataclass
class StressTestResult:
    """Stress testing result"""
    scenario_name: str
    portfolio_return: float
    portfolio_value_change: float
    worst_asset: str
    worst_asset_return: float
    best_asset: str
    best_asset_return: float
    correlations_breakdown: bool  # True if correlations increased significantly

class RiskManager:
    """Advanced risk management calculations"""
    
    def __init__(self, cache_client=None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.risk_management.calculator')
    
    def calculate_var(self, 
                     returns: Union[pd.Series, np.ndarray],
                     confidence_levels: List[float] = [0.95, 0.99],
                     method: str = 'historical',
                     portfolio_value: float = 100000,
                     time_horizon: int = 1) -> VaRResult:
        """
        Calculate Value at Risk using multiple methods
        
        Args:
            returns: Portfolio returns (daily)
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            method: 'historical', 'parametric', 'monte_carlo'
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days
        """
        
        if isinstance(returns, pd.Series):
            returns_array = returns.dropna().values
        else:
            returns_array = np.array(returns)
        
        if len(returns_array) < 30:
            raise ValueError("Need at least 30 return observations for VaR calculation")
        
        # Scale for time horizon if needed
        if time_horizon > 1:
            returns_array = returns_array * np.sqrt(time_horizon)
        
        var_results = {}
        
        if method == 'historical':
            for conf in confidence_levels:
                percentile = (1 - conf) * 100
                var_value = np.percentile(returns_array, percentile)
                var_results[f'var_{int(conf*100)}'] = var_value
                
        elif method == 'parametric':
            # Assume normal distribution
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            for conf in confidence_levels:
                z_score = norm.ppf(1 - conf)
                var_value = mean_return + z_score * std_return
                var_results[f'var_{int(conf*100)}'] = var_value
                
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # Generate random scenarios
            n_simulations = 10000
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            for conf in confidence_levels:
                percentile = (1 - conf) * 100
                var_value = np.percentile(simulated_returns, percentile)
                var_results[f'var_{int(conf*100)}'] = var_value
        
        elif method == 't_distribution':
            # Student's t-distribution (accounts for fat tails)
            # Fit t-distribution to returns
            params = stats.t.fit(returns_array)
            df, loc, scale = params
            
            for conf in confidence_levels:
                var_value = stats.t.ppf(1 - conf, df, loc, scale)
                var_results[f'var_{int(conf*100)}'] = var_value
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Convert to dollar amounts
        var_95_dollar = abs(var_results.get('var_95', 0)) * portfolio_value
        var_99_dollar = abs(var_results.get('var_99', 0)) * portfolio_value
        
        return VaRResult(
            var_95=var_results.get('var_95', 0),
            var_99=var_results.get('var_99', 0),
            var_95_percent=abs(var_results.get('var_95', 0)) * 100,
            var_99_percent=abs(var_results.get('var_99', 0)) * 100,
            confidence_level=confidence_levels[0] if confidence_levels else 0.95,
            time_horizon=time_horizon,
            method=method,
            portfolio_value=portfolio_value
        )
    
    def calculate_cvar(self, 
                      returns: Union[pd.Series, np.ndarray],
                      confidence_levels: List[float] = [0.95, 0.99],
                      method: str = 'historical') -> CVaRResult:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Portfolio returns
            confidence_levels: List of confidence levels
            method: 'historical' or 'parametric'
        """
        
        if isinstance(returns, pd.Series):
            returns_array = returns.dropna().values
        else:
            returns_array = np.array(returns)
        
        cvar_results = {}
        
        if method == 'historical':
            for conf in confidence_levels:
                # Find VaR threshold
                percentile = (1 - conf) * 100
                var_threshold = np.percentile(returns_array, percentile)
                
                # Calculate expected return of tail (CVaR)
                tail_returns = returns_array[returns_array <= var_threshold]
                if len(tail_returns) > 0:
                    cvar_value = np.mean(tail_returns)
                else:
                    cvar_value = var_threshold
                
                cvar_results[f'cvar_{int(conf*100)}'] = cvar_value
                cvar_results[f'es_{int(conf*100)}'] = abs(cvar_value)
        
        elif method == 'parametric':
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            for conf in confidence_levels:
                # For normal distribution
                z_score = norm.ppf(1 - conf)
                cvar_value = mean_return - std_return * norm.pdf(z_score) / (1 - conf)
                
                cvar_results[f'cvar_{int(conf*100)}'] = cvar_value
                cvar_results[f'es_{int(conf*100)}'] = abs(cvar_value)
        
        # Tail expectation (average of worst 5% returns)
        worst_5_percent = np.percentile(returns_array, 5)
        tail_returns = returns_array[returns_array <= worst_5_percent]
        tail_expectation = np.mean(tail_returns) if len(tail_returns) > 0 else worst_5_percent
        
        return CVaRResult(
            cvar_95=cvar_results.get('cvar_95', 0),
            cvar_99=cvar_results.get('cvar_99', 0),
            cvar_95_percent=abs(cvar_results.get('cvar_95', 0)) * 100,
            cvar_99_percent=abs(cvar_results.get('cvar_99', 0)) * 100,
            expected_shortfall_95=cvar_results.get('es_95', 0),
            expected_shortfall_99=cvar_results.get('es_99', 0),
            tail_expectation=abs(tail_expectation) * 100,
            method=method
        )
    
    def calculate_portfolio_risk_metrics(self,
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series = None,
                                       portfolio_value: float = 100000) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio"""
        
        # Basic risk calculations
        var_result = self.calculate_var(portfolio_returns, portfolio_value=portfolio_value)
        cvar_result = self.calculate_cvar(portfolio_returns)
        
        # Additional risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        skewness = stats.skew(portfolio_returns.dropna())
        kurtosis = stats.kurtosis(portfolio_returns.dropna())
        
        # Drawdown calculation
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Downside deviation (semi-deviation)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Benchmark-relative metrics
        beta, tracking_error, information_ratio, correlation = 0, 0, 0, 0
        upside_capture, downside_capture = 1, 1
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
            if len(aligned_data) > 10:
                port_ret = aligned_data.iloc[:, 0]
                bench_ret = aligned_data.iloc[:, 1]
                
                # Beta calculation
                covariance = port_ret.cov(bench_ret)
                benchmark_variance = bench_ret.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                # Tracking error
                active_returns = port_ret - bench_ret
                tracking_error = active_returns.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
                
                # Correlation
                correlation = port_ret.corr(bench_ret)
                
                # Upside/Downside capture
                up_months = bench_ret > 0
                down_months = bench_ret < 0
                
                if up_months.sum() > 0:
                    upside_capture = port_ret[up_months].mean() / bench_ret[up_months].mean()
                if down_months.sum() > 0:
                    downside_capture = port_ret[down_months].mean() / bench_ret[down_months].mean()
        
        return RiskMetrics(
            var_results=var_result,
            cvar_results=cvar_result,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            max_drawdown=max_drawdown,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            correlation_with_market=correlation,
            downside_deviation=downside_deviation,
            upside_capture=upside_capture,
            downside_capture=downside_capture
        )
    
    def stress_test_portfolio(self,
                            portfolio_weights: Dict[str, float],
                            tickers: List[str],
                            scenarios: Dict[str, Dict[str, float]] = None) -> Dict[str, StressTestResult]:
        """
        Perform stress testing on portfolio
        
        Args:
            portfolio_weights: Dictionary of ticker weights
            tickers: List of tickers in portfolio
            scenarios: Custom stress scenarios
        """
        
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        results = {}
        
        for scenario_name, asset_shocks in scenarios.items():
            portfolio_return = 0
            asset_returns = {}
            
            # Calculate portfolio return under stress
            for ticker in tickers:
                weight = portfolio_weights.get(ticker, 0)
                shock = asset_shocks.get(ticker, asset_shocks.get('market', 0))  # Default to market shock
                
                asset_returns[ticker] = shock
                portfolio_return += weight * shock
            
            # Find worst and best performing assets
            worst_asset = min(asset_returns, key=asset_returns.get)
            best_asset = max(asset_returns, key=asset_returns.get)
            
            # Check for correlation breakdown (simplified)
            correlation_breakdown = any(abs(shock) > 0.3 for shock in asset_shocks.values())
            
            results[scenario_name] = StressTestResult(
                scenario_name=scenario_name,
                portfolio_return=portfolio_return,
                portfolio_value_change=portfolio_return,  # Assuming portfolio_value = 1
                worst_asset=worst_asset,
                worst_asset_return=asset_returns[worst_asset],
                best_asset=best_asset,
                best_asset_return=asset_returns[best_asset],
                correlations_breakdown=correlation_breakdown
            )
        
        return results
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Get default stress testing scenarios"""
        return {
            'market_crash': {
                'market': -0.20,  # 20% market drop
                'SPY': -0.20,
                'QQQ': -0.25,  # Tech hit harder
                'IWM': -0.30,  # Small caps hit hardest
                'TSLA': -0.35,
                'AAPL': -0.15,
                'MSFT': -0.18
            },
            'tech_selloff': {
                'market': -0.05,
                'AAPL': -0.25,
                'MSFT': -0.22,
                'GOOGL': -0.28,
                'META': -0.30,
                'TSLA': -0.40,
                'NVDA': -0.35,
                'QQQ': -0.20
            },
            'inflation_shock': {
                'market': -0.10,
                'BND': -0.15,  # Bonds hit by rising rates
                'TLT': -0.20,  # Long-term bonds hit harder
                'REIT': -0.15,
                'utilities': -0.12,
                'financials': 0.05,  # Banks benefit from higher rates
                'energy': 0.10  # Energy benefits from inflation
            },
            'recession_scenario': {
                'market': -0.30,
                'consumer_discretionary': -0.40,
                'financials': -0.35,
                'industrials': -0.32,
                'consumer_staples': -0.15,  # Defensive
                'healthcare': -0.10,  # Defensive
                'utilities': -0.08   # Most defensive
            },
            'geopolitical_crisis': {
                'market': -0.15,
                'VIX': 1.00,  # Volatility spike
                'oil': 0.30,  # Oil price spike
                'gold': 0.15,  # Flight to safety
                'emerging_markets': -0.25,
                'european_stocks': -0.20,
                'defense_stocks': 0.10
            }
        }
    
    def calculate_marginal_var(self,
                             portfolio_returns: pd.Series,
                             individual_returns: Dict[str, pd.Series],
                             weights: Dict[str, float],
                             confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Marginal VaR - the contribution of each asset to portfolio VaR
        """
        
        portfolio_var = self.calculate_var(
            portfolio_returns, 
            confidence_levels=[confidence_level]
        ).var_95
        
        marginal_vars = {}
        epsilon = 0.01  # Small change for numerical differentiation
        
        for ticker, weight in weights.items():
            if ticker not in individual_returns:
                marginal_vars[ticker] = 0
                continue
            
            # Create slightly modified portfolio
            modified_weights = weights.copy()
            modified_weights[ticker] = weight + epsilon
            
            # Normalize weights
            total_weight = sum(modified_weights.values())
            modified_weights = {k: v/total_weight for k, v in modified_weights.items()}
            
            # Calculate modified portfolio returns
            modified_returns = pd.Series(0, index=portfolio_returns.index)
            for t, w in modified_weights.items():
                if t in individual_returns:
                    aligned_returns = individual_returns[t].reindex(portfolio_returns.index).fillna(0)
                    modified_returns += w * aligned_returns
            
            # Calculate VaR of modified portfolio
            modified_var = self.calculate_var(
                modified_returns,
                confidence_levels=[confidence_level]
            ).var_95
            
            # Marginal VaR = (Modified VaR - Original VaR) / epsilon
            marginal_vars[ticker] = (modified_var - portfolio_var) / epsilon
        
        return marginal_vars
    
    def calculate_component_var(self,
                              marginal_vars: Dict[str, float],
                              weights: Dict[str, float],
                              portfolio_var: float) -> Dict[str, float]:
        """
        Calculate Component VaR - the absolute contribution of each asset to portfolio VaR
        """
        
        component_vars = {}
        
        for ticker in weights.keys():
            marginal_var = marginal_vars.get(ticker, 0)
            weight = weights.get(ticker, 0)
            
            # Component VaR = Weight * Marginal VaR
            component_vars[ticker] = weight * marginal_var
        
        # Verify components sum to portfolio VaR (approximately)
        total_component = sum(component_vars.values())
        if abs(total_component - portfolio_var) > 0.01:
            self.logger.warning(f"Component VaR sum ({total_component:.4f}) != Portfolio VaR ({portfolio_var:.4f})")
        
        return component_vars
    
    def risk_budgeting_optimization(self,
                                  expected_returns: Dict[str, float],
                                  risk_budget: Dict[str, float],
                                  covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio weights based on risk budgeting
        Each asset should contribute a target percentage to total portfolio risk
        """
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            tickers = list(expected_returns.keys())
            
            def risk_contribution_objective(weights):
                """Objective function for risk budgeting"""
                weights = np.array(weights)
                
                # Portfolio variance
                portfolio_var = np.dot(weights.T, np.dot(covariance_matrix.values, weights))
                portfolio_vol = np.sqrt(portfolio_var)
                
                # Marginal risk contributions
                marginal_contrib = np.dot(covariance_matrix.values, weights) / portfolio_vol
                
                # Risk contributions
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                # Target risk contributions
                target_contrib = np.array([risk_budget.get(ticker, 1/n_assets) for ticker in tickers])
                
                # Minimize difference between actual and target risk contributions
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds (positive weights only)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_contribution_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = dict(zip(tickers, result.x))
                return optimal_weights
            else:
                self.logger.warning("Risk budgeting optimization failed, using equal weights")
                return {ticker: 1/n_assets for ticker in tickers}
                
        except Exception as e:
            self.logger.error(f"Risk budgeting optimization error: {e}")
            return {ticker: 1/len(expected_returns) for ticker in expected_returns.keys()}


def calculate_portfolio_var_cvar(portfolio_returns: pd.Series,
                               portfolio_value: float = 100000,
                               confidence_levels: List[float] = [0.95, 0.99],
                               methods: List[str] = ['historical', 'parametric']) -> Dict[str, Any]:
    """
    Convenience function to calculate VaR and CVaR for a portfolio
    
    Returns:
        Dictionary with VaR and CVaR results for all methods and confidence levels
    """
    
    risk_manager = RiskManager()
    results = {}
    
    for method in methods:
        try:
            var_result = risk_manager.calculate_var(
                portfolio_returns, 
                confidence_levels=confidence_levels,
                method=method,
                portfolio_value=portfolio_value
            )
            
            cvar_result = risk_manager.calculate_cvar(
                portfolio_returns,
                confidence_levels=confidence_levels,
                method=method
            )
            
            results[method] = {
                'var': asdict(var_result),
                'cvar': asdict(cvar_result)
            }
            
        except Exception as e:
            logger.error(f"Error calculating {method} VaR/CVaR: {e}")
            results[method] = {'error': str(e)}
    
    return results


def stress_test_quick(tickers: List[str], 
                     weights: List[float] = None,
                     scenarios: List[str] = None) -> Dict[str, Any]:
    """
    Quick stress test for a list of tickers
    
    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights (equal weight if None)
        scenarios: List of scenario names (default scenarios if None)
    
    Returns:
        Stress test results
    """
    
    if weights is None:
        weights = [1/len(tickers)] * len(tickers)
    
    portfolio_weights = dict(zip(tickers, weights))
    
    risk_manager = RiskManager()
    
    if scenarios:
        # Filter default scenarios
        all_scenarios = risk_manager._get_default_stress_scenarios()
        filtered_scenarios = {k: v for k, v in all_scenarios.items() if k in scenarios}
    else:
        filtered_scenarios = None
    
    return risk_manager.stress_test_portfolio(
        portfolio_weights, 
        tickers, 
        filtered_scenarios
    )