#!/usr/bin/env python3
"""
Advanced Portfolio Optimizer with Enhanced Features
- Black-Litterman Optimization
- Risk Parity and Hierarchical Risk Parity
- Multi-Objective Optimization
- Machine Learning Enhanced Strategies
- Monte Carlo Simulation
- Advanced Risk Metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioOptimizer:
    def __init__(self, returns, risk_free_rate=0.02, market_cap_weights=None):
        self.returns = returns
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.tickers = returns.columns.tolist()
        
        # Fix the market cap weights assignment
        if market_cap_weights is None:
            self.market_cap_weights = np.array([1/self.n_assets] * self.n_assets)
        else:
            self.market_cap_weights = market_cap_weights
        
    def portfolio_performance(self, weights):
        """Enhanced portfolio performance metrics"""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Additional metrics
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns.fillna(0).dot(weights) ** 2)) * np.sqrt(252)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': downside_deviation
        }
    
    def black_litterman_optimization(self, views_matrix=None, views_uncertainty=None, tau=0.05):
        """
        Black-Litterman optimization incorporating investor views
        
        Parameters:
        - views_matrix: Matrix of investor views on expected returns
        - views_uncertainty: Uncertainty in the views
        - tau: Scalar representing uncertainty in the prior
        """
        try:
            # Market equilibrium returns (reverse optimization)
            pi = self.risk_free_rate + np.dot(self.cov_matrix, self.market_cap_weights)
            
            if views_matrix is None or views_uncertainty is None:
                # Use market equilibrium if no views provided
                mu_bl = pi
            else:
                # Black-Litterman formula
                tau_sigma = tau * self.cov_matrix
                omega = views_uncertainty
                
                # Calculate Black-Litterman expected returns
                term1 = np.linalg.inv(tau_sigma)
                term2 = np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_matrix))
                term3 = np.dot(np.linalg.inv(tau_sigma), pi)
                term4 = np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_matrix.mean(axis=0)))
                
                mu_bl = np.dot(np.linalg.inv(term1 + term2), term3 + term4)
            
            # Optimize using Black-Litterman returns
            def objective(weights):
                portfolio_return = np.sum(mu_bl * weights)
                portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
                return -portfolio_return + 0.5 * portfolio_var  # Mean-variance utility
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = self.market_cap_weights
            
            result = minimize(objective, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x, mu_bl
            else:
                raise Exception(f"Black-Litterman optimization failed: {result.message}")
                
        except Exception as e:
            raise Exception(f"Black-Litterman optimization error: {str(e)}")
    
    def hierarchical_risk_parity(self, linkage_method='ward'):
        """
        Hierarchical Risk Parity (HRP) using machine learning clustering
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
            from scipy.spatial.distance import squareform
            
            # Calculate correlation matrix and distance matrix
            corr_matrix = self.returns.corr()
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Hierarchical clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method=linkage_method)
            
            # Get clusters
            clusters = cut_tree(linkage_matrix, n_clusters=min(5, self.n_assets//2))
            
            # Allocate weights hierarchically
            weights = np.zeros(self.n_assets)
            
            def _get_cluster_weights(cluster_indices):
                if len(cluster_indices) == 1:
                    return np.array([1.0])
                
                # Calculate cluster variance
                cluster_cov = self.cov_matrix[np.ix_(cluster_indices, cluster_indices)]
                cluster_ivp = 1.0 / np.diag(cluster_cov)
                cluster_weights = cluster_ivp / cluster_ivp.sum()
                
                return cluster_weights
            
            # Assign weights within each cluster
            unique_clusters = np.unique(clusters.flatten())
            for cluster_id in unique_clusters:
                cluster_mask = (clusters.flatten() == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]
                cluster_weights = _get_cluster_weights(cluster_indices)
                weights[cluster_indices] = cluster_weights / len(unique_clusters)
            
            return weights
            
        except Exception as e:
            # Fall back to equal weight if HRP fails
            print(f"HRP failed: {str(e)}, falling back to equal weight")
            return np.array([1/self.n_assets] * self.n_assets)
    
    def multi_objective_optimization(self, objectives=['sharpe', 'sortino', 'calmar'], weights_obj=None):
        """
        Multi-objective optimization balancing different objectives
        
        Objectives:
        - sharpe: Sharpe ratio
        - sortino: Sortino ratio 
        - calmar: Calmar ratio (return/max drawdown)
        - min_vol: Minimum volatility
        - max_div: Maximum diversification
        """
        try:
            weights_obj = weights_obj or [1/len(objectives)] * len(objectives)
            
            def multi_objective(weights):
                perf = self.portfolio_performance(weights)
                
                # Calculate individual objectives
                obj_values = []
                
                if 'sharpe' in objectives:
                    obj_values.append(-perf['sharpe_ratio'])  # Negative because we minimize
                    
                if 'sortino' in objectives:
                    obj_values.append(-perf['sortino_ratio'])
                    
                if 'min_vol' in objectives:
                    obj_values.append(perf['volatility'])
                    
                if 'calmar' in objectives:
                    # Approximate Calmar ratio using volatility as proxy for max drawdown
                    calmar_approx = perf['return'] / (perf['volatility'] * 2)  # Rough approximation
                    obj_values.append(-calmar_approx)
                    
                if 'max_div' in objectives:
                    # Diversification ratio: weighted average vol / portfolio vol
                    individual_vols = np.sqrt(np.diag(self.cov_matrix))
                    diversification_ratio = np.sum(weights * individual_vols) / perf['volatility']
                    obj_values.append(-diversification_ratio)
                
                # Weighted combination of objectives
                return np.sum(np.array(obj_values) * np.array(weights_obj))
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(multi_objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            
            if result.success:
                return result.x
            else:
                raise Exception(f"Multi-objective optimization failed: {result.message}")
                
        except Exception as e:
            raise Exception(f"Multi-objective optimization error: {str(e)}")
    
    def risk_budgeting_optimization(self, risk_budgets=None):
        """
        Advanced Risk Parity with custom risk budgets
        
        Parameters:
        - risk_budgets: Target risk contributions for each asset (must sum to 1)
        """
        try:
            if risk_budgets is None:
                risk_budgets = np.array([1/self.n_assets] * self.n_assets)
            
            def risk_budget_objective(weights):
                # Portfolio volatility
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                
                # Marginal risk contributions
                marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
                
                # Risk contributions
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                # Objective: minimize squared deviations from target risk budgets
                return np.sum((risk_contrib - risk_budgets) ** 2)
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0.001, 1) for _ in range(self.n_assets))
            initial_guess = risk_budgets
            
            result = minimize(risk_budget_objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            
            if result.success:
                return result.x
            else:
                raise Exception(f"Risk budgeting optimization failed: {result.message}")
                
        except Exception as e:
            # Fall back to equal risk budget
            return self.risk_parity_portfolio()
    
    def constrained_optimization(self, constraints_dict):
        """
        Portfolio optimization with advanced constraints
        
        constraints_dict can include:
        - min_weights: minimum allocation per asset
        - max_weights: maximum allocation per asset  
        - sector_limits: sector allocation limits
        - turnover_limit: maximum turnover from current portfolio
        - target_return: target portfolio return
        """
        try:
            # Use CVXPY for more complex constraints
            w = cp.Variable(self.n_assets)
            
            # Objective: maximize Sharpe ratio (or minimize negative Sharpe)
            portfolio_return = self.mean_returns.values @ w
            portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
            
            # Constraints
            constraints = [cp.sum(w) == 1]  # Weights sum to 1
            
            # Basic bounds
            if 'min_weights' in constraints_dict:
                min_w = constraints_dict['min_weights']
                if isinstance(min_w, (int, float)):
                    constraints.append(w >= min_w)
                else:
                    constraints.append(w >= np.array(min_w))
                    
            if 'max_weights' in constraints_dict:
                max_w = constraints_dict['max_weights']
                if isinstance(max_w, (int, float)):
                    constraints.append(w <= max_w)
                else:
                    constraints.append(w <= np.array(max_w))
            
            # Target return constraint
            if 'target_return' in constraints_dict:
                target = constraints_dict['target_return']
                constraints.append(portfolio_return >= target)
            
            # Sector constraints (if provided)
            if 'sector_weights' in constraints_dict and 'sector_limits' in constraints_dict:
                sector_weights = constraints_dict['sector_weights']  # Which sector each asset belongs to
                sector_limits = constraints_dict['sector_limits']    # Min/max for each sector
                
                for sector, limits in sector_limits.items():
                    sector_mask = np.array([sector_weights.get(ticker, '') == sector 
                                          for ticker in self.tickers])
                    sector_allocation = cp.sum(w[sector_mask])
                    
                    if 'min' in limits:
                        constraints.append(sector_allocation >= limits['min'])
                    if 'max' in limits:
                        constraints.append(sector_allocation <= limits['max'])
            
            # Formulate optimization problem
            if 'target_return' in constraints_dict:
                # Minimize risk for target return
                objective = cp.Minimize(portfolio_variance)
            else:
                # Maximize Sharpe ratio approximation
                objective = cp.Maximize(portfolio_return - 0.5 * portfolio_variance)
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)
            
            if w.value is not None:
                return w.value
            else:
                raise Exception("Constrained optimization failed to converge")
                
        except Exception as e:
            raise Exception(f"Constrained optimization error: {str(e)}")
    
    def risk_parity_portfolio(self):
        """Enhanced Risk Parity implementation"""
        try:
            def risk_budget_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                if portfolio_vol == 0:
                    return 1e10
                marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                return np.sum((contrib - contrib.mean()) ** 2)
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0.001, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(risk_budget_objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            
            if result.success:
                return result.x
            else:
                return np.array([1/self.n_assets] * self.n_assets)
                
        except Exception:
            return np.array([1/self.n_assets] * self.n_assets)
    
    def monte_carlo_simulation(self, weights, num_simulations=10000, time_horizon=252):
        """
        Monte Carlo simulation for portfolio returns
        
        Returns:
        - Distribution of portfolio values
        - VaR and CVaR estimates
        - Probability metrics
        """
        try:
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Generate random returns
            daily_returns = np.random.normal(
                portfolio_return / 252, 
                portfolio_vol / np.sqrt(252), 
                (num_simulations, time_horizon)
            )
            
            # Calculate cumulative portfolio values
            portfolio_values = np.cumprod(1 + daily_returns, axis=1)
            final_values = portfolio_values[:, -1]
            
            # Calculate risk metrics
            var_95 = np.percentile(final_values, 5)
            var_99 = np.percentile(final_values, 1)
            cvar_95 = np.mean(final_values[final_values <= var_95])
            cvar_99 = np.mean(final_values[final_values <= var_99])
            
            # Maximum drawdown simulation
            running_max = np.maximum.accumulate(portfolio_values, axis=1)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdowns = np.min(drawdowns, axis=1)
            
            return {
                'final_values': final_values,
                'portfolio_values': portfolio_values,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdowns': max_drawdowns,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'prob_loss': np.sum(final_values < 1) / num_simulations,
                'prob_outperform_market': np.sum(final_values > 1.07) / num_simulations  # Assuming 7% market return
            }
            
        except Exception as e:
            raise Exception(f"Monte Carlo simulation error: {str(e)}")
    
    def calculate_risk_decomposition(self, weights):
        """
        Calculate detailed risk decomposition
        """
        try:
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # Component risk contributions
            component_contrib = weights * marginal_contrib
            
            # Percentage risk contributions
            pct_contrib = component_contrib / portfolio_vol
            
            return {
                'total_risk': portfolio_vol,
                'marginal_contributions': marginal_contrib,
                'component_contributions': component_contrib,
                'percentage_contributions': pct_contrib,
                'risk_concentration': np.sum(pct_contrib ** 2)  # Herfindahl index for risk concentration
            }
            
        except Exception as e:
            raise Exception(f"Risk decomposition error: {str(e)}")


def get_sector_data(tickers):
    """
    Get sector information for tickers
    """
    sector_map = {}
    try:
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                sector_map[ticker] = sector
            except:
                sector_map[ticker] = 'Unknown'
        return sector_map
    except Exception:
        return {ticker: 'Unknown' for ticker in tickers}


def get_market_cap_weights(tickers):
    """
    Get market cap weights for tickers (for Black-Litterman)
    """
    try:
        market_caps = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 1e9)  # Default 1B if not available
                market_caps[ticker] = market_cap
            except:
                market_caps[ticker] = 1e9
        
        total_cap = sum(market_caps.values())
        weights = np.array([market_caps[ticker] / total_cap for ticker in tickers])
        return weights
        
    except Exception:
        return np.array([1/len(tickers)] * len(tickers))