#!/usr/bin/env python3
"""
Portfolio optimization tests
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import json
from routes.optimize import optimize_bp
from routes.advanced_optimize import advanced_optimize_bp
from advanced_optimizer import AdvancedPortfolioOptimizer


class TestBasicOptimization:
    """Test basic portfolio optimization endpoints"""
    
    def test_optimize_endpoint_success(self, client):
        """Test successful portfolio optimization"""
        with patch('routes.optimize.yf.download') as mock_download:
            # Mock stock data for multiple tickers
            mock_data = pd.DataFrame({
                'Close': np.random.randn(252) * 0.02 + 1  # 1 year of returns
            }, index=pd.date_range('2023-01-01', periods=252))
            mock_download.return_value = mock_data
            
            payload = {
                'tickers': ['AAPL', 'GOOGL', 'MSFT'],
                'strategy': 'sharpe',
                'start': '2023-01-01',
                'end': '2024-01-01',
                'investment_amount': 10000
            }
            
            response = client.post('/api/optimize', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = response.get_json()
            
            assert 'weights' in data
            assert 'expected_return' in data
            assert 'volatility' in data
            assert 'sharpe_ratio' in data
            assert len(data['weights']) == 3
            assert abs(sum(data['weights'].values()) - 1.0) < 0.01  # Weights sum to 1
            
    def test_optimize_endpoint_invalid_tickers(self, client):
        """Test optimization with insufficient tickers"""
        payload = {
            'tickers': ['AAPL'],  # Need at least 2 tickers
            'strategy': 'sharpe',
            'start': '2023-01-01',
            'end': '2024-01-01'
        }
        
        response = client.post('/api/optimize',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        
    def test_optimize_endpoint_invalid_strategy(self, client):
        """Test optimization with invalid strategy"""
        payload = {
            'tickers': ['AAPL', 'GOOGL'],
            'strategy': 'invalid_strategy',
            'start': '2023-01-01',
            'end': '2024-01-01'
        }
        
        response = client.post('/api/optimize',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        
    def test_optimize_endpoint_date_validation(self, client):
        """Test optimization with invalid date range"""
        payload = {
            'tickers': ['AAPL', 'GOOGL'],
            'strategy': 'sharpe',
            'start': '2024-01-01',  # Start after end
            'end': '2023-01-01'
        }
        
        response = client.post('/api/optimize',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data


class TestAdvancedOptimization:
    """Test advanced portfolio optimization"""
    
    def test_advanced_optimize_black_litterman(self, client):
        """Test Black-Litterman optimization"""
        with patch('routes.advanced_optimize.yf.download') as mock_download:
            # Mock stock data
            mock_data = pd.DataFrame({
                'Close': np.random.randn(252) * 0.02 + 100
            }, index=pd.date_range('2023-01-01', periods=252))
            mock_download.return_value = mock_data
            
            payload = {
                'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                'strategy': 'black_litterman',
                'start': '2023-01-01',
                'end': '2024-01-01',
                'constraints': {
                    'min_weights': 0.05,
                    'max_weights': 0.40
                },
                'optimization_params': {
                    'risk_free_rate': 0.02,
                    'tau': 0.05
                },
                'include_monte_carlo': True,
                'include_efficient_frontier': True
            }
            
            response = client.post('/api/advanced-optimize',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = response.get_json()
            
            assert 'weights' in data
            assert 'expected_return' in data
            assert 'volatility' in data
            assert 'monte_carlo_results' in data
            assert 'efficient_frontier' in data
            
            # Check weight constraints
            weights = data['weights']
            for weight in weights.values():
                assert weight >= 0.05 - 0.01  # Allow small tolerance
                assert weight <= 0.40 + 0.01
                
    def test_advanced_optimize_multi_objective(self, client):
        """Test multi-objective optimization"""
        with patch('routes.advanced_optimize.yf.download') as mock_download:
            mock_data = pd.DataFrame({
                'Close': np.random.randn(252) * 0.02 + 100
            }, index=pd.date_range('2023-01-01', periods=252))
            mock_download.return_value = mock_data
            
            payload = {
                'tickers': ['AAPL', 'GOOGL', 'MSFT'],
                'strategy': 'multi_objective',
                'start': '2023-01-01',
                'end': '2024-01-01',
                'optimization_params': {
                    'objectives': ['sharpe', 'sortino'],
                    'objective_weights': [0.6, 0.4]
                }
            }
            
            response = client.post('/api/advanced-optimize',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = response.get_json()
            
            assert 'weights' in data
            assert 'multi_objective_score' in data


class TestAdvancedPortfolioOptimizer:
    """Test AdvancedPortfolioOptimizer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = AdvancedPortfolioOptimizer()
        
        # Create sample returns data
        np.random.seed(42)
        self.sample_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252),
            'AMZN': np.random.normal(0.0009, 0.03, 252)
        }, index=pd.date_range('2023-01-01', periods=252))
        
    def test_black_litterman_optimization(self):
        """Test Black-Litterman model implementation"""
        # Define investor views
        views_matrix = np.array([
            [1, -1, 0, 0],  # AAPL outperforms GOOGL
            [0, 0, 1, -1]   # MSFT outperforms AMZN
        ])
        views_returns = np.array([0.02, 0.01])  # Expected outperformance
        
        result = self.optimizer.black_litterman_optimization(
            returns=self.sample_returns,
            views_matrix=views_matrix,
            views_returns=views_returns,
            risk_free_rate=0.02,
            tau=0.05
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert abs(sum(result['weights']) - 1.0) < 0.01
        
    def test_risk_parity_optimization(self):
        """Test risk parity optimization"""
        result = self.optimizer.risk_parity_optimization(self.sample_returns)
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert abs(sum(result['weights']) - 1.0) < 0.01
        
        # Risk parity should have relatively equal risk contributions
        weights = np.array(list(result['weights']))
        cov_matrix = self.sample_returns.cov().values
        risk_contributions = (weights * (cov_matrix @ weights)) / np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Risk contributions should be similar (within reasonable tolerance)
        assert np.std(risk_contributions) < 0.1
        
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization"""
        objectives = ['sharpe', 'sortino', 'max_diversification']
        weights = [0.5, 0.3, 0.2]
        
        result = self.optimizer.multi_objective_optimization(
            returns=self.sample_returns,
            objectives=objectives,
            objective_weights=weights
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'multi_objective_score' in result
        assert abs(sum(result['weights']) - 1.0) < 0.01
        
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo portfolio simulation"""
        weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
        
        result = self.optimizer.monte_carlo_simulation(
            returns=self.sample_returns,
            weights=weights,
            num_simulations=1000,
            time_horizon=252
        )
        
        assert 'simulated_returns' in result
        assert 'percentiles' in result
        assert 'var_95' in result
        assert 'cvar_95' in result
        assert len(result['simulated_returns']) == 1000
        
        # Check VaR and CVaR are reasonable
        assert result['var_95'] < 0  # VaR should be negative (loss)
        assert result['cvar_95'] < result['var_95']  # CVaR should be worse than VaR
        
    def test_efficient_frontier_generation(self):
        """Test efficient frontier calculation"""
        result = self.optimizer.generate_efficient_frontier(
            returns=self.sample_returns,
            num_portfolios=50
        )
        
        assert 'frontier_returns' in result
        assert 'frontier_volatilities' in result
        assert 'frontier_weights' in result
        assert len(result['frontier_returns']) == 50
        assert len(result['frontier_volatilities']) == 50
        
        # Check that frontier is properly ordered (increasing volatility)
        assert all(result['frontier_volatilities'][i] <= result['frontier_volatilities'][i+1] 
                  for i in range(len(result['frontier_volatilities'])-1))
        
    def test_constraints_validation(self):
        """Test portfolio constraints validation"""
        constraints = {
            'min_weights': 0.1,
            'max_weights': 0.4,
            'sector_limits': {
                'Technology': {'min': 0.2, 'max': 0.6}
            }
        }
        
        # Test valid weights
        valid_weights = [0.25, 0.25, 0.25, 0.25]
        assert self.optimizer.validate_constraints(valid_weights, constraints)
        
        # Test invalid weights (below minimum)
        invalid_weights = [0.05, 0.25, 0.35, 0.35]
        assert not self.optimizer.validate_constraints(invalid_weights, constraints)
        
    def test_performance_metrics_calculation(self):
        """Test portfolio performance metrics"""
        weights = [0.25, 0.25, 0.25, 0.25]
        
        metrics = self.optimizer.calculate_performance_metrics(
            returns=self.sample_returns,
            weights=weights,
            risk_free_rate=0.02
        )
        
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        
        # Check metrics are reasonable
        assert isinstance(metrics['expected_return'], (int, float))
        assert metrics['volatility'] > 0
        assert isinstance(metrics['sharpe_ratio'], (int, float))


class TestOptimizationEdgeCases:
    """Test edge cases and error handling"""
    
    def test_insufficient_data(self):
        """Test optimization with insufficient historical data"""
        optimizer = AdvancedPortfolioOptimizer()
        
        # Very limited data
        limited_returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.005, -0.01, 0.015]
        })
        
        with pytest.raises(ValueError):
            optimizer.black_litterman_optimization(limited_returns)
            
    def test_singular_covariance_matrix(self):
        """Test handling of singular covariance matrices"""
        optimizer = AdvancedPortfolioOptimizer()
        
        # Create perfectly correlated returns (singular covariance)
        base_returns = np.random.normal(0.001, 0.02, 100)
        correlated_returns = pd.DataFrame({
            'STOCK1': base_returns,
            'STOCK2': base_returns * 2,  # Perfectly correlated
            'STOCK3': base_returns * 0.5
        })
        
        result = optimizer.risk_parity_optimization(correlated_returns)
        
        # Should handle gracefully and return reasonable result
        assert 'weights' in result
        assert abs(sum(result['weights']) - 1.0) < 0.01
        
    def test_extreme_constraints(self):
        """Test optimization with extreme constraints"""
        optimizer = AdvancedPortfolioOptimizer()
        
        sample_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252)
        })
        
        # Impossible constraints (min > max)
        impossible_constraints = {
            'min_weights': 0.8,
            'max_weights': 0.4
        }
        
        with pytest.raises(ValueError):
            optimizer.constrained_optimization(
                returns=sample_returns,
                strategy='sharpe',
                constraints=impossible_constraints
            )
            
    def test_negative_returns_handling(self):
        """Test handling of assets with negative expected returns"""
        optimizer = AdvancedPortfolioOptimizer()
        
        # Create returns where one asset has negative expected return
        negative_returns = pd.DataFrame({
            'GOOD_STOCK': np.random.normal(0.002, 0.02, 252),
            'BAD_STOCK': np.random.normal(-0.001, 0.03, 252),
            'NEUTRAL_STOCK': np.random.normal(0.0005, 0.015, 252)
        })
        
        result = optimizer.black_litterman_optimization(negative_returns)
        
        # Should handle negative returns and potentially give zero or minimal weight to bad stock
        assert 'weights' in result
        assert result['weights']['BAD_STOCK'] <= result['weights']['GOOD_STOCK']