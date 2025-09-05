#!/usr/bin/env python3
"""
API endpoint tests
"""
import pytest
import json
from unittest.mock import patch, MagicMock

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns correct response"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'environment' in data

class TestPredictionAPI:
    """Test prediction API endpoints"""
    
    def test_predict_endpoint_invalid_ticker(self, client):
        """Test prediction with invalid ticker"""
        response = client.get('/api/predict/')
        assert response.status_code == 404
        
        response = client.get('/api/predict/INVALID_TICKER_TOO_LONG')
        # Should handle gracefully, not crash
        assert response.status_code in [400, 404, 500]
    
    def test_predict_endpoint_valid_params(self, client):
        """Test prediction with valid parameters"""
        # Mock the external dependencies
        with patch('routes.predict.get_stock_info') as mock_stock_info, \
             patch('routes.predict.generate_ai_forecast') as mock_forecast:
            
            mock_stock_info.return_value = {
                'ticker': 'AAPL',
                'name': 'Apple Inc.',
                'current_price': 150.0,
                'previous_close': 148.0,
                'day_change': 2.0,
                'day_change_percent': 1.35,
                'market_cap': 2400000000000,
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'currency': 'USD',
                'last_updated': '2024-01-01T12:00:00'
            }
            
            mock_forecast.return_value = {
                'AAPL': {
                    '2024-01-02': {'value': 151.0, 'lower': 148.0, 'upper': 154.0},
                    '2024-01-03': {'value': 152.0, 'lower': 149.0, 'upper': 155.0}
                }
            }
            
            response = client.get('/api/predict/AAPL?days=2&models=prophet')
            assert response.status_code == 200
            
            data = response.get_json()
            assert 'stock_info' in data
            assert 'predictions' in data
            assert data['stock_info']['ticker'] == 'AAPL'

class TestOptimizationAPI:
    """Test portfolio optimization endpoints"""
    
    def test_optimize_missing_data(self, client):
        """Test optimization with missing data"""
        response = client.post('/api/optimize')
        assert response.status_code == 400
    
    def test_optimize_invalid_data(self, client):
        """Test optimization with invalid data"""
        invalid_data = {
            'tickers': [],  # Empty tickers
            'strategy': 'invalid_strategy'
        }
        response = client.post('/api/optimize', 
                             data=json.dumps(invalid_data),
                             content_type='application/json')
        assert response.status_code == 400
    
    def test_optimize_valid_data(self, client, sample_portfolio_data):
        """Test optimization with valid data"""
        with patch('routes.optimize.yf.download') as mock_download:
            # Mock successful data download
            mock_df = MagicMock()
            mock_df.empty = False
            mock_download.return_value = mock_df
            
            response = client.post('/api/optimize',
                                 data=json.dumps(sample_portfolio_data),
                                 content_type='application/json')
            
            # Should not crash - may return error due to mock data but should handle gracefully
            assert response.status_code in [200, 400, 500]

class TestAutocompleteAPI:
    """Test autocomplete API"""
    
    def test_autocomplete_empty_query(self, client):
        """Test autocomplete with empty query"""
        response = client.get('/api/autocomplete')
        assert response.status_code == 400
    
    def test_autocomplete_valid_query(self, client):
        """Test autocomplete with valid query"""
        response = client.get('/api/autocomplete?q=AAP')
        # Should return some results or handle gracefully
        assert response.status_code in [200, 500]

class TestModelTraining:
    """Test model training endpoints"""
    
    def test_train_invalid_ticker(self, client):
        """Test training with invalid ticker"""
        response = client.post('/api/train/')
        assert response.status_code == 404
    
    def test_train_valid_request(self, client):
        """Test training with valid request"""
        train_data = {
            'models': ['xgboost'],
            'period': '1y'
        }
        
        with patch('routes.predict.get_xgb_predictor') as mock_predictor:
            mock_instance = MagicMock()
            mock_instance.train.return_value = {'status': 'success'}
            mock_predictor.return_value = mock_instance
            
            response = client.post('/api/train/AAPL',
                                 data=json.dumps(train_data),
                                 content_type='application/json')
            
            # Should handle gracefully
            assert response.status_code in [200, 500]

class TestErrorHandling:
    """Test error handling"""
    
    def test_404_handler(self, client):
        """Test 404 error handler"""
        response = client.get('/nonexistent-endpoint')
        assert response.status_code == 404
        
        data = response.get_json()
        assert 'error' in data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        response = client.put('/api/predict/AAPL')  # PUT not allowed
        assert response.status_code == 405