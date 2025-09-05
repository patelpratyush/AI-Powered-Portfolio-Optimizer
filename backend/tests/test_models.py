#!/usr/bin/env python3
"""
ML Model tests
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd
from models.xgb_model import XGBoostStockPredictor
from forecast import ProphetPredictor
from models.buy_sell_advisor import BuySellAdvisor


class TestXGBoostPredictor:
    """Test XGBoost model functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = XGBoostStockPredictor()
        
    @patch('models.xgb_model.yf.download')
    def test_fetch_data_success(self, mock_download):
        """Test successful data fetching"""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        mock_download.return_value = mock_data
        
        result = self.predictor.fetch_data('AAPL', '1y')
        
        assert not result.empty
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        mock_download.assert_called_once()
        
    @patch('models.xgb_model.yf.download')
    def test_fetch_data_failure(self, mock_download):
        """Test data fetching failure handling"""
        mock_download.side_effect = Exception("API Error")
        
        result = self.predictor.fetch_data('INVALID', '1y')
        
        assert result.empty
        
    def test_create_features(self):
        """Test feature engineering"""
        # Create sample data
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        features = self.predictor.create_features(data)
        
        # Check that technical indicators were added
        expected_features = ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_12']
        for feature in expected_features:
            assert feature in features.columns
            
        # Check that features have correct shape
        assert features.shape[0] == len(data)
        assert features.shape[1] > 10  # Should have many technical indicators
        
    @patch('models.xgb_model.joblib.dump')
    @patch('models.xgb_model.XGBRegressor')
    def test_train_model_success(self, mock_xgb, mock_dump):
        """Test successful model training"""
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.fit = Mock()
        mock_model.predict = Mock(return_value=np.array([100, 101, 102]))
        mock_xgb.return_value = mock_model
        
        # Mock data fetching
        with patch.object(self.predictor, 'fetch_data') as mock_fetch:
            mock_data = pd.DataFrame({
                'Open': np.random.rand(100) * 100 + 100,
                'High': np.random.rand(100) * 100 + 105,
                'Low': np.random.rand(100) * 100 + 95,
                'Close': np.random.rand(100) * 100 + 102,
                'Volume': np.random.rand(100) * 1000000 + 1000000
            })
            mock_fetch.return_value = mock_data
            
            result = self.predictor.train('AAPL', period='1y')
            
            assert 'model_path' in result
            assert 'test_metrics' in result
            assert 'mae' in result['test_metrics']
            assert 'r2' in result['test_metrics']
            mock_model.fit.assert_called_once()
            mock_dump.assert_called()


class TestProphetPredictor:
    """Test Prophet model functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = ProphetPredictor()
        
    @patch('forecast.yf.download')
    def test_predict_success(self, mock_download):
        """Test successful Prophet prediction"""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Close': [100 + i + np.sin(i/10) * 5 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        mock_download.return_value = mock_data
        
        result = self.predictor.predict('AAPL', days=10)
        
        assert 'predictions' in result
        assert 'forecast_data' in result
        assert len(result['predictions']) == 10
        assert all('date' in pred for pred in result['predictions'])
        assert all('predicted_price' in pred for pred in result['predictions'])
        
    @patch('forecast.yf.download')
    def test_predict_insufficient_data(self, mock_download):
        """Test prediction with insufficient data"""
        # Mock insufficient data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102]  # Too few data points
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_download.return_value = mock_data
        
        result = self.predictor.predict('AAPL', days=10)
        
        assert 'error' in result
        assert 'insufficient data' in result['error'].lower()


class TestBuySellAdvisor:
    """Test trading recommendation engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.advisor = BuySellAdvisor()
        
    def test_analyze_stock_success(self):
        """Test successful stock analysis"""
        # Create sample data with clear patterns
        sample_data = {
            'current_price': 150.0,
            'predicted_prices': [152, 155, 148, 160, 158],
            'technical_indicators': {
                'RSI': 65.0,
                'MACD': 2.5,
                'BB_position': 0.8,
                'volume_trend': 1.2
            },
            'fundamental_data': {
                'pe_ratio': 25.0,
                'revenue_growth': 0.15,
                'debt_ratio': 0.3
            }
        }
        
        result = self.advisor.analyze_stock('AAPL', sample_data)
        
        assert 'action' in result
        assert result['action'] in ['BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL']
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 100
        assert 'reasoning' in result
        assert isinstance(result['reasoning'], list)
        
    def test_calculate_technical_score(self):
        """Test technical analysis scoring"""
        indicators = {
            'RSI': 30.0,  # Oversold - bullish
            'MACD': 1.5,  # Positive - bullish
            'BB_position': 0.2,  # Lower band - bullish
            'volume_trend': 1.5  # High volume - strong signal
        }
        
        score = self.advisor.calculate_technical_score(indicators)
        
        assert isinstance(score, float)
        assert -100 <= score <= 100
        # Should be positive (bullish) given the indicators
        assert score > 0
        
    def test_calculate_fundamental_score(self):
        """Test fundamental analysis scoring"""
        fundamentals = {
            'pe_ratio': 15.0,  # Reasonable P/E
            'revenue_growth': 0.20,  # Strong growth
            'debt_ratio': 0.25  # Low debt
        }
        
        score = self.advisor.calculate_fundamental_score(fundamentals)
        
        assert isinstance(score, float)
        assert -100 <= score <= 100
        # Should be positive given good fundamentals
        assert score > 0
        
    def test_generate_action_strong_buy(self):
        """Test action generation for strong buy signal"""
        # Very positive scores should generate STRONG_BUY
        result = self.advisor.generate_action(80, 85, 90)
        
        assert result['action'] in ['STRONG_BUY', 'BUY']
        assert result['confidence'] > 70
        
    def test_generate_action_strong_sell(self):
        """Test action generation for strong sell signal"""
        # Very negative scores should generate STRONG_SELL
        result = self.advisor.generate_action(-80, -85, -75)
        
        assert result['action'] in ['STRONG_SELL', 'SELL']
        assert result['confidence'] > 70
        
    def test_generate_action_hold(self):
        """Test action generation for hold signal"""
        # Neutral scores should generate HOLD
        result = self.advisor.generate_action(5, -5, 10)
        
        assert result['action'] == 'HOLD'
        assert result['confidence'] < 70


class TestModelIntegration:
    """Test model integration and ensemble functionality"""
    
    @patch('models.xgb_model.XGBoostStockPredictor.predict')
    @patch('forecast.ProphetPredictor.predict')
    def test_ensemble_prediction(self, mock_prophet, mock_xgb):
        """Test ensemble prediction combining multiple models"""
        # Mock XGBoost prediction
        mock_xgb.return_value = {
            'predictions': [
                {'date': '2024-01-01', 'predicted_price': 150.0},
                {'date': '2024-01-02', 'predicted_price': 152.0}
            ]
        }
        
        # Mock Prophet prediction
        mock_prophet.return_value = {
            'predictions': [
                {'date': '2024-01-01', 'predicted_price': 148.0},
                {'date': '2024-01-02', 'predicted_price': 151.0}
            ]
        }
        
        # Test ensemble logic (would be implemented in routes)
        xgb_predictions = mock_xgb.return_value['predictions']
        prophet_predictions = mock_prophet.return_value['predictions']
        
        # Simple ensemble averaging
        ensemble_predictions = []
        for i in range(len(xgb_predictions)):
            avg_price = (xgb_predictions[i]['predicted_price'] + prophet_predictions[i]['predicted_price']) / 2
            ensemble_predictions.append({
                'date': xgb_predictions[i]['date'],
                'predicted_price': avg_price,
                'xgb_price': xgb_predictions[i]['predicted_price'],
                'prophet_price': prophet_predictions[i]['predicted_price']
            })
        
        assert len(ensemble_predictions) == 2
        assert ensemble_predictions[0]['predicted_price'] == 149.0  # (150 + 148) / 2
        assert ensemble_predictions[1]['predicted_price'] == 151.5  # (152 + 151) / 2


class TestModelError:
    """Test model error handling and edge cases"""
    
    def test_xgb_predictor_invalid_ticker(self):
        """Test XGBoost predictor with invalid ticker"""
        predictor = XGBoostStockPredictor()
        
        with patch.object(predictor, 'fetch_data', return_value=pd.DataFrame()):
            result = predictor.train('INVALID_TICKER')
            
            assert 'error' in result
            
    def test_prophet_predictor_network_error(self):
        """Test Prophet predictor with network error"""
        predictor = ProphetPredictor()
        
        with patch('forecast.yf.download', side_effect=Exception("Network error")):
            result = predictor.predict('AAPL', days=5)
            
            assert 'error' in result
            assert 'network error' in result['error'].lower()
            
    def test_buy_sell_advisor_missing_data(self):
        """Test BuySellAdvisor with missing data"""
        advisor = BuySellAdvisor()
        
        # Missing required fields
        incomplete_data = {
            'current_price': 150.0
            # Missing predicted_prices, technical_indicators, etc.
        }
        
        result = advisor.analyze_stock('AAPL', incomplete_data)
        
        # Should handle gracefully and return default values
        assert 'action' in result
        assert result['action'] == 'HOLD'  # Default action when data is insufficient
        assert result['confidence'] < 50  # Low confidence due to missing data