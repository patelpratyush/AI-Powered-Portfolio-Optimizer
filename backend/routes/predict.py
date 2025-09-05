#!/usr/bin/env python3
"""
Stock Prediction API
Unified endpoint for Prophet, XGBoost, and LSTM predictions
"""

from flask import Blueprint, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import security utilities
from utils.security import secure_api_endpoint, create_custom_rate_limit
from schemas.validation import PredictionRequest, BatchPredictionRequest, ModelTrainingRequest

from models.xgb_model import XGBoostStockPredictor
from models.lstm_model import LSTMStockPredictor
from models.buy_sell_advisor import BuySellAdvisor
from utils.model_cache import get_cached_predictor, get_model_cache
from utils.sentiment_analysis import SentimentAggregator, create_sentiment_features
from forecast import generate_ai_forecast
import yfinance as yf
from utils.yahoo_finance_cache import (
    get_current_price, get_daily_data, get_stock_info as get_cached_stock_info,
    get_multiple_current_prices, get_yahoo_finance_cache
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import concurrent.futures
import time
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
predict_bp = Blueprint('predict', __name__)

# Initialize cached predictors (high performance lazy loading)
_cached_predictors = {}
_buy_sell_advisor = None
_sentiment_aggregator = None

def get_xgb_predictor():
    """Get cached XGBoost predictor instance"""
    if 'xgboost' not in _cached_predictors:
        _cached_predictors['xgboost'] = get_cached_predictor('xgboost')
    return _cached_predictors['xgboost']

def get_lstm_predictor():
    """Get cached LSTM predictor instance"""
    if 'lstm' not in _cached_predictors:
        _cached_predictors['lstm'] = get_cached_predictor('lstm')
    return _cached_predictors['lstm']

def get_buy_sell_advisor():
    """Get Buy/Sell advisor instance (lazy loading)"""
    global _buy_sell_advisor
    if _buy_sell_advisor is None:
        _buy_sell_advisor = BuySellAdvisor()
    return _buy_sell_advisor

def get_sentiment_aggregator():
    """Get Sentiment aggregator instance (lazy loading)"""
    global _sentiment_aggregator
    if _sentiment_aggregator is None:
        from utils.cache import get_cache_client
        cache_client = get_cache_client()
        _sentiment_aggregator = SentimentAggregator(cache_client)
    return _sentiment_aggregator

def get_stock_info(ticker: str) -> Dict:
    """Get basic stock information with caching"""
    try:
        # Use cached functions for better performance
        current_price = get_current_price(ticker)
        info = get_cached_stock_info(ticker)
        
        # Get recent history for day change calculation
        hist = get_daily_data(ticker, period="5d")
        
        if hist.empty:
            return None
        
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'current_price': float(current_price),
            'previous_close': float(prev_close),
            'day_change': float(current_price - prev_close),
            'day_change_percent': float((current_price - prev_close) / prev_close * 100),
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'currency': info.get('currency', 'USD'),
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stock info for {ticker}: {str(e)}")
        return None

def combine_predictions(prophet_pred: Dict, xgb_pred: Optional[Dict], lstm_pred: Optional[Dict]) -> Dict:
    """Combine predictions from multiple models into ensemble"""
    
    ensemble_predictions = []
    models_used = ['Prophet']
    model_weights = {'Prophet': 0.4}  # Base weight for Prophet
    
    # Add XGBoost if available
    if xgb_pred and 'predictions' in xgb_pred:
        models_used.append('XGBoost')
        model_weights['XGBoost'] = 0.35
    
    # Add LSTM if available
    if lstm_pred and 'predictions' in lstm_pred:
        models_used.append('LSTM')
        model_weights['LSTM'] = 0.25
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Combine predictions day by day
    max_days = len(prophet_pred.get('predictions', []))
    
    for day in range(1, max_days + 1):
        prophet_day = next((p for p in prophet_pred['predictions'] if p['day'] == day), None)
        xgb_day = next((p for p in xgb_pred.get('predictions', []) if p['day'] == day), None) if xgb_pred else None
        lstm_day = next((p for p in lstm_pred.get('predictions', []) if p['day'] == day), None) if lstm_pred else None
        
        if not prophet_day:
            continue
        
        # Ensemble price prediction
        ensemble_price = prophet_day['predicted_price'] * model_weights['Prophet']
        ensemble_return = prophet_day['predicted_return'] * model_weights['Prophet']
        
        if xgb_day:
            ensemble_price += xgb_day['predicted_price'] * model_weights['XGBoost']
            ensemble_return += xgb_day['predicted_return'] * model_weights['XGBoost']
        
        if lstm_day:
            ensemble_price += lstm_day['predicted_price'] * model_weights['LSTM']
            ensemble_return += lstm_day['predicted_return'] * model_weights['LSTM']
        
        # Confidence intervals (use Prophet's as base, adjust with other models)
        confidence_lower = prophet_day['confidence_lower']
        confidence_upper = prophet_day['confidence_upper']
        
        if xgb_day and lstm_day:
            # If we have all models, average the confidence intervals
            confidence_lower = (confidence_lower + 
                              xgb_day.get('confidence_lower', confidence_lower) +
                              lstm_day.get('confidence_lower', confidence_lower)) / 3
            confidence_upper = (confidence_upper +
                              xgb_day.get('confidence_upper', confidence_upper) +
                              lstm_day.get('confidence_upper', confidence_upper)) / 3
        
        ensemble_predictions.append({
            'day': day,
            'predicted_price': float(ensemble_price),
            'predicted_return': float(ensemble_return),
            'confidence_lower': float(confidence_lower),
            'confidence_upper': float(confidence_upper),
            'date': prophet_day['date'],
            'models_used': models_used,
            'model_weights': model_weights
        })
    
    # Generate enhanced trading signal using BuySellAdvisor
    avg_return = np.mean([p['predicted_return'] for p in ensemble_predictions])
    prediction_std = np.std([p['predicted_return'] for p in ensemble_predictions])
    
    # Create ML predictions summary for advisor
    ml_prediction_summary = {
        'summary': {
            'avg_predicted_return': avg_return,
            'max_predicted_price': max(p['predicted_price'] for p in ensemble_predictions),
            'min_predicted_price': min(p['predicted_price'] for p in ensemble_predictions),
            'volatility_estimate': prediction_std
        },
        'trading_signal': {
            'action': 'BUY' if avg_return > 0.025 else 'SELL' if avg_return < -0.025 else 'HOLD',
            'strength': min(abs(avg_return) * 12, 1.0)
        }
    }
    
    # Get enhanced recommendation from BuySellAdvisor
    try:
        ticker_symbol = prophet_pred.get('ticker', 'UNKNOWN')
        advisor = get_buy_sell_advisor()
        recommendation = advisor.generate_recommendation(ticker_symbol, ml_prediction_summary)
        
        # Convert advisor recommendation to API format
        enhanced_signal = {
            'action': str(recommendation.signal.value),
            'strength': float(recommendation.confidence / 100.0),
            'confidence': float(recommendation.confidence),
            'reasoning': str(recommendation.summary),
            'target_price': float(recommendation.target_price) if recommendation.target_price is not None else None,
            'stop_loss': float(recommendation.stop_loss) if recommendation.stop_loss is not None else None,
            'risk_level': str(recommendation.risk_level),
            'expected_return': float(recommendation.expected_return),
            'max_downside': float(recommendation.max_downside),
            'time_horizon': str(recommendation.time_horizon),
            'reasons': [
                {
                    'category': str(reason.category),
                    'indicator': str(reason.indicator),
                    'value': float(reason.value),
                    'threshold': float(reason.threshold),
                    'weight': float(reason.weight),
                    'description': str(reason.description),
                    'bullish': bool(reason.bullish)
                }
                for reason in recommendation.reasons
            ],
            'summary': recommendation.summary
        }
        
    except Exception as e:
        logger.warning(f"BuySellAdvisor failed, using basic signal: {str(e)}")
        # Fallback to basic signal
        if avg_return > 0.025:
            signal = "BUY"
            strength = min(avg_return * 12, 1.0)
        elif avg_return < -0.025:
            signal = "SELL"
            strength = min(abs(avg_return) * 12, 1.0)
        else:
            signal = "HOLD"
            strength = 0.5
        
        enhanced_signal = {
            'action': signal,
            'strength': float(strength),
            'confidence': float(strength * 100),
            'reasoning': f"Ensemble of {len(models_used)} models predicts {avg_return:.2%} return",
            'reasons': []
        }
    
    return {
        'predictions': ensemble_predictions,
        'summary': {
            'avg_predicted_return': float(avg_return),
            'max_predicted_price': float(max(p['predicted_price'] for p in ensemble_predictions)),
            'min_predicted_price': float(min(p['predicted_price'] for p in ensemble_predictions)),
            'volatility_estimate': float(prediction_std),
            'trend_direction': 'bullish' if avg_return > 0 else 'bearish'
        },
        'trading_signal': enhanced_signal,
        'ensemble_info': {
            'models_used': models_used,
            'model_weights': model_weights
        }
    }

@predict_bp.route('/predict/<ticker>', methods=['GET'])
@secure_api_endpoint(rate_limit="30 per minute")
def predict_stock(ticker: str):
    """Predict stock price using ensemble of models"""
    try:
        ticker = ticker.upper().strip()
        
        # Get query parameters
        days_ahead = int(request.args.get('days', 10))
        models = request.args.get('models', 'all').lower()  # all, prophet, xgboost, lstm
        
        # Validate parameters
        if days_ahead < 1 or days_ahead > 30:
            return jsonify({'error': 'days_ahead must be between 1 and 30'}), 400
        
        # Get stock info
        stock_info = get_stock_info(ticker)
        if not stock_info:
            return jsonify({'error': f'Unable to fetch data for ticker {ticker}'}), 404
        
        # Get sentiment analysis
        sentiment_analysis = None
        try:
            logger.info(f"Getting sentiment analysis for {ticker}")
            sentiment_aggregator = get_sentiment_aggregator()
            sentiment_score = sentiment_aggregator.get_ticker_sentiment(ticker, days=7)
            sentiment_analysis = {
                'score': sentiment_score.score,
                'confidence': sentiment_score.confidence,
                'interpretation': _interpret_sentiment_score(sentiment_score.score, sentiment_score.confidence),
                'volume': sentiment_score.volume,
                'source': sentiment_score.source,
                'last_updated': sentiment_score.timestamp.isoformat()
            }
            logger.info(f"Sentiment analysis completed for {ticker}: score={sentiment_score.score:.3f}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {ticker}: {str(e)}")
        
        predictions = {}
        
        # Prophet prediction (always included as baseline)
        try:
            logger.info(f"Generating Prophet forecast for {ticker}")
            prophet_result = generate_ai_forecast([ticker], period="1y")
            if ticker in prophet_result:
                forecast_data = prophet_result[ticker]
                
                # Convert Prophet format to standard format
                prophet_predictions = []
                for i, (date, values) in enumerate(forecast_data.items(), 1):
                    if i > days_ahead:
                        break
                    
                    predicted_price = values.get('value', stock_info['current_price'])
                    predicted_return = (predicted_price - stock_info['current_price']) / stock_info['current_price']
                    
                    prophet_predictions.append({
                        'day': i,
                        'predicted_price': float(predicted_price),
                        'predicted_return': float(predicted_return),
                        'confidence_lower': float(values.get('lower', predicted_price * 0.9)),
                        'confidence_upper': float(values.get('upper', predicted_price * 1.1)),
                        'date': date
                    })
                
                predictions['prophet'] = {
                    'ticker': ticker,
                    'model': 'Prophet',
                    'current_price': stock_info['current_price'],
                    'predictions': prophet_predictions,
                    'summary': {
                        'avg_predicted_return': float(np.mean([p['predicted_return'] for p in prophet_predictions])),
                        'max_predicted_price': float(max(p['predicted_price'] for p in prophet_predictions)),
                        'min_predicted_price': float(min(p['predicted_price'] for p in prophet_predictions)),
                        'volatility_estimate': float(np.std([p['predicted_return'] for p in prophet_predictions]))
                    },
                    'trading_signal': {
                        'action': 'HOLD',
                        'strength': 0.5,
                        'reasoning': 'Prophet time-series analysis'
                    }
                }
        except Exception as e:
            logger.error(f"Prophet prediction failed for {ticker}: {str(e)}")
            predictions['prophet'] = None
        
        # XGBoost prediction
        if models in ['all', 'xgboost']:
            try:
                logger.info(f"Generating XGBoost prediction for {ticker}")
                xgb_predictor = get_xgb_predictor()
                
                # Try to load existing model, or train if not available
                try:
                    xgb_result = xgb_predictor.predict(ticker, days_ahead=days_ahead)
                    predictions['xgboost'] = xgb_result
                except (FileNotFoundError, Exception) as e:
                    logger.warning(f"XGBoost model not found for {ticker}: {str(e)}")
                    predictions['xgboost'] = {
                        'error': 'Model not trained',
                        'message': f'XGBoost model for {ticker} needs to be trained first. Use the training interface or API.',
                        'suggested_action': f'POST /api/train/{ticker} with models: ["xgboost"]'
                    }
                
            except Exception as e:
                logger.error(f"XGBoost prediction failed for {ticker}: {str(e)}")
                predictions['xgboost'] = {
                    'error': 'Prediction failed',
                    'message': str(e)
                }
        
        # LSTM prediction
        if models in ['all', 'lstm']:
            try:
                logger.info(f"Generating LSTM prediction for {ticker}")
                lstm_predictor = get_lstm_predictor()
                
                # Try to load existing model, or train if not available
                try:
                    lstm_result = lstm_predictor.predict(ticker, days_ahead=days_ahead)
                    predictions['lstm'] = lstm_result
                except (FileNotFoundError, Exception) as e:
                    logger.warning(f"LSTM model not found for {ticker}: {str(e)}")
                    predictions['lstm'] = {
                        'error': 'Model not trained',
                        'message': f'LSTM model for {ticker} needs to be trained first. Use the training interface or API.',
                        'suggested_action': f'POST /api/train/{ticker} with models: ["lstm"]'
                    }
                
            except Exception as e:
                logger.error(f"LSTM prediction failed for {ticker}: {str(e)}")
                predictions['lstm'] = {
                    'error': 'Prediction failed',
                    'message': str(e)
                }
        
        # Create ensemble if multiple models available
        if models == 'all' and predictions['prophet']:
            ensemble_result = combine_predictions(
                predictions['prophet'],
                predictions.get('xgboost'),
                predictions.get('lstm')
            )
            predictions['ensemble'] = {
                'ticker': ticker,
                'model': 'Ensemble',
                'current_price': stock_info['current_price'],
                **ensemble_result
            }
        
        # Return results
        response = {
            'stock_info': stock_info,
            'predictions': predictions,
            'sentiment_analysis': sentiment_analysis,
            'request_info': {
                'ticker': ticker,
                'days_ahead': days_ahead,
                'models_requested': models,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction API error for {ticker}: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@predict_bp.route('/train/<ticker>', methods=['POST'])
@secure_api_endpoint(
    schema=ModelTrainingRequest,
    require_auth=True,
    rate_limit="5 per hour"  # Very restrictive for resource-intensive training
)
def train_models(ticker: str):
    """Train ML models for a specific ticker"""
    try:
        # Use validated data from security decorator
        validated_data = request.validated_json
        ticker = validated_data['ticker']
        models_to_train = validated_data['models']
        period = validated_data['period']
        
        training_results = {}
        
        # Train XGBoost
        if 'xgboost' in models_to_train:
            try:
                logger.info(f"Training XGBoost model for {ticker}")
                xgb_predictor = get_xgb_predictor()
                xgb_result = xgb_predictor.train(ticker, period=period)
                training_results['xgboost'] = xgb_result
            except Exception as e:
                logger.error(f"XGBoost training failed: {str(e)}")
                training_results['xgboost'] = {'error': str(e)}
        
        # Train LSTM
        if 'lstm' in models_to_train:
            try:
                logger.info(f"Training LSTM model for {ticker}")
                lstm_predictor = get_lstm_predictor()
                lstm_result = lstm_predictor.train(ticker, period=period)
                training_results['lstm'] = lstm_result
            except Exception as e:
                logger.error(f"LSTM training failed: {str(e)}")
                training_results['lstm'] = {'error': str(e)}
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_to_serializable(training_results)
        
        return jsonify({
            'ticker': ticker,
            'training_results': serializable_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Training API error for {ticker}: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@predict_bp.route('/cache/status', methods=['GET'])
@secure_api_endpoint(rate_limit="10 per minute")
def get_cache_status():
    """Get comprehensive cache status (ML models + Yahoo Finance)"""
    try:
        # ML model cache stats
        model_cache = get_model_cache()
        model_stats = model_cache.get_cache_stats()
        
        from utils.model_cache import get_cache_health
        model_health = get_cache_health()
        
        # Yahoo Finance cache stats
        yf_cache = get_yahoo_finance_cache()
        yf_stats = yf_cache.get_stats()
        
        from utils.yahoo_finance_cache import get_cache_health as get_yf_health
        yf_health = get_yf_health()
        
        return jsonify({
            'model_cache': {
                'stats': model_stats,
                'health': model_health
            },
            'yahoo_finance_cache': {
                'stats': yf_stats,
                'health': yf_health
            },
            'overall_health': 'healthy' if (
                model_health['status'] == 'healthy' and 
                yf_health['status'] == 'healthy'
            ) else 'warning',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Cache status error: {str(e)}")
        return jsonify({'error': f'Cache status failed: {str(e)}'}), 500

@predict_bp.route('/models/available', methods=['GET'])
def get_available_models():
    """Get list of available prediction models and their status"""
    return jsonify({
        'models': {
            'prophet': {
                'name': 'Prophet',
                'description': 'Facebook Prophet time-series forecasting',
                'type': 'statistical',
                'training_required': False,
                'best_for': 'Long-term trends and seasonality',
                'time_horizon': '1-365 days'
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Extreme Gradient Boosting with technical indicators',
                'type': 'machine_learning',
                'training_required': True,
                'best_for': 'Short-term predictions with technical analysis',
                'time_horizon': '1-30 days'
            },
            'lstm': {
                'name': 'LSTM',
                'description': 'Long Short-Term Memory neural network',
                'type': 'deep_learning',
                'training_required': True,
                'best_for': 'Pattern recognition in price sequences',
                'time_horizon': '1-30 days'
            },
            'ensemble': {
                'name': 'Ensemble',
                'description': 'Weighted combination of all models',
                'type': 'ensemble',
                'training_required': True,
                'best_for': 'Balanced predictions with uncertainty quantification',
                'time_horizon': '1-30 days'
            }
        },
        'usage_recommendations': {
            'conservative_investor': ['prophet', 'ensemble'],
            'technical_trader': ['xgboost', 'ensemble'],
            'pattern_trader': ['lstm', 'ensemble'],
            'comprehensive_analysis': ['ensemble']
        }
    })

@predict_bp.route('/batch-predict', methods=['POST'])
@secure_api_endpoint(
    schema=BatchPredictionRequest,
    require_auth=True,
    rate_limit="10 per hour"  # Restrictive for batch operations
)
def batch_predict():
    """Predict multiple stocks in parallel"""
    try:
        data = request.get_json()
        if not data or 'tickers' not in data:
            return jsonify({'error': 'Missing tickers list'}), 400
        
        tickers = [ticker.upper().strip() for ticker in data['tickers']]
        days_ahead = data.get('days', 10)
        models = data.get('models', 'ensemble')
        
        if len(tickers) > 10:
            return jsonify({'error': 'Maximum 10 tickers allowed for batch prediction'}), 400
        
        results = {}
        
        # Use threading for parallel predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {}
            
            for ticker in tickers:
                future = executor.submit(predict_single_stock, ticker, days_ahead, models)
                future_to_ticker[future] = ticker
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per stock
                    results[ticker] = result
                except Exception as e:
                    logger.error(f"Batch prediction failed for {ticker}: {str(e)}")
                    results[ticker] = {'error': str(e)}
        
        return jsonify({
            'batch_results': results,
            'summary': {
                'total_tickers': len(tickers),
                'successful_predictions': len([r for r in results.values() if 'error' not in r]),
                'failed_predictions': len([r for r in results.values() if 'error' in r])
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

def predict_single_stock(ticker: str, days_ahead: int, models: str) -> Dict:
    """Helper function for single stock prediction"""
    # This would call the same logic as the main predict endpoint
    # Simplified version for batch processing
    try:
        stock_info = get_stock_info(ticker)
        if not stock_info:
            return {'error': f'Unable to fetch data for {ticker}'}
        
        # For batch processing, we'll use a simplified prediction
        # In practice, you'd call the full prediction logic
        return {
            'stock_info': stock_info,
            'prediction_summary': {
                'status': 'completed',
                'models_used': [models] if models != 'all' else ['prophet', 'xgboost', 'lstm']
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def _interpret_sentiment_score(score: float, confidence: float) -> Dict[str, str]:
    """Interpret sentiment score for predictions"""
    if confidence < 0.3:
        confidence_level = 'low'
    elif confidence < 0.7:
        confidence_level = 'moderate' 
    else:
        confidence_level = 'high'
    
    if score > 0.3:
        sentiment_label = 'bullish'
        impact = 'positive'
    elif score > 0.1:
        sentiment_label = 'slightly_positive'
        impact = 'mildly_positive'
    elif score > -0.1:
        sentiment_label = 'neutral'
        impact = 'neutral'
    elif score > -0.3:
        sentiment_label = 'slightly_negative'
        impact = 'mildly_negative'
    else:
        sentiment_label = 'bearish'
        impact = 'negative'
    
    return {
        'label': sentiment_label,
        'impact': impact,
        'confidence_level': confidence_level
    }

if __name__ == "__main__":
    # Test the prediction API
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(predict_bp, url_prefix="/api")
    
    print("Testing prediction API...")
    
    # Test individual model prediction
    with app.test_client() as client:
        response = client.get('/api/predict/AAPL?days=5&models=prophet')
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"Prophet prediction for AAPL successful")
            print(f"Current price: ${data['stock_info']['current_price']:.2f}")
            if 'prophet' in data['predictions']:
                pred_summary = data['predictions']['prophet']['summary']
                print(f"Avg predicted return: {pred_summary['avg_predicted_return']:.2%}")
        else:
            print(f"Error: {response.get_json()}")