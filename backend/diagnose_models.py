#!/usr/bin/env python3
"""
Model Diagnostics Script
Test XGBoost and LSTM models to see what they're actually predicting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.xgb_model import XGBoostStockPredictor
from models.lstm_model import LSTMStockPredictor
import numpy as np

def test_xgboost_model(ticker='AAPL'):
    print(f"\n🔍 Testing XGBoost model for {ticker}")
    print("=" * 50)
    
    try:
        predictor = XGBoostStockPredictor()
        result = predictor.predict(ticker, days_ahead=5)
        
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Model Loaded: {predictor.is_trained}")
        print(f"Feature Count: {len(predictor.feature_names)}")
        
        print("\nPredictions:")
        for pred in result['predictions']:
            print(f"Day {pred['day']}: ${pred['predicted_price']:.2f} ({pred['predicted_return']*100:.3f}% return)")
        
        print(f"\nSummary:")
        print(f"Avg Return: {result['summary']['avg_predicted_return']*100:.3f}%")
        print(f"Trading Signal: {result['trading_signal']['action']} (strength: {result['trading_signal']['strength']:.3f})")
        
        # Check if model is making meaningful predictions
        returns = [p['predicted_return'] for p in result['predictions']]
        if all(abs(r) < 0.001 for r in returns):  # Less than 0.1% return
            print("⚠️  WARNING: Model is making very conservative predictions (< 0.1% returns)")
        
        return result
        
    except Exception as e:
        print(f"❌ XGBoost Error: {str(e)}")
        return None

def test_lstm_model(ticker='AAPL'):
    print(f"\n🔍 Testing LSTM model for {ticker}")
    print("=" * 50)
    
    try:
        predictor = LSTMStockPredictor()
        result = predictor.predict(ticker, days_ahead=5)
        
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Model Loaded: {predictor.is_trained}")
        print(f"Feature Count: {len(predictor.feature_names)}")
        
        print("\nPredictions:")
        for pred in result['predictions']:
            print(f"Day {pred['day']}: ${pred['predicted_price']:.2f} ({pred['predicted_return']*100:.3f}% return)")
        
        print(f"\nSummary:")
        print(f"Avg Return: {result['summary']['avg_predicted_return']*100:.3f}%")
        print(f"Trading Signal: {result['trading_signal']['action']} (strength: {result['trading_signal']['strength']:.3f})")
        
        # Check if model is making meaningful predictions
        returns = [p['predicted_return'] for p in result['predictions']]
        if all(abs(r) < 0.001 for r in returns):  # Less than 0.1% return
            print("⚠️  WARNING: Model is making very conservative predictions (< 0.1% returns)")
        
        return result
        
    except Exception as e:
        print(f"❌ LSTM Error: {str(e)}")
        return None

def diagnose_models():
    print("🚀 Starting Model Diagnostics")
    print("Testing both XGBoost and LSTM models with available tickers")
    
    # Test multiple tickers to see if issue is ticker-specific
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"TESTING {ticker}")
        print(f"{'='*60}")
        
        xgb_result = test_xgboost_model(ticker)
        lstm_result = test_lstm_model(ticker)
        
        # Summary for this ticker
        if xgb_result and lstm_result:
            xgb_returns = [p['predicted_return'] for p in xgb_result['predictions']]
            lstm_returns = [p['predicted_return'] for p in lstm_result['predictions']]
            
            print(f"\n📊 {ticker} Summary:")
            print(f"XGBoost avg return: {np.mean(xgb_returns)*100:.3f}%")
            print(f"LSTM avg return: {np.mean(lstm_returns)*100:.3f}%")
            
            if all(abs(r) < 0.001 for r in xgb_returns + lstm_returns):
                print(f"❌ Both models are too conservative for {ticker}")
            else:
                print(f"✅ At least one model is making meaningful predictions for {ticker}")

if __name__ == "__main__":
    diagnose_models()