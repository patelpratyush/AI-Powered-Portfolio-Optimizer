#!/usr/bin/env python3
"""
Quick test to see if we can load the model files directly
"""

import os
import joblib

def test_file_loading():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models/saved")
    ticker = "aapl"
    
    model_path = os.path.join(model_dir, f"xgb_{ticker}.joblib")
    scaler_path = os.path.join(model_dir, f"xgb_scaler_{ticker}.joblib")
    features_path = os.path.join(model_dir, f"xgb_features_{ticker}.joblib")
    
    print(f"Checking files for {ticker}:")
    print(f"Model path: {model_path} - Exists: {os.path.exists(model_path)}")
    print(f"Scaler path: {scaler_path} - Exists: {os.path.exists(scaler_path)}")
    print(f"Features path: {features_path} - Exists: {os.path.exists(features_path)}")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully: {type(model)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    if os.path.exists(features_path):
        try:
            features = joblib.load(features_path)
            print(f"✅ Features loaded successfully: {len(features)} features")
            print(f"Feature names: {features[:5]}...")  # First 5 features
        except Exception as e:
            print(f"❌ Error loading features: {e}")

if __name__ == "__main__":
    test_file_loading()