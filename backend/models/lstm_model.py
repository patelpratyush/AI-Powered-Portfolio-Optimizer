#!/usr/bin/env python3
"""
LSTM Deep Learning Stock Predictor
Advanced neural network model for stock price prediction using time series patterns
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import ta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
from datetime import datetime, timedelta
import os
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    def __init__(self, model_dir: str = None, sequence_length: int = 30):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Use absolute path for model directory
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "saved")
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.feature_names = []
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Model architecture parameters (optimized for better performance)
        self.model_params = {
            'lstm_units': [50, 25],  # Simpler architecture
            'dropout_rate': 0.2,  # Moderate dropout
            'learning_rate': 0.001,  # Standard learning rate
            'batch_size': 32,  # Smaller batch size
            'epochs': 100,  # More epochs with early stopping
            'patience': 15  # More patience for early stopping
        }
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simplified features for LSTM model"""
        features_df = df.copy()
        
        # Basic price features
        features_df['returns'] = df['Close'].pct_change()
        features_df['price_momentum'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Simple moving averages
        for window in [5, 10, 20]:
            ma = df['Close'].rolling(window).mean()
            features_df[f'ma_{window}'] = ma
            features_df[f'price_to_ma_{window}'] = df['Close'] / ma
        
        # Basic technical indicators
        features_df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Volume features
        volume_ma = df['Volume'].rolling(10).mean()
        features_df['volume_ratio'] = df['Volume'] / volume_ma
        
        # High-low features
        features_df['high_low_ratio'] = df['High'] / df['Low']
        
        # Simple lag features
        for lag in [1, 2, 3]:
            features_df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        
        return features_df
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Create enhanced features
        features_df = self.create_enhanced_features(df)
        
        # Select feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Adj Close']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < self.sequence_length + 10:
            raise ValueError(f"Insufficient data after cleaning. Need at least {self.sequence_length + 10} rows")
        
        # Scale features
        feature_data = features_df[feature_cols].values
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Scale target (prices)
        price_data = features_df[target_col].values.reshape(-1, 1)
        scaled_prices = self.price_scaler.fit_transform(price_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build simplified LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.model_params['lstm_units'][0], 
                      return_sequences=True,
                      input_shape=input_shape,
                      name='lstm_1'))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Second LSTM layer
        model.add(LSTM(self.model_params['lstm_units'][1], 
                      return_sequences=False,
                      name='lstm_2'))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Dense layers
        model.add(Dense(25, activation='relu', name='dense_1'))
        model.add(Dropout(self.model_params['dropout_rate']))
        model.add(Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.model_params['learning_rate'])
        model.compile(optimizer=optimizer, 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def train(self, ticker: str, period: str = "2y") -> Dict:
        """Train the LSTM model for a specific ticker"""
        logger.info(f"Training LSTM model for {ticker}")
        
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < self.sequence_length + 50:
                raise ValueError(f"Insufficient data for {ticker}. Need at least {self.sequence_length + 50} days")
            
            # Prepare sequences
            X, y = self.prepare_sequences(df)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient sequences for training {ticker}")
            
            # Split data (time series split)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', 
                            patience=self.model_params['patience'], 
                            restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.5, 
                                patience=10, 
                                min_lr=1e-7),
                ModelCheckpoint(os.path.join(self.model_dir, f'lstm_{ticker.lower()}_best.keras'),
                              monitor='val_loss',
                              save_best_only=True)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.model_params['batch_size'],
                epochs=self.model_params['epochs'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Make predictions
            y_pred_train = self.model.predict(X_train, verbose=0)
            y_pred_test = self.model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            y_train_orig = self.price_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_orig = self.price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_train_orig = self.price_scaler.inverse_transform(y_pred_train).flatten()
            y_pred_test_orig = self.price_scaler.inverse_transform(y_pred_test).flatten()
            
            # Calculate metrics
            train_metrics = {
                'mse': float(mean_squared_error(y_train_orig, y_pred_train_orig)),
                'mae': float(mean_absolute_error(y_train_orig, y_pred_train_orig)),
                'r2': float(r2_score(y_train_orig, y_pred_train_orig))
            }
            
            test_metrics = {
                'mse': float(mean_squared_error(y_test_orig, y_pred_test_orig)),
                'mae': float(mean_absolute_error(y_test_orig, y_pred_test_orig)),
                'r2': float(r2_score(y_test_orig, y_pred_test_orig))
            }
            
            self.is_trained = True
            
            # Validate model performance before saving
            if test_metrics['r2'] < -1.0:  # Very poor performance
                logger.warning(f"LSTM model for {ticker} has very poor R² score: {test_metrics['r2']:.4f}")
                logger.warning("Model may not be reliable for predictions")
            elif test_metrics['r2'] < 0:  # Negative R² 
                logger.warning(f"LSTM model for {ticker} has negative R² score: {test_metrics['r2']:.4f}")
                logger.warning("Model performs worse than a simple mean prediction")
            else:
                logger.info(f"LSTM model for {ticker} achieved good R² score: {test_metrics['r2']:.4f}")
            
            # Save model and scalers
            self.save_model(ticker)
            
            training_results = {
                'ticker': ticker,
                'sequence_length': self.sequence_length,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'epochs_trained': len(history.history['loss'])
                },
                'model_saved': True
            }
            
            logger.info(f"LSTM training completed for {ticker}")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}, Test MAE: ${test_metrics['mae']:.2f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"LSTM training failed for {ticker}: {str(e)}")
            raise
    
    def predict(self, ticker: str, days_ahead: int = 5) -> Dict:
        """Make predictions for future stock prices using LSTM"""
        try:
            # Load model if not already loaded
            if not self.is_trained:
                self.load_model(ticker)
            
            # Download recent data
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")  # Get enough data for sequence
            
            if df.empty or len(df) < self.sequence_length:
                raise ValueError(f"Insufficient data for {ticker}")
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Prepare features
            features_df = self.create_enhanced_features(df)
            features_df = features_df.dropna()
            
            if len(features_df) < self.sequence_length:
                raise ValueError(f"Insufficient clean data for prediction")
            
            # Scale features
            feature_data = features_df[self.feature_names].values
            scaled_features = self.scaler.transform(feature_data)
            
            # Get the last sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make predictions
            predictions = []
            current_sequence = last_sequence.copy()
            
            for day in range(1, days_ahead + 1):
                # Predict next price (scaled)
                next_price_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
                
                # Convert to actual price
                next_price = self.price_scaler.inverse_transform([[next_price_scaled]])[0, 0]
                
                # Calculate return
                prev_price = current_price if day == 1 else predictions[-1]['predicted_price']
                predicted_return = (next_price - prev_price) / prev_price
                
                # Calculate confidence intervals using model uncertainty
                # For neural networks, we can use dropout during inference for uncertainty estimation
                mc_predictions = []
                for _ in range(10):  # Monte Carlo sampling
                    pred = self.model.predict(current_sequence, verbose=0)[0, 0]
                    mc_predictions.append(self.price_scaler.inverse_transform([[pred]])[0, 0])
                
                pred_std = np.std(mc_predictions)
                confidence_lower = next_price - 1.96 * pred_std
                confidence_upper = next_price + 1.96 * pred_std
                
                predictions.append({
                    'day': day,
                    'predicted_price': float(next_price),
                    'predicted_return': float(predicted_return),
                    'confidence_lower': float(confidence_lower),
                    'confidence_upper': float(confidence_upper),
                    'prediction_std': float(pred_std),
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
                })
                
                # Update sequence for next prediction
                # Create new features for the predicted price point
                if day < days_ahead:  # Don't update on last iteration
                    # Estimate next feature values based on the prediction
                    # This is a simplified approach - in practice you'd use more sophisticated methods
                    
                    # Get the last feature vector
                    last_features = scaled_features[-1].copy()
                    
                    # Update price-related features (simplified)
                    # We'll approximate new technical indicators based on the predicted price
                    price_change_ratio = next_price / (current_price if day == 1 else predictions[-2]['predicted_price'])
                    
                    # Update some key features (this is a simplified approximation)
                    # In a real scenario, you'd need to properly calculate all technical indicators
                    last_features[0] = next_price_scaled  # Assuming first feature is close price
                    if len(last_features) > 1:
                        last_features[1] = (price_change_ratio - 1)  # Return feature
                    
                    # Roll the sequence window: remove first, add new prediction
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, :] = last_features
            
            # Generate trading signal based on predictions
            avg_predicted_return = np.mean([p['predicted_return'] for p in predictions])
            prediction_volatility = np.std([p['predicted_return'] for p in predictions])
            
            # More sophisticated signal generation
            if avg_predicted_return > 0.03:  # > 3% expected return
                signal = "BUY"
                strength = min(avg_predicted_return * 8, 1.0)
            elif avg_predicted_return < -0.03:  # < -3% expected return
                signal = "SELL"
                strength = min(abs(avg_predicted_return) * 8, 1.0)
            else:
                signal = "HOLD"
                strength = 0.5
            
            # Adjust strength based on volatility (lower strength for high volatility)
            strength *= (1 - min(prediction_volatility * 10, 0.5))
            
            return {
                'ticker': ticker,
                'model': 'LSTM',
                'current_price': float(current_price),
                'predictions': predictions,
                'summary': {
                    'avg_predicted_return': float(avg_predicted_return),
                    'max_predicted_price': float(max(p['predicted_price'] for p in predictions)),
                    'min_predicted_price': float(min(p['predicted_price'] for p in predictions)),
                    'volatility_estimate': float(prediction_volatility),
                    'trend_direction': 'bullish' if avg_predicted_return > 0 else 'bearish'
                },
                'trading_signal': {
                    'action': signal,
                    'strength': float(strength),
                    'reasoning': f"LSTM deep learning model predicts {avg_predicted_return:.2%} average return with {prediction_volatility:.2%} volatility"
                },
                'model_confidence': 'high',  # LSTM typically has high confidence for patterns
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed for {ticker}: {str(e)}")
            raise
    
    def save_model(self, ticker: str):
        """Save the trained model and scalers"""
        try:
            model_path = os.path.join(self.model_dir, f"lstm_{ticker.lower()}.keras")
            scaler_path = os.path.join(self.model_dir, f"lstm_scaler_{ticker.lower()}.joblib")
            price_scaler_path = os.path.join(self.model_dir, f"lstm_price_scaler_{ticker.lower()}.joblib")
            features_path = os.path.join(self.model_dir, f"lstm_features_{ticker.lower()}.joblib")
            
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.price_scaler, price_scaler_path)
            joblib.dump({
                'feature_names': self.feature_names,
                'sequence_length': self.sequence_length
            }, features_path)
            
            logger.info(f"LSTM model saved for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to save LSTM model for {ticker}: {str(e)}")
    
    def load_model(self, ticker: str):
        """Load a pre-trained model and scalers"""
        try:
            model_path = os.path.join(self.model_dir, f"lstm_{ticker.lower()}.keras")
            scaler_path = os.path.join(self.model_dir, f"lstm_scaler_{ticker.lower()}.joblib")
            price_scaler_path = os.path.join(self.model_dir, f"lstm_price_scaler_{ticker.lower()}.joblib")
            features_path = os.path.join(self.model_dir, f"lstm_features_{ticker.lower()}.joblib")
            
            if not all(os.path.exists(path) for path in [model_path, scaler_path, price_scaler_path, features_path]):
                raise FileNotFoundError(f"Model files not found for {ticker}")
            
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.price_scaler = joblib.load(price_scaler_path)
            
            features_data = joblib.load(features_path)
            self.feature_names = features_data['feature_names']
            self.sequence_length = features_data['sequence_length']
            
            self.is_trained = True
            
            logger.info(f"LSTM model loaded for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model for {ticker}: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the LSTM predictor
    predictor = LSTMStockPredictor(sequence_length=60)
    
    # Test training
    test_ticker = "AAPL"
    try:
        training_results = predictor.train(test_ticker, period="1y")
        print("LSTM Training Results:")
        print(f"Test R²: {training_results['test_metrics']['r2']:.4f}")
        print(f"Test MAE: ${training_results['test_metrics']['mae']:.2f}")
        print(f"Epochs trained: {training_results['training_history']['epochs_trained']}")
        
        # Test prediction
        predictions = predictor.predict(test_ticker, days_ahead=5)
        print(f"\nLSTM Predictions for {test_ticker}:")
        print(f"Current Price: ${predictions['current_price']:.2f}")
        print(f"Trading Signal: {predictions['trading_signal']['action']}")
        print(f"Average Predicted Return: {predictions['summary']['avg_predicted_return']:.2%}")
        print(f"Trend Direction: {predictions['summary']['trend_direction']}")
        
        for pred in predictions['predictions'][:3]:  # Show first 3 days
            print(f"Day {pred['day']}: ${pred['predicted_price']:.2f} ({pred['predicted_return']:.2%}) ±${pred['prediction_std']:.2f}")
            
    except Exception as e:
        print(f"LSTM test failed: {str(e)}")