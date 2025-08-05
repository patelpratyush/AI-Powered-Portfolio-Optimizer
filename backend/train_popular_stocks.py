#!/usr/bin/env python3
"""
Pre-train models for popular stocks
This script trains XGBoost and LSTM models for commonly traded stocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.xgb_model import XGBoostStockPredictor
from models.lstm_model import LSTMStockPredictor
import logging
import time
import concurrent.futures
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Popular stocks to pre-train
POPULAR_STOCKS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'DIS', 'NKE',
    
    # Other Popular
    'V', 'MA', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD'
]

# Training periods for different model types
TRAINING_PERIODS = {
    'xgboost': '1y',  # XGBoost needs less data, trains faster
    'lstm': '2y'      # LSTM benefits from more data
}

class ModelTrainer:
    def __init__(self):
        self.xgb_predictor = XGBoostStockPredictor()
        self.lstm_predictor = LSTMStockPredictor()
        self.results = {}
        self.failed_trainings = {}
    
    def train_xgboost(self, ticker: str) -> Dict:
        """Train XGBoost model for a ticker"""
        try:
            logger.info(f"Training XGBoost model for {ticker}")
            start_time = time.time()
            
            result = self.xgb_predictor.train(ticker, period=TRAINING_PERIODS['xgboost'])
            
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info(f"XGBoost training completed for {ticker} in {training_time:.1f}s")
            logger.info(f"Test R²: {result['test_metrics']['r2']:.4f}, Test MAE: {result['test_metrics']['mae']:.4f}")
            
            return {
                'success': True,
                'training_time': training_time,
                'model_type': 'xgboost',
                'ticker': ticker,
                'metrics': result['test_metrics'],
                'training_samples': result['training_samples'],
                'test_samples': result['test_samples']
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed for {ticker}: {str(e)}")
            return {
                'success': False,
                'model_type': 'xgboost',
                'ticker': ticker,
                'error': str(e)
            }
    
    def train_lstm(self, ticker: str) -> Dict:
        """Train LSTM model for a ticker"""
        try:
            logger.info(f"Training LSTM model for {ticker}")
            start_time = time.time()
            
            result = self.lstm_predictor.train(ticker, period=TRAINING_PERIODS['lstm'])
            
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info(f"LSTM training completed for {ticker} in {training_time:.1f}s")
            logger.info(f"Test R²: {result['test_metrics']['r2']:.4f}, Test MAE: ${result['test_metrics']['mae']:.2f}")
            
            return {
                'success': True,
                'training_time': training_time,
                'model_type': 'lstm',
                'ticker': ticker,
                'metrics': result['test_metrics'],
                'training_samples': result['training_samples'],
                'test_samples': result['test_samples'],
                'epochs_trained': result['training_history']['epochs_trained']
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed for {ticker}: {str(e)}")
            return {
                'success': False,
                'model_type': 'lstm',
                'ticker': ticker,
                'error': str(e)
            }
    
    def train_ticker_models(self, ticker: str, models: List[str] = ['xgboost', 'lstm']) -> Dict:
        """Train all specified models for a ticker"""
        ticker_results = {}
        
        for model_type in models:
            if model_type == 'xgboost':
                result = self.train_xgboost(ticker)
            elif model_type == 'lstm':
                result = self.train_lstm(ticker)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            ticker_results[model_type] = result
            
            if not result['success']:
                if ticker not in self.failed_trainings:
                    self.failed_trainings[ticker] = []
                self.failed_trainings[ticker].append(result)
        
        return ticker_results
    
    def train_batch_sequential(self, tickers: List[str], models: List[str] = ['xgboost', 'lstm']):
        """Train models sequentially (safer for memory)"""
        logger.info(f"Starting sequential training for {len(tickers)} tickers with models: {models}")
        
        total_start_time = time.time()
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Training {ticker} ({i+1}/{len(tickers)})")
            
            ticker_results = self.train_ticker_models(ticker, models)
            self.results[ticker] = ticker_results
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch training completed in {total_time:.1f}s")
        
        return self.results
    
    def train_batch_parallel(self, tickers: List[str], models: List[str] = ['xgboost'], max_workers: int = 3):
        """Train models in parallel (faster but more memory intensive)"""
        logger.info(f"Starting parallel training for {len(tickers)} tickers with models: {models}")
        
        total_start_time = time.time()
        
        # Only parallelize XGBoost training - LSTM is too memory intensive
        if 'lstm' in models and 'xgboost' in models:
            logger.info("LSTM detected - using sequential training for memory safety")
            return self.train_batch_sequential(tickers, models)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}
            
            for ticker in tickers:
                future = executor.submit(self.train_ticker_models, ticker, models)
                future_to_ticker[future] = ticker
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=1800)  # 30 minute timeout per ticker
                    self.results[ticker] = result
                    logger.info(f"Completed training for {ticker}")
                except Exception as e:
                    logger.error(f"Parallel training failed for {ticker}: {str(e)}")
                    self.results[ticker] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        logger.info(f"Parallel training completed in {total_time:.1f}s")
        
        return self.results
    
    def save_results(self, filename: str = 'training_results.json'):
        """Save training results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'results': self.results,
                    'failed_trainings': self.failed_trainings,
                    'timestamp': time.time(),
                    'total_tickers': len(self.results),
                    'successful_trainings': len([r for r in self.results.values() if 'error' not in r]),
                    'failed_trainings_count': len(self.failed_trainings)
                }, f, indent=2)
            
            logger.info(f"Training results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def print_summary(self):
        """Print training summary"""
        logger.info("=== Training Summary ===")
        logger.info(f"Total tickers processed: {len(self.results)}")
        
        successful_xgb = 0
        successful_lstm = 0
        failed_xgb = 0
        failed_lstm = 0
        
        for ticker, ticker_results in self.results.items():
            if 'xgboost' in ticker_results:
                if ticker_results['xgboost']['success']:
                    successful_xgb += 1
                else:
                    failed_xgb += 1
            
            if 'lstm' in ticker_results:
                if ticker_results['lstm']['success']:
                    successful_lstm += 1
                else:
                    failed_lstm += 1
        
        logger.info(f"XGBoost: {successful_xgb} successful, {failed_xgb} failed")
        logger.info(f"LSTM: {successful_lstm} successful, {failed_lstm} failed")
        
        if self.failed_trainings:
            logger.info("Failed trainings:")
            for ticker, failures in self.failed_trainings.items():
                for failure in failures:
                    logger.info(f"  {ticker} ({failure['model_type']}): {failure['error']}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models for popular stocks')
    parser.add_argument('--models', nargs='+', default=['xgboost'], 
                       choices=['xgboost', 'lstm'], 
                       help='Models to train (default: xgboost)')
    parser.add_argument('--tickers', nargs='+', default=POPULAR_STOCKS[:10],
                       help='Tickers to train (default: top 10 popular stocks)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel training (faster but more memory)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum parallel workers (default: 3)')
    parser.add_argument('--output', default='training_results.json',
                       help='Output file for results (default: training_results.json)')
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    logger.info(f"Models: {args.models}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Parallel: {args.parallel}")
    
    trainer = ModelTrainer()
    
    try:
        if args.parallel and 'lstm' not in args.models:
            results = trainer.train_batch_parallel(args.tickers, args.models, args.max_workers)
        else:
            results = trainer.train_batch_sequential(args.tickers, args.models)
        
        trainer.print_summary()
        trainer.save_results(args.output)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.print_summary()
        trainer.save_results(f"interrupted_{args.output}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        trainer.print_summary()
        trainer.save_results(f"failed_{args.output}")
        raise

if __name__ == "__main__":
    main()
