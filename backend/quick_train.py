#!/usr/bin/env python3
"""
Quick Training Script for Essential Stocks
Train XGBoost models for the most popular stocks to get started quickly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.xgb_model import XGBoostStockPredictor
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential stocks to train first (fast XGBoost training only)
ESSENTIAL_STOCKS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft  
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    'NVDA',   # NVIDIA
    'JPM',    # JPMorgan
    'JNJ',    # Johnson & Johnson
    'V',      # Visa
    'PG'      # Procter & Gamble
]

def train_essential_stocks():
    """Train XGBoost models for essential stocks"""
    logger.info("Starting quick training for essential stocks")
    logger.info(f"Training {len(ESSENTIAL_STOCKS)} stocks: {', '.join(ESSENTIAL_STOCKS)}")
    
    predictor = XGBoostStockPredictor()
    results = {}
    
    start_time = time.time()
    
    for i, ticker in enumerate(ESSENTIAL_STOCKS):
        logger.info(f"Training {ticker} ({i+1}/{len(ESSENTIAL_STOCKS)})")
        
        try:
            ticker_start = time.time()
            result = predictor.train(ticker, period="1y", target_days=1)
            ticker_time = time.time() - ticker_start
            
            results[ticker] = {
                'success': True,
                'training_time': ticker_time,
                'test_r2': result['test_metrics']['r2'],
                'test_mae': result['test_metrics']['mae'],
                'training_samples': result['training_samples']
            }
            
            logger.info(f"✅ {ticker} completed in {ticker_time:.1f}s - R²: {result['test_metrics']['r2']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ {ticker} failed: {str(e)}")
            results[ticker] = {
                'success': False,
                'error': str(e)
            }
        
        # Small delay to prevent system overload
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    
    # Print summary
    successful = [ticker for ticker, result in results.items() if result['success']]
    failed = [ticker for ticker, result in results.items() if not result['success']]
    
    logger.info("=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Successful: {len(successful)}/{len(ESSENTIAL_STOCKS)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info(f"✅ Successfully trained: {', '.join(successful)}")
        avg_r2 = sum(results[ticker]['test_r2'] for ticker in successful) / len(successful)
        logger.info(f"Average R² score: {avg_r2:.3f}")
    
    if failed:
        logger.info(f"❌ Failed to train: {', '.join(failed)}")
    
    logger.info("=" * 50)
    logger.info("You can now use these models for predictions!")
    logger.info("Example: GET /api/predict/AAPL?models=xgboost")
    logger.info("=" * 50)
    
    return results

if __name__ == "__main__":
    print("🚀 AI Portfolio Optimizer - Quick Model Training")
    print("This will train XGBoost models for 10 popular stocks (~5-10 minutes)")
    print()
    
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)
    
    try:
        results = train_essential_stocks()
        
        successful_count = sum(1 for result in results.values() if result['success'])
        if successful_count > 0:
            print(f"\n🎉 Training completed! {successful_count} models ready for predictions.")
            print("\nTry these examples:")
            print("- Visit the StockForecastChart component")
            print("- Use the prediction API: /api/predict/AAPL")
            print("- Check available models: /api/models/available")
        else:
            print("\n😞 No models were successfully trained. Check the logs for errors.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n💥 Training failed: {str(e)}")
        sys.exit(1)