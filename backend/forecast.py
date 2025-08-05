#!/usr/bin/env python3
"""
Simple Prophet-based forecasting for stock prices
Provides generate_ai_forecast function for the prediction API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_ai_forecast(tickers: List[str], period: str = "1y", days_ahead: int = 10) -> Dict:
    """
    Generate AI forecasts for multiple tickers using statistical methods
    Simplified version that doesn't require Prophet installation
    """
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Generating forecast for {ticker}")
            
            # Download stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 30:
                logger.warning(f"Insufficient data for {ticker}")
                results[ticker] = None
                continue
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Simple trend-based forecasting
            # Calculate recent trend (last 30 days)
            recent_prices = df['Close'].tail(30)
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]  # Linear trend
            
            # Calculate volatility
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Generate forecasts
            forecasts = {}
            base_date = datetime.now()
            
            for day in range(1, days_ahead + 1):
                forecast_date = (base_date + timedelta(days=day)).strftime('%Y-%m-%d')
                
                # Simple trend projection with noise
                trend_price = current_price + (trend * day)
                
                # Add some random walk component
                random_factor = np.random.normal(0, volatility * np.sqrt(day)) * current_price
                predicted_price = max(trend_price + random_factor, current_price * 0.5)  # Prevent negative prices
                
                # Simple confidence intervals (±2 standard deviations)
                price_std = volatility * current_price * np.sqrt(day)
                confidence_lower = max(predicted_price - 2 * price_std, current_price * 0.3)
                confidence_upper = predicted_price + 2 * price_std
                
                forecasts[forecast_date] = {
                    'value': float(predicted_price),
                    'lower': float(confidence_lower),
                    'upper': float(confidence_upper),
                    'trend': float(trend),
                    'volatility': float(volatility)
                }
            
            results[ticker] = forecasts
            logger.info(f"Forecast completed for {ticker}")
            
        except Exception as e:
            logger.error(f"Forecast failed for {ticker}: {str(e)}")
            results[ticker] = None
    
    return results

def get_simple_forecast(ticker: str, days_ahead: int = 10) -> Optional[Dict]:
    """
    Get a simple forecast for a single ticker
    """
    forecasts = generate_ai_forecast([ticker], days_ahead=days_ahead)
    return forecasts.get(ticker)

if __name__ == "__main__":
    # Test the forecasting
    test_tickers = ['AAPL', 'MSFT']
    
    print("Testing forecast generation...")
    results = generate_ai_forecast(test_tickers, period="6mo", days_ahead=5)
    
    for ticker, forecast in results.items():
        if forecast:
            print(f"\n{ticker} Forecast:")
            for date, values in list(forecast.items())[:3]:  # Show first 3 days
                print(f"  {date}: ${values['value']:.2f} (${values['lower']:.2f} - ${values['upper']:.2f})")
        else:
            print(f"\n{ticker}: Forecast failed")