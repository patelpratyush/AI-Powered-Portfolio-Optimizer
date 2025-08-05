#!/usr/bin/env python3
"""
Portfolio Import and Validation Module
Handles CSV parsing, ticker validation, and market data enrichment
"""

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Blueprint, request, jsonify
import io
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import concurrent.futures
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
import_bp = Blueprint('import_portfolio', __name__)

class PortfolioValidator:
    def __init__(self):
        self.valid_exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        self.cache = {}  # Simple in-memory cache for market data
        self.cache_expiry = 300  # 5 minutes cache
        
    def validate_ticker_format(self, ticker: str) -> bool:
        """Validate ticker symbol format"""
        if not ticker or not isinstance(ticker, str):
            return False
        
        # Remove whitespace and convert to uppercase
        ticker = ticker.strip().upper()
        
        # Basic format validation (1-5 characters, letters only for most cases)
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            # Allow some special cases like BRK.A, BRK.B
            if not re.match(r'^[A-Z]{1,4}\.[A-Z]$', ticker):
                return False
        
        return True
    
    def get_market_data_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get market data for multiple tickers efficiently"""
        market_data = {}
        valid_tickers = []
        invalid_tickers = []
        
        # Filter and validate tickers
        for ticker in tickers:
            if self.validate_ticker_format(ticker):
                ticker_clean = ticker.strip().upper()
                
                # Check cache first
                cache_key = f"{ticker_clean}_{int(time.time() // self.cache_expiry)}"
                if cache_key in self.cache:
                    market_data[ticker_clean] = self.cache[cache_key]
                    valid_tickers.append(ticker_clean)
                else:
                    valid_tickers.append(ticker_clean)
            else:
                invalid_tickers.append(ticker)
        
        # Batch download market data for uncached tickers
        uncached_tickers = [t for t in valid_tickers if t not in market_data]
        
        if uncached_tickers:
            try:
                # Use threading for faster API calls
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_ticker = {
                        executor.submit(self._get_single_ticker_data, ticker): ticker 
                        for ticker in uncached_tickers
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            data = future.result(timeout=10)  # 10 second timeout per ticker
                            if data:
                                market_data[ticker] = data
                                # Cache the result
                                cache_key = f"{ticker}_{int(time.time() // self.cache_expiry)}"
                                self.cache[cache_key] = data
                            else:
                                invalid_tickers.append(ticker)
                        except Exception as e:
                            logger.warning(f"Failed to get data for {ticker}: {str(e)}")
                            invalid_tickers.append(ticker)
                            
            except Exception as e:
                logger.error(f"Batch market data fetch failed: {str(e)}")
                invalid_tickers.extend(uncached_tickers)
        
        return market_data, invalid_tickers
    
    def _get_single_ticker_data(self, ticker: str) -> Optional[Dict]:
        """Get market data for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current price and basic info
            info = stock.info
            hist = stock.history(period="5d")  # Get recent history for validation
            
            if hist.empty or not info:
                return None
            
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            if current_price is None or np.isnan(current_price):
                return None
            
            return {
                'price': float(current_price),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'volume': float(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                'prev_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price,
                'day_change': float(current_price - hist['Close'].iloc[-2]) if len(hist) > 1 else 0,
                'day_change_percent': float((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def validate_portfolio_data(self, holdings: List[Dict]) -> Dict:
        """Validate portfolio holdings data"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_holdings': [],
            'summary': {}
        }
        
        if not holdings:
            validation_result['valid'] = False
            validation_result['errors'].append('No holdings provided')
            return validation_result
        
        # Extract tickers for batch validation
        tickers = []
        for holding in holdings:
            ticker = holding.get('ticker', '').strip().upper()
            if ticker:
                tickers.append(ticker)
        
        # Get market data
        market_data, invalid_tickers = self.get_market_data_batch(tickers)
        
        total_value = 0
        total_cost_basis = 0
        sector_allocation = {}
        
        for i, holding in enumerate(holdings):
            validated_holding = holding.copy()
            errors = []
            warnings = []
            
            # Validate required fields
            ticker = holding.get('ticker', '').strip().upper()
            shares = holding.get('shares', 0)
            avg_price = holding.get('avgPrice', 0)
            
            if not ticker:
                errors.append(f"Row {i+1}: Ticker is required")
            elif ticker in invalid_tickers:
                errors.append(f"Row {i+1}: Invalid or unknown ticker '{ticker}'")
            
            try:
                shares = float(shares)
                if shares <= 0:
                    errors.append(f"Row {i+1}: Shares must be greater than 0")
            except (ValueError, TypeError):
                errors.append(f"Row {i+1}: Invalid shares value")
                shares = 0
            
            try:
                avg_price = float(avg_price)
                if avg_price <= 0:
                    errors.append(f"Row {i+1}: Average price must be greater than 0")
            except (ValueError, TypeError):
                errors.append(f"Row {i+1}: Invalid average price value")
                avg_price = 0
            
            # Enrich with market data if available
            if ticker in market_data:
                market_info = market_data[ticker]
                current_price = market_info['price']
                current_value = shares * current_price
                cost_basis = shares * avg_price
                
                validated_holding.update({
                    'ticker': ticker,
                    'shares': shares,
                    'avgPrice': avg_price,
                    'marketPrice': current_price,
                    'currentValue': current_value,
                    'costBasis': cost_basis,
                    'gainLoss': current_value - cost_basis,
                    'gainLossPercent': ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0,
                    'dayChange': market_info['day_change'],
                    'dayChangePercent': market_info['day_change_percent'],
                    'sector': market_info['sector'],
                    'industry': market_info['industry'],
                    'currency': market_info['currency'],
                    'volume': market_info['volume']
                })
                
                total_value += current_value
                total_cost_basis += cost_basis
                
                # Track sector allocation
                sector = market_info['sector']
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += current_value
                
                # Generate warnings for significant changes
                price_diff_percent = abs((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
                if price_diff_percent > 50:
                    warnings.append(f"Row {i+1}: Large price difference ({price_diff_percent:.1f}%) between average price and current market price")
            
            if errors:
                validation_result['errors'].extend(errors)
                validation_result['valid'] = False
            
            if warnings:
                validation_result['warnings'].extend(warnings)
            
            validation_result['validated_holdings'].append(validated_holding)
        
        # Generate portfolio summary
        if total_value > 0:
            total_gain_loss = total_value - total_cost_basis
            sector_percentages = {k: (v / total_value * 100) for k, v in sector_allocation.items()}
            
            validation_result['summary'] = {
                'total_positions': len([h for h in validation_result['validated_holdings'] if h.get('ticker')]),
                'total_value': total_value,
                'total_cost_basis': total_cost_basis,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': (total_gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0,
                'sector_allocation': sector_percentages,
                'largest_position': max(validation_result['validated_holdings'], 
                                      key=lambda x: x.get('currentValue', 0), 
                                      default={}).get('ticker', 'N/A'),
                'diversification_score': self._calculate_diversification_score(validation_result['validated_holdings'])
            }
        
        return validation_result
    
    def _calculate_diversification_score(self, holdings: List[Dict]) -> float:
        """Calculate a simple diversification score (0-100)"""
        if not holdings:
            return 0
        
        total_value = sum(h.get('currentValue', 0) for h in holdings)
        if total_value == 0:
            return 0
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        weights = [h.get('currentValue', 0) / total_value for h in holdings]
        hhi = sum(w**2 for w in weights)
        
        # Convert to diversification score (lower HHI = higher diversification)
        # Perfect diversification (equal weights) would give HHI = 1/n
        n = len(holdings)
        min_hhi = 1.0 / n  # Perfect diversification
        max_hhi = 1.0      # Complete concentration
        
        # Normalize to 0-100 scale
        diversification_score = (1 - (hhi - min_hhi) / (max_hhi - min_hhi)) * 100
        return max(0, min(100, diversification_score))

# Initialize validator
validator = PortfolioValidator()

@import_bp.route('/validate-portfolio', methods=['POST'])
def validate_portfolio():
    """Validate portfolio holdings and enrich with market data"""
    try:
        data = request.get_json()
        
        if not data or 'holdings' not in data:
            return jsonify({
                'error': 'Missing holdings data',
                'valid': False
            }), 400
        
        holdings = data['holdings']
        
        # Validate the portfolio
        validation_result = validator.validate_portfolio_data(holdings)
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error(f"Portfolio validation error: {str(e)}")
        return jsonify({
            'error': f'Validation failed: {str(e)}',
            'valid': False
        }), 500

@import_bp.route('/parse-csv', methods=['POST'])
def parse_csv():
    """Parse CSV file and extract portfolio holdings"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV content
        csv_content = file.read().decode('utf-8')
        
        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(csv_content))
        except Exception as e:
            return jsonify({'error': f'CSV parsing failed: {str(e)}'}), 400
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Map common column variations
        column_mapping = {
            'symbol': 'ticker',
            'stock': 'ticker',
            'quantity': 'shares',
            'qty': 'shares',
            'price': 'avgPrice',
            'avg_price': 'avgPrice',
            'average_price': 'avgPrice',
            'cost_basis': 'avgPrice'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Validate required columns
        required_columns = ['ticker', 'shares', 'avgPrice']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'available_columns': list(df.columns),
                'required_columns': required_columns
            }), 400
        
        # Convert to holdings format
        holdings = []
        for _, row in df.iterrows():
            holding = {
                'ticker': str(row.get('ticker', '')).strip().upper(),
                'shares': float(row.get('shares', 0)) if pd.notna(row.get('shares')) else 0,
                'avgPrice': float(row.get('avgPrice', 0)) if pd.notna(row.get('avgPrice')) else 0
            }
            
            # Only add if ticker is not empty
            if holding['ticker']:
                holdings.append(holding)
        
        return jsonify({
            'holdings': holdings,
            'total_positions': len(holdings),
            'message': f'Successfully parsed {len(holdings)} positions from CSV'
        })
        
    except Exception as e:
        logger.error(f"CSV parsing error: {str(e)}")
        return jsonify({'error': f'CSV parsing failed: {str(e)}'}), 500

@import_bp.route('/portfolio-template', methods=['GET'])
def get_portfolio_template():
    """Download portfolio CSV template"""
    template_data = [
        {'ticker': 'AAPL', 'shares': 100, 'avgPrice': 150.00},
        {'ticker': 'GOOGL', 'shares': 50, 'avgPrice': 2500.00},
        {'ticker': 'MSFT', 'shares': 75, 'avgPrice': 300.00},
        {'ticker': 'TSLA', 'shares': 25, 'avgPrice': 800.00},
        {'ticker': 'AMZN', 'shares': 30, 'avgPrice': 3200.00}
    ]
    
    # Create CSV content
    csv_content = "ticker,shares,avgPrice\n"
    for item in template_data:
        csv_content += f"{item['ticker']},{item['shares']},{item['avgPrice']:.2f}\n"
    
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=portfolio_template.csv'
    }

if __name__ == '__main__':
    # Test the validator
    test_holdings = [
        {'ticker': 'AAPL', 'shares': 100, 'avgPrice': 150.00},
        {'ticker': 'GOOGL', 'shares': 50, 'avgPrice': 2500.00},
        {'ticker': 'INVALID', 'shares': 25, 'avgPrice': 100.00}
    ]
    
    result = validator.validate_portfolio_data(test_holdings)
    print("Validation Result:")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    if result['summary']:
        print(f"Total Value: ${result['summary']['total_value']:,.2f}")
        print(f"Total Gain/Loss: ${result['summary']['total_gain_loss']:,.2f}")