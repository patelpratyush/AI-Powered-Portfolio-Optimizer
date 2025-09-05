#!/usr/bin/env python3
"""
Alternative Assets Integration
Support for cryptocurrency, commodities, bonds, and other alternative investments
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('portfolio_optimizer.alternative_assets')

@dataclass
class AssetInfo:
    """Information about an alternative asset"""
    symbol: str
    name: str
    asset_type: str  # 'crypto', 'commodity', 'bond', 'reit', 'etf'
    category: str  # e.g., 'precious_metals', 'energy', 'government_bond'
    currency: str
    market_cap: Optional[float] = None
    trading_volume: Optional[float] = None
    expense_ratio: Optional[float] = None  # For ETFs
    yield_rate: Optional[float] = None  # For bonds
    description: str = ""

@dataclass
class CryptoMetrics:
    """Cryptocurrency-specific metrics"""
    market_cap_rank: int
    circulating_supply: float
    total_supply: float
    max_supply: Optional[float]
    all_time_high: float
    all_time_low: float
    price_change_24h: float
    volume_24h: float
    fear_greed_index: Optional[int] = None

class AlternativeAssetsManager:
    """Manage alternative asset data and analysis"""
    
    def __init__(self, cache_client=None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.alternative_assets.manager')
        
        # API keys
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Asset mappings
        self.asset_categories = self._initialize_asset_categories()
    
    def _initialize_asset_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize asset categories and their mappings"""
        return {
            'cryptocurrency': {
                'major_coins': {
                    'BTC-USD': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin'},
                    'ETH-USD': {'name': 'Ethereum', 'coingecko_id': 'ethereum'},
                    'BNB-USD': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin'},
                    'XRP-USD': {'name': 'Ripple', 'coingecko_id': 'ripple'},
                    'ADA-USD': {'name': 'Cardano', 'coingecko_id': 'cardano'},
                    'DOGE-USD': {'name': 'Dogecoin', 'coingecko_id': 'dogecoin'},
                    'MATIC-USD': {'name': 'Polygon', 'coingecko_id': 'matic-network'},
                    'SOL-USD': {'name': 'Solana', 'coingecko_id': 'solana'},
                    'DOT-USD': {'name': 'Polkadot', 'coingecko_id': 'polkadot'},
                    'AVAX-USD': {'name': 'Avalanche', 'coingecko_id': 'avalanche-2'}
                },
                'defi_tokens': {
                    'UNI-USD': {'name': 'Uniswap', 'coingecko_id': 'uniswap'},
                    'LINK-USD': {'name': 'Chainlink', 'coingecko_id': 'chainlink'},
                    'AAVE-USD': {'name': 'Aave', 'coingecko_id': 'aave'},
                    'SUSHI-USD': {'name': 'SushiSwap', 'coingecko_id': 'sushi'},
                    'CRV-USD': {'name': 'Curve DAO', 'coingecko_id': 'curve-dao-token'}
                }
            },
            'commodities': {
                'precious_metals': {
                    'GLD': {'name': 'Gold ETF', 'underlying': 'Gold'},
                    'SLV': {'name': 'Silver ETF', 'underlying': 'Silver'},
                    'PPLT': {'name': 'Platinum ETF', 'underlying': 'Platinum'},
                    'PALL': {'name': 'Palladium ETF', 'underlying': 'Palladium'}
                },
                'energy': {
                    'USO': {'name': 'Oil ETF', 'underlying': 'Crude Oil'},
                    'UNG': {'name': 'Natural Gas ETF', 'underlying': 'Natural Gas'},
                    'BNO': {'name': 'Brent Oil ETF', 'underlying': 'Brent Crude'},
                    'ICLN': {'name': 'Clean Energy ETF', 'underlying': 'Renewable Energy'}
                },
                'agriculture': {
                    'DBA': {'name': 'Agriculture ETF', 'underlying': 'Agricultural Commodities'},
                    'CORN': {'name': 'Corn ETF', 'underlying': 'Corn'},
                    'WEAT': {'name': 'Wheat ETF', 'underlying': 'Wheat'},
                    'SOYB': {'name': 'Soybean ETF', 'underlying': 'Soybeans'}
                }
            },
            'bonds': {
                'government': {
                    'TLT': {'name': '20+ Year Treasury Bond ETF', 'duration': 'long', 'credit': 'AAA'},
                    'IEF': {'name': '7-10 Year Treasury ETF', 'duration': 'intermediate', 'credit': 'AAA'},
                    'SHY': {'name': '1-3 Year Treasury ETF', 'duration': 'short', 'credit': 'AAA'},
                    'TIPS': {'name': 'TIPS ETF', 'duration': 'mixed', 'credit': 'AAA', 'inflation_protected': True}
                },
                'corporate': {
                    'LQD': {'name': 'Investment Grade Corporate Bond ETF', 'credit': 'A+'},
                    'HYG': {'name': 'High Yield Corporate Bond ETF', 'credit': 'BB'},
                    'JNK': {'name': 'High Yield ETF', 'credit': 'B'},
                    'VCIT': {'name': 'Intermediate Corporate Bond ETF', 'credit': 'A'}
                },
                'international': {
                    'BNDX': {'name': 'International Bond ETF', 'region': 'Developed'},
                    'EMB': {'name': 'Emerging Markets Bond ETF', 'region': 'Emerging'},
                    'VTEB': {'name': 'Tax-Exempt Bond ETF', 'tax_status': 'exempt'}
                }
            },
            'real_estate': {
                'reit_etfs': {
                    'VNQ': {'name': 'Real Estate ETF', 'focus': 'broad'},
                    'IYR': {'name': 'Real Estate ETF', 'focus': 'broad'},
                    'XLRE': {'name': 'Real Estate Select Sector SPDR', 'focus': 'broad'},
                    'RWR': {'name': 'SPDR DJ Wilshire REIT ETF', 'focus': 'broad'}
                },
                'sector_specific': {
                    'FREL': {'name': 'Residential REIT ETF', 'focus': 'residential'},
                    'PLD': {'name': 'Prologis Industrial REIT', 'focus': 'industrial'},
                    'AMT': {'name': 'American Tower REIT', 'focus': 'infrastructure'},
                    'O': {'name': 'Realty Income REIT', 'focus': 'commercial'}
                }
            },
            'alternative_strategies': {
                'volatility': {
                    'VIX': {'name': 'VIX Volatility Index', 'description': 'Market volatility measure'},
                    'UVXY': {'name': 'Ultra VIX Short-Term Futures ETF', 'leverage': '2x'},
                    'SVXY': {'name': 'Short VIX Short-Term Futures ETF', 'direction': 'inverse'}
                },
                'currencies': {
                    'UUP': {'name': 'Dollar Index ETF', 'underlying': 'US Dollar'},
                    'FXE': {'name': 'Euro ETF', 'underlying': 'EUR'},
                    'FXY': {'name': 'Yen ETF', 'underlying': 'JPY'},
                    'FXB': {'name': 'British Pound ETF', 'underlying': 'GBP'}
                }
            }
        }
    
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get detailed information about an asset"""
        # Search through all categories
        for category, subcategories in self.asset_categories.items():
            for subcategory, assets in subcategories.items():
                if symbol in assets:
                    asset_data = assets[symbol]
                    return AssetInfo(
                        symbol=symbol,
                        name=asset_data['name'],
                        asset_type=category,
                        category=subcategory,
                        currency='USD',  # Most are USD-denominated
                        description=asset_data.get('description', '')
                    )
        
        return None
    
    def get_crypto_data(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Get cryptocurrency price data"""
        crypto_data = {}
        
        # Get data from Yahoo Finance (most reliable for price history)
        for symbol in symbols:
            try:
                # Ensure symbol has -USD suffix for Yahoo Finance
                if not symbol.endswith('-USD'):
                    yf_symbol = f"{symbol}-USD"
                else:
                    yf_symbol = symbol
                
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    crypto_data[symbol] = hist
                    self.logger.info(f"Retrieved {len(hist)} days of data for {symbol}")
                else:
                    self.logger.warning(f"No data available for {yf_symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching crypto data for {symbol}: {e}")
        
        return crypto_data
    
    def get_crypto_metrics(self, coin_id: str) -> Optional[CryptoMetrics]:
        """Get detailed crypto metrics from CoinGecko API"""
        if not self.coingecko_api_key:
            self.logger.warning("CoinGecko API key not configured")
            return None
        
        try:
            # CoinGecko API call
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {'x_cg_demo_api_key': self.coingecko_api_key} if self.coingecko_api_key else {}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            market_data = data.get('market_data', {})
            
            return CryptoMetrics(
                market_cap_rank=data.get('market_cap_rank', 0),
                circulating_supply=market_data.get('circulating_supply', 0),
                total_supply=market_data.get('total_supply', 0),
                max_supply=market_data.get('max_supply'),
                all_time_high=market_data.get('ath', {}).get('usd', 0),
                all_time_low=market_data.get('atl', {}).get('usd', 0),
                price_change_24h=market_data.get('price_change_percentage_24h', 0),
                volume_24h=market_data.get('total_volume', {}).get('usd', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto metrics for {coin_id}: {e}")
            return None
    
    def get_commodity_data(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Get commodity ETF data"""
        commodity_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    commodity_data[symbol] = hist
                    self.logger.info(f"Retrieved commodity data for {symbol}")
                else:
                    self.logger.warning(f"No data available for commodity {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching commodity data for {symbol}: {e}")
        
        return commodity_data
    
    def get_bond_data(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Get bond ETF data with yield information"""
        bond_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    # Add yield information if available
                    hist['yield'] = info.get('yield', np.nan)
                    hist['duration'] = info.get('duration', np.nan)
                    bond_data[symbol] = hist
                    self.logger.info(f"Retrieved bond data for {symbol}")
                else:
                    self.logger.warning(f"No data available for bond {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching bond data for {symbol}: {e}")
        
        return bond_data
    
    def get_reit_data(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Get REIT data"""
        reit_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    # Add REIT-specific metrics
                    hist['dividend_yield'] = info.get('dividendYield', np.nan)
                    hist['funds_from_operations'] = info.get('ffo', np.nan)
                    reit_data[symbol] = hist
                    self.logger.info(f"Retrieved REIT data for {symbol}")
                else:
                    self.logger.warning(f"No data available for REIT {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching REIT data for {symbol}: {e}")
        
        return reit_data
    
    def get_alternative_universe(self, categories: List[str] = None) -> Dict[str, List[str]]:
        """Get universe of alternative assets by category"""
        if categories is None:
            categories = list(self.asset_categories.keys())
        
        universe = {}
        for category in categories:
            if category in self.asset_categories:
                assets = []
                for subcategory, asset_dict in self.asset_categories[category].items():
                    assets.extend(list(asset_dict.keys()))
                universe[category] = assets
        
        return universe
    
    def analyze_alternative_correlation(self, 
                                     alternative_data: Dict[str, pd.DataFrame],
                                     traditional_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze correlation between alternative and traditional assets"""
        
        # Combine all data
        all_data = {}
        all_data.update(alternative_data)
        if traditional_data:
            all_data.update(traditional_data)
        
        if len(all_data) < 2:
            return {'error': 'Need at least 2 assets for correlation analysis'}
        
        # Extract closing prices and calculate returns
        price_data = {}
        for symbol, data in all_data.items():
            if 'Close' in data.columns:
                price_data[symbol] = data['Close']
        
        if not price_data:
            return {'error': 'No valid price data found'}
        
        # Combine into DataFrame
        combined_prices = pd.DataFrame(price_data).dropna()
        returns = combined_prices.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Identify asset types
        alternative_assets = list(alternative_data.keys())
        traditional_assets = list(traditional_data.keys()) if traditional_data else []
        
        # Calculate average correlations
        analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'summary': {
                'alternative_assets': alternative_assets,
                'traditional_assets': traditional_assets,
                'period_analyzed': f"{returns.index[0].date()} to {returns.index[-1].date()}",
                'data_points': len(returns)
            },
            'insights': []
        }
        
        # Generate insights
        if traditional_assets and alternative_assets:
            # Average correlation between alternative and traditional
            alt_trad_corrs = []
            for alt in alternative_assets:
                for trad in traditional_assets:
                    if alt in correlation_matrix.index and trad in correlation_matrix.columns:
                        corr = correlation_matrix.loc[alt, trad]
                        if not pd.isna(corr):
                            alt_trad_corrs.append(abs(corr))
            
            if alt_trad_corrs:
                avg_correlation = np.mean(alt_trad_corrs)
                analysis['insights'].append({
                    'type': 'diversification_benefit',
                    'metric': 'average_correlation',
                    'value': avg_correlation,
                    'interpretation': _interpret_correlation(avg_correlation)
                })
        
        # Find highly correlated pairs (potential redundancy)
        high_corr_pairs = []
        for i, asset1 in enumerate(correlation_matrix.index):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[asset1, asset2]
                    if not pd.isna(corr) and abs(corr) > 0.8:
                        high_corr_pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': corr
                        })
        
        if high_corr_pairs:
            analysis['high_correlation_pairs'] = high_corr_pairs
            analysis['insights'].append({
                'type': 'redundancy_warning',
                'count': len(high_corr_pairs),
                'message': 'Some assets are highly correlated and may not provide diversification benefits'
            })
        
        return analysis
    
    def calculate_alternative_risk_metrics(self, 
                                         alternative_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics specific to alternative assets"""
        
        risk_metrics = {}
        
        for symbol, data in alternative_data.items():
            if 'Close' not in data.columns:
                continue
            
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 30:
                continue
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95%)
            var_95 = returns.quantile(0.05)
            
            # Tail risk (average of worst 5% returns)
            tail_threshold = returns.quantile(0.05)
            tail_returns = returns[returns <= tail_threshold]
            tail_risk = tail_returns.mean() if len(tail_returns) > 0 else var_95
            
            asset_info = self.get_asset_info(symbol)
            
            risk_metrics[symbol] = {
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'downside_deviation': downside_deviation,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'tail_risk': tail_risk,
                'asset_type': asset_info.asset_type if asset_info else 'unknown',
                'category': asset_info.category if asset_info else 'unknown'
            }
        
        return risk_metrics
    
    def suggest_alternative_allocation(self, 
                                     traditional_portfolio: Dict[str, float],
                                     risk_tolerance: str = 'moderate',
                                     investment_goals: List[str] = None) -> Dict[str, Any]:
        """Suggest alternative asset allocation based on traditional portfolio"""
        
        if investment_goals is None:
            investment_goals = ['diversification']
        
        suggestions = {
            'recommended_allocation': {},
            'rationale': {},
            'asset_suggestions': {},
            'total_alternative_percent': 0
        }
        
        # Base allocation percentages by risk tolerance
        allocation_ranges = {
            'conservative': {'min': 5, 'max': 15},
            'moderate': {'min': 10, 'max': 25},
            'aggressive': {'min': 15, 'max': 40}
        }
        
        target_range = allocation_ranges.get(risk_tolerance, allocation_ranges['moderate'])
        
        # Suggest allocations based on goals
        if 'diversification' in investment_goals:
            suggestions['recommended_allocation']['commodities'] = 5
            suggestions['recommended_allocation']['real_estate'] = 5
            suggestions['rationale']['diversification'] = 'Commodities and REITs provide diversification from traditional stocks and bonds'
        
        if 'inflation_hedge' in investment_goals:
            suggestions['recommended_allocation']['commodities'] = 8
            suggestions['recommended_allocation']['real_estate'] = 7
            suggestions['recommended_allocation']['tips'] = 5
            suggestions['rationale']['inflation_hedge'] = 'Real assets and TIPS provide inflation protection'
        
        if 'growth' in investment_goals and risk_tolerance in ['moderate', 'aggressive']:
            suggestions['recommended_allocation']['cryptocurrency'] = 3 if risk_tolerance == 'moderate' else 8
            suggestions['rationale']['growth'] = 'Cryptocurrency provides high growth potential with high risk'
        
        if 'income' in investment_goals:
            suggestions['recommended_allocation']['real_estate'] = 10
            suggestions['recommended_allocation']['high_yield_bonds'] = 5
            suggestions['rationale']['income'] = 'REITs and high-yield bonds provide regular income'
        
        # Specific asset suggestions
        suggestions['asset_suggestions'] = {
            'commodities': ['GLD', 'SLV', 'DBA', 'USO'],
            'real_estate': ['VNQ', 'O', 'PLD', 'AMT'],
            'cryptocurrency': ['BTC-USD', 'ETH-USD'] if risk_tolerance != 'conservative' else [],
            'bonds': ['TIP', 'HYG', 'EMB'],
            'alternatives': ['VIX', 'UUP']
        }
        
        # Calculate total alternative allocation
        suggestions['total_alternative_percent'] = sum(suggestions['recommended_allocation'].values())
        
        # Ensure within target range
        if suggestions['total_alternative_percent'] < target_range['min']:
            # Scale up allocations
            scale_factor = target_range['min'] / suggestions['total_alternative_percent']
            suggestions['recommended_allocation'] = {
                k: v * scale_factor for k, v in suggestions['recommended_allocation'].items()
            }
        elif suggestions['total_alternative_percent'] > target_range['max']:
            # Scale down allocations
            scale_factor = target_range['max'] / suggestions['total_alternative_percent']
            suggestions['recommended_allocation'] = {
                k: v * scale_factor for k, v in suggestions['recommended_allocation'].items()
            }
        
        suggestions['total_alternative_percent'] = sum(suggestions['recommended_allocation'].values())
        
        return suggestions


def _interpret_correlation(correlation: float) -> str:
    """Interpret correlation coefficient"""
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        return 'Low correlation - Good diversification benefit'
    elif abs_corr < 0.7:
        return 'Moderate correlation - Some diversification benefit'
    else:
        return 'High correlation - Limited diversification benefit'


def get_popular_alternatives_by_category() -> Dict[str, List[Dict[str, str]]]:
    """Get popular alternative assets organized by category"""
    return {
        'Cryptocurrency': [
            {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'risk': 'Very High'},
            {'symbol': 'ETH-USD', 'name': 'Ethereum', 'risk': 'Very High'},
            {'symbol': 'BNB-USD', 'name': 'Binance Coin', 'risk': 'Very High'}
        ],
        'Commodities': [
            {'symbol': 'GLD', 'name': 'Gold ETF', 'risk': 'Medium'},
            {'symbol': 'SLV', 'name': 'Silver ETF', 'risk': 'Medium-High'},
            {'symbol': 'USO', 'name': 'Oil ETF', 'risk': 'High'},
            {'symbol': 'DBA', 'name': 'Agriculture ETF', 'risk': 'Medium-High'}
        ],
        'Real Estate': [
            {'symbol': 'VNQ', 'name': 'Real Estate ETF', 'risk': 'Medium'},
            {'symbol': 'O', 'name': 'Realty Income REIT', 'risk': 'Medium'},
            {'symbol': 'PLD', 'name': 'Prologis REIT', 'risk': 'Medium'},
            {'symbol': 'AMT', 'name': 'American Tower REIT', 'risk': 'Medium'}
        ],
        'Bonds': [
            {'symbol': 'TLT', 'name': '20+ Year Treasury ETF', 'risk': 'Low-Medium'},
            {'symbol': 'HYG', 'name': 'High Yield Bond ETF', 'risk': 'Medium'},
            {'symbol': 'EMB', 'name': 'Emerging Markets Bonds', 'risk': 'Medium-High'},
            {'symbol': 'TIP', 'name': 'TIPS ETF', 'risk': 'Low'}
        ]
    }