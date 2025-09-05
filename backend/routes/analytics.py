#!/usr/bin/env python3
"""
Advanced Analytics and Reporting Routes
Comprehensive portfolio analytics, risk metrics, and performance reporting
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

from models.database import get_db, Portfolio, PredictionHistory, User
from schemas import *
from utils.error_handlers import *
from utils.cache import cached, cache

logger = logging.getLogger(__name__)

# Create Blueprint
analytics_bp = Blueprint('analytics', __name__)

class PortfolioAnalytics:
    """Advanced portfolio analytics engine"""
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.DataFrame) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        cumulative = (1 + PortfolioAnalytics.calculate_returns(prices)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        
        # Find drawdown period
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative.loc[:max_dd_end].idxmax()
        
        return {
            'max_drawdown': float(max_drawdown),
            'start_date': max_dd_start.isoformat() if pd.notna(max_dd_start) else None,
            'end_date': max_dd_end.isoformat() if pd.notna(max_dd_end) else None,
            'duration_days': (max_dd_end - max_dd_start).days if pd.notna(max_dd_start) and pd.notna(max_dd_end) else 0
        }
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta against market"""
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_sector_allocation(holdings: List[Dict], stock_info: Dict) -> Dict[str, float]:
        """Calculate sector allocation"""
        sector_allocation = {}
        total_value = sum(holding['shares'] * holding.get('current_price', 0) for holding in holdings)
        
        if total_value == 0:
            return sector_allocation
        
        for holding in holdings:
            ticker = holding['ticker']
            value = holding['shares'] * holding.get('current_price', 0)
            sector = stock_info.get(ticker, {}).get('sector', 'Unknown')
            
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += (value / total_value) * 100
        
        return sector_allocation

# Request/Response schemas
class AnalyticsRequest(BaseModel):
    portfolio_id: int = Field(..., description="Portfolio ID to analyze")
    start_date: Optional[str] = Field(None, description="Start date for analysis (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for analysis (YYYY-MM-DD)")
    benchmark: Optional[str] = Field("SPY", description="Benchmark ticker for comparison")
    include_predictions: bool = Field(True, description="Include ML predictions in analysis")

class RiskMetricsRequest(BaseModel):
    portfolio_id: int
    confidence_levels: List[float] = Field([0.05, 0.01], description="VaR confidence levels")
    time_horizons: List[int] = Field([1, 5, 22], description="Time horizons in days")

@analytics_bp.route('/portfolio/<int:portfolio_id>/summary', methods=['GET'])
@jwt_required()
@safe_api_call
@cached(ttl=300, key_prefix="portfolio_summary")
def get_portfolio_summary(portfolio_id: int):
    """Get comprehensive portfolio summary"""
    current_user_id = get_jwt_identity()
    db = next(get_db())
    
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user_id
        ).first()
        
        if not portfolio:
            raise NotFoundError("Portfolio not found")
        
        holdings = portfolio.holdings or []
        if not holdings:
            return jsonify({
                'portfolio_id': portfolio_id,
                'summary': {'total_value': 0, 'holdings_count': 0},
                'message': 'Portfolio is empty'
            })
        
        # Get current prices
        tickers = [holding['ticker'] for holding in holdings]
        
        # Fetch current data
        current_prices = {}
        stock_info = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for ticker in tickers:
                futures[ticker] = executor.submit(fetch_stock_data, ticker)
            
            for ticker, future in futures.items():
                try:
                    data = future.result(timeout=10)
                    current_prices[ticker] = data['current_price']
                    stock_info[ticker] = data
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    current_prices[ticker] = holdings[next(i for i, h in enumerate(holdings) if h['ticker'] == ticker)].get('avg_price', 0)
        
        # Calculate portfolio metrics
        total_value = 0
        total_cost = 0
        positions = []
        
        for holding in holdings:
            ticker = holding['ticker']
            shares = holding['shares']
            avg_price = holding.get('avg_price', 0)
            current_price = current_prices.get(ticker, avg_price)
            
            position_value = shares * current_price
            position_cost = shares * avg_price
            position_gain_loss = position_value - position_cost
            position_gain_loss_pct = (position_gain_loss / position_cost) * 100 if position_cost > 0 else 0
            
            total_value += position_value
            total_cost += position_cost
            
            positions.append({
                'ticker': ticker,
                'shares': shares,
                'avg_price': avg_price,
                'current_price': current_price,
                'position_value': position_value,
                'position_cost': position_cost,
                'gain_loss': position_gain_loss,
                'gain_loss_percent': position_gain_loss_pct,
                'weight': 0  # Will calculate after total_value is known
            })
        
        # Calculate weights
        for position in positions:
            position['weight'] = (position['position_value'] / total_value) * 100 if total_value > 0 else 0
        
        # Portfolio level metrics
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
        
        # Sector allocation
        sector_allocation = PortfolioAnalytics.calculate_sector_allocation(
            [{'ticker': p['ticker'], 'shares': p['shares'], 'current_price': p['current_price']} for p in positions],
            stock_info
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'portfolio_name': portfolio.name,
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': total_gain_loss_pct,
                'holdings_count': len(positions),
                'largest_position': max(positions, key=lambda x: x['position_value'])['ticker'] if positions else None,
                'most_profitable': max(positions, key=lambda x: x['gain_loss_percent'])['ticker'] if positions else None,
                'least_profitable': min(positions, key=lambda x: x['gain_loss_percent'])['ticker'] if positions else None
            },
            'positions': positions,
            'sector_allocation': sector_allocation
        })
        
    finally:
        db.close()

@analytics_bp.route('/portfolio/<int:portfolio_id>/risk-metrics', methods=['POST'])
@jwt_required()
@safe_api_call
def calculate_risk_metrics(portfolio_id: int):
    """Calculate comprehensive risk metrics"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = RiskMetricsRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid request data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user_id
        ).first()
        
        if not portfolio:
            raise NotFoundError("Portfolio not found")
        
        holdings = portfolio.holdings or []
        if not holdings:
            raise ValidationException("Portfolio is empty")
        
        # Get historical data for risk calculation
        tickers = [holding['ticker'] for holding in holdings]
        weights = np.array([holding.get('weight', 1/len(holdings)) for holding in holdings])
        
        # Fetch 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        historical_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        if historical_data.empty:
            raise ValidationException("Unable to fetch historical data")
        
        # Calculate returns
        returns = historical_data.pct_change().dropna()
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Risk metrics
        analytics = PortfolioAnalytics()
        
        # Basic metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Value at Risk
        var_metrics = {}
        for confidence in request_data.confidence_levels:
            for horizon in request_data.time_horizons:
                key = f"var_{int(confidence*100)}_{horizon}d"
                var_metrics[key] = analytics.calculate_var(
                    portfolio_returns.tail(252), confidence
                ) * np.sqrt(horizon)
        
        # Maximum drawdown
        portfolio_prices = (1 + portfolio_returns).cumprod()
        max_drawdown = analytics.calculate_max_drawdown(
            pd.DataFrame({'portfolio': portfolio_prices})
        )
        
        # Correlation matrix
        correlation_matrix = returns.corr().round(3).to_dict()
        
        # Beta (against SPY)
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
        spy_returns = spy_data.pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(spy_returns.index)
        if len(common_dates) > 0:
            portfolio_beta = analytics.calculate_beta(
                portfolio_returns.loc[common_dates],
                spy_returns.loc[common_dates]
            )
        else:
            portfolio_beta = 1.0
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'risk_metrics': {
                'portfolio_volatility': float(portfolio_volatility),
                'portfolio_beta': float(portfolio_beta),
                'max_drawdown': max_drawdown,
                'value_at_risk': var_metrics,
                'individual_volatilities': {
                    ticker: float(vol) for ticker, vol in volatility.items()
                },
                'correlation_matrix': correlation_matrix
            },
            'analysis_period': {
                'start_date': start_date.date().isoformat(),
                'end_date': end_date.date().isoformat(),
                'trading_days': len(portfolio_returns)
            }
        })
        
    finally:
        db.close()

@analytics_bp.route('/portfolio/<int:portfolio_id>/performance', methods=['GET'])
@jwt_required()
@safe_api_call
@cached(ttl=600, key_prefix="portfolio_performance")
def get_performance_analysis(portfolio_id: int):
    """Get detailed performance analysis"""
    current_user_id = get_jwt_identity()
    
    # Get query parameters
    benchmark = request.args.get('benchmark', 'SPY')
    period = request.args.get('period', '1y')  # 1y, 6m, 3m, 1m
    
    db = next(get_db())
    
    try:
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user_id
        ).first()
        
        if not portfolio:
            raise NotFoundError("Portfolio not found")
        
        # Calculate performance metrics
        performance_data = calculate_portfolio_performance(
            portfolio, benchmark, period
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'performance': performance_data,
            'benchmark': benchmark,
            'period': period
        })
        
    finally:
        db.close()

@analytics_bp.route('/market/sector-analysis', methods=['GET'])
@safe_api_call
@cached(ttl=1800, key_prefix="sector_analysis")
def get_sector_analysis():
    """Get market sector analysis"""
    try:
        # Sector ETFs for analysis
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Communication Services': 'XLC',
            'Industrial': 'XLI',
            'Consumer Defensive': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        # Fetch sector performance
        sector_data = {}
        tickers = list(sector_etfs.values())
        
        # Get 1 year of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        for sector, ticker in sector_etfs.items():
            if ticker in data.columns:
                prices = data[ticker].dropna()
                if len(prices) > 0:
                    returns = prices.pct_change().dropna()
                    
                    sector_data[sector] = {
                        'ticker': ticker,
                        'current_price': float(prices.iloc[-1]),
                        'ytd_return': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
                        'volatility': float(returns.std() * np.sqrt(252) * 100),
                        'max_drawdown': PortfolioAnalytics.calculate_max_drawdown(
                            pd.DataFrame({ticker: prices})
                        )['max_drawdown'] * 100
                    }
        
        return jsonify({
            'sector_analysis': sector_data,
            'analysis_date': datetime.now().isoformat(),
            'period': '1 year'
        })
        
    except Exception as e:
        logger.error(f"Error in sector analysis: {e}")
        raise ExternalAPIError("Yahoo Finance", "Unable to fetch sector data")

def fetch_stock_data(ticker: str) -> Dict:
    """Fetch comprehensive stock data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")
        
        if hist.empty:
            raise ValueError(f"No data found for {ticker}")
        
        current_price = float(hist['Close'].iloc[-1])
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', 0),
            'dividend_yield': info.get('dividendYield', 0)
        }
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return {
            'ticker': ticker,
            'current_price': 0,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0
        }

def calculate_portfolio_performance(portfolio: Portfolio, benchmark: str, period: str) -> Dict:
    """Calculate comprehensive portfolio performance"""
    # This is a placeholder for the actual performance calculation
    # In a real implementation, you would:
    # 1. Get historical portfolio values
    # 2. Compare with benchmark performance
    # 3. Calculate risk-adjusted metrics
    
    return {
        'total_return': 0.0,  # Calculate from historical data
        'annualized_return': 0.0,
        'benchmark_return': 0.0,
        'alpha': 0.0,
        'beta': 1.0,
        'sharpe_ratio': 0.0,
        'information_ratio': 0.0,
        'tracking_error': 0.0
    }