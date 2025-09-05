#!/usr/bin/env python3
"""
Personalized User Dashboard Routes
Customizable dashboard with widgets, preferences, and personalized insights
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, JSON
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Any
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.database import get_db, User, Portfolio, PredictionHistory, Base
from schemas import *
from utils.error_handlers import *
from utils.cache import cached

logger = logging.getLogger(__name__)

# Create Blueprint
dashboard_bp = Blueprint('dashboard', __name__)

# Dashboard widget types
class DashboardWidget(Base):
    """User dashboard widget configuration"""
    __tablename__ = 'dashboard_widgets'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Widget configuration
    widget_type = Column(String(50), nullable=False)  # 'portfolio_summary', 'watchlist', 'news', etc.
    title = Column(String(200))
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    width = Column(Integer, default=4)
    height = Column(Integer, default=3)
    
    # Widget settings (JSON)
    settings = Column(JSON, default={})
    
    # State
    is_visible = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    def to_dict(self):
        return {
            'id': self.id,
            'widget_type': self.widget_type,
            'title': self.title,
            'position': {'x': self.position_x, 'y': self.position_y},
            'size': {'width': self.width, 'height': self.height},
            'settings': self.settings,
            'is_visible': self.is_visible,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserWatchlist(Base):
    """User stock watchlist"""
    __tablename__ = 'user_watchlists'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    
    # Custom settings
    notes = Column(Text)
    price_target = Column(Float)
    stop_loss = Column(Float)
    
    # Metadata
    added_at = Column(DateTime, server_default=func.now())
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'notes': self.notes,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'added_at': self.added_at.isoformat() if self.added_at else None
        }

# Request/Response schemas
class CreateWidgetRequest(BaseModel):
    widget_type: str = Field(..., min_length=1)
    title: Optional[str] = Field(None)
    position: Dict[str, int] = Field(default={'x': 0, 'y': 0})
    size: Dict[str, int] = Field(default={'width': 4, 'height': 3})
    settings: Dict[str, Any] = Field(default_factory=dict)

class UpdateWidgetRequest(BaseModel):
    title: Optional[str] = None
    position: Optional[Dict[str, int]] = None
    size: Optional[Dict[str, int]] = None
    settings: Optional[Dict[str, Any]] = None
    is_visible: Optional[bool] = None

class WatchlistRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    notes: Optional[str] = Field(None, max_length=1000)
    price_target: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()

# Dashboard data aggregators
class DashboardDataService:
    """Service to aggregate dashboard data"""
    
    @staticmethod
    def get_portfolio_summary(user_id: int) -> Dict[str, Any]:
        """Get portfolio summary for dashboard"""
        db = next(get_db())
        
        try:
            portfolios = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
            
            if not portfolios:
                return {
                    'total_portfolios': 0,
                    'total_value': 0,
                    'total_gain_loss': 0,
                    'best_performer': None,
                    'worst_performer': None
                }
            
            total_value = 0
            total_cost = 0
            portfolio_performances = []
            
            for portfolio in portfolios:
                holdings = portfolio.holdings or []
                portfolio_value = 0
                portfolio_cost = 0
                
                for holding in holdings:
                    # Get current price (simplified - would use caching in production)
                    try:
                        stock = yf.Ticker(holding['ticker'])
                        hist = stock.history(period="1d")
                        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else holding.get('avg_price', 0)
                    except:
                        current_price = holding.get('avg_price', 0)
                    
                    holding_value = holding['shares'] * current_price
                    holding_cost = holding['shares'] * holding.get('avg_price', 0)
                    
                    portfolio_value += holding_value
                    portfolio_cost += holding_cost
                
                portfolio_gain_loss = ((portfolio_value - portfolio_cost) / portfolio_cost * 100) if portfolio_cost > 0 else 0
                
                portfolio_performances.append({
                    'name': portfolio.name,
                    'value': portfolio_value,
                    'gain_loss_percent': portfolio_gain_loss
                })
                
                total_value += portfolio_value
                total_cost += portfolio_cost
            
            total_gain_loss = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            
            best_performer = max(portfolio_performances, key=lambda x: x['gain_loss_percent']) if portfolio_performances else None
            worst_performer = min(portfolio_performances, key=lambda x: x['gain_loss_percent']) if portfolio_performances else None
            
            return {
                'total_portfolios': len(portfolios),
                'total_value': total_value,
                'total_gain_loss': total_gain_loss,
                'best_performer': best_performer,
                'worst_performer': worst_performer,
                'portfolio_breakdown': portfolio_performances
            }
            
        finally:
            db.close()
    
    @staticmethod
    def get_market_overview() -> Dict[str, Any]:
        """Get market overview data"""
        try:
            # Major indices
            indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'VIX': '^VIX'
            }
            
            market_data = {}
            
            for name, ticker in indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="2d")
                    
                    if len(hist) >= 2:
                        current = float(hist['Close'].iloc[-1])
                        previous = float(hist['Close'].iloc[-2])
                        change_percent = ((current - previous) / previous) * 100
                        
                        market_data[name] = {
                            'value': current,
                            'change_percent': change_percent,
                            'ticker': ticker
                        }
                except Exception as e:
                    logger.error(f"Error fetching {name} data: {e}")
            
            return {
                'indices': market_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in market overview: {e}")
            return {'indices': {}, 'last_updated': datetime.now().isoformat()}
    
    @staticmethod
    def get_watchlist_data(user_id: int) -> List[Dict[str, Any]]:
        """Get watchlist with current prices"""
        db = next(get_db())
        
        try:
            watchlist_items = db.query(UserWatchlist).filter(
                UserWatchlist.user_id == user_id
            ).all()
            
            if not watchlist_items:
                return []
            
            # Fetch current prices for all tickers
            tickers = [item.ticker for item in watchlist_items]
            current_prices = {}
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {
                    executor.submit(DashboardDataService._fetch_ticker_data, ticker): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        price_data = future.result(timeout=5)
                        current_prices[ticker] = price_data
                    except Exception as e:
                        logger.error(f"Error fetching {ticker} data: {e}")
                        current_prices[ticker] = {'price': 0, 'change_percent': 0}
            
            # Combine watchlist items with price data
            watchlist_data = []
            for item in watchlist_items:
                price_data = current_prices.get(item.ticker, {'price': 0, 'change_percent': 0})
                
                item_data = item.to_dict()
                item_data.update({
                    'current_price': price_data['price'],
                    'change_percent': price_data['change_percent'],
                    'distance_to_target': None,
                    'distance_to_stop': None
                })
                
                # Calculate distances to targets
                if item.price_target and price_data['price'] > 0:
                    item_data['distance_to_target'] = ((item.price_target - price_data['price']) / price_data['price']) * 100
                
                if item.stop_loss and price_data['price'] > 0:
                    item_data['distance_to_stop'] = ((price_data['price'] - item.stop_loss) / price_data['price']) * 100
                
                watchlist_data.append(item_data)
            
            return watchlist_data
            
        finally:
            db.close()
    
    @staticmethod
    def _fetch_ticker_data(ticker: str) -> Dict[str, float]:
        """Fetch current price and change for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            
            if len(hist) >= 2:
                current = float(hist['Close'].iloc[-1])
                previous = float(hist['Close'].iloc[-2])
                change_percent = ((current - previous) / previous) * 100
                
                return {'price': current, 'change_percent': change_percent}
            elif len(hist) == 1:
                current = float(hist['Close'].iloc[-1])
                return {'price': current, 'change_percent': 0}
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
        
        return {'price': 0, 'change_percent': 0}
    
    @staticmethod
    def get_recent_predictions(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent ML predictions for user's portfolios"""
        db = next(get_db())
        
        try:
            # Get user's portfolio tickers
            portfolios = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
            user_tickers = set()
            
            for portfolio in portfolios:
                holdings = portfolio.holdings or []
                for holding in holdings:
                    user_tickers.add(holding['ticker'])
            
            if not user_tickers:
                return []
            
            # Get recent predictions for user's tickers
            recent_predictions = db.query(PredictionHistory).filter(
                PredictionHistory.ticker.in_(user_tickers)
            ).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
            
            predictions_data = []
            for prediction in recent_predictions:
                pred_data = prediction.to_dict()
                # Simplify prediction data for dashboard
                if pred_data['summary_stats']:
                    pred_data['expected_return'] = pred_data['summary_stats'].get('avg_predicted_return', 0)
                    pred_data['confidence'] = pred_data['summary_stats'].get('volatility_estimate', 0)
                
                predictions_data.append(pred_data)
            
            return predictions_data
            
        finally:
            db.close()

# Routes
@dashboard_bp.route('/dashboard', methods=['GET'])
@jwt_required()
@safe_api_call
@cached(ttl=300, key_prefix="user_dashboard")
def get_dashboard():
    """Get personalized dashboard data"""
    current_user_id = get_jwt_identity()
    
    # Get user widgets
    db = next(get_db())
    
    try:
        widgets = db.query(DashboardWidget).filter(
            DashboardWidget.user_id == current_user_id,
            DashboardWidget.is_visible == True
        ).order_by(DashboardWidget.position_y, DashboardWidget.position_x).all()
        
        # If no widgets, create default layout
        if not widgets:
            default_widgets = create_default_dashboard_layout(current_user_id)
            widgets = default_widgets
        
        # Gather widget data
        dashboard_data = {
            'layout': [widget.to_dict() for widget in widgets],
            'data': {}
        }
        
        # Get data for each widget type
        widget_types = {widget.widget_type for widget in widgets}
        
        service = DashboardDataService()
        
        if 'portfolio_summary' in widget_types:
            dashboard_data['data']['portfolio_summary'] = service.get_portfolio_summary(current_user_id)
        
        if 'market_overview' in widget_types:
            dashboard_data['data']['market_overview'] = service.get_market_overview()
        
        if 'watchlist' in widget_types:
            dashboard_data['data']['watchlist'] = service.get_watchlist_data(current_user_id)
        
        if 'recent_predictions' in widget_types:
            dashboard_data['data']['recent_predictions'] = service.get_recent_predictions(current_user_id)
        
        return jsonify({
            'dashboard': dashboard_data,
            'last_updated': datetime.now().isoformat()
        })
        
    finally:
        db.close()

@dashboard_bp.route('/dashboard/widgets', methods=['POST'])
@jwt_required()
@safe_api_call
def create_widget():
    """Create a new dashboard widget"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = CreateWidgetRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid widget data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        widget = DashboardWidget(
            user_id=current_user_id,
            widget_type=request_data.widget_type,
            title=request_data.title,
            position_x=request_data.position['x'],
            position_y=request_data.position['y'],
            width=request_data.size['width'],
            height=request_data.size['height'],
            settings=request_data.settings
        )
        
        db.add(widget)
        db.commit()
        db.refresh(widget)
        
        return jsonify({
            'message': 'Widget created successfully',
            'widget': widget.to_dict()
        }), 201
        
    finally:
        db.close()

@dashboard_bp.route('/dashboard/widgets/<int:widget_id>', methods=['PUT'])
@jwt_required()
@safe_api_call
def update_widget(widget_id: int):
    """Update dashboard widget"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = UpdateWidgetRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid widget data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        widget = db.query(DashboardWidget).filter(
            DashboardWidget.id == widget_id,
            DashboardWidget.user_id == current_user_id
        ).first()
        
        if not widget:
            raise NotFoundError("Widget not found")
        
        # Update fields
        if request_data.title is not None:
            widget.title = request_data.title
        if request_data.position is not None:
            widget.position_x = request_data.position['x']
            widget.position_y = request_data.position['y']
        if request_data.size is not None:
            widget.width = request_data.size['width']
            widget.height = request_data.size['height']
        if request_data.settings is not None:
            widget.settings = request_data.settings
        if request_data.is_visible is not None:
            widget.is_visible = request_data.is_visible
        
        widget.updated_at = datetime.now()
        db.commit()
        
        return jsonify({
            'message': 'Widget updated successfully',
            'widget': widget.to_dict()
        })
        
    finally:
        db.close()

@dashboard_bp.route('/dashboard/widgets/<int:widget_id>', methods=['DELETE'])
@jwt_required()
@safe_api_call
def delete_widget(widget_id: int):
    """Delete dashboard widget"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        widget = db.query(DashboardWidget).filter(
            DashboardWidget.id == widget_id,
            DashboardWidget.user_id == current_user_id
        ).first()
        
        if not widget:
            raise NotFoundError("Widget not found")
        
        db.delete(widget)
        db.commit()
        
        return jsonify({'message': 'Widget deleted successfully'})
        
    finally:
        db.close()

@dashboard_bp.route('/watchlist', methods=['GET'])
@jwt_required()
@safe_api_call
def get_watchlist():
    """Get user's watchlist"""
    current_user_id = get_jwt_identity()
    
    watchlist_data = DashboardDataService.get_watchlist_data(current_user_id)
    
    return jsonify({
        'watchlist': watchlist_data,
        'count': len(watchlist_data)
    })

@dashboard_bp.route('/watchlist', methods=['POST'])
@jwt_required()
@safe_api_call
def add_to_watchlist():
    """Add stock to watchlist"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = WatchlistRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid watchlist data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        # Check if already exists
        existing = db.query(UserWatchlist).filter(
            UserWatchlist.user_id == current_user_id,
            UserWatchlist.ticker == request_data.ticker
        ).first()
        
        if existing:
            raise ValidationException(f"{request_data.ticker} is already in your watchlist")
        
        watchlist_item = UserWatchlist(
            user_id=current_user_id,
            ticker=request_data.ticker,
            notes=request_data.notes,
            price_target=request_data.price_target,
            stop_loss=request_data.stop_loss
        )
        
        db.add(watchlist_item)
        db.commit()
        db.refresh(watchlist_item)
        
        return jsonify({
            'message': f'{request_data.ticker} added to watchlist',
            'item': watchlist_item.to_dict()
        }), 201
        
    finally:
        db.close()

@dashboard_bp.route('/watchlist/<int:item_id>', methods=['DELETE'])
@jwt_required()
@safe_api_call
def remove_from_watchlist(item_id: int):
    """Remove stock from watchlist"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        item = db.query(UserWatchlist).filter(
            UserWatchlist.id == item_id,
            UserWatchlist.user_id == current_user_id
        ).first()
        
        if not item:
            raise NotFoundError("Watchlist item not found")
        
        ticker = item.ticker
        db.delete(item)
        db.commit()
        
        return jsonify({'message': f'{ticker} removed from watchlist'})
        
    finally:
        db.close()

def create_default_dashboard_layout(user_id: int) -> List[DashboardWidget]:
    """Create default dashboard layout for new users"""
    db = next(get_db())
    
    try:
        default_widgets = [
            {
                'widget_type': 'portfolio_summary',
                'title': 'Portfolio Overview',
                'position_x': 0,
                'position_y': 0,
                'width': 6,
                'height': 4,
                'settings': {}
            },
            {
                'widget_type': 'market_overview',
                'title': 'Market Overview',
                'position_x': 6,
                'position_y': 0,
                'width': 6,
                'height': 4,
                'settings': {}
            },
            {
                'widget_type': 'watchlist',
                'title': 'Watchlist',
                'position_x': 0,
                'position_y': 4,
                'width': 6,
                'height': 5,
                'settings': {}
            },
            {
                'widget_type': 'recent_predictions',
                'title': 'Recent Predictions',
                'position_x': 6,
                'position_y': 4,
                'width': 6,
                'height': 5,
                'settings': {}
            }
        ]
        
        widgets = []
        for widget_config in default_widgets:
            widget = DashboardWidget(
                user_id=user_id,
                **widget_config
            )
            db.add(widget)
            widgets.append(widget)
        
        db.commit()
        
        for widget in widgets:
            db.refresh(widget)
        
        return widgets
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating default dashboard layout: {e}")
        return []
    finally:
        db.close()