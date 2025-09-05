#!/usr/bin/env python3
"""
WebSocket Routes for Real-time Updates
Provides live stock price updates, portfolio changes, and notifications
"""
from flask import Blueprint
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_jwt_extended import decode_token, jwt_required
import yfinance as yf
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Set, List
import json

logger = logging.getLogger(__name__)

# Global variables for WebSocket management
socketio = None
connected_users: Dict[str, Dict] = {}  # session_id -> user_info
stock_subscriptions: Dict[str, Set[str]] = {}  # ticker -> set of session_ids
portfolio_subscriptions: Dict[int, Set[str]] = {}  # user_id -> set of session_ids
background_thread = None
thread_lock = threading.Lock()

def init_socketio(app):
    """Initialize SocketIO with the Flask app"""
    global socketio
    
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",  # Configure based on your CORS needs
        async_mode='threading',
        ping_timeout=60,
        ping_interval=25,
        logger=True,
        engineio_logger=True
    )
    
    setup_socketio_handlers()
    return socketio

def setup_socketio_handlers():
    """Set up WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect(auth=None):
        """Handle client connection"""
        logger.info(f"Client connected: {request.sid}")
        
        # Optional JWT authentication
        user_id = None
        user_info = {}
        
        if auth and auth.get('token'):
            try:
                token = auth['token']
                # Remove 'Bearer ' if present
                if token.startswith('Bearer '):
                    token = token[7:]
                
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']  # JWT subject
                user_info = {
                    'user_id': user_id,
                    'email': decoded_token.get('email'),
                    'first_name': decoded_token.get('first_name'),
                    'authenticated': True
                }
                logger.info(f"Authenticated user connected: {user_info['email']}")
            except Exception as e:
                logger.warning(f"Invalid token in WebSocket connection: {e}")
                user_info = {'authenticated': False}
        else:
            user_info = {'authenticated': False}
        
        # Store connection info
        connected_users[request.sid] = {
            'user_id': user_id,
            'connected_at': datetime.now(),
            'subscriptions': {
                'stocks': set(),
                'portfolios': set(),
                'notifications': user_id is not None
            },
            **user_info
        }
        
        emit('connected', {
            'status': 'connected',
            'authenticated': user_info.get('authenticated', False),
            'server_time': datetime.now().isoformat()
        })
        
        # Start background thread if not already running
        start_background_thread()
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info(f"Client disconnected: {request.sid}")
        
        if request.sid in connected_users:
            user_info = connected_users[request.sid]
            
            # Remove from stock subscriptions
            for ticker in user_info['subscriptions']['stocks']:
                if ticker in stock_subscriptions:
                    stock_subscriptions[ticker].discard(request.sid)
                    if not stock_subscriptions[ticker]:
                        del stock_subscriptions[ticker]
            
            # Remove from portfolio subscriptions
            if user_info['user_id']:
                user_id = user_info['user_id']
                if user_id in portfolio_subscriptions:
                    portfolio_subscriptions[user_id].discard(request.sid)
                    if not portfolio_subscriptions[user_id]:
                        del portfolio_subscriptions[user_id]
            
            # Remove user
            del connected_users[request.sid]
    
    @socketio.on('subscribe_stocks')
    def handle_subscribe_stocks(data):
        """Subscribe to real-time stock price updates"""
        tickers = data.get('tickers', [])
        
        if not isinstance(tickers, list) or len(tickers) > 50:
            emit('error', {'message': 'Invalid tickers list or too many tickers (max 50)'})
            return
        
        session_id = request.sid
        
        if session_id not in connected_users:
            emit('error', {'message': 'Not connected'})
            return
        
        # Clean tickers
        tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
        
        # Update subscriptions
        user_subscriptions = connected_users[session_id]['subscriptions']['stocks']
        
        # Remove old subscriptions
        for ticker in list(user_subscriptions):
            if ticker in stock_subscriptions:
                stock_subscriptions[ticker].discard(session_id)
                if not stock_subscriptions[ticker]:
                    del stock_subscriptions[ticker]
        
        # Add new subscriptions
        user_subscriptions.clear()
        for ticker in tickers:
            user_subscriptions.add(ticker)
            if ticker not in stock_subscriptions:
                stock_subscriptions[ticker] = set()
            stock_subscriptions[ticker].add(session_id)
        
        logger.info(f"Client {session_id} subscribed to stocks: {tickers}")
        emit('subscribed', {
            'type': 'stocks',
            'tickers': tickers,
            'count': len(tickers)
        })
    
    @socketio.on('unsubscribe_stocks')
    def handle_unsubscribe_stocks():
        """Unsubscribe from stock price updates"""
        session_id = request.sid
        
        if session_id not in connected_users:
            return
        
        # Remove from all stock subscriptions
        user_subscriptions = connected_users[session_id]['subscriptions']['stocks']
        for ticker in list(user_subscriptions):
            if ticker in stock_subscriptions:
                stock_subscriptions[ticker].discard(session_id)
                if not stock_subscriptions[ticker]:
                    del stock_subscriptions[ticker]
        
        user_subscriptions.clear()
        logger.info(f"Client {session_id} unsubscribed from all stocks")
        emit('unsubscribed', {'type': 'stocks'})
    
    @socketio.on('subscribe_portfolio')
    def handle_subscribe_portfolio(data):
        """Subscribe to portfolio updates (authenticated users only)"""
        session_id = request.sid
        
        if session_id not in connected_users:
            emit('error', {'message': 'Not connected'})
            return
        
        user_info = connected_users[session_id]
        if not user_info.get('authenticated') or not user_info.get('user_id'):
            emit('error', {'message': 'Authentication required for portfolio updates'})
            return
        
        user_id = user_info['user_id']
        
        # Add to portfolio subscriptions
        if user_id not in portfolio_subscriptions:
            portfolio_subscriptions[user_id] = set()
        portfolio_subscriptions[user_id].add(session_id)
        
        connected_users[session_id]['subscriptions']['portfolios'].add(user_id)
        
        logger.info(f"User {user_id} subscribed to portfolio updates")
        emit('subscribed', {
            'type': 'portfolio',
            'user_id': user_id
        })
    
    @socketio.on('get_stock_price')
    def handle_get_stock_price(data):
        """Get current stock price on demand"""
        ticker = data.get('ticker', '').upper().strip()
        
        if not ticker:
            emit('error', {'message': 'Ticker required'})
            return
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                emit('error', {'message': f'No data found for ticker {ticker}'})
                return
            
            current_price = float(hist['Close'].iloc[-1])
            
            emit('stock_price', {
                'ticker': ticker,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            emit('error', {'message': f'Error fetching price for {ticker}'})
    
    @socketio.on('ping')
    def handle_ping():
        """Handle ping from client"""
        emit('pong', {'timestamp': datetime.now().isoformat()})
    
    @socketio.on_error_default
    def default_error_handler(e):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {e}")
        emit('error', {'message': 'Server error occurred'})

def start_background_thread():
    """Start background thread for periodic updates"""
    global background_thread
    
    with thread_lock:
        if background_thread is None:
            background_thread = socketio.start_background_task(background_task)

def background_task():
    """Background task to send periodic updates"""
    logger.info("Background WebSocket task started")
    
    while True:
        try:
            # Send stock price updates
            if stock_subscriptions:
                update_stock_prices()
            
            # Send portfolio updates (if any portfolios are being tracked)
            if portfolio_subscriptions:
                update_portfolio_values()
            
            # Send system status updates
            send_system_status()
            
            # Sleep for 5 seconds between updates
            socketio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in background WebSocket task: {e}")
            socketio.sleep(5)

def update_stock_prices():
    """Fetch and broadcast stock price updates"""
    if not stock_subscriptions:
        return
    
    try:
        # Get all subscribed tickers
        tickers = list(stock_subscriptions.keys())
        
        if not tickers:
            return
        
        # Fetch prices (batch request)
        tickers_string = ' '.join(tickers)
        data = yf.download(
            tickers_string,
            period="1d",
            interval="1m",
            progress=False,
            show_errors=False
        )
        
        if data.empty:
            return
        
        # Process and send updates
        current_time = datetime.now().isoformat()
        
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker
                    current_price = float(data['Close'].iloc[-1])
                else:
                    # Multiple tickers
                    current_price = float(data['Close'][ticker].iloc[-1])
                
                # Send to all subscribers of this ticker
                subscriber_sessions = list(stock_subscriptions.get(ticker, []))
                
                price_update = {
                    'ticker': ticker,
                    'price': current_price,
                    'timestamp': current_time,
                    'type': 'price_update'
                }
                
                for session_id in subscriber_sessions:
                    if session_id in connected_users:
                        socketio.emit('stock_update', price_update, room=session_id)
                
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(f"Error processing price for {ticker}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error updating stock prices: {e}")

def update_portfolio_values():
    """Calculate and broadcast portfolio value updates"""
    # This is a placeholder - would need actual portfolio data from database
    # For now, just send a status update
    
    for user_id, session_ids in portfolio_subscriptions.items():
        try:
            # TODO: Calculate actual portfolio value from database
            portfolio_update = {
                'user_id': user_id,
                'total_value': 0,  # Calculate from database
                'daily_change': 0,  # Calculate from database
                'timestamp': datetime.now().isoformat(),
                'type': 'portfolio_update'
            }
            
            for session_id in list(session_ids):
                if session_id in connected_users:
                    socketio.emit('portfolio_update', portfolio_update, room=session_id)
                    
        except Exception as e:
            logger.error(f"Error updating portfolio for user {user_id}: {e}")

def send_system_status():
    """Send periodic system status updates"""
    if not connected_users:
        return
    
    try:
        status_update = {
            'connected_users': len(connected_users),
            'active_stock_subscriptions': len(stock_subscriptions),
            'server_time': datetime.now().isoformat(),
            'type': 'system_status'
        }
        
        socketio.emit('system_status', status_update, broadcast=True)
        
    except Exception as e:
        logger.error(f"Error sending system status: {e}")

def send_notification(user_id: int, notification: Dict):
    """Send notification to specific user"""
    if user_id not in portfolio_subscriptions:
        return
    
    try:
        notification_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'notification',
            **notification
        }
        
        for session_id in list(portfolio_subscriptions[user_id]):
            if session_id in connected_users:
                socketio.emit('notification', notification_data, room=session_id)
                
    except Exception as e:
        logger.error(f"Error sending notification to user {user_id}: {e}")

def broadcast_market_alert(alert: Dict):
    """Broadcast market alert to all connected users"""
    try:
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'market_alert',
            **alert
        }
        
        socketio.emit('market_alert', alert_data, broadcast=True)
        
    except Exception as e:
        logger.error(f"Error broadcasting market alert: {e}")

# Utility functions for other routes to use
def notify_portfolio_change(user_id: int, change_type: str, details: Dict):
    """Notify user about portfolio changes"""
    if socketio:
        notification = {
            'title': 'Portfolio Update',
            'message': f'Your portfolio has been {change_type}',
            'change_type': change_type,
            'details': details
        }
        send_notification(user_id, notification)

def notify_prediction_complete(user_id: int, ticker: str, result: Dict):
    """Notify user when ML prediction is complete"""
    if socketio:
        notification = {
            'title': 'Prediction Complete',
            'message': f'ML prediction for {ticker} is ready',
            'ticker': ticker,
            'result': result
        }
        send_notification(user_id, notification)