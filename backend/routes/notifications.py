#!/usr/bin/env python3
"""
Notification System Routes
Email notifications, alerts, and in-app notification management
"""
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, Enum as SQLEnum
from datetime import datetime, timedelta
import smtplib
import threading
import queue
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Any
import json
from enum import Enum

from models.database import get_db, User, Base
from schemas import *
from utils.error_handlers import *
from routes.websocket import send_notification, broadcast_market_alert

logger = logging.getLogger(__name__)

# Create Blueprint
notifications_bp = Blueprint('notifications', __name__)

# Notification types and priority levels
class NotificationType(str, Enum):
    PRICE_ALERT = "price_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    PREDICTION_COMPLETE = "prediction_complete"
    MARKET_NEWS = "market_news"
    SYSTEM_UPDATE = "system_update"
    SECURITY_ALERT = "security_alert"

class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    BOTH = "both"

# Database models
class Notification(Base):
    """Notification model"""
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Notification content
    type = Column(SQLEnum(NotificationType), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(Text)  # JSON data
    
    # Notification settings
    priority = Column(SQLEnum(NotificationPriority), default=NotificationPriority.MEDIUM)
    channel = Column(SQLEnum(NotificationChannel), default=NotificationChannel.IN_APP)
    
    # Status tracking
    is_read = Column(Boolean, default=False)
    is_sent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    read_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)  # Optional expiration
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'type': self.type.value if self.type else None,
            'title': self.title,
            'message': self.message,
            'data': json.loads(self.data) if self.data else {},
            'priority': self.priority.value if self.priority else None,
            'channel': self.channel.value if self.channel else None,
            'is_read': self.is_read,
            'is_sent': self.is_sent,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class PriceAlert(Base):
    """Price alert model"""
    __tablename__ = 'price_alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    
    # Alert conditions
    condition = Column(String(20), nullable=False)  # 'above', 'below', 'change_percent'
    target_value = Column(Float, nullable=False)
    current_value = Column(Float)  # Last known value
    
    # Alert settings
    is_active = Column(Boolean, default=True)
    is_triggered = Column(Boolean, default=False)
    notification_channel = Column(SQLEnum(NotificationChannel), default=NotificationChannel.BOTH)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    triggered_at = Column(DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'ticker': self.ticker,
            'condition': self.condition,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'is_active': self.is_active,
            'is_triggered': self.is_triggered,
            'notification_channel': self.notification_channel.value if self.notification_channel else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None
        }

# Request/Response schemas
class CreateNotificationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    type: NotificationType = Field(default=NotificationType.SYSTEM_UPDATE)
    priority: NotificationPriority = Field(default=NotificationPriority.MEDIUM)
    channel: NotificationChannel = Field(default=NotificationChannel.IN_APP)
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    expires_at: Optional[str] = Field(None, description="Expiration datetime (ISO format)")

class CreatePriceAlertRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    condition: str = Field(..., regex=r'^(above|below|change_percent)$')
    target_value: float = Field(..., description="Target price or percentage change")
    notification_channel: NotificationChannel = Field(default=NotificationChannel.BOTH)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()

class NotificationPreferencesRequest(BaseModel):
    email_enabled: bool = Field(True)
    price_alerts: bool = Field(True)
    portfolio_updates: bool = Field(True)
    market_news: bool = Field(False)
    system_updates: bool = Field(True)
    security_alerts: bool = Field(True)

# Email notification system
class EmailNotificationService:
    """Email notification service"""
    
    def __init__(self):
        self.smtp_server = current_app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = current_app.config.get('SMTP_PORT', 587)
        self.smtp_username = current_app.config.get('SMTP_USERNAME', '')
        self.smtp_password = current_app.config.get('SMTP_PASSWORD', '')
        self.from_email = current_app.config.get('FROM_EMAIL', 'noreply@portfoliooptimizer.com')
        
        # Email queue for background processing
        self.email_queue = queue.Queue()
        self.worker_thread = None
        self.start_worker()
    
    def start_worker(self):
        """Start background email worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._email_worker, daemon=True)
            self.worker_thread.start()
            logger.info("Email worker thread started")
    
    def _email_worker(self):
        """Background worker to process email queue"""
        while True:
            try:
                email_data = self.email_queue.get(timeout=60)  # 1 minute timeout
                self._send_email_sync(email_data)
                self.email_queue.task_done()
            except queue.Empty:
                continue  # Timeout, keep running
            except Exception as e:
                logger.error(f"Email worker error: {e}")
    
    def send_email(self, to_email: str, subject: str, message: str, 
                   html_message: Optional[str] = None, attachments: List = None):
        """Queue email for sending"""
        email_data = {
            'to_email': to_email,
            'subject': subject,
            'message': message,
            'html_message': html_message,
            'attachments': attachments or []
        }
        
        try:
            self.email_queue.put(email_data, block=False)
            logger.info(f"Email queued for {to_email}: {subject}")
        except queue.Full:
            logger.error("Email queue is full, dropping email")
    
    def _send_email_sync(self, email_data: Dict):
        """Send email synchronously"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = email_data['to_email']
            msg['Subject'] = email_data['subject']
            
            # Add text content
            text_part = MIMEText(email_data['message'], 'plain')
            msg.attach(text_part)
            
            # Add HTML content if provided
            if email_data.get('html_message'):
                html_part = MIMEText(email_data['html_message'], 'html')
                msg.attach(html_part)
            
            # Add attachments if any
            for attachment in email_data.get('attachments', []):
                with open(attachment['path'], 'rb') as file:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(file.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["name"]}'
                    )
                    msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {email_data['to_email']}")
            
        except Exception as e:
            logger.error(f"Failed to send email to {email_data['to_email']}: {e}")

# Global email service
email_service = None

def init_email_service(app):
    """Initialize email service"""
    global email_service
    with app.app_context():
        email_service = EmailNotificationService()
    return email_service

# Routes
@notifications_bp.route('/notifications', methods=['GET'])
@jwt_required()
@safe_api_call
def get_notifications():
    """Get user notifications"""
    current_user_id = get_jwt_identity()
    
    # Query parameters
    limit = min(int(request.args.get('limit', 50)), 100)
    offset = int(request.args.get('offset', 0))
    unread_only = request.args.get('unread_only', 'false').lower() == 'true'
    
    db = next(get_db())
    
    try:
        query = db.query(Notification).filter(Notification.user_id == current_user_id)
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        # Filter expired notifications
        query = query.filter(
            (Notification.expires_at.is_(None)) | 
            (Notification.expires_at > datetime.now())
        )
        
        total_count = query.count()
        notifications = query.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()
        
        unread_count = db.query(Notification).filter(
            Notification.user_id == current_user_id,
            Notification.is_read == False,
            (Notification.expires_at.is_(None)) | (Notification.expires_at > datetime.now())
        ).count()
        
        return jsonify({
            'notifications': [notification.to_dict() for notification in notifications],
            'pagination': {
                'total_count': total_count,
                'unread_count': unread_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }
        })
        
    finally:
        db.close()

@notifications_bp.route('/notifications/<int:notification_id>/read', methods=['POST'])
@jwt_required()
@safe_api_call
def mark_notification_read(notification_id: int):
    """Mark notification as read"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == current_user_id
        ).first()
        
        if not notification:
            raise NotFoundError("Notification not found")
        
        if not notification.is_read:
            notification.is_read = True
            notification.read_at = datetime.now()
            db.commit()
        
        return jsonify({
            'message': 'Notification marked as read',
            'notification': notification.to_dict()
        })
        
    finally:
        db.close()

@notifications_bp.route('/notifications/read-all', methods=['POST'])
@jwt_required()
@safe_api_call
def mark_all_read():
    """Mark all notifications as read"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        updated_count = db.query(Notification).filter(
            Notification.user_id == current_user_id,
            Notification.is_read == False
        ).update({
            'is_read': True,
            'read_at': datetime.now()
        })
        
        db.commit()
        
        return jsonify({
            'message': f'Marked {updated_count} notifications as read',
            'updated_count': updated_count
        })
        
    finally:
        db.close()

@notifications_bp.route('/price-alerts', methods=['GET'])
@jwt_required()
@safe_api_call
def get_price_alerts():
    """Get user's price alerts"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        alerts = db.query(PriceAlert).filter(
            PriceAlert.user_id == current_user_id
        ).order_by(PriceAlert.created_at.desc()).all()
        
        return jsonify({
            'price_alerts': [alert.to_dict() for alert in alerts]
        })
        
    finally:
        db.close()

@notifications_bp.route('/price-alerts', methods=['POST'])
@jwt_required()
@safe_api_call
def create_price_alert():
    """Create a new price alert"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = CreatePriceAlertRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid price alert data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        # Check if user already has an alert for this ticker and condition
        existing_alert = db.query(PriceAlert).filter(
            PriceAlert.user_id == current_user_id,
            PriceAlert.ticker == request_data.ticker,
            PriceAlert.condition == request_data.condition,
            PriceAlert.target_value == request_data.target_value,
            PriceAlert.is_active == True
        ).first()
        
        if existing_alert:
            raise ValidationException("Similar price alert already exists")
        
        # Create new alert
        alert = PriceAlert(
            user_id=current_user_id,
            ticker=request_data.ticker,
            condition=request_data.condition,
            target_value=request_data.target_value,
            notification_channel=request_data.notification_channel
        )
        
        db.add(alert)
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Price alert created: User {current_user_id}, Ticker {request_data.ticker}")
        
        return jsonify({
            'message': 'Price alert created successfully',
            'alert': alert.to_dict()
        }), 201
        
    finally:
        db.close()

@notifications_bp.route('/price-alerts/<int:alert_id>', methods=['DELETE'])
@jwt_required()
@safe_api_call
def delete_price_alert(alert_id: int):
    """Delete a price alert"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        alert = db.query(PriceAlert).filter(
            PriceAlert.id == alert_id,
            PriceAlert.user_id == current_user_id
        ).first()
        
        if not alert:
            raise NotFoundError("Price alert not found")
        
        db.delete(alert)
        db.commit()
        
        return jsonify({'message': 'Price alert deleted successfully'})
        
    finally:
        db.close()

# Utility functions
def create_notification(user_id: int, title: str, message: str, 
                       notification_type: NotificationType = NotificationType.SYSTEM_UPDATE,
                       priority: NotificationPriority = NotificationPriority.MEDIUM,
                       channel: NotificationChannel = NotificationChannel.IN_APP,
                       data: Dict = None,
                       expires_at: Optional[datetime] = None) -> int:
    """Create a new notification"""
    db = next(get_db())
    
    try:
        notification = Notification(
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            priority=priority,
            channel=channel,
            data=json.dumps(data or {}),
            expires_at=expires_at
        )
        
        db.add(notification)
        db.commit()
        db.refresh(notification)
        
        # Send via WebSocket if user is connected
        if channel in [NotificationChannel.IN_APP, NotificationChannel.BOTH]:
            send_notification(user_id, {
                'id': notification.id,
                'title': title,
                'message': message,
                'type': notification_type.value,
                'priority': priority.value,
                'data': data or {}
            })
        
        # Send email if requested
        if channel in [NotificationChannel.EMAIL, NotificationChannel.BOTH] and email_service:
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.email:
                email_service.send_email(
                    to_email=user.email,
                    subject=f"Portfolio Optimizer: {title}",
                    message=message
                )
        
        logger.info(f"Notification created: User {user_id}, Type {notification_type.value}")
        return notification.id
        
    finally:
        db.close()

def check_price_alerts():
    """Check and trigger price alerts (called by background scheduler)"""
    db = next(get_db())
    
    try:
        # Get all active alerts
        active_alerts = db.query(PriceAlert).filter(
            PriceAlert.is_active == True,
            PriceAlert.is_triggered == False
        ).all()
        
        if not active_alerts:
            return
        
        # Group alerts by ticker for efficient price fetching
        ticker_alerts = {}
        for alert in active_alerts:
            if alert.ticker not in ticker_alerts:
                ticker_alerts[alert.ticker] = []
            ticker_alerts[alert.ticker].append(alert)
        
        # Check each ticker
        for ticker, alerts in ticker_alerts.items():
            try:
                # Fetch current price
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                
                if hist.empty:
                    continue
                
                current_price = float(hist['Close'].iloc[-1])
                
                # Check each alert for this ticker
                for alert in alerts:
                    triggered = False
                    
                    if alert.condition == 'above' and current_price > alert.target_value:
                        triggered = True
                    elif alert.condition == 'below' and current_price < alert.target_value:
                        triggered = True
                    elif alert.condition == 'change_percent' and alert.current_value:
                        change_pct = ((current_price - alert.current_value) / alert.current_value) * 100
                        if abs(change_pct) >= abs(alert.target_value):
                            triggered = True
                    
                    # Update current value
                    alert.current_value = current_price
                    
                    if triggered:
                        # Trigger alert
                        alert.is_triggered = True
                        alert.triggered_at = datetime.now()
                        
                        # Create notification
                        title = f"Price Alert: {ticker}"
                        message = f"{ticker} price ({current_price:.2f}) triggered your alert condition: {alert.condition} {alert.target_value}"
                        
                        create_notification(
                            user_id=alert.user_id,
                            title=title,
                            message=message,
                            notification_type=NotificationType.PRICE_ALERT,
                            priority=NotificationPriority.HIGH,
                            channel=alert.notification_channel,
                            data={'ticker': ticker, 'price': current_price, 'alert_id': alert.id}
                        )
                        
                        logger.info(f"Price alert triggered: Alert {alert.id}, {ticker} @ {current_price}")
                
            except Exception as e:
                logger.error(f"Error checking price alerts for {ticker}: {e}")
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in check_price_alerts: {e}")
        db.rollback()
    finally:
        db.close()

# Background scheduler would call this function periodically
# For example, using APScheduler:
# scheduler.add_job(check_price_alerts, 'interval', minutes=5)