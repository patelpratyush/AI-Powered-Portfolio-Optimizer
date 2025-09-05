#!/usr/bin/env python3
"""
Database models and configuration using SQLAlchemy with connection pooling
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import logging
from utils.database_pool import initialize_database_pool, get_database_pool, get_db_session

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """User model for authentication and preferences"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # User preferences
    preferred_currency = Column(String(3), default='USD')
    risk_tolerance = Column(Float, default=0.5)  # 0-1 scale
    investment_horizon = Column(String(20), default='medium')  # short, medium, long
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'preferred_currency': self.preferred_currency,
            'risk_tolerance': self.risk_tolerance,
            'investment_horizon': self.investment_horizon
        }

class Portfolio(Base):
    """Portfolio model for saving user portfolios"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)  # Optional for guest users
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Portfolio composition (JSON field)
    holdings = Column(JSON)  # [{"ticker": "AAPL", "shares": 100, "avg_price": 150.0}]
    
    # Optimization settings
    strategy = Column(String(50), default='sharpe')
    target_return = Column(Float)
    risk_tolerance = Column(Float)
    constraints = Column(JSON)
    
    # Results cache
    last_optimization_result = Column(JSON)
    last_optimization_date = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'holdings': self.holdings,
            'strategy': self.strategy,
            'target_return': self.target_return,
            'risk_tolerance': self.risk_tolerance,
            'constraints': self.constraints,
            'last_optimization_result': self.last_optimization_result,
            'last_optimization_date': self.last_optimization_date.isoformat() if self.last_optimization_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class PredictionHistory(Base):
    """Store prediction history and results"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # prophet, xgboost, lstm, ensemble
    
    # Prediction parameters
    days_ahead = Column(Integer, default=10)
    prediction_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Prediction results (JSON)
    predictions = Column(JSON)  # Full prediction result
    summary_stats = Column(JSON)  # Summary statistics
    
    # Model performance tracking
    actual_price = Column(Float)  # Fill in later for backtesting
    prediction_accuracy = Column(Float)  # Calculated after actual results
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, ticker='{self.ticker}', model='{self.model_type}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_type': self.model_type,
            'days_ahead': self.days_ahead,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'predictions': self.predictions,
            'summary_stats': self.summary_stats,
            'actual_price': self.actual_price,
            'prediction_accuracy': self.prediction_accuracy,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ModelMetadata(Base):
    """Track ML model training and performance"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # xgboost, lstm
    
    # Training information
    training_start_date = Column(DateTime(timezone=True))
    training_end_date = Column(DateTime(timezone=True))
    training_period = Column(String(10))  # 1y, 2y, etc.
    
    # Model performance metrics
    r2_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    training_time_seconds = Column(Float)
    
    # Model file information
    model_file_path = Column(String(500))
    scaler_file_path = Column(String(500))
    feature_names = Column(JSON)
    
    # Metadata
    model_version = Column(String(50), default='1.0')
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, ticker='{self.ticker}', model='{self.model_type}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_type': self.model_type,
            'training_start_date': self.training_start_date.isoformat() if self.training_start_date else None,
            'training_end_date': self.training_end_date.isoformat() if self.training_end_date else None,
            'training_period': self.training_period,
            'r2_score': self.r2_score,
            'mae': self.mae,
            'mse': self.mse,
            'training_time_seconds': self.training_time_seconds,
            'model_file_path': self.model_file_path,
            'scaler_file_path': self.scaler_file_path,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

class APIUsageLog(Base):
    """Track API usage for monitoring and rate limiting"""
    __tablename__ = 'api_usage_log'
    
    id = Column(Integer, primary_key=True)
    endpoint = Column(String(200), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    
    # Request info
    user_id = Column(Integer)  # If authenticated
    ip_address = Column(String(45))  # IPv4/IPv6
    user_agent = Column(String(500))
    
    # Response info
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    error_message = Column(Text)
    
    # Request parameters (be careful with sensitive data)
    request_params = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<APIUsageLog(id={self.id}, endpoint='{self.endpoint}')>"

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = os.environ.get('DATABASE_URL', 'sqlite:///portfolio_optimizer.db')
        
        self.engine = create_engine(
            database_url,
            echo=os.environ.get('FLASK_ENV') == 'development',  # Log SQL in development
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database initialized: {database_url}")
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()

# Global database manager instance
db_manager = None

def init_database(database_url: str = None):
    """Initialize database with connection pooling"""
    global db_manager
    
    # Initialize connection pool
    from config import config
    import os
    
    env = os.environ.get('FLASK_ENV', 'development')
    from utils.database_pool import get_pool_config
    pool_config = get_pool_config(env)
    
    # Initialize pool with environment-specific configuration
    pool = initialize_database_pool(database_url, **pool_config)
    
    # Create tables using pooled connection
    Base.metadata.create_all(pool.engine)
    
    db_manager = DatabaseManager(database_url, use_pool=True)
    logger.info(f"Database initialized with connection pooling: {env} environment")
    
    return db_manager

def get_db() -> Session:
    """Get database session - for use in Flask routes with connection pooling"""
    try:
        # Use pooled session
        with get_db_session() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise

# Database utility functions
def save_portfolio(session: Session, portfolio_data: Dict[str, Any], user_id: Optional[int] = None) -> Portfolio:
    """Save portfolio to database"""
    portfolio = Portfolio(
        user_id=user_id,
        name=portfolio_data.get('name', f'Portfolio {datetime.now().strftime("%Y-%m-%d %H:%M")}'),
        description=portfolio_data.get('description'),
        holdings=portfolio_data.get('holdings', []),
        strategy=portfolio_data.get('strategy', 'sharpe'),
        target_return=portfolio_data.get('target_return'),
        risk_tolerance=portfolio_data.get('risk_tolerance'),
        constraints=portfolio_data.get('constraints', {}),
        last_optimization_result=portfolio_data.get('optimization_result')
    )
    
    session.add(portfolio)
    session.commit()
    session.refresh(portfolio)
    
    logger.info(f"Portfolio saved: {portfolio.id}")
    return portfolio

def save_prediction(session: Session, ticker: str, model_type: str, 
                   prediction_data: Dict[str, Any], days_ahead: int = 10) -> PredictionHistory:
    """Save prediction results to database"""
    prediction = PredictionHistory(
        ticker=ticker.upper(),
        model_type=model_type,
        days_ahead=days_ahead,
        predictions=prediction_data.get('predictions', []),
        summary_stats=prediction_data.get('summary', {})
    )
    
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    
    logger.info(f"Prediction saved: {ticker} - {model_type}")
    return prediction

def save_model_metadata(session: Session, ticker: str, model_type: str, 
                       metadata: Dict[str, Any]) -> ModelMetadata:
    """Save model training metadata"""
    model_meta = ModelMetadata(
        ticker=ticker.upper(),
        model_type=model_type,
        training_period=metadata.get('training_period'),
        r2_score=metadata.get('r2_score'),
        mae=metadata.get('mae'),
        mse=metadata.get('mse'),
        training_time_seconds=metadata.get('training_time'),
        model_file_path=metadata.get('model_file_path'),
        scaler_file_path=metadata.get('scaler_file_path'),
        feature_names=metadata.get('feature_names', [])
    )
    
    session.add(model_meta)
    session.commit()
    session.refresh(model_meta)
    
    logger.info(f"Model metadata saved: {ticker} - {model_type}")
    return model_meta

def get_user_portfolios(session: Session, user_id: int) -> List[Portfolio]:
    """Get all portfolios for a user"""
    return session.query(Portfolio).filter(Portfolio.user_id == user_id).all()

def get_prediction_history(session: Session, ticker: str, days: int = 30) -> List[PredictionHistory]:
    """Get prediction history for a ticker"""
    cutoff_date = datetime.now() - timedelta(days=days)
    return session.query(PredictionHistory).filter(
        PredictionHistory.ticker == ticker.upper(),
        PredictionHistory.created_at >= cutoff_date
    ).order_by(PredictionHistory.created_at.desc()).all()

def get_model_info(session: Session, ticker: str, model_type: str) -> Optional[ModelMetadata]:
    """Get model metadata"""
    return session.query(ModelMetadata).filter(
        ModelMetadata.ticker == ticker.upper(),
        ModelMetadata.model_type == model_type,
        ModelMetadata.is_active == True
    ).order_by(ModelMetadata.created_at.desc()).first()