#!/usr/bin/env python3
"""
Comprehensive Input Validation Schemas
Secure Pydantic schemas for all API endpoints
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum
import re

# Security constants
MAX_TICKER_LENGTH = 10
MAX_TICKERS_PER_REQUEST = 50
MIN_PASSWORD_LENGTH = 8
MAX_STRING_LENGTH = 255
MAX_TEXT_LENGTH = 2000
MAX_ALLOCATION = 1.0
MIN_ALLOCATION = 0.0

class OptimizationStrategy(str, Enum):
    """Valid optimization strategies"""
    SHARPE = "sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    TARGET_RETURN = "target_return"

class ModelType(str, Enum):
    """Valid ML model types"""
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    ALL = "all"

class RiskLevel(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

# Validators
def validate_ticker(ticker: str) -> str:
    """Validate stock ticker format"""
    if not ticker or len(ticker) > MAX_TICKER_LENGTH:
        raise ValueError(f"Ticker must be 1-{MAX_TICKER_LENGTH} characters")
    
    # Allow only alphanumeric characters, dots, and hyphens
    if not re.match(r'^[A-Z0-9.-]+$', ticker.upper()):
        raise ValueError("Ticker contains invalid characters")
    
    return ticker.upper().strip()

def validate_email(email: str) -> str:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email format")
    return email.lower().strip()

def validate_password_strength(password: str) -> str:
    """Validate password strength"""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    
    # Check for complexity
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        raise ValueError("Password must contain uppercase, lowercase, digit, and special character")
    
    return password

# Portfolio Optimization Schemas
class PortfolioHolding(BaseModel):
    """Individual portfolio holding"""
    ticker: str = Field(..., description="Stock ticker symbol")
    weight: float = Field(..., ge=MIN_ALLOCATION, le=MAX_ALLOCATION, description="Portfolio weight (0-1)")
    
    @validator('ticker')
    def validate_ticker_format(cls, v):
        return validate_ticker(v)

class OptimizationRequest(BaseModel):
    """Portfolio optimization request"""
    tickers: List[str] = Field(..., min_items=2, max_items=MAX_TICKERS_PER_REQUEST)
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.SHARPE)
    start_date: date = Field(..., description="Start date for historical data")
    end_date: date = Field(..., description="End date for historical data")
    target_return: Optional[float] = Field(None, ge=0.0, le=2.0, description="Target return for target_return strategy")
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=0.2, description="Risk-free rate")
    
    @validator('tickers', each_item=True)
    def validate_tickers(cls, v):
        return validate_ticker(v)
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        if v > date.today():
            raise ValueError("End date cannot be in the future")
        return v
    
    @root_validator
    def validate_target_return_strategy(cls, values):
        if values.get('strategy') == OptimizationStrategy.TARGET_RETURN:
            if not values.get('target_return'):
                raise ValueError("target_return is required for target_return strategy")
        return values

class AdvancedOptimizationRequest(OptimizationRequest):
    """Advanced portfolio optimization with additional constraints"""
    min_weights: Optional[Dict[str, float]] = Field(None, description="Minimum weights per ticker")
    max_weights: Optional[Dict[str, float]] = Field(None, description="Maximum weights per ticker")
    sector_constraints: Optional[Dict[str, float]] = Field(None, description="Sector allocation constraints")
    max_volatility: Optional[float] = Field(None, ge=0.0, le=2.0, description="Maximum portfolio volatility")
    
    @validator('min_weights', 'max_weights')
    def validate_weight_constraints(cls, v):
        if v:
            for ticker, weight in v.items():
                validate_ticker(ticker)
                if not MIN_ALLOCATION <= weight <= MAX_ALLOCATION:
                    raise ValueError(f"Weight for {ticker} must be between {MIN_ALLOCATION} and {MAX_ALLOCATION}")
        return v

# ML Prediction Schemas
class PredictionRequest(BaseModel):
    """Stock prediction request"""
    ticker: str = Field(..., description="Stock ticker symbol")
    days_ahead: int = Field(default=10, ge=1, le=30, description="Days to predict ahead")
    models: ModelType = Field(default=ModelType.ALL, description="Models to use for prediction")
    
    @validator('ticker')
    def validate_ticker_format(cls, v):
        return validate_ticker(v)

class BatchPredictionRequest(BaseModel):
    """Batch stock prediction request"""
    tickers: List[str] = Field(..., min_items=1, max_items=10, description="Stock ticker symbols")
    days_ahead: int = Field(default=10, ge=1, le=30, description="Days to predict ahead")
    models: ModelType = Field(default=ModelType.ENSEMBLE, description="Models to use for prediction")
    
    @validator('tickers', each_item=True)
    def validate_tickers(cls, v):
        return validate_ticker(v)

class ModelTrainingRequest(BaseModel):
    """Model training request"""
    ticker: str = Field(..., description="Stock ticker symbol")
    models: List[ModelType] = Field(default=[ModelType.XGBOOST], description="Models to train")
    period: str = Field(default="2y", regex=r'^[1-9]\d*[dmy]$', description="Training data period")
    
    @validator('ticker')
    def validate_ticker_format(cls, v):
        return validate_ticker(v)
    
    @validator('models')
    def validate_training_models(cls, v):
        # Prophet doesn't need training
        valid_training_models = {ModelType.XGBOOST, ModelType.LSTM}
        for model in v:
            if model not in valid_training_models:
                raise ValueError(f"Model {model} does not support training")
        return v

# Authentication Schemas
class UserRegistrationRequest(BaseModel):
    """User registration request"""
    email: str = Field(..., max_length=MAX_STRING_LENGTH)
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    risk_tolerance: RiskLevel = Field(default=RiskLevel.MODERATE)
    investment_horizon: int = Field(default=5, ge=1, le=50, description="Investment horizon in years")
    
    @validator('email')
    def validate_email_format(cls, v):
        return validate_email(v)
    
    @validator('password')
    def validate_password_strength(cls, v):
        return validate_password_strength(v)
    
    @validator('first_name', 'last_name')
    def validate_names(cls, v):
        v = v.strip()
        if not v or not re.match(r'^[a-zA-Z\s\'-]+$', v):
            raise ValueError("Names must contain only letters, spaces, hyphens, and apostrophes")
        return v

class UserLoginRequest(BaseModel):
    """User login request"""
    email: str = Field(..., max_length=MAX_STRING_LENGTH)
    password: str = Field(..., description="User password")
    
    @validator('email')
    def validate_email_format(cls, v):
        return validate_email(v)

class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, description="New password")
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        return validate_password_strength(v)

# Portfolio Import Schemas
class PortfolioImportRequest(BaseModel):
    """Portfolio CSV import request"""
    csv_data: str = Field(..., max_length=100000, description="CSV data as string")
    has_header: bool = Field(default=True, description="Whether CSV has header row")
    
    @validator('csv_data')
    def validate_csv_content(cls, v):
        # Basic CSV validation
        if not v.strip():
            raise ValueError("CSV data cannot be empty")
        
        lines = v.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("CSV must have at least 2 lines")
        
        return v

# Notification Schemas
class PriceAlertRequest(BaseModel):
    """Price alert creation request"""
    ticker: str = Field(..., description="Stock ticker symbol")
    condition: str = Field(..., regex=r'^(above|below|change)$', description="Alert condition")
    target_price: float = Field(..., gt=0, description="Target price for alert")
    percentage_change: Optional[float] = Field(None, ge=-100, le=1000, description="Percentage change threshold")
    
    @validator('ticker')
    def validate_ticker_format(cls, v):
        return validate_ticker(v)
    
    @root_validator
    def validate_condition_parameters(cls, values):
        condition = values.get('condition')
        target_price = values.get('target_price')
        percentage_change = values.get('percentage_change')
        
        if condition in ['above', 'below'] and not target_price:
            raise ValueError(f"target_price is required for {condition} condition")
        
        if condition == 'change' and percentage_change is None:
            raise ValueError("percentage_change is required for change condition")
        
        return values

# Analytics Schemas
class PortfolioAnalyticsRequest(BaseModel):
    """Portfolio analytics request"""
    holdings: List[PortfolioHolding] = Field(..., min_items=1, max_items=MAX_TICKERS_PER_REQUEST)
    benchmark: Optional[str] = Field(default="SPY", description="Benchmark ticker")
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")
    
    @validator('holdings')
    def validate_portfolio_weights(cls, v):
        total_weight = sum(holding.weight for holding in v)
        if not (0.95 <= total_weight <= 1.05):  # Allow small rounding errors
            raise ValueError("Portfolio weights must sum to approximately 1.0")
        return v
    
    @validator('benchmark')
    def validate_benchmark_ticker(cls, v):
        if v:
            return validate_ticker(v)
        return v

# Generic Response Schemas
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)

class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = Field(default=True)
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now)

# Utility function for schema validation
def validate_request(schema_class: type, data: Dict[str, Any]) -> BaseModel:
    """
    Validate request data against schema
    Raises ValueError with detailed error messages
    """
    try:
        return schema_class(**data)
    except ValueError as e:
        raise ValueError(f"Validation error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid request format: {str(e)}")