#!/usr/bin/env python3
"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class OptimizationStrategy(str, Enum):
    """Portfolio optimization strategies"""
    SHARPE = "sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    TARGET_RETURN = "target_return"

class MLModel(str, Enum):
    """Available ML models"""
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    ALL = "all"

class Signal(str, Enum):
    """Trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Portfolio Optimization Schemas
class PortfolioHolding(BaseModel):
    """Individual portfolio holding"""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    shares: float = Field(..., gt=0, description="Number of shares")
    avg_price: Optional[float] = Field(None, gt=0, description="Average purchase price")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()

class OptimizePortfolioRequest(BaseModel):
    """Basic portfolio optimization request"""
    tickers: List[str] = Field(..., min_items=2, max_items=20, description="List of stock tickers")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.SHARPE)
    start_date: str = Field(..., description="Start date for historical data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for historical data (YYYY-MM-DD)")
    investment_amount: Optional[float] = Field(10000, gt=0, le=10000000, description="Total investment amount")
    
    @validator('tickers')
    def validate_tickers(cls, v):
        return [ticker.upper().strip() for ticker in v]

class AdvancedOptimizeRequest(OptimizePortfolioRequest):
    """Advanced portfolio optimization request"""
    target_return: Optional[float] = Field(None, ge=0, le=1, description="Target annual return (0-1)")
    risk_tolerance: Optional[float] = Field(None, ge=0, le=1, description="Risk tolerance (0-1)")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional constraints")

# Prediction Schemas
class PredictionRequest(BaseModel):
    """Stock prediction request"""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    days: int = Field(default=10, ge=1, le=30, description="Number of days to predict")
    models: MLModel = Field(default=MLModel.ALL, description="ML models to use")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    tickers: List[str] = Field(..., min_items=1, max_items=10, description="List of stock tickers")
    days: int = Field(default=10, ge=1, le=30, description="Number of days to predict")
    models: MLModel = Field(default=MLModel.ENSEMBLE, description="ML models to use")
    
    @validator('tickers')
    def validate_tickers(cls, v):
        return [ticker.upper().strip() for ticker in v]

class TrainingRequest(BaseModel):
    """Model training request"""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    models: List[MLModel] = Field(default=[MLModel.XGBOOST], description="Models to train")
    period: str = Field(default="2y", description="Training period (1y, 2y, 5y, etc.)")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()
    
    @validator('models')
    def validate_models(cls, v):
        allowed = [MLModel.XGBOOST, MLModel.LSTM]
        for model in v:
            if model not in allowed:
                raise ValueError(f"Model {model} not allowed for training")
        return v

# Response Schemas
class StockInfo(BaseModel):
    """Stock information response"""
    ticker: str
    name: str
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    market_cap: Optional[int]
    sector: Optional[str]
    industry: Optional[str]
    currency: str = "USD"
    last_updated: datetime

class PredictionPoint(BaseModel):
    """Individual prediction point"""
    day: int
    predicted_price: float
    predicted_return: float
    confidence_lower: float
    confidence_upper: float
    date: str
    models_used: Optional[List[str]] = None
    model_weights: Optional[Dict[str, float]] = None

class TradingSignal(BaseModel):
    """Trading signal response"""
    action: Signal
    strength: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=100)
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    expected_return: Optional[float] = None
    max_downside: Optional[float] = None
    time_horizon: Optional[str] = None

class PredictionSummary(BaseModel):
    """Prediction summary statistics"""
    avg_predicted_return: float
    max_predicted_price: float
    min_predicted_price: float
    volatility_estimate: float
    trend_direction: Literal["bullish", "bearish"]

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    ticker: str
    model: str
    current_price: float
    predictions: List[PredictionPoint]
    summary: PredictionSummary
    trading_signal: TradingSignal
    ensemble_info: Optional[Dict[str, Any]] = None

class OptimizationResult(BaseModel):
    """Portfolio optimization result"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]
    allocation: Dict[str, float]
    total_value: float

# Error Response Schema
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"]
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.now)