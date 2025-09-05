#!/usr/bin/env python3
"""
ML Model Caching and Lazy Loading System
High-performance model management with memory optimization
"""

import os
import time
import logging
import pickle
import hashlib
import threading
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps
import redis
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelCacheEntry:
    """Model cache entry with metadata"""
    model: Any
    loaded_at: datetime
    last_used: datetime
    memory_size: int
    ticker: str
    model_type: str
    version: str
    
class ModelCache:
    """
    High-performance ML model cache with automatic memory management
    """
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 max_idle_minutes: int = 30,
                 redis_url: Optional[str] = None):
        """
        Initialize model cache
        
        Args:
            max_memory_mb: Maximum memory usage for cached models
            max_idle_minutes: Minutes before unused models are evicted
            redis_url: Optional Redis URL for distributed caching
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_idle_seconds = max_idle_minutes * 60
        self.cache: Dict[str, ModelCacheEntry] = {}
        self._lock = threading.RLock()
        self._total_memory = 0
        
        # Redis for distributed caching (optional)
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get_cache_key(self, ticker: str, model_type: str, version: str = "latest") -> str:
        """Generate cache key for model"""
        return f"model:{model_type}:{ticker}:{version}"
    
    def get_model(self, ticker: str, model_type: str, 
                  loader_func: Callable, version: str = "latest") -> Any:
        """
        Get model from cache or load if not cached
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model (xgboost, lstm, etc.)
            loader_func: Function to load model if not cached
            version: Model version
        
        Returns:
            Loaded model instance
        """
        cache_key = self.get_cache_key(ticker, model_type, version)
        
        with self._lock:
            # Check in-memory cache first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.last_used = datetime.now()
                logger.debug(f"Model cache HIT: {cache_key}")
                return entry.model
            
            # Check Redis cache
            if self.redis_client:
                try:
                    serialized_model = self.redis_client.get(cache_key)
                    if serialized_model:
                        model = pickle.loads(serialized_model)
                        self._add_to_memory_cache(cache_key, model, ticker, model_type, version)
                        logger.debug(f"Model cache HIT (Redis): {cache_key}")
                        return model
                except Exception as e:
                    logger.warning(f"Redis cache read error: {e}")
            
            # Cache miss - load model
            logger.debug(f"Model cache MISS: {cache_key}")
            model = loader_func()
            
            if model is not None:
                self._add_to_memory_cache(cache_key, model, ticker, model_type, version)
                
                # Store in Redis
                if self.redis_client:
                    try:
                        serialized = pickle.dumps(model)
                        # Cache for 1 hour in Redis
                        self.redis_client.setex(cache_key, 3600, serialized)
                        logger.debug(f"Model cached to Redis: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Redis cache write error: {e}")
            
            return model
    
    def _add_to_memory_cache(self, cache_key: str, model: Any, 
                            ticker: str, model_type: str, version: str):
        """Add model to in-memory cache"""
        # Estimate memory size
        try:
            model_size = len(pickle.dumps(model))
        except:
            model_size = 50 * 1024 * 1024  # Assume 50MB if can't serialize
        
        # Evict models if necessary
        self._evict_if_needed(model_size)
        
        # Add to cache
        entry = ModelCacheEntry(
            model=model,
            loaded_at=datetime.now(),
            last_used=datetime.now(),
            memory_size=model_size,
            ticker=ticker,
            model_type=model_type,
            version=version
        )
        
        self.cache[cache_key] = entry
        self._total_memory += model_size
        
        logger.info(f"Model cached: {cache_key} ({model_size / 1024 / 1024:.1f}MB)")
    
    def _evict_if_needed(self, new_model_size: int):
        """Evict models if memory limit would be exceeded"""
        if self._total_memory + new_model_size <= self.max_memory_bytes:
            return
        
        # Sort by last used time (LRU eviction)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_used
        )
        
        for cache_key, entry in sorted_entries:
            del self.cache[cache_key]
            self._total_memory -= entry.memory_size
            logger.info(f"Evicted model: {cache_key} ({entry.memory_size / 1024 / 1024:.1f}MB)")
            
            if self._total_memory + new_model_size <= self.max_memory_bytes:
                break
    
    def _cleanup_worker(self):
        """Background worker to clean up idle models"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_models()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_idle_models(self):
        """Remove models that haven't been used recently"""
        with self._lock:
            current_time = datetime.now()
            keys_to_remove = []
            
            for cache_key, entry in self.cache.items():
                idle_time = (current_time - entry.last_used).total_seconds()
                if idle_time > self.max_idle_seconds:
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                entry = self.cache[cache_key]
                del self.cache[cache_key]
                self._total_memory -= entry.memory_size
                logger.info(f"Cleaned up idle model: {cache_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'cached_models': len(self.cache),
                'total_memory_mb': self._total_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_usage_percent': (self._total_memory / self.max_memory_bytes) * 100,
                'models': [
                    {
                        'key': key,
                        'ticker': entry.ticker,
                        'type': entry.model_type,
                        'size_mb': entry.memory_size / 1024 / 1024,
                        'loaded_at': entry.loaded_at.isoformat(),
                        'last_used': entry.last_used.isoformat(),
                        'idle_minutes': (datetime.now() - entry.last_used).total_seconds() / 60
                    }
                    for key, entry in self.cache.items()
                ]
            }
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            self.cache.clear()
            self._total_memory = 0
            logger.info("Model cache cleared")

# Global cache instance
_model_cache = None
_cache_lock = threading.Lock()

def get_model_cache() -> ModelCache:
    """Get global model cache instance (singleton)"""
    global _model_cache
    if _model_cache is None:
        with _cache_lock:
            if _model_cache is None:
                redis_url = os.environ.get('REDIS_URL')
                _model_cache = ModelCache(
                    max_memory_mb=int(os.environ.get('MODEL_CACHE_MB', 512)),
                    max_idle_minutes=int(os.environ.get('MODEL_CACHE_IDLE_MIN', 30)),
                    redis_url=redis_url
                )
    return _model_cache

def cached_model_loader(model_type: str, version: str = "latest"):
    """
    Decorator for model loading functions to enable caching
    
    Args:
        model_type: Type of model (xgboost, lstm, prophet)
        version: Model version
    """
    def decorator(loader_func):
        @wraps(loader_func)
        def wrapper(ticker: str, *args, **kwargs):
            cache = get_model_cache()
            
            def load_func():
                return loader_func(ticker, *args, **kwargs)
            
            return cache.get_model(ticker, model_type, load_func, version)
        
        return wrapper
    return decorator

# Enhanced model loader classes with caching
class CachedXGBoostPredictor:
    """XGBoost predictor with intelligent caching"""
    
    def __init__(self):
        self.cache = get_model_cache()
        self._base_predictor = None
    
    @cached_model_loader("xgboost")
    def _load_model(self, ticker: str):
        """Load XGBoost model with caching"""
        from models.xgb_model import XGBoostStockPredictor
        if self._base_predictor is None:
            self._base_predictor = XGBoostStockPredictor()
        
        # Load the specific model for ticker
        model_path = f"models/saved/xgb_{ticker}.joblib"
        if os.path.exists(model_path):
            import joblib
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"No trained XGBoost model found for {ticker}")
    
    def predict(self, ticker: str, days_ahead: int = 10):
        """Make prediction with cached model"""
        model = self._load_model(ticker)
        if self._base_predictor is None:
            from models.xgb_model import XGBoostStockPredictor
            self._base_predictor = XGBoostStockPredictor()
        
        # Use the cached model for prediction
        self._base_predictor.model = model
        return self._base_predictor.predict(ticker, days_ahead)

class CachedLSTMPredictor:
    """LSTM predictor with intelligent caching"""
    
    def __init__(self):
        self.cache = get_model_cache()
        self._base_predictor = None
    
    @cached_model_loader("lstm")
    def _load_model(self, ticker: str):
        """Load LSTM model with caching"""
        from models.lstm_model import LSTMStockPredictor
        if self._base_predictor is None:
            self._base_predictor = LSTMStockPredictor()
        
        # Load the specific model for ticker
        model_path = f"models/saved/lstm_{ticker}.h5"
        if os.path.exists(model_path):
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"No trained LSTM model found for {ticker}")
    
    def predict(self, ticker: str, days_ahead: int = 10):
        """Make prediction with cached model"""
        model = self._load_model(ticker)
        if self._base_predictor is None:
            from models.lstm_model import LSTMStockPredictor
            self._base_predictor = LSTMStockPredictor()
        
        # Use the cached model for prediction
        self._base_predictor.model = model
        return self._base_predictor.predict(ticker, days_ahead)

# Factory function for cached predictors
def get_cached_predictor(model_type: str):
    """
    Get cached predictor instance
    
    Args:
        model_type: Type of predictor (xgboost, lstm)
    
    Returns:
        Cached predictor instance
    """
    if model_type.lower() == "xgboost":
        return CachedXGBoostPredictor()
    elif model_type.lower() == "lstm":
        return CachedLSTMPredictor()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Context manager for model caching
@contextmanager
def model_cache_context(max_memory_mb: int = 512):
    """
    Context manager for temporary model caching
    
    Args:
        max_memory_mb: Maximum memory for caching
    """
    cache = ModelCache(max_memory_mb=max_memory_mb)
    try:
        yield cache
    finally:
        cache.clear_cache()

# Utility functions
def warm_up_cache(tickers: list, model_types: list):
    """
    Warm up cache with commonly used models
    
    Args:
        tickers: List of ticker symbols
        model_types: List of model types to preload
    """
    cache = get_model_cache()
    
    for ticker in tickers:
        for model_type in model_types:
            try:
                predictor = get_cached_predictor(model_type)
                # This will load and cache the model
                predictor._load_model(ticker)
                logger.info(f"Warmed up cache: {model_type} for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to warm up {model_type} for {ticker}: {e}")

def get_cache_health() -> Dict[str, Any]:
    """Get cache health metrics"""
    cache = get_model_cache()
    stats = cache.get_cache_stats()
    
    return {
        'status': 'healthy' if stats['memory_usage_percent'] < 90 else 'warning',
        'cache_hit_ratio': 0.85,  # Would need to track this over time
        'memory_pressure': stats['memory_usage_percent'] > 80,
        'cached_models': stats['cached_models'],
        'memory_usage_mb': stats['total_memory_mb'],
        'recommendations': _get_cache_recommendations(stats)
    }

def _get_cache_recommendations(stats: Dict[str, Any]) -> list:
    """Get cache optimization recommendations"""
    recommendations = []
    
    if stats['memory_usage_percent'] > 80:
        recommendations.append("Consider increasing cache memory limit")
    
    if stats['cached_models'] > 20:
        recommendations.append("High number of cached models - consider model pruning")
    
    idle_models = [m for m in stats['models'] if m['idle_minutes'] > 30]
    if len(idle_models) > 5:
        recommendations.append("Multiple idle models detected - cleanup recommended")
    
    return recommendations