#!/usr/bin/env python3
"""
Caching utilities for performance optimization
"""
import redis
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info("Redis cache initialized successfully")
        except (redis.RedisError, ConnectionError):
            logger.warning("Redis cache unavailable, using in-memory fallback")
            self.redis_client = None
            self.available = False
            self._memory_cache: Dict[str, Any] = {}
    
    def _get_key(self, key: str, prefix: str = "portfolio_app") -> str:
        """Generate cache key with prefix"""
        return f"{prefix}:{key}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        cache_key = self._get_key(key)
        
        try:
            if self.available:
                value = self.redis_client.get(cache_key)
                if value:
                    return json.loads(value)
            else:
                # Fallback to in-memory cache
                if cache_key in self._memory_cache:
                    item = self._memory_cache[cache_key]
                    if item['expires'] > datetime.now():
                        return item['value']
                    else:
                        del self._memory_cache[cache_key]
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL in seconds"""
        cache_key = self._get_key(key)
        
        try:
            if self.available:
                return self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            else:
                # Fallback to in-memory cache
                self._memory_cache[cache_key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
                
                # Simple cleanup - remove expired items
                current_time = datetime.now()
                expired_keys = [
                    k for k, v in self._memory_cache.items() 
                    if v['expires'] < current_time
                ]
                for k in expired_keys:
                    del self._memory_cache[k]
                
                return True
        except (redis.RedisError, TypeError) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._get_key(key)
        
        try:
            if self.available:
                return bool(self.redis_client.delete(cache_key))
            else:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    return True
        except redis.RedisError as e:
            logger.error(f"Cache delete error: {e}")
        
        return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern"""
        if not self.available:
            # Simple pattern matching for in-memory cache
            pattern_key = self._get_key(pattern.replace('*', ''))
            matching_keys = [
                k for k in self._memory_cache.keys() 
                if k.startswith(pattern_key)
            ]
            for k in matching_keys:
                del self._memory_cache[k]
            return len(matching_keys)
        
        try:
            keys = self.redis_client.keys(self._get_key(pattern))
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0

# Global cache instance
cache = CacheManager()

def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def cached(ttl: int = 300, key_prefix: str = "func"):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_stock_data(ticker: str, data: Dict[str, Any], ttl: int = 300):
    """Cache stock data with ticker-specific key"""
    cache_key = f"stock_data:{ticker.upper()}"
    cache.set(cache_key, data, ttl)

def get_cached_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get cached stock data"""
    cache_key = f"stock_data:{ticker.upper()}"
    return cache.get(cache_key)

def cache_prediction(ticker: str, days: int, models: str, prediction: Dict[str, Any], ttl: int = 1800):
    """Cache prediction results"""
    cache_key = f"prediction:{ticker.upper()}:{days}:{models}"
    cache.set(cache_key, prediction, ttl)

def get_cached_prediction(ticker: str, days: int, models: str) -> Optional[Dict[str, Any]]:
    """Get cached prediction"""
    cache_key = f"prediction:{ticker.upper()}:{days}:{models}"
    return cache.get(cache_key)

def cache_optimization_result(tickers: List[str], strategy: str, params: Dict[str, Any], 
                            result: Dict[str, Any], ttl: int = 900):
    """Cache portfolio optimization results"""
    # Sort tickers for consistent cache key
    sorted_tickers = sorted([t.upper() for t in tickers])
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    
    cache_key = f"optimization:{':'.join(sorted_tickers)}:{strategy}:{params_hash}"
    cache.set(cache_key, result, ttl)

def get_cached_optimization_result(tickers: List[str], strategy: str, 
                                 params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get cached optimization result"""
    sorted_tickers = sorted([t.upper() for t in tickers])
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    
    cache_key = f"optimization:{':'.join(sorted_tickers)}:{strategy}:{params_hash}"
    return cache.get(cache_key)

def invalidate_stock_cache(ticker: str):
    """Invalidate all cache entries for a specific ticker"""
    ticker = ticker.upper()
    patterns = [
        f"stock_data:{ticker}",
        f"prediction:{ticker}:*",
        f"*:{ticker}:*"
    ]
    
    for pattern in patterns:
        cache.clear_pattern(pattern)

def clear_expired_cache():
    """Clear expired cache entries - maintenance function"""
    if cache.available:
        try:
            # Redis automatically handles TTL, but we can do maintenance here
            logger.info("Cache maintenance completed")
        except Exception as e:
            logger.error(f"Cache maintenance error: {e}")

# Model-specific caching
class ModelCache:
    """Cache for ML model instances and predictions"""
    
    def __init__(self):
        self._model_instances: Dict[str, Any] = {}
        self._prediction_cache: Dict[str, Any] = {}
    
    def get_model(self, model_type: str, ticker: str) -> Optional[Any]:
        """Get cached model instance"""
        key = f"{model_type}:{ticker.upper()}"
        return self._model_instances.get(key)
    
    def set_model(self, model_type: str, ticker: str, model: Any):
        """Cache model instance"""
        key = f"{model_type}:{ticker.upper()}"
        self._model_instances[key] = model
    
    def clear_models(self, model_type: str = None):
        """Clear model cache"""
        if model_type:
            keys_to_remove = [k for k in self._model_instances.keys() if k.startswith(model_type)]
            for key in keys_to_remove:
                del self._model_instances[key]
        else:
            self._model_instances.clear()

model_cache = ModelCache()