#!/usr/bin/env python3
"""
Yahoo Finance API Caching Layer
High-performance caching for market data with intelligent refresh strategies
"""

import os
import time
import logging
import hashlib
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import redis
import yfinance as yf
import pandas as pd
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    expiry: float
    ticker: str
    data_type: str
    size_bytes: int

class YahooFinanceCache:
    """
    High-performance Yahoo Finance data cache with intelligent refresh
    """
    
    def __init__(self, 
                 redis_url: Optional[str] = None,
                 default_ttl: int = 300,  # 5 minutes
                 max_memory_mb: int = 100):
        """
        Initialize Yahoo Finance cache
        
        Args:
            redis_url: Optional Redis URL for distributed caching
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage
        """
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._memory_usage = 0
        
        # Redis client for distributed caching
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Yahoo Finance cache: Redis initialized")
            except Exception as e:
                logger.warning(f"Yahoo Finance cache: Redis unavailable: {e}")
        
        # Cache configuration for different data types
        self.ttl_config = {
            'current_price': 60,        # 1 minute
            'intraday_data': 300,       # 5 minutes
            'daily_data': 3600,         # 1 hour
            'weekly_data': 21600,       # 6 hours
            'monthly_data': 86400,      # 24 hours
            'financial_data': 86400,    # 24 hours
            'info': 21600,              # 6 hours
            'recommendations': 3600,    # 1 hour
            'earnings': 21600,          # 6 hours
            'dividends': 86400,         # 24 hours
        }
        
        # Start background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _get_cache_key(self, ticker: str, data_type: str, **kwargs) -> str:
        """Generate cache key"""
        # Create deterministic key from parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"yf:{data_type}:{ticker.upper()}:{params_hash}"
    
    def _get_ttl(self, data_type: str) -> int:
        """Get TTL for data type"""
        return self.ttl_config.get(data_type, self.default_ttl)
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        if isinstance(data, pd.DataFrame):
            return data.to_json(date_format='iso').encode('utf-8')
        elif isinstance(data, pd.Series):
            return data.to_json(date_format='iso').encode('utf-8')
        elif isinstance(data, dict):
            return json.dumps(data, default=str).encode('utf-8')
        else:
            return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize_data(self, data_bytes: bytes, data_type: str) -> Any:
        """Deserialize data from storage"""
        try:
            data_str = data_bytes.decode('utf-8')
            
            if 'historical' in data_type or 'data' in data_type:
                # Try to reconstruct DataFrame
                try:
                    return pd.read_json(data_str, date_format='iso')
                except:
                    return json.loads(data_str)
            else:
                return json.loads(data_str)
        except Exception as e:
            logger.warning(f"Failed to deserialize data: {e}")
            return None
    
    def get(self, ticker: str, data_type: str, **kwargs) -> Optional[Any]:
        """
        Get data from cache
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data (current_price, daily_data, etc.)
            **kwargs: Additional parameters for cache key
        
        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._get_cache_key(ticker, data_type, **kwargs)
        current_time = time.time()
        
        with self._lock:
            # Check in-memory cache first
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if current_time < entry.expiry:
                    logger.debug(f"Cache HIT (memory): {cache_key}")
                    return entry.data
                else:
                    # Expired - remove from cache
                    del self._cache[cache_key]
                    self._memory_usage -= entry.size_bytes
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = self._deserialize_data(cached_data, data_type)
                    if data is not None:
                        # Add back to memory cache
                        ttl = self._get_ttl(data_type)
                        self._add_to_memory_cache(cache_key, data, ticker, data_type, ttl)
                        logger.debug(f"Cache HIT (Redis): {cache_key}")
                        return data
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def set(self, ticker: str, data_type: str, data: Any, **kwargs):
        """
        Set data in cache
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data
            data: Data to cache
            **kwargs: Additional parameters for cache key
        """
        if data is None:
            return
        
        cache_key = self._get_cache_key(ticker, data_type, **kwargs)
        ttl = self._get_ttl(data_type)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, data, ticker, data_type, ttl)
        
        # Add to Redis cache
        if self.redis_client:
            try:
                serialized_data = self._serialize_data(data)
                self.redis_client.setex(cache_key, ttl, serialized_data)
                logger.debug(f"Data cached to Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, data: Any, 
                            ticker: str, data_type: str, ttl: int):
        """Add data to in-memory cache"""
        current_time = time.time()
        
        # Estimate size
        try:
            size_bytes = len(self._serialize_data(data))
        except:
            size_bytes = 1024  # Default size
        
        with self._lock:
            # Evict if necessary
            self._evict_if_needed(size_bytes)
            
            # Add entry
            entry = CacheEntry(
                data=data,
                timestamp=current_time,
                expiry=current_time + ttl,
                ticker=ticker,
                data_type=data_type,
                size_bytes=size_bytes
            )
            
            self._cache[cache_key] = entry
            self._memory_usage += size_bytes
            
            logger.debug(f"Data cached to memory: {cache_key} ({size_bytes / 1024:.1f}KB)")
    
    def _evict_if_needed(self, new_size: int):
        """Evict old entries if memory limit would be exceeded"""
        if self._memory_usage + new_size <= self.max_memory_bytes:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        for cache_key, entry in sorted_entries:
            del self._cache[cache_key]
            self._memory_usage -= entry.size_bytes
            logger.debug(f"Evicted cache entry: {cache_key}")
            
            if self._memory_usage + new_size <= self.max_memory_bytes:
                break
    
    def _cleanup_worker(self):
        """Background worker to clean up expired entries"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries from memory cache"""
        current_time = time.time()
        keys_to_remove = []
        
        with self._lock:
            for cache_key, entry in self._cache.items():
                if current_time >= entry.expiry:
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                entry = self._cache[cache_key]
                del self._cache[cache_key]
                self._memory_usage -= entry.size_bytes
                logger.debug(f"Cleaned up expired entry: {cache_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            memory_mb = self._memory_usage / 1024 / 1024
            
            # Group by data type
            type_stats = {}
            for entry in self._cache.values():
                data_type = entry.data_type
                if data_type not in type_stats:
                    type_stats[data_type] = {'count': 0, 'size_mb': 0}
                type_stats[data_type]['count'] += 1
                type_stats[data_type]['size_mb'] += entry.size_bytes / 1024 / 1024
            
            return {
                'total_entries': total_entries,
                'memory_usage_mb': memory_mb,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_usage_percent': (self._memory_usage / self.max_memory_bytes) * 100,
                'type_breakdown': type_stats,
                'redis_available': self.redis_client is not None
            }

# Global cache instance
_yf_cache = None
_cache_lock = threading.Lock()

def get_yahoo_finance_cache() -> YahooFinanceCache:
    """Get global Yahoo Finance cache instance"""
    global _yf_cache
    if _yf_cache is None:
        with _cache_lock:
            if _yf_cache is None:
                redis_url = os.environ.get('REDIS_URL')
                _yf_cache = YahooFinanceCache(
                    redis_url=redis_url,
                    default_ttl=int(os.environ.get('YF_CACHE_TTL', 300)),
                    max_memory_mb=int(os.environ.get('YF_CACHE_MB', 100))
                )
    return _yf_cache

def cached_yfinance_call(data_type: str):
    """
    Decorator for caching yfinance API calls
    
    Args:
        data_type: Type of data being cached
    """
    def decorator(func):
        @wraps(func)
        def wrapper(ticker: str, *args, **kwargs):
            cache = get_yahoo_finance_cache()
            
            # Try to get from cache
            cached_data = cache.get(ticker, data_type, args=args, kwargs=kwargs)
            if cached_data is not None:
                return cached_data
            
            # Cache miss - fetch from API
            try:
                data = func(ticker, *args, **kwargs)
                if data is not None:
                    cache.set(ticker, data_type, data, args=args, kwargs=kwargs)
                return data
            except Exception as e:
                logger.error(f"Yahoo Finance API error for {ticker}: {e}")
                raise
        
        return wrapper
    return decorator

# Cached Yahoo Finance functions
@cached_yfinance_call("current_price")
def get_current_price(ticker: str) -> float:
    """Get current stock price with caching"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d", interval="1m")
    if not hist.empty:
        return float(hist['Close'].iloc[-1])
    raise ValueError(f"No price data available for {ticker}")

@cached_yfinance_call("daily_data")
def get_daily_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get daily historical data with caching"""
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

@cached_yfinance_call("intraday_data")
def get_intraday_data(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """Get intraday data with caching"""
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

@cached_yfinance_call("info")
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get stock info with caching"""
    stock = yf.Ticker(ticker)
    return stock.info

@cached_yfinance_call("financial_data")
def get_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    """Get financial statements with caching"""
    stock = yf.Ticker(ticker)
    return {
        'income_stmt': stock.financials,
        'balance_sheet': stock.balance_sheet,
        'cash_flow': stock.cashflow
    }

@cached_yfinance_call("dividends")
def get_dividends(ticker: str) -> pd.Series:
    """Get dividend history with caching"""
    stock = yf.Ticker(ticker)
    return stock.dividends

@cached_yfinance_call("recommendations")
def get_recommendations(ticker: str) -> pd.DataFrame:
    """Get analyst recommendations with caching"""
    stock = yf.Ticker(ticker)
    return stock.recommendations

# Batch operations with caching
def get_multiple_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Get current prices for multiple tickers with caching"""
    cache = get_yahoo_finance_cache()
    results = {}
    uncached_tickers = []
    
    # Check cache for each ticker
    for ticker in tickers:
        cached_price = cache.get(ticker, "current_price")
        if cached_price is not None:
            results[ticker] = cached_price
        else:
            uncached_tickers.append(ticker)
    
    # Fetch uncached tickers in batch
    if uncached_tickers:
        try:
            # Use yfinance download for batch fetching
            data = yf.download(uncached_tickers, period="1d", interval="1m", 
                             group_by='ticker', progress=False)
            
            for ticker in uncached_tickers:
                try:
                    if len(uncached_tickers) == 1:
                        price_data = data['Close']
                    else:
                        price_data = data[ticker]['Close']
                    
                    if not price_data.empty:
                        current_price = float(price_data.iloc[-1])
                        results[ticker] = current_price
                        cache.set(ticker, "current_price", current_price)
                except Exception as e:
                    logger.warning(f"Failed to get price for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Batch price fetch failed: {e}")
            # Fallback to individual calls
            for ticker in uncached_tickers:
                try:
                    results[ticker] = get_current_price(ticker)
                except Exception as e:
                    logger.warning(f"Individual price fetch failed for {ticker}: {e}")
    
    return results

# Cache management functions
def warm_up_cache(tickers: List[str]):
    """Warm up cache with commonly used data"""
    cache = get_yahoo_finance_cache()
    
    for ticker in tickers:
        try:
            # Warm up with current price and basic info
            get_current_price(ticker)
            get_stock_info(ticker)
            logger.info(f"Warmed up cache for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to warm up cache for {ticker}: {e}")

def get_cache_health() -> Dict[str, Any]:
    """Get cache health status"""
    cache = get_yahoo_finance_cache()
    stats = cache.get_stats()
    
    return {
        'status': 'healthy' if stats['memory_usage_percent'] < 80 else 'warning',
        'stats': stats,
        'recommendations': _get_cache_recommendations(stats)
    }

def _get_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Get cache optimization recommendations"""
    recommendations = []
    
    if stats['memory_usage_percent'] > 80:
        recommendations.append("Consider increasing cache memory limit")
    
    if not stats['redis_available']:
        recommendations.append("Redis not available - consider enabling for better performance")
    
    return recommendations