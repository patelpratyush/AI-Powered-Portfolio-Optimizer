#!/usr/bin/env python3
"""
ML Model Performance Optimization
- Model caching and lazy loading
- Batch prediction optimization
- Model versioning and A/B testing
- Performance monitoring
"""
import os
import pickle
import hashlib
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import redis
import logging
from threading import Lock
import joblib
import time
from pathlib import Path

logger = logging.getLogger('portfolio_optimizer.ml_optimization')


@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning"""
    model_id: str
    ticker: str
    model_type: str
    version: str
    created_at: datetime
    accuracy_metrics: Dict[str, float]
    training_data_size: int
    feature_count: int
    file_path: str
    checksum: str
    performance_metrics: Dict[str, float]
    is_active: bool = True


class ModelCache:
    """Intelligent model caching system"""
    
    def __init__(self, redis_client: redis.Redis, max_memory_models: int = 10):
        self.redis_client = redis_client
        self.max_memory_models = max_memory_models
        self.memory_cache = {}
        self.cache_lock = Lock()
        self.access_times = {}
        self.logger = logging.getLogger('portfolio_optimizer.model_cache')
        
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from cache with LRU eviction"""
        with self.cache_lock:
            # Check memory cache first
            if model_id in self.memory_cache:
                self.access_times[model_id] = time.time()
                self.logger.debug(f"Model {model_id} loaded from memory cache")
                return self.memory_cache[model_id]
            
            # Check Redis cache
            model_data = self.redis_client.get(f"model_cache:{model_id}")
            if model_data:
                try:
                    model = pickle.loads(model_data)
                    self._add_to_memory_cache(model_id, model)
                    self.logger.debug(f"Model {model_id} loaded from Redis cache")
                    return model
                except Exception as e:
                    self.logger.error(f"Failed to deserialize model {model_id}: {e}")
                    self.redis_client.delete(f"model_cache:{model_id}")
            
            return None
    
    def cache_model(self, model_id: str, model: Any, ttl: int = 3600):
        """Cache model in both memory and Redis"""
        try:
            # Cache in Redis
            model_data = pickle.dumps(model)
            self.redis_client.setex(f"model_cache:{model_id}", ttl, model_data)
            
            # Cache in memory
            self._add_to_memory_cache(model_id, model)
            
            self.logger.info(f"Model {model_id} cached successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cache model {model_id}: {e}")
    
    def _add_to_memory_cache(self, model_id: str, model: Any):
        """Add model to memory cache with LRU eviction"""
        with self.cache_lock:
            # Remove least recently used model if cache is full
            if len(self.memory_cache) >= self.max_memory_models:
                lru_model_id = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.memory_cache[lru_model_id]
                del self.access_times[lru_model_id]
                self.logger.debug(f"Evicted model {lru_model_id} from memory cache")
            
            self.memory_cache[model_id] = model
            self.access_times[model_id] = time.time()
    
    def invalidate_model(self, model_id: str):
        """Remove model from all caches"""
        with self.cache_lock:
            if model_id in self.memory_cache:
                del self.memory_cache[model_id]
                del self.access_times[model_id]
        
        self.redis_client.delete(f"model_cache:{model_id}")
        self.logger.info(f"Model {model_id} invalidated from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            memory_models = list(self.memory_cache.keys())
        
        # Get Redis cache info
        redis_keys = list(self.redis_client.scan_iter(match="model_cache:*"))
        redis_models = [key.decode().replace("model_cache:", "") for key in redis_keys]
        
        return {
            'memory_cache_size': len(memory_models),
            'memory_cache_max': self.max_memory_models,
            'memory_cached_models': memory_models,
            'redis_cache_size': len(redis_models),
            'redis_cached_models': redis_models,
            'cache_hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would require more sophisticated tracking
        # For now, return a placeholder
        return 0.85


class ModelVersionManager:
    """Model versioning and A/B testing system"""
    
    def __init__(self, redis_client: redis.Redis, models_dir: str):
        self.redis_client = redis_client
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('portfolio_optimizer.model_version')
    
    def register_model(self, 
                      ticker: str,
                      model_type: str,
                      model_obj: Any,
                      accuracy_metrics: Dict[str, float],
                      training_data_size: int,
                      feature_count: int) -> ModelMetadata:
        """Register a new model version"""
        
        # Generate model ID and version
        model_id = self._generate_model_id(ticker, model_type)
        version = self._generate_version()
        
        # Save model to disk
        file_path = self.models_dir / f"{model_id}_{version}.joblib"
        joblib.dump(model_obj, file_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            ticker=ticker,
            model_type=model_type,
            version=version,
            created_at=datetime.utcnow(),
            accuracy_metrics=accuracy_metrics,
            training_data_size=training_data_size,
            feature_count=feature_count,
            file_path=str(file_path),
            checksum=checksum,
            performance_metrics={},
            is_active=True
        )
        
        # Store metadata in Redis
        self._store_metadata(metadata)
        
        # Update active model for ticker/type
        self._set_active_model(ticker, model_type, model_id, version)
        
        self.logger.info(
            f"Model registered: {model_id} v{version}",
            extra={
                'model_id': model_id,
                'version': version,
                'ticker': ticker,
                'model_type': model_type,
                'accuracy_metrics': accuracy_metrics
            }
        )
        
        return metadata
    
    def get_active_model(self, ticker: str, model_type: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """Get the currently active model for a ticker/type"""
        active_key = f"active_model:{ticker}:{model_type}"
        active_info = self.redis_client.hgetall(active_key)
        
        if not active_info:
            return None
        
        model_id = active_info[b'model_id'].decode()
        version = active_info[b'version'].decode()
        
        # Get metadata
        metadata = self._get_metadata(model_id, version)
        if not metadata:
            return None
        
        # Load model
        try:
            model = joblib.load(metadata.file_path)
            return model, metadata
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id} v{version}: {e}")
            return None
    
    def list_model_versions(self, ticker: str, model_type: str) -> List[ModelMetadata]:
        """List all versions of a model"""
        pattern = f"model_metadata:{ticker}:{model_type}:*"
        keys = list(self.redis_client.scan_iter(match=pattern))
        
        models = []
        for key in keys:
            try:
                data = self.redis_client.hgetall(key)
                if data:
                    metadata = self._deserialize_metadata(data)
                    models.append(metadata)
            except Exception as e:
                self.logger.error(f"Failed to deserialize metadata for {key}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def rollback_model(self, ticker: str, model_type: str, version: str) -> bool:
        """Rollback to a previous model version"""
        # Find the model
        model_id = self._generate_model_id(ticker, model_type)
        metadata = self._get_metadata(model_id, version)
        
        if not metadata:
            self.logger.error(f"Model version not found: {model_id} v{version}")
            return False
        
        # Check if model file exists
        if not Path(metadata.file_path).exists():
            self.logger.error(f"Model file not found: {metadata.file_path}")
            return False
        
        # Set as active model
        self._set_active_model(ticker, model_type, model_id, version)
        
        self.logger.info(
            f"Model rolled back: {model_id} v{version}",
            extra={
                'model_id': model_id,
                'version': version,
                'ticker': ticker,
                'model_type': model_type
            }
        )
        
        return True
    
    def setup_ab_test(self, 
                     ticker: str,
                     model_type: str,
                     version_a: str,
                     version_b: str,
                     traffic_split: float = 0.5) -> str:
        """Set up A/B test between two model versions"""
        
        test_id = f"abtest_{ticker}_{model_type}_{int(time.time())}"
        
        test_config = {
            'test_id': test_id,
            'ticker': ticker,
            'model_type': model_type,
            'version_a': version_a,
            'version_b': version_b,
            'traffic_split': traffic_split,
            'start_time': datetime.utcnow().isoformat(),
            'metrics': {'a': {}, 'b': {}},
            'active': True
        }
        
        # Store test configuration
        self.redis_client.hset(
            f"ab_test:{test_id}",
            mapping={k: str(v) for k, v in test_config.items()}
        )
        self.redis_client.expire(f"ab_test:{test_id}", 86400 * 30)  # 30 days
        
        # Set test as active for this ticker/model_type
        self.redis_client.setex(
            f"active_ab_test:{ticker}:{model_type}",
            86400 * 30,
            test_id
        )
        
        self.logger.info(
            f"A/B test started: {test_id}",
            extra={
                'test_id': test_id,
                'ticker': ticker,
                'model_type': model_type,
                'versions': [version_a, version_b],
                'traffic_split': traffic_split
            }
        )
        
        return test_id
    
    def get_model_for_request(self, ticker: str, model_type: str, request_id: str = None) -> Optional[Tuple[Any, ModelMetadata, str]]:
        """Get model considering A/B tests"""
        
        # Check if there's an active A/B test
        test_id = self.redis_client.get(f"active_ab_test:{ticker}:{model_type}")
        
        if test_id:
            test_id = test_id.decode()
            test_config = self.redis_client.hgetall(f"ab_test:{test_id}")
            
            if test_config and test_config.get(b'active') == b'True':
                # Determine which version to use
                traffic_split = float(test_config[b'traffic_split'])
                
                # Use request_id or timestamp for consistent routing
                if request_id:
                    hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                else:
                    hash_val = int(time.time() * 1000000)
                
                use_version_a = (hash_val % 100) < (traffic_split * 100)
                version = test_config[b'version_a' if use_version_a else b'version_b'].decode()
                variant = 'a' if use_version_a else 'b'
                
                # Load specific version
                model_id = self._generate_model_id(ticker, model_type)
                metadata = self._get_metadata(model_id, version)
                
                if metadata and Path(metadata.file_path).exists():
                    try:
                        model = joblib.load(metadata.file_path)
                        return model, metadata, f"abtest_{variant}"
                    except Exception as e:
                        self.logger.error(f"Failed to load A/B test model: {e}")
        
        # Fall back to active model
        result = self.get_active_model(ticker, model_type)
        if result:
            model, metadata = result
            return model, metadata, "active"
        
        return None
    
    def record_prediction_metrics(self, 
                                model_id: str, 
                                version: str,
                                prediction_time: float,
                                accuracy_score: Optional[float] = None):
        """Record prediction performance metrics"""
        
        metrics_key = f"model_performance:{model_id}:{version}"
        
        # Update performance metrics
        pipe = self.redis_client.pipeline()
        pipe.lpush(f"{metrics_key}:response_times", prediction_time)
        pipe.ltrim(f"{metrics_key}:response_times", 0, 999)  # Keep last 1000
        pipe.expire(f"{metrics_key}:response_times", 86400)  # 24 hours
        
        if accuracy_score is not None:
            pipe.lpush(f"{metrics_key}:accuracy_scores", accuracy_score)
            pipe.ltrim(f"{metrics_key}:accuracy_scores", 0, 999)
            pipe.expire(f"{metrics_key}:accuracy_scores", 86400)
        
        pipe.incr(f"{metrics_key}:prediction_count")
        pipe.expire(f"{metrics_key}:prediction_count", 86400)
        
        pipe.execute()
    
    def get_model_performance_stats(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get performance statistics for a model version"""
        
        metrics_key = f"model_performance:{model_id}:{version}"
        
        # Get response times
        response_times = self.redis_client.lrange(f"{metrics_key}:response_times", 0, -1)
        response_times = [float(rt) for rt in response_times]
        
        # Get accuracy scores
        accuracy_scores = self.redis_client.lrange(f"{metrics_key}:accuracy_scores", 0, -1)
        accuracy_scores = [float(acc) for acc in accuracy_scores]
        
        # Get prediction count
        prediction_count = int(self.redis_client.get(f"{metrics_key}:prediction_count") or 0)
        
        stats = {
            'prediction_count': prediction_count,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
        }
        
        if accuracy_scores:
            stats.update({
                'avg_accuracy': np.mean(accuracy_scores),
                'min_accuracy': np.min(accuracy_scores),
                'max_accuracy': np.max(accuracy_scores)
            })
        
        return stats
    
    def _generate_model_id(self, ticker: str, model_type: str) -> str:
        """Generate consistent model ID"""
        return f"{ticker}_{model_type}"
    
    def _generate_version(self) -> str:
        """Generate version string"""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of model file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _store_metadata(self, metadata: ModelMetadata):
        """Store model metadata in Redis"""
        key = f"model_metadata:{metadata.ticker}:{metadata.model_type}:{metadata.version}"
        
        data = {
            'model_id': metadata.model_id,
            'ticker': metadata.ticker,
            'model_type': metadata.model_type,
            'version': metadata.version,
            'created_at': metadata.created_at.isoformat(),
            'accuracy_metrics': str(metadata.accuracy_metrics),
            'training_data_size': str(metadata.training_data_size),
            'feature_count': str(metadata.feature_count),
            'file_path': metadata.file_path,
            'checksum': metadata.checksum,
            'is_active': str(metadata.is_active)
        }
        
        self.redis_client.hset(key, mapping=data)
        self.redis_client.expire(key, 86400 * 365)  # 1 year
    
    def _get_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata from Redis"""
        # Extract ticker and model_type from model_id
        parts = model_id.split('_', 1)
        if len(parts) != 2:
            return None
        
        ticker, model_type = parts
        key = f"model_metadata:{ticker}:{model_type}:{version}"
        
        data = self.redis_client.hgetall(key)
        if not data:
            return None
        
        return self._deserialize_metadata(data)
    
    def _deserialize_metadata(self, data: Dict[bytes, bytes]) -> ModelMetadata:
        """Deserialize metadata from Redis hash"""
        return ModelMetadata(
            model_id=data[b'model_id'].decode(),
            ticker=data[b'ticker'].decode(),
            model_type=data[b'model_type'].decode(),
            version=data[b'version'].decode(),
            created_at=datetime.fromisoformat(data[b'created_at'].decode()),
            accuracy_metrics=eval(data[b'accuracy_metrics'].decode()),
            training_data_size=int(data[b'training_data_size']),
            feature_count=int(data[b'feature_count']),
            file_path=data[b'file_path'].decode(),
            checksum=data[b'checksum'].decode(),
            performance_metrics={},
            is_active=data[b'is_active'].decode() == 'True'
        )
    
    def _set_active_model(self, ticker: str, model_type: str, model_id: str, version: str):
        """Set active model for ticker/model_type"""
        key = f"active_model:{ticker}:{model_type}"
        self.redis_client.hset(key, mapping={
            'model_id': model_id,
            'version': version,
            'updated_at': datetime.utcnow().isoformat()
        })


class BatchPredictionOptimizer:
    """Optimize batch predictions for better throughput"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger('portfolio_optimizer.batch_optimizer')
    
    async def batch_predict(self, 
                          model_func: callable,
                          inputs: List[Any],
                          batch_size: int = 32) -> List[Any]:
        """Run batch predictions with optimal batching"""
        
        if not inputs:
            return []
        
        # Split inputs into batches
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._process_batch, model_func, batch): batch 
                for batch in batches
            }
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch prediction failed: {e}")
                    # Return empty results for failed batch
                    batch = future_to_batch[future]
                    all_results.extend([None] * len(batch))
        
        return all_results
    
    def _process_batch(self, model_func: callable, batch: List[Any]) -> List[Any]:
        """Process a single batch"""
        try:
            return [model_func(item) for item in batch]
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [None] * len(batch)


def setup_ml_optimization(redis_client: redis.Redis, models_dir: str) -> Dict[str, Any]:
    """Set up ML optimization components"""
    
    model_cache = ModelCache(redis_client)
    version_manager = ModelVersionManager(redis_client, models_dir)
    batch_optimizer = BatchPredictionOptimizer()
    
    return {
        'model_cache': model_cache,
        'version_manager': version_manager,
        'batch_optimizer': batch_optimizer
    }