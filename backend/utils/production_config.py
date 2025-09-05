#!/usr/bin/env python3
"""
Production readiness configuration and utilities
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets
from cryptography.fernet import Fernet
import redis
from flask import Flask
import psutil

logger = logging.getLogger('portfolio_optimizer.production')

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    debug: bool
    testing: bool
    log_level: str
    database_pool_size: int
    redis_max_connections: int
    worker_timeout: int
    max_content_length: int
    rate_limit_per_minute: int
    
    @classmethod
    def from_env(cls) -> 'EnvironmentConfig':
        """Create config from environment variables"""
        env_name = os.getenv('FLASK_ENV', 'development')
        
        configs = {
            'development': cls(
                name='development',
                debug=True,
                testing=False,
                log_level='DEBUG',
                database_pool_size=5,
                redis_max_connections=10,
                worker_timeout=30,
                max_content_length=16 * 1024 * 1024,  # 16MB
                rate_limit_per_minute=1000
            ),
            'staging': cls(
                name='staging',
                debug=False,
                testing=False,
                log_level='INFO',
                database_pool_size=10,
                redis_max_connections=50,
                worker_timeout=60,
                max_content_length=32 * 1024 * 1024,  # 32MB
                rate_limit_per_minute=500
            ),
            'production': cls(
                name='production',
                debug=False,
                testing=False,
                log_level='WARNING',
                database_pool_size=20,
                redis_max_connections=100,
                worker_timeout=120,
                max_content_length=64 * 1024 * 1024,  # 64MB
                rate_limit_per_minute=100
            )
        }
        
        return configs.get(env_name, configs['development'])


class SecretsManager:
    """Secure secrets management for production"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            # Generate a new key for development
            self.encryption_key = Fernet.generate_key().decode()
            logger.warning("No encryption key found, generated new key for development")
        
        self.cipher = Fernet(self.encryption_key.encode())
        self.secrets_cache = {}
        
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value"""
        return self.cipher.decrypt(encrypted_value.encode()).decode()
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret from environment or cache"""
        # Check cache first
        if key in self.secrets_cache:
            return self.secrets_cache[key]
        
        # Try environment variable
        env_value = os.getenv(key, default)
        if env_value:
            # Check if it's encrypted (starts with gAAAAAB)
            if env_value.startswith('gAAAAA'):
                try:
                    decrypted = self.decrypt_secret(env_value)
                    self.secrets_cache[key] = decrypted
                    return decrypted
                except Exception as e:
                    logger.error(f"Failed to decrypt secret {key}: {e}")
                    return default
            else:
                self.secrets_cache[key] = env_value
                return env_value
        
        return default
    
    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret value"""
        try:
            encrypted = self.encrypt_secret(new_value)
            # In production, this would update the secret store
            logger.info(f"Secret rotation completed for {key}")
            self.secrets_cache[key] = new_value
            return True
        except Exception as e:
            logger.error(f"Failed to rotate secret {key}: {e}")
            return False


class MetricsCollector:
    """Collect application metrics for monitoring"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.metrics')
        
    def record_request(self, endpoint: str, method: str, status_code: int, duration_ms: float):
        """Record API request metrics"""
        try:
            if self.redis_client:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d:%H:%M')
                
                # Request count
                self.redis_client.incr(f"metrics:requests:{endpoint}:{method}:{timestamp}")
                self.redis_client.expire(f"metrics:requests:{endpoint}:{method}:{timestamp}", 3600)
                
                # Status code distribution
                self.redis_client.incr(f"metrics:status:{status_code}:{timestamp}")
                self.redis_client.expire(f"metrics:status:{status_code}:{timestamp}", 3600)
                
                # Response time (histogram buckets)
                bucket = self._get_duration_bucket(duration_ms)
                self.redis_client.incr(f"metrics:duration:{bucket}:{timestamp}")
                self.redis_client.expire(f"metrics:duration:{bucket}:{timestamp}", 3600)
                
        except Exception as e:
            self.logger.error(f"Failed to record request metrics: {e}")
    
    def record_custom_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record custom application metric"""
        try:
            if self.redis_client:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d:%H:%M')
                tag_str = '_'.join(f"{k}:{v}" for k, v in (tags or {}).items())
                key = f"metrics:custom:{name}:{tag_str}:{timestamp}"
                
                self.redis_client.lpush(key, value)
                self.redis_client.expire(key, 3600)
                
        except Exception as e:
            self.logger.error(f"Failed to record custom metric {name}: {e}")
    
    def _get_duration_bucket(self, duration_ms: float) -> str:
        """Get duration bucket for histogram"""
        if duration_ms < 10:
            return "0-10ms"
        elif duration_ms < 50:
            return "10-50ms"
        elif duration_ms < 100:
            return "50-100ms"
        elif duration_ms < 500:
            return "100-500ms"
        elif duration_ms < 1000:
            return "500ms-1s"
        elif duration_ms < 5000:
            return "1-5s"
        else:
            return "5s+"
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        try:
            if not self.redis_client:
                return {}
            
            summary = {
                'total_requests': 0,
                'status_codes': {},
                'duration_distribution': {},
                'endpoints': {},
                'custom_metrics': {}
            }
            
            now = datetime.utcnow()
            for hour in range(hours):
                hour_key = (now - timedelta(hours=hour)).strftime('%Y-%m-%d:%H')
                
                # Scan for all metrics in this hour
                for pattern in ['metrics:requests:*', 'metrics:status:*', 'metrics:duration:*']:
                    for key in self.redis_client.scan_iter(match=f"{pattern}:{hour_key}"):
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        value = int(self.redis_client.get(key) or 0)
                        
                        if 'requests:' in key_str:
                            summary['total_requests'] += value
                            endpoint = key_str.split(':')[2]
                            summary['endpoints'][endpoint] = summary['endpoints'].get(endpoint, 0) + value
                        elif 'status:' in key_str:
                            status = key_str.split(':')[2]
                            summary['status_codes'][status] = summary['status_codes'].get(status, 0) + value
                        elif 'duration:' in key_str:
                            bucket = key_str.split(':')[2]
                            summary['duration_distribution'][bucket] = summary['duration_distribution'].get(bucket, 0) + value
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
        
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            if not self.redis_client:
                return "# No Redis client available\n"
            
            summary = self.get_metrics_summary(24)
            prometheus_lines = [
                "# HELP portfolio_requests_total Total HTTP requests",
                "# TYPE portfolio_requests_total counter",
                f"portfolio_requests_total {summary.get('total_requests', 0)}"
            ]
            
            # Status code metrics
            for status, count in summary.get('status_codes', {}).items():
                prometheus_lines.append(f'portfolio_requests_total{{status="{status}"}} {count}')
            
            # Duration distribution
            prometheus_lines.extend([
                "# HELP portfolio_request_duration_bucket Request duration buckets",
                "# TYPE portfolio_request_duration_bucket histogram"
            ])
            
            for bucket, count in summary.get('duration_distribution', {}).items():
                prometheus_lines.append(f'portfolio_request_duration_bucket{{le="{bucket}"}} {count}')
            
            # Endpoint metrics
            prometheus_lines.extend([
                "# HELP portfolio_endpoint_requests_total Requests per endpoint",
                "# TYPE portfolio_endpoint_requests_total counter"
            ])
            
            for endpoint, count in summary.get('endpoints', {}).items():
                prometheus_lines.append(f'portfolio_endpoint_requests_total{{endpoint="{endpoint}"}} {count}')
            
            return '\n'.join(prometheus_lines) + '\n'
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return "# Error exporting metrics\n"


class ResourceMonitor:
    """Monitor system resources and auto-scaling triggers"""
    
    def __init__(self):
        self.logger = logging.getLogger('portfolio_optimizer.resources')
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 80.0,
            'memory_critical': 90.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0
        }
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = 'healthy'
            alerts = []
            
            # Check thresholds
            if cpu_percent > self.thresholds['cpu_critical']:
                status = 'critical'
                alerts.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > self.thresholds['cpu_warning']:
                status = 'warning'
                alerts.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent > self.thresholds['memory_critical']:
                status = 'critical'
                alerts.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > self.thresholds['memory_warning']:
                if status != 'critical':
                    status = 'warning'
                alerts.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.thresholds['disk_critical']:
                status = 'critical'
                alerts.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > self.thresholds['disk_warning']:
                if status not in ['critical']:
                    status = 'warning'
                alerts.append(f"Disk usage high: {disk_percent:.1f}%")
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk_percent,
                'alerts': alerts,
                'timestamp': datetime.utcnow().isoformat(),
                'scaling_recommendation': self._get_scaling_recommendation(cpu_percent, memory.percent)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _get_scaling_recommendation(self, cpu: float, memory: float) -> str:
        """Get auto-scaling recommendation"""
        if cpu > 85 or memory > 90:
            return 'scale_up_urgent'
        elif cpu > 70 or memory > 80:
            return 'scale_up'
        elif cpu < 20 and memory < 40:
            return 'scale_down'
        else:
            return 'stable'


class ProductionMiddleware:
    """Production-ready middleware for Flask applications"""
    
    def __init__(self, app: Flask, metrics_collector: MetricsCollector):
        self.app = app
        self.metrics_collector = metrics_collector
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup production middleware"""
        
        @self.app.before_request
        def before_request():
            """Before request middleware"""
            from flask import request, g
            import time
            
            g.start_time = time.time()
            g.request_id = secrets.token_urlsafe(16)
            
            # Log incoming request
            logger.info(
                f"Request started: {request.method} {request.path}",
                extra={
                    'request_id': g.request_id,
                    'method': request.method,
                    'path': request.path,
                    'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                    'user_agent': request.headers.get('User-Agent', '')[:100]
                }
            )
        
        @self.app.after_request
        def after_request(response):
            """After request middleware"""
            from flask import request, g
            import time
            
            if hasattr(g, 'start_time'):
                duration = (time.time() - g.start_time) * 1000
                
                # Record metrics
                self.metrics_collector.record_request(
                    endpoint=request.endpoint or 'unknown',
                    method=request.method,
                    status_code=response.status_code,
                    duration_ms=duration
                )
                
                # Log response
                logger.info(
                    f"Request completed: {request.method} {request.path} - {response.status_code}",
                    extra={
                        'request_id': getattr(g, 'request_id', 'unknown'),
                        'status_code': response.status_code,
                        'duration_ms': round(duration, 2),
                        'response_size': response.content_length or 0
                    }
                )
            
            # Add security headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
            return response


def setup_production_features(app: Flask, redis_client: Optional[redis.Redis] = None) -> Dict[str, Any]:
    """Setup all production readiness features"""
    
    # Environment configuration
    env_config = EnvironmentConfig.from_env()
    
    # Configure Flask app
    app.config.update({
        'MAX_CONTENT_LENGTH': env_config.max_content_length,
        'PERMANENT_SESSION_LIFETIME': 3600 if env_config.name == 'production' else 7200,
        'SESSION_COOKIE_SECURE': env_config.name == 'production',
        'SESSION_COOKIE_HTTPONLY': True,
        'SESSION_COOKIE_SAMESITE': 'Lax',
    })
    
    # Secrets management
    secrets_manager = SecretsManager()
    
    # Metrics collection
    metrics_collector = MetricsCollector(redis_client)
    
    # Resource monitoring
    resource_monitor = ResourceMonitor()
    
    # Production middleware
    ProductionMiddleware(app, metrics_collector)
    
    # Add production endpoints
    @app.route('/metrics')
    def metrics_endpoint():
        """Prometheus-compatible metrics endpoint"""
        from flask import Response
        
        try:
            prometheus_metrics = metrics_collector.export_prometheus_metrics()
            return Response(prometheus_metrics, mimetype='text/plain')
            
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return Response("# Error generating metrics\n", mimetype='text/plain'), 500
    
    @app.route('/resources')
    def resources_endpoint():
        """Resource monitoring endpoint"""
        from flask import jsonify
        
        resources = resource_monitor.check_resources()
        return jsonify(resources)
    
    # Return all production components
    return {
        'env_config': env_config,
        'secrets_manager': secrets_manager,
        'metrics_collector': metrics_collector,
        'resource_monitor': resource_monitor
    }