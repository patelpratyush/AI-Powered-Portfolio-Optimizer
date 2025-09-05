#!/usr/bin/env python3
"""
Application monitoring and health checks
"""
import os
import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
import redis
import sqlalchemy as sa
from sqlalchemy import text


@dataclass
class HealthCheckResult:
    """Health check result data class"""
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('portfolio_optimizer.monitoring')
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': {
                        '1m': load_avg[0],
                        '5m': load_avg[1],
                        '15m': load_avg[2]
                    }
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'percent': memory.percent
                },
                'swap': {
                    'total_gb': round(swap.total / (1024**3), 2),
                    'used_gb': round(swap.used / (1024**3), 2),
                    'percent': swap.percent
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def check_resource_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if resources exceed warning thresholds"""
        warnings = []
        
        try:
            # CPU threshold (>80%)
            if metrics.get('cpu', {}).get('percent', 0) > 80:
                warnings.append(f"High CPU usage: {metrics['cpu']['percent']:.1f}%")
            
            # Memory threshold (>85%)
            if metrics.get('memory', {}).get('percent', 0) > 85:
                warnings.append(f"High memory usage: {metrics['memory']['percent']:.1f}%")
            
            # Disk threshold (>90%)
            if metrics.get('disk', {}).get('percent', 0) > 90:
                warnings.append(f"High disk usage: {metrics['disk']['percent']:.1f}%")
            
            # Load average threshold (>CPU count * 2)
            cpu_count = metrics.get('cpu', {}).get('count', 1)
            load_1m = metrics.get('cpu', {}).get('load_avg', {}).get('1m', 0)
            if load_1m > cpu_count * 2:
                warnings.append(f"High load average: {load_1m:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error checking resource thresholds: {e}")
            warnings.append("Error checking system resources")
        
        return warnings


class DatabaseHealthCheck:
    """Database connectivity and performance monitoring"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.logger = logging.getLogger('portfolio_optimizer.monitoring')
        
    def check_health(self) -> HealthCheckResult:
        """Check database health"""
        start_time = time.time()
        
        try:
            # Create engine for health check
            engine = sa.create_engine(self.database_url, pool_pre_ping=True)
            
            # Test connection
            with engine.connect() as conn:
                # Simple query to test connectivity
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database stats
                db_stats = self._get_database_stats(conn)
                
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on response time and stats
            if response_time > 5000:  # 5 seconds
                status = 'degraded'
                message = f"Database responding slowly ({response_time:.0f}ms)"
            else:
                status = 'healthy'
                message = f"Database healthy ({response_time:.0f}ms)"
            
            return HealthCheckResult(
                service='database',
                status=status,
                response_time_ms=response_time,
                message=message,
                details=db_stats
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Database health check failed: {e}")
            
            return HealthCheckResult(
                service='database',
                status='unhealthy',
                response_time_ms=response_time,
                message=f"Database connection failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _get_database_stats(self, conn) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Get connection count (PostgreSQL specific)
            try:
                result = conn.execute(text("""
                    SELECT count(*) as active_connections
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """))
                active_connections = result.fetchone()[0]
            except:
                active_connections = "unknown"
            
            # Get database size (PostgreSQL specific)
            try:
                result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size = result.fetchone()[0]
            except:
                db_size = "unknown"
            
            return {
                'active_connections': active_connections,
                'database_size': db_size,
                'engine_pool_size': conn.engine.pool.size(),
                'engine_pool_checked_in': conn.engine.pool.checkedin(),
                'engine_pool_checked_out': conn.engine.pool.checkedout(),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}


class RedisHealthCheck:
    """Redis connectivity and performance monitoring"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.logger = logging.getLogger('portfolio_optimizer.monitoring')
        
    def check_health(self) -> HealthCheckResult:
        """Check Redis health"""
        start_time = time.time()
        
        try:
            # Connect to Redis
            r = redis.from_url(self.redis_url)
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            r.delete(test_key)
            
            if value != b"test_value":
                raise Exception("Redis read/write test failed")
            
            # Get Redis info
            redis_info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check status based on memory usage and response time
            memory_usage = redis_info.get('used_memory_rss', 0)
            max_memory = redis_info.get('maxmemory', 0)
            
            if response_time > 1000:  # 1 second
                status = 'degraded'
                message = f"Redis responding slowly ({response_time:.0f}ms)"
            elif max_memory > 0 and memory_usage > max_memory * 0.9:
                status = 'degraded'
                message = f"Redis memory usage high ({memory_usage}/{max_memory})"
            else:
                status = 'healthy'
                message = f"Redis healthy ({response_time:.0f}ms)"
            
            return HealthCheckResult(
                service='redis',
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    'version': redis_info.get('redis_version'),
                    'connected_clients': redis_info.get('connected_clients'),
                    'used_memory_human': redis_info.get('used_memory_human'),
                    'keyspace_hits': redis_info.get('keyspace_hits'),
                    'keyspace_misses': redis_info.get('keyspace_misses'),
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Redis health check failed: {e}")
            
            return HealthCheckResult(
                service='redis',
                status='unhealthy',
                response_time_ms=response_time,
                message=f"Redis connection failed: {str(e)}",
                details={'error': str(e)}
            )


class ExternalServiceHealthCheck:
    """External service monitoring (Yahoo Finance, etc.)"""
    
    def __init__(self):
        self.logger = logging.getLogger('portfolio_optimizer.monitoring')
        
    def check_yahoo_finance(self) -> HealthCheckResult:
        """Check Yahoo Finance API availability"""
        import yfinance as yf
        start_time = time.time()
        
        try:
            # Test with a simple ticker
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            
            if not info or 'symbol' not in info:
                raise Exception("Invalid response from Yahoo Finance")
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service='yahoo_finance',
                status='healthy' if response_time < 5000 else 'degraded',
                response_time_ms=response_time,
                message=f"Yahoo Finance API healthy ({response_time:.0f}ms)",
                details={
                    'test_symbol': info.get('symbol'),
                    'last_test': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Yahoo Finance health check failed: {e}")
            
            return HealthCheckResult(
                service='yahoo_finance',
                status='unhealthy',
                response_time_ms=response_time,
                message=f"Yahoo Finance API failed: {str(e)}",
                details={'error': str(e)}
            )


class ApplicationHealthMonitor:
    """Main application health monitoring"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.logger = logging.getLogger('portfolio_optimizer.monitoring')
        self.system_monitor = SystemMonitor()
        
        # Initialize health checkers
        database_url = app.config.get('DATABASE_URL')
        redis_url = app.config.get('REDIS_URL')
        
        self.db_health = DatabaseHealthCheck(database_url) if database_url else None
        self.redis_health = RedisHealthCheck(redis_url) if redis_url else None
        self.external_health = ExternalServiceHealthCheck()
        
        # Add health check routes
        self._register_health_routes()
        
    def _register_health_routes(self):
        """Register health check endpoints"""
        
        @self.app.route('/health')
        def health_check():
            """Basic health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': self.app.config.get('VERSION', '1.0.0'),
                'environment': self.app.config.get('FLASK_ENV', 'development')
            })
        
        @self.app.route('/health/detailed')
        def detailed_health_check():
            """Detailed health check with all services"""
            results = self.run_all_health_checks()
            
            # Determine overall status
            statuses = [result.status for result in results.values()]
            if 'unhealthy' in statuses:
                overall_status = 'unhealthy'
            elif 'degraded' in statuses:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            return jsonify({
                'status': overall_status,
                'timestamp': datetime.utcnow().isoformat(),
                'version': self.app.config.get('VERSION', '1.0.0'),
                'environment': self.app.config.get('FLASK_ENV', 'development'),
                'services': {
                    service: {
                        'status': result.status,
                        'response_time_ms': result.response_time_ms,
                        'message': result.message,
                        'details': result.details,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None
                    }
                    for service, result in results.items()
                }
            }), 200 if overall_status == 'healthy' else 503
        
        @self.app.route('/health/metrics')
        def system_metrics():
            """System metrics endpoint"""
            metrics = self.system_monitor.get_system_metrics()
            warnings = self.system_monitor.check_resource_thresholds(metrics)
            
            return jsonify({
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'warnings': warnings,
                'status': 'warning' if warnings else 'ok'
            })
        
        @self.app.route('/health/readiness')
        def readiness_check():
            """Kubernetes readiness probe endpoint"""
            results = self.run_critical_health_checks()
            
            # Only check critical services for readiness
            critical_healthy = all(
                result.status in ['healthy', 'degraded'] 
                for result in results.values()
            )
            
            if critical_healthy:
                return jsonify({'status': 'ready'}), 200
            else:
                return jsonify({
                    'status': 'not_ready',
                    'failing_services': [
                        service for service, result in results.items()
                        if result.status == 'unhealthy'
                    ]
                }), 503
        
        @self.app.route('/health/liveness')
        def liveness_check():
            """Kubernetes liveness probe endpoint"""
            # Simple liveness check - just ensure app is responding
            return jsonify({'status': 'alive'}), 200
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        
        # Database health check
        if self.db_health:
            results['database'] = self.db_health.check_health()
        
        # Redis health check
        if self.redis_health:
            results['redis'] = self.redis_health.check_health()
        
        # External services
        results['yahoo_finance'] = self.external_health.check_yahoo_finance()
        
        return results
    
    def run_critical_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run only critical health checks (for readiness probe)"""
        results = {}
        
        # Only check database and Redis for readiness
        if self.db_health:
            results['database'] = self.db_health.check_health()
        
        if self.redis_health:
            results['redis'] = self.redis_health.check_health()
        
        return results
    
    def log_health_summary(self):
        """Log health check summary"""
        try:
            results = self.run_all_health_checks()
            metrics = self.system_monitor.get_system_metrics()
            warnings = self.system_monitor.check_resource_thresholds(metrics)
            
            # Log overall health
            unhealthy_services = [
                service for service, result in results.items()
                if result.status == 'unhealthy'
            ]
            
            if unhealthy_services:
                self.logger.error(
                    f"Health check failed - Unhealthy services: {unhealthy_services}",
                    extra={
                        'health_check': 'summary',
                        'unhealthy_services': unhealthy_services,
                        'all_results': {s: r.status for s, r in results.items()}
                    }
                )
            elif warnings:
                self.logger.warning(
                    f"Health check warnings: {warnings}",
                    extra={
                        'health_check': 'summary',
                        'warnings': warnings,
                        'system_metrics': metrics
                    }
                )
            else:
                self.logger.info(
                    "Health check passed - All services healthy",
                    extra={
                        'health_check': 'summary',
                        'all_results': {s: r.status for s, r in results.items()}
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error running health check summary: {e}")


def setup_monitoring(app: Flask) -> ApplicationHealthMonitor:
    """Set up application monitoring"""
    monitor = ApplicationHealthMonitor(app)
    
    # Schedule periodic health checks (in production, use celery or similar)
    # For now, just log at startup
    monitor.log_health_summary()
    
    return monitor