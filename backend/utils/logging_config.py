#!/usr/bin/env python3
"""
Advanced logging configuration with correlation IDs, structured logging, and performance monitoring
"""
import logging
import logging.handlers
import json
import uuid
import time
import functools
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from flask import request, g, has_request_context
import threading

# Thread-local storage for correlation IDs
_local = threading.local()


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID and request context to log records"""
    
    def filter(self, record):
        # Add correlation ID
        correlation_id = getattr(_local, 'correlation_id', None)
        if not correlation_id and has_request_context():
            correlation_id = request.headers.get('X-Correlation-ID')
            if not correlation_id:
                correlation_id = str(uuid.uuid4())
                _local.correlation_id = correlation_id
        
        record.correlation_id = correlation_id or 'system'
        
        # Add request context if available
        if has_request_context():
            record.request_method = request.method
            record.request_path = request.path
            record.request_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            record.user_agent = request.headers.get('User-Agent', '')
        else:
            record.request_method = None
            record.request_path = None
            record.request_ip = None
            record.user_agent = None
            
        return True


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""
    
    def __init__(self, include_extra_fields=True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
        
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'system'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request context
        if hasattr(record, 'request_method') and record.request_method:
            log_entry['request'] = {
                'method': record.request_method,
                'path': record.request_path,
                'ip': record.request_ip,
                'user_agent': record.user_agent
            }
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        if self.include_extra_fields:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                              'relativeCreated', 'thread', 'threadName', 'processName',
                              'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info',
                              'correlation_id', 'request_method', 'request_path', 'request_ip',
                              'user_agent']:
                    extra_fields[key] = value
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Performance monitoring and logging"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_request_performance(self, start_time: float, end_time: float, 
                              operation: str, **kwargs):
        """Log request performance metrics"""
        duration = end_time - start_time
        
        self.logger.info(
            f"Performance: {operation} completed",
            extra={
                'operation': operation,
                'duration_ms': round(duration * 1000, 2),
                'performance_category': 'request',
                **kwargs
            }
        )
        
        # Log warnings for slow operations
        if duration > 5.0:  # 5 seconds
            self.logger.warning(
                f"Slow operation detected: {operation}",
                extra={
                    'operation': operation,
                    'duration_ms': round(duration * 1000, 2),
                    'performance_category': 'slow_operation',
                    **kwargs
                }
            )
            
    def log_ml_performance(self, model_name: str, operation: str, 
                          duration: float, **metrics):
        """Log ML model performance"""
        self.logger.info(
            f"ML Performance: {model_name} {operation}",
            extra={
                'model_name': model_name,
                'operation': operation,
                'duration_ms': round(duration * 1000, 2),
                'performance_category': 'ml_model',
                **metrics
            }
        )
        
    def log_database_performance(self, query_type: str, table: str, 
                               duration: float, row_count: Optional[int] = None):
        """Log database query performance"""
        extra_data = {
            'query_type': query_type,
            'table': table,
            'duration_ms': round(duration * 1000, 2),
            'performance_category': 'database'
        }
        
        if row_count is not None:
            extra_data['row_count'] = row_count
            
        self.logger.info(
            f"Database: {query_type} on {table}",
            extra=extra_data
        )
        
        # Warn on slow queries
        if duration > 1.0:  # 1 second
            self.logger.warning(
                f"Slow database query: {query_type} on {table}",
                extra={**extra_data, 'performance_category': 'slow_database'}
            )


class SecurityLogger:
    """Security event logging"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_authentication_attempt(self, username: str, success: bool, 
                                 ip_address: str, user_agent: str = ""):
        """Log authentication attempts"""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'}: {username}",
            extra={
                'event_type': 'authentication',
                'username': username,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'security_category': 'auth'
            }
        )
        
    def log_authorization_failure(self, username: str, resource: str, 
                                action: str, ip_address: str):
        """Log authorization failures"""
        self.logger.warning(
            f"Authorization denied: {username} attempted {action} on {resource}",
            extra={
                'event_type': 'authorization_failure',
                'username': username,
                'resource': resource,
                'action': action,
                'ip_address': ip_address,
                'security_category': 'authz'
            }
        )
        
    def log_suspicious_activity(self, event_type: str, description: str, 
                              ip_address: str, **kwargs):
        """Log suspicious activities"""
        self.logger.error(
            f"Suspicious activity detected: {event_type}",
            extra={
                'event_type': 'suspicious_activity',
                'description': description,
                'ip_address': ip_address,
                'security_category': 'threat',
                **kwargs
            }
        )
        
    def log_rate_limit_exceeded(self, ip_address: str, endpoint: str, 
                              limit: int, window: int):
        """Log rate limit violations"""
        self.logger.warning(
            f"Rate limit exceeded: {ip_address} hit {endpoint}",
            extra={
                'event_type': 'rate_limit_exceeded',
                'ip_address': ip_address,
                'endpoint': endpoint,
                'limit': limit,
                'window_seconds': window,
                'security_category': 'rate_limit'
            }
        )


def setup_logging(app_config: Dict[str, Any]) -> Dict[str, logging.Logger]:
    """Set up comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    import os
    log_dir = app_config.get('LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(app_config.get('LOG_LEVEL', 'INFO'))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    structured_formatter = StructuredFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(correlation_id)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create correlation ID filter
    correlation_filter = CorrelationIdFilter()
    
    # Console handler (for development)
    if app_config.get('LOG_TO_CONSOLE', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(correlation_filter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'app.log'),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(structured_formatter)
    file_handler.addFilter(correlation_filter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Error handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'errors.log'),
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(structured_formatter)
    error_handler.addFilter(correlation_filter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)
    
    # Performance handler
    performance_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'performance.log'),
        maxBytes=30 * 1024 * 1024,  # 30MB
        backupCount=7,
        encoding='utf-8'
    )
    performance_handler.setFormatter(structured_formatter)
    performance_handler.addFilter(correlation_filter)
    performance_handler.addFilter(lambda record: hasattr(record, 'performance_category'))
    performance_handler.setLevel(logging.INFO)
    root_logger.addHandler(performance_handler)
    
    # Security handler
    security_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'security.log'),
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    security_handler.setFormatter(structured_formatter)
    security_handler.addFilter(correlation_filter)
    security_handler.addFilter(lambda record: hasattr(record, 'security_category'))
    security_handler.setLevel(logging.INFO)
    root_logger.addHandler(security_handler)
    
    # Create specialized loggers
    app_logger = logging.getLogger('portfolio_optimizer')
    ml_logger = logging.getLogger('portfolio_optimizer.ml')
    api_logger = logging.getLogger('portfolio_optimizer.api')
    security_logger = logging.getLogger('portfolio_optimizer.security')
    performance_logger = logging.getLogger('portfolio_optimizer.performance')
    
    # Create helper classes
    perf_logger = PerformanceLogger(performance_logger)
    sec_logger = SecurityLogger(security_logger)
    
    return {
        'app': app_logger,
        'ml': ml_logger,
        'api': api_logger,
        'security': security_logger,
        'performance': performance_logger,
        'perf_helper': perf_logger,
        'sec_helper': sec_logger
    }


def get_correlation_id() -> str:
    """Get current correlation ID"""
    return getattr(_local, 'correlation_id', 'system')


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current thread"""
    _local.correlation_id = correlation_id


def clear_correlation_id():
    """Clear correlation ID for current thread"""
    if hasattr(_local, 'correlation_id'):
        delattr(_local, 'correlation_id')


def log_performance(operation: str, logger: Optional[logging.Logger] = None):
    """Decorator to log function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            _logger = logger or logging.getLogger(f'portfolio_optimizer.{func.__module__}')
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                _logger.info(
                    f"Performance: {operation} completed successfully",
                    extra={
                        'operation': operation,
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2),
                        'performance_category': 'function'
                    }
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                _logger.error(
                    f"Performance: {operation} failed with error",
                    extra={
                        'operation': operation,
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2),
                        'error': str(e),
                        'performance_category': 'function_error'
                    }
                )
                raise
                
        return wrapper
    return decorator


def log_api_call(logger: Optional[logging.Logger] = None):
    """Decorator to log API calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            _logger = logger or logging.getLogger('portfolio_optimizer.api')
            
            # Log request start
            if has_request_context():
                _logger.info(
                    f"API Request started: {request.method} {request.path}",
                    extra={
                        'api_endpoint': func.__name__,
                        'request_size': request.content_length or 0,
                        'api_category': 'request_start'
                    }
                )
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Log successful response
                status_code = getattr(result, 'status_code', 200)
                _logger.info(
                    f"API Request completed: {request.method} {request.path}",
                    extra={
                        'api_endpoint': func.__name__,
                        'status_code': status_code,
                        'duration_ms': round(duration * 1000, 2),
                        'api_category': 'request_complete'
                    }
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                _logger.error(
                    f"API Request failed: {request.method} {request.path}",
                    extra={
                        'api_endpoint': func.__name__,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'duration_ms': round(duration * 1000, 2),
                        'api_category': 'request_error'
                    }
                )
                raise
                
        return wrapper
    return decorator


class RequestLoggingMiddleware:
    """Middleware to log all HTTP requests"""
    
    def __init__(self, app, logger: logging.Logger):
        self.app = app
        self.logger = logger
        
    def __call__(self, environ, start_response):
        start_time = time.time()
        
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
        # Add correlation ID to WSGI environ
        environ['HTTP_X_CORRELATION_ID'] = correlation_id
        
        def logging_start_response(status, headers):
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract status code
            status_code = int(status.split()[0])
            
            # Log request completion
            self.logger.info(
                f"HTTP {environ['REQUEST_METHOD']} {environ['PATH_INFO']}",
                extra={
                    'method': environ['REQUEST_METHOD'],
                    'path': environ['PATH_INFO'],
                    'status_code': status_code,
                    'duration_ms': round(duration * 1000, 2),
                    'remote_addr': environ.get('REMOTE_ADDR'),
                    'user_agent': environ.get('HTTP_USER_AGENT', ''),
                    'api_category': 'http_request'
                }
            )
            
            # Add correlation ID to response headers
            headers.append(('X-Correlation-ID', correlation_id))
            
            return start_response(status, headers)
        
        try:
            return self.app(environ, logging_start_response)
        finally:
            clear_correlation_id()