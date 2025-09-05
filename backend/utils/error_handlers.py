#!/usr/bin/env python3
"""
Centralized error handling utilities with enhanced monitoring and recovery
"""
import logging
import traceback
import uuid
from functools import wraps
from typing import Dict, Any, Optional, Tuple, Callable
from flask import jsonify, current_app, request, has_request_context
from datetime import datetime, timedelta
from pydantic import ValidationError
import sys
import redis
from utils.logging_config import get_correlation_id

logger = logging.getLogger('portfolio_optimizer.errors')

class APIException(Exception):
    """Base API exception class"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(APIException):
    """Validation error exception"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, details)

class NotFoundError(APIException):
    """Resource not found exception"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404)

class ModelNotTrainedError(APIException):
    """Model not trained exception"""
    def __init__(self, ticker: str, model: str):
        message = f"{model} model for {ticker} is not trained"
        details = {
            'ticker': ticker,
            'model': model,
            'suggestion': f'Train the model using POST /api/train/{ticker}'
        }
        super().__init__(message, 422, details)

class ExternalAPIError(APIException):
    """External API error exception"""
    def __init__(self, service: str, message: str = "External service unavailable"):
        details = {'service': service}
        super().__init__(message, 503, details)

class TrainingError(APIException):
    """Model training error exception"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)

def handle_validation_error(error: ValidationError) -> Tuple[Dict[str, Any], int]:
    """Handle Pydantic validation errors"""
    errors = []
    for err in error.errors():
        field = '.'.join(str(loc) for loc in err['loc'])
        errors.append({
            'field': field,
            'message': err['msg'],
            'type': err['type']
        })
    
    return {
        'error': 'Validation failed',
        'message': 'Invalid input data',
        'details': {'validation_errors': errors},
        'timestamp': datetime.now().isoformat()
    }, 400

def handle_api_exception(error: APIException) -> Tuple[Dict[str, Any], int]:
    """Handle custom API exceptions"""
    return {
        'error': error.__class__.__name__,
        'message': error.message,
        'details': error.details,
        'timestamp': datetime.now().isoformat()
    }, error.status_code

def handle_generic_exception(error: Exception) -> Tuple[Dict[str, Any], int]:
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception: {str(error)}\n{traceback.format_exc()}")
    
    # Don't expose internal error details in production
    if current_app.config.get('DEBUG'):
        details = {
            'error_type': error.__class__.__name__,
            'traceback': traceback.format_exc()
        }
    else:
        details = {}
    
    return {
        'error': 'InternalServerError',
        'message': 'An unexpected error occurred',
        'details': details,
        'timestamp': datetime.now().isoformat()
    }, 500

def safe_api_call(func):
    """Decorator for safe API calls with comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            response_data, status_code = handle_validation_error(e)
            return jsonify(response_data), status_code
        except APIException as e:
            response_data, status_code = handle_api_exception(e)
            return jsonify(response_data), status_code
        except Exception as e:
            response_data, status_code = handle_generic_exception(e)
            return jsonify(response_data), status_code
    
    return wrapper

def log_api_call(endpoint: str, method: str, params: Dict[str, Any] = None):
    """Log API call for monitoring"""
    log_data = {
        'endpoint': endpoint,
        'method': method,
        'timestamp': datetime.now().isoformat(),
        'params': params or {}
    }
    logger.info(f"API Call: {log_data}")

def validate_ticker(ticker: str) -> str:
    """Validate and normalize stock ticker"""
    if not ticker or not isinstance(ticker, str):
        raise ValidationException("Ticker is required and must be a string")
    
    ticker = ticker.upper().strip()
    if len(ticker) < 1 or len(ticker) > 10:
        raise ValidationException("Ticker must be 1-10 characters")
    
    # Basic ticker format validation
    if not ticker.isalnum() and not all(c.isalnum() or c in ['.', '-'] for c in ticker):
        raise ValidationException("Ticker contains invalid characters")
    
    return ticker


class ErrorTracker:
    """Track and analyze application errors"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.error_tracker')
        
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track error occurrence for analysis"""
        error_id = str(uuid.uuid4())
        correlation_id = get_correlation_id()
        
        error_data = {
            'error_id': error_id,
            'correlation_id': correlation_id,
            'error_type': error.__class__.__name__,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context or {}
        }
        
        # Add request context if available
        if has_request_context():
            error_data['request'] = {
                'method': request.method,
                'path': request.path,
                'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                'user_agent': request.headers.get('User-Agent', ''),
                'args': dict(request.args),
                'json': request.get_json(silent=True)
            }
        
        # Store in Redis for analysis (if available)
        if self.redis_client:
            try:
                # Store error details
                self.redis_client.hset(
                    f"error:{error_id}", 
                    mapping={k: str(v) for k, v in error_data.items()}
                )
                self.redis_client.expire(f"error:{error_id}", 86400 * 7)  # 7 days
                
                # Track error frequency
                error_key = f"error_count:{error.__class__.__name__}:{datetime.utcnow().strftime('%Y-%m-%d:%H')}"
                self.redis_client.incr(error_key)
                self.redis_client.expire(error_key, 86400)  # 24 hours
                
            except Exception as e:
                self.logger.error(f"Failed to track error in Redis: {e}")
        
        # Log error with structured data
        self.logger.error(
            f"Error tracked: {error.__class__.__name__}",
            extra={
                'error_id': error_id,
                'error_tracking': True,
                **error_data
            }
        )
        
        return error_id
    
    def get_error_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.redis_client:
            return {}
        
        try:
            stats = {}
            now = datetime.utcnow()
            
            for hour in range(hours):
                hour_key = (now - timedelta(hours=hour)).strftime('%Y-%m-%d:%H')
                pattern = f"error_count:*:{hour_key}"
                
                for key in self.redis_client.scan_iter(match=pattern):
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    error_type = key_str.split(':')[1]
                    count = int(self.redis_client.get(key) or 0)
                    
                    if error_type not in stats:
                        stats[error_type] = {'total': 0, 'hourly': {}}
                    
                    stats[error_type]['total'] += count
                    stats[error_type]['hourly'][hour_key] = count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get error stats: {e}")
            return {}


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, 
                 failure_threshold: int = 5, 
                 recovery_timeout: int = 60,
                 redis_client: Optional[redis.Redis] = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.circuit_breaker')
        
    def call(self, service_name: str, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        state_key = f"circuit_breaker:{service_name}"
        failure_key = f"circuit_breaker_failures:{service_name}"
        
        try:
            # Check circuit state
            if self.redis_client:
                state = self.redis_client.get(state_key)
                if state == b'OPEN':
                    # Check if recovery timeout has passed
                    last_failure = self.redis_client.get(f"{failure_key}:last")
                    if last_failure:
                        last_failure_time = datetime.fromisoformat(last_failure.decode())
                        if datetime.utcnow() - last_failure_time < timedelta(seconds=self.recovery_timeout):
                            raise ExternalAPIError(
                                service_name, 
                                f"Service {service_name} is unavailable (circuit breaker OPEN)"
                            )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Reset failure count on success
            if self.redis_client:
                self.redis_client.delete(failure_key)
                self.redis_client.set(state_key, 'CLOSED', ex=3600)
            
            return result
            
        except Exception as e:
            # Increment failure count
            if self.redis_client:
                failures = self.redis_client.incr(failure_key)
                self.redis_client.expire(failure_key, 3600)  # 1 hour
                
                # Open circuit if threshold reached
                if failures >= self.failure_threshold:
                    self.redis_client.set(state_key, 'OPEN', ex=self.recovery_timeout)
                    self.redis_client.set(
                        f"{failure_key}:last", 
                        datetime.utcnow().isoformat(),
                        ex=self.recovery_timeout
                    )
                    
                    self.logger.error(
                        f"Circuit breaker OPENED for {service_name} after {failures} failures",
                        extra={
                            'service': service_name,
                            'failures': failures,
                            'threshold': self.failure_threshold,
                            'circuit_breaker': True
                        }
                    )
            
            raise e


def enhanced_safe_api_call(error_tracker: Optional[ErrorTracker] = None):
    """Enhanced decorator with error tracking and monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except ValidationError as e:
                if error_tracker:
                    error_tracker.track_error(e, {'validation_errors': e.errors()})
                response_data, status_code = handle_validation_error(e)
                return jsonify(response_data), status_code
                
            except APIException as e:
                if error_tracker:
                    error_tracker.track_error(e, {'api_exception': True})
                response_data, status_code = handle_api_exception(e)
                return jsonify(response_data), status_code
                
            except Exception as e:
                error_id = None
                if error_tracker:
                    error_id = error_tracker.track_error(e, {'unexpected_error': True})
                
                response_data, status_code = handle_generic_exception(e)
                
                # Add error ID to response for tracking
                if error_id:
                    response_data['error_id'] = error_id
                
                return jsonify(response_data), status_code
                
        return wrapper
    return decorator


class RetryHandler:
    """Retry handler for transient failures"""
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.logger = logging.getLogger('portfolio_optimizer.retry')
        
    def retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Don't retry certain types of errors
                if isinstance(e, (ValidationException, NotFoundError)):
                    raise e
                
                if attempt < self.max_attempts - 1:
                    delay = self.delay * (self.backoff ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s",
                        extra={
                            'attempt': attempt + 1,
                            'max_attempts': self.max_attempts,
                            'delay': delay,
                            'error': str(e),
                            'retry_handler': True
                        }
                    )
                    import time
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_attempts} attempts failed",
                        extra={
                            'max_attempts': self.max_attempts,
                            'final_error': str(e),
                            'retry_handler': True
                        }
                    )
        
        raise last_exception


def setup_error_handlers(app, redis_client: Optional[redis.Redis] = None):
    """Set up comprehensive error handling for Flask app"""
    error_tracker = ErrorTracker(redis_client)
    
    @app.errorhandler(400)
    def bad_request(error):
        error_tracker.track_error(Exception("Bad Request"), {'http_error': 400})
        return jsonify({
            'error': 'BadRequest',
            'message': 'Invalid request',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        error_tracker.track_error(Exception("Unauthorized"), {'http_error': 401})
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        error_tracker.track_error(Exception("Forbidden"), {'http_error': 403})
        return jsonify({
            'error': 'Forbidden',
            'message': 'Access denied',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'NotFound',
            'message': 'Resource not found',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 404
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        error_tracker.track_error(Exception("Rate Limit Exceeded"), {'http_error': 429})
        return jsonify({
            'error': 'RateLimitExceeded',
            'message': 'Rate limit exceeded',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        error_tracker.track_error(error.original_exception if hasattr(error, 'original_exception') else Exception("Internal Server Error"))
        return jsonify({
            'error': 'InternalServerError',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': get_correlation_id()
        }), 500
    
    # Add error monitoring endpoints
    @app.route('/api/admin/errors/stats')
    def get_error_stats():
        """Get error statistics (admin only)"""
        stats = error_tracker.get_error_stats()
        return jsonify({
            'error_stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    return error_tracker

def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """Validate date range"""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValidationException("Dates must be in YYYY-MM-DD format")
    
    if start >= end:
        raise ValidationException("Start date must be before end date")
    
    if (end - start).days < 30:
        raise ValidationException("Date range must be at least 30 days")
    
    if end > datetime.now():
        raise ValidationException("End date cannot be in the future")
    
    return start, end

def validate_prediction_params(days: int, models: str) -> Tuple[int, str]:
    """Validate prediction parameters"""
    if not isinstance(days, int) or days < 1 or days > 30:
        raise ValidationException("Days must be an integer between 1 and 30")
    
    valid_models = ['prophet', 'xgboost', 'lstm', 'ensemble', 'all']
    if models not in valid_models:
        raise ValidationException(f"Model must be one of: {', '.join(valid_models)}")
    
    return days, models