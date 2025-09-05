#!/usr/bin/env python3
"""
Centralized Security Configuration
Security settings and middleware for the entire application
"""

import os
import secrets
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from datetime import timedelta

class SecurityConfig:
    """Centralized security configuration"""
    
    # JWT Security Settings
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_ALGORITHM = "HS256"
    JWT_BLACKLIST_ENABLED = True
    JWT_BLACKLIST_TOKEN_CHECKS = ['access', 'refresh']
    
    # Rate Limiting Settings
    RATE_LIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    
    # Default Rate Limits (per endpoint type)
    RATE_LIMITS = {
        'public': "100 per minute",
        'authenticated': "200 per minute", 
        'ml_prediction': "30 per minute",
        'ml_training': "5 per hour",
        'admin': "50 per minute",
        'batch_operations': "10 per hour"
    }
    
    # Content Security Policy
    CSP = {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline'",
        'img-src': "'self' data: https:",
        'font-src': "'self'",
        'connect-src': "'self'",
        'frame-ancestors': "'none'",
        'base-uri': "'self'",
        'form-action': "'self'"
    }
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    # Input Validation Limits
    MAX_REQUEST_SIZE = 16 * 1024 * 1024  # 16MB
    MAX_JSON_DEPTH = 10
    MAX_ARRAY_SIZE = 1000
    MAX_STRING_LENGTH = 10000
    
    # File Upload Security
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    # Session Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

def apply_security_middleware(app: Flask) -> Flask:
    """
    Apply comprehensive security middleware to Flask app
    
    Args:
        app: Flask application instance
    
    Returns:
        Configured Flask app with security middleware
    """
    
    # 1. Configure Flask-Talisman for security headers
    if not app.config.get('TESTING'):  # Skip in testing
        talisman_config = {
            'force_https': app.config.get('FLASK_ENV') == 'production',
            'content_security_policy': SecurityConfig.CSP,
            'content_security_policy_nonce_in': ['script-src'],
            'referrer_policy': 'strict-origin-when-cross-origin',
            'feature_policy': {
                'geolocation': "'none'",
                'microphone': "'none'",
                'camera': "'none'"
            }
        }
        
        Talisman(app, **talisman_config)
    
    # 2. Configure session security
    app.config.update({
        'SESSION_COOKIE_SECURE': SecurityConfig.SESSION_COOKIE_SECURE,
        'SESSION_COOKIE_HTTPONLY': SecurityConfig.SESSION_COOKIE_HTTPONLY,
        'SESSION_COOKIE_SAMESITE': SecurityConfig.SESSION_COOKIE_SAMESITE,
        'PERMANENT_SESSION_LIFETIME': SecurityConfig.PERMANENT_SESSION_LIFETIME
    })
    
    # 3. Configure JWT security
    app.config.update({
        'JWT_ACCESS_TOKEN_EXPIRES': SecurityConfig.JWT_ACCESS_TOKEN_EXPIRES,
        'JWT_REFRESH_TOKEN_EXPIRES': SecurityConfig.JWT_REFRESH_TOKEN_EXPIRES,
        'JWT_ALGORITHM': SecurityConfig.JWT_ALGORITHM,
        'JWT_BLACKLIST_ENABLED': SecurityConfig.JWT_BLACKLIST_ENABLED,
        'JWT_BLACKLIST_TOKEN_CHECKS': SecurityConfig.JWT_BLACKLIST_TOKEN_CHECKS
    })
    
    # 4. Configure request limits
    app.config['MAX_CONTENT_LENGTH'] = SecurityConfig.MAX_REQUEST_SIZE
    
    # 5. Add security headers middleware
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses"""
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
    
    # 6. Add request validation middleware
    @app.before_request
    def validate_request():
        """Validate incoming requests for security"""
        from flask import request, abort
        import json
        
        # Check content type for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            if request.content_type and not request.content_type.startswith('application/json'):
                if not request.content_type.startswith('multipart/form-data'):
                    abort(400, "Invalid content type")
        
        # Validate JSON structure depth
        if request.is_json:
            try:
                data = request.get_json()
                if data is not None:
                    _validate_json_structure(data)
            except (ValueError, json.JSONDecodeError):
                abort(400, "Invalid JSON structure")
        
        # Check for suspicious user agents
        user_agent = request.headers.get('User-Agent', '').lower()
        suspicious_agents = ['sqlmap', 'nikto', 'nmap', 'masscan', 'burp']
        if any(agent in user_agent for agent in suspicious_agents):
            abort(403, "Blocked")
    
    # 7. Add error handler for security errors
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle request too large errors"""
        return {
            'error': 'RequestTooLarge',
            'message': 'Request entity too large'
        }, 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit exceeded errors"""
        return {
            'error': 'RateLimitExceeded',
            'message': 'Too many requests',
            'retry_after': str(error.retry_after) if hasattr(error, 'retry_after') else '60'
        }, 429
    
    return app

def _validate_json_structure(data, max_depth=10, current_depth=0):
    """
    Validate JSON structure to prevent deeply nested payloads
    
    Args:
        data: JSON data to validate
        max_depth: Maximum allowed nesting depth
        current_depth: Current nesting depth
    
    Raises:
        ValueError: If structure violates security constraints
    """
    if current_depth > max_depth:
        raise ValueError("JSON structure too deeply nested")
    
    if isinstance(data, dict):
        if len(data) > 100:  # Limit number of keys
            raise ValueError("Too many keys in JSON object")
        
        for key, value in data.items():
            if isinstance(key, str) and len(key) > 100:
                raise ValueError("JSON key too long")
            _validate_json_structure(value, max_depth, current_depth + 1)
    
    elif isinstance(data, list):
        if len(data) > SecurityConfig.MAX_ARRAY_SIZE:
            raise ValueError("JSON array too large")
        
        for item in data:
            _validate_json_structure(item, max_depth, current_depth + 1)
    
    elif isinstance(data, str):
        if len(data) > SecurityConfig.MAX_STRING_LENGTH:
            raise ValueError("JSON string too long")

def generate_secure_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure key
    
    Args:
        length: Key length in bytes
    
    Returns:
        Secure random key as hex string
    """
    return secrets.token_hex(length)

def create_rate_limiter(app: Flask) -> Limiter:
    """
    Create and configure rate limiter
    
    Args:
        app: Flask application
    
    Returns:
        Configured Limiter instance
    """
    return Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri=SecurityConfig.RATE_LIMIT_STORAGE_URL,
        default_limits=[SecurityConfig.RATE_LIMITS['public']],
        headers_enabled=True
    )

# Security audit functions
def log_security_event(event_type: str, details: dict, app: Flask):
    """Log security events for monitoring"""
    security_logger = app.logger.getChild('security')
    security_logger.warning(f"SECURITY_EVENT: {event_type} - {details}")

def check_environment_security():
    """
    Check environment for security misconfigurations
    
    Returns:
        List of security warnings
    """
    warnings = []
    
    # Check for required environment variables
    required_vars = ['SECRET_KEY', 'JWT_SECRET_KEY']
    for var in required_vars:
        if not os.environ.get(var):
            warnings.append(f"Missing required environment variable: {var}")
    
    # Check secret key strength
    secret_key = os.environ.get('SECRET_KEY', '')
    if len(secret_key) < 32:
        warnings.append("SECRET_KEY should be at least 32 characters long")
    
    jwt_secret = os.environ.get('JWT_SECRET_KEY', '')
    if len(jwt_secret) < 32:
        warnings.append("JWT_SECRET_KEY should be at least 32 characters long")
    
    if secret_key == jwt_secret:
        warnings.append("SECRET_KEY and JWT_SECRET_KEY should be different")
    
    # Check for development defaults in production
    if os.environ.get('FLASK_ENV') == 'production':
        if 'dev-secret' in secret_key.lower():
            warnings.append("Using development secret key in production")
        
        if not os.environ.get('DATABASE_URL', '').startswith('postgresql'):
            warnings.append("Production should use PostgreSQL database")
    
    return warnings

# Export security utilities
__all__ = [
    'SecurityConfig',
    'apply_security_middleware',
    'generate_secure_key',
    'create_rate_limiter',
    'log_security_event',
    'check_environment_security'
]