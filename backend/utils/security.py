#!/usr/bin/env python3
"""
Security Utilities and Decorators
Enhanced security functions for API endpoints
"""

from functools import wraps
from flask import request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import ValidationError
from typing import Dict, Any, Type, Optional
import logging
import time
import hashlib
import secrets
import re

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute", "1000 per hour"]
)

def secure_api_endpoint(
    schema: Optional[Type] = None,
    require_auth: bool = False,
    rate_limit: str = "60 per minute",
    admin_only: bool = False
):
    """
    Comprehensive security decorator for API endpoints
    
    Args:
        schema: Pydantic schema for request validation
        require_auth: Whether JWT authentication is required
        rate_limit: Rate limit string (e.g., "60 per minute")
        admin_only: Whether endpoint requires admin privileges
    """
    def decorator(f):
        @wraps(f)
        @limiter.limit(rate_limit)
        def decorated_function(*args, **kwargs):
            try:
                # 1. Authentication check
                if require_auth or admin_only:
                    try:
                        from flask_jwt_extended import verify_jwt_in_request
                        verify_jwt_in_request()
                        user_id = get_jwt_identity()
                        
                        if not user_id:
                            return jsonify({
                                'error': 'AuthenticationRequired',
                                'message': 'Valid authentication token required'
                            }), 401
                            
                        # Admin check
                        if admin_only:
                            # You would implement admin check logic here
                            # For now, assuming admin status is in JWT claims
                            from flask_jwt_extended import get_jwt
                            claims = get_jwt()
                            if not claims.get('is_admin', False):
                                return jsonify({
                                    'error': 'InsufficientPermissions',
                                    'message': 'Admin privileges required'
                                }), 403
                                
                    except Exception as e:
                        logger.warning(f"Authentication failed: {str(e)}")
                        return jsonify({
                            'error': 'AuthenticationFailed',
                            'message': 'Invalid or expired token'
                        }), 401
                
                # 2. Input validation
                if schema and request.method in ['POST', 'PUT', 'PATCH']:
                    try:
                        json_data = request.get_json()
                        if json_data is None:
                            return jsonify({
                                'error': 'InvalidRequest',
                                'message': 'Request must contain valid JSON'
                            }), 400
                        
                        # Validate against schema
                        validated_data = schema(**json_data)
                        request.validated_json = validated_data.dict()
                        
                    except ValidationError as e:
                        logger.warning(f"Validation error for {request.endpoint}: {e}")
                        return jsonify({
                            'error': 'ValidationError',
                            'message': 'Request data validation failed',
                            'details': e.errors()
                        }), 400
                    except Exception as e:
                        logger.error(f"Unexpected validation error: {str(e)}")
                        return jsonify({
                            'error': 'ValidationError',
                            'message': 'Invalid request format'
                        }), 400
                
                # 3. Security headers
                response = f(*args, **kwargs)
                
                # Add security headers to response
                if hasattr(response, 'headers'):
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    response.headers['X-Frame-Options'] = 'DENY'
                    response.headers['X-XSS-Protection'] = '1; mode=block'
                    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
                    response.headers['Content-Security-Policy'] = "default-src 'self'"
                
                return response
                
            except Exception as e:
                logger.error(f"Unexpected error in {request.endpoint}: {str(e)}")
                return jsonify({
                    'error': 'InternalServerError',
                    'message': 'An unexpected error occurred'
                }), 500
                
        return decorated_function
    return decorator

def sanitize_input(input_string: str, max_length: int = 255) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        input_string: Raw input string
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ""
    
    # Truncate to max length
    sanitized = input_string[:max_length]
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';\\]', '', sanitized)
    
    # Strip whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def validate_sql_query(query: str) -> bool:
    """
    Validate SQL query for potential injection attacks
    
    Args:
        query: SQL query string
    
    Returns:
        True if query appears safe, False otherwise
    """
    if not isinstance(query, str):
        return False
    
    # Convert to lowercase for checking
    query_lower = query.lower()
    
    # Check for dangerous SQL keywords
    dangerous_keywords = [
        'drop', 'delete', 'truncate', 'alter', 'create',
        'insert', 'update', 'exec', 'execute', 'sp_',
        'xp_', 'union', 'script', 'javascript', 'vbscript'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
            return False
    
    # Check for SQL comment markers
    if '--' in query or '/*' in query or '*/' in query:
        logger.warning("SQL comment markers detected")
        return False
    
    return True

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token
    
    Args:
        length: Length of the token
    
    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(length)

def hash_sensitive_data(data: str, salt: Optional[str] = None) -> tuple:
    """
    Hash sensitive data with salt
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
    
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Create hash using SHA-256 with salt
    hash_obj = hashlib.sha256((data + salt).encode('utf-8'))
    hash_value = hash_obj.hexdigest()
    
    return hash_value, salt

def verify_hash(data: str, hash_value: str, salt: str) -> bool:
    """
    Verify data against hash and salt
    
    Args:
        data: Original data
        hash_value: Hash to verify against
        salt: Salt used in hashing
    
    Returns:
        True if data matches hash, False otherwise
    """
    computed_hash, _ = hash_sensitive_data(data, salt)
    return computed_hash == hash_value

class SecurityAudit:
    """Security audit logging"""
    
    @staticmethod
    def log_security_event(event_type: str, details: Dict[str, Any], user_id: Optional[str] = None):
        """
        Log security-related events
        
        Args:
            event_type: Type of security event
            details: Event details
            user_id: User ID if available
        """
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': get_remote_address(),
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'details': details
        }
        
        logger.info(f"SECURITY_AUDIT: {audit_entry}")
    
    @staticmethod
    def log_failed_authentication(email: str, reason: str):
        """Log failed authentication attempts"""
        SecurityAudit.log_security_event(
            'FAILED_AUTHENTICATION',
            {'email': email, 'reason': reason}
        )
    
    @staticmethod
    def log_suspicious_activity(activity_type: str, details: Dict[str, Any]):
        """Log suspicious activities"""
        SecurityAudit.log_security_event(
            'SUSPICIOUS_ACTIVITY',
            {'activity_type': activity_type, **details}
        )

# Input sanitization for specific use cases
def sanitize_ticker(ticker: str) -> str:
    """Sanitize stock ticker input"""
    if not ticker:
        return ""
    
    # Only allow alphanumeric, dots, and hyphens
    sanitized = re.sub(r'[^A-Z0-9.-]', '', ticker.upper())
    return sanitized[:10]  # Max 10 characters

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    if not filename:
        return ""
    
    # Remove path traversal attempts
    sanitized = filename.replace('..', '').replace('/', '').replace('\\', '')
    
    # Only allow safe characters
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
    
    return sanitized[:100]  # Max 100 characters

def validate_json_structure(data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
    """
    Validate JSON structure to prevent deeply nested payloads
    
    Args:
        data: JSON data to validate
        max_depth: Maximum allowed nesting depth
        current_depth: Current nesting depth
    
    Returns:
        True if structure is safe, False otherwise
    """
    if current_depth > max_depth:
        return False
    
    if isinstance(data, dict):
        if len(data) > 100:  # Limit number of keys
            return False
        return all(validate_json_structure(v, max_depth, current_depth + 1) for v in data.values())
    
    elif isinstance(data, list):
        if len(data) > 1000:  # Limit array size
            return False
        return all(validate_json_structure(item, max_depth, current_depth + 1) for item in data)
    
    return True

# Rate limiting helpers
def get_user_rate_limit_key():
    """Get rate limit key for authenticated users"""
    try:
        user_id = get_jwt_identity()
        if user_id:
            return f"user_{user_id}"
    except:
        pass
    
    return get_remote_address()

def create_custom_rate_limit(rate: str):
    """Create custom rate limit decorator"""
    def decorator(f):
        @wraps(f)
        @limiter.limit(rate, key_func=get_user_rate_limit_key)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return decorator