#!/usr/bin/env python3
"""
Advanced security features including JWT refresh rotation, rate limiting, 
account lockout, and intrusion detection
"""
import hashlib
import hmac
import secrets
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import redis
from flask import request, current_app
from flask_jwt_extended import get_jwt, decode_token, create_access_token, create_refresh_token
import logging
from werkzeug.security import check_password_hash

logger = logging.getLogger('portfolio_optimizer.security')


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: str
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: str  # 'low', 'medium', 'high', 'critical'


class AdvancedRateLimiter:
    """Advanced rate limiting with sliding window and user-based limits"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.rate_limiter')
        
    def is_rate_limited(self, 
                       identifier: str, 
                       limit: int, 
                       window_seconds: int,
                       burst_limit: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if identifier is rate limited using sliding window
        
        Args:
            identifier: IP address or user ID
            limit: Requests per window
            window_seconds: Time window in seconds
            burst_limit: Maximum burst requests (optional)
        """
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Clean old entries
        key = f"rate_limit:{identifier}"
        pipe = self.redis_client.pipeline()
        
        # Remove entries outside window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window_seconds + 1)
        
        results = pipe.execute()
        current_requests = results[1]
        
        # Check burst limit (last minute)
        if burst_limit:
            burst_window = current_time - 60  # 1 minute
            burst_count = self.redis_client.zcount(key, burst_window, current_time)
            if burst_count > burst_limit:
                return True, {
                    'reason': 'burst_limit_exceeded',
                    'current_requests': burst_count,
                    'burst_limit': burst_limit,
                    'window': '1 minute'
                }
        
        # Check main rate limit
        is_limited = current_requests >= limit
        
        if is_limited:
            self.logger.warning(
                f"Rate limit exceeded for {identifier}",
                extra={
                    'identifier': identifier,
                    'current_requests': current_requests,
                    'limit': limit,
                    'window_seconds': window_seconds,
                    'security_event': True
                }
            )
        
        return is_limited, {
            'current_requests': current_requests,
            'limit': limit,
            'window_seconds': window_seconds,
            'reset_time': current_time + window_seconds
        }
    
    def get_rate_limit_info(self, identifier: str, window_seconds: int) -> Dict[str, Any]:
        """Get current rate limit status"""
        current_time = int(time.time())
        window_start = current_time - window_seconds
        key = f"rate_limit:{identifier}"
        
        # Count requests in current window
        current_requests = self.redis_client.zcount(key, window_start, current_time)
        
        return {
            'current_requests': current_requests,
            'window_seconds': window_seconds,
            'reset_time': current_time + window_seconds
        }


class AccountSecurityManager:
    """Advanced account security with lockout and suspicious activity detection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.account_security')
        
        # Configuration
        self.max_failed_attempts = 5
        self.lockout_duration = 3600  # 1 hour
        self.suspicious_threshold = 3
        self.monitoring_window = 1800  # 30 minutes
        
    def record_login_attempt(self, username: str, ip_address: str, success: bool, 
                           user_agent: str = "") -> Dict[str, Any]:
        """Record login attempt and check for suspicious activity"""
        timestamp = int(time.time())
        
        # Keys for different tracking
        failed_key = f"failed_logins:{username}"
        ip_failed_key = f"failed_logins_ip:{ip_address}"
        attempt_key = f"login_attempts:{username}"
        
        if success:
            # Clear failed attempts on successful login
            self.redis_client.delete(failed_key)
            self.redis_client.delete(ip_failed_key)
            
            # Log successful login
            self.logger.info(
                f"Successful login: {username}",
                extra={
                    'username': username,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'security_event': True,
                    'event_type': 'successful_login'
                }
            )
            
            return {'status': 'success', 'account_locked': False}
            
        else:
            # Record failed attempt
            pipe = self.redis_client.pipeline()
            
            # Increment failed attempts for user
            pipe.incr(failed_key)
            pipe.expire(failed_key, self.lockout_duration)
            
            # Increment failed attempts for IP
            pipe.incr(ip_failed_key)
            pipe.expire(ip_failed_key, self.lockout_duration)
            
            # Record attempt details
            attempt_data = {
                'timestamp': timestamp,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'success': False
            }
            
            pipe.lpush(attempt_key, json.dumps(attempt_data))
            pipe.ltrim(attempt_key, 0, 99)  # Keep last 100 attempts
            pipe.expire(attempt_key, 86400)  # 24 hours
            
            results = pipe.execute()
            failed_count = results[0]
            ip_failed_count = results[1]
            
            # Check if account should be locked
            account_locked = failed_count >= self.max_failed_attempts
            
            if account_locked:
                self.redis_client.setex(f"account_locked:{username}", self.lockout_duration, timestamp)
                
                self.logger.error(
                    f"Account locked: {username} after {failed_count} failed attempts",
                    extra={
                        'username': username,
                        'failed_attempts': failed_count,
                        'ip_address': ip_address,
                        'security_event': True,
                        'event_type': 'account_locked'
                    }
                )
            
            # Check for suspicious IP activity
            if ip_failed_count >= self.suspicious_threshold:
                self.redis_client.setex(f"suspicious_ip:{ip_address}", self.monitoring_window, timestamp)
                
                self.logger.warning(
                    f"Suspicious IP activity: {ip_address}",
                    extra={
                        'ip_address': ip_address,
                        'failed_attempts': ip_failed_count,
                        'security_event': True,
                        'event_type': 'suspicious_ip'
                    }
                )
            
            # Log failed attempt
            self.logger.warning(
                f"Failed login attempt: {username}",
                extra={
                    'username': username,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'failed_attempts': failed_count,
                    'security_event': True,
                    'event_type': 'failed_login'
                }
            )
            
            return {
                'status': 'failed',
                'failed_attempts': failed_count,
                'account_locked': account_locked,
                'max_attempts': self.max_failed_attempts,
                'lockout_duration': self.lockout_duration
            }
    
    def is_account_locked(self, username: str) -> Tuple[bool, Optional[int]]:
        """Check if account is locked"""
        locked_until = self.redis_client.get(f"account_locked:{username}")
        
        if locked_until:
            locked_timestamp = int(locked_until)
            unlock_time = locked_timestamp + self.lockout_duration
            current_time = int(time.time())
            
            if current_time < unlock_time:
                return True, unlock_time - current_time
            else:
                # Lock expired, clean up
                self.redis_client.delete(f"account_locked:{username}")
        
        return False, None
    
    def is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP is flagged as suspicious"""
        return self.redis_client.exists(f"suspicious_ip:{ip_address}")
    
    def get_login_history(self, username: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get login attempt history"""
        attempts = self.redis_client.lrange(f"login_attempts:{username}", 0, limit - 1)
        
        history = []
        for attempt in attempts:
            try:
                data = json.loads(attempt)
                data['datetime'] = datetime.fromtimestamp(data['timestamp']).isoformat()
                history.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        
        return history


class JWTSecurityManager:
    """Advanced JWT security with refresh token rotation and blacklisting"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.jwt_security')
        
        # Token lifetimes
        self.access_token_lifetime = timedelta(minutes=15)
        self.refresh_token_lifetime = timedelta(days=30)
        self.max_refresh_tokens = 5  # Maximum refresh tokens per user
        
    def create_token_pair(self, user_identity: Dict[str, Any]) -> Dict[str, str]:
        """Create access and refresh token pair with rotation"""
        user_id = user_identity['user_id']
        
        # Generate new token IDs
        access_jti = secrets.token_urlsafe(32)
        refresh_jti = secrets.token_urlsafe(32)
        
        # Create tokens
        access_token = create_access_token(
            identity=user_identity,
            expires_delta=self.access_token_lifetime,
            additional_claims={'jti': access_jti, 'token_type': 'access'}
        )
        
        refresh_token = create_refresh_token(
            identity=user_identity,
            expires_delta=self.refresh_token_lifetime,
            additional_claims={'jti': refresh_jti, 'token_type': 'refresh'}
        )
        
        # Store refresh token info
        refresh_key = f"refresh_tokens:{user_id}"
        token_data = {
            'jti': refresh_jti,
            'created_at': int(time.time()),
            'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
            'user_agent': request.headers.get('User-Agent', '')[:200]  # Truncate
        }
        
        # Add to user's refresh token list
        pipe = self.redis_client.pipeline()
        pipe.lpush(refresh_key, json.dumps(token_data))
        pipe.ltrim(refresh_key, 0, self.max_refresh_tokens - 1)  # Keep only recent tokens
        pipe.expire(refresh_key, int(self.refresh_token_lifetime.total_seconds()))
        pipe.execute()
        
        self.logger.info(
            f"Token pair created for user {user_id}",
            extra={
                'user_id': user_id,
                'access_jti': access_jti,
                'refresh_jti': refresh_jti,
                'security_event': True,
                'event_type': 'token_created'
            }
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': int(self.access_token_lifetime.total_seconds())
        }
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token with rotation"""
        try:
            # Decode refresh token
            token_data = decode_token(refresh_token)
            user_identity = token_data['sub']
            old_refresh_jti = token_data['jti']
            
            # Check if refresh token is blacklisted
            if self.is_token_blacklisted(old_refresh_jti):
                raise ValueError("Refresh token is blacklisted")
            
            # Verify refresh token exists in user's token list
            if not self._verify_refresh_token(user_identity['user_id'], old_refresh_jti):
                raise ValueError("Refresh token not found or invalid")
            
            # Blacklist old refresh token
            self.blacklist_token(old_refresh_jti)
            
            # Create new token pair
            new_tokens = self.create_token_pair(user_identity)
            
            self.logger.info(
                f"Access token refreshed for user {user_identity['user_id']}",
                extra={
                    'user_id': user_identity['user_id'],
                    'old_refresh_jti': old_refresh_jti,
                    'security_event': True,
                    'event_type': 'token_refreshed'
                }
            )
            
            return new_tokens
            
        except Exception as e:
            self.logger.error(
                f"Token refresh failed: {e}",
                extra={
                    'error': str(e),
                    'security_event': True,
                    'event_type': 'token_refresh_failed'
                }
            )
            raise ValueError("Invalid refresh token")
    
    def blacklist_token(self, jti: str, expires_in: Optional[int] = None):
        """Blacklist a token"""
        if expires_in is None:
            expires_in = int(self.refresh_token_lifetime.total_seconds())
        
        self.redis_client.setex(f"blacklist:{jti}", expires_in, int(time.time()))
        
        self.logger.info(
            f"Token blacklisted: {jti}",
            extra={
                'jti': jti,
                'security_event': True,
                'event_type': 'token_blacklisted'
            }
        )
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        return self.redis_client.exists(f"blacklist:{jti}")
    
    def revoke_all_user_tokens(self, user_id: int):
        """Revoke all tokens for a user"""
        # Get all refresh tokens
        refresh_key = f"refresh_tokens:{user_id}"
        tokens = self.redis_client.lrange(refresh_key, 0, -1)
        
        # Blacklist all refresh tokens
        for token in tokens:
            try:
                token_data = json.loads(token)
                self.blacklist_token(token_data['jti'])
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Clear refresh token list
        self.redis_client.delete(refresh_key)
        
        self.logger.info(
            f"All tokens revoked for user {user_id}",
            extra={
                'user_id': user_id,
                'tokens_revoked': len(tokens),
                'security_event': True,
                'event_type': 'all_tokens_revoked'
            }
        )
    
    def _verify_refresh_token(self, user_id: int, jti: str) -> bool:
        """Verify refresh token exists in user's active tokens"""
        refresh_key = f"refresh_tokens:{user_id}"
        tokens = self.redis_client.lrange(refresh_key, 0, -1)
        
        for token in tokens:
            try:
                token_data = json.loads(token)
                if token_data['jti'] == jti:
                    return True
            except (json.JSONDecodeError, KeyError):
                continue
        
        return False


class IntrusionDetectionSystem:
    """Detect and respond to potential intrusion attempts"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger('portfolio_optimizer.ids')
        
        # Detection thresholds
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(\bunion\s+select\b)",
            r"(\bor\s+1\s*=\s*1\b)",
            r"(\band\s+1\s*=\s*1\b)",
            r"(--\s*$|#\s*$|\/\*.*\*\/)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"eval\s*\("
        ]
        
    def analyze_request(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze request for suspicious patterns"""
        events = []
        
        # Check for SQL injection attempts
        for pattern in self.sql_injection_patterns:
            if self._check_pattern(pattern, request_data):
                events.append(SecurityEvent(
                    event_type='sql_injection_attempt',
                    user_id=request_data.get('user_id'),
                    ip_address=request_data.get('ip_address', ''),
                    user_agent=request_data.get('user_agent', ''),
                    timestamp=datetime.utcnow(),
                    details={'pattern': pattern, 'request_data': request_data},
                    severity='high'
                ))
        
        # Check for XSS attempts
        for pattern in self.xss_patterns:
            if self._check_pattern(pattern, request_data):
                events.append(SecurityEvent(
                    event_type='xss_attempt',
                    user_id=request_data.get('user_id'),
                    ip_address=request_data.get('ip_address', ''),
                    user_agent=request_data.get('user_agent', ''),
                    timestamp=datetime.utcnow(),
                    details={'pattern': pattern, 'request_data': request_data},
                    severity='high'
                ))
        
        # Check for unusual request patterns
        if self._is_unusual_request(request_data):
            events.append(SecurityEvent(
                event_type='unusual_request_pattern',
                user_id=request_data.get('user_id'),
                ip_address=request_data.get('ip_address', ''),
                user_agent=request_data.get('user_agent', ''),
                timestamp=datetime.utcnow(),
                details={'request_data': request_data},
                severity='medium'
            ))
        
        # Log events
        for event in events:
            self.logger.error(
                f"Security threat detected: {event.event_type}",
                extra={
                    'security_event': True,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'ip_address': event.ip_address,
                    'user_id': event.user_id,
                    'details': event.details
                }
            )
        
        return events
    
    def _check_pattern(self, pattern: str, request_data: Dict[str, Any]) -> bool:
        """Check if pattern matches in request data"""
        import re
        
        # Convert request data to searchable string
        search_text = json.dumps(request_data, default=str).lower()
        
        return bool(re.search(pattern, search_text, re.IGNORECASE))
    
    def _is_unusual_request(self, request_data: Dict[str, Any]) -> bool:
        """Check for unusual request patterns"""
        # Check for unusually large payloads
        payload_size = len(json.dumps(request_data, default=str))
        if payload_size > 100000:  # 100KB
            return True
        
        # Check for unusual parameter counts
        if isinstance(request_data.get('params'), dict):
            if len(request_data['params']) > 50:
                return True
        
        # Add more pattern checks as needed
        return False


def setup_advanced_security(app, redis_client: redis.Redis):
    """Set up advanced security features"""
    rate_limiter = AdvancedRateLimiter(redis_client)
    account_security = AccountSecurityManager(redis_client)
    jwt_security = JWTSecurityManager(redis_client)
    ids = IntrusionDetectionSystem(redis_client)
    
    # Store in app context
    app.security_managers = {
        'rate_limiter': rate_limiter,
        'account_security': account_security,
        'jwt_security': jwt_security,
        'ids': ids
    }
    
    return {
        'rate_limiter': rate_limiter,
        'account_security': account_security,
        'jwt_security': jwt_security,
        'ids': ids
    }