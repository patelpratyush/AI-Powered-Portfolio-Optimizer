#!/usr/bin/env python3
"""
User Authentication Routes
JWT-based authentication with registration, login, and user management
"""
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token, 
    create_refresh_token, get_jwt_identity, get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import User, get_db
from schemas import *
from utils.security import secure_api_endpoint
from utils.error_handlers import *
import re
import logging

logger = logging.getLogger(__name__)

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

# JWT blacklist for token revocation
blacklisted_tokens = set()

def init_jwt(app):
    """Initialize JWT extension"""
    app.config['JWT_SECRET_KEY'] = app.config.get('JWT_SECRET_KEY')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(
        seconds=app.config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)
    )
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    
    jwt = JWTManager(app)
    
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        return jwt_payload['jti'] in blacklisted_tokens
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'TokenExpired', 'message': 'Token has expired'}), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'InvalidToken', 'message': 'Invalid token'}), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({'error': 'AuthorizationRequired', 'message': 'Access token required'}), 401
    
    return jwt

# Validation schemas
class UserRegistrationRequest(BaseModel):
    email: str = Field(..., regex=r'^[^\s@]+@[^\s@]+\.[^\s@]+$', description="Valid email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    preferred_currency: str = Field("USD", regex=r'^[A-Z]{3}$')
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Risk tolerance (0-1)")
    investment_horizon: str = Field("medium", regex=r'^(short|medium|long)$')
    
    @validator('password')
    def validate_password_strength(cls, v):
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v

class UserLoginRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")

class UserUpdateRequest(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    preferred_currency: Optional[str] = Field(None, regex=r'^[A-Z]{3}$')
    risk_tolerance: Optional[float] = Field(None, ge=0, le=1)
    investment_horizon: Optional[str] = Field(None, regex=r'^(short|medium|long)$')

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v

# Authentication utilities
def hash_password(password: str) -> str:
    """Hash password using Werkzeug"""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

def verify_password(password_hash: str, password: str) -> bool:
    """Verify password against hash"""
    return check_password_hash(password_hash, password)

def create_user_tokens(user: User) -> Dict[str, str]:
    """Create access and refresh tokens for user"""
    additional_claims = {
        "user_id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name
    }
    
    access_token = create_access_token(
        identity=user.id,
        additional_claims=additional_claims
    )
    refresh_token = create_refresh_token(identity=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds()
    }

# Routes
@auth_bp.route('/register', methods=['POST'])
@safe_api_call
def register():
    """Register new user"""
    try:
        # Validate request data
        request_data = UserRegistrationRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid registration data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request_data.email.lower()).first()
        if existing_user:
            raise ValidationException("User with this email already exists")
        
        # Create new user
        new_user = User(
            email=request_data.email.lower(),
            password_hash=hash_password(request_data.password),
            first_name=request_data.first_name,
            last_name=request_data.last_name,
            preferred_currency=request_data.preferred_currency,
            risk_tolerance=request_data.risk_tolerance,
            investment_horizon=request_data.investment_horizon
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create tokens
        tokens = create_user_tokens(new_user)
        
        logger.info(f"New user registered: {new_user.email}")
        
        return jsonify({
            "message": "User registered successfully",
            "user": new_user.to_dict(),
            "tokens": tokens
        }), 201
        
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

@auth_bp.route('/login', methods=['POST'])
@safe_api_call
def login():
    """Authenticate user and return tokens"""
    try:
        request_data = UserLoginRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid login data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        # Find user
        user = db.query(User).filter(User.email == request_data.email.lower()).first()
        
        if not user or not verify_password(user.password_hash, request_data.password):
            raise ValidationException("Invalid email or password")
        
        if not user.is_active:
            raise ValidationException("Account is deactivated")
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        # Update last login
        user.updated_at = datetime.now()
        db.commit()
        
        logger.info(f"User logged in: {user.email}")
        
        return jsonify({
            "message": "Login successful",
            "user": user.to_dict(),
            "tokens": tokens
        })
        
    finally:
        db.close()

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
@safe_api_call
def refresh():
    """Refresh access token using refresh token"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        user = db.query(User).filter(User.id == current_user_id).first()
        
        if not user or not user.is_active:
            raise NotFoundError("User not found or inactive")
        
        # Create new access token
        additional_claims = {
            "user_id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name
        }
        
        new_access_token = create_access_token(
            identity=user.id,
            additional_claims=additional_claims
        )
        
        return jsonify({
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds()
        })
        
    finally:
        db.close()

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
@safe_api_call
def logout():
    """Logout user and blacklist token"""
    jti = get_jwt()['jti']
    blacklisted_tokens.add(jti)
    
    logger.info(f"User logged out: {get_jwt_identity()}")
    
    return jsonify({"message": "Successfully logged out"})

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
@safe_api_call
def get_current_user():
    """Get current user profile"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        user = db.query(User).filter(User.id == current_user_id).first()
        
        if not user:
            raise NotFoundError("User not found")
        
        return jsonify({"user": user.to_dict()})
        
    finally:
        db.close()

@auth_bp.route('/me', methods=['PUT'])
@jwt_required()
@safe_api_call
def update_current_user():
    """Update current user profile"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = UserUpdateRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid update data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        user = db.query(User).filter(User.id == current_user_id).first()
        
        if not user:
            raise NotFoundError("User not found")
        
        # Update fields
        update_data = request_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
        
        logger.info(f"User profile updated: {user.email}")
        
        return jsonify({
            "message": "Profile updated successfully",
            "user": user.to_dict()
        })
        
    finally:
        db.close()

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
@safe_api_call
def change_password():
    """Change user password"""
    current_user_id = get_jwt_identity()
    
    try:
        request_data = ChangePasswordRequest(**request.get_json())
    except ValidationError as e:
        raise ValidationException("Invalid password data", details=dict(e.errors()))
    
    db = next(get_db())
    
    try:
        user = db.query(User).filter(User.id == current_user_id).first()
        
        if not user:
            raise NotFoundError("User not found")
        
        # Verify current password
        if not verify_password(user.password_hash, request_data.current_password):
            raise ValidationException("Current password is incorrect")
        
        # Update password
        user.password_hash = hash_password(request_data.new_password)
        user.updated_at = datetime.now()
        db.commit()
        
        logger.info(f"Password changed for user: {user.email}")
        
        return jsonify({"message": "Password changed successfully"})
        
    finally:
        db.close()

@auth_bp.route('/deactivate', methods=['POST'])
@jwt_required()
@safe_api_call
def deactivate_account():
    """Deactivate user account"""
    current_user_id = get_jwt_identity()
    
    db = next(get_db())
    
    try:
        user = db.query(User).filter(User.id == current_user_id).first()
        
        if not user:
            raise NotFoundError("User not found")
        
        user.is_active = False
        user.updated_at = datetime.now()
        db.commit()
        
        # Blacklist current token
        jti = get_jwt()['jti']
        blacklisted_tokens.add(jti)
        
        logger.info(f"Account deactivated: {user.email}")
        
        return jsonify({"message": "Account deactivated successfully"})
        
    finally:
        db.close()

# Utility endpoint for frontend
@auth_bp.route('/check-email', methods=['POST'])
@safe_api_call
def check_email_availability():
    """Check if email is available for registration"""
    data = request.get_json()
    email = data.get('email', '').lower().strip()
    
    if not email or '@' not in email:
        raise ValidationException("Invalid email format")
    
    db = next(get_db())
    
    try:
        existing_user = db.query(User).filter(User.email == email).first()
        
        return jsonify({
            "email": email,
            "available": existing_user is None
        })
        
    finally:
        db.close()

# Admin routes (for future use)
@auth_bp.route('/admin/users', methods=['GET'])
@secure_api_endpoint(
    require_auth=True,
    admin_only=True,
    rate_limit="10 per minute"
)
def list_users():
    """List all users (admin only)"""
    current_user_id = get_jwt_identity()
    
    # This endpoint is now properly protected by the security decorator
    # Only admin users can access this endpoint
    raise ValidationException("Admin functionality not yet implemented")

if __name__ == "__main__":
    # Test authentication functions
    test_password = "testpassword123"
    hashed = hash_password(test_password)
    print(f"Password hash: {hashed}")
    print(f"Verification: {verify_password(hashed, test_password)}")
    print(f"Wrong password: {verify_password(hashed, 'wrongpassword')}")