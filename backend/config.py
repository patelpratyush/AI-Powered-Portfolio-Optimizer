#!/usr/bin/env python3
"""
Configuration management for Flask application
"""
import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required for security")
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///portfolio_optimizer.db'
    
    # API Configuration
    API_HOST = os.environ.get('API_HOST') or '0.0.0.0'
    API_PORT = int(os.environ.get('API_PORT') or 5000)
    
    # ML Configuration
    MODEL_DIR = os.environ.get('MODEL_DIR') or 'models/saved'
    TRAINING_TIMEOUT = int(os.environ.get('TRAINING_TIMEOUT') or 1800)
    MAX_TRAINING_WORKERS = int(os.environ.get('MAX_TRAINING_WORKERS') or 3)
    
    # External APIs
    YAHOO_FINANCE_TIMEOUT = int(os.environ.get('YAHOO_FINANCE_TIMEOUT') or 10)
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE') or 100)
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'logs/app.log'
    LOG_DIR = os.environ.get('LOG_DIR') or 'logs'
    LOG_TO_CONSOLE = os.environ.get('LOG_TO_CONSOLE', 'True').lower() == 'true'
    
    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY environment variable is required for security")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES') or 3600)
    
    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'
    
    # Override secrets for development environment only
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'dev-jwt-secret-change-in-production'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'
    REDIS_URL = 'redis://localhost:6379/1'
    
    # Override secrets for testing environment only
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'test-secret-key-for-testing'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'test-jwt-secret-for-testing'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Override with more secure defaults for production
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE') or 60)

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}