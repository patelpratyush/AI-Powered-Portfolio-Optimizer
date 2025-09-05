import os
import logging
import redis
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager
from config import config
from security_config import apply_security_middleware, create_rate_limiter, check_environment_security
from utils.production_config import setup_production_features
from utils.monitoring import setup_monitoring
from utils.error_handlers import setup_error_handlers
from utils.logging_config import setup_logging
from routes.optimize import optimize_bp
from routes.autocomplete import autocomplete_bp
from routes.advanced_optimize import advanced_optimize_bp
from routes.import_portfolio import import_bp
from routes.predict import predict_bp
from routes.auth import auth_bp, init_jwt
from routes.websocket import init_socketio
from routes.analytics import analytics_bp
from routes.notifications import notifications_bp, init_email_service
from routes.dashboard import dashboard_bp
from routes.sentiment import sentiment_bp
from routes.backtesting import backtesting_bp
from routes.risk import risk_bp
from routes.alternatives import alternatives_bp
from models.database import init_database

def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Check for security misconfigurations
    security_warnings = check_environment_security()
    if security_warnings:
        for warning in security_warnings:
            app.logger.warning(f"SECURITY WARNING: {warning}")
    
    # Apply comprehensive security middleware
    app = apply_security_middleware(app)
    
    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize JWT
    init_jwt(app)
    
    # Initialize rate limiter
    limiter = create_rate_limiter(app)
    
    # Initialize database
    init_database(app.config.get('DATABASE_URL'))
    
    # Initialize Redis client for production features
    redis_client = None
    if app.config.get('REDIS_URL'):
        try:
            redis_client = redis.from_url(app.config['REDIS_URL'])
            redis_client.ping()  # Test connection
        except Exception as e:
            app.logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    # Setup production features
    production_components = setup_production_features(app, redis_client)
    
    # Setup monitoring and health checks
    health_monitor = setup_monitoring(app)
    
    # Setup enhanced error handling
    error_tracker = setup_error_handlers(app, redis_client)
    
    # Configure advanced logging with correlation IDs
    loggers = setup_logging(app.config)
    
    # Store components in app context for easy access
    app.loggers = loggers
    app.production_components = production_components
    app.health_monitor = health_monitor
    app.error_tracker = error_tracker
    app.redis_client = redis_client
    
    # Wrap WSGI app with request logging middleware
    from utils.logging_config import RequestLoggingMiddleware
    app.wsgi_app = RequestLoggingMiddleware(app.wsgi_app, loggers['api'])
    
    # Set application logger level
    app.logger.setLevel(logging.INFO)
    app.logger.info('Portfolio Optimizer startup with production features')
    
    # Initialize WebSocket
    socketio = init_socketio(app)
    
    # Initialize email service
    init_email_service(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(notifications_bp, url_prefix="/api/notifications")
    app.register_blueprint(dashboard_bp, url_prefix="/api")
    app.register_blueprint(optimize_bp, url_prefix="/api")
    app.register_blueprint(autocomplete_bp, url_prefix="/api")
    app.register_blueprint(advanced_optimize_bp, url_prefix="/api")
    app.register_blueprint(import_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(sentiment_bp, url_prefix="/api")
    app.register_blueprint(backtesting_bp, url_prefix="/api")
    app.register_blueprint(risk_bp, url_prefix="/api")
    app.register_blueprint(alternatives_bp, url_prefix="/api")
    
    # Basic health check endpoint (detailed ones are added by monitoring setup)
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'environment': config_name
        })
    
    return app, socketio

app, socketio = create_app()

if __name__ == "__main__":
    config_name = os.environ.get('FLASK_ENV', 'development')
    app_config = config[config_name]
    
    # Use SocketIO's run method instead of Flask's
    socketio.run(
        app,
        host=app_config.API_HOST,
        port=app_config.API_PORT,
        debug=app_config.DEBUG
    )
