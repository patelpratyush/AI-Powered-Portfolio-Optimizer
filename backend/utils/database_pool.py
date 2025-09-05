#!/usr/bin/env python3
"""
Database Connection Pooling
High-performance database connection management with pooling
"""

import os
import logging
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import DisconnectionError, TimeoutError
import time

logger = logging.getLogger(__name__)

class DatabasePool:
    """
    High-performance database connection pool manager
    """
    
    def __init__(self, 
                 database_url: str,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600,
                 pool_pre_ping: bool = True):
        """
        Initialize database connection pool
        
        Args:
            database_url: Database connection URL
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of additional connections
            pool_timeout: Seconds to wait for connection from pool
            pool_recycle: Seconds before recreating a connection
            pool_pre_ping: Verify connections before use
        """
        self.database_url = database_url
        self._lock = threading.RLock()
        
        # Configure connection pool
        pool_config = {
            'poolclass': QueuePool,
            'pool_size': pool_size,
            'max_overflow': max_overflow,
            'pool_timeout': pool_timeout,
            'pool_recycle': pool_recycle,
            'pool_pre_ping': pool_pre_ping,
            'echo': os.environ.get('SQL_ECHO', 'false').lower() == 'true'
        }
        
        # Special handling for SQLite (in-memory or development)
        if database_url.startswith('sqlite'):
            pool_config.update({
                'poolclass': StaticPool,
                'pool_size': 1,
                'max_overflow': 0,
                'connect_args': {'check_same_thread': False}
            })
        
        # PostgreSQL optimizations
        elif database_url.startswith('postgresql'):
            pool_config.update({
                'connect_args': {
                    'connect_timeout': 10,
                    'application_name': 'portfolio_optimizer',
                    'options': '-c default_transaction_isolation=read_committed'
                }
            })
        
        try:
            self.engine = create_engine(database_url, **pool_config)
            self._setup_engine_events()
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(self.session_factory)
            
            # Test connection
            self._test_connection()
            
            logger.info(f"Database pool initialized: {pool_size} connections, {max_overflow} overflow")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    def _setup_engine_events(self):
        """Setup SQLAlchemy engine events for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Handle new database connections"""
            logger.debug("New database connection established")
            
            # PostgreSQL session optimizations
            if self.database_url.startswith('postgresql'):
                with dbapi_connection.cursor() as cursor:
                    # Optimize for read-heavy workloads
                    cursor.execute("SET default_transaction_isolation = 'read committed'")
                    cursor.execute("SET statement_timeout = '30s'")
                    cursor.execute("SET lock_timeout = '10s'")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool"""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Handle connection return to pool"""
            logger.debug("Connection returned to pool")
        
        @event.listens_for(self.engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation"""
            logger.warning(f"Connection invalidated: {exception}")
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        
        Yields:
            SQLAlchemy session with automatic transaction management
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_scoped_session(self):
        """
        Get thread-local scoped session
        
        Returns:
            Scoped session instance
        """
        return self.Session()
    
    def remove_scoped_session(self):
        """Remove current thread-local session"""
        self.Session.remove()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status
        
        Returns:
            Dictionary with pool statistics
        """
        pool = self.engine.pool
        
        return {
            'pool_size': pool.size(),
            'checked_in_connections': pool.checkedin(),
            'checked_out_connections': pool.checkedout(),
            'overflow_connections': pool.overflow(),
            'invalid_connections': pool.invalid(),
            'total_connections': pool.size() + pool.overflow(),
            'pool_timeout': pool._timeout,
            'pool_recycle': pool._recycle,
            'connection_info': {
                'url': str(self.engine.url).replace(self.engine.url.password or '', '***'),
                'driver': self.engine.driver,
                'dialect': str(self.engine.dialect.name)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check
        
        Returns:
            Health check results
        """
        start_time = time.time()
        
        try:
            with self.get_session() as session:
                # Simple query to test connection
                session.execute("SELECT 1")
                
            response_time = (time.time() - start_time) * 1000
            pool_status = self.get_pool_status()
            
            # Determine health status
            is_healthy = (
                response_time < 1000 and  # Response time < 1s
                pool_status['checked_out_connections'] < pool_status['pool_size'] * 0.8  # < 80% utilization
            )
            
            return {
                'status': 'healthy' if is_healthy else 'warning',
                'response_time_ms': response_time,
                'pool_status': pool_status,
                'warnings': self._get_health_warnings(pool_status, response_time),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000,
                'timestamp': time.time()
            }
    
    def _get_health_warnings(self, pool_status: Dict[str, Any], response_time: float) -> list:
        """Generate health warnings based on pool status"""
        warnings = []
        
        if response_time > 500:
            warnings.append(f"High database response time: {response_time:.1f}ms")
        
        utilization = pool_status['checked_out_connections'] / pool_status['pool_size']
        if utilization > 0.8:
            warnings.append(f"High pool utilization: {utilization:.1%}")
        
        if pool_status['invalid_connections'] > 0:
            warnings.append(f"Invalid connections detected: {pool_status['invalid_connections']}")
        
        if pool_status['overflow_connections'] > pool_status['pool_size'] * 0.5:
            warnings.append("High overflow connection usage")
        
        return warnings
    
    def close(self):
        """Close all connections in the pool"""
        try:
            self.Session.remove()
            self.engine.dispose()
            logger.info("Database pool closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")

# Global pool instance
_db_pool = None
_pool_lock = threading.Lock()

def initialize_database_pool(database_url: str, **kwargs) -> DatabasePool:
    """
    Initialize global database pool
    
    Args:
        database_url: Database connection URL
        **kwargs: Additional pool configuration
    
    Returns:
        DatabasePool instance
    """
    global _db_pool
    
    with _pool_lock:
        if _db_pool is not None:
            _db_pool.close()
        
        _db_pool = DatabasePool(database_url, **kwargs)
        return _db_pool

def get_database_pool() -> DatabasePool:
    """
    Get global database pool instance
    
    Returns:
        DatabasePool instance
    
    Raises:
        RuntimeError: If pool not initialized
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized. Call initialize_database_pool() first.")
    
    return _db_pool

@contextmanager
def get_db_session():
    """
    Context manager for database sessions using global pool
    
    Yields:
        SQLAlchemy session
    """
    pool = get_database_pool()
    with pool.get_session() as session:
        yield session

def get_scoped_db_session():
    """
    Get thread-local scoped session from global pool
    
    Returns:
        Scoped session instance
    """
    pool = get_database_pool()
    return pool.get_scoped_session()

# Performance monitoring decorator
def monitor_db_performance(operation_name: str):
    """
    Decorator to monitor database operation performance
    
    Args:
        operation_name: Name of the database operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > 1000:  # Log slow queries
                    logger.warning(f"Slow database operation: {operation_name} took {execution_time:.1f}ms")
                else:
                    logger.debug(f"Database operation: {operation_name} took {execution_time:.1f}ms")
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Database operation failed: {operation_name} failed after {execution_time:.1f}ms - {e}")
                raise
        
        return wrapper
    return decorator

# Connection pool configuration for different environments
def get_pool_config(environment: str) -> Dict[str, Any]:
    """
    Get connection pool configuration for environment
    
    Args:
        environment: Environment name (development, testing, production)
    
    Returns:
        Pool configuration dictionary
    """
    configs = {
        'development': {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600
        },
        'testing': {
            'pool_size': 2,
            'max_overflow': 5,
            'pool_timeout': 10,
            'pool_recycle': -1  # No recycling for tests
        },
        'production': {
            'pool_size': 20,
            'max_overflow': 30,
            'pool_timeout': 30,
            'pool_recycle': 1800  # 30 minutes
        }
    }
    
    return configs.get(environment, configs['development'])