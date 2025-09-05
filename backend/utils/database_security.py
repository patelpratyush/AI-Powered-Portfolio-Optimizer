#!/usr/bin/env python3
"""
Database Security Utilities
Secure database operations with SQL injection prevention
"""

import logging
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import text, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseSecurityError(Exception):
    """Custom exception for database security violations"""
    pass

class SecureQueryBuilder:
    """
    Secure query builder that prevents SQL injection attacks
    """
    
    @staticmethod
    def validate_column_name(column_name: str) -> bool:
        """
        Validate column name to prevent SQL injection
        Only allows alphanumeric characters, underscores, and dots
        """
        if not isinstance(column_name, str):
            return False
        
        # Allow only safe characters for column names
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\.]*$', column_name):
            return False
        
        # Prevent dangerous keywords
        dangerous_keywords = [
            'drop', 'delete', 'truncate', 'alter', 'create',
            'insert', 'update', 'exec', 'execute', 'union',
            'script', '--', '/*', '*/', ';'
        ]
        
        column_lower = column_name.lower()
        for keyword in dangerous_keywords:
            if keyword in column_lower:
                return False
        
        return True
    
    @staticmethod
    def validate_table_name(table_name: str) -> bool:
        """
        Validate table name to prevent SQL injection
        """
        if not isinstance(table_name, str):
            return False
        
        # Allow only alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
            return False
        
        # Prevent dangerous keywords
        dangerous_keywords = [
            'information_schema', 'sys', 'mysql', 'pg_',
            'sqlite_master', 'sqlite_temp_master'
        ]
        
        table_lower = table_name.lower()
        for keyword in dangerous_keywords:
            if keyword in table_lower:
                return False
        
        return True
    
    @staticmethod
    def sanitize_order_by(order_by: str) -> str:
        """
        Sanitize ORDER BY clause to prevent injection
        """
        if not order_by:
            return ""
        
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w\s,.]', '', order_by)
        
        # Validate that it only contains safe patterns
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\.]*(\s+(ASC|DESC))?(\s*,\s*[a-zA-Z][a-zA-Z0-9_\.]*(\s+(ASC|DESC))?)*$', sanitized.strip(), re.IGNORECASE):
            raise DatabaseSecurityError(f"Invalid ORDER BY clause: {order_by}")
        
        return sanitized
    
    @staticmethod
    def build_where_clause(filters: Dict[str, Any]) -> tuple:
        """
        Build parameterized WHERE clause from filters dictionary
        
        Args:
            filters: Dictionary of column_name: value pairs
        
        Returns:
            Tuple of (where_clause, parameters)
        """
        if not filters:
            return "", {}
        
        conditions = []
        parameters = {}
        
        for i, (column, value) in enumerate(filters.items()):
            # Validate column name
            if not SecureQueryBuilder.validate_column_name(column):
                raise DatabaseSecurityError(f"Invalid column name: {column}")
            
            param_name = f"param_{i}"
            
            if value is None:
                conditions.append(f"{column} IS NULL")
            elif isinstance(value, (list, tuple)):
                # Handle IN clause
                in_params = []
                for j, item in enumerate(value):
                    in_param = f"{param_name}_{j}"
                    parameters[in_param] = item
                    in_params.append(f":{in_param}")
                conditions.append(f"{column} IN ({', '.join(in_params)})")
            else:
                conditions.append(f"{column} = :{param_name}")
                parameters[param_name] = value
        
        where_clause = " AND ".join(conditions)
        return where_clause, parameters

class SecureRepository:
    """
    Base repository class with secure database operations
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    def safe_execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Safely execute a parameterized query
        
        Args:
            query: SQL query with named parameters
            parameters: Dictionary of parameter values
        
        Returns:
            List of result dictionaries
        """
        try:
            # Validate the query doesn't contain dangerous patterns
            self._validate_query(query)
            
            # Execute with parameters
            result = self.session.execute(text(query), parameters or {})
            
            # Convert to list of dictionaries
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            else:
                return []
                
        except SQLAlchemyError as e:
            logger.error(f"Database query error: {str(e)}")
            self.session.rollback()
            raise DatabaseSecurityError(f"Query execution failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in safe_execute_query: {str(e)}")
            self.session.rollback()
            raise DatabaseSecurityError(f"Unexpected database error: {str(e)}")
    
    def safe_select(
        self,
        table_name: str,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> List[Dict[str, Any]]:
        """
        Safely execute a SELECT query
        
        Args:
            table_name: Name of the table
            columns: List of column names to select
            filters: WHERE clause filters
            order_by: ORDER BY clause
            limit: LIMIT value
            offset: OFFSET value
        
        Returns:
            List of result dictionaries
        """
        # Validate table name
        if not SecureQueryBuilder.validate_table_name(table_name):
            raise DatabaseSecurityError(f"Invalid table name: {table_name}")
        
        # Validate column names
        if columns:
            for column in columns:
                if not SecureQueryBuilder.validate_column_name(column):
                    raise DatabaseSecurityError(f"Invalid column name: {column}")
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        # Build query
        query = f"SELECT {select_clause} FROM {table_name}"
        parameters = {}
        
        # Add WHERE clause
        if filters:
            where_clause, where_params = SecureQueryBuilder.build_where_clause(filters)
            query += f" WHERE {where_clause}"
            parameters.update(where_params)
        
        # Add ORDER BY clause
        if order_by:
            sanitized_order = SecureQueryBuilder.sanitize_order_by(order_by)
            query += f" ORDER BY {sanitized_order}"
        
        # Add LIMIT and OFFSET
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise DatabaseSecurityError("LIMIT must be a non-negative integer")
            query += f" LIMIT {limit}"
        
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise DatabaseSecurityError("OFFSET must be a non-negative integer")
            query += f" OFFSET {offset}"
        
        return self.safe_execute_query(query, parameters)
    
    def safe_insert(
        self,
        table_name: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Safely execute an INSERT query
        
        Args:
            table_name: Name of the table
            data: Dictionary of column_name: value pairs
        
        Returns:
            Number of affected rows
        """
        if not SecureQueryBuilder.validate_table_name(table_name):
            raise DatabaseSecurityError(f"Invalid table name: {table_name}")
        
        if not data:
            raise DatabaseSecurityError("Insert data cannot be empty")
        
        # Validate column names
        for column in data.keys():
            if not SecureQueryBuilder.validate_column_name(column):
                raise DatabaseSecurityError(f"Invalid column name: {column}")
        
        # Build INSERT query
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{key}" for key in data.keys()])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        try:
            result = self.session.execute(text(query), data)
            return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"Database insert error: {str(e)}")
            self.session.rollback()
            raise DatabaseSecurityError(f"Insert operation failed: {str(e)}")
    
    def safe_update(
        self,
        table_name: str,
        data: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> int:
        """
        Safely execute an UPDATE query
        
        Args:
            table_name: Name of the table
            data: Dictionary of column_name: new_value pairs
            filters: WHERE clause filters
        
        Returns:
            Number of affected rows
        """
        if not SecureQueryBuilder.validate_table_name(table_name):
            raise DatabaseSecurityError(f"Invalid table name: {table_name}")
        
        if not data:
            raise DatabaseSecurityError("Update data cannot be empty")
        
        if not filters:
            raise DatabaseSecurityError("UPDATE queries must have WHERE clause for safety")
        
        # Validate column names
        for column in data.keys():
            if not SecureQueryBuilder.validate_column_name(column):
                raise DatabaseSecurityError(f"Invalid column name: {column}")
        
        # Build SET clause
        set_clauses = []
        parameters = {}
        
        for i, (column, value) in enumerate(data.items()):
            param_name = f"set_param_{i}"
            set_clauses.append(f"{column} = :{param_name}")
            parameters[param_name] = value
        
        # Build WHERE clause
        where_clause, where_params = SecureQueryBuilder.build_where_clause(filters)
        parameters.update(where_params)
        
        # Build query
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"
        
        try:
            result = self.session.execute(text(query), parameters)
            return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"Database update error: {str(e)}")
            self.session.rollback()
            raise DatabaseSecurityError(f"Update operation failed: {str(e)}")
    
    def safe_delete(
        self,
        table_name: str,
        filters: Dict[str, Any]
    ) -> int:
        """
        Safely execute a DELETE query
        
        Args:
            table_name: Name of the table
            filters: WHERE clause filters
        
        Returns:
            Number of affected rows
        """
        if not SecureQueryBuilder.validate_table_name(table_name):
            raise DatabaseSecurityError(f"Invalid table name: {table_name}")
        
        if not filters:
            raise DatabaseSecurityError("DELETE queries must have WHERE clause for safety")
        
        # Build WHERE clause
        where_clause, parameters = SecureQueryBuilder.build_where_clause(filters)
        
        # Build query
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        
        try:
            result = self.session.execute(text(query), parameters)
            return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"Database delete error: {str(e)}")
            self.session.rollback()
            raise DatabaseSecurityError(f"Delete operation failed: {str(e)}")
    
    def _validate_query(self, query: str) -> None:
        """
        Validate query for dangerous patterns
        
        Args:
            query: SQL query to validate
        
        Raises:
            DatabaseSecurityError: If dangerous patterns are found
        """
        if not isinstance(query, str):
            raise DatabaseSecurityError("Query must be a string")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r';\s*drop\s+',
            r';\s*delete\s+',
            r';\s*truncate\s+',
            r';\s*alter\s+',
            r';\s*create\s+',
            r'union\s+select',
            r'@@version',
            r'information_schema',
            r'pg_sleep',
            r'waitfor\s+delay',
            r'benchmark\s*\(',
            r'sleep\s*\(',
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                raise DatabaseSecurityError(f"Query contains dangerous pattern: {pattern}")
        
        # Check for excessive comment usage
        if query.count('--') > 1 or query.count('/*') > 1:
            raise DatabaseSecurityError("Excessive comment usage detected")
        
        # Check for suspicious semicolon usage
        if query.count(';') > 1:
            raise DatabaseSecurityError("Multiple statements not allowed")

@contextmanager
def secure_transaction(session: Session):
    """
    Context manager for secure database transactions
    
    Args:
        session: SQLAlchemy session
    
    Yields:
        SecureRepository instance
    """
    repo = SecureRepository(session)
    try:
        yield repo
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction rolled back due to error: {str(e)}")
        raise
    finally:
        session.close()

def create_secure_connection_string(
    host: str,
    database: str,
    username: str,
    password: str,
    port: int = 5432,
    driver: str = "postgresql"
) -> str:
    """
    Create a secure database connection string with validation
    
    Args:
        host: Database host
        database: Database name
        username: Username
        password: Password
        port: Database port
        driver: Database driver
    
    Returns:
        Secure connection string
    """
    # Validate inputs
    if not all([host, database, username, password]):
        raise DatabaseSecurityError("All connection parameters must be provided")
    
    # Validate host format
    if not re.match(r'^[a-zA-Z0-9.-]+$', host):
        raise DatabaseSecurityError("Invalid host format")
    
    # Validate database name
    if not re.match(r'^[a-zA-Z0-9_-]+$', database):
        raise DatabaseSecurityError("Invalid database name")
    
    # Validate username
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise DatabaseSecurityError("Invalid username")
    
    # Validate port
    if not (1 <= port <= 65535):
        raise DatabaseSecurityError("Port must be between 1 and 65535")
    
    # URL-encode password to handle special characters
    from urllib.parse import quote_plus
    encoded_password = quote_plus(password)
    
    return f"{driver}://{username}:{encoded_password}@{host}:{port}/{database}"

# Example usage and utility functions
def get_secure_repository(session: Session) -> SecureRepository:
    """
    Get a secure repository instance
    
    Args:
        session: SQLAlchemy session
    
    Returns:
        SecureRepository instance
    """
    return SecureRepository(session)