#!/usr/bin/env python3
"""
Pytest configuration and fixtures
"""
import pytest
import os
import tempfile
from app import create_app
from config import TestingConfig

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Create a temporary file to serve as a test database
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app('testing')
    app.config['DATABASE_URL'] = f'sqlite:///{db_path}'
    app.config['TESTING'] = True
    
    with app.app_context():
        # Initialize the test database
        yield app
    
    # Clean up
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

@pytest.fixture
def sample_tickers():
    """Sample ticker symbols for testing"""
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'strategy': 'sharpe',
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'investment_amount': 10000
    }

@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for testing"""
    return {
        'ticker': 'AAPL',
        'days': 10,
        'models': 'prophet'
    }