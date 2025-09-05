#!/usr/bin/env python3
"""
Portfolio Backtesting API Routes
"""
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from utils.backtesting import (
    PortfolioBacktester, BacktestConfig, 
    EqualWeightStrategy, MarketCapWeightStrategy, MeanReversionStrategy,
    MomentumStrategy, MinVarianceStrategy, MaxSharpeStrategy,
    create_strategy_from_config, quick_backtest
)
from utils.error_handlers import safe_api_call
from models.database import get_user_by_id

# Create blueprint
backtesting_bp = Blueprint('backtesting', __name__)
logger = logging.getLogger('portfolio_optimizer.routes.backtesting')

@backtesting_bp.route('/backtest/single', methods=['POST'])
@jwt_required()
@safe_api_call
def backtest_single_strategy():
    """Run backtest for a single strategy"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['tickers', 'start_date', 'end_date', 'strategy']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Parse and validate inputs
        tickers = [t.upper().strip() for t in data['tickers']]
        if not tickers or len(tickers) > 20:
            return jsonify({'error': 'Must provide 1-20 tickers'}), 400
        
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        if start_date >= end_date:
            return jsonify({'error': 'Start date must be before end date'}), 400
        
        if (end_date - start_date).days < 30:
            return jsonify({'error': 'Minimum backtest period is 30 days'}), 400
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=data.get('initial_capital', 100000),
            rebalance_frequency=data.get('rebalance_frequency', 'monthly'),
            transaction_cost=data.get('transaction_cost', 0.001),
            benchmark=data.get('benchmark', 'SPY'),
            risk_free_rate=data.get('risk_free_rate', 0.02)
        )
        
        # Create strategy
        strategy = create_strategy_from_config(data['strategy'])
        
        # Run backtest
        logger.info(f"Running backtest for {strategy.name} with {len(tickers)} tickers (user: {user_id})")
        backtester = PortfolioBacktester()
        result = backtester.backtest_strategy(strategy, tickers, config)
        
        # Format response
        response = {
            'strategy_name': strategy.name,
            'config': {
                'tickers': tickers,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'initial_capital': config.initial_capital,
                'rebalance_frequency': config.rebalance_frequency,
                'transaction_cost': config.transaction_cost,
                'benchmark': config.benchmark
            },
            'performance': {
                'total_return': result.metrics.total_return,
                'annualized_return': result.metrics.annualized_return,
                'volatility': result.metrics.volatility,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown,
                'calmar_ratio': result.metrics.calmar_ratio,
                'sortino_ratio': result.metrics.sortino_ratio,
                'beta': result.metrics.beta,
                'alpha': result.metrics.alpha,
                'information_ratio': result.metrics.information_ratio,
                'win_rate': result.metrics.win_rate,
                'profit_factor': result.metrics.profit_factor,
                'value_at_risk_95': result.metrics.value_at_risk_95,
                'conditional_var_95': result.metrics.conditional_var_95
            },
            'benchmark_performance': {
                'total_return': result.benchmark_metrics.total_return,
                'annualized_return': result.benchmark_metrics.annualized_return,
                'volatility': result.benchmark_metrics.volatility,
                'sharpe_ratio': result.benchmark_metrics.sharpe_ratio,
                'max_drawdown': result.benchmark_metrics.max_drawdown
            },
            'portfolio_values': [
                {'date': date.isoformat(), 'value': float(value)}
                for date, value in result.portfolio_values.items()
            ],
            'monthly_returns': [
                {'month': date.strftime('%Y-%m'), 'return': float(ret)}
                for date, ret in result.monthly_returns.items()
            ],
            'yearly_returns': [
                {'year': date.year, 'return': float(ret)}
                for date, ret in result.yearly_returns.items()
            ],
            'drawdown_periods': result.drawdown_periods,
            'trade_count': len(result.trades),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Backtest completed. Total return: {result.metrics.total_return:.2%}, Sharpe: {result.metrics.sharpe_ratio:.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Single strategy backtest error: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@backtesting_bp.route('/backtest/compare', methods=['POST'])
@jwt_required()
@safe_api_call
def compare_strategies():
    """Compare multiple strategies"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['tickers', 'start_date', 'end_date', 'strategies']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Parse inputs
        tickers = [t.upper().strip() for t in data['tickers']]
        if not tickers or len(tickers) > 15:
            return jsonify({'error': 'Must provide 1-15 tickers for comparison'}), 400
        
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        strategies_config = data['strategies']
        if not strategies_config or len(strategies_config) > 6:
            return jsonify({'error': 'Must provide 1-6 strategies to compare'}), 400
        
        # Create strategies
        strategies = []
        for strategy_config in strategies_config:
            try:
                strategy = create_strategy_from_config(strategy_config)
                strategies.append(strategy)
            except Exception as e:
                return jsonify({'error': f'Invalid strategy configuration: {e}'}), 400
        
        # Create configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=data.get('initial_capital', 100000),
            rebalance_frequency=data.get('rebalance_frequency', 'monthly'),
            transaction_cost=data.get('transaction_cost', 0.001),
            benchmark=data.get('benchmark', 'SPY'),
            risk_free_rate=data.get('risk_free_rate', 0.02)
        )
        
        # Run comparison
        logger.info(f"Comparing {len(strategies)} strategies with {len(tickers)} tickers (user: {user_id})")
        backtester = PortfolioBacktester()
        results = backtester.compare_strategies(strategies, tickers, config)
        
        # Format response
        comparison_data = {}
        summary_metrics = []
        
        for strategy_name, result in results.items():
            if result is None:
                comparison_data[strategy_name] = {'error': 'Backtest failed'}
                continue
            
            comparison_data[strategy_name] = {
                'performance': {
                    'total_return': result.metrics.total_return,
                    'annualized_return': result.metrics.annualized_return,
                    'volatility': result.metrics.volatility,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'max_drawdown': result.metrics.max_drawdown,
                    'calmar_ratio': result.metrics.calmar_ratio,
                    'sortino_ratio': result.metrics.sortino_ratio,
                    'win_rate': result.metrics.win_rate,
                    'profit_factor': result.metrics.profit_factor
                },
                'portfolio_values': [
                    {'date': date.isoformat(), 'value': float(value)}
                    for date, value in result.portfolio_values.items()
                ],
                'drawdown_periods': len(result.drawdown_periods),
                'trade_count': len(result.trades)
            }
            
            summary_metrics.append({
                'strategy': strategy_name,
                'total_return': result.metrics.total_return,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown,
                'volatility': result.metrics.volatility
            })
        
        # Rank strategies by Sharpe ratio
        summary_metrics.sort(key=lambda x: x.get('sharpe_ratio', -999), reverse=True)
        
        response = {
            'comparison_results': comparison_data,
            'summary': summary_metrics,
            'config': {
                'tickers': tickers,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'strategies_compared': len(strategies),
                'period_days': (end_date - start_date).days
            },
            'generated_at': datetime.now().isoformat()
        }
        
        best_strategy = summary_metrics[0] if summary_metrics else None
        if best_strategy:
            logger.info(f"Best strategy: {best_strategy['strategy']} (Sharpe: {best_strategy['sharpe_ratio']:.2f})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Strategy comparison error: {e}")
        return jsonify({'error': f'Strategy comparison failed: {str(e)}'}), 500

@backtesting_bp.route('/backtest/quick', methods=['POST'])
@safe_api_call
def quick_backtest_endpoint():
    """Quick backtest with minimal configuration"""
    try:
        data = request.get_json()
        
        # Basic validation
        if not data.get('tickers'):
            return jsonify({'error': 'Tickers are required'}), 400
        
        tickers = [t.upper().strip() for t in data['tickers']]
        if len(tickers) > 10:
            return jsonify({'error': 'Maximum 10 tickers for quick backtest'}), 400
        
        # Default parameters for quick backtest
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        strategy_type = data.get('strategy', 'equal_weight')
        rebalance_frequency = data.get('rebalance_frequency', 'monthly')
        
        # Run quick backtest
        logger.info(f"Running quick backtest for {len(tickers)} tickers")
        result = quick_backtest(tickers, start_date, end_date, strategy_type, rebalance_frequency)
        
        # Simplified response
        response = {
            'strategy': strategy_type.replace('_', ' ').title(),
            'period': f"{start_date} to {end_date}",
            'tickers': tickers,
            'summary': {
                'total_return': f"{result.metrics.total_return:.2%}",
                'annualized_return': f"{result.metrics.annualized_return:.2%}",
                'volatility': f"{result.metrics.volatility:.2%}",
                'sharpe_ratio': f"{result.metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{result.metrics.max_drawdown:.2%}",
                'win_rate': f"{result.metrics.win_rate:.1%}"
            },
            'final_value': float(result.portfolio_values.iloc[-1]),
            'initial_value': float(result.portfolio_values.iloc[0]),
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Quick backtest error: {e}")
        return jsonify({'error': f'Quick backtest failed: {str(e)}'}), 500

@backtesting_bp.route('/backtest/strategies', methods=['GET'])
@safe_api_call
def get_available_strategies():
    """Get list of available backtesting strategies"""
    try:
        strategies = {
            'equal_weight': {
                'name': 'Equal Weight',
                'description': 'Allocates equal weight to all assets',
                'parameters': {},
                'suitable_for': 'Diversified portfolios, beginners',
                'risk_level': 'Medium'
            },
            'market_cap': {
                'name': 'Market Cap Weighted',
                'description': 'Weights assets by market capitalization',
                'parameters': {},
                'suitable_for': 'Index-like exposure',
                'risk_level': 'Medium'
            },
            'mean_reversion': {
                'name': 'Mean Reversion',
                'description': 'Overweights underperforming assets expecting reversion',
                'parameters': {
                    'lookback_period': {
                        'type': 'integer',
                        'default': 20,
                        'min': 5,
                        'max': 100,
                        'description': 'Days to calculate mean prices'
                    }
                },
                'suitable_for': 'Contrarian investors, range-bound markets',
                'risk_level': 'Medium-High'
            },
            'momentum': {
                'name': 'Momentum',
                'description': 'Overweights assets with positive momentum',
                'parameters': {
                    'lookback_period': {
                        'type': 'integer',
                        'default': 60,
                        'min': 10,
                        'max': 252,
                        'description': 'Days to calculate momentum'
                    }
                },
                'suitable_for': 'Trending markets, growth investors',
                'risk_level': 'High'
            },
            'min_variance': {
                'name': 'Minimum Variance',
                'description': 'Minimizes portfolio volatility',
                'parameters': {
                    'lookback_period': {
                        'type': 'integer',
                        'default': 252,
                        'min': 60,
                        'max': 500,
                        'description': 'Days for covariance calculation'
                    }
                },
                'suitable_for': 'Risk-averse investors, defensive strategies',
                'risk_level': 'Low'
            },
            'max_sharpe': {
                'name': 'Maximum Sharpe Ratio',
                'description': 'Maximizes risk-adjusted returns',
                'parameters': {
                    'lookback_period': {
                        'type': 'integer',
                        'default': 252,
                        'min': 60,
                        'max': 500,
                        'description': 'Days for optimization'
                    }
                },
                'suitable_for': 'Balanced risk-return profile',
                'risk_level': 'Medium'
            }
        }
        
        return jsonify({
            'strategies': strategies,
            'rebalance_frequencies': {
                'daily': 'Rebalance every trading day (high turnover)',
                'weekly': 'Rebalance every week',
                'monthly': 'Rebalance monthly (recommended)',
                'quarterly': 'Rebalance every quarter (low turnover)'
            },
            'common_benchmarks': ['SPY', 'QQQ', 'IWM', 'VTI', 'BND'],
            'notes': [
                'All strategies support portfolio constraints and transaction costs',
                'Minimum backtest period is 30 days',
                'Maximum 20 tickers per backtest for performance',
                'Historical performance does not guarantee future results'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        return jsonify({'error': 'Failed to get strategies'}), 500

@backtesting_bp.route('/backtest/presets', methods=['GET'])
@safe_api_call
def get_backtest_presets():
    """Get common backtesting presets"""
    try:
        presets = {
            'conservative': {
                'name': 'Conservative Portfolio',
                'description': 'Low-risk, defensive strategy',
                'strategy': {'type': 'min_variance', 'lookback_period': 252},
                'rebalance_frequency': 'quarterly',
                'transaction_cost': 0.001,
                'suitable_for': 'Risk-averse investors, retirement planning'
            },
            'balanced': {
                'name': 'Balanced Portfolio',
                'description': 'Balanced risk-return approach',
                'strategy': {'type': 'max_sharpe', 'lookback_period': 180},
                'rebalance_frequency': 'monthly',
                'transaction_cost': 0.001,
                'suitable_for': 'Long-term investors, moderate risk tolerance'
            },
            'aggressive': {
                'name': 'Aggressive Growth',
                'description': 'High-growth momentum strategy',
                'strategy': {'type': 'momentum', 'lookback_period': 60},
                'rebalance_frequency': 'monthly',
                'transaction_cost': 0.0015,
                'suitable_for': 'High risk tolerance, growth-focused'
            },
            'contrarian': {
                'name': 'Contrarian Strategy',
                'description': 'Mean reversion approach',
                'strategy': {'type': 'mean_reversion', 'lookback_period': 30},
                'rebalance_frequency': 'monthly',
                'transaction_cost': 0.001,
                'suitable_for': 'Range-bound markets, value investors'
            },
            'index_like': {
                'name': 'Index-Like Exposure',
                'description': 'Market cap weighted approach',
                'strategy': {'type': 'market_cap'},
                'rebalance_frequency': 'quarterly',
                'transaction_cost': 0.0005,
                'suitable_for': 'Passive investors, benchmark tracking'
            }
        }
        
        return jsonify({
            'presets': presets,
            'usage': 'Use preset configurations as starting points for your backtests'
        })
        
    except Exception as e:
        logger.error(f"Error getting backtest presets: {e}")
        return jsonify({'error': 'Failed to get presets'}), 500