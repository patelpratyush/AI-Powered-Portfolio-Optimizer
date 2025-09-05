#!/usr/bin/env python3
"""
Risk Management API Routes
VaR, CVaR, and advanced risk analytics
"""
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import yfinance as yf

from utils.risk_management import (
    RiskManager, calculate_portfolio_var_cvar, stress_test_quick
)
from utils.error_handlers import safe_api_call
from models.database import get_user_by_id

# Create blueprint
risk_bp = Blueprint('risk', __name__)
logger = logging.getLogger('portfolio_optimizer.routes.risk')

@risk_bp.route('/risk/var-cvar', methods=['POST'])
@jwt_required()
@safe_api_call
def calculate_var_cvar():
    """Calculate VaR and CVaR for a portfolio"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['tickers', 'weights']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        tickers = [t.upper().strip() for t in data['tickers']]
        weights = data['weights']
        
        if len(tickers) != len(weights):
            return jsonify({'error': 'Number of tickers must match number of weights'}), 400
        
        if abs(sum(weights) - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        # Get parameters
        portfolio_value = data.get('portfolio_value', 100000)
        confidence_levels = data.get('confidence_levels', [0.95, 0.99])
        methods = data.get('methods', ['historical', 'parametric'])
        period = data.get('period', '1y')
        
        # Validate inputs
        if portfolio_value <= 0:
            return jsonify({'error': 'Portfolio value must be positive'}), 400
        
        valid_methods = ['historical', 'parametric', 'monte_carlo', 't_distribution']
        methods = [m for m in methods if m in valid_methods]
        if not methods:
            methods = ['historical']
        
        # Fetch historical data
        logger.info(f"Calculating VaR/CVaR for {len(tickers)} tickers (user: {user_id})")
        
        try:
            # Get price data for all tickers
            price_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if hist.empty:
                    return jsonify({'error': f'No data available for {ticker}'}), 400
                price_data[ticker] = hist['Close']
            
            # Combine into DataFrame and calculate returns
            prices_df = pd.DataFrame(price_data).dropna()
            if len(prices_df) < 30:
                return jsonify({'error': 'Insufficient historical data (need at least 30 days)'}), 400
            
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_weights = np.array(weights)
            portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return jsonify({'error': 'Failed to fetch historical data'}), 500
        
        # Calculate VaR and CVaR
        try:
            risk_results = calculate_portfolio_var_cvar(
                portfolio_returns,
                portfolio_value=portfolio_value,
                confidence_levels=confidence_levels,
                methods=methods
            )
            
            # Calculate additional risk metrics
            risk_manager = RiskManager()
            
            # Get benchmark (SPY) for comparison
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period=period)
            spy_returns = spy_hist['Close'].pct_change().dropna()
            
            # Align with portfolio returns
            aligned_data = pd.concat([portfolio_returns, spy_returns], axis=1, join='inner').dropna()
            if len(aligned_data) > 10:
                aligned_portfolio = aligned_data.iloc[:, 0]
                aligned_benchmark = aligned_data.iloc[:, 1]
            else:
                aligned_portfolio = portfolio_returns
                aligned_benchmark = None
            
            comprehensive_metrics = risk_manager.calculate_portfolio_risk_metrics(
                aligned_portfolio,
                aligned_benchmark,
                portfolio_value
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return jsonify({'error': 'Failed to calculate risk metrics'}), 500
        
        # Format response
        response = {
            'portfolio_info': {
                'tickers': tickers,
                'weights': weights,
                'portfolio_value': portfolio_value,
                'period_analyzed': period,
                'data_points': len(portfolio_returns)
            },
            'var_cvar_results': risk_results,
            'comprehensive_metrics': {
                'volatility': comprehensive_metrics.volatility,
                'skewness': comprehensive_metrics.skewness,
                'kurtosis': comprehensive_metrics.kurtosis,
                'max_drawdown': comprehensive_metrics.max_drawdown,
                'beta': comprehensive_metrics.beta,
                'correlation_with_market': comprehensive_metrics.correlation_with_market,
                'downside_deviation': comprehensive_metrics.downside_deviation,
                'tracking_error': comprehensive_metrics.tracking_error,
                'information_ratio': comprehensive_metrics.information_ratio,
                'upside_capture': comprehensive_metrics.upside_capture,
                'downside_capture': comprehensive_metrics.downside_capture
            },
            'risk_interpretation': _interpret_risk_metrics(comprehensive_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Risk analysis completed. VaR(95%): {risk_results.get('historical', {}).get('var', {}).get('var_95_percent', 0):.2f}%")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"VaR/CVaR calculation error: {e}")
        return jsonify({'error': f'Risk calculation failed: {str(e)}'}), 500

@risk_bp.route('/risk/stress-test', methods=['POST'])
@jwt_required()
@safe_api_call
def stress_test_portfolio():
    """Perform stress testing on portfolio"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate required fields
        if not data.get('tickers'):
            return jsonify({'error': 'Tickers are required'}), 400
        
        tickers = [t.upper().strip() for t in data['tickers']]
        weights = data.get('weights')
        
        if weights and len(tickers) != len(weights):
            return jsonify({'error': 'Number of tickers must match number of weights'}), 400
        
        if weights and abs(sum(weights) - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        scenarios = data.get('scenarios')  # Optional: specific scenarios to test
        portfolio_value = data.get('portfolio_value', 100000)
        
        # Run stress test
        logger.info(f"Running stress test for {len(tickers)} tickers (user: {user_id})")
        
        stress_results = stress_test_quick(
            tickers=tickers,
            weights=weights,
            scenarios=scenarios
        )
        
        # Format results with portfolio value impact
        formatted_results = {}
        for scenario_name, result in stress_results.items():
            if hasattr(result, 'portfolio_return'):
                value_impact = result.portfolio_return * portfolio_value
                formatted_results[scenario_name] = {
                    'scenario_name': result.scenario_name,
                    'portfolio_return_percent': result.portfolio_return * 100,
                    'portfolio_value_change': value_impact,
                    'new_portfolio_value': portfolio_value + value_impact,
                    'worst_asset': result.worst_asset,
                    'worst_asset_return_percent': result.worst_asset_return * 100,
                    'best_asset': result.best_asset,
                    'best_asset_return_percent': result.best_asset_return * 100,
                    'correlation_breakdown': result.correlations_breakdown,
                    'severity': _classify_stress_severity(result.portfolio_return)
                }
        
        # Summary statistics
        if formatted_results:
            worst_case = min(formatted_results.values(), key=lambda x: x['portfolio_return_percent'])
            best_case = max(formatted_results.values(), key=lambda x: x['portfolio_return_percent'])
            
            summary = {
                'worst_case_scenario': worst_case['scenario_name'],
                'worst_case_loss_percent': worst_case['portfolio_return_percent'],
                'worst_case_loss_dollar': abs(worst_case['portfolio_value_change']),
                'best_case_scenario': best_case['scenario_name'],
                'best_case_return_percent': best_case['portfolio_return_percent'],
                'scenarios_tested': len(formatted_results),
                'severe_scenarios': len([r for r in formatted_results.values() if r['severity'] == 'severe']),
                'portfolio_resilience': _assess_portfolio_resilience(formatted_results)
            }
        else:
            summary = {}
        
        response = {
            'portfolio_info': {
                'tickers': tickers,
                'weights': weights if weights else [1/len(tickers)] * len(tickers),
                'portfolio_value': portfolio_value
            },
            'stress_test_results': formatted_results,
            'summary': summary,
            'risk_recommendations': _generate_risk_recommendations(formatted_results),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Stress test completed. Worst case: {summary.get('worst_case_loss_percent', 0):.1f}%")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Stress test error: {e}")
        return jsonify({'error': f'Stress test failed: {str(e)}'}), 500

@risk_bp.route('/risk/component-analysis', methods=['POST'])
@jwt_required()
@safe_api_call
def component_risk_analysis():
    """Analyze risk contribution of individual portfolio components"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate inputs
        tickers = [t.upper().strip() for t in data.get('tickers', [])]
        weights = data.get('weights', [])
        
        if not tickers or len(tickers) != len(weights):
            return jsonify({'error': 'Must provide equal number of tickers and weights'}), 400
        
        if abs(sum(weights) - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        period = data.get('period', '1y')
        confidence_level = data.get('confidence_level', 0.95)
        
        # Fetch data and calculate returns
        logger.info(f"Analyzing component risk for {len(tickers)} assets (user: {user_id})")
        
        try:
            price_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if hist.empty:
                    return jsonify({'error': f'No data available for {ticker}'}), 400
                price_data[ticker] = hist['Close']
            
            prices_df = pd.DataFrame(price_data).dropna()
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_weights = dict(zip(tickers, weights))
            portfolio_returns = (returns_df * np.array(weights)).sum(axis=1)
            
        except Exception as e:
            return jsonify({'error': 'Failed to fetch historical data'}), 500
        
        # Calculate component risk metrics
        risk_manager = RiskManager()
        
        # Portfolio VaR
        var_result = risk_manager.calculate_var(
            portfolio_returns,
            confidence_levels=[confidence_level]
        )
        portfolio_var = var_result.var_95
        
        # Individual asset returns for marginal VaR calculation
        individual_returns = {ticker: returns_df[ticker] for ticker in tickers}
        
        # Calculate marginal and component VaR
        marginal_vars = risk_manager.calculate_marginal_var(
            portfolio_returns,
            individual_returns,
            portfolio_weights,
            confidence_level
        )
        
        component_vars = risk_manager.calculate_component_var(
            marginal_vars,
            portfolio_weights,
            portfolio_var
        )
        
        # Calculate individual asset risk metrics
        individual_metrics = {}
        for ticker in tickers:
            asset_returns = returns_df[ticker].dropna()
            if len(asset_returns) > 10:
                individual_metrics[ticker] = {
                    'volatility': asset_returns.std() * np.sqrt(252),
                    'var_95': np.percentile(asset_returns, 5),
                    'max_drawdown': _calculate_max_drawdown(asset_returns),
                    'skewness': float(asset_returns.skew()),
                    'weight': portfolio_weights[ticker],
                    'marginal_var': marginal_vars.get(ticker, 0),
                    'component_var': component_vars.get(ticker, 0),
                    'risk_contribution_percent': (abs(component_vars.get(ticker, 0)) / abs(portfolio_var)) * 100 if portfolio_var != 0 else 0
                }
        
        # Risk decomposition
        total_component_var = sum(abs(v) for v in component_vars.values())
        risk_contributions = {}
        for ticker in tickers:
            contrib = abs(component_vars.get(ticker, 0))
            risk_contributions[ticker] = (contrib / total_component_var) * 100 if total_component_var != 0 else 0
        
        response = {
            'portfolio_metrics': {
                'portfolio_var_95': portfolio_var,
                'portfolio_var_95_percent': abs(portfolio_var) * 100,
                'portfolio_volatility': portfolio_returns.std() * np.sqrt(252)
            },
            'component_analysis': individual_metrics,
            'risk_contributions': risk_contributions,
            'risk_budget_vs_weight': {
                ticker: {
                    'weight_percent': weights[i] * 100,
                    'risk_contribution_percent': risk_contributions.get(ticker, 0),
                    'risk_efficiency': risk_contributions.get(ticker, 0) / (weights[i] * 100) if weights[i] > 0 else 0
                }
                for i, ticker in enumerate(tickers)
            },
            'recommendations': _generate_risk_budget_recommendations(individual_metrics, risk_contributions, weights, tickers),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Component risk analysis completed for {len(tickers)} assets")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Component risk analysis error: {e}")
        return jsonify({'error': f'Component analysis failed: {str(e)}'}), 500

@risk_bp.route('/risk/scenarios', methods=['GET'])
@safe_api_call
def get_stress_scenarios():
    """Get available stress test scenarios"""
    try:
        risk_manager = RiskManager()
        scenarios = risk_manager._get_default_stress_scenarios()
        
        formatted_scenarios = {}
        for name, shocks in scenarios.items():
            formatted_scenarios[name] = {
                'name': name.replace('_', ' ').title(),
                'description': _get_scenario_description(name),
                'severity': _classify_scenario_severity(shocks),
                'affected_assets': list(shocks.keys()),
                'market_impact': shocks.get('market', 0),
                'example_shocks': {k: f"{v*100:+.1f}%" for k, v in list(shocks.items())[:5]}
            }
        
        return jsonify({
            'scenarios': formatted_scenarios,
            'usage': 'Select scenarios to test your portfolio against various market conditions',
            'custom_scenarios': 'You can also provide custom shock values for specific assets'
        })
        
    except Exception as e:
        logger.error(f"Error getting stress scenarios: {e}")
        return jsonify({'error': 'Failed to get scenarios'}), 500

def _interpret_risk_metrics(metrics) -> Dict[str, str]:
    """Interpret risk metrics for user understanding"""
    interpretation = {}
    
    # Volatility interpretation
    vol = metrics.volatility
    if vol < 0.10:
        interpretation['volatility'] = 'Low risk - Conservative portfolio'
    elif vol < 0.20:
        interpretation['volatility'] = 'Moderate risk - Balanced portfolio'
    else:
        interpretation['volatility'] = 'High risk - Aggressive portfolio'
    
    # Sharpe interpretation via information ratio
    ir = metrics.information_ratio
    if ir > 0.5:
        interpretation['risk_efficiency'] = 'Excellent risk-adjusted returns'
    elif ir > 0.0:
        interpretation['risk_efficiency'] = 'Good risk-adjusted performance'
    else:
        interpretation['risk_efficiency'] = 'Poor risk-adjusted performance'
    
    # Beta interpretation
    beta = metrics.beta
    if beta > 1.2:
        interpretation['market_sensitivity'] = 'High sensitivity to market movements'
    elif beta > 0.8:
        interpretation['market_sensitivity'] = 'Moderate sensitivity to market movements'
    else:
        interpretation['market_sensitivity'] = 'Low sensitivity to market movements'
    
    # Drawdown interpretation
    dd = abs(metrics.max_drawdown)
    if dd > 0.30:
        interpretation['downside_risk'] = 'High downside risk - Significant losses possible'
    elif dd > 0.15:
        interpretation['downside_risk'] = 'Moderate downside risk'
    else:
        interpretation['downside_risk'] = 'Low downside risk - Relatively stable'
    
    return interpretation

def _classify_stress_severity(portfolio_return: float) -> str:
    """Classify stress test severity"""
    loss = abs(portfolio_return)
    if loss > 0.30:
        return 'severe'
    elif loss > 0.15:
        return 'moderate'
    elif loss > 0.05:
        return 'mild'
    else:
        return 'minimal'

def _assess_portfolio_resilience(stress_results: Dict) -> str:
    """Assess overall portfolio resilience"""
    severe_count = len([r for r in stress_results.values() if r['severity'] == 'severe'])
    total_count = len(stress_results)
    
    if severe_count == 0:
        return 'high_resilience'
    elif severe_count / total_count < 0.3:
        return 'moderate_resilience'
    else:
        return 'low_resilience'

def _generate_risk_recommendations(stress_results: Dict) -> List[str]:
    """Generate risk management recommendations"""
    recommendations = []
    
    if not stress_results:
        return recommendations
    
    # Check for concentration risk
    severe_scenarios = [r for r in stress_results.values() if r['severity'] == 'severe']
    if len(severe_scenarios) > 2:
        recommendations.append("Consider diversifying across sectors to reduce concentration risk")
    
    # Check tech exposure
    if 'tech_selloff' in stress_results and stress_results['tech_selloff']['severity'] == 'severe':
        recommendations.append("High technology exposure detected - consider reducing tech concentration")
    
    # Check market sensitivity
    if 'market_crash' in stress_results and stress_results['market_crash']['portfolio_return_percent'] < -25:
        recommendations.append("Portfolio highly sensitive to market crashes - consider defensive assets")
    
    # Check inflation protection
    if 'inflation_shock' in stress_results and stress_results['inflation_shock']['portfolio_return_percent'] < -15:
        recommendations.append("Limited inflation protection - consider inflation-hedged assets")
    
    return recommendations

def _generate_risk_budget_recommendations(individual_metrics: Dict, 
                                        risk_contributions: Dict,
                                        weights: List[float],
                                        tickers: List[str]) -> List[str]:
    """Generate risk budgeting recommendations"""
    recommendations = []
    
    # Find assets with high risk contribution relative to weight
    for i, ticker in enumerate(tickers):
        weight_pct = weights[i] * 100
        risk_contrib_pct = risk_contributions.get(ticker, 0)
        
        if risk_contrib_pct > weight_pct * 1.5:  # Risk contribution 50% higher than weight
            recommendations.append(f"{ticker} contributes {risk_contrib_pct:.1f}% of risk but only {weight_pct:.1f}% of weight - consider reducing allocation")
    
    # Check for concentration
    max_risk_contrib = max(risk_contributions.values()) if risk_contributions else 0
    if max_risk_contrib > 40:
        max_ticker = max(risk_contributions, key=risk_contributions.get)
        recommendations.append(f"High risk concentration in {max_ticker} ({max_risk_contrib:.1f}% of total risk)")
    
    return recommendations

def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown for a return series"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def _get_scenario_description(scenario_name: str) -> str:
    """Get description for stress scenario"""
    descriptions = {
        'market_crash': 'Broad market decline similar to 2008 or March 2020',
        'tech_selloff': 'Technology sector rotation or bubble burst',
        'inflation_shock': 'Unexpected inflation surge leading to rate hikes',
        'recession_scenario': 'Economic recession with sector-specific impacts',
        'geopolitical_crisis': 'Geopolitical tensions affecting global markets'
    }
    return descriptions.get(scenario_name, 'Custom stress scenario')

def _classify_scenario_severity(shocks: Dict[str, float]) -> str:
    """Classify scenario severity based on shock magnitudes"""
    max_shock = max(abs(v) for v in shocks.values())
    if max_shock > 0.30:
        return 'severe'
    elif max_shock > 0.15:
        return 'moderate'
    else:
        return 'mild'