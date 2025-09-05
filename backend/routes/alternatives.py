#!/usr/bin/env python3
"""
Alternative Assets API Routes
Cryptocurrency, commodities, bonds, REITs integration
"""
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from utils.alternative_assets import (
    AlternativeAssetsManager, get_popular_alternatives_by_category
)
from utils.error_handlers import safe_api_call
from models.database import get_user_by_id

# Create blueprint
alternatives_bp = Blueprint('alternatives', __name__)
logger = logging.getLogger('portfolio_optimizer.routes.alternatives')

@alternatives_bp.route('/alternatives/universe', methods=['GET'])
@safe_api_call
def get_alternatives_universe():
    """Get universe of available alternative assets"""
    try:
        manager = AlternativeAssetsManager()
        
        # Get all categories
        categories = request.args.getlist('categories')  # Optional filter
        universe = manager.get_alternative_universe(categories if categories else None)
        
        # Get popular alternatives by category
        popular_alternatives = get_popular_alternatives_by_category()
        
        response = {
            'universe': universe,
            'popular_by_category': popular_alternatives,
            'categories': {
                'cryptocurrency': {
                    'description': 'Digital currencies and tokens',
                    'risk_level': 'Very High',
                    'liquidity': 'High',
                    'correlation_with_stocks': 'Low to Medium'
                },
                'commodities': {
                    'description': 'Physical goods and commodity ETFs',
                    'risk_level': 'Medium to High',
                    'liquidity': 'Medium to High',
                    'correlation_with_stocks': 'Low'
                },
                'bonds': {
                    'description': 'Government and corporate bonds',
                    'risk_level': 'Low to Medium',
                    'liquidity': 'High',
                    'correlation_with_stocks': 'Low to Negative'
                },
                'real_estate': {
                    'description': 'REITs and real estate ETFs',
                    'risk_level': 'Medium',
                    'liquidity': 'High',
                    'correlation_with_stocks': 'Medium'
                },
                'alternative_strategies': {
                    'description': 'Volatility, currency, and strategy ETFs',
                    'risk_level': 'High',
                    'liquidity': 'Medium to High',
                    'correlation_with_stocks': 'Variable'
                }
            },
            'usage_guidelines': {
                'beginner': 'Start with 5-10% allocation to low-risk alternatives like bonds and REITs',
                'intermediate': '10-20% allocation across multiple alternative categories',
                'advanced': '15-30% allocation with sophisticated risk management'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting alternatives universe: {e}")
        return jsonify({'error': 'Failed to get alternatives universe'}), 500

@alternatives_bp.route('/alternatives/data', methods=['POST'])
@jwt_required()
@safe_api_call
def get_alternative_assets_data():
    """Get price data for alternative assets"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate inputs
        if not data.get('assets'):
            return jsonify({'error': 'Assets list is required'}), 400
        
        assets = data['assets']
        period = data.get('period', '1y')
        
        if len(assets) > 20:
            return jsonify({'error': 'Maximum 20 assets allowed'}), 400
        
        # Categorize assets by type
        manager = AlternativeAssetsManager()
        categorized_assets = {
            'crypto': [],
            'commodities': [],
            'bonds': [],
            'reits': [],
            'other': []
        }
        
        for asset in assets:
            asset_info = manager.get_asset_info(asset)
            if asset_info:
                if asset_info.asset_type == 'cryptocurrency':
                    categorized_assets['crypto'].append(asset)
                elif asset_info.asset_type == 'commodities':
                    categorized_assets['commodities'].append(asset)
                elif asset_info.asset_type == 'bonds':
                    categorized_assets['bonds'].append(asset)
                elif asset_info.asset_type == 'real_estate':
                    categorized_assets['reits'].append(asset)
                else:
                    categorized_assets['other'].append(asset)
            else:
                categorized_assets['other'].append(asset)
        
        logger.info(f"Fetching data for {len(assets)} alternative assets (user: {user_id})")
        
        # Fetch data by category
        all_data = {}
        
        if categorized_assets['crypto']:
            crypto_data = manager.get_crypto_data(categorized_assets['crypto'], period)
            all_data.update(crypto_data)
        
        if categorized_assets['commodities']:
            commodity_data = manager.get_commodity_data(categorized_assets['commodities'], period)
            all_data.update(commodity_data)
        
        if categorized_assets['bonds']:
            bond_data = manager.get_bond_data(categorized_assets['bonds'], period)
            all_data.update(bond_data)
        
        if categorized_assets['reits']:
            reit_data = manager.get_reit_data(categorized_assets['reits'], period)
            all_data.update(reit_data)
        
        # Handle other assets with general method
        if categorized_assets['other']:
            import yfinance as yf
            for symbol in categorized_assets['other']:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        all_data[symbol] = hist
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Format response
        formatted_data = {}
        summary_stats = {}
        
        for symbol, data in all_data.items():
            if data.empty:
                continue
            
            # Basic price data
            formatted_data[symbol] = {
                'prices': [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume']) if 'Volume' in row else 0
                    }
                    for date, row in data.tail(100).iterrows()  # Last 100 days
                ]
            }
            
            # Calculate summary statistics
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                summary_stats[symbol] = {
                    'current_price': float(data['Close'].iloc[-1]),
                    'period_return': float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100),
                    'volatility': float(returns.std() * np.sqrt(252) * 100),
                    'max_price': float(data['High'].max()),
                    'min_price': float(data['Low'].min()),
                    'average_volume': float(data['Volume'].mean()) if 'Volume' in data else 0,
                    'data_points': len(data)
                }
        
        response = {
            'data': formatted_data,
            'summary_statistics': summary_stats,
            'categorization': categorized_assets,
            'period_analyzed': period,
            'total_assets': len(formatted_data),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Retrieved data for {len(formatted_data)} alternative assets")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting alternative assets data: {e}")
        return jsonify({'error': f'Failed to get asset data: {str(e)}'}), 500

@alternatives_bp.route('/alternatives/correlation-analysis', methods=['POST'])
@jwt_required()
@safe_api_call
def analyze_alternative_correlations():
    """Analyze correlations between alternative and traditional assets"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate inputs
        alternative_assets = data.get('alternative_assets', [])
        traditional_assets = data.get('traditional_assets', [])
        period = data.get('period', '1y')
        
        if not alternative_assets:
            return jsonify({'error': 'Alternative assets list is required'}), 400
        
        if len(alternative_assets) + len(traditional_assets) > 15:
            return jsonify({'error': 'Maximum 15 total assets for correlation analysis'}), 400
        
        logger.info(f"Analyzing correlations for {len(alternative_assets)} alternative and {len(traditional_assets)} traditional assets")
        
        manager = AlternativeAssetsManager()
        
        # Get alternative assets data
        alt_data = {}
        for asset in alternative_assets:
            asset_info = manager.get_asset_info(asset)
            if asset_info and asset_info.asset_type == 'cryptocurrency':
                crypto_data = manager.get_crypto_data([asset], period)
                alt_data.update(crypto_data)
            else:
                # Use general method
                import yfinance as yf
                try:
                    ticker = yf.Ticker(asset)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        alt_data[asset] = hist
                except Exception as e:
                    logger.error(f"Error fetching alternative data for {asset}: {e}")
        
        # Get traditional assets data
        trad_data = {}
        if traditional_assets:
            import yfinance as yf
            for asset in traditional_assets:
                try:
                    ticker = yf.Ticker(asset)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        trad_data[asset] = hist
                except Exception as e:
                    logger.error(f"Error fetching traditional data for {asset}: {e}")
        
        # Perform correlation analysis
        correlation_analysis = manager.analyze_alternative_correlation(alt_data, trad_data)
        
        if 'error' in correlation_analysis:
            return jsonify({'error': correlation_analysis['error']}), 400
        
        # Add interpretation and recommendations
        correlation_analysis['recommendations'] = _generate_correlation_recommendations(
            correlation_analysis, alternative_assets, traditional_assets
        )
        
        response = {
            'correlation_analysis': correlation_analysis,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info("Correlation analysis completed")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return jsonify({'error': f'Correlation analysis failed: {str(e)}'}), 500

@alternatives_bp.route('/alternatives/risk-metrics', methods=['POST'])
@jwt_required()
@safe_api_call
def calculate_alternative_risk_metrics():
    """Calculate risk metrics for alternative assets"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Validate inputs
        assets = data.get('assets', [])
        period = data.get('period', '1y')
        
        if not assets:
            return jsonify({'error': 'Assets list is required'}), 400
        
        if len(assets) > 15:
            return jsonify({'error': 'Maximum 15 assets for risk analysis'}), 400
        
        logger.info(f"Calculating risk metrics for {len(assets)} alternative assets")
        
        manager = AlternativeAssetsManager()
        
        # Get asset data
        asset_data = {}
        for asset in assets:
            asset_info = manager.get_asset_info(asset)
            if asset_info and asset_info.asset_type == 'cryptocurrency':
                crypto_data = manager.get_crypto_data([asset], period)
                asset_data.update(crypto_data)
            else:
                # Use general method
                import yfinance as yf
                try:
                    ticker = yf.Ticker(asset)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        asset_data[asset] = hist
                except Exception as e:
                    logger.error(f"Error fetching data for {asset}: {e}")
        
        # Calculate risk metrics
        risk_metrics = manager.calculate_alternative_risk_metrics(asset_data)
        
        # Add risk interpretations
        interpreted_metrics = {}
        for asset, metrics in risk_metrics.items():
            interpreted_metrics[asset] = {
                **metrics,
                'risk_level': _classify_risk_level(metrics['volatility']),
                'tail_risk_level': _classify_tail_risk(metrics['tail_risk']),
                'stability': _assess_stability(metrics['max_drawdown'], metrics['volatility'])
            }
        
        # Generate risk comparison
        if len(risk_metrics) > 1:
            risk_comparison = _generate_risk_comparison(risk_metrics)
        else:
            risk_comparison = {}
        
        response = {
            'risk_metrics': interpreted_metrics,
            'risk_comparison': risk_comparison,
            'benchmark_comparisons': _get_benchmark_comparisons(risk_metrics),
            'recommendations': _generate_risk_recommendations(interpreted_metrics),
            'period_analyzed': period,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Risk metrics calculated for {len(risk_metrics)} assets")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error calculating alternative risk metrics: {e}")
        return jsonify({'error': f'Risk calculation failed: {str(e)}'}), 500

@alternatives_bp.route('/alternatives/allocation-suggestions', methods=['POST'])
@jwt_required()
@safe_api_call
def get_allocation_suggestions():
    """Get alternative asset allocation suggestions"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Parse inputs
        traditional_portfolio = data.get('traditional_portfolio', {})
        risk_tolerance = data.get('risk_tolerance', 'moderate')
        investment_goals = data.get('investment_goals', ['diversification'])
        portfolio_size = data.get('portfolio_size', 100000)
        
        # Validate risk tolerance
        if risk_tolerance not in ['conservative', 'moderate', 'aggressive']:
            return jsonify({'error': 'Risk tolerance must be conservative, moderate, or aggressive'}), 400
        
        logger.info(f"Generating allocation suggestions for {risk_tolerance} risk tolerance")
        
        manager = AlternativeAssetsManager()
        
        # Get allocation suggestions
        suggestions = manager.suggest_alternative_allocation(
            traditional_portfolio,
            risk_tolerance,
            investment_goals
        )
        
        # Add specific asset recommendations with dollar amounts
        detailed_recommendations = {}
        total_alternative_allocation = portfolio_size * (suggestions['total_alternative_percent'] / 100)
        
        for category, percentage in suggestions['recommended_allocation'].items():
            allocation_amount = portfolio_size * (percentage / 100)
            
            # Get specific assets for this category
            category_assets = suggestions['asset_suggestions'].get(category, [])
            
            detailed_recommendations[category] = {
                'allocation_percentage': percentage,
                'allocation_amount': allocation_amount,
                'suggested_assets': category_assets[:3],  # Top 3 suggestions
                'rationale': suggestions['rationale'].get(category, ''),
                'risk_considerations': _get_category_risk_considerations(category, risk_tolerance)
            }
        
        # Generate implementation timeline
        implementation_timeline = _generate_implementation_timeline(
            detailed_recommendations, risk_tolerance
        )
        
        response = {
            'allocation_summary': {
                'total_alternative_percentage': suggestions['total_alternative_percent'],
                'total_alternative_amount': total_alternative_allocation,
                'remaining_traditional_percentage': 100 - suggestions['total_alternative_percent'],
                'risk_tolerance': risk_tolerance,
                'investment_goals': investment_goals
            },
            'detailed_recommendations': detailed_recommendations,
            'implementation_timeline': implementation_timeline,
            'risk_warnings': _generate_risk_warnings(suggestions, risk_tolerance),
            'monitoring_guidelines': _generate_monitoring_guidelines(detailed_recommendations),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Allocation suggestions generated: {suggestions['total_alternative_percent']:.1f}% alternatives")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating allocation suggestions: {e}")
        return jsonify({'error': f'Allocation suggestions failed: {str(e)}'}), 500

@alternatives_bp.route('/alternatives/crypto-metrics/<coin_id>', methods=['GET'])
@safe_api_call
def get_crypto_detailed_metrics(coin_id):
    """Get detailed cryptocurrency metrics"""
    try:
        manager = AlternativeAssetsManager()
        
        # Get detailed crypto metrics
        crypto_metrics = manager.get_crypto_metrics(coin_id)
        
        if not crypto_metrics:
            return jsonify({'error': 'Unable to fetch crypto metrics'}), 404
        
        # Convert to dict and add interpretations
        metrics_dict = {
            'market_cap_rank': crypto_metrics.market_cap_rank,
            'circulating_supply': crypto_metrics.circulating_supply,
            'total_supply': crypto_metrics.total_supply,
            'max_supply': crypto_metrics.max_supply,
            'all_time_high': crypto_metrics.all_time_high,
            'all_time_low': crypto_metrics.all_time_low,
            'price_change_24h': crypto_metrics.price_change_24h,
            'volume_24h': crypto_metrics.volume_24h,
            'supply_analysis': {
                'inflation_rate': ((crypto_metrics.total_supply - crypto_metrics.circulating_supply) / 
                                 crypto_metrics.circulating_supply * 100) if crypto_metrics.circulating_supply > 0 else 0,
                'scarcity_score': _calculate_scarcity_score(crypto_metrics),
                'supply_type': 'capped' if crypto_metrics.max_supply else 'uncapped'
            }
        }
        
        response = {
            'coin_id': coin_id,
            'metrics': metrics_dict,
            'risk_assessment': _assess_crypto_risk(crypto_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting crypto metrics for {coin_id}: {e}")
        return jsonify({'error': 'Failed to get crypto metrics'}), 500

# Helper functions
def _generate_correlation_recommendations(analysis, alt_assets, trad_assets):
    """Generate recommendations based on correlation analysis"""
    recommendations = []
    
    if 'high_correlation_pairs' in analysis:
        high_corr_pairs = analysis['high_correlation_pairs']
        if high_corr_pairs:
            recommendations.append({
                'type': 'diversification',
                'message': f"Found {len(high_corr_pairs)} highly correlated asset pairs - consider reducing overlap",
                'priority': 'high'
            })
    
    # Check for insights
    for insight in analysis.get('insights', []):
        if insight['type'] == 'diversification_benefit':
            if insight['value'] < 0.5:
                recommendations.append({
                    'type': 'positive',
                    'message': 'Good diversification benefit - alternative assets have low correlation with traditional assets',
                    'priority': 'medium'
                })
            else:
                recommendations.append({
                    'type': 'warning',
                    'message': 'Limited diversification benefit - consider different alternative assets',
                    'priority': 'medium'
                })
    
    return recommendations

def _classify_risk_level(volatility):
    """Classify risk level based on volatility"""
    if volatility < 0.15:
        return 'Low'
    elif volatility < 0.30:
        return 'Medium'
    elif volatility < 0.60:
        return 'High'
    else:
        return 'Very High'

def _classify_tail_risk(tail_risk):
    """Classify tail risk level"""
    tail_risk_abs = abs(tail_risk)
    if tail_risk_abs < 0.05:
        return 'Low'
    elif tail_risk_abs < 0.10:
        return 'Medium'
    else:
        return 'High'

def _assess_stability(max_drawdown, volatility):
    """Assess asset stability"""
    drawdown_abs = abs(max_drawdown)
    if drawdown_abs < 0.20 and volatility < 0.25:
        return 'Stable'
    elif drawdown_abs < 0.40 and volatility < 0.50:
        return 'Moderate'
    else:
        return 'Volatile'

def _generate_risk_comparison(risk_metrics):
    """Generate risk comparison between assets"""
    if not risk_metrics:
        return {}
    
    volatilities = [m['volatility'] for m in risk_metrics.values()]
    max_drawdowns = [abs(m['max_drawdown']) for m in risk_metrics.values()]
    
    # Find extremes
    lowest_vol = min(risk_metrics.keys(), key=lambda x: risk_metrics[x]['volatility'])
    highest_vol = max(risk_metrics.keys(), key=lambda x: risk_metrics[x]['volatility'])
    
    return {
        'lowest_volatility': {
            'asset': lowest_vol,
            'volatility': risk_metrics[lowest_vol]['volatility']
        },
        'highest_volatility': {
            'asset': highest_vol,
            'volatility': risk_metrics[highest_vol]['volatility']
        },
        'average_volatility': np.mean(volatilities),
        'average_max_drawdown': np.mean(max_drawdowns)
    }

def _get_benchmark_comparisons(risk_metrics):
    """Compare risk metrics to common benchmarks"""
    benchmarks = {
        'S&P 500': {'volatility': 0.16, 'max_drawdown': -0.34},
        'Bitcoin': {'volatility': 0.80, 'max_drawdown': -0.84},
        'Gold': {'volatility': 0.20, 'max_drawdown': -0.45},
        '10-Year Treasury': {'volatility': 0.08, 'max_drawdown': -0.12}
    }
    
    comparisons = {}
    for asset, metrics in risk_metrics.items():
        asset_comparisons = {}
        for benchmark, bench_metrics in benchmarks.items():
            asset_comparisons[benchmark] = {
                'volatility_ratio': metrics['volatility'] / bench_metrics['volatility'],
                'risk_level': 'Higher' if metrics['volatility'] > bench_metrics['volatility'] else 'Lower'
            }
        comparisons[asset] = asset_comparisons
    
    return comparisons

def _generate_risk_recommendations(risk_metrics):
    """Generate risk-based recommendations"""
    recommendations = []
    
    for asset, metrics in risk_metrics.items():
        if metrics['volatility'] > 0.60:  # Very high volatility
            recommendations.append({
                'asset': asset,
                'type': 'warning',
                'message': f'{asset} has very high volatility ({metrics["volatility"]*100:.1f}%) - limit allocation to <5%'
            })
        
        if abs(metrics['max_drawdown']) > 0.50:  # Large drawdowns
            recommendations.append({
                'asset': asset,
                'type': 'caution',
                'message': f'{asset} has experienced large drawdowns - ensure adequate risk tolerance'
            })
    
    return recommendations

def _get_category_risk_considerations(category, risk_tolerance):
    """Get risk considerations for each category"""
    considerations = {
        'commodities': {
            'conservative': 'Commodities can be volatile but provide inflation protection',
            'moderate': 'Good diversification tool, but monitor correlation during market stress',
            'aggressive': 'Can add significant volatility, consider tactical allocation'
        },
        'cryptocurrency': {
            'conservative': 'Not recommended - extremely high volatility',
            'moderate': 'Very high risk - limit to 2-5% maximum allocation',
            'aggressive': 'High growth potential but extreme volatility - maximum 10%'
        },
        'real_estate': {
            'conservative': 'Generally stable with income generation',
            'moderate': 'Good diversification with moderate risk',
            'aggressive': 'Can add leverage and sector concentration risk'
        },
        'bonds': {
            'conservative': 'Lower risk, good diversification from equities',
            'moderate': 'Interest rate and credit risk considerations',
            'aggressive': 'High yield bonds add credit risk but higher returns'
        }
    }
    
    return considerations.get(category, {}).get(risk_tolerance, 'Monitor risk carefully')

def _generate_implementation_timeline(recommendations, risk_tolerance):
    """Generate implementation timeline for alternative allocations"""
    if risk_tolerance == 'conservative':
        timeline = 'Implement over 6-12 months with gradual allocation'
    elif risk_tolerance == 'moderate':
        timeline = 'Implement over 3-6 months with measured allocation'
    else:
        timeline = 'Can implement over 1-3 months with careful monitoring'
    
    phases = [
        {'phase': 1, 'duration': 'Month 1-2', 'action': 'Start with lowest risk alternatives (bonds, REITs)'},
        {'phase': 2, 'duration': 'Month 3-4', 'action': 'Add commodity exposure gradually'},
        {'phase': 3, 'duration': 'Month 5-6', 'action': 'Consider higher risk alternatives if appropriate'}
    ]
    
    return {'timeline': timeline, 'phases': phases}

def _generate_risk_warnings(suggestions, risk_tolerance):
    """Generate risk warnings based on suggestions"""
    warnings = []
    
    if 'cryptocurrency' in suggestions['recommended_allocation']:
        warnings.append('Cryptocurrency is extremely volatile and speculative - only invest what you can afford to lose')
    
    if suggestions['total_alternative_percent'] > 30:
        warnings.append('High alternative allocation increases portfolio complexity and may reduce liquidity')
    
    if risk_tolerance == 'conservative' and suggestions['total_alternative_percent'] > 15:
        warnings.append('Alternative allocation may be too high for conservative risk tolerance')
    
    return warnings

def _generate_monitoring_guidelines(recommendations):
    """Generate monitoring guidelines for alternative allocations"""
    return [
        'Review alternative allocations quarterly',
        'Monitor correlation changes during market stress',
        'Rebalance when allocations drift >20% from targets',
        'Stay informed about regulatory changes affecting alternatives',
        'Consider tax implications of alternative investments'
    ]

def _calculate_scarcity_score(crypto_metrics):
    """Calculate scarcity score for cryptocurrency"""
    if not crypto_metrics.max_supply:
        return 0  # No supply cap
    
    remaining_supply = crypto_metrics.max_supply - crypto_metrics.circulating_supply
    scarcity_ratio = remaining_supply / crypto_metrics.max_supply
    
    # Higher score = more scarce
    return max(0, 100 - (scarcity_ratio * 100))

def _assess_crypto_risk(crypto_metrics):
    """Assess cryptocurrency risk based on metrics"""
    risk_factors = []
    risk_score = 0  # 0-100 scale
    
    # Volatility risk (based on 24h change)
    daily_change = abs(crypto_metrics.price_change_24h)
    if daily_change > 20:
        risk_factors.append('Extremely high daily volatility')
        risk_score += 30
    elif daily_change > 10:
        risk_factors.append('High daily volatility')
        risk_score += 20
    elif daily_change > 5:
        risk_factors.append('Moderate daily volatility')
        risk_score += 10
    
    # Market cap rank (higher rank = higher risk)
    if crypto_metrics.market_cap_rank > 100:
        risk_factors.append('Low market cap ranking - higher volatility risk')
        risk_score += 25
    elif crypto_metrics.market_cap_rank > 50:
        risk_factors.append('Medium market cap ranking')
        risk_score += 15
    elif crypto_metrics.market_cap_rank > 10:
        risk_factors.append('Top tier market cap')
        risk_score += 5
    
    # Supply characteristics
    if not crypto_metrics.max_supply:
        risk_factors.append('Unlimited supply - inflation risk')
        risk_score += 15
    
    # Overall assessment
    if risk_score >= 60:
        risk_level = 'Very High'
    elif risk_score >= 40:
        risk_level = 'High'
    elif risk_score >= 20:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors
    }