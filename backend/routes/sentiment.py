#!/usr/bin/env python3
"""
Sentiment analysis API routes
"""
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.sentiment_analysis import SentimentAggregator, create_sentiment_features
from utils.error_handlers import safe_api_call
from utils.cache import get_cache_client
from models.database import get_user_by_id

# Create blueprint
sentiment_bp = Blueprint('sentiment', __name__)
logger = logging.getLogger('portfolio_optimizer.routes.sentiment')

# Initialize sentiment aggregator
cache_client = get_cache_client()
sentiment_aggregator = SentimentAggregator(cache_client)

@sentiment_bp.route('/sentiment/<ticker>')
@jwt_required()
@safe_api_call
def get_ticker_sentiment(ticker):
    """Get sentiment analysis for a specific ticker"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get query parameters
        days = request.args.get('days', default=7, type=int)
        days = min(max(days, 1), 30)  # Limit to 1-30 days
        
        # Get sentiment analysis
        sentiment_score = sentiment_aggregator.get_ticker_sentiment(ticker.upper(), days)
        
        # Format response
        result = {
            'ticker': ticker.upper(),
            'sentiment': {
                'score': sentiment_score.score,
                'confidence': sentiment_score.confidence,
                'interpretation': _interpret_sentiment(sentiment_score.score, sentiment_score.confidence),
                'volume': sentiment_score.volume,
                'source': sentiment_score.source,
                'last_updated': sentiment_score.timestamp.isoformat(),
                'text_sample': sentiment_score.text_sample
            },
            'analysis_period_days': days,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Sentiment analysis completed for {ticker} (user: {user_id})")
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Invalid ticker format: {e}")
        return jsonify({'error': 'Invalid ticker symbol'}), 400
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}")
        return jsonify({'error': 'Failed to analyze sentiment'}), 500

@sentiment_bp.route('/sentiment/portfolio')
@jwt_required()
@safe_api_call
def get_portfolio_sentiment():
    """Get sentiment analysis for user's portfolio"""
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get request parameters
        data = request.get_json() or {}
        tickers = data.get('tickers', [])
        days = data.get('days', 7)
        days = min(max(days, 1), 30)  # Limit to 1-30 days
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Validate tickers
        tickers = [t.upper() for t in tickers if t and len(t) <= 10]
        if not tickers:
            return jsonify({'error': 'No valid tickers provided'}), 400
        
        # Limit number of tickers to prevent abuse
        tickers = tickers[:20]
        
        # Get sentiment for all tickers
        sentiment_scores = sentiment_aggregator.get_market_sentiment(tickers, days)
        
        # Format response
        portfolio_sentiment = {}
        overall_scores = []
        overall_confidences = []
        total_volume = 0
        
        for ticker, sentiment in sentiment_scores.items():
            portfolio_sentiment[ticker] = {
                'score': sentiment.score,
                'confidence': sentiment.confidence,
                'interpretation': _interpret_sentiment(sentiment.score, sentiment.confidence),
                'volume': sentiment.volume,
                'source': sentiment.source,
                'last_updated': sentiment.timestamp.isoformat(),
                'text_sample': sentiment.text_sample[:100] + "..." if len(sentiment.text_sample) > 100 else sentiment.text_sample
            }
            
            if sentiment.confidence > 0.3:  # Only include confident predictions
                overall_scores.append(sentiment.score)
                overall_confidences.append(sentiment.confidence)
            
            total_volume += sentiment.volume
        
        # Calculate overall portfolio sentiment
        if overall_scores:
            weighted_scores = [score * conf for score, conf in zip(overall_scores, overall_confidences)]
            total_weight = sum(overall_confidences)
            
            overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
            overall_confidence = sum(overall_confidences) / len(overall_confidences)
        else:
            overall_score = 0.0
            overall_confidence = 0.0
        
        result = {
            'portfolio_sentiment': {
                'overall_score': overall_score,
                'overall_confidence': overall_confidence,
                'overall_interpretation': _interpret_sentiment(overall_score, overall_confidence),
                'total_volume': total_volume,
                'tickers_analyzed': len(tickers),
                'high_confidence_tickers': len([s for s in sentiment_scores.values() if s.confidence > 0.7])
            },
            'individual_sentiment': portfolio_sentiment,
            'analysis_period_days': days,
            'generated_at': datetime.now().isoformat(),
            'features': create_sentiment_features(sentiment_scores)
        }
        
        logger.info(f"Portfolio sentiment analysis completed for {len(tickers)} tickers (user: {user_id})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting portfolio sentiment: {e}")
        return jsonify({'error': 'Failed to analyze portfolio sentiment'}), 500

@sentiment_bp.route('/sentiment/market')
@safe_api_call
def get_market_sentiment():
    """Get overall market sentiment from major indices"""
    try:
        # Major market tickers for sentiment analysis
        market_tickers = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        days = request.args.get('days', default=7, type=int)
        days = min(max(days, 1), 30)
        
        # Get sentiment for market tickers
        sentiment_scores = sentiment_aggregator.get_market_sentiment(market_tickers, days)
        
        # Calculate market sentiment metrics
        scores = [s.score for s in sentiment_scores.values() if s.confidence > 0.3]
        confidences = [s.confidence for s in sentiment_scores.values() if s.confidence > 0.3]
        volumes = [s.volume for s in sentiment_scores.values()]
        
        if scores:
            market_score = sum(score * conf for score, conf in zip(scores, confidences)) / sum(confidences)
            market_confidence = sum(confidences) / len(confidences)
        else:
            market_score = 0.0
            market_confidence = 0.0
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in scores if s > 0.1)
        negative_count = sum(1 for s in scores if s < -0.1)
        neutral_count = len(scores) - positive_count - negative_count
        
        result = {
            'market_sentiment': {
                'score': market_score,
                'confidence': market_confidence,
                'interpretation': _interpret_sentiment(market_score, market_confidence),
                'distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count,
                    'total_sources': sum(volumes)
                }
            },
            'index_sentiment': {
                ticker: {
                    'score': sentiment.score,
                    'confidence': sentiment.confidence,
                    'interpretation': _interpret_sentiment(sentiment.score, sentiment.confidence),
                    'volume': sentiment.volume
                }
                for ticker, sentiment in sentiment_scores.items()
            },
            'analysis_period_days': days,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Market sentiment analysis completed")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        return jsonify({'error': 'Failed to analyze market sentiment'}), 500

@sentiment_bp.route('/sentiment/trending')
@safe_api_call
def get_trending_sentiment():
    """Get sentiment for trending stocks"""
    try:
        # Popular/trending tickers (this could be made dynamic)
        trending_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
        ]
        
        days = request.args.get('days', default=3, type=int)
        days = min(max(days, 1), 7)  # Trending is short-term
        
        # Get sentiment for trending tickers
        sentiment_scores = sentiment_aggregator.get_market_sentiment(trending_tickers, days)
        
        # Sort by sentiment strength (high confidence + extreme sentiment)
        trending_data = []
        for ticker, sentiment in sentiment_scores.items():
            strength = abs(sentiment.score) * sentiment.confidence
            trending_data.append({
                'ticker': ticker,
                'score': sentiment.score,
                'confidence': sentiment.confidence,
                'strength': strength,
                'interpretation': _interpret_sentiment(sentiment.score, sentiment.confidence),
                'volume': sentiment.volume,
                'last_updated': sentiment.timestamp.isoformat()
            })
        
        # Sort by strength
        trending_data.sort(key=lambda x: x['strength'], reverse=True)
        
        result = {
            'trending_sentiment': trending_data,
            'analysis_period_days': days,
            'generated_at': datetime.now().isoformat(),
            'disclaimer': 'Sentiment analysis is for informational purposes only and should not be used as sole investment criteria'
        }
        
        logger.info("Trending sentiment analysis completed")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting trending sentiment: {e}")
        return jsonify({'error': 'Failed to analyze trending sentiment'}), 500

@sentiment_bp.route('/sentiment/sources')
@safe_api_call
def get_sentiment_sources():
    """Get information about sentiment data sources"""
    try:
        sources_info = {
            'news_sources': {
                'newsapi': {
                    'name': 'News API',
                    'description': 'Global news articles from 80,000+ sources',
                    'enabled': bool(sentiment_aggregator.news_collector.news_api_key),
                    'coverage': 'Global'
                },
                'alpha_vantage': {
                    'name': 'Alpha Vantage News Sentiment',
                    'description': 'Financial news with pre-calculated sentiment scores',
                    'enabled': bool(sentiment_aggregator.news_collector.alpha_vantage_key),
                    'coverage': 'US Markets'
                },
                'rss_feeds': {
                    'name': 'Financial RSS Feeds',
                    'description': 'Reuters, MarketWatch, Bloomberg RSS feeds',
                    'enabled': True,
                    'coverage': 'Global Financial News'
                }
            },
            'social_sources': {
                'twitter': {
                    'name': 'Twitter/X',
                    'description': 'Real-time social media sentiment',
                    'enabled': sentiment_aggregator.social_collector.twitter_available,
                    'coverage': 'Global Social Media'
                }
            },
            'analysis_methods': {
                'finbert': {
                    'name': 'FinBERT',
                    'description': 'Financial domain-specific BERT model',
                    'enabled': sentiment_aggregator.sentiment_analyzer.finbert_available,
                    'accuracy': 'High for financial content'
                },
                'textblob': {
                    'name': 'TextBlob',
                    'description': 'General-purpose sentiment analysis',
                    'enabled': True,
                    'accuracy': 'Moderate fallback method'
                }
            },
            'update_frequency': {
                'real_time': 'Social media data',
                'hourly': 'Cached results refresh',
                'daily': 'Historical analysis'
            },
            'limitations': [
                'Sentiment analysis is not financial advice',
                'Results may vary based on data availability',
                'Social media data requires API access',
                'News articles may have publication delays'
            ]
        }
        
        return jsonify(sources_info)
        
    except Exception as e:
        logger.error(f"Error getting sentiment sources info: {e}")
        return jsonify({'error': 'Failed to get sources information'}), 500

def _interpret_sentiment(score: float, confidence: float) -> Dict[str, Any]:
    """Interpret sentiment score with human-readable description"""
    
    # Confidence levels
    if confidence < 0.3:
        confidence_level = 'low'
    elif confidence < 0.7:
        confidence_level = 'moderate'
    else:
        confidence_level = 'high'
    
    # Sentiment interpretation
    if score > 0.3:
        sentiment_label = 'bullish'
        description = 'Positive sentiment suggests optimism'
        color = 'green'
        signal = 'buy_interest'
    elif score > 0.1:
        sentiment_label = 'slightly_positive'
        description = 'Mildly positive sentiment'
        color = 'light_green'
        signal = 'weak_buy'
    elif score > -0.1:
        sentiment_label = 'neutral'
        description = 'Mixed or neutral sentiment'
        color = 'gray'
        signal = 'hold'
    elif score > -0.3:
        sentiment_label = 'slightly_negative'
        description = 'Mildly negative sentiment'
        color = 'light_red'
        signal = 'weak_sell'
    else:
        sentiment_label = 'bearish'
        description = 'Negative sentiment suggests concern'
        color = 'red'
        signal = 'sell_pressure'
    
    return {
        'label': sentiment_label,
        'description': description,
        'confidence_level': confidence_level,
        'color': color,
        'signal': signal,
        'numeric_score': score,
        'confidence_score': confidence
    }