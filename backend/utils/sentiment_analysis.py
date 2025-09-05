#!/usr/bin/env python3
"""
Sentiment Analysis for Stock Predictions
Integrates news and social media sentiment into ML models
"""
import os
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import feedparser
import tweepy
import redis
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger('portfolio_optimizer.sentiment')

@dataclass
class SentimentScore:
    """Sentiment score with confidence and metadata"""
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    text_sample: str
    volume: int  # number of articles/posts

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    url: str
    published: datetime
    source: str
    ticker: str

class FinancialSentimentAnalyzer:
    """Advanced sentiment analysis for financial content"""
    
    def __init__(self, cache_client: Optional[redis.Redis] = None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.sentiment.analyzer')
        
        # Initialize financial sentiment model (FinBERT)
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert", cache_dir="./models/finbert"
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert", cache_dir="./models/finbert"
            )
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=-1  # CPU
            )
            self.finbert_available = True
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.warning(f"FinBERT not available, using TextBlob: {e}")
            self.finbert_available = False
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text"""
        try:
            if self.finbert_available and len(text) > 10:
                # Use FinBERT for financial sentiment
                result = self.finbert_pipeline(text[:512])  # Truncate for BERT
                
                # Convert FinBERT output to standardized format
                label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                sentiment_score = label_map.get(result[0]['label'].lower(), 0.0)
                confidence = result[0]['score']
                
                return {
                    'score': sentiment_score,
                    'confidence': confidence,
                    'method': 'finbert',
                    'raw_result': result[0]
                }
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert subjectivity to confidence (inverse)
                confidence = 1.0 - subjectivity if subjectivity <= 1.0 else 0.0
                
                return {
                    'score': polarity,
                    'confidence': confidence,
                    'method': 'textblob',
                    'raw_result': {'polarity': polarity, 'subjectivity': subjectivity}
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'method': 'error',
                'raw_result': {'error': str(e)}
            }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts efficiently"""
        results = []
        
        if self.finbert_available and len(texts) > 5:
            try:
                # Batch process with FinBERT
                truncated_texts = [text[:512] for text in texts]
                finbert_results = self.finbert_pipeline(truncated_texts)
                
                label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                
                for i, result in enumerate(finbert_results):
                    sentiment_score = label_map.get(result['label'].lower(), 0.0)
                    confidence = result['score']
                    
                    results.append({
                        'score': sentiment_score,
                        'confidence': confidence,
                        'method': 'finbert_batch',
                        'raw_result': result
                    })
                    
            except Exception as e:
                self.logger.error(f"Batch FinBERT analysis failed: {e}")
                # Fallback to individual analysis
                for text in texts:
                    results.append(self.analyze_text(text))
        else:
            # Process individually
            for text in texts:
                results.append(self.analyze_text(text))
        
        return results


class NewsDataCollector:
    """Collect news data from multiple sources"""
    
    def __init__(self, cache_client: Optional[redis.Redis] = None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.sentiment.news')
        
        # News API configuration
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Rate limiting
        self.last_api_call = {}
        self.api_rate_limits = {
            'newsapi': 1.0,  # 1 second between calls
            'alpha_vantage': 12.0,  # 12 seconds (5 calls per minute)
            'rss': 0.5  # 0.5 seconds between RSS feeds
        }
    
    def _rate_limit(self, service: str):
        """Implement rate limiting for APIs"""
        if service in self.last_api_call:
            elapsed = time.time() - self.last_api_call[service]
            required_wait = self.api_rate_limits.get(service, 1.0)
            
            if elapsed < required_wait:
                time.sleep(required_wait - elapsed)
        
        self.last_api_call[service] = time.time()
    
    def get_news_api_articles(self, ticker: str, days: int = 7) -> List[NewsArticle]:
        """Get articles from News API"""
        if not self.news_api_key:
            self.logger.warning("News API key not configured")
            return []
        
        try:
            self._rate_limit('newsapi')
            
            # Get company name for better search
            company_info = self._get_company_info(ticker)
            query = f"{ticker} OR {company_info.get('name', ticker)}"
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                if article.get('title') and article.get('description'):
                    articles.append(NewsArticle(
                        title=article['title'],
                        content=article.get('description', '') + ' ' + article.get('content', ''),
                        url=article.get('url', ''),
                        published=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                        source=article.get('source', {}).get('name', 'NewsAPI'),
                        ticker=ticker
                    ))
            
            self.logger.info(f"Retrieved {len(articles)} articles for {ticker} from News API")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching News API articles for {ticker}: {e}")
            return []
    
    def get_alpha_vantage_news(self, ticker: str) -> List[NewsArticle]:
        """Get news from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            self.logger.warning("Alpha Vantage API key not configured")
            return []
        
        try:
            self._rate_limit('alpha_vantage')
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data.get('feed', []):
                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('summary', ''),
                    url=item.get('url', ''),
                    published=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                    source=item.get('source', 'Alpha Vantage'),
                    ticker=ticker
                ))
            
            self.logger.info(f"Retrieved {len(articles)} articles for {ticker} from Alpha Vantage")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage news for {ticker}: {e}")
            return []
    
    def get_rss_feeds(self, ticker: str) -> List[NewsArticle]:
        """Get articles from financial RSS feeds"""
        rss_feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://www.marketwatch.com/rss/topstories",
            "https://feeds.bloomberg.com/markets/news.rss"
        ]
        
        articles = []
        
        for feed_url in rss_feeds:
            try:
                self._rate_limit('rss')
                
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Limit per feed
                    # Filter for ticker relevance
                    content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    if ticker.lower() in content.lower():
                        articles.append(NewsArticle(
                            title=entry.get('title', ''),
                            content=entry.get('summary', ''),
                            url=entry.get('link', ''),
                            published=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                            source=feed.feed.get('title', 'RSS Feed'),
                            ticker=ticker
                        ))
                
            except Exception as e:
                self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                continue
        
        self.logger.info(f"Retrieved {len(articles)} articles for {ticker} from RSS feeds")
        return articles
    
    def _get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information for better news search"""
        cache_key = f"company_info:{ticker}"
        
        if self.cache_client:
            cached = self.cache_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_data = {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', '')
            }
            
            if self.cache_client:
                self.cache_client.setex(cache_key, 86400, json.dumps(company_data))  # Cache for 1 day
            
            return company_data
            
        except Exception as e:
            self.logger.error(f"Error getting company info for {ticker}: {e}")
            return {'name': ticker}


class SocialMediaCollector:
    """Collect sentiment from social media (Twitter/X)"""
    
    def __init__(self, cache_client: Optional[redis.Redis] = None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.sentiment.social')
        
        # Twitter API v2 credentials
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if self.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
                self.twitter_available = True
                self.logger.info("Twitter API client initialized")
            except Exception as e:
                self.logger.warning(f"Twitter API not available: {e}")
                self.twitter_available = False
        else:
            self.twitter_available = False
    
    def get_twitter_sentiment(self, ticker: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent tweets about a ticker"""
        if not self.twitter_available:
            return []
        
        try:
            # Search for tweets mentioning the ticker
            query = f"${ticker} OR {ticker} -is:retweet lang:en"
            
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                max_results=min(count, 100),
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            ).flatten(limit=count)
            
            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics,
                    'id': tweet.id
                })
            
            self.logger.info(f"Retrieved {len(tweet_data)} tweets for {ticker}")
            return tweet_data
            
        except Exception as e:
            self.logger.error(f"Error fetching tweets for {ticker}: {e}")
            return []


class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self, cache_client: Optional[redis.Redis] = None):
        self.cache_client = cache_client
        self.logger = logging.getLogger('portfolio_optimizer.sentiment.aggregator')
        
        self.sentiment_analyzer = FinancialSentimentAnalyzer(cache_client)
        self.news_collector = NewsDataCollector(cache_client)
        self.social_collector = SocialMediaCollector(cache_client)
    
    def get_ticker_sentiment(self, ticker: str, days: int = 7) -> SentimentScore:
        """Get comprehensive sentiment analysis for a ticker"""
        cache_key = f"sentiment:{ticker}:{days}"
        
        # Check cache first
        if self.cache_client:
            cached = self.cache_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return SentimentScore(**data)
        
        try:
            all_texts = []
            all_sources = []
            all_timestamps = []
            
            # Collect news articles
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.news_collector.get_news_api_articles, ticker, days): 'newsapi',
                    executor.submit(self.news_collector.get_alpha_vantage_news, ticker): 'alphavantage',
                    executor.submit(self.news_collector.get_rss_feeds, ticker): 'rss'
                }
                
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        articles = future.result(timeout=30)
                        for article in articles:
                            text = f"{article.title} {article.content}"
                            all_texts.append(text)
                            all_sources.append(f"news_{source}")
                            all_timestamps.append(article.published)
                    except Exception as e:
                        self.logger.error(f"Error processing {source}: {e}")
            
            # Collect social media sentiment
            try:
                tweets = self.social_collector.get_twitter_sentiment(ticker, 50)
                for tweet in tweets:
                    all_texts.append(tweet['text'])
                    all_sources.append('twitter')
                    all_timestamps.append(tweet['created_at'])
            except Exception as e:
                self.logger.error(f"Error collecting social media sentiment: {e}")
            
            if not all_texts:
                self.logger.warning(f"No sentiment data found for {ticker}")
                return SentimentScore(
                    score=0.0,
                    confidence=0.0,
                    source='none',
                    timestamp=datetime.now(),
                    text_sample='',
                    volume=0
                )
            
            # Analyze all texts
            sentiment_results = self.sentiment_analyzer.batch_analyze(all_texts)
            
            # Aggregate results with time weighting (more recent = higher weight)
            total_weighted_score = 0.0
            total_weight = 0.0
            confidence_scores = []
            
            now = datetime.now()
            for i, (result, timestamp) in enumerate(zip(sentiment_results, all_timestamps)):
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=None)
                    
                # Time decay weight (more recent = higher weight)
                days_old = (now.replace(tzinfo=None) - timestamp.replace(tzinfo=None)).days
                time_weight = max(0.1, 1.0 - (days_old / (days * 2)))  # Decay over 2x the period
                
                # Source reliability weight
                source_weight = {
                    'news_newsapi': 1.0,
                    'news_alphavantage': 0.9,
                    'news_rss': 0.8,
                    'twitter': 0.6
                }.get(all_sources[i], 0.5)
                
                final_weight = time_weight * source_weight * result['confidence']
                
                total_weighted_score += result['score'] * final_weight
                total_weight += final_weight
                confidence_scores.append(result['confidence'])
            
            # Calculate final sentiment
            if total_weight > 0:
                final_score = total_weighted_score / total_weight
                final_confidence = np.mean(confidence_scores)
            else:
                final_score = 0.0
                final_confidence = 0.0
            
            # Create result
            sentiment_result = SentimentScore(
                score=final_score,
                confidence=final_confidence,
                source=f"aggregated_{len(all_texts)}_sources",
                timestamp=datetime.now(),
                text_sample=all_texts[0][:100] + "..." if all_texts else "",
                volume=len(all_texts)
            )
            
            # Cache result
            if self.cache_client:
                cache_data = {
                    'score': sentiment_result.score,
                    'confidence': sentiment_result.confidence,
                    'source': sentiment_result.source,
                    'timestamp': sentiment_result.timestamp.isoformat(),
                    'text_sample': sentiment_result.text_sample,
                    'volume': sentiment_result.volume
                }
                self.cache_client.setex(cache_key, 3600, json.dumps(cache_data))  # Cache for 1 hour
            
            self.logger.info(f"Sentiment analysis completed for {ticker}: score={final_score:.3f}, confidence={final_confidence:.3f}, volume={len(all_texts)}")
            return sentiment_result
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {ticker}: {e}")
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                source='error',
                timestamp=datetime.now(),
                text_sample=str(e),
                volume=0
            )
    
    def get_market_sentiment(self, tickers: List[str], days: int = 7) -> Dict[str, SentimentScore]:
        """Get sentiment for multiple tickers"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self.get_ticker_sentiment, ticker, days): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result(timeout=60)
                except Exception as e:
                    self.logger.error(f"Error getting sentiment for {ticker}: {e}")
                    results[ticker] = SentimentScore(
                        score=0.0,
                        confidence=0.0,
                        source='error',
                        timestamp=datetime.now(),
                        text_sample=str(e),
                        volume=0
                    )
        
        return results


def create_sentiment_features(sentiment_scores: Dict[str, SentimentScore]) -> Dict[str, float]:
    """Create features for ML models from sentiment data"""
    features = {}
    
    for ticker, sentiment in sentiment_scores.items():
        prefix = f"{ticker}_sentiment"
        
        features.update({
            f"{prefix}_score": sentiment.score,
            f"{prefix}_confidence": sentiment.confidence,
            f"{prefix}_volume": min(sentiment.volume / 100.0, 1.0),  # Normalize volume
            f"{prefix}_weighted": sentiment.score * sentiment.confidence,
            f"{prefix}_signal": 1.0 if sentiment.score > 0.1 and sentiment.confidence > 0.6 else (
                -1.0 if sentiment.score < -0.1 and sentiment.confidence > 0.6 else 0.0
            )
        })
    
    # Market-wide sentiment features
    if sentiment_scores:
        all_scores = [s.score for s in sentiment_scores.values()]
        all_confidences = [s.confidence for s in sentiment_scores.values()]
        all_volumes = [s.volume for s in sentiment_scores.values()]
        
        features.update({
            'market_sentiment_mean': np.mean(all_scores),
            'market_sentiment_std': np.std(all_scores),
            'market_confidence_mean': np.mean(all_confidences),
            'market_volume_total': sum(all_volumes),
            'market_positive_ratio': sum(1 for s in all_scores if s > 0.1) / len(all_scores),
            'market_negative_ratio': sum(1 for s in all_scores if s < -0.1) / len(all_scores)
        })
    
    return features