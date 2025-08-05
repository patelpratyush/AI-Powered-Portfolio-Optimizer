#!/usr/bin/env python3
"""
Buy/Sell Recommendation Engine
Advanced AI-powered trading advisor using multiple models and technical analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import ta
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class RecommendationReason:
    category: str
    indicator: str
    value: float
    threshold: float
    weight: float
    description: str
    bullish: bool

@dataclass
class TradingRecommendation:
    ticker: str
    signal: Signal
    confidence: float  # 0-100
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str  # "short", "medium", "long"
    reasons: List[RecommendationReason]
    risk_level: str  # "low", "medium", "high"
    expected_return: float
    max_downside: float
    summary: str

class BuySellAdvisor:
    def __init__(self):
        self.technical_weights = {
            'momentum': 0.25,
            'trend': 0.30,
            'volatility': 0.15,
            'volume': 0.10,
            'support_resistance': 0.20
        }
        
        self.fundamental_weights = {
            'valuation': 0.40,
            'growth': 0.30,
            'profitability': 0.20,
            'financial_health': 0.10
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            Signal.STRONG_BUY: 80,
            Signal.BUY: 60,
            Signal.HOLD: 40,
            Signal.SELL: 20,
            Signal.STRONG_SELL: 0
        }
    
    def get_technical_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[float, List[RecommendationReason]]:
        """Perform comprehensive technical analysis"""
        reasons = []
        
        if df.empty or len(df) < 50:
            return 50.0, reasons  # Neutral if insufficient data
        
        current_price = df['Close'].iloc[-1]
        
        # 1. Momentum Indicators
        rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
        stoch_k = ta.momentum.stoch(df['High'], df['Low'], df['Close']).iloc[-1]
        williams_r = ta.momentum.williams_r(df['High'], df['Low'], df['Close']).iloc[-1]
        
        # RSI Analysis
        if rsi < 30:
            reasons.append(RecommendationReason(
                category="momentum",
                indicator="RSI",
                value=rsi,
                threshold=30,
                weight=0.8,
                description=f"RSI at {rsi:.1f} indicates oversold condition",
                bullish=True
            ))
        elif rsi > 70:
            reasons.append(RecommendationReason(
                category="momentum",
                indicator="RSI",
                value=rsi,
                threshold=70,
                weight=0.8,
                description=f"RSI at {rsi:.1f} indicates overbought condition",
                bullish=False
            ))
        
        # 2. Trend Indicators
        # Moving Averages
        sma_20 = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
        sma_50 = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1]
        ema_12 = ta.trend.ema_indicator(df['Close'], window=12).iloc[-1]
        ema_26 = ta.trend.ema_indicator(df['Close'], window=26).iloc[-1]
        
        # Price vs Moving Averages
        if current_price > sma_20 > sma_50:
            reasons.append(RecommendationReason(
                category="trend",
                indicator="Moving Average Alignment",
                value=current_price,
                threshold=sma_20,
                weight=0.9,
                description="Price above both 20-day and 50-day SMA in bullish alignment",
                bullish=True
            ))
        elif current_price < sma_20 < sma_50:
            reasons.append(RecommendationReason(
                category="trend",
                indicator="Moving Average Alignment",
                value=current_price,
                threshold=sma_20,
                weight=0.9,
                description="Price below both 20-day and 50-day SMA in bearish alignment",
                bullish=False
            ))
        
        # MACD
        macd_line = ta.trend.macd_diff(df['Close']).iloc[-1]
        macd_signal = ta.trend.macd_signal(df['Close']).iloc[-1]
        
        if macd_line > macd_signal and macd_line > 0:
            reasons.append(RecommendationReason(
                category="trend",
                indicator="MACD",
                value=macd_line,
                threshold=macd_signal,
                weight=0.7,
                description="MACD line above signal line and zero - bullish momentum",
                bullish=True
            ))
        elif macd_line < macd_signal and macd_line < 0:
            reasons.append(RecommendationReason(
                category="trend",
                indicator="MACD",
                value=macd_line,
                threshold=macd_signal,
                weight=0.7,
                description="MACD line below signal line and zero - bearish momentum",
                bullish=False
            ))
        
        # 3. Volatility Analysis
        bb_upper = ta.volatility.bollinger_hband(df['Close']).iloc[-1]
        bb_lower = ta.volatility.bollinger_lband(df['Close']).iloc[-1]
        bb_mid = ta.volatility.bollinger_mavg(df['Close']).iloc[-1]
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position < 0.2:
            reasons.append(RecommendationReason(
                category="volatility",
                indicator="Bollinger Bands",
                value=bb_position,
                threshold=0.2,
                weight=0.6,
                description="Price near lower Bollinger Band - potential bounce opportunity",
                bullish=True
            ))
        elif bb_position > 0.8:
            reasons.append(RecommendationReason(
                category="volatility",
                indicator="Bollinger Bands",
                value=bb_position,
                threshold=0.8,
                weight=0.6,
                description="Price near upper Bollinger Band - potential resistance",
                bullish=False
            ))
        
        # 4. Volume Analysis
        volume_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
        
        if volume_ratio > 1.5:
            price_change = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            if price_change > 0:
                reasons.append(RecommendationReason(
                    category="volume",
                    indicator="Volume Surge",
                    value=volume_ratio,
                    threshold=1.5,
                    weight=0.8,
                    description=f"High volume ({volume_ratio:.1f}x average) with price increase",
                    bullish=True
                ))
            else:
                reasons.append(RecommendationReason(
                    category="volume",
                    indicator="Volume Surge",
                    value=volume_ratio,
                    threshold=1.5,
                    weight=0.8,
                    description=f"High volume ({volume_ratio:.1f}x average) with price decrease",
                    bullish=False
                ))
        
        # 5. Support and Resistance
        high_20 = df['High'].rolling(20).max().iloc[-1]
        low_20 = df['Low'].rolling(20).min().iloc[-1]
        
        resistance_distance = (high_20 - current_price) / current_price
        support_distance = (current_price - low_20) / current_price
        
        if support_distance < 0.02:  # Within 2% of support
            reasons.append(RecommendationReason(
                category="support_resistance",
                indicator="Support Level",
                value=current_price,
                threshold=low_20,
                weight=0.7,
                description=f"Price near 20-day support level at ${low_20:.2f}",
                bullish=True
            ))
        elif resistance_distance < 0.02:  # Within 2% of resistance
            reasons.append(RecommendationReason(
                category="support_resistance",
                indicator="Resistance Level",
                value=current_price,
                threshold=high_20,
                weight=0.7,
                description=f"Price near 20-day resistance level at ${high_20:.2f}",
                bullish=False
            ))
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(reasons)
        
        return technical_score, reasons
    
    def get_fundamental_analysis(self, ticker: str) -> Tuple[float, List[RecommendationReason]]:
        """Perform fundamental analysis using available data"""
        reasons = []
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return 50.0, reasons  # Neutral if no data
            
            # 1. Valuation Metrics
            pe_ratio = info.get('trailingPE', None)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    reasons.append(RecommendationReason(
                        category="valuation",
                        indicator="P/E Ratio",
                        value=pe_ratio,
                        threshold=15,
                        weight=0.8,
                        description=f"Low P/E ratio of {pe_ratio:.1f} suggests undervaluation",
                        bullish=True
                    ))
                elif pe_ratio > 30:
                    reasons.append(RecommendationReason(
                        category="valuation",
                        indicator="P/E Ratio",
                        value=pe_ratio,
                        threshold=30,
                        weight=0.8,
                        description=f"High P/E ratio of {pe_ratio:.1f} suggests overvaluation",
                        bullish=False
                    ))
            
            # 2. Growth Metrics
            revenue_growth = info.get('revenueGrowth', None)
            if revenue_growth and revenue_growth > 0.1:  # 10% growth
                reasons.append(RecommendationReason(
                    category="growth",
                    indicator="Revenue Growth",
                    value=revenue_growth,
                    threshold=0.1,
                    weight=0.9,
                    description=f"Strong revenue growth of {revenue_growth*100:.1f}%",
                    bullish=True
                ))
            elif revenue_growth and revenue_growth < -0.05:  # -5% decline
                reasons.append(RecommendationReason(
                    category="growth",
                    indicator="Revenue Growth",
                    value=revenue_growth,
                    threshold=-0.05,
                    weight=0.9,
                    description=f"Revenue declining by {abs(revenue_growth*100):.1f}%",
                    bullish=False
                ))
            
            # 3. Profitability
            profit_margin = info.get('profitMargins', None)
            if profit_margin and profit_margin > 0.15:  # 15% margin
                reasons.append(RecommendationReason(
                    category="profitability",
                    indicator="Profit Margin",
                    value=profit_margin,
                    threshold=0.15,
                    weight=0.7,
                    description=f"High profit margin of {profit_margin*100:.1f}%",
                    bullish=True
                ))
            elif profit_margin and profit_margin < 0.05:  # 5% margin
                reasons.append(RecommendationReason(
                    category="profitability",
                    indicator="Profit Margin",
                    value=profit_margin,
                    threshold=0.05,
                    weight=0.7,
                    description=f"Low profit margin of {profit_margin*100:.1f}%",
                    bullish=False
                ))
            
            # 4. Financial Health
            debt_to_equity = info.get('debtToEquity', None)
            if debt_to_equity:
                if debt_to_equity < 30:  # Low debt
                    reasons.append(RecommendationReason(
                        category="financial_health",
                        indicator="Debt-to-Equity",
                        value=debt_to_equity,
                        threshold=30,
                        weight=0.6,
                        description=f"Low debt-to-equity ratio of {debt_to_equity:.1f}%",
                        bullish=True
                    ))
                elif debt_to_equity > 100:  # High debt
                    reasons.append(RecommendationReason(
                        category="financial_health",
                        indicator="Debt-to-Equity",
                        value=debt_to_equity,
                        threshold=100,
                        weight=0.6,
                        description=f"High debt-to-equity ratio of {debt_to_equity:.1f}%",
                        bullish=False
                    ))
            
        except Exception as e:
            logger.warning(f"Fundamental analysis failed for {ticker}: {str(e)}")
            return 50.0, reasons
        
        # Calculate fundamental score
        fundamental_score = self._calculate_fundamental_score(reasons)
        
        return fundamental_score, reasons
    
    def _calculate_technical_score(self, reasons: List[RecommendationReason]) -> float:
        """Calculate weighted technical analysis score"""
        if not reasons:
            return 50.0
        
        category_scores = {}
        category_weights = {}
        
        for reason in reasons:
            if reason.category not in category_scores:
                category_scores[reason.category] = []
                category_weights[reason.category] = []
            
            # Convert bullish/bearish to score
            base_score = 75 if reason.bullish else 25
            weighted_score = base_score * reason.weight
            
            category_scores[reason.category].append(weighted_score)
            category_weights[reason.category].append(reason.weight)
        
        # Calculate weighted average for each category
        final_score = 0
        total_weight = 0
        
        for category, scores in category_scores.items():
            weights = category_weights[category]
            category_score = np.average(scores, weights=weights)
            category_weight = self.technical_weights.get(category, 0.1)
            
            final_score += category_score * category_weight
            total_weight += category_weight
        
        return final_score / total_weight if total_weight > 0 else 50.0
    
    def _calculate_fundamental_score(self, reasons: List[RecommendationReason]) -> float:
        """Calculate weighted fundamental analysis score"""
        if not reasons:
            return 50.0
        
        category_scores = {}
        category_weights = {}
        
        for reason in reasons:
            if reason.category not in category_scores:
                category_scores[reason.category] = []
                category_weights[reason.category] = []
            
            # Convert bullish/bearish to score
            base_score = 75 if reason.bullish else 25
            weighted_score = base_score * reason.weight
            
            category_scores[reason.category].append(weighted_score)
            category_weights[reason.category].append(reason.weight)
        
        # Calculate weighted average for each category
        final_score = 0
        total_weight = 0
        
        for category, scores in category_scores.items():
            weights = category_weights[category]
            category_score = np.average(scores, weights=weights)
            category_weight = self.fundamental_weights.get(category, 0.1)
            
            final_score += category_score * category_weight
            total_weight += category_weight
        
        return final_score / total_weight if total_weight > 0 else 50.0
    
    def generate_recommendation(self, ticker: str, ml_predictions: Optional[Dict] = None) -> TradingRecommendation:
        """Generate comprehensive buy/sell recommendation"""
        logger.info(f"Generating recommendation for {ticker}")
        
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")  # 3 months for analysis
            
            if df.empty:
                raise ValueError(f"No data available for {ticker}")
            
            current_price = df['Close'].iloc[-1]
            
            # Technical Analysis
            technical_score, technical_reasons = self.get_technical_analysis(df, ticker)
            
            # Fundamental Analysis
            fundamental_score, fundamental_reasons = self.get_fundamental_analysis(ticker)
            
            # Combine ML predictions if available
            ml_score = 50.0
            ml_reasons = []
            
            if ml_predictions:
                # Extract insights from ML predictions
                avg_return = ml_predictions.get('summary', {}).get('avg_predicted_return', 0)
                signal_action = ml_predictions.get('trading_signal', {}).get('action', 'HOLD')
                signal_strength = ml_predictions.get('trading_signal', {}).get('strength', 0.5)
                
                if signal_action == 'BUY':
                    ml_score = 50 + (signal_strength * 40)  # 50-90 range
                elif signal_action == 'SELL':
                    ml_score = 50 - (signal_strength * 40)  # 10-50 range
                else:
                    ml_score = 50  # HOLD
                
                ml_reasons.append(RecommendationReason(
                    category="ai_prediction",
                    indicator="ML Ensemble",
                    value=avg_return,
                    threshold=0.02,
                    weight=1.0,
                    description=f"AI models predict {avg_return*100:.1f}% return with {signal_strength*100:.0f}% confidence",
                    bullish=avg_return > 0.02
                ))
            
            # Calculate overall score
            weights = {'technical': 0.4, 'fundamental': 0.3, 'ml': 0.3}
            if not ml_predictions:
                weights = {'technical': 0.6, 'fundamental': 0.4, 'ml': 0.0}
            
            overall_score = (
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                ml_score * weights['ml']
            )
            
            # Determine signal
            signal = Signal.HOLD
            for sig, threshold in sorted(self.signal_thresholds.items(), key=lambda x: x[1], reverse=True):
                if overall_score >= threshold:
                    signal = sig
                    break
            
            # Calculate confidence
            confidence = min(95, max(5, overall_score))
            
            # Estimate target price and stop loss
            target_price = None
            stop_loss = None
            
            if signal in [Signal.BUY, Signal.STRONG_BUY]:
                # Target price based on technical resistance or ML prediction
                if ml_predictions:
                    max_predicted = ml_predictions.get('summary', {}).get('max_predicted_price', current_price)
                    target_price = max_predicted
                else:
                    # Use technical resistance
                    high_20 = df['High'].rolling(20).max().iloc[-1]
                    target_price = max(current_price * 1.1, high_20)
                
                # Stop loss at recent support
                low_10 = df['Low'].rolling(10).min().iloc[-1]
                stop_loss = max(current_price * 0.95, low_10)
                
            elif signal in [Signal.SELL, Signal.STRONG_SELL]:
                # Target price (downside)
                if ml_predictions:
                    min_predicted = ml_predictions.get('summary', {}).get('min_predicted_price', current_price)
                    target_price = min_predicted
                else:
                    low_20 = df['Low'].rolling(20).min().iloc[-1]
                    target_price = min(current_price * 0.9, low_20)
                
                # Stop loss at recent resistance
                high_10 = df['High'].rolling(10).max().iloc[-1]
                stop_loss = min(current_price * 1.05, high_10)
            
            # Risk assessment
            volatility = df['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            
            if volatility < 0.2:
                risk_level = "low"
            elif volatility < 0.4:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Expected return and downside
            if target_price:
                expected_return = (target_price - current_price) / current_price
            else:
                expected_return = 0.0
            
            if stop_loss:
                max_downside = (stop_loss - current_price) / current_price
            else:
                max_downside = -volatility * 0.5  # Approximate downside based on volatility
            
            # Generate summary
            all_reasons = technical_reasons + fundamental_reasons + ml_reasons
            summary = self._generate_summary(signal, confidence, all_reasons, ticker)
            
            # Time horizon
            time_horizon = "short" if ml_predictions else "medium"
            
            return TradingRecommendation(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                time_horizon=time_horizon,
                reasons=all_reasons,
                risk_level=risk_level,
                expected_return=expected_return,
                max_downside=max_downside,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed for {ticker}: {str(e)}")
            # Return neutral recommendation on error
            return TradingRecommendation(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=50.0,
                target_price=None,
                stop_loss=None,
                time_horizon="medium",
                reasons=[],
                risk_level="medium",
                expected_return=0.0,
                max_downside=-0.1,
                summary=f"Unable to analyze {ticker} due to insufficient data"
            )
    
    def _generate_summary(self, signal: Signal, confidence: float, reasons: List[RecommendationReason], ticker: str) -> str:
        """Generate human-readable summary of recommendation"""
        
        # Count bullish vs bearish reasons
        bullish_reasons = [r for r in reasons if r.bullish]
        bearish_reasons = [r for r in reasons if not r.bullish]
        
        # Signal description
        signal_desc = {
            Signal.STRONG_BUY: "Strong Buy",
            Signal.BUY: "Buy",
            Signal.HOLD: "Hold",
            Signal.SELL: "Sell",
            Signal.STRONG_SELL: "Strong Sell"
        }
        
        summary = f"{signal_desc[signal]} recommendation for {ticker} with {confidence:.0f}% confidence. "
        
        if len(bullish_reasons) > len(bearish_reasons):
            summary += f"Analysis shows {len(bullish_reasons)} bullish indicators vs {len(bearish_reasons)} bearish. "
            if bullish_reasons:
                top_reason = max(bullish_reasons, key=lambda x: x.weight)
                summary += f"Key positive: {top_reason.description}. "
        elif len(bearish_reasons) > len(bullish_reasons):
            summary += f"Analysis shows {len(bearish_reasons)} bearish indicators vs {len(bullish_reasons)} bullish. "
            if bearish_reasons:
                top_reason = max(bearish_reasons, key=lambda x: x.weight)
                summary += f"Key concern: {top_reason.description}. "
        else:
            summary += "Mixed signals from technical and fundamental analysis suggest a neutral stance. "
        
        return summary

if __name__ == "__main__":
    # Test the advisor
    advisor = BuySellAdvisor()
    
    test_ticker = "AAPL"
    recommendation = advisor.generate_recommendation(test_ticker)
    
    print(f"Recommendation for {test_ticker}:")
    print(f"Signal: {recommendation.signal.value}")
    print(f"Confidence: {recommendation.confidence:.1f}%")
    print(f"Target Price: ${recommendation.target_price:.2f}" if recommendation.target_price else "Target Price: N/A")
    print(f"Stop Loss: ${recommendation.stop_loss:.2f}" if recommendation.stop_loss else "Stop Loss: N/A")
    print(f"Risk Level: {recommendation.risk_level}")
    print(f"Expected Return: {recommendation.expected_return:.2%}")
    print(f"Summary: {recommendation.summary}")
    
    print(f"\nTop Reasons ({len(recommendation.reasons)}):")
    for reason in sorted(recommendation.reasons, key=lambda x: x.weight, reverse=True)[:5]:
        print(f"- {reason.description} ({reason.category}, weight: {reason.weight})")