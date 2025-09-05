#!/usr/bin/env python3
"""
Portfolio Backtesting System
Historical performance simulation with different strategies
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization imports
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

logger = logging.getLogger('portfolio_optimizer.backtesting')

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_cost: float = 0.001  # 0.1% per trade
    benchmark: str = 'SPY'
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    value_at_risk_95: float
    conditional_var_95: float

@dataclass
class BacktestResult:
    """Complete backtesting result"""
    config: BacktestConfig
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    drawdown_periods: List[Dict[str, Any]]
    monthly_returns: pd.Series
    yearly_returns: pd.Series

class PortfolioStrategy:
    """Base class for portfolio strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_weights(self, 
                        data: pd.DataFrame, 
                        current_date: datetime,
                        **kwargs) -> Dict[str, float]:
        """Generate portfolio weights for given date"""
        raise NotImplementedError
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime, frequency: str) -> bool:
        """Check if portfolio should be rebalanced"""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif frequency == 'monthly':
            return current_date.month != last_rebalance.month
        elif frequency == 'quarterly':
            return (current_date.month - 1) // 3 != (last_rebalance.month - 1) // 3
        return False

class EqualWeightStrategy(PortfolioStrategy):
    """Equal weight portfolio strategy"""
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        n_assets = len(data.columns)
        return {col: 1.0 / n_assets for col in data.columns}

class MarketCapWeightStrategy(PortfolioStrategy):
    """Market capitalization weighted strategy"""
    
    def __init__(self):
        super().__init__("Market Cap Weight")
        self.market_caps = {}
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        # Get market caps (simplified - in practice would need real market cap data)
        total_cap = 0
        weights = {}
        
        for ticker in data.columns:
            if ticker not in self.market_caps:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    self.market_caps[ticker] = info.get('marketCap', 1e9)  # Default 1B if not available
                except:
                    self.market_caps[ticker] = 1e9
            
            total_cap += self.market_caps[ticker]
        
        for ticker in data.columns:
            weights[ticker] = self.market_caps[ticker] / total_cap
        
        return weights

class MeanReversionStrategy(PortfolioStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Mean Reversion")
        self.lookback_period = lookback_period
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        # Calculate rolling means and current deviations
        rolling_mean = data.rolling(window=self.lookback_period).mean()
        current_prices = data.iloc[-1]
        mean_prices = rolling_mean.iloc[-1]
        
        # Calculate deviations (negative means undervalued, positive means overvalued)
        deviations = (current_prices - mean_prices) / mean_prices
        
        # Invert deviations for mean reversion (buy undervalued, sell overvalued)
        scores = -deviations
        
        # Convert to weights (softmax-like normalization)
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        weights_array = exp_scores / exp_scores.sum()
        
        return dict(zip(data.columns, weights_array))

class MomentumStrategy(PortfolioStrategy):
    """Momentum strategy"""
    
    def __init__(self, lookback_period: int = 60):
        super().__init__("Momentum")
        self.lookback_period = lookback_period
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        # Calculate momentum scores (returns over lookback period)
        returns = data.pct_change().dropna()
        momentum_scores = returns.rolling(window=self.lookback_period).mean().iloc[-1]
        
        # Only invest in positive momentum assets
        positive_momentum = momentum_scores[momentum_scores > 0]
        
        if len(positive_momentum) == 0:
            # Equal weight if no positive momentum
            return {col: 1.0 / len(data.columns) for col in data.columns}
        
        # Weight by momentum strength
        weights_sum = positive_momentum.sum()
        weights = {}
        
        for ticker in data.columns:
            if ticker in positive_momentum.index:
                weights[ticker] = positive_momentum[ticker] / weights_sum
            else:
                weights[ticker] = 0.0
        
        return weights

class MinVarianceStrategy(PortfolioStrategy):
    """Minimum variance strategy"""
    
    def __init__(self, lookback_period: int = 252):
        super().__init__("Minimum Variance")
        self.lookback_period = lookback_period
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        try:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            if len(returns) < self.lookback_period:
                # Fall back to equal weight if not enough data
                n_assets = len(data.columns)
                return {col: 1.0 / n_assets for col in data.columns}
            
            # Use recent data for covariance calculation
            recent_returns = returns.tail(self.lookback_period)
            
            # Calculate covariance matrix
            cov_matrix = risk_models.sample_cov(recent_returns, frequency=252)
            
            # Optimize for minimum variance
            ef = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 1))
            weights = ef.min_volatility()
            
            return ef.clean_weights()
            
        except Exception as e:
            logger.error(f"Min variance optimization failed: {e}")
            # Fall back to equal weight
            n_assets = len(data.columns)
            return {col: 1.0 / n_assets for col in data.columns}

class MaxSharpeStrategy(PortfolioStrategy):
    """Maximum Sharpe ratio strategy"""
    
    def __init__(self, lookback_period: int = 252):
        super().__init__("Max Sharpe")
        self.lookback_period = lookback_period
    
    def generate_weights(self, data: pd.DataFrame, current_date: datetime, **kwargs) -> Dict[str, float]:
        try:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            if len(returns) < self.lookback_period:
                # Fall back to equal weight if not enough data
                n_assets = len(data.columns)
                return {col: 1.0 / n_assets for col in data.columns}
            
            # Use recent data
            recent_returns = returns.tail(self.lookback_period)
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(recent_returns, frequency=252)
            S = risk_models.sample_cov(recent_returns, frequency=252)
            
            # Optimize for maximum Sharpe ratio
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            weights = ef.max_sharpe()
            
            return ef.clean_weights()
            
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {e}")
            # Fall back to equal weight
            n_assets = len(data.columns)
            return {col: 1.0 / n_assets for col in data.columns}

class PortfolioBacktester:
    """Main backtesting engine"""
    
    def __init__(self):
        self.logger = logging.getLogger('portfolio_optimizer.backtesting.engine')
    
    def get_price_data(self, 
                      tickers: List[str], 
                      start_date: datetime, 
                      end_date: datetime,
                      chunk_size: int = 10) -> pd.DataFrame:
        """Get historical price data for multiple tickers"""
        
        def fetch_ticker_data(ticker: str) -> pd.Series:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if hist.empty:
                    self.logger.warning(f"No data available for {ticker}")
                    return None
                return hist['Close'].rename(ticker)
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {e}")
                return None
        
        # Fetch data in parallel
        all_data = []
        with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
            future_to_ticker = {executor.submit(fetch_ticker_data, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_data.append(data)
                except Exception as e:
                    self.logger.error(f"Error processing {ticker}: {e}")
        
        if not all_data:
            raise ValueError("No price data available for any ticker")
        
        # Combine all data
        price_data = pd.concat(all_data, axis=1)
        
        # Forward fill missing values and drop rows with all NaN
        price_data = price_data.fillna(method='ffill').dropna()
        
        self.logger.info(f"Loaded price data for {len(price_data.columns)} tickers from {start_date} to {end_date}")
        return price_data
    
    def calculate_performance_metrics(self, 
                                    returns: pd.Series, 
                                    benchmark_returns: pd.Series,
                                    risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0
        
        # Beta and Alpha (relative to benchmark)
        if len(benchmark_returns) > 0 and len(returns) > 0:
            # Align returns
            aligned_returns = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
            if len(aligned_returns) > 1:
                portfolio_ret = aligned_returns.iloc[:, 0]
                benchmark_ret = aligned_returns.iloc[:, 1]
                
                covariance = portfolio_ret.cov(benchmark_ret)
                benchmark_var = benchmark_ret.var()
                
                beta = covariance / benchmark_var if benchmark_var != 0 else 1
                
                portfolio_annual = portfolio_ret.mean() * 252
                benchmark_annual = benchmark_ret.mean() * 252
                alpha = portfolio_annual - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))
                
                # Information ratio
                active_return = portfolio_ret - benchmark_ret
                tracking_error = active_return.std() * np.sqrt(252)
                information_ratio = active_return.mean() * 252 / tracking_error if tracking_error != 0 else 0
            else:
                beta, alpha, information_ratio = 1, 0, 0
        else:
            beta, alpha, information_ratio = 1, 0, 0
        
        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns != 0 else float('inf')
        
        # Value at Risk (95th percentile)
        var_95 = returns.quantile(0.05)  # 5th percentile for 95% VaR
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95
        )
    
    def backtest_strategy(self, 
                         strategy: PortfolioStrategy,
                         tickers: List[str],
                         config: BacktestConfig) -> BacktestResult:
        """Run backtest for a specific strategy"""
        
        self.logger.info(f"Starting backtest for {strategy.name} strategy")
        
        # Get price data
        price_data = self.get_price_data(tickers, config.start_date, config.end_date)
        
        # Get benchmark data
        benchmark_data = self.get_price_data([config.benchmark], config.start_date, config.end_date)
        benchmark_returns = benchmark_data.pct_change().dropna().iloc[:, 0]
        
        # Initialize tracking variables
        portfolio_values = []
        returns = []
        positions = []
        trades = []
        
        current_weights = {}
        current_value = config.initial_capital
        last_rebalance_date = config.start_date
        transaction_costs = 0
        
        # Iterate through each trading day
        for i, date in enumerate(price_data.index):
            current_prices = price_data.loc[date]
            
            # Check if we should rebalance
            should_rebalance = (i == 0 or 
                              strategy.should_rebalance(date, last_rebalance_date, config.rebalance_frequency))
            
            if should_rebalance:
                # Generate new weights
                historical_data = price_data.loc[:date]
                if len(historical_data) >= 20:  # Minimum data requirement
                    try:
                        new_weights = strategy.generate_weights(historical_data, date)
                        
                        # Calculate transaction costs
                        if current_weights:
                            weight_changes = sum(abs(new_weights.get(ticker, 0) - current_weights.get(ticker, 0)) 
                                               for ticker in set(list(new_weights.keys()) + list(current_weights.keys())))
                            transaction_cost = current_value * weight_changes * config.transaction_cost
                            transaction_costs += transaction_cost
                            current_value -= transaction_cost
                            
                            # Record trades
                            for ticker in set(list(new_weights.keys()) + list(current_weights.keys())):
                                old_weight = current_weights.get(ticker, 0)
                                new_weight = new_weights.get(ticker, 0)
                                if abs(new_weight - old_weight) > 0.01:  # Only record significant changes
                                    trades.append({
                                        'date': date,
                                        'ticker': ticker,
                                        'old_weight': old_weight,
                                        'new_weight': new_weight,
                                        'transaction_cost': current_value * abs(new_weight - old_weight) * config.transaction_cost
                                    })
                        
                        current_weights = new_weights
                        last_rebalance_date = date
                        
                    except Exception as e:
                        self.logger.error(f"Error generating weights on {date}: {e}")
                        # Keep existing weights
            
            # Calculate portfolio value
            if i > 0:
                # Calculate returns based on price changes and weights
                prev_prices = price_data.iloc[i-1]
                price_returns = (current_prices - prev_prices) / prev_prices
                
                portfolio_return = sum(current_weights.get(ticker, 0) * price_returns.get(ticker, 0) 
                                     for ticker in current_weights.keys())
                
                current_value *= (1 + portfolio_return)
                returns.append(portfolio_return)
            else:
                returns.append(0)
            
            portfolio_values.append(current_value)
            
            # Record positions
            position_record = {'date': date, 'total_value': current_value}
            position_record.update(current_weights)
            positions.append(position_record)
        
        # Convert to pandas objects
        portfolio_values = pd.Series(portfolio_values, index=price_data.index)
        returns = pd.Series(returns, index=price_data.index)
        positions = pd.DataFrame(positions).set_index('date')
        trades = pd.DataFrame(trades)
        
        # Calculate performance metrics
        portfolio_metrics = self.calculate_performance_metrics(returns.dropna(), benchmark_returns, config.risk_free_rate)
        benchmark_metrics = self.calculate_performance_metrics(benchmark_returns, benchmark_returns, config.risk_free_rate)
        
        # Calculate drawdown periods
        drawdown_periods = self._calculate_drawdown_periods(portfolio_values)
        
        # Calculate monthly and yearly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        result = BacktestResult(
            config=config,
            portfolio_values=portfolio_values,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics,
            drawdown_periods=drawdown_periods,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns
        )
        
        self.logger.info(f"Backtest completed for {strategy.name}. Total return: {portfolio_metrics.total_return:.2%}")
        return result
    
    def _calculate_drawdown_periods(self, portfolio_values: pd.Series) -> List[Dict[str, Any]]:
        """Calculate drawdown periods"""
        cumulative = portfolio_values / portfolio_values.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        max_dd = 0
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_date = date
                max_dd = dd
            elif dd < max_dd and in_drawdown:
                max_dd = dd
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'max_drawdown': max_dd,
                    'duration_days': (date - start_date).days
                })
                max_dd = 0
        
        return drawdown_periods
    
    def compare_strategies(self, 
                          strategies: List[PortfolioStrategy],
                          tickers: List[str],
                          config: BacktestConfig) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        
        results = {}
        
        # Run backtests in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(len(strategies), 4)) as executor:
            future_to_strategy = {
                executor.submit(self.backtest_strategy, strategy, tickers, config): strategy 
                for strategy in strategies
            }
            
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy.name] = result
                except Exception as e:
                    self.logger.error(f"Backtest failed for {strategy.name}: {e}")
                    results[strategy.name] = None
        
        return results


def create_strategy_from_config(strategy_config: Dict[str, Any]) -> PortfolioStrategy:
    """Create strategy instance from configuration"""
    strategy_type = strategy_config.get('type', 'equal_weight')
    
    if strategy_type == 'equal_weight':
        return EqualWeightStrategy()
    elif strategy_type == 'market_cap':
        return MarketCapWeightStrategy()
    elif strategy_type == 'mean_reversion':
        lookback = strategy_config.get('lookback_period', 20)
        return MeanReversionStrategy(lookback)
    elif strategy_type == 'momentum':
        lookback = strategy_config.get('lookback_period', 60)
        return MomentumStrategy(lookback)
    elif strategy_type == 'min_variance':
        lookback = strategy_config.get('lookback_period', 252)
        return MinVarianceStrategy(lookback)
    elif strategy_type == 'max_sharpe':
        lookback = strategy_config.get('lookback_period', 252)
        return MaxSharpeStrategy(lookback)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# Utility functions for common backtesting scenarios
def quick_backtest(tickers: List[str], 
                  start_date: str, 
                  end_date: str,
                  strategy_type: str = 'equal_weight',
                  rebalance_frequency: str = 'monthly') -> BacktestResult:
    """Quick backtest with common parameters"""
    
    config = BacktestConfig(
        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
        initial_capital=100000,
        rebalance_frequency=rebalance_frequency,
        transaction_cost=0.001,
        benchmark='SPY'
    )
    
    strategy = create_strategy_from_config({'type': strategy_type})
    backtester = PortfolioBacktester()
    
    return backtester.backtest_strategy(strategy, tickers, config)