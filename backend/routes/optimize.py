from flask import Blueprint, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import re
import requests
from datetime import datetime, timedelta
from prophet import Prophet
warnings.filterwarnings('ignore')

optimize_bp = Blueprint('optimize', __name__)

# Stock ticker validation
def validate_ticker(ticker):
    """Validate if ticker is a valid stock symbol format"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic format validation (1-5 alphanumeric characters)
    if not re.match(r'^[A-Z]{1,5}$', ticker.upper()):
        return False
        
    return True

def validate_tickers_exist(tickers, timeout=10):
    """Validate that tickers exist by attempting to fetch basic info"""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            # Quick validation using yfinance info
            stock = yf.Ticker(ticker)
            # Try to get basic info with timeout
            info = stock.info
            if info and 'symbol' in info:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def forecast_portfolio_growth(portfolio_growth_dict, periods=90):
    """Forecast portfolio growth using Prophet"""
    try:
        if not portfolio_growth_dict or len(portfolio_growth_dict) < 10:
            raise ValueError("Insufficient historical data for forecasting")
        
        # Prepare data for Prophet
        df = pd.DataFrame([
            {'ds': pd.to_datetime(date), 'y': value}
            for date, value in portfolio_growth_dict.items()
        ])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.8
        )
        
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Extract forecast data (only future periods)
        future_forecast = forecast.tail(periods)
        
        # Format forecast results
        forecast_dict = {}
        for _, row in future_forecast.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            forecast_dict[date_str] = {
                'value': float(row['yhat']),
                'lower': float(row['yhat_lower']),
                'upper': float(row['yhat_upper'])
            }
        
        return forecast_dict
        
    except Exception as e:
        raise Exception(f"Forecasting failed: {str(e)}")

class PortfolioOptimizer:
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def portfolio_performance(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Objective function for maximizing Sharpe ratio"""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Objective function for minimizing volatility"""
        return self.portfolio_performance(weights)[1]
    
    def optimize_max_sharpe(self):
        """Find portfolio with maximum Sharpe ratio"""
        try:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                self.negative_sharpe,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise Exception(f"Optimization failed: {result.message}")
                
            return result.x
            
        except Exception as e:
            raise Exception(f"Max Sharpe optimization failed: {str(e)}")
    
    def optimize_min_volatility(self):
        """Find minimum volatility portfolio"""
        try:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                self.portfolio_volatility,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise Exception(f"Optimization failed: {result.message}")
                
            return result.x
            
        except Exception as e:
            raise Exception(f"Min volatility optimization failed: {str(e)}")
    
    def optimize_target_return(self, target_return):
        """Find minimum volatility portfolio for a target return"""
        # Check if target return is feasible
        min_possible_return = self.mean_returns.min()
        max_possible_return = self.mean_returns.max()
        
        if target_return < min_possible_return or target_return > max_possible_return:
            return None, f"Target return {target_return:.1%} is not feasible. Range: {min_possible_return:.1%} to {max_possible_return:.1%}"
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            return result.x, None
        else:
            return None, f"Optimization failed: {result.message}"
    
    def equal_weight_portfolio(self):
        """Equal weight portfolio (1/N rule)"""
        return np.array([1/self.n_assets] * self.n_assets)
    
    def risk_parity_portfolio(self):
        """Risk parity portfolio (equal risk contribution)"""
        try:
            def risk_budget_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                if portfolio_vol == 0:
                    return 1e10  # Large penalty for zero volatility
                marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                return np.sum((contrib - contrib.mean()) ** 2)
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0.001, 1) for _ in range(self.n_assets))  # Small minimum to avoid division by zero
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                risk_budget_objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                print(f"Risk parity optimization failed: {result.message}, falling back to equal weight")
                return self.equal_weight_portfolio()
                
            return result.x
            
        except Exception as e:
            print(f"Risk parity optimization error: {str(e)}, falling back to equal weight")
            return self.equal_weight_portfolio()
    
    def efficient_frontier(self, num_portfolios=100):
        """Generate efficient frontier"""
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        for target in target_returns:
            try:
                weights, error = self.optimize_target_return(target)
                if weights is not None:
                    ret, vol, sharpe = self.portfolio_performance(weights)
                    efficient_portfolios.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': sharpe,
                        'weights': weights.tolist()
                    })
            except:
                continue
                
        return efficient_portfolios

@optimize_bp.route("/optimize", methods=["POST"])
def optimize_portfolio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        tickers = data.get("tickers", [])
        start = data.get("start")
        end = data.get("end")
        strategy = data.get("strategy", "max_sharpe")
        target_return = data.get("target_return", 0.1)
        risk_free_rate = data.get("risk_free_rate", 0.02)
        include_efficient_frontier = data.get("include_efficient_frontier", False)
        include_portfolio_growth = data.get("include_portfolio_growth", True)

        # Input validation
        if not tickers or not start or not end:
            return jsonify({"error": "Missing required fields: tickers, start, or end date"}), 400
            
        if not isinstance(tickers, list) or len(tickers) == 0:
            return jsonify({"error": "Tickers must be a non-empty list"}), 400
            
        if len(tickers) > 20:
            return jsonify({"error": "Too many tickers (maximum 20 allowed)"}), 400
            
        # Validate ticker formats
        invalid_format_tickers = []
        for ticker in tickers:
            if not validate_ticker(ticker):
                invalid_format_tickers.append(ticker)
                
        if invalid_format_tickers:
            return jsonify({
                "error": f"Invalid ticker format: {', '.join(invalid_format_tickers)}",
                "details": "Tickers must be 1-5 uppercase letters"
            }), 400
            
        # Validate dates
        try:
            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')
            
            if start_date >= end_date:
                return jsonify({"error": "Start date must be before end date"}), 400
                
            if end_date > datetime.now():
                return jsonify({"error": "End date cannot be in the future"}), 400
                
            if (end_date - start_date).days < 30:
                return jsonify({"error": "Date range must be at least 30 days"}), 400
                
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
            
        # Validate strategy
        valid_strategies = ['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity', 'target_return']
        if strategy not in valid_strategies:
            return jsonify({"error": f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"}), 400
            
        # Validate target return if needed
        if strategy == 'target_return':
            if not isinstance(target_return, (int, float)) or target_return <= 0:
                return jsonify({"error": "Target return must be a positive number"}), 400
                
        # Validate tickers exist (with timeout)
        print(f"Validating tickers: {tickers}")
        valid_tickers, invalid_tickers = validate_tickers_exist(tickers, timeout=10)
        
        if invalid_tickers:
            return jsonify({
                "error": f"Invalid tickers: {', '.join(invalid_tickers)}",
                "details": "These tickers do not exist or cannot be accessed",
                "valid_tickers": valid_tickers
            }), 400
            
        tickers = valid_tickers  # Use only valid tickers

        # Download stock data with timeout handling
        print(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
        try:
            # Set timeout for yfinance download (note: yfinance doesn't directly support timeout)
            raw_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            
            if raw_data.empty:
                return jsonify({"error": "No data available for the specified tickers and date range"}), 400
                
        except Exception as download_error:
            return jsonify({
                "error": "Failed to download stock data",
                "details": str(download_error)
            }), 500
        
        # Handle different data structures based on number of tickers
        if len(tickers) == 1:
            # Single ticker case
            if 'Close' in raw_data.columns:
                df = raw_data[['Close']].copy()
                df.columns = tickers
            else:
                available_cols = raw_data.columns.tolist()
                price_cols = ['Close', 'Adj Close', 'close', 'adj_close']
                price_col = None
                for col in price_cols:
                    if col in available_cols:
                        price_col = col
                        break
                
                if price_col:
                    df = raw_data[[price_col]].copy()
                    df.columns = tickers
                else:
                    df = raw_data.iloc[:, [0]].copy()
                    df.columns = tickers
        else:
            # Multiple tickers case
            if isinstance(raw_data.columns, pd.MultiIndex):
                try:
                    if 'Close' in raw_data.columns.get_level_values(0):
                        df = raw_data['Close'].copy()
                    elif 'Close' in raw_data.columns.get_level_values(1):
                        df = raw_data.xs('Close', axis=1, level=1)
                    elif 'Adj Close' in raw_data.columns.get_level_values(0):
                        df = raw_data['Adj Close'].copy()
                    else:
                        level_0_values = raw_data.columns.get_level_values(0).unique().tolist()
                        level_1_values = raw_data.columns.get_level_values(1).unique().tolist()
                        return jsonify({
                            "error": f"Could not find Close prices. Available level 0: {level_0_values}, level 1: {level_1_values}"
                        }), 500
                except Exception as e:
                    return jsonify({"error": f"Error extracting Close prices: {str(e)}"}), 500
            else:
                return jsonify({"error": f"Expected MultiIndex for multiple tickers. Columns: {raw_data.columns.tolist()}"}), 500
        
        # Clean data
        df = df.dropna()
        if df.empty:
            return jsonify({"error": "No valid data after removing missing values"}), 400
        
        # Calculate returns
        returns = df.pct_change().dropna()
        if returns.empty:
            return jsonify({"error": "No valid returns data"}), 400

        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns, risk_free_rate)

        # Optimize based on strategy with proper error handling
        try:
            if strategy == "max_sharpe":
                weights = optimizer.optimize_max_sharpe()
                description = "Maximum Sharpe Ratio Portfolio"
            elif strategy == "min_volatility":
                weights = optimizer.optimize_min_volatility()
                description = "Minimum Volatility Portfolio"
            elif strategy == "equal_weight":
                weights = optimizer.equal_weight_portfolio()
                description = "Equal Weight Portfolio"
            elif strategy == "risk_parity":
                weights = optimizer.risk_parity_portfolio()
                description = "Risk Parity Portfolio"
            elif strategy == "target_return":
                weights, error_msg = optimizer.optimize_target_return(target_return)
                description = f"Minimum Volatility Portfolio for {target_return:.1%} Target Return"
                if weights is None:
                    return jsonify({
                        "error": error_msg,
                        "suggestion": f"Try a target return between {optimizer.mean_returns.min():.1%} and {optimizer.mean_returns.max():.1%}"
                    }), 400
            else:
                return jsonify({"error": f"Unknown strategy: {strategy}"}), 400
                
        except Exception as opt_error:
            return jsonify({
                "error": "Portfolio optimization failed",
                "details": str(opt_error),
                "suggestion": "Try a different optimization strategy or check your data"
            }), 500
            
        if weights is None:
            return jsonify({
                "error": "Optimization failed to converge",
                "suggestion": "Try a different strategy or adjust parameters"
            }), 500

        # Calculate portfolio performance
        portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_performance(weights)

        response = {
            "strategy": strategy,
            "description": description,
            "tickers": df.columns.tolist(),
            "weights": weights.round(4).tolist(),
            "expected_return": round(portfolio_return, 4),
            "volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "risk_free_rate": risk_free_rate,
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].strftime('%Y-%m-%d'),
                "end": df.index[-1].strftime('%Y-%m-%d')
            }
        }

        # Add portfolio growth time series
        if include_portfolio_growth:
            try:
                # Calculate cumulative returns for the portfolio
                cumulative_returns = (1 + returns).cumprod()
                portfolio_growth = cumulative_returns.dot(weights)
                
                # Convert to dictionary with date strings as keys
                portfolio_growth_dict = {
                    date.strftime('%Y-%m-%d'): round(value, 6)
                    for date, value in portfolio_growth.items()
                }
                
                response["portfolio_growth"] = portfolio_growth_dict
                
                # Add forecasting with better error handling
                try:
                    if len(portfolio_growth_dict) >= 30:  # Need sufficient data
                        forecasted_growth = forecast_portfolio_growth(portfolio_growth_dict, periods=90)
                        response["forecasted_growth"] = forecasted_growth
                        print(f"Forecast generated successfully for {len(forecasted_growth)} periods")
                    else:
                        response["forecast_warning"] = "Insufficient data for forecasting (minimum 30 data points required)"
                except Exception as e:
                    print(f"Forecasting error: {str(e)}")
                    response["forecast_error"] = f"Forecasting failed: {str(e)}"

                # Add individual asset growth for comparison
                individual_growth = {}
                for ticker in df.columns:
                    ticker_cumulative = (1 + returns[ticker]).cumprod()
                    individual_growth[ticker] = {
                        date.strftime('%Y-%m-%d'): round(value, 6)
                        for date, value in ticker_cumulative.items()
                    }
                response["individual_asset_growth"] = individual_growth
                
                # Add portfolio vs benchmark comparison
                equal_weight_growth = cumulative_returns.dot(optimizer.equal_weight_portfolio())
                response["equal_weight_benchmark"] = {
                    date.strftime('%Y-%m-%d'): round(value, 6)
                    for date, value in equal_weight_growth.items()
                }
                
            except Exception as e:
                response["portfolio_growth_error"] = str(e)

        # Add efficient frontier if requested
        if include_efficient_frontier and len(tickers) > 1:
            try:
                efficient_frontier = optimizer.efficient_frontier(50)
                response["efficient_frontier"] = efficient_frontier
            except Exception as e:
                response["efficient_frontier_error"] = str(e)

        return jsonify(response)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Unexpected error in optimize_portfolio: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return more helpful error messages
        if "timeout" in error_msg.lower():
            return jsonify({
                "error": "Request timed out",
                "details": "The optimization took too long. Try with fewer tickers or a shorter date range."
            }), 408
        elif "memory" in error_msg.lower():
            return jsonify({
                "error": "Insufficient memory",
                "details": "Try with fewer tickers or a shorter date range."
            }), 507
        else:
            return jsonify({
                "error": "Internal server error",
                "details": error_msg
            }), 500

@optimize_bp.route("/portfolio-info", methods=["POST"])
def get_portfolio_info():
    """Get portfolio statistics and feasible return range"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        tickers = data.get("tickers", [])
        start = data.get("start")
        end = data.get("end")
        risk_free_rate = data.get("risk_free_rate", 0.02)

        # Basic validation
        if not tickers or not start or not end:
            return jsonify({"error": "Missing required fields: tickers, start, or end date"}), 400
            
        # Validate ticker formats
        invalid_tickers = [ticker for ticker in tickers if not validate_ticker(ticker)]
        if invalid_tickers:
            return jsonify({
                "error": f"Invalid ticker format: {', '.join(invalid_tickers)}"
            }), 400
        # Download and process data (same logic as optimize endpoint)
        raw_data = yf.download(tickers, start=start, end=end, auto_adjust=True)
        
        if len(tickers) == 1:
            if 'Close' in raw_data.columns:
                df = raw_data[['Close']].copy()
                df.columns = tickers
            else:
                available_cols = raw_data.columns.tolist()
                price_cols = ['Close', 'Adj Close', 'close', 'adj_close']
                price_col = None
                for col in price_cols:
                    if col in available_cols:
                        price_col = col
                        break
                
                if price_col:
                    df = raw_data[[price_col]].copy()
                    df.columns = tickers
                else:
                    df = raw_data.iloc[:, [0]].copy()
                    df.columns = tickers
        else:
            if isinstance(raw_data.columns, pd.MultiIndex):
                try:
                    if 'Close' in raw_data.columns.get_level_values(0):
                        df = raw_data['Close'].copy()
                    elif 'Close' in raw_data.columns.get_level_values(1):
                        df = raw_data.xs('Close', axis=1, level=1)
                    elif 'Adj Close' in raw_data.columns.get_level_values(0):
                        df = raw_data['Adj Close'].copy()
                    else:
                        return jsonify({"error": "Could not find Close prices"}), 500
                except Exception as e:
                    return jsonify({"error": f"Error extracting Close prices: {str(e)}"}), 500
            else:
                return jsonify({"error": "Expected MultiIndex for multiple tickers"}), 500
        
        df = df.dropna()
        returns = df.pct_change().dropna()
        
        if df.empty or returns.empty:
            return jsonify({"error": "No valid data"}), 400

        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        
        # Calculate different portfolio metrics
        equal_weights = optimizer.equal_weight_portfolio()
        equal_weight_return, equal_weight_vol, equal_weight_sharpe = optimizer.portfolio_performance(equal_weights)
        
        min_vol_weights = optimizer.optimize_min_volatility()
        if min_vol_weights is not None:
            min_vol_return, min_vol_volatility, min_vol_sharpe = optimizer.portfolio_performance(min_vol_weights)
        else:
            min_vol_return, min_vol_volatility, min_vol_sharpe = None, None, None
            
        max_sharpe_weights = optimizer.optimize_max_sharpe()
        if max_sharpe_weights is not None:
            max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = optimizer.portfolio_performance(max_sharpe_weights)
        else:
            max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = None, None, None

        return jsonify({
            "tickers": df.columns.tolist(),
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].strftime('%Y-%m-%d'),
                "end": df.index[-1].strftime('%Y-%m-%d')
            },
            "individual_returns": dict(zip(df.columns.tolist(), optimizer.mean_returns.round(4).tolist())),
            "return_range": {
                "min_possible": round(float(optimizer.mean_returns.min()), 4),
                "max_possible": round(float(optimizer.mean_returns.max()), 4),
                "recommended_min": round(float(optimizer.mean_returns.min() * 0.95), 4),  # 5% buffer
                "recommended_max": round(float(optimizer.mean_returns.max() * 0.95), 4)   # 5% buffer
            },
            "portfolio_examples": {
                "equal_weight": {
                    "return": round(equal_weight_return, 4),
                    "volatility": round(equal_weight_vol, 4),
                    "sharpe_ratio": round(equal_weight_sharpe, 4),
                    "weights": equal_weights.round(4).tolist()
                },
                "min_volatility": {
                    "return": round(min_vol_return, 4) if min_vol_return else None,
                    "volatility": round(min_vol_volatility, 4) if min_vol_volatility else None,
                    "sharpe_ratio": round(min_vol_sharpe, 4) if min_vol_sharpe else None,
                    "weights": min_vol_weights.round(4).tolist() if min_vol_weights is not None else None
                },
                "max_sharpe": {
                    "return": round(max_sharpe_return, 4) if max_sharpe_return else None,
                    "volatility": round(max_sharpe_vol, 4) if max_sharpe_vol else None,
                    "sharpe_ratio": round(max_sharpe_ratio, 4) if max_sharpe_ratio else None,
                    "weights": max_sharpe_weights.round(4).tolist() if max_sharpe_weights is not None else None
                }
            },
            "target_return_guidance": {
                "conservative": round(float(optimizer.mean_returns.min() + (optimizer.mean_returns.max() - optimizer.mean_returns.min()) * 0.2), 4),
                "moderate": round(float(optimizer.mean_returns.min() + (optimizer.mean_returns.max() - optimizer.mean_returns.min()) * 0.5), 4),
                "aggressive": round(float(optimizer.mean_returns.min() + (optimizer.mean_returns.max() - optimizer.mean_returns.min()) * 0.8), 4)
            }
        })

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@optimize_bp.route("/strategies", methods=["GET"])
def get_strategies():
    """Get available optimization strategies"""
    strategies = {
        "max_sharpe": {
            "name": "Maximum Sharpe Ratio",
            "description": "Maximizes risk-adjusted returns",
            "parameters": ["risk_free_rate"]
        },
        "min_volatility": {
            "name": "Minimum Volatility",
            "description": "Minimizes portfolio risk",
            "parameters": []
        },
        "equal_weight": {
            "name": "Equal Weight",
            "description": "Equal allocation to all assets (1/N rule)",
            "parameters": []
        },
        "risk_parity": {
            "name": "Risk Parity",
            "description": "Equal risk contribution from all assets",
            "parameters": []
        },
        "target_return": {
            "name": "Target Return",
            "description": "Minimum volatility for a target return",
            "parameters": ["target_return"],
            "notes": "Use /portfolio-info endpoint to get feasible return range"
        }
    }
    return jsonify(strategies)

