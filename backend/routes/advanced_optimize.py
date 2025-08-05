from flask import Blueprint, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from advanced_optimizer import AdvancedPortfolioOptimizer, get_sector_data, get_market_cap_weights
from routes.optimize import validate_ticker, validate_tickers_exist, forecast_portfolio_growth

warnings.filterwarnings('ignore')

advanced_optimize_bp = Blueprint('advanced_optimize', __name__)

@advanced_optimize_bp.route("/advanced-optimize", methods=["POST"])
def advanced_optimize_portfolio():
    """Advanced portfolio optimization with enhanced features"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Extract parameters
        tickers = data.get("tickers", [])
        start = data.get("start")
        end = data.get("end")
        strategy = data.get("strategy", "black_litterman")
        constraints = data.get("constraints", {})
        optimization_params = data.get("optimization_params", {})
        include_monte_carlo = data.get("include_monte_carlo", True)
        include_risk_decomposition = data.get("include_risk_decomposition", True)
        
        # Validation
        if not tickers or not start or not end:
            return jsonify({"error": "Missing required fields"}), 400
            
        # Validate tickers
        invalid_tickers = [ticker for ticker in tickers if not validate_ticker(ticker)]
        if invalid_tickers:
            return jsonify({"error": f"Invalid tickers: {invalid_tickers}"}), 400
            
        valid_tickers, invalid_tickers = validate_tickers_exist(tickers)
        if invalid_tickers:
            return jsonify({"error": f"Tickers not found: {invalid_tickers}"}), 400
            
        tickers = valid_tickers
        
        # Download data
        print(f"Downloading data for advanced optimization: {tickers}")
        raw_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        
        if raw_data.empty:
            return jsonify({"error": "No data available"}), 400
            
        # Process data
        if len(tickers) == 1:
            if 'Close' in raw_data.columns:
                df = raw_data[['Close']].copy()
                df.columns = tickers
            else:
                df = raw_data.iloc[:, [0]].copy()
                df.columns = tickers
        else:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if 'Close' in raw_data.columns.get_level_values(0):
                    df = raw_data['Close'].copy()
                else:
                    df = raw_data.xs('Close', axis=1, level=1)
            else:
                return jsonify({"error": "Unexpected data structure"}), 500
        
        df = df.dropna()
        if df.empty:
            return jsonify({"error": "No valid data after cleaning"}), 400
            
        returns = df.pct_change().dropna()
        if returns.empty:
            return jsonify({"error": "No valid returns data"}), 400
        
        # Get additional data
        sector_data = get_sector_data(tickers)
        market_cap_weights = get_market_cap_weights(tickers)
        
        # Initialize advanced optimizer
        risk_free_rate = optimization_params.get("risk_free_rate", 0.02)
        optimizer = AdvancedPortfolioOptimizer(returns, risk_free_rate, market_cap_weights)
        
        # Execute optimization strategy
        weights = None
        strategy_info = {}
        
        try:
            if strategy == "black_litterman":
                # Black-Litterman with optional investor views
                views_matrix = optimization_params.get("views_matrix")
                views_uncertainty = optimization_params.get("views_uncertainty")
                tau = optimization_params.get("tau", 0.05)
                
                weights, bl_returns = optimizer.black_litterman_optimization(
                    views_matrix, views_uncertainty, tau
                )
                strategy_info = {
                    "name": "Black-Litterman Optimization",
                    "description": "Incorporates market equilibrium and investor views",
                    "equilibrium_returns": bl_returns.tolist()
                }
                
            elif strategy == "hierarchical_risk_parity":
                weights = optimizer.hierarchical_risk_parity()
                strategy_info = {
                    "name": "Hierarchical Risk Parity",
                    "description": "Machine learning based risk parity using asset clustering"
                }
                
            elif strategy == "multi_objective":
                objectives = optimization_params.get("objectives", ["sharpe", "sortino"])
                objective_weights = optimization_params.get("objective_weights")
                
                weights = optimizer.multi_objective_optimization(objectives, objective_weights)
                strategy_info = {
                    "name": "Multi-Objective Optimization",
                    "description": f"Optimizes for: {', '.join(objectives)}",
                    "objectives": objectives
                }
                
            elif strategy == "risk_budgeting":
                risk_budgets = optimization_params.get("risk_budgets")
                weights = optimizer.risk_budgeting_optimization(risk_budgets)
                strategy_info = {
                    "name": "Risk Budgeting",
                    "description": "Custom risk contribution targets for each asset"
                }
                
            elif strategy == "constrained":
                weights = optimizer.constrained_optimization(constraints)
                strategy_info = {
                    "name": "Constrained Optimization",
                    "description": "Portfolio optimization with advanced constraints",
                    "constraints": constraints
                }
                
            else:
                return jsonify({"error": f"Unknown advanced strategy: {strategy}"}), 400
                
        except Exception as opt_error:
            return jsonify({
                "error": "Advanced optimization failed",
                "details": str(opt_error)
            }), 500
        
        if weights is None:
            return jsonify({"error": "Optimization failed to converge"}), 500
            
        # Calculate performance metrics
        performance = optimizer.portfolio_performance(weights)
        
        # Risk decomposition
        risk_decomposition = None
        if include_risk_decomposition:
            try:
                risk_decomposition = optimizer.calculate_risk_decomposition(weights)
                # Convert numpy arrays to lists for JSON serialization
                risk_decomposition = {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in risk_decomposition.items()
                }
            except Exception as e:
                print(f"Risk decomposition failed: {e}")
        
        # Monte Carlo simulation
        monte_carlo_results = None
        if include_monte_carlo:
            try:
                mc_params = optimization_params.get("monte_carlo", {})
                num_simulations = mc_params.get("num_simulations", 5000)
                time_horizon = mc_params.get("time_horizon", 252)
                
                monte_carlo_results = optimizer.monte_carlo_simulation(
                    weights, num_simulations, time_horizon
                )
                # Convert numpy arrays to lists for JSON serialization
                monte_carlo_results['final_values'] = monte_carlo_results['final_values'].tolist()
                monte_carlo_results['max_drawdowns'] = monte_carlo_results['max_drawdowns'].tolist()
                # Don't include full portfolio_values array (too large), just summary stats
                portfolio_values = monte_carlo_results['portfolio_values']
                monte_carlo_results['portfolio_values_summary'] = {
                    'percentiles': {
                        '5th': np.percentile(portfolio_values[:, -1], 5),
                        '25th': np.percentile(portfolio_values[:, -1], 25),
                        '50th': np.percentile(portfolio_values[:, -1], 50),
                        '75th': np.percentile(portfolio_values[:, -1], 75),
                        '95th': np.percentile(portfolio_values[:, -1], 95)
                    }
                }
                del monte_carlo_results['portfolio_values']  # Remove large array
                
            except Exception as e:
                print(f"Monte Carlo simulation failed: {e}")
        
        # Portfolio growth calculation
        cumulative_returns = (1 + returns).cumprod()
        portfolio_growth = cumulative_returns.dot(weights)
        portfolio_growth_dict = {
            date.strftime('%Y-%m-%d'): round(value, 6)
            for date, value in portfolio_growth.items()
        }
        
        # Forecasting
        forecasted_growth = None
        try:
            if len(portfolio_growth_dict) >= 30:
                forecasted_growth = forecast_portfolio_growth(portfolio_growth_dict, periods=90)
        except Exception as e:
            print(f"Forecasting failed: {e}")
        
        # Efficient frontier (enhanced)
        efficient_frontier = []
        try:
            target_returns = np.linspace(
                optimizer.mean_returns.min(), 
                optimizer.mean_returns.max(), 
                50
            )
            
            for target in target_returns:
                try:
                    ef_weights = optimizer.constrained_optimization({
                        'target_return': target,
                        'min_weights': 0,
                        'max_weights': 1
                    })
                    if ef_weights is not None:
                        ef_performance = optimizer.portfolio_performance(ef_weights)
                        efficient_frontier.append({
                            'return': ef_performance['return'],
                            'volatility': ef_performance['volatility'],
                            'sharpe_ratio': ef_performance['sharpe_ratio'],
                            'sortino_ratio': ef_performance['sortino_ratio'],
                            'weights': ef_weights.tolist()
                        })
                except:
                    continue
        except Exception as e:
            print(f"Efficient frontier calculation failed: {e}")
        
        # Build response
        response = {
            "strategy": strategy,
            "strategy_info": strategy_info,
            "tickers": tickers,
            "weights": weights.round(6).tolist(),
            "performance": {
                "expected_return": round(performance['return'], 6),
                "volatility": round(performance['volatility'], 6), 
                "sharpe_ratio": round(performance['sharpe_ratio'], 6),
                "sortino_ratio": round(performance['sortino_ratio'], 6),
                "downside_deviation": round(performance['downside_deviation'], 6)
            },
            "sector_allocation": {},
            "portfolio_growth": portfolio_growth_dict,
            "efficient_frontier": efficient_frontier,
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].strftime('%Y-%m-%d'),
                "end": df.index[-1].strftime('%Y-%m-%d')
            }
        }
        
        # Add sector allocation
        for ticker, weight in zip(tickers, weights):
            sector = sector_data.get(ticker, 'Unknown')
            if sector not in response["sector_allocation"]:
                response["sector_allocation"][sector] = 0
            response["sector_allocation"][sector] += weight
        
        # Add optional components
        if risk_decomposition:
            response["risk_decomposition"] = risk_decomposition
            
        if monte_carlo_results:
            response["monte_carlo"] = monte_carlo_results
            
        if forecasted_growth:
            response["forecasted_growth"] = forecasted_growth
            
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Advanced optimization error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Advanced optimization failed",
            "details": str(e)
        }), 500


@advanced_optimize_bp.route("/optimization-strategies", methods=["GET"])
def get_advanced_strategies():
    """Get available advanced optimization strategies"""
    strategies = {
        "black_litterman": {
            "name": "Black-Litterman",
            "description": "Incorporates market equilibrium and investor views",
            "parameters": [
                "views_matrix", "views_uncertainty", "tau", "risk_free_rate"
            ],
            "features": ["Market equilibrium", "Investor views", "Bayesian approach"]
        },
        "hierarchical_risk_parity": {
            "name": "Hierarchical Risk Parity",
            "description": "ML-based clustering for risk allocation",
            "parameters": ["linkage_method"],
            "features": ["Machine learning", "Asset clustering", "Risk diversification"]
        },
        "multi_objective": {
            "name": "Multi-Objective Optimization",
            "description": "Balance multiple objectives simultaneously",
            "parameters": ["objectives", "objective_weights"],
            "features": ["Multiple objectives", "Customizable weights", "Pareto efficiency"]
        },
        "risk_budgeting": {
            "name": "Risk Budgeting",
            "description": "Target specific risk contributions",
            "parameters": ["risk_budgets"],
            "features": ["Custom risk allocation", "Risk parity", "Active management"]
        },
        "constrained": {
            "name": "Constrained Optimization",
            "description": "Advanced constraints and limits",
            "parameters": ["min_weights", "max_weights", "sector_limits", "target_return"],
            "features": ["Position limits", "Sector constraints", "Regulatory compliance"]
        }
    }
    return jsonify(strategies)


@advanced_optimize_bp.route("/risk-metrics", methods=["POST"])
def calculate_risk_metrics():
    """Calculate comprehensive risk metrics for a portfolio"""
    try:
        data = request.get_json()
        tickers = data.get("tickers", [])
        weights = data.get("weights", [])
        start = data.get("start")
        end = data.get("end")
        
        if not all([tickers, weights, start, end]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Download data
        raw_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        
        # Process data (simplified for this endpoint)
        if len(tickers) == 1:
            df = pd.DataFrame(raw_data['Close']).rename(columns={'Close': tickers[0]})
        else:
            df = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
            
        returns = df.pct_change().dropna()
        
        # Initialize optimizer for risk calculations
        optimizer = AdvancedPortfolioOptimizer(returns)
        weights_array = np.array(weights)
        
        # Calculate comprehensive risk metrics
        performance = optimizer.portfolio_performance(weights_array)
        risk_decomposition = optimizer.calculate_risk_decomposition(weights_array)
        monte_carlo = optimizer.monte_carlo_simulation(weights_array, num_simulations=5000)
        
        # Additional risk metrics
        portfolio_returns = returns.dot(weights_array)
        
        # VaR calculations
        var_95_historical = np.percentile(portfolio_returns, 5)
        var_99_historical = np.percentile(portfolio_returns, 1)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Beta calculation (assuming SPY as market proxy)
        try:
            spy_data = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            portfolio_aligned = portfolio_returns.loc[common_dates]
            spy_aligned = spy_returns.loc[common_dates]
            
            covariance = np.cov(portfolio_aligned, spy_aligned)[0, 1]
            market_variance = np.var(spy_aligned)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
        except:
            beta = 1.0  # Default if SPY data unavailable
        
        risk_metrics = {
            "performance": performance,
            "risk_decomposition": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in risk_decomposition.items()
            },
            "var_metrics": {
                "var_95_historical": float(var_95_historical),
                "var_99_historical": float(var_99_historical),
                "var_95_monte_carlo": float(monte_carlo['var_95']),
                "var_99_monte_carlo": float(monte_carlo['var_99']),
                "cvar_95": float(monte_carlo['cvar_95']),
                "cvar_99": float(monte_carlo['cvar_99'])
            },
            "drawdown_metrics": {
                "max_drawdown_historical": float(max_drawdown),
                "max_drawdown_monte_carlo": float(np.mean(monte_carlo['max_drawdowns']))
            },
            "market_metrics": {
                "beta": float(beta),
                "correlation_with_market": float(np.corrcoef(portfolio_aligned, spy_aligned)[0, 1]) if 'portfolio_aligned' in locals() else None
            },
            "probability_metrics": {
                "prob_loss": float(monte_carlo['prob_loss']),
                "prob_outperform_market": float(monte_carlo['prob_outperform_market'])
            }
        }
        
        return jsonify(risk_metrics)
        
    except Exception as e:
        return jsonify({
            "error": "Risk metrics calculation failed",
            "details": str(e)
        }), 500