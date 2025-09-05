import { useLocation, Link } from 'react-router-dom';
import { useDarkMode } from '@/hooks/useDarkMode';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { ArrowLeft, TrendingUp, Target, BarChart3, CheckCircle } from "lucide-react";
import { BarChart, Bar } from 'recharts';
import { PieChart, Pie, Cell, Legend } from 'recharts';
import { ScatterChart, Scatter, ZAxis } from 'recharts';
import { AreaChart, Area } from "recharts";
import { 
  PerformanceMetricsCard, 
  EnhancedAllocationChart, 
  EnhancedEfficientFrontier, 
  EnhancedGrowthChart 
} from "@/components/EnhancedCharts";


const Results = () => {
  const { isDarkMode } = useDarkMode();
  const location = useLocation();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-100/50 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900/20">
        <Card className="max-w-md bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
          <CardHeader className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <CardTitle className="text-xl bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
              No Portfolio Data
            </CardTitle>
            <CardDescription className="text-base text-slate-600 dark:text-slate-400">
              Submit a portfolio optimization request first to view your results.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Link to="/">
                <Button className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Home
                </Button>
              </Link>
              <Link to="/ai-hub">
                <Button variant="outline" className="w-full">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  AI Analysis Hub
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const metrics = {
    expected_return: result.expected_return,
    volatility: result.volatility,
    sharpe_ratio: result.sharpe_ratio,
    sortino_ratio: result.sortino_ratio,
    downside_deviation: result.downside_deviation,
    portfolio_value: result.portfolio_value || 100000  // fallback if not present
  };
  
  const composition = result.tickers.map((ticker: string, index: number) => ({
    ticker,
    weight: result.weights[index],
    percentage: (result.weights[index] * 100).toFixed(2)
  }));
  
  const frontier = result.efficient_frontier || [];
  const growthData = result.portfolio_growth
  ? Object.entries(result.portfolio_growth).map(([date, growth]) => ({
      date,
      value: growth as number
    }))
  : [];

  const forecastData = result.forecast_growth
  ? Object.entries(result.forecast_growth).map(([date, forecast]: [string, { value?: number; lower?: number; upper?: number } | number]) => ({
      date,
      value: typeof forecast === 'object' ? forecast.value : forecast,
      lower: typeof forecast === 'object' ? forecast.lower : undefined,
      upper: typeof forecast === 'object' ? forecast.upper : undefined
    }))
  : [];
  




  const colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4', '#84CC16', '#F97316'];

  const mergedGrowthData = [
    ...growthData.map(d => ({ ...d, type: "actual" })),
    ...forecastData.map(d => ({ ...d, type: "forecast" }))
  ];


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-100/50 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900/20 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-green-400/10 to-blue-600/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-indigo-400/10 to-purple-600/10 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/3 left-1/4 w-60 h-60 bg-gradient-to-br from-emerald-400/5 to-cyan-600/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 p-8 space-y-8">
        {/* Enhanced Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex items-start space-x-6">
            <div className="flex items-center space-x-3">
              <Link to="/">
                <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Home
                </Button>
              </Link>
              <Link to="/basic">
                <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                  Back to Optimizer
                </Button>
              </Link>
              <Link to="/ai-hub">
                <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  AI Analysis
                </Button>
              </Link>
            </div>
            <div>
              <div className="flex items-center space-x-4 mb-2">
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 via-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <TrendingUp className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-green-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Portfolio Results
                </h1>
              </div>
              <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
                Smart optimization results with professional-grade analytics and performance insights
              </p>
              <div className="flex flex-wrap gap-3 mt-4">
                <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {result.strategy}
                </Badge>
                <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                  <Target className="w-3 h-3 mr-1" />
                  {result.tickers.length} Assets
                </Badge>
                <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                  <BarChart3 className="w-3 h-3 mr-1" />
                  {result.data_points} Data Points
                </Badge>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
              <CheckCircle className="w-3 h-3 mr-1" />
              Optimized
            </Badge>
          </div>
        </div>
        
        {/* Enhanced Performance Metrics */}
        <PerformanceMetricsCard metrics={metrics} />

        {/* Enhanced Efficient Frontier */}
        {frontier && frontier.length > 0 && (
          <EnhancedEfficientFrontier 
            data={frontier} 
            currentPortfolio={{
              return: metrics.expected_return,
              volatility: metrics.volatility,
              sharpe_ratio: metrics.sharpe_ratio
            }}
          />
        )}

        {/* Enhanced Portfolio Allocation */}
        <EnhancedAllocationChart data={composition} title="Smart Portfolio Allocation" />

        {/* Enhanced Growth Chart */}
        {(growthData.length > 0 || forecastData.length > 0) && (
          <EnhancedGrowthChart 
            data={growthData} 
            forecastData={forecastData}
            title="Portfolio Performance & AI Forecast"
          />
        )}

        {/* Enhanced Quick Stats Footer */}
        <Card className="bg-gradient-to-r from-slate-100/80 via-green-50/80 to-blue-100/80 dark:from-slate-800/80 dark:via-slate-700/80 dark:to-green-900/20 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-green-500/5 to-transparent"></div>
          <CardContent className="relative p-8">
            <div className="text-center mb-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Optimization Summary</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">Professional portfolio optimization completed successfully</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <TrendingUp className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">
                  {(metrics.expected_return * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Expected Return</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Annualized</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-orange-500 to-amber-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <Target className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-1">
                  {(metrics.volatility * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Risk Level</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Volatility</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <BarChart3 className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                  {metrics.sharpe_ratio.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Sharpe Ratio</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Risk-adjusted</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <CheckCircle className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">
                  {result.tickers.length}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Assets</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Diversified</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Results;
