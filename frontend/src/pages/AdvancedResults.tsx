import { useLocation, Link } from 'react-router-dom';
import { useDarkMode } from '@/hooks/useDarkMode';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  Legend,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap
} from "recharts";
import { 
  ArrowLeft, 
  TrendingUp, 
  Shield, 
  Target, 
  Brain, 
  BarChart3, 
  PieChart as PieChartIcon,
  Activity,
  AlertTriangle,
  CheckCircle,
  Info
} from "lucide-react";
import { 
  PerformanceMetricsCard, 
  EnhancedAllocationChart, 
  EnhancedEfficientFrontier, 
  EnhancedGrowthChart,
  EnhancedRiskMetrics,
  EnhancedMonteCarloResults
} from "@/components/EnhancedCharts";

const AdvancedResults = () => {
  const { isDarkMode } = useDarkMode();
  const location = useLocation();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-purple-50/30 to-indigo-100/50 dark:from-slate-900 dark:via-purple-900/20 dark:to-indigo-900/20">
        <Card className="max-w-md bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
          <CardHeader className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <CardTitle className="text-xl bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent">
              No Portfolio Data
            </CardTitle>
            <CardDescription className="text-base text-slate-600 dark:text-slate-400">
              Run an advanced AI optimization first to view professional analytics.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/">
              <Button className="mt-4 w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white shadow-lg">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Home
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Process data for visualizations
  const performance = result.performance;
  const weights = result.weights;
  const tickers = result.tickers;
  
  // Portfolio composition data
  const compositionData = tickers.map((ticker: string, index: number) => ({
    ticker,
    weight: weights[index],
    percentage: (weights[index] * 100).toFixed(2)
  }));
  
  // Sector allocation data
  const sectorData = Object.entries(result.sector_allocation || {}).map(([sector, weight]) => ({
    sector,
    weight: weight as number,
    percentage: ((weight as number) * 100).toFixed(2)
  }));
  
  // Enhanced efficient frontier with multiple metrics
  const frontierData = result.efficient_frontier || [];
  
  // Risk decomposition data
  const riskDecomp = result.risk_decomposition;
  const riskContributionData = riskDecomp ? tickers.map((ticker: string, index: number) => ({
    ticker,
    contribution: riskDecomp.percentage_contributions[index] * 100,
    marginal: riskDecomp.marginal_contributions[index]
  })) : [];
  
  // Monte Carlo results
  const monteCarlo = result.monte_carlo;
  const monteCarloSummary = monteCarlo ? [
    { metric: "Expected Return", value: (monteCarlo.expected_return * 100).toFixed(2) + "%" },
    { metric: "Volatility", value: (monteCarlo.volatility * 100).toFixed(2) + "%" },
    { metric: "VaR (95%)", value: ((1 - monteCarlo.var_95) * 100).toFixed(2) + "%" },
    { metric: "CVaR (95%)", value: ((1 - monteCarlo.cvar_95) * 100).toFixed(2) + "%" },
    { metric: "Prob. of Loss", value: (monteCarlo.prob_loss * 100).toFixed(1) + "%" },
    { metric: "Prob. Outperform Market", value: (monteCarlo.prob_outperform_market * 100).toFixed(1) + "%" }
  ] : [];
  
  // Performance metrics for radar chart
  const performanceRadar = [
    { metric: 'Sharpe Ratio', value: Math.max(0, Math.min(5, performance.sharpe_ratio)) },
    { metric: 'Sortino Ratio', value: Math.max(0, Math.min(5, performance.sortino_ratio)) },
    { metric: 'Expected Return', value: Math.max(0, Math.min(5, performance.expected_return * 10)) },
    { metric: 'Risk Level', value: Math.max(0, Math.min(5, 5 - performance.volatility * 10)) },
    { metric: 'Diversification', value: riskDecomp ? Math.max(0, Math.min(5, 5 - riskDecomp.risk_concentration * 10)) : 3 }
  ];
  
  // Portfolio growth data
  const growthData = result.portfolio_growth ? 
    Object.entries(result.portfolio_growth).map(([date, value]) => ({
      date: new Date(date).toLocaleDateString(),
      value: value as number,
      formatted_value: ((value as number) * 100).toFixed(2) + '%'
    })) : [];
  
  // Forecast data
  const forecastData = result.forecasted_growth ?
    Object.entries(result.forecasted_growth).map(([date, forecast]: [string, { value: number; lower: number; upper: number }]) => ({
      date: new Date(date).toLocaleDateString(),
      value: forecast.value,
      lower: forecast.lower,
      upper: forecast.upper
    })) : [];

  const colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4', '#84CC16', '#F97316'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-purple-50/30 to-indigo-100/50 dark:from-slate-900 dark:via-purple-900/20 dark:to-indigo-900/20 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-purple-400/10 to-indigo-600/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-indigo-400/10 to-purple-600/10 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/3 left-1/4 w-60 h-60 bg-gradient-to-br from-violet-400/5 to-purple-600/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
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
              <Link to="/advanced">
                <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                  Back to Optimizer
                </Button>
              </Link>
              <Link to="/ai-hub">
                <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                  <Brain className="w-4 h-4 mr-2" />
                  AI Analysis
                </Button>
              </Link>
            </div>
            <div>
              <div className="flex items-center space-x-4 mb-2">
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 via-indigo-600 to-violet-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <Brain className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-purple-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent">
                  Advanced Analytics
                </h1>
              </div>
              <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
                Professional-grade AI portfolio analysis with institutional algorithms and risk management
              </p>
              <div className="flex flex-wrap gap-3 mt-4">
                <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                  <Brain className="w-3 h-3 mr-1" />
                  {result.strategy_info?.name || 'AI Optimization'}
                </Badge>
                <Badge className="bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300">
                  <Target className="w-3 h-3 mr-1" />
                  {tickers.length} Assets
                </Badge>
                <Badge className="bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300">
                  <BarChart3 className="w-3 h-3 mr-1" />
                  {result.data_points} Data Points
                </Badge>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse"></div>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
              <CheckCircle className="w-3 h-3 mr-1" />
              AI Enhanced
            </Badge>
          </div>
        </div>

        {/* Enhanced Strategy Info Card */}
        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-indigo-500/5 to-transparent"></div>
          <CardHeader className="relative">
            <CardTitle className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl bg-gradient-to-r from-purple-700 to-indigo-700 dark:from-purple-300 dark:to-indigo-300 bg-clip-text text-transparent">
                {result.strategy_info?.name || 'Advanced AI Strategy'}
              </span>
            </CardTitle>
            <CardDescription className="text-base text-slate-600 dark:text-slate-400 ml-13">
              {result.strategy_info?.description || 'Professional portfolio optimization using advanced machine learning algorithms and institutional-grade risk management techniques.'}
            </CardDescription>
            {result.strategy_info?.features && (
              <div className="flex flex-wrap gap-2 mt-4 ml-13">
                {result.strategy_info.features.map((feature: string, idx: number) => (
                  <Badge key={idx} variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                    {feature}
                  </Badge>
                ))}
              </div>
            )}
          </CardHeader>
        </Card>

        {/* Enhanced Performance Summary */}
        <PerformanceMetricsCard metrics={performance} />

        {/* Enhanced Main Analysis Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 p-2">
            <TabsTrigger value="overview" className="data-[state=active]:bg-purple-100 data-[state=active]:text-purple-800 dark:data-[state=active]:bg-purple-900/30 dark:data-[state=active]:text-purple-300">
              Overview
            </TabsTrigger>
            <TabsTrigger value="allocation" className="data-[state=active]:bg-indigo-100 data-[state=active]:text-indigo-800 dark:data-[state=active]:bg-indigo-900/30 dark:data-[state=active]:text-indigo-300">
              Allocation
            </TabsTrigger>
            <TabsTrigger value="risk" className="data-[state=active]:bg-red-100 data-[state=active]:text-red-800 dark:data-[state=active]:bg-red-900/30 dark:data-[state=active]:text-red-300">
              Risk Analysis
            </TabsTrigger>
            <TabsTrigger value="performance" className="data-[state=active]:bg-green-100 data-[state=active]:text-green-800 dark:data-[state=active]:bg-green-900/30 dark:data-[state=active]:text-green-300">
              Performance
            </TabsTrigger>
            <TabsTrigger value="montecarlo" className="data-[state=active]:bg-orange-100 data-[state=active]:text-orange-800 dark:data-[state=active]:bg-orange-900/30 dark:data-[state=active]:text-orange-300">
              Monte Carlo
            </TabsTrigger>
            <TabsTrigger value="forecast" className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-800 dark:data-[state=active]:bg-blue-900/30 dark:data-[state=active]:text-blue-300">
              Forecast
            </TabsTrigger>
          </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Portfolio Performance Radar */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="w-5 h-5" />
                  <span>Performance Metrics</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={performanceRadar}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis domain={[0, 5]} />
                    <Radar
                      name="Portfolio"
                      dataKey="value"
                      stroke="#8B5CF6"
                      fill="#8B5CF6"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Enhanced Efficient Frontier */}
            <EnhancedEfficientFrontier 
              data={frontierData}
              currentPortfolio={{
                return: performance.expected_return,
                volatility: performance.volatility,
                sharpe_ratio: performance.sharpe_ratio
              }}
            />
          </div>
        </TabsContent>

        {/* Allocation Tab */}
        <TabsContent value="allocation" className="space-y-6">
          {/* Enhanced Asset Allocation */}
          <EnhancedAllocationChart 
            data={compositionData} 
            title="Advanced Portfolio Allocation"
          />

          {/* Sector Allocation */}
          {sectorData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5" />
                  <span>Sector Allocation</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={sectorData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sector" />
                    <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                    <Bar dataKey="weight" fill="#8B5CF6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Risk Analysis Tab */}
        <TabsContent value="risk" className="space-y-6">
          {riskDecomp && (
            <EnhancedRiskMetrics 
              riskDecomposition={riskDecomp}
              tickers={tickers}
            />
          )}
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          {/* Enhanced Portfolio Growth Chart */}
          {(growthData.length > 0 || forecastData.length > 0) && (
            <EnhancedGrowthChart 
              data={growthData} 
              forecastData={forecastData}
              title="Advanced Portfolio Performance Analysis"
            />
          )}
        </TabsContent>

        {/* Monte Carlo Tab */}
        <TabsContent value="montecarlo" className="space-y-6">
          {monteCarlo && (
            <EnhancedMonteCarloResults monteCarlo={monteCarlo} />
          )}
        </TabsContent>

        {/* Forecast Tab */}
        <TabsContent value="forecast" className="space-y-6">
          {forecastData.length > 0 && (
            <EnhancedGrowthChart 
              data={[]} 
              forecastData={forecastData}
              title="AI-Powered Portfolio Forecast"
            />
          )}
          
          {/* Forecast Summary */}
          <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-700">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="w-5 h-5 text-blue-600" />
                <span>Prophet AI Forecast Insights</span>
              </CardTitle>
              <CardDescription>
                Advanced time-series forecasting using Facebook's Prophet algorithm with confidence intervals
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-700">
                  <div className="text-sm text-blue-700 dark:text-blue-300 mb-1">Forecast Horizon</div>
                  <div className="text-lg font-bold text-blue-800 dark:text-blue-200">12 Months</div>
                </div>
                <div className="p-4 rounded-lg bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-700">
                  <div className="text-sm text-blue-700 dark:text-blue-300 mb-1">Confidence Level</div>
                  <div className="text-lg font-bold text-blue-800 dark:text-blue-200">95%</div>
                </div>
                <div className="p-4 rounded-lg bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-700">
                  <div className="text-sm text-blue-700 dark:text-blue-300 mb-1">Model Accuracy</div>
                  <div className="text-lg font-bold text-blue-800 dark:text-blue-200">High</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Enhanced Analytics Summary Footer */}
      <Card className="bg-gradient-to-r from-slate-100/80 via-purple-50/80 to-indigo-100/80 dark:from-slate-800/80 dark:via-purple-900/20 dark:to-indigo-900/20 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500/5 to-transparent"></div>
        <CardContent className="relative p-8">
          <div className="text-center mb-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Advanced Analytics Summary</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">Professional-grade AI portfolio optimization with institutional algorithms</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center group">
              <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">
                {(performance.expected_return * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">AI Expected Return</div>
              <div className="text-xs text-slate-500 dark:text-slate-500">Machine Learning</div>
            </div>
            <div className="text-center group">
              <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-red-500 to-pink-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-1">
                {(performance.volatility * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Risk Management</div>
              <div className="text-xs text-slate-500 dark:text-slate-500">Advanced modeling</div>
            </div>
            <div className="text-center group">
              <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Target className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                {performance.sharpe_ratio.toFixed(2)}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Sharpe Ratio</div>
              <div className="text-xs text-slate-500 dark:text-slate-500">Efficiency score</div>
            </div>
            <div className="text-center group">
              <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <CheckCircle className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">
                {tickers.length}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Optimized Assets</div>
              <div className="text-xs text-slate-500 dark:text-slate-500">AI selected</div>
            </div>
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
};

export default AdvancedResults;