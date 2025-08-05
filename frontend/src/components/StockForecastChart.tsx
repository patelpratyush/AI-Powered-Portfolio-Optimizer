import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import TickerAutocomplete from "@/components/TickerAutocomplete";
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  AreaChart,
  Area,
  ReferenceLine,
  ComposedChart,
  Bar,
  ScatterChart,
  Scatter
} from "recharts";
import { 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  Zap, 
  Target,
  AlertCircle,
  RefreshCw,
  Activity,
  BarChart3,
  Eye,
  Settings
} from "lucide-react";
import EnhancedRecommendations from "@/components/EnhancedRecommendations";

interface PredictionData {
  day: number;
  predicted_price: number;
  predicted_return: number;
  confidence_lower: number;
  confidence_upper: number;
  date: string;
  models_used?: string[];
  model_weights?: Record<string, number>;
}

interface TradingSignal {
  action: 'BUY' | 'SELL' | 'HOLD' | 'STRONG_BUY' | 'STRONG_SELL';
  strength: number;
  reasoning: string;
  confidence?: number;
  target_price?: number;
  stop_loss?: number;
  risk_level?: string;
  expected_return?: number;
  max_downside?: number;
  time_horizon?: string;
  reasons?: Array<{
    category: string;
    indicator: string;
    value: number;
    threshold: number;
    weight: number;
    description: string;
    bullish: boolean;
  }>;
  summary?: string;
}

interface ModelPrediction {
  ticker: string;
  model: string;
  current_price: number;
  predictions: PredictionData[];
  summary: {
    avg_predicted_return: number;
    max_predicted_price: number;
    min_predicted_price: number;
    volatility_estimate: number;
    trend_direction?: string;
  };
  trading_signal: TradingSignal;
  model_confidence?: string;
  last_updated: string;
}

interface StockInfo {
  ticker: string;
  name: string;
  current_price: number;
  previous_close: number;
  day_change: number;
  day_change_percent: number;
  market_cap: number;
  sector: string;
  industry: string;
  currency: string;
}

interface StockForecastChartProps {
  ticker: string;
  onTickerChange?: (ticker: string) => void;
}

export const StockForecastChart: React.FC<StockForecastChartProps> = ({ 
  ticker, 
  onTickerChange 
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [predictions, setPredictions] = useState<Record<string, ModelPrediction>>({});
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [selectedDays, setSelectedDays] = useState<number>(10);
  const [activeTab, setActiveTab] = useState('forecast');

  // Fetch predictions from API
  const fetchPredictions = async (targetTicker: string, days: number, models: string = 'all') => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/predict/${targetTicker}?days=${days}&models=${models}`);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      setStockInfo(data.stock_info);
      setPredictions(data.predictions);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
      console.error('Prediction fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (ticker) {
      fetchPredictions(ticker, selectedDays);
    }
  }, [ticker, selectedDays]);

  const handleRefresh = () => {
    if (ticker) {
      fetchPredictions(ticker, selectedDays, 'all');
    }
  };

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
  };

  const handleDaysChange = (days: string) => {
    setSelectedDays(parseInt(days));
  };

  // Prepare chart data
  const getChartData = () => {
    const selectedPrediction = predictions[selectedModel];
    if (!selectedPrediction || !stockInfo || !selectedPrediction.predictions || !Array.isArray(selectedPrediction.predictions)) return [];

    const baseData = selectedPrediction.predictions.map(pred => ({
      ...pred,
      date_formatted: new Date(pred.date).toLocaleDateString(),
      price_change: pred.predicted_price - stockInfo.current_price,
      confidence_range: pred.confidence_upper - pred.confidence_lower
    }));

    // Add current price as day 0
    return [
      {
        day: 0,
        predicted_price: stockInfo.current_price,
        predicted_return: 0,
        confidence_lower: stockInfo.current_price,
        confidence_upper: stockInfo.current_price,
        date: new Date().toISOString().split('T')[0],
        date_formatted: 'Today',
        price_change: 0,
        confidence_range: 0
      },
      ...baseData
    ];
  };

  const chartData = getChartData();
  const currentPrediction = predictions[selectedModel];

  // Get signal color
  const getSignalColor = (signal: TradingSignal) => {
    switch (signal.action) {
      case 'STRONG_BUY': return 'text-green-800 bg-green-50 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-700';
      case 'BUY': return 'text-green-600 bg-green-50 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-700';
      case 'SELL': return 'text-red-600 bg-red-50 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-700';
      case 'STRONG_SELL': return 'text-red-800 bg-red-50 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-700';
      case 'HOLD': return 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-700';
      default: return 'text-gray-600 bg-gray-50 border-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-700';
    }
  };

  const getSignalIcon = (action: string) => {
    switch (action) {
      case 'STRONG_BUY': return <TrendingUp className="w-4 h-4" />;
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      case 'STRONG_SELL': return <TrendingDown className="w-4 h-4" />;
      case 'HOLD': return <Target className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getModelIcon = (model: string) => {
    switch (model.toLowerCase()) {
      case 'prophet': return <BarChart3 className="w-4 h-4" />;
      case 'xgboost': return <Zap className="w-4 h-4" />;
      case 'lstm': return <Brain className="w-4 h-4" />;
      case 'ensemble': return <Settings className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-700">
        <CardContent className="p-8 text-center">
          <div className="relative">
            <div className="w-20 h-20 mx-auto mb-6 relative">
              <div className="absolute inset-0 rounded-full border-4 border-blue-200 dark:border-blue-700"></div>
              <div className="absolute inset-0 rounded-full border-4 border-blue-600 border-t-transparent animate-spin"></div>
              <Brain className="w-8 h-8 absolute inset-0 m-auto text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-2">
              Generating AI Predictions...
            </h3>
            <p className="text-blue-600 dark:text-blue-400 mb-4">
              Training models and analyzing market data for {ticker}
            </p>
            <div className="flex justify-center space-x-4 text-sm text-blue-500 dark:text-blue-400">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                <span>Prophet Analysis</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse delay-200"></div>
                <span>XGBoost Training</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse delay-400"></div>
                <span>LSTM Processing</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-red-200 dark:border-red-700">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-red-500 to-pink-600 rounded-2xl flex items-center justify-center shadow-lg">
            <AlertCircle className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-semibold text-red-800 dark:text-red-300 mb-2">
            Prediction Failed
          </h3>
          <p className="text-red-600 dark:text-red-400 mb-4">
            {error}
          </p>
          <div className="flex justify-center space-x-3">
            <Button 
              onClick={handleRefresh}
              className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry Analysis
            </Button>
            <Button 
              variant="outline" 
              onClick={() => {
                setError(null);
                if (onTickerChange) {
                  onTickerChange('AAPL'); // Fallback to AAPL
                }
              }}
              className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-600 dark:text-red-400"
            >
              Try Different Stock
            </Button>
          </div>
          <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
            <p className="text-xs text-red-700 dark:text-red-300">
              <strong>Common issues:</strong> Invalid ticker symbol, API rate limit, or temporary service unavailability
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!stockInfo || Object.keys(predictions).length === 0) {
    return (
      <Card className="bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-800 dark:to-gray-800 border-slate-200 dark:border-slate-700">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-slate-400 to-gray-500 rounded-2xl flex items-center justify-center shadow-lg">
            <Eye className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-2">
            No Predictions Available
          </h3>
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            Enter a ticker symbol above to get AI-powered stock predictions
          </p>
          <div className="flex justify-center space-x-2 mb-4">
            {['AAPL', 'MSFT', 'GOOGL', 'TSLA'].map(symbol => (
              <Button
                key={symbol}
                variant="outline"
                size="sm"
                onClick={() => onTickerChange && onTickerChange(symbol)}
                className="text-xs"
              >
                {symbol}
              </Button>
            ))}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Try these popular stocks to see our AI analysis in action
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stock Info Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-3">
                <span className="text-2xl font-bold">{stockInfo.ticker}</span>
                <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                  {stockInfo.sector}
                </Badge>
              </CardTitle>
              <CardDescription className="text-base">
                {stockInfo.name} • {stockInfo.industry}
              </CardDescription>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold">
                ${stockInfo.current_price.toFixed(2)}
              </div>
              <div className={`flex items-center space-x-1 ${
                stockInfo.day_change >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {stockInfo.day_change >= 0 ? 
                  <TrendingUp className="w-4 h-4" /> : 
                  <TrendingDown className="w-4 h-4" />
                }
                <span>
                  {stockInfo.day_change >= 0 ? '+' : ''}${stockInfo.day_change.toFixed(2)} 
                  ({stockInfo.day_change_percent.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Model:</label>
            <Select value={selectedModel} onValueChange={handleModelChange}>
              <SelectTrigger className="w-48 mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(predictions).map(model => (
                  <SelectItem key={model} value={model}>
                    <div className="flex items-center space-x-2">
                      {getModelIcon(model)}
                      <span className="capitalize">{model}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Forecast Days:</label>
            <Select value={selectedDays.toString()} onValueChange={handleDaysChange}>
              <SelectTrigger className="w-32 mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5">5 days</SelectItem>
                <SelectItem value="10">10 days</SelectItem>
                <SelectItem value="15">15 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Enhanced Trading Signal */}
      {currentPrediction?.trading_signal && stockInfo && (
        <EnhancedRecommendations 
          tradingSignal={currentPrediction.trading_signal}
          currentPrice={stockInfo.current_price}
          ticker={ticker}
        />
      )}

      {/* Main Charts */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="forecast">Price Forecast</TabsTrigger>
          <TabsTrigger value="returns">Expected Returns</TabsTrigger>
          <TabsTrigger value="confidence">Confidence Bands</TabsTrigger>
        </TabsList>

        {/* Enhanced Price Forecast Tab */}
        <TabsContent value="forecast" className="space-y-4">
          <Card className="bg-gradient-to-br from-white to-blue-50/30 dark:from-slate-800 dark:to-blue-900/10 border-blue-200 dark:border-blue-800">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-t-lg">
              <CardTitle className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                  {getModelIcon(selectedModel)}
                </div>
                <span className="text-xl bg-gradient-to-r from-blue-700 to-indigo-700 dark:from-blue-300 dark:to-indigo-300 bg-clip-text text-transparent">
                  {selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} Price Forecast
                </span>
              </CardTitle>
              <CardDescription className="text-slate-600 dark:text-slate-400">
                AI-powered stock price predictions with confidence intervals and trend analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="h-[500px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart 
                    data={chartData} 
                    margin={{ top: 30, right: 40, bottom: 60, left: 60 }}
                  >
                    <defs>
                      {/* Enhanced gradients */}
                      <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.4}/>
                        <stop offset="50%" stopColor="#60A5FA" stopOpacity={0.2}/>
                        <stop offset="100%" stopColor="#93C5FD" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#1E40AF" stopOpacity={0.8}/>
                        <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.4}/>
                      </linearGradient>
                      <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                        <feMerge> 
                          <feMergeNode in="coloredBlur"/>
                          <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                      </filter>
                    </defs>
                    
                    {/* Enhanced grid */}
                    <CartesianGrid 
                      strokeDasharray="2 4" 
                      stroke="#E2E8F0" 
                      strokeOpacity={0.5}
                      horizontal={true}
                      vertical={false}
                    />
                    
                    {/* Improved axes */}
                    <XAxis 
                      dataKey="date_formatted" 
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      interval={0}
                    />
                    <YAxis 
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                      domain={['dataMin - 2%', 'dataMax + 2%']}
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      width={80}
                    />
                    
                    {/* Enhanced tooltip */}
                    <Tooltip 
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700">
                              <p className="font-semibold text-slate-900 dark:text-slate-100 mb-2">{label}</p>
                              {payload.map((entry, index) => (
                                <div key={index} className="flex items-center space-x-2 text-sm">
                                  <div 
                                    className="w-3 h-3 rounded-full" 
                                    style={{ backgroundColor: entry.color }}
                                  ></div>
                                  <span className="text-slate-600 dark:text-slate-400">
                                    {entry.name === 'predicted_price' && `Predicted: $${entry.value.toFixed(2)}`}
                                    {entry.name === 'confidence_upper' && `Upper: $${entry.value.toFixed(2)}`}
                                    {entry.name === 'confidence_lower' && `Lower: $${entry.value.toFixed(2)}`}
                                  </span>
                                </div>
                              ))}
                              {data && (
                                <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700">
                                  <p className="text-xs text-slate-500">
                                    Return: {(data.predicted_return * 100).toFixed(2)}%
                                  </p>
                                </div>
                              )}
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    
                    {/* Confidence bands as separate lines */}
                    <Line
                      type="monotone"
                      dataKey="confidence_upper"
                      stroke="#60A5FA"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={false}
                      connectNulls={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="confidence_lower"
                      stroke="#60A5FA"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={false}
                      connectNulls={false}
                    />
                    
                    {/* Current price reference line - enhanced */}
                    <ReferenceLine 
                      y={stockInfo?.current_price} 
                      stroke="#059669" 
                      strokeWidth={2}
                      strokeDasharray="8 4"
                      label={{ 
                        value: `Current: $${stockInfo?.current_price?.toFixed(2)}`, 
                        position: "topLeft",
                        style: { 
                          fontSize: '12px', 
                          fontWeight: 'bold',
                          fill: '#059669'
                        }
                      }}
                    />
                    
                    {/* Main prediction line - enhanced */}
                    <Line
                      type="monotone"
                      dataKey="predicted_price"
                      stroke="#1E40AF"
                      strokeWidth={4}
                      dot={{ 
                        fill: '#1E40AF', 
                        strokeWidth: 2, 
                        r: 5,
                        filter: "url(#glow)"
                      }}
                      activeDot={{ 
                        r: 8, 
                        fill: '#3B82F6',
                        stroke: '#1E40AF',
                        strokeWidth: 3,
                        filter: "url(#glow)"
                      }}
                      connectNulls={false}
                    />
                    
                    {/* Enhanced legend */}
                    <Legend 
                      wrapperStyle={{
                        paddingTop: '20px',
                        fontSize: '12px'
                      }}
                      formatter={(value) => {
                        if (value === 'predicted_price') return 'AI Prediction';
                        if (value === 'confidence_upper') return 'Confidence Band';
                        return value;
                      }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              
              {/* Chart insights */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {chartData.length > 1 ? `${selectedDays}` : '0'}
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Days Forecast</div>
                </div>
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {currentPrediction?.summary?.avg_predicted_return 
                      ? `${(currentPrediction.summary.avg_predicted_return * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Avg Return</div>
                </div>
                <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {currentPrediction?.model_confidence || 'Medium'}
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Confidence</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enhanced Returns Tab */}
        <TabsContent value="returns" className="space-y-4">
          <Card className="bg-gradient-to-br from-white to-green-50/30 dark:from-slate-800 dark:to-green-900/10 border-green-200 dark:border-green-800">
            <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-t-lg">
              <CardTitle className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl bg-gradient-to-r from-green-700 to-emerald-700 dark:from-green-300 dark:to-emerald-300 bg-clip-text text-transparent">
                  Expected Returns Analysis
                </span>
              </CardTitle>
              <CardDescription className="text-slate-600 dark:text-slate-400">
                Daily expected returns and cumulative performance outlook
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart 
                    data={chartData.slice(1)} 
                    margin={{ top: 20, right: 40, bottom: 40, left: 60 }}
                  >
                    <defs>
                      <linearGradient id="returnsGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10B981" stopOpacity={0.8}/>
                        <stop offset="50%" stopColor="#34D399" stopOpacity={0.4}/>
                        <stop offset="100%" stopColor="#6EE7B7" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="negativeReturns" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#EF4444" stopOpacity={0.8}/>
                        <stop offset="50%" stopColor="#F87171" stopOpacity={0.4}/>
                        <stop offset="100%" stopColor="#FCA5A5" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    
                    <CartesianGrid 
                      strokeDasharray="2 4" 
                      stroke="#E2E8F0" 
                      strokeOpacity={0.5}
                      horizontal={true}
                      vertical={false}
                    />
                    
                    <XAxis 
                      dataKey="date_formatted" 
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    
                    <YAxis 
                      tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      width={80}
                    />
                    
                    <Tooltip 
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const value = payload[0].value;
                          const isPositive = value >= 0;
                          return (
                            <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700">
                              <p className="font-semibold text-slate-900 dark:text-slate-100 mb-2">{label}</p>
                              <div className="flex items-center space-x-2">
                                <div 
                                  className={`w-3 h-3 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
                                ></div>
                                <span className={`font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                                  {isPositive ? '+' : ''}{(value * 100).toFixed(2)}%
                                </span>
                              </div>
                              <p className="text-xs text-slate-500 mt-1">
                                Expected daily return
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    
                    {/* Positive and negative return areas */}
                    <Area
                      type="monotone"
                      dataKey="predicted_return"
                      stroke="#10B981"
                      fill="url(#returnsGradient)"
                      strokeWidth={3}
                      fillOpacity={0.6}
                    />
                    
                    {/* Zero line */}
                    <ReferenceLine 
                      y={0} 
                      stroke="#6B7280" 
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      label={{ 
                        value: "Break Even", 
                        position: "topRight",
                        style: { fontSize: '11px', fill: '#6B7280', fontWeight: 'bold' }
                      }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              
              {/* Returns insights */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-xl font-bold text-green-600 dark:text-green-400">
                    {currentPrediction?.summary?.avg_predicted_return 
                      ? `${currentPrediction.summary.avg_predicted_return >= 0 ? '+' : ''}${(currentPrediction.summary.avg_predicted_return * 100).toFixed(2)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Avg Daily Return</div>
                </div>
                <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {currentPrediction?.summary?.max_predicted_price && stockInfo?.current_price
                      ? `${((currentPrediction.summary.max_predicted_price - stockInfo.current_price) / stockInfo.current_price * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Max Upside</div>
                </div>
                <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
                    {currentPrediction?.summary?.volatility_estimate 
                      ? `${(currentPrediction.summary.volatility_estimate * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Volatility</div>
                </div>
                <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
                    {currentPrediction?.trading_signal?.strength 
                      ? `${(currentPrediction.trading_signal.strength * 100).toFixed(0)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Signal Strength</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enhanced Confidence Tab */}
        <TabsContent value="confidence" className="space-y-4">
          <Card className="bg-gradient-to-br from-white to-purple-50/30 dark:from-slate-800 dark:to-purple-900/10 border-purple-200 dark:border-purple-800">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-t-lg">
              <CardTitle className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-violet-600 rounded-lg flex items-center justify-center">
                  <Eye className="w-4 h-4 text-white" />
                </div>
                <span className="text-xl bg-gradient-to-r from-purple-700 to-violet-700 dark:from-purple-300 dark:to-violet-300 bg-clip-text text-transparent">
                  Prediction Confidence Analysis
                </span>
              </CardTitle>
              <CardDescription className="text-slate-600 dark:text-slate-400">
                Model uncertainty assessment and confidence interval visualization
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart 
                    data={chartData.slice(1)} 
                    margin={{ top: 30, right: 40, bottom: 60, left: 60 }}
                  >
                    <defs>
                      <linearGradient id="confidenceScatter" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                        <stop offset="50%" stopColor="#A78BFA" stopOpacity={0.6}/>
                        <stop offset="100%" stopColor="#C4B5FD" stopOpacity={0.4}/>
                      </linearGradient>
                      <linearGradient id="confidenceArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#8B5CF6" stopOpacity={0.3}/>
                        <stop offset="100%" stopColor="#C4B5FD" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    
                    {/* Enhanced grid */}
                    <CartesianGrid 
                      strokeDasharray="2 4" 
                      stroke="#E2E8F0" 
                      strokeOpacity={0.5}
                      horizontal={true}
                      vertical={false}
                    />
                    
                    {/* Improved axes */}
                    <XAxis 
                      dataKey="day" 
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      tickFormatter={(value) => `Day ${value}`}
                      domain={[1, selectedDays]}
                    />
                    <YAxis 
                      dataKey="confidence_range"
                      tickFormatter={(value) => `$${value.toFixed(2)}`}
                      tick={{ fontSize: 11, fill: '#64748B' }}
                      tickLine={{ stroke: '#CBD5E1' }}
                      axisLine={{ stroke: '#CBD5E1' }}
                      width={80}
                    />
                    
                    {/* Enhanced tooltip */}
                    <Tooltip 
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700">
                              <p className="font-semibold text-slate-900 dark:text-slate-100 mb-2">Day {label}</p>
                              <div className="space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                                  <span className="text-sm text-slate-600 dark:text-slate-400">
                                    Confidence Range: <span className="font-medium">${data.confidence_range?.toFixed(2)}</span>
                                  </span>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                  <span className="text-sm text-slate-600 dark:text-slate-400">
                                    Upper Bound: <span className="font-medium">${data.confidence_upper?.toFixed(2)}</span>
                                  </span>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                  <span className="text-sm text-slate-600 dark:text-slate-400">
                                    Lower Bound: <span className="font-medium">${data.confidence_lower?.toFixed(2)}</span>
                                  </span>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                                  <span className="text-sm text-slate-600 dark:text-slate-400">
                                    Predicted: <span className="font-medium">${data.predicted_price?.toFixed(2)}</span>
                                  </span>
                                </div>
                              </div>
                              <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700">
                                <p className="text-xs text-slate-500">
                                  Uncertainty: {((data.confidence_range / data.predicted_price) * 100).toFixed(1)}%
                                </p>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    
                    {/* Confidence area visualization */}
                    <Area
                      type="monotone"
                      dataKey="confidence_range"
                      stroke="none"
                      fill="url(#confidenceArea)"
                      fillOpacity={0.6}
                    />
                    
                    {/* Scatter plot for confidence ranges */}
                    <Scatter 
                      dataKey="confidence_range" 
                      fill="url(#confidenceScatter)"
                      name="Confidence Range"
                      shape="circle"
                    />
                    
                    {/* Trend line for confidence evolution */}
                    <Line
                      type="monotone"
                      dataKey="confidence_range"
                      stroke="#8B5CF6"
                      strokeWidth={3}
                      dot={{ 
                        fill: '#8B5CF6', 
                        strokeWidth: 2, 
                        r: 4
                      }}
                      activeDot={{ 
                        r: 6, 
                        fill: '#A78BFA',
                        stroke: '#8B5CF6',
                        strokeWidth: 2
                      }}
                      connectNulls={false}
                    />
                    
                    {/* Enhanced legend */}
                    <Legend 
                      wrapperStyle={{
                        paddingTop: '20px',
                        fontSize: '12px'
                      }}
                      formatter={(value) => {
                        if (value === 'confidence_range') return 'Prediction Uncertainty';
                        return value;
                      }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              
              {/* Confidence insights */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
                    {chartData.length > 1 
                      ? `±$${(chartData.slice(1).reduce((sum, item) => sum + item.confidence_range, 0) / chartData.slice(1).length).toFixed(2)}`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Avg Uncertainty</div>
                </div>
                <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {chartData.length > 1 
                      ? `${((chartData.slice(1).reduce((sum, item) => sum + (item.confidence_range / item.predicted_price), 0) / chartData.slice(1).length) * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Relative Error</div>
                </div>
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-xl font-bold text-green-600 dark:text-green-400">
                    {chartData.length > 1 
                      ? `$${Math.max(...chartData.slice(1).map(item => item.confidence_upper)).toFixed(2)}`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Max Upper Bound</div>
                </div>
                <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
                    {currentPrediction?.model_confidence || 'Medium'}
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">Model Confidence</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Model Summary */}
      {currentPrediction?.summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">Avg Return</div>
              <div className={`text-2xl font-bold ${
                currentPrediction.summary.avg_predicted_return >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {(currentPrediction.summary.avg_predicted_return * 100).toFixed(2)}%
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">Max Price</div>
              <div className="text-2xl font-bold text-blue-600">
                ${currentPrediction.summary.max_predicted_price.toFixed(2)}
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">Min Price</div>
              <div className="text-2xl font-bold text-purple-600">
                ${currentPrediction.summary.min_predicted_price.toFixed(2)}
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">Volatility</div>
              <div className="text-2xl font-bold text-orange-600">
                {(currentPrediction.summary.volatility_estimate * 100).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default StockForecastChart;