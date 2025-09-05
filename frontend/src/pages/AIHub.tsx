import React, { useState } from 'react';
import { useDarkMode } from '@/hooks/useDarkMode';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import TickerAutocomplete from "@/components/TickerAutocomplete";
import { Link } from 'react-router-dom';
import { 
  Brain, 
  Upload, 
  TrendingUp, 
  Settings, 
  Database,
  ChartBar,
  Target,
  Zap,
  ArrowLeft,
  Sparkles
} from "lucide-react";

// Import our new components
import PortfolioUploader from "@/components/PortfolioUploader";
import StockForecastChart from "@/components/StockForecastChart";
import ModelTrainingCenter from "@/components/ModelTrainingCenter";

interface PortfolioHolding {
  id: string;
  ticker: string;
  shares: number;
  avgPrice: number;
  currentValue?: number;
  marketPrice?: number;
  gainLoss?: number;
  gainLossPercent?: number;
}

interface AnalysisResult {
  error?: string;
  predictions?: {
    ensemble?: {
      trading_signal?: {
        action: string;
        target_price?: number;
        confidence?: number;
      };
      summary?: {
        avg_predicted_return?: number;
      };
    };
    prophet?: {
      trading_signal?: {
        action: string;
        target_price?: number;
        confidence?: number;
      };
      summary?: {
        avg_predicted_return?: number;
      };
    };
  };
  stock_info?: {
    current_price?: number;
  };
}

const AIHub: React.FC = () => {
  const { isDarkMode } = useDarkMode();
  const [activeTab, setActiveTab] = useState('forecast');
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [importedPortfolio, setImportedPortfolio] = useState<PortfolioHolding[]>([]);
  const [portfolioAnalysisResults, setPortfolioAnalysisResults] = useState<Record<string, AnalysisResult>>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handlePortfolioImport = (holdings: PortfolioHolding[]) => {
    setImportedPortfolio(holdings);
    console.log('Portfolio imported:', holdings);
    // Seamlessly transition to analysis tab
    setActiveTab('analysis');
    // Show success notification
    console.log(`✅ Successfully imported ${holdings.length} portfolio positions`);
  };

  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker.toUpperCase());
  };

  const analyzeBatchPortfolio = async () => {
    if (importedPortfolio.length === 0) return;
    
    setIsAnalyzing(true);
    try {
      const tickers = importedPortfolio.map(h => h.ticker);
      const response = await fetch('/api/batch-predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tickers: tickers,
          days: 10,
          models: 'ensemble'
        })
      });

      if (response.ok) {
        const results = await response.json();
        setPortfolioAnalysisResults(results.batch_results || {});
        console.log('✅ Portfolio analysis completed:', results);
      } else {
        console.error('❌ Portfolio analysis failed');
      }
    } catch (error) {
      console.error('❌ Error analyzing portfolio:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const popularTickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-100/50 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900/20 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-cyan-400/10 to-blue-600/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-purple-400/10 to-indigo-600/10 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/3 left-1/4 w-60 h-60 bg-gradient-to-br from-emerald-400/5 to-cyan-600/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 p-8 space-y-8">
        {/* Enhanced Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex items-start space-x-6">
            <Link to="/">
              <Button variant="outline" size="sm" className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:bg-white dark:hover:bg-slate-700 border-white/20 dark:border-slate-700/20">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Home
              </Button>
            </Link>
            <div>
              <div className="flex items-center space-x-4 mb-2">
                <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <Brain className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  AI Investment Hub
                </h1>
              </div>
              <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
                Professional-grade AI-powered portfolio analysis, stock predictions, and machine learning models
              </p>
              <div className="flex flex-wrap gap-3 mt-4">
                <Badge className="bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300">
                  <Zap className="w-3 h-3 mr-1" />
                  XGBoost
                </Badge>
                <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                  <Brain className="w-3 h-3 mr-1" />
                  LSTM
                </Badge>
                <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  Prophet
                </Badge>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
              <Sparkles className="w-3 h-3 mr-1" />
              AI Enabled
            </Badge>
          </div>
        </div>

        {/* Enhanced Feature Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="group cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 overflow-hidden" 
                onClick={() => setActiveTab('forecast')}>
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-cyan-500/5 to-transparent group-hover:from-blue-500/20 group-hover:via-cyan-500/10 transition-all duration-300"></div>
            <CardContent className="relative p-6 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <TrendingUp className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-2">AI Predictions</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                XGBoost, LSTM & Prophet ensemble forecasts
              </p>
              <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 text-xs">
                Multi-Model
              </Badge>
            </CardContent>
          </Card>

          <Card className="group cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 overflow-hidden"
                onClick={() => setActiveTab('import')}>
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 via-emerald-500/5 to-transparent group-hover:from-green-500/20 group-hover:via-emerald-500/10 transition-all duration-300"></div>
            <CardContent className="relative p-6 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-2">Portfolio Import</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                CSV upload, manual entry & live market data
              </p>
              <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs">
                Real-time
              </Badge>
            </CardContent>
          </Card>

          <Card className="group cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 overflow-hidden"
                onClick={() => setActiveTab('training')}>
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-violet-500/5 to-transparent group-hover:from-purple-500/20 group-hover:via-violet-500/10 transition-all duration-300"></div>
            <CardContent className="relative p-6 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-2">Model Training</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                Train AI models for any stock ticker
              </p>
              <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 text-xs">
                Custom Models
              </Badge>
            </CardContent>
          </Card>

          <Card className="group cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 overflow-hidden"
                onClick={() => setActiveTab('analysis')}>
            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 via-amber-500/5 to-transparent group-hover:from-orange-500/20 group-hover:via-amber-500/10 transition-all duration-300"></div>
            <CardContent className="relative p-6 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-orange-500 to-amber-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                <Target className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-2">Smart Signals</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                BUY/SELL recommendations with confidence
              </p>
              <Badge className="bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300 text-xs">
                AI Driven
              </Badge>
            </CardContent>
          </Card>
        </div>

        {/* Enhanced Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 p-2">
            <TabsTrigger value="forecast" className="flex items-center space-x-2 data-[state=active]:bg-blue-100 data-[state=active]:text-blue-800 dark:data-[state=active]:bg-blue-900/30 dark:data-[state=active]:text-blue-300">
              <TrendingUp className="w-4 h-4" />
              <span className="hidden sm:inline">AI Forecast</span>
              <span className="sm:hidden">Forecast</span>
            </TabsTrigger>
            <TabsTrigger value="import" className="flex items-center space-x-2 data-[state=active]:bg-green-100 data-[state=active]:text-green-800 dark:data-[state=active]:bg-green-900/30 dark:data-[state=active]:text-green-300">
              <Upload className="w-4 h-4" />
              <span className="hidden sm:inline">Import Portfolio</span>
              <span className="sm:hidden">Import</span>
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center space-x-2 data-[state=active]:bg-purple-100 data-[state=active]:text-purple-800 dark:data-[state=active]:bg-purple-900/30 dark:data-[state=active]:text-purple-300">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Train Models</span>
              <span className="sm:hidden">Train</span>
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center space-x-2 data-[state=active]:bg-orange-100 data-[state=active]:text-orange-800 dark:data-[state=active]:bg-orange-900/30 dark:data-[state=active]:text-orange-300">
              <ChartBar className="w-4 h-4" />
              <span className="hidden sm:inline">Portfolio Analysis</span>
              <span className="sm:hidden">Analysis</span>
            </TabsTrigger>
          </TabsList>

          {/* Enhanced AI Forecast Tab */}
          <TabsContent value="forecast" className="space-y-6">
            <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
              <CardHeader className="bg-gradient-to-r from-blue-50 via-cyan-50 to-indigo-50 dark:from-blue-900/20 dark:via-cyan-900/20 dark:to-indigo-900/20 rounded-t-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg">
                      <TrendingUp className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-xl bg-gradient-to-r from-blue-700 to-cyan-700 dark:from-blue-300 dark:to-cyan-300 bg-clip-text text-transparent">
                        AI Stock Predictions
                      </CardTitle>
                      <CardDescription className="text-slate-600 dark:text-slate-400">
                        Ensemble predictions from XGBoost, LSTM, and Prophet models
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs">
                      Live
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-8">
                {/* Enhanced Ticker Selection */}
                <div className="mb-8">
                  <div className="flex items-center space-x-2 mb-4">
                    <Target className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                    <label className="text-lg font-semibold text-slate-900 dark:text-slate-100">Select Stock to Analyze</label>
                  </div>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-4">
                      <div className="flex-1 max-w-sm">
                        <TickerAutocomplete
                          value={selectedTicker}
                          onChange={setSelectedTicker}
                          onSelect={(suggestion) => setSelectedTicker(suggestion.symbol)}
                          placeholder="Enter ticker (e.g., AAPL)"
                          showPopular={true}
                          className="h-12 text-lg"
                        />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">Popular Stocks:</p>
                      <div className="flex flex-wrap gap-2">
                        {popularTickers.map(ticker => (
                          <Button
                            key={ticker}
                            variant={selectedTicker === ticker ? "default" : "outline"}
                            size="sm"
                            onClick={() => setSelectedTicker(ticker)}
                            className={`transition-all duration-200 ${
                              selectedTicker === ticker 
                                ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg' 
                                : 'hover:bg-blue-50 dark:hover:bg-blue-900/20'
                            }`}
                          >
                            {ticker}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Stock Forecast Chart */}
                <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg">
                  <StockForecastChart 
                    ticker={selectedTicker}
                    onTickerChange={handleTickerChange}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

        {/* Portfolio Import Tab */}
        <TabsContent value="import" className="space-y-6">
          <PortfolioUploader 
            onPortfolioImported={handlePortfolioImport}
            existingHoldings={importedPortfolio}
          />

          {/* Show imported portfolio summary */}
          {importedPortfolio.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Imported Portfolio Summary</CardTitle>
                <CardDescription>
                  {importedPortfolio.length} positions imported successfully
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {importedPortfolio.slice(0, 5).map(holding => (
                    <div key={holding.id} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span className="font-medium">{holding.ticker}</span>
                      <span>{holding.shares} shares @ ${holding.avgPrice.toFixed(2)}</span>
                      {holding.currentValue && (
                        <span className="text-green-600">
                          ${holding.currentValue.toLocaleString()}
                        </span>
                      )}
                    </div>
                  ))}
                  {importedPortfolio.length > 5 && (
                    <p className="text-sm text-gray-600">
                      ...and {importedPortfolio.length - 5} more positions
                    </p>
                  )}
                </div>
                
                <div className="mt-4 flex space-x-2">
                  <Button onClick={() => setActiveTab('analysis')}>
                    <ChartBar className="w-4 h-4 mr-2" />
                    Analyze Portfolio
                  </Button>
                  <Link to="/advanced-results">
                    <Button variant="outline">
                      <Target className="w-4 h-4 mr-2" />
                      Advanced Optimization
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Model Training Tab */}
        <TabsContent value="training" className="space-y-6">
          <ModelTrainingCenter />
        </TabsContent>

        {/* Portfolio Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          {importedPortfolio.length > 0 ? (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>AI Portfolio Analysis</CardTitle>
                  <CardDescription>
                    Get predictions and recommendations for your imported portfolio
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {importedPortfolio.slice(0, 6).map(holding => (
                      <Card key={holding.id} className="p-4">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-bold text-lg">{holding.ticker}</span>
                          <Badge variant="outline">
                            {holding.shares} shares
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600 mb-3">
                          Avg: ${holding.avgPrice.toFixed(2)}
                          {holding.currentValue && (
                            <span className={`ml-2 ${holding.gainLoss && holding.gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {holding.gainLoss && holding.gainLoss >= 0 ? '+' : ''}
                              ${holding.gainLoss?.toFixed(2)} ({holding.gainLossPercent?.toFixed(1)}%)
                            </span>
                          )}
                        </div>
                        <Button 
                          size="sm" 
                          variant="outline" 
                          className="w-full"
                          onClick={() => {
                            setSelectedTicker(holding.ticker);
                            setActiveTab('forecast');
                          }}
                        >
                          <TrendingUp className="w-4 h-4 mr-2" />
                          Get AI Forecast
                        </Button>
                      </Card>
                    ))}
                  </div>

                  <div className="mt-6 flex flex-wrap gap-4">
                    <Button 
                      onClick={analyzeBatchPortfolio}
                      disabled={isAnalyzing}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Brain className="w-4 h-4 mr-2" />
                          Run AI Analysis
                        </>
                      )}
                    </Button>
                    <Button onClick={() => setActiveTab('forecast')} variant="outline">
                      <TrendingUp className="w-4 h-4 mr-2" />
                      Individual Analysis
                    </Button>
                    <Link to="/advanced">
                      <Button variant="outline">
                        <Settings className="w-4 h-4 mr-2" />
                        Portfolio Optimization
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>

              {/* Batch Analysis Results */}
              {Object.keys(portfolioAnalysisResults).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="w-5 h-5 text-blue-600" />
                      <span>AI Portfolio Analysis Results</span>
                    </CardTitle>
                    <CardDescription>
                      Comprehensive AI predictions and recommendations for your portfolio
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(portfolioAnalysisResults).map(([ticker, result]: [string, AnalysisResult]) => {
                        if (result.error) {
                          return (
                            <Card key={ticker} className="border-red-200 bg-red-50 dark:bg-red-900/20">
                              <CardContent className="p-4">
                                <div className="flex justify-between items-center mb-2">
                                  <span className="font-bold text-lg">{ticker}</span>
                                  <Badge variant="destructive">Error</Badge>
                                </div>
                                <p className="text-sm text-red-600 dark:text-red-400">{result.error}</p>
                              </CardContent>
                            </Card>
                          );
                        }

                        const prediction = result.predictions?.ensemble || result.predictions?.prophet;
                        const signal = prediction?.trading_signal;
                        
                        if (!prediction || !signal) return null;

                        const signalColor = signal.action === 'BUY' || signal.action === 'STRONG_BUY' 
                          ? 'text-green-600 bg-green-50 border-green-200' 
                          : signal.action === 'SELL' || signal.action === 'STRONG_SELL'
                          ? 'text-red-600 bg-red-50 border-red-200'
                          : 'text-yellow-600 bg-yellow-50 border-yellow-200';

                        return (
                          <Card key={ticker} className="hover:shadow-lg transition-shadow">
                            <CardContent className="p-4">
                              <div className="flex justify-between items-center mb-3">
                                <span className="font-bold text-lg">{ticker}</span>
                                <Badge className={signalColor}>
                                  {signal.action}
                                </Badge>
                              </div>
                              
                              <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                  <span>Current:</span>
                                  <span className="font-medium">${result.stock_info?.current_price?.toFixed(2) || 'N/A'}</span>
                                </div>
                                
                                {signal.target_price && (
                                  <div className="flex justify-between">
                                    <span>Target:</span>
                                    <span className="font-medium text-green-600">${signal.target_price.toFixed(2)}</span>
                                  </div>
                                )}
                                
                                {signal.confidence && (
                                  <div className="flex justify-between">
                                    <span>Confidence:</span>
                                    <span className="font-medium">{signal.confidence.toFixed(0)}%</span>
                                  </div>
                                )}
                                
                                {prediction.summary?.avg_predicted_return && (
                                  <div className="flex justify-between">
                                    <span>Expected Return:</span>
                                    <span className={`font-medium ${prediction.summary.avg_predicted_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                      {(prediction.summary.avg_predicted_return * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                )}
                              </div>

                              <Button 
                                size="sm" 
                                variant="outline" 
                                className="w-full mt-3"
                                onClick={() => {
                                  setSelectedTicker(ticker);
                                  setActiveTab('forecast');
                                }}
                              >
                                <TrendingUp className="w-4 h-4 mr-2" />
                                View Details
                              </Button>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Database className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <h3 className="text-xl font-semibold mb-2">No Portfolio Imported</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Import your portfolio first to get AI-powered analysis and recommendations
                </p>
                <Button onClick={() => setActiveTab('import')}>
                  <Upload className="w-4 h-4 mr-2" />
                  Import Portfolio
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

        {/* Enhanced Quick Stats Footer */}
        <Card className="bg-gradient-to-r from-slate-100/80 via-blue-50/80 to-indigo-100/80 dark:from-slate-800/80 dark:via-slate-700/80 dark:to-indigo-900/20 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/5 to-transparent"></div>
          <CardContent className="relative p-8">
            <div className="text-center mb-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Platform Statistics</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">Real-time AI-powered investment platform metrics</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">3</div>
                <div className="text-sm text-slate-600 dark:text-slate-400">AI Models</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">XGBoost, LSTM, Prophet</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <Database className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">1000+</div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Stocks Supported</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Real-time data</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <Settings className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">74</div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Technical Indicators</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">Advanced analysis</div>
              </div>
              <div className="text-center group">
                <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-br from-orange-500 to-amber-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-1">44%</div>
                <div className="text-sm text-slate-600 dark:text-slate-400">LSTM Accuracy</div>
                <div className="text-xs text-slate-500 dark:text-slate-500">AAPL predictions</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AIHub;