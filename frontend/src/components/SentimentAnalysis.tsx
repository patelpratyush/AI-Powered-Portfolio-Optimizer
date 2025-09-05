import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  BarChart3, 
  MessageCircle, 
  Newspaper,
  RefreshCw,
  AlertCircle,
  Info
} from 'lucide-react';
import { apiClient } from '@/lib/api-client';

interface SentimentScore {
  score: number;
  confidence: number;
  interpretation: {
    label: string;
    impact: string;
    confidence_level: string;
  };
  volume: number;
  source: string;
  last_updated: string;
}

interface SentimentAnalysisProps {
  ticker?: string;
  tickers?: string[];
  className?: string;
}

interface MarketSentiment {
  score: number;
  confidence: number;
  interpretation: {
    label: string;
    impact: string;
    confidence_level: string;
  };
  distribution: {
    positive: number;
    negative: number;
    neutral: number;
    total_sources: number;
  };
}

const SentimentAnalysis: React.FC<SentimentAnalysisProps> = ({ 
  ticker, 
  tickers, 
  className 
}) => {
  const [sentiment, setSentiment] = useState<SentimentScore | null>(null);
  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment | null>(null);
  const [portfolioSentiment, setPortfolioSentiment] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(7);

  const fetchSentiment = async () => {
    if (!ticker && !tickers?.length) return;

    setLoading(true);
    setError(null);

    try {
      if (ticker) {
        // Single ticker sentiment
        const response = await apiClient.get(`/api/sentiment/${ticker}?days=${days}`);
        setSentiment(response.data.sentiment);
      } else if (tickers?.length) {
        // Portfolio sentiment
        const response = await apiClient.post('/api/sentiment/portfolio', {
          tickers,
          days
        });
        setPortfolioSentiment(response.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to fetch sentiment analysis');
    } finally {
      setLoading(false);
    }
  };

  const fetchMarketSentiment = async () => {
    try {
      const response = await apiClient.get(`/api/sentiment/market?days=${days}`);
      setMarketSentiment(response.data.market_sentiment);
    } catch (err) {
      console.error('Failed to fetch market sentiment:', err);
    }
  };

  useEffect(() => {
    fetchSentiment();
    fetchMarketSentiment();
  }, [ticker, tickers, days]);

  const getSentimentIcon = (score: number) => {
    if (score > 0.1) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (score < -0.1) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Minus className="w-4 h-4 text-gray-500" />;
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.3) return 'bg-green-500';
    if (score > 0.1) return 'bg-green-300';
    if (score > -0.1) return 'bg-gray-300';
    if (score > -0.3) return 'bg-red-300';
    return 'bg-red-500';
  };

  const getSentimentBadgeVariant = (label: string) => {
    switch (label) {
      case 'bullish': return 'default';
      case 'slightly_positive': return 'secondary';
      case 'neutral': return 'outline';
      case 'slightly_negative': return 'destructive';
      case 'bearish': return 'destructive';
      default: return 'outline';
    }
  };

  const formatSentimentScore = (score: number): string => {
    return `${score > 0 ? '+' : ''}${(score * 100).toFixed(1)}%`;
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence > 0.7) return 'text-green-600';
    if (confidence > 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <Card className={`${className} border-0 shadow-sm bg-white`}>
        <CardHeader className="pb-8">
          <CardTitle className="flex items-center gap-3 text-xl font-semibold text-gray-900">
            <div className="p-2 bg-blue-100 rounded-lg">
              <MessageCircle className="w-5 h-5 text-blue-600" />
            </div>
            Sentiment Analysis
            <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />
          </CardTitle>
          <CardDescription className="text-gray-600 mt-2">
            AI-powered sentiment from news and social media
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="h-6 bg-gradient-to-r from-gray-200 to-gray-300 rounded-lg animate-pulse" />
              <div className="h-6 bg-gradient-to-r from-gray-200 to-gray-300 rounded-lg animate-pulse w-4/5" />
              <div className="h-6 bg-gradient-to-r from-gray-200 to-gray-300 rounded-lg animate-pulse w-3/5" />
            </div>
            <div className="flex justify-center py-8">
              <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`${className} border-0 shadow-sm bg-white`}>
        <CardHeader className="pb-6">
          <CardTitle className="flex items-center gap-3 text-xl font-semibold text-red-600">
            <div className="p-2 bg-red-100 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-600" />
            </div>
            Sentiment Analysis Error
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-6">
            <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-red-700 text-sm leading-relaxed">{error}</p>
            </div>
            <div className="flex justify-center">
              <Button 
                onClick={fetchSentiment} 
                variant="outline" 
                size="lg"
                className="px-8 py-3 border-red-200 text-red-600 hover:bg-red-50 rounded-xl"
              >
                <RefreshCw className="w-4 h-4 mr-3" />
                Try Again
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`${className} border-0 shadow-sm bg-white overflow-hidden`}>
      <CardHeader className="pb-8 bg-gradient-to-r from-blue-50 to-indigo-50">
        <CardTitle className="flex items-center gap-3 text-xl font-semibold text-gray-900">
          <div className="p-2 bg-blue-100 rounded-lg">
            <MessageCircle className="w-5 h-5 text-blue-600" />
          </div>
          Sentiment Analysis
        </CardTitle>
        <CardDescription className="text-gray-600 mt-2 text-base">
          AI-powered sentiment from news and social media
        </CardDescription>
      </CardHeader>
      <CardContent className="p-8">
        <Tabs defaultValue={ticker ? "individual" : "portfolio"} className="space-y-8">
          <TabsList className="grid w-full grid-cols-3 bg-gray-100 p-1 rounded-xl h-12">
            {ticker && (
              <TabsTrigger 
                value="individual"
                className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                {ticker} Sentiment
              </TabsTrigger>
            )}
            {tickers?.length && (
              <TabsTrigger 
                value="portfolio"
                className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                Portfolio
              </TabsTrigger>
            )}
            <TabsTrigger 
              value="market"
              className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
            >
              Market
            </TabsTrigger>
          </TabsList>

          {/* Individual Stock Sentiment */}
          {ticker && sentiment && (
            <TabsContent value="individual" className="space-y-8 mt-6">
              {/* Main Sentiment Display */}
              <div className="text-center space-y-6 p-8 bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-3">
                    {getSentimentIcon(sentiment.score)}
                    <span className="text-3xl font-bold text-gray-900">
                      {formatSentimentScore(sentiment.score)}
                    </span>
                  </div>
                  <Badge 
                    variant={getSentimentBadgeVariant(sentiment.interpretation.label)}
                    className="text-base px-4 py-2 rounded-full"
                  >
                    {sentiment.interpretation.label.replace('_', ' ')}
                  </Badge>
                </div>
                
                <div className="max-w-md mx-auto">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-gray-600">Sentiment Range</span>
                    <span className="text-sm font-medium text-gray-600">Confidence: 
                      <span className={`ml-1 ${getConfidenceColor(sentiment.confidence)}`}>
                        {(sentiment.confidence * 100).toFixed(0)}%
                      </span>
                    </span>
                  </div>
                  <div className="relative">
                    <Progress 
                      value={((sentiment.score + 1) / 2) * 100} 
                      className="h-3 bg-gray-200"
                    />
                    <div className="flex justify-between mt-2 text-xs text-gray-500">
                      <span>Bearish</span>
                      <span>Neutral</span>
                      <span>Bullish</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-6 bg-blue-50 rounded-xl border border-blue-100">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Newspaper className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">Data Sources</h3>
                      <p className="text-sm text-gray-600">News and social media analysis</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Sources</span>
                      <span className="font-semibold text-gray-900">{sentiment.volume}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Analysis Method</span>
                      <span className="text-sm font-medium text-blue-600">FinBERT AI</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-purple-50 rounded-xl border border-purple-100">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <BarChart3 className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">Impact Assessment</h3>
                      <p className="text-sm text-gray-600">Expected market influence</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Market Impact</span>
                      <span className="font-semibold text-gray-900 capitalize">
                        {sentiment.interpretation.impact.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Reliability</span>
                      <span className={`font-semibold ${getConfidenceColor(sentiment.confidence)}`}>
                        {sentiment.interpretation.confidence_level}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Analysis Period Info */}
              <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <Info className="w-5 h-5 text-green-600" />
                  </div>
                  <h3 className="font-semibold text-gray-900">Analysis Details</h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Analysis Period:</span>
                    <span className="ml-2 font-medium text-gray-900">Last {days} days</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Last Updated:</span>
                    <span className="ml-2 font-medium text-gray-900">
                      {new Date(sentiment.last_updated).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </TabsContent>
          )}

          {/* Portfolio Sentiment */}
          {tickers?.length && portfolioSentiment && (
            <TabsContent value="portfolio" className="space-y-8 mt-6">
              {/* Portfolio Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl border border-blue-200">
                  <div className="space-y-3">
                    <div className="flex items-center justify-center gap-3">
                      {getSentimentIcon(portfolioSentiment.portfolio_sentiment.overall_score)}
                      <span className="text-3xl font-bold text-gray-900">
                        {formatSentimentScore(portfolioSentiment.portfolio_sentiment.overall_score)}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-gray-600">Overall Sentiment</p>
                    <Badge 
                      variant={getSentimentBadgeVariant(portfolioSentiment.portfolio_sentiment.overall_interpretation?.label || 'neutral')}
                      className="px-3 py-1 rounded-full"
                    >
                      {portfolioSentiment.portfolio_sentiment.overall_interpretation?.label?.replace('_', ' ') || 'Neutral'}
                    </Badge>
                  </div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-2xl border border-indigo-200">
                  <div className="space-y-3">
                    <div className="text-3xl font-bold text-indigo-600">
                      {portfolioSentiment.portfolio_sentiment.tickers_analyzed}
                    </div>
                    <p className="text-sm font-medium text-gray-600">Stocks Analyzed</p>
                    <div className="text-xs text-gray-500">
                      {portfolioSentiment.portfolio_sentiment.total_volume || 0} total sources
                    </div>
                  </div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-2xl border border-green-200">
                  <div className="space-y-3">
                    <div className="text-3xl font-bold text-green-600">
                      {portfolioSentiment.portfolio_sentiment.high_confidence_tickers}
                    </div>
                    <p className="text-sm font-medium text-gray-600">High Confidence</p>
                    <div className="text-xs text-gray-500">
                      ≥70% confidence level
                    </div>
                  </div>
                </div>
              </div>

              {/* Individual Stock Breakdown */}
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-gray-600" />
                  Individual Stock Sentiment
                </h4>
                <div className="bg-gray-50 rounded-2xl p-6">
                  <div className="grid gap-3 max-h-80 overflow-y-auto">
                    {Object.entries(portfolioSentiment.individual_sentiment).map(([ticker, data]: [string, any]) => (
                      <div key={ticker} className="flex items-center justify-between p-4 bg-white rounded-xl border border-gray-200 hover:shadow-sm transition-shadow">
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-2">
                            {getSentimentIcon(data.score)}
                            <span className="font-semibold text-gray-900 text-lg">{ticker}</span>
                          </div>
                          <div className="text-sm text-gray-600">
                            {data.volume} sources
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge 
                            variant={getSentimentBadgeVariant(data.interpretation.label)}
                            className="px-3 py-1 rounded-full"
                          >
                            {formatSentimentScore(data.score)}
                          </Badge>
                          <div className="text-right">
                            <div className={`text-sm font-medium ${getConfidenceColor(data.confidence)}`}>
                              {(data.confidence * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-gray-500">confidence</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </TabsContent>
          )}

          {/* Market Sentiment */}
          {marketSentiment && (
            <TabsContent value="market" className="space-y-8 mt-6">
              {/* Market Overview */}
              <div className="text-center space-y-6 p-8 bg-gradient-to-br from-slate-50 to-slate-100 rounded-2xl">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-3">
                    {getSentimentIcon(marketSentiment.score)}
                    <span className="text-4xl font-bold text-gray-900">
                      {formatSentimentScore(marketSentiment.score)}
                    </span>
                  </div>
                  <Badge 
                    variant={getSentimentBadgeVariant(marketSentiment.interpretation.label)}
                    className="text-base px-6 py-2 rounded-full"
                  >
                    {marketSentiment.interpretation.label.replace('_', ' ')}
                  </Badge>
                  <p className="text-gray-600">Overall Market Sentiment</p>
                </div>
                
                <div className="max-w-lg mx-auto">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-gray-600">Market Range</span>
                    <span className={`text-sm font-medium ${getConfidenceColor(marketSentiment.confidence)}`}>
                      {(marketSentiment.confidence * 100).toFixed(0)}% Confidence
                    </span>
                  </div>
                  <Progress 
                    value={((marketSentiment.score + 1) / 2) * 100} 
                    className="h-3 bg-gray-200"
                  />
                  <div className="flex justify-between mt-2 text-xs text-gray-500">
                    <span>Bearish</span>
                    <span>Neutral</span>
                    <span>Bullish</span>
                  </div>
                </div>
              </div>

              {/* Sentiment Distribution */}
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-gray-600" />
                  Sentiment Distribution
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-xl border border-green-200">
                    <div className="space-y-2">
                      <TrendingUp className="w-8 h-8 text-green-600 mx-auto" />
                      <div className="text-3xl font-bold text-green-600">
                        {marketSentiment.distribution.positive}
                      </div>
                      <div className="text-sm font-medium text-gray-600">Positive Sources</div>
                      <div className="text-xs text-gray-500">
                        {marketSentiment.distribution.total_sources > 0 ? 
                          ((marketSentiment.distribution.positive / marketSentiment.distribution.total_sources) * 100).toFixed(1) : 0}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-center p-6 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border border-gray-200">
                    <div className="space-y-2">
                      <Minus className="w-8 h-8 text-gray-600 mx-auto" />
                      <div className="text-3xl font-bold text-gray-600">
                        {marketSentiment.distribution.neutral}
                      </div>
                      <div className="text-sm font-medium text-gray-600">Neutral Sources</div>
                      <div className="text-xs text-gray-500">
                        {marketSentiment.distribution.total_sources > 0 ? 
                          ((marketSentiment.distribution.neutral / marketSentiment.distribution.total_sources) * 100).toFixed(1) : 0}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-center p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border border-red-200">
                    <div className="space-y-2">
                      <TrendingDown className="w-8 h-8 text-red-600 mx-auto" />
                      <div className="text-3xl font-bold text-red-600">
                        {marketSentiment.distribution.negative}
                      </div>
                      <div className="text-sm font-medium text-gray-600">Negative Sources</div>
                      <div className="text-xs text-gray-500">
                        {marketSentiment.distribution.total_sources > 0 ? 
                          ((marketSentiment.distribution.negative / marketSentiment.distribution.total_sources) * 100).toFixed(1) : 0}%
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="text-center p-4 bg-blue-50 rounded-xl border border-blue-100">
                  <div className="text-sm text-blue-700">
                    <strong>Total Sources Analyzed:</strong> {marketSentiment.distribution.total_sources}
                  </div>
                </div>
              </div>
            </TabsContent>
          )}
        </Tabs>

        {/* Controls Section */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 mt-8 p-6 bg-gray-50 rounded-2xl">
          <div className="flex items-center gap-4">
            <label className="text-sm font-semibold text-gray-700">Analysis Period:</label>
            <select 
              value={days} 
              onChange={(e) => setDays(Number(e.target.value))}
              className="text-sm border-2 border-gray-200 rounded-lg px-4 py-2 bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-colors"
            >
              <option value={3}>3 days</option>
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
            </select>
          </div>
          <Button 
            onClick={() => { fetchSentiment(); fetchMarketSentiment(); }} 
            variant="outline" 
            size="lg"
            className="px-6 py-3 border-2 border-blue-200 text-blue-600 hover:bg-blue-50 rounded-xl transition-colors"
          >
            <RefreshCw className="w-4 h-4 mr-3" />
            Refresh Analysis
          </Button>
        </div>

        {/* Information Footer */}
        <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl border border-blue-100">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-blue-100 rounded-lg flex-shrink-0">
              <Info className="w-5 h-5 text-blue-600" />
            </div>
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-900">About Sentiment Analysis</h4>
              <div className="text-sm text-gray-700 leading-relaxed space-y-2">
                <p>
                  Our AI-powered sentiment analysis combines news articles and social media data using the 
                  <strong className="text-blue-700"> FinBERT</strong> model, specifically trained for financial content.
                </p>
                <p>
                  Higher confidence scores indicate more reliable predictions. This analysis provides valuable 
                  market insights but should be used alongside technical analysis and fundamental research.
                </p>
                <p className="text-xs text-blue-600 font-medium">
                  ⚠️ Not financial advice • For informational purposes only
                </p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default SentimentAnalysis;