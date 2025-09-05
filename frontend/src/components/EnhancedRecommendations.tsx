import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Brain, 
  BarChart3,
  Shield,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Activity,
  Zap,
  Eye,
  DollarSign
} from "lucide-react";

interface RecommendationReason {
  category: string;
  indicator: string;
  value: number;
  threshold: number;
  weight: number;
  description: string;
  bullish: boolean;
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
  reasons?: RecommendationReason[];
  summary?: string;
}

interface EnhancedRecommendationsProps {
  tradingSignal: TradingSignal;
  currentPrice: number;
  ticker: string;
}

const EnhancedRecommendations: React.FC<EnhancedRecommendationsProps> = ({
  tradingSignal,
  currentPrice,
  ticker
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const getSignalColor = (action: string) => {
    switch (action) {
      case 'STRONG_BUY': return 'text-green-800 bg-green-50 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-700';
      case 'BUY': return 'text-green-700 bg-green-50 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-700';
      case 'SELL': return 'text-red-700 bg-red-50 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-700';
      case 'STRONG_SELL': return 'text-red-800 bg-red-50 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-700';
      case 'HOLD': return 'text-yellow-700 bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-700';
      default: return 'text-gray-600 bg-gray-50 border-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-700';
    }
  };

  const getSignalIcon = (action: string) => {
    switch (action) {
      case 'STRONG_BUY': return <TrendingUp className="w-5 h-5" />;
      case 'BUY': return <TrendingUp className="w-5 h-5" />;
      case 'SELL': return <TrendingDown className="w-5 h-5" />;
      case 'STRONG_SELL': return <TrendingDown className="w-5 h-5" />;
      case 'HOLD': return <Target className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-300';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'momentum': return <Zap className="w-4 h-4" />;
      case 'trend': return <TrendingUp className="w-4 h-4" />;
      case 'volatility': return <Activity className="w-4 h-4" />;
      case 'volume': return <BarChart3 className="w-4 h-4" />;
      case 'support_resistance': return <Shield className="w-4 h-4" />;
      case 'valuation': return <DollarSign className="w-4 h-4" />;
      case 'growth': return <TrendingUp className="w-4 h-4" />;
      case 'profitability': return <Target className="w-4 h-4" />;
      case 'financial_health': return <Shield className="w-4 h-4" />;
      case 'ai_prediction': return <Brain className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'momentum': return 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300';
      case 'trend': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'volatility': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300';
      case 'volume': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'support_resistance': return 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300';
      case 'valuation': return 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300';
      case 'growth': return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300';
      case 'profitability': return 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300';
      case 'financial_health': return 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300';
      case 'ai_prediction': return 'bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
    }
  };

  const confidence = tradingSignal.confidence || tradingSignal.strength * 100;
  const reasons = tradingSignal.reasons || [];
  const bullishReasons = reasons.filter(r => r.bullish);
  const bearishReasons = reasons.filter(r => !r.bullish);

  return (
    <TooltipProvider>
      <Card className="border-0 shadow-sm bg-white overflow-hidden">
        <CardHeader className="pb-8 bg-gradient-to-r from-slate-50 to-gray-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${getSignalColor(tradingSignal.action)} shadow-sm`}>
                {getSignalIcon(tradingSignal.action)}
              </div>
              <div className="space-y-2">
                <CardTitle className="text-2xl font-bold text-gray-900">
                  {tradingSignal.action.replace('_', ' ')} SIGNAL
                </CardTitle>
                <CardDescription className="text-base text-gray-600 leading-relaxed max-w-md">
                  {tradingSignal.summary || tradingSignal.reasoning}
                </CardDescription>
              </div>
            </div>
            <div className="text-right space-y-2">
              <div className="text-sm font-medium text-gray-600">AI Confidence</div>
              <div className="text-4xl font-bold text-gray-900">
                {confidence.toFixed(0)}%
              </div>
              <div className="w-24">
                <Progress value={confidence} className="h-3 bg-gray-200" />
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="p-8">
          {/* Key Metrics Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            {tradingSignal.target_price && (
              <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl border border-green-200">
                <div className="text-sm font-medium text-gray-600 mb-2">Target Price</div>
                <div className="text-2xl font-bold text-green-600 mb-1">
                  ${tradingSignal.target_price.toFixed(2)}
                </div>
                <div className="text-sm text-green-700 font-medium">
                  {((tradingSignal.target_price - currentPrice) / currentPrice * 100).toFixed(1)}% upside
                </div>
              </div>
            )}

            {tradingSignal.stop_loss && (
              <div className="text-center p-6 bg-gradient-to-br from-red-50 to-rose-50 rounded-2xl border border-red-200">
                <div className="text-sm font-medium text-gray-600 mb-2">Stop Loss</div>
                <div className="text-2xl font-bold text-red-600 mb-1">
                  ${tradingSignal.stop_loss.toFixed(2)}
                </div>
                <div className="text-sm text-red-700 font-medium">
                  {((tradingSignal.stop_loss - currentPrice) / currentPrice * 100).toFixed(1)}% risk
                </div>
              </div>
            )}

            {tradingSignal.risk_level && (
              <div className="text-center p-6 bg-gradient-to-br from-orange-50 to-amber-50 rounded-2xl border border-orange-200">
                <div className="text-sm font-medium text-gray-600 mb-2">Risk Level</div>
                <div className="flex justify-center mb-1">
                  <Badge className={`text-base font-bold px-4 py-2 rounded-full ${getRiskLevelColor(tradingSignal.risk_level)}`}>
                    <Shield className="w-4 h-4 mr-2" />
                    {tradingSignal.risk_level.toUpperCase()}
                  </Badge>
                </div>
              </div>
            )}

            {tradingSignal.time_horizon && (
              <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-200">
                <div className="text-sm font-medium text-gray-600 mb-2">Time Horizon</div>
                <div className="text-2xl font-bold text-blue-600">
                  {tradingSignal.time_horizon.charAt(0).toUpperCase() + tradingSignal.time_horizon.slice(1)}
                </div>
              </div>
            )}
          </div>

          {/* Toggle Details Button */}
          <div className="flex justify-center mb-8">
            <Button
              variant="outline"
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-3 px-6 py-3 rounded-xl border-2 border-gray-200 text-gray-700 hover:bg-gray-50 font-medium"
              size="lg"
            >
              <Eye className="w-5 h-5" />
              <span>{showDetails ? 'Hide' : 'Show'} Detailed Analysis</span>
              {showDetails ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </Button>
          </div>

          {/* Detailed Analysis */}
          {showDetails && reasons.length > 0 && (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3 bg-gray-100 p-1 rounded-xl h-12">
                <TabsTrigger 
                  value="overview"
                  className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
                >
                  Overview
                </TabsTrigger>
                <TabsTrigger 
                  value="bullish"
                  className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
                >
                  Bullish ({bullishReasons.length})
                </TabsTrigger>
                <TabsTrigger 
                  value="bearish"
                  className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm"
                >
                  Bearish ({bearishReasons.length})
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="mt-8">
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-0 shadow-sm bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200">
                      <CardContent className="p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <div className="p-2 bg-green-100 rounded-lg">
                            <TrendingUp className="w-5 h-5 text-green-600" />
                          </div>
                          <span className="font-semibold text-green-800 text-lg">
                            Bullish Factors
                          </span>
                        </div>
                        <div className="text-3xl font-bold text-green-600 mb-2">
                          {bullishReasons.length}
                        </div>
                        <div className="text-sm text-green-700 leading-relaxed">
                          Supporting {tradingSignal.action.includes('BUY') ? 'buy' : 'hold'} recommendation
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700">
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-2 mb-2">
                          <TrendingDown className="w-5 h-5 text-red-600" />
                          <span className="font-semibold text-red-800 dark:text-red-300">
                            Bearish Factors
                          </span>
                        </div>
                        <div className="text-2xl font-bold text-red-600 mb-1">
                          {bearishReasons.length}
                        </div>
                        <div className="text-sm text-red-700 dark:text-red-400">
                          Risk factors to consider
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Top weighted reasons */}
                  <div>
                    <h4 className="font-semibold mb-3">Key Decision Factors</h4>
                    <div className="space-y-3">
                      {reasons
                        .sort((a, b) => b.weight - a.weight)
                        .slice(0, 3)
                        .map((reason, index) => (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getCategoryColor(reason.category)}`}>
                              {getCategoryIcon(reason.category)}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-1">
                                <span className="font-medium">{reason.indicator}</span>
                                <Badge className={`text-xs ${getCategoryColor(reason.category)}`}>
                                  {reason.category}
                                </Badge>
                                <Badge variant={reason.bullish ? "default" : "destructive"} className="text-xs">
                                  {reason.bullish ? 'BULLISH' : 'BEARISH'}
                                </Badge>
                              </div>
                              <p className="text-sm text-slate-600 dark:text-slate-400">
                                {reason.description}
                              </p>
                              <div className="flex items-center space-x-2 mt-2">
                                <span className="text-xs text-slate-500">Weight: {(reason.weight * 100).toFixed(0)}%</span>
                                <span className="text-xs text-slate-500">
                                  Value: {typeof reason.value === 'number' ? reason.value.toFixed(2) : reason.value}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="bullish" className="mt-4">
                <div className="space-y-3">
                  {bullishReasons.length > 0 ? (
                    bullishReasons
                      .sort((a, b) => b.weight - a.weight)
                      .map((reason, index) => (
                        <div key={index} className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700 rounded-lg">
                          <div className="flex items-start space-x-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getCategoryColor(reason.category)}`}>
                              {getCategoryIcon(reason.category)}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h5 className="font-semibold text-green-800 dark:text-green-300">
                                  {reason.indicator}
                                </h5>
                                <Badge className={`text-xs ${getCategoryColor(reason.category)}`}>
                                  {reason.category}
                                </Badge>
                              </div>
                              <p className="text-sm text-green-700 dark:text-green-400 mb-2">
                                {reason.description}
                              </p>
                              <div className="flex items-center space-x-4 text-xs text-green-600 dark:text-green-500">
                                <span>Weight: {(reason.weight * 100).toFixed(0)}%</span>
                                <span>Current: {typeof reason.value === 'number' ? reason.value.toFixed(2) : reason.value}</span>
                                <span>Threshold: {typeof reason.threshold === 'number' ? reason.threshold.toFixed(2) : reason.threshold}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                  ) : (
                    <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                      <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>No bullish factors identified in current analysis</p>
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="bearish" className="mt-4">
                <div className="space-y-3">
                  {bearishReasons.length > 0 ? (
                    bearishReasons
                      .sort((a, b) => b.weight - a.weight)
                      .map((reason, index) => (
                        <div key={index} className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-lg">
                          <div className="flex items-start space-x-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getCategoryColor(reason.category)}`}>
                              {getCategoryIcon(reason.category)}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h5 className="font-semibold text-red-800 dark:text-red-300">
                                  {reason.indicator}
                                </h5>
                                <Badge className={`text-xs ${getCategoryColor(reason.category)}`}>
                                  {reason.category}
                                </Badge>
                              </div>
                              <p className="text-sm text-red-700 dark:text-red-400 mb-2">
                                {reason.description}
                              </p>
                              <div className="flex items-center space-x-4 text-xs text-red-600 dark:text-red-500">
                                <span>Weight: {(reason.weight * 100).toFixed(0)}%</span>
                                <span>Current: {typeof reason.value === 'number' ? reason.value.toFixed(2) : reason.value}</span>
                                <span>Threshold: {typeof reason.threshold === 'number' ? reason.threshold.toFixed(2) : reason.threshold}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                  ) : (
                    <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                      <TrendingDown className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>No bearish factors identified in current analysis</p>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default EnhancedRecommendations;