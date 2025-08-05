import React from 'react';
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
  ComposedChart
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Target, Shield, Activity, AlertTriangle } from "lucide-react";

// Enhanced color palettes
const colorPalettes = {
  primary: ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4', '#84CC16', '#F97316'],
  gradient: ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4', '#84CC16', '#F97316'],
  performance: {
    positive: '#10B981',
    negative: '#EF4444',
    neutral: '#6B7280'
  },
  risk: {
    low: '#10B981',
    medium: '#F59E0B', 
    high: '#EF4444'
  }
};

interface EnhancedTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: any;
  formatter?: (value: any, name: string) => [string, string];
  labelFormatter?: (label: any) => string;
  contentStyle?: React.CSSProperties;
}

// Custom Tooltip Component
const EnhancedTooltip: React.FC<EnhancedTooltipProps> = ({ 
  active, 
  payload, 
  label, 
  formatter,
  labelFormatter 
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-lg border border-slate-200 dark:border-slate-600 backdrop-blur-sm">
        {label && (
          <p className="font-semibold text-slate-900 dark:text-slate-100 mb-2">
            {labelFormatter ? labelFormatter(label) : label}
          </p>
        )}
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center space-x-2 mb-1">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-slate-600 dark:text-slate-400">
              {entry.name}:
            </span>
            <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
              {formatter ? formatter(entry.value, entry.name)[0] : entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// Enhanced Performance Metrics Card
interface PerformanceMetricsProps {
  metrics: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    sortino_ratio?: number;
    downside_deviation?: number;
  };
}

export const PerformanceMetricsCard: React.FC<PerformanceMetricsProps> = ({ metrics }) => {
  const getPerformanceColor = (value: number, type: 'return' | 'ratio' | 'risk') => {
    switch (type) {
      case 'return':
        return value > 0 ? colorPalettes.performance.positive : colorPalettes.performance.negative;
      case 'ratio':
        return value > 1 ? colorPalettes.performance.positive : 
               value > 0.5 ? colorPalettes.performance.neutral : colorPalettes.performance.negative;
      case 'risk':
        return value < 0.15 ? colorPalettes.risk.low :
               value < 0.25 ? colorPalettes.risk.medium : colorPalettes.risk.high;
      default:
        return colorPalettes.performance.neutral;
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatRatio = (value: number) => value.toFixed(3);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
      {/* Expected Return */}
      <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/10 dark:to-emerald-900/10" />
        <CardContent className="relative p-6">
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-green-600" />
            <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
              Annual
            </Badge>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-green-700 dark:text-green-300">Expected Return</p>
            <p className="text-3xl font-bold" style={{ color: getPerformanceColor(metrics.expected_return, 'return') }}>
              {formatPercentage(metrics.expected_return)}
            </p>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-xs text-green-600 dark:text-green-400">Annualized</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Volatility */}
      <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-900/10 dark:to-amber-900/10" />
        <CardContent className="relative p-6">
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8 text-orange-600" />
            <Badge variant="secondary" className="bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300">
              Risk
            </Badge>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-orange-700 dark:text-orange-300">Volatility</p>
            <p className="text-3xl font-bold" style={{ color: getPerformanceColor(metrics.volatility, 'risk') }}>
              {formatPercentage(metrics.volatility)}
            </p>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-orange-500" />
              <span className="text-xs text-orange-600 dark:text-orange-400">Standard Deviation</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sharpe Ratio */}
      <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/10 dark:to-cyan-900/10" />
        <CardContent className="relative p-6">
          <div className="flex items-center justify-between mb-4">
            <Target className="w-8 h-8 text-blue-600" />
            <Badge variant="secondary" className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
              Efficiency
            </Badge>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Sharpe Ratio</p>
            <p className="text-3xl font-bold" style={{ color: getPerformanceColor(metrics.sharpe_ratio, 'ratio') }}>
              {formatRatio(metrics.sharpe_ratio)}
            </p>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="text-xs text-blue-600 dark:text-blue-400">Risk-Adjusted Return</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sortino Ratio */}
      {metrics.sortino_ratio !== undefined && (
        <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/10 dark:to-violet-900/10" />
          <CardContent className="relative p-6">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-purple-600" />
              <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                Downside
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Sortino Ratio</p>
              <p className="text-3xl font-bold" style={{ color: getPerformanceColor(metrics.sortino_ratio, 'ratio') }}>
                {formatRatio(metrics.sortino_ratio)}
              </p>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-purple-500" />
                <span className="text-xs text-purple-600 dark:text-purple-400">Downside Risk Focus</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Downside Deviation */}
      {metrics.downside_deviation !== undefined && (
        <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/10 dark:to-pink-900/10" />
          <CardContent className="relative p-6">
            <div className="flex items-center justify-between mb-4">
              <TrendingDown className="w-8 h-8 text-red-600" />
              <Badge variant="secondary" className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300">
                Downside
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-red-700 dark:text-red-300">Downside Deviation</p>
              <p className="text-3xl font-bold" style={{ color: getPerformanceColor(metrics.downside_deviation, 'risk') }}>
                {formatPercentage(metrics.downside_deviation)}
              </p>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-xs text-red-600 dark:text-red-400">Negative Returns Only</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

// Enhanced Portfolio Allocation Chart
interface AllocationChartProps {
  data: Array<{
    ticker: string;
    weight: number;
    percentage: string;
  }>;
  title?: string;
}

export const EnhancedAllocationChart: React.FC<AllocationChartProps> = ({ data, title = "Portfolio Allocation" }) => {
  const RADIAN = Math.PI / 180;
  
  const renderCustomizedLabel = ({
    cx, cy, midAngle, innerRadius, outerRadius, percent, ticker
  }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return percent > 0.05 ? (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        className="font-semibold text-sm"
      >
        {`${ticker}`}
      </text>
    ) : null;
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg" />
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <defs>
                  {colorPalettes.gradient.map((color, index) => (
                    <linearGradient key={index} id={`gradient-${index}`} x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor={color} stopOpacity={0.8} />
                      <stop offset="100%" stopColor={color} stopOpacity={1} />
                    </linearGradient>
                  ))}
                </defs>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="weight"
                  stroke="#fff"
                  strokeWidth={2}
                >
                  {data.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`url(#gradient-${index % colorPalettes.gradient.length})`}
                    />
                  ))}
                </Pie>
                <Tooltip 
                  content={<EnhancedTooltip 
                    formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Weight']}
                  />}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          {/* Allocation Details */}
          <div className="space-y-4">
            <h4 className="font-semibold text-lg text-slate-900 dark:text-slate-100">Asset Breakdown</h4>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {data.map((item, index) => (
                <div key={item.ticker} className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded-full shadow-sm"
                      style={{ backgroundColor: colorPalettes.gradient[index % colorPalettes.gradient.length] }}
                    />
                    <div>
                      <span className="font-medium text-slate-900 dark:text-slate-100">{item.ticker}</span>
                      <div className="text-xs text-slate-600 dark:text-slate-400">
                        ${((item.weight * 100000)).toLocaleString()} @ 100K
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-slate-900 dark:text-slate-100">{item.percentage}%</div>
                    <div className="w-16 bg-slate-200 dark:bg-slate-600 rounded-full h-2 mt-1">
                      <div 
                        className="h-2 rounded-full transition-all duration-500"
                        style={{ 
                          width: `${item.weight * 100}%`,
                          backgroundColor: colorPalettes.gradient[index % colorPalettes.gradient.length]
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Enhanced Efficient Frontier Chart
interface EfficientFrontierProps {
  data: Array<{
    return: number;
    volatility: number;
    sharpe_ratio: number;
    sortino_ratio?: number;
  }>;
  currentPortfolio?: {
    return: number;
    volatility: number;
    sharpe_ratio: number;
  };
}

export const EnhancedEfficientFrontier: React.FC<EfficientFrontierProps> = ({ data, currentPortfolio }) => {
  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg" />
          <span>Enhanced Efficient Frontier</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={data} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
              <defs>
                <linearGradient id="frontierGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="50%" stopColor="#EC4899" />
                  <stop offset="100%" stopColor="#F59E0B" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                type="number" 
                dataKey="volatility" 
                name="Risk (Volatility)"
                tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                domain={['dataMin - 0.01', 'dataMax + 0.01']}
              />
              <YAxis 
                type="number" 
                dataKey="return" 
                name="Expected Return"
                tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                domain={['dataMin - 0.01', 'dataMax + 0.01']}
              />
              <Tooltip 
                content={<EnhancedTooltip 
                  formatter={(value: number, name: string) => {
                    if (name === 'return' || name === 'volatility') {
                      return [`${(value * 100).toFixed(2)}%`, name === 'return' ? 'Expected Return' : 'Volatility'];
                    }
                    return [value.toFixed(3), name === 'sharpe_ratio' ? 'Sharpe Ratio' : 'Sortino Ratio'];
                  }}
                />}
              />
              <Scatter 
                name="Efficient Portfolios" 
                dataKey="sharpe_ratio" 
                fill="url(#frontierGradient)"
                r={6}
              />
              {currentPortfolio && (
                <Scatter 
                  data={[currentPortfolio]}
                  fill="#EF4444"
                  r={10}
                  name="Your Portfolio"
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500" />
            <span>Efficient Frontier</span>
          </div>
          {currentPortfolio && (
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>Your Portfolio</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Enhanced Growth Chart
interface GrowthChartProps {
  data: Array<{
    date: string;
    value: number;
  }>;
  forecastData?: Array<{
    date: string;
    value: number;
    lower?: number;
    upper?: number;
  }>;
  title?: string;
}

export const EnhancedGrowthChart: React.FC<GrowthChartProps> = ({ 
  data, 
  forecastData, 
  title = "Portfolio Growth Over Time" 
}) => {
  const combinedData = [
    ...data.map(d => ({ ...d, type: 'historical' })),
    ...(forecastData || []).map(d => ({ ...d, type: 'forecast' }))
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-green-500 to-blue-600 rounded-lg" />
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={combinedData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
              <defs>
                <linearGradient id="growthGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.2}/>
                </linearGradient>
                <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.2}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                content={<EnhancedTooltip 
                  formatter={(value: number, name: string) => [`${(value * 100).toFixed(2)}%`, 'Growth']}
                  labelFormatter={(label) => `Date: ${label}`}
                />}
              />
              
              {/* Historical Area */}
              <Area
                type="monotone"
                dataKey="value"
                stroke="#10B981"
                fillOpacity={1}
                fill="url(#growthGradient)"
                strokeWidth={3}
                dot={false}
                connectNulls={false}
              />
              
              {/* Forecast Area */}
              {forecastData && (
                <>
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="none"
                    fillOpacity={0.3}
                    fill="#3B82F6"
                    connectNulls={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="none"
                    fillOpacity={0.3}
                    fill="#ffffff"
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    connectNulls={false}
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Historical Performance</span>
          </div>
          {forecastData && (
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span>AI Forecast</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Enhanced Risk Metrics Card
interface RiskMetricsProps {
  riskDecomposition: {
    total_risk: number;
    risk_concentration: number;
    percentage_contributions: number[];
    marginal_contributions: number[];
  };
  tickers: string[];
}

export const EnhancedRiskMetrics: React.FC<RiskMetricsProps> = ({ riskDecomposition, tickers }) => {
  const riskContributionData = tickers.map((ticker, index) => ({
    ticker,
    contribution: riskDecomposition.percentage_contributions[index] * 100,
    marginal: riskDecomposition.marginal_contributions[index],
    color: colorPalettes.gradient[index % colorPalettes.gradient.length]
  }));

  const diversificationScore = (1 - riskDecomposition.risk_concentration) * 100;

  return (
    <div className="space-y-6">
      {/* Risk Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/10 dark:to-pink-900/10" />
          <CardContent className="relative p-6">
            <div className="flex items-center justify-between mb-4">
              <Shield className="w-8 h-8 text-red-600" />
              <Badge variant="secondary" className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300">
                Total Risk
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-red-700 dark:text-red-300">Portfolio Risk</p>
              <p className="text-3xl font-bold text-red-800 dark:text-red-200">
                {(riskDecomposition.total_risk * 100).toFixed(2)}%
              </p>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-xs text-red-600 dark:text-red-400">Annual Volatility</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-900/10 dark:to-amber-900/10" />
          <CardContent className="relative p-6">
            <div className="flex items-center justify-between mb-4">
              <Target className="w-8 h-8 text-orange-600" />
              <Badge variant="secondary" className="bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300">
                Concentration
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-orange-700 dark:text-orange-300">Risk Concentration</p>
              <p className="text-3xl font-bold text-orange-800 dark:text-orange-200">
                {(riskDecomposition.risk_concentration * 100).toFixed(1)}%
              </p>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-orange-500" />
                <span className="text-xs text-orange-600 dark:text-orange-400">Herfindahl Index</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/10 dark:to-emerald-900/10" />
          <CardContent className="relative p-6">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-green-600" />
              <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                Diversification
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-green-700 dark:text-green-300">Diversification Score</p>
              <p className="text-3xl font-bold text-green-800 dark:text-green-200">
                {diversificationScore.toFixed(0)}%
              </p>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-xs text-green-600 dark:text-green-400">Portfolio Balance</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Contribution Chart */}
      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-red-500 to-orange-600 rounded-lg" />
            <span>Risk Contribution Analysis</span>
          </CardTitle>
          <CardDescription>
            How much each asset contributes to total portfolio risk
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskContributionData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
                <defs>
                  {riskContributionData.map((_, index) => (
                    <linearGradient key={index} id={`riskGradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={colorPalettes.gradient[index % colorPalettes.gradient.length]} stopOpacity={0.8} />
                      <stop offset="100%" stopColor={colorPalettes.gradient[index % colorPalettes.gradient.length]} stopOpacity={0.6} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="ticker" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(value) => `${value.toFixed(1)}%`} tick={{ fontSize: 12 }} />
                <Tooltip 
                  content={<EnhancedTooltip 
                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Risk Contribution']}
                  />}
                />
                <Bar dataKey="contribution" radius={[4, 4, 0, 0]}>
                  {riskContributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`url(#riskGradient-${index})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Enhanced Monte Carlo Results
interface MonteCarloResultsProps {
  monteCarlo: {
    expected_return: number;
    volatility: number;
    var_95: number;
    var_99: number;
    cvar_95: number;
    cvar_99: number;
    prob_loss: number;
    prob_outperform_market: number;
    max_drawdowns: number[];
  };
}

export const EnhancedMonteCarloResults: React.FC<MonteCarloResultsProps> = ({ monteCarlo }) => {
  const metricsData = [
    {
      label: "Expected Return",
      value: `${(monteCarlo.expected_return * 100).toFixed(2)}%`,
      icon: TrendingUp,
      color: "green",
      description: "Annualized expected return"
    },
    {
      label: "Volatility",
      value: `${(monteCarlo.volatility * 100).toFixed(2)}%`,
      icon: Activity,
      color: "orange",
      description: "Standard deviation of returns"
    },
    {
      label: "VaR (95%)",
      value: `${((1 - monteCarlo.var_95) * 100).toFixed(2)}%`,
      icon: Shield,
      color: "red",
      description: "Maximum loss at 95% confidence"
    },
    {
      label: "Expected Shortfall",
      value: `${((1 - monteCarlo.cvar_95) * 100).toFixed(2)}%`,
      icon: AlertTriangle,
      color: "red",
      description: "Average loss beyond VaR"
    },
    {
      label: "Probability of Loss",
      value: `${(monteCarlo.prob_loss * 100).toFixed(1)}%`,
      icon: TrendingDown,
      color: "red", 
      description: "Chance of negative returns"
    },
    {
      label: "Beat Market Odds",
      value: `${(monteCarlo.prob_outperform_market * 100).toFixed(1)}%`,
      icon: Target,
      color: "blue",
      description: "Probability of outperforming market"
    }
  ];

  const getColorClasses = (color: string) => {
    const colorMap = {
      green: {
        bg: "from-green-50 to-emerald-50 dark:from-green-900/10 dark:to-emerald-900/10",
        border: "border-green-200 dark:border-green-700",
        text: "text-green-700 dark:text-green-300",
        value: "text-green-800 dark:text-green-200",
        icon: "text-green-600",
        badge: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
      },
      orange: {
        bg: "from-orange-50 to-amber-50 dark:from-orange-900/10 dark:to-amber-900/10",
        border: "border-orange-200 dark:border-orange-700",
        text: "text-orange-700 dark:text-orange-300",
        value: "text-orange-800 dark:text-orange-200",
        icon: "text-orange-600",
        badge: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
      },
      red: {
        bg: "from-red-50 to-pink-50 dark:from-red-900/10 dark:to-pink-900/10",
        border: "border-red-200 dark:border-red-700",
        text: "text-red-700 dark:text-red-300",
        value: "text-red-800 dark:text-red-200",
        icon: "text-red-600",
        badge: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
      },
      blue: {
        bg: "from-blue-50 to-cyan-50 dark:from-blue-900/10 dark:to-cyan-900/10",
        border: "border-blue-200 dark:border-blue-700",
        text: "text-blue-700 dark:text-blue-300",
        value: "text-blue-800 dark:text-blue-200",
        icon: "text-blue-600",
        badge: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
      }
    };
    return colorMap[color as keyof typeof colorMap];
  };

  return (
    <div className="space-y-6">
      {/* Monte Carlo Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metricsData.map((metric, index) => {
          const Icon = metric.icon;
          const colors = getColorClasses(metric.color);
          
          return (
            <Card key={index} className={`relative overflow-hidden group hover:shadow-lg transition-all duration-300 ${colors.border}`}>
              <div className={`absolute inset-0 bg-gradient-to-br ${colors.bg}`} />
              <CardContent className="relative p-6">
                <div className="flex items-center justify-between mb-4">
                  <Icon className={`w-8 h-8 ${colors.icon}`} />
                  <Badge variant="secondary" className={colors.badge}>
                    Monte Carlo
                  </Badge>
                </div>
                <div className="space-y-2">
                  <p className={`text-sm font-medium ${colors.text}`}>{metric.label}</p>
                  <p className={`text-3xl font-bold ${colors.value}`}>
                    {metric.value}
                  </p>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full bg-${metric.color}-500`} />
                    <span className={`text-xs ${colors.text}`}>{metric.description}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Risk Distribution Chart */}
      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg" />
            <span>Risk Distribution Analysis</span>
          </CardTitle>
          <CardDescription>
            Distribution of potential maximum drawdowns from Monte Carlo simulation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-semibold text-lg">Risk Metrics Summary</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-800">
                  <span className="text-sm font-medium">Average Max Drawdown</span>
                  <span className="font-bold text-red-600">
                    {(Math.abs(monteCarlo.max_drawdowns.reduce((a, b) => a + b, 0) / monteCarlo.max_drawdowns.length) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-800">
                  <span className="text-sm font-medium">Worst Case Scenario</span>
                  <span className="font-bold text-red-600">
                    {(Math.abs(Math.min(...monteCarlo.max_drawdowns)) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-800">
                  <span className="text-sm font-medium">Best Case Scenario</span>
                  <span className="font-bold text-green-600">
                    {(Math.abs(Math.max(...monteCarlo.max_drawdowns)) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold text-lg">Success Probabilities</h4>
              <div className="space-y-3">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Probability of Profit</span>
                    <span className="font-bold text-green-600">
                      {((1 - monteCarlo.prob_loss) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full bg-green-500 transition-all duration-500"
                      style={{ width: `${(1 - monteCarlo.prob_loss) * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Outperform Market</span>
                    <span className="font-bold text-blue-600">
                      {(monteCarlo.prob_outperform_market * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${monteCarlo.prob_outperform_market * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default {
  PerformanceMetricsCard,
  EnhancedAllocationChart,
  EnhancedEfficientFrontier,
  EnhancedGrowthChart,
  EnhancedRiskMetrics,
  EnhancedMonteCarloResults
};