import { useLocation, Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { ArrowLeft } from "lucide-react";
import { BarChart, Bar } from 'recharts';
import { PieChart, Pie, Cell, Legend } from 'recharts';
import { ScatterChart, Scatter, ZAxis } from 'recharts';
import { AreaChart, Area } from "recharts";


const Results = () => {
  const location = useLocation();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-slate-900">
        <Card className="max-w-md p-6 shadow-lg border border-gray-200 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">No Portfolio Data</CardTitle>
            <CardDescription className="text-sm text-slate-600 dark:text-slate-400">
              Submit a portfolio optimization request first.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/">
              <Button className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Form
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const metrics = {
    expected_return: result.expected_return,
    volatility: result.volatility,
    sharpe_ratio: result.sharpe_ratio,
    portfolio_value: result.portfolio_value || 100000  // fallback if not present
  };
  
  const composition = result.tickers.map((ticker: string, index: number) => ({
    ticker,
    weight: result.weights[index]
  }));
  
  const frontier = result.efficient_frontier || [];
  const growthData = result.portfolio_growth
  ? Object.entries(result.portfolio_growth).map(([date, growth]) => ({
      date,
      value: parseFloat((growth as number * metrics.portfolio_value).toFixed(2))
    }))
  : [];

  const forecastData = result.forecast_growth
  ? Object.entries(result.forecast_growth).map(([date, value]) => ({
      date,
      value: parseFloat((value as number).toFixed(2))
    }))
  : [];
  




  const colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4', '#84CC16', '#F97316'];

  const mergedGrowthData = [
    ...growthData.map(d => ({ ...d, type: "actual" })),
    ...forecastData.map(d => ({ ...d, type: "forecast" }))
  ];


  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8 space-y-10 transition-colors duration-300">
      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-4">
            <div className="text-sm text-gray-500 dark:text-slate-400">Expected Return</div>
            <div className="text-xl font-bold text-green-600 dark:text-green-400">
              {(metrics.expected_return * 100).toFixed(2)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-sm text-gray-500 dark:text-slate-400">Volatility</div>
            <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
              {(metrics.volatility * 100).toFixed(2)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-sm text-gray-500 dark:text-slate-400">Sharpe Ratio</div>
            <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
              {metrics.sharpe_ratio.toFixed(2)}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-sm text-gray-500 dark:text-slate-400">Portfolio Value</div>
            <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
              ${metrics.portfolio_value.toLocaleString()}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Efficient Frontier Chart */}
      {frontier && frontier.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">Efficient Frontier</CardTitle>
          </CardHeader>
          <CardContent className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="risk" label={{ value: 'Risk (%)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Return (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(val: number) => `${(val * 100).toFixed(2)}%`} />
                <Line type="monotone" dataKey="return" stroke="#3B82F6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {frontier && frontier.length > 0 && (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Sharpe Ratio vs Volatility (Bubble Chart)
          </CardTitle>
        </CardHeader>
        <CardContent className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="volatility"
                name="Volatility"
                tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
              <YAxis
                type="number"
                dataKey="return"
                name="Return"
                tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
              <ZAxis
                type="number"
                dataKey="sharpe_ratio"
                name="Sharpe Ratio"
                range={[60, 400]}
              />
              <Tooltip
                formatter={(val: number, name: string) => {
                  return name === 'sharpe_ratio'
                    ? val.toFixed(2)
                    : `${(val * 100).toFixed(2)}%`;
                }}
              />
              <Scatter
                name="Portfolios"
                data={frontier}
                fill="#6366F1"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    )}


      {/* Portfolio Composition */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">Portfolio Allocation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {composition.map((item: any, index: number) => (
            <div key={item.ticker} className="flex justify-between items-center">
              <div className="flex items-center space-x-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: colors[index % colors.length] }}
                ></div>
                <span className="text-sm text-slate-700 dark:text-slate-200">{item.ticker}</span>
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">
                {(item.weight * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Portfolio Weights Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Portfolio Weights (Bar Chart)
          </CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={composition}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ticker" />
              <YAxis />
              <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
              <Bar dataKey="weight">
                {composition.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      
      {/* Portfolio Composition Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Portfolio Composition (Pie Chart)
          </CardTitle>
        </CardHeader>
        <CardContent className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={composition}
                dataKey="weight"
                nameKey="ticker"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
              >
                {composition.map((entry, index) => (
                  <Cell key={`slice-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {growthData.length > 0 && (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Portfolio Value Over Time
          </CardTitle>
        </CardHeader>
        <CardContent className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={growthData}>
              <defs>
                <linearGradient id="valueFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis
                domain={['dataMin', 'dataMax']}
                tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
              />
              <CartesianGrid strokeDasharray="3 3" />
              <Tooltip formatter={(val: number) => `$${val.toFixed(2)}`} />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#3B82F6"
                fillOpacity={1}
                fill="url(#valueFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    )}

    {forecastData.length > 0 && (
  <Card>
    <CardHeader>
      <CardTitle className="text-lg font-semibold text-slate-800 dark:text-slate-100">Forecasted Portfolio Growth</CardTitle>
    </CardHeader>
    <CardContent className="h-96">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={forecastData}>
          <defs>
            <linearGradient id="colorBounds" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip formatter={(val: number) => `$${val.toFixed(2)}`} />
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#colorBounds)"
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#ffffff"
            activeDot={false}
          />
          <Line type="monotone" dataKey="value" stroke="#10B981" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
)}




    </div>
  );
};

export default Results;
