
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Percent } from "lucide-react";

interface PortfolioCompositionProps {
  tickers: string[];
}

const PortfolioComposition = ({ tickers }: PortfolioCompositionProps) => {
  // Mock weights for demonstration
  const getRandomWeight = () => Math.random() * 40 + 10;
  const weights = tickers.map(() => getRandomWeight());
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const normalizedWeights = weights.map(weight => (weight / totalWeight) * 100);

  const colors = [
    'bg-blue-500',
    'bg-green-500', 
    'bg-purple-500',
    'bg-orange-500',
    'bg-pink-500',
    'bg-indigo-500',
    'bg-red-500',
    'bg-yellow-500'
  ];

  return (
    <Card className="bg-white/70 backdrop-blur-sm border-slate-200 shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <PieChart className="w-5 h-5 text-blue-600" />
          <span>Portfolio Composition</span>
        </CardTitle>
        <CardDescription>
          Optimized asset allocation weights
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-64 bg-gradient-to-br from-slate-50 to-blue-50 rounded-lg border border-slate-200 flex items-center justify-center relative overflow-hidden mb-4">
          {/* Mock pie chart visualization */}
          <div className="w-32 h-32 rounded-full bg-gradient-to-br from-blue-400 via-purple-400 to-green-400 opacity-80 relative">
            <div className="absolute inset-4 bg-white rounded-full flex items-center justify-center">
              <div className="text-center">
                <PieChart className="w-6 h-6 text-blue-600 mx-auto mb-1" />
                <p className="text-xs text-slate-600">Optimized</p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {tickers.map((ticker, index) => (
            <div key={ticker} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${colors[index % colors.length]}`}></div>
                <span className="font-medium text-slate-700">{ticker}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-semibold text-slate-900">
                  {normalizedWeights[index].toFixed(1)}%
                </span>
                <Percent className="w-3 h-3 text-slate-400" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default PortfolioComposition;
