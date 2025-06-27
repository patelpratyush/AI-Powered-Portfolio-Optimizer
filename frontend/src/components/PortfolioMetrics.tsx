
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, Shield, DollarSign, BarChart } from "lucide-react";

const PortfolioMetrics = () => {
  const metrics = [
    {
      title: "Expected Return",
      value: "14.2%",
      change: "+2.1%",
      icon: TrendingUp,
      color: "text-green-600",
      bgColor: "bg-green-50"
    },
    {
      title: "Sharpe Ratio",
      value: "1.84",
      change: "+0.23",
      icon: BarChart,
      color: "text-blue-600",
      bgColor: "bg-blue-50"
    },
    {
      title: "Risk (Volatility)",
      value: "12.4%",
      change: "-1.2%",
      icon: Shield,
      color: "text-orange-600",
      bgColor: "bg-orange-50"
    },
    {
      title: "Portfolio Value",
      value: "$125,847",
      change: "+$8,234",
      icon: DollarSign,
      color: "text-purple-600",
      bgColor: "bg-purple-50"
    }
  ];

  return (
    <Card className="bg-white/70 backdrop-blur-sm border-slate-200 shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <BarChart className="w-5 h-5 text-blue-600" />
          <span>Portfolio Metrics</span>
        </CardTitle>
        <CardDescription>
          Key performance indicators for your optimized portfolio
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {metrics.map((metric) => (
            <div 
              key={metric.title}
              className="p-4 rounded-lg border border-slate-100 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between mb-2">
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  <metric.icon className={`w-4 h-4 ${metric.color}`} />
                </div>
                <span className={`text-xs font-medium ${metric.color}`}>
                  {metric.change}
                </span>
              </div>
              <div className="text-2xl font-bold text-slate-900 mb-1">
                {metric.value}
              </div>
              <div className="text-sm text-slate-600">
                {metric.title}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default PortfolioMetrics;
