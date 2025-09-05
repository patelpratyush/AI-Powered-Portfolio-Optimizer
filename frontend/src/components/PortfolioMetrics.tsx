
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
      bgColor: "bg-green-100",
      gradientBg: "bg-gradient-to-br from-green-50 to-emerald-50",
      borderColor: "border-green-200"
    },
    {
      title: "Sharpe Ratio",
      value: "1.84",
      change: "+0.23",
      icon: BarChart,
      color: "text-blue-600",
      bgColor: "bg-blue-100",
      gradientBg: "bg-gradient-to-br from-blue-50 to-indigo-50",
      borderColor: "border-blue-200"
    },
    {
      title: "Risk (Volatility)",
      value: "12.4%",
      change: "-1.2%",
      icon: Shield,
      color: "text-orange-600",
      bgColor: "bg-orange-100",
      gradientBg: "bg-gradient-to-br from-orange-50 to-amber-50",
      borderColor: "border-orange-200"
    },
    {
      title: "Portfolio Value",
      value: "$125,847",
      change: "+$8,234",
      icon: DollarSign,
      color: "text-purple-600",
      bgColor: "bg-purple-100",
      gradientBg: "bg-gradient-to-br from-purple-50 to-violet-50",
      borderColor: "border-purple-200"
    }
  ];

  return (
    <Card className="border-0 shadow-sm bg-white overflow-hidden">
      <CardHeader className="pb-8 bg-gradient-to-r from-slate-50 to-gray-50">
        <CardTitle className="flex items-center gap-3 text-xl font-semibold text-gray-900">
          <div className="p-2 bg-slate-100 rounded-lg">
            <BarChart className="w-5 h-5 text-slate-600" />
          </div>
          Portfolio Metrics
        </CardTitle>
        <CardDescription className="text-gray-600 mt-2 text-base">
          Key performance indicators for your optimized portfolio
        </CardDescription>
      </CardHeader>
      <CardContent className="p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {metrics.map((metric) => (
            <div 
              key={metric.title}
              className={`p-6 rounded-2xl border ${metric.borderColor} ${metric.gradientBg} hover:shadow-md transition-all duration-200 hover:scale-[1.02]`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-xl ${metric.bgColor}`}>
                  <metric.icon className={`w-5 h-5 ${metric.color}`} />
                </div>
                <div className="text-right">
                  <span className={`text-sm font-semibold px-3 py-1 rounded-full ${metric.color} ${metric.bgColor}/50`}>
                    {metric.change}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-3xl font-bold text-gray-900">
                  {metric.value}
                </div>
                <div className="text-sm font-medium text-gray-600">
                  {metric.title}
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default PortfolioMetrics;
