import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Target, BarChart3, Loader2, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import StockTickerInput from "./StockTickerInput";
import DateRangeSelector from "./DateRangeSelector";
import axios from 'axios';

interface StrategyMeta {
  name: string;
  description: string;
  parameters: string[];
}

const PortfolioForm = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [tickers, setTickers] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL', 'AMZN']);
  const [dateRange, setDateRange] = useState({ start: '2023-01-01', end: '2024-01-01' });
  const [strategy, setStrategy] = useState<string>('');
  const [strategies, setStrategies] = useState<Record<string, StrategyMeta>>({});
  const [targetReturn, setTargetReturn] = useState<string>('');
  const [targetReturnGuidance, setTargetReturnGuidance] = useState<null | {
    conservative: number;
    moderate: number;
    aggressive: number;
  }>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/strategies");
        setStrategies(res.data);
      } catch (err) {
        console.error("Failed to fetch strategies", err);
        toast({
          title: "Error loading strategies",
          description: "Unable to load portfolio strategies from the backend.",
          variant: "destructive"
        });
      }
    };
    fetchStrategies();
  }, [toast]);

  useEffect(() => {
    const fetchPortfolioInfo = async () => {
      try {
        const res = await axios.post("http://localhost:5000/api/portfolio-info", {
          tickers,
          start: dateRange.start,
          end: dateRange.end,
        });
        setTargetReturnGuidance(res.data.target_return_guidance || null);
      } catch (err) {
        console.warn("Failed to fetch portfolio info:", err);
        setTargetReturnGuidance(null);
      }
    };

    if (tickers.length >= 2 && dateRange.start && dateRange.end) {
      fetchPortfolioInfo();
    }
  }, [tickers, dateRange]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!strategy) {
      toast({
        title: "Strategy Required",
        description: "Please select an optimization strategy.",
        variant: "destructive",
      });
      return;
    }

    const strategyParams = strategies[strategy]?.parameters ?? [];
    if (strategyParams.includes("target_return") && (!targetReturn || isNaN(parseFloat(targetReturn)))) {
      toast({
        title: "Target Return Required",
        description: "Please enter a valid target return.",
        variant: "destructive",
      });
      return;
    }

    if (tickers.length < 2) {
      toast({
        title: "Insufficient Tickers",
        description: "Please add at least 2 stock tickers.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    const payload: any = {
      tickers,
      start: dateRange.start,
      end: dateRange.end,
      strategy,
      include_efficient_frontier: true
    };

    if (strategyParams.includes("risk_free_rate")) {
      payload.risk_free_rate = 0.02;
    }
    if (strategyParams.includes("target_return")) {
      const target = parseFloat(targetReturn) / 100;
      const max = targetReturnGuidance?.aggressive;
      const min = targetReturnGuidance?.conservative;
    
      if (!target || isNaN(target)) {
        toast({
          title: "Target Return Required",
          description: "Please enter a valid target return.",
          variant: "destructive",
        });
        return;
      }
    
      if ((min !== undefined && target < min) || (max !== undefined && target > max)) {
        toast({
          title: "Target Return Out of Range",
          description: `Please enter a return between ${(min * 100).toFixed(1)}% and ${(max * 100).toFixed(1)}%.`,
          variant: "destructive",
        });
        return;
      }
    
      payload.target_return = target;
    }

    try {
      const res = await axios.post("http://localhost:5000/api/optimize", payload);
      navigate('/results', { state: { result: res.data } });
      toast({
        title: "Portfolio Optimized",
        description: "Your portfolio has been successfully optimized!",
      });
    } catch (err) {
      console.error("Optimization failed:", err);
      toast({
        title: "Optimization Failed",
        description: "There was a problem optimizing your portfolio.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300">
      <CardHeader className="pb-6">
        <CardTitle className="flex items-center space-x-3 text-2xl">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
            <Target className="w-5 h-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
            Portfolio Configuration
          </span>
        </CardTitle>
        <CardDescription className="text-base text-slate-600 dark:text-slate-400">
          Customize your investment preferences for optimization
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <form onSubmit={handleSubmit} className="space-y-8">
          <StockTickerInput tickers={tickers} onTickersChange={setTickers} />
          <DateRangeSelector dateRange={dateRange} onDateRangeChange={setDateRange} />

          <div className="space-y-4">
            <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
              <div className="w-2 h-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"></div>
              <span>Optimization Strategy</span>
            </Label>
            <Select value={strategy} onValueChange={setStrategy}>
              <SelectTrigger className="w-full h-12 bg-white/80 dark:bg-slate-700/80 border-slate-200 dark:border-slate-600">
                <SelectValue placeholder="Select strategy" />
              </SelectTrigger>
              <SelectContent className="bg-white/95 dark:bg-slate-800/95">
                {Object.entries(strategies).map(([key, meta]) => (
                  <SelectItem key={key} value={key} className="text-slate-900 dark:text-slate-100">
                    <div className="flex flex-col py-1">
                      <span className="font-semibold">{meta.name}</span>
                      <span className="text-sm text-slate-500 dark:text-slate-400">{meta.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {strategies[strategy]?.parameters?.includes("target_return") && (
            <div className="space-y-2">
              <Label className="text-base font-semibold text-slate-700 dark:text-slate-300">
                🎯 Target Return (%)
              </Label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={targetReturn}
                onChange={(e) => setTargetReturn(e.target.value)}
                placeholder="e.g. 15 for 15%"
                className="w-full h-12 px-4 border rounded-md bg-white/80 dark:bg-slate-700/80 text-slate-900 dark:text-slate-100"
              />
              {targetReturnGuidance && (
                <div className="text-sm text-slate-500 dark:text-slate-400">
                  Suggested Target Returns:
                  <ul className="list-disc list-inside ml-2 mt-1">
                    <li>Conservative: <strong>{(targetReturnGuidance.conservative * 100).toFixed(1)}%</strong></li>
                    <li>Moderate: <strong>{(targetReturnGuidance.moderate * 100).toFixed(1)}%</strong></li>
                    <li>Aggressive: <strong>{(targetReturnGuidance.aggressive * 100).toFixed(1)}%</strong></li>
                  </ul>
                </div>
              )}
            </div>
          )}

          <Button
            type="submit"
            disabled={isSubmitting}
            className="w-full h-12 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:scale-105 text-white font-semibold shadow-lg transition"
          >
            {isSubmitting ? (
              <div className="flex items-center space-x-3">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Optimizing...</span>
                <Sparkles className="w-4 h-4 animate-pulse" />
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5" />
                <span>Optimize Portfolio</span>
                <Sparkles className="w-4 h-4" />
              </div>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default PortfolioForm;
