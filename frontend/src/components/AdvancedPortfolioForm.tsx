import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Target, BarChart3, Loader2, Sparkles, Settings, TrendingUp, Shield, Brain } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import StockTickerInput from "./StockTickerInput";
import DateRangeSelector from "./DateRangeSelector";
import axios from 'axios';

interface AdvancedStrategy {
  name: string;
  description: string;
  parameters: string[];
  features: string[];
}

interface Constraints {
  min_weights?: number;
  max_weights?: number;
  sector_limits?: Record<string, { min: number; max: number }>;
  target_return?: number;
}

interface OptimizationParams {
  risk_free_rate: number;
  objectives?: string[];
  objective_weights?: number[];
  monte_carlo?: {
    num_simulations: number;
    time_horizon: number;
  };
  views_matrix?: number[][];
  views_uncertainty?: number[][];
  tau?: number;
  risk_budgets?: number[];
}

const AdvancedPortfolioForm = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Basic form state
  const [tickers, setTickers] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']);
  const [dateRange, setDateRange] = useState({ start: '2023-01-01', end: '2024-01-01' });
  const [strategy, setStrategy] = useState<string>('black_litterman');
  const [strategies, setStrategies] = useState<Record<string, AdvancedStrategy>>({});
  
  // Advanced configuration state
  const [constraints, setConstraints] = useState<Constraints>({
    min_weights: 0.05,
    max_weights: 0.40,
    target_return: undefined
  });
  
  const [optimizationParams, setOptimizationParams] = useState<OptimizationParams>({
    risk_free_rate: 0.02,
    objectives: ['sharpe', 'sortino'],
    objective_weights: [0.6, 0.4],
    monte_carlo: {
      num_simulations: 5000,
      time_horizon: 252
    },
    tau: 0.05
  });
  
  const [includeOptions, setIncludeOptions] = useState({
    monte_carlo: true,
    risk_decomposition: true,
    efficient_frontier: true,
    forecasting: true
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/optimization-strategies");
        setStrategies(res.data);
      } catch (err) {
        console.error("Failed to fetch advanced strategies", err);
        toast({
          title: "Error loading strategies",
          description: "Unable to load advanced optimization strategies.",
          variant: "destructive"
        });
      }
    };
    fetchStrategies();
  }, [toast]);
  
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
    
    if (tickers.length < 2) {
      toast({
        title: "Insufficient Tickers",
        description: "Please add at least 2 stock tickers.",
        variant: "destructive",
      });
      return;
    }
    
    setIsSubmitting(true);
    
    const payload = {
      tickers,
      start: dateRange.start,
      end: dateRange.end,
      strategy,
      constraints,
      optimization_params: optimizationParams,
      include_monte_carlo: includeOptions.monte_carlo,
      include_risk_decomposition: includeOptions.risk_decomposition,
      include_efficient_frontier: includeOptions.efficient_frontier
    };
    
    try {
      const res = await axios.post("http://localhost:5000/api/advanced-optimize", payload);
      navigate('/advanced-results', { state: { result: res.data } });
      toast({
        title: "Portfolio Optimized",
        description: "Your advanced portfolio has been successfully optimized!",
      });
    } catch (err: unknown) {
      console.error("Advanced optimization failed:", err);
      const errorMsg = (err as { response?: { data?: { details?: string } } }).response?.data?.details || "Advanced optimization failed";
      toast({
        title: "Optimization Failed",
        description: errorMsg,
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const updateObjectiveWeight = (index: number, value: number) => {
    const newWeights = [...(optimizationParams.objective_weights || [0.5, 0.5])];
    newWeights[index] = value / 100;
    setOptimizationParams(prev => ({
      ...prev,
      objective_weights: newWeights
    }));
  };
  
  return (
    <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300">
      <CardHeader className="pb-6">
        <CardTitle className="flex items-center space-x-3 text-2xl">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent">
            Advanced Portfolio Optimization
          </span>
        </CardTitle>
        <CardDescription className="text-base text-slate-600 dark:text-slate-400">
          Professional-grade portfolio optimization with advanced algorithms and risk management
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-8">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic Setup</TabsTrigger>
              <TabsTrigger value="strategy">Strategy</TabsTrigger>
              <TabsTrigger value="constraints">Constraints</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basic" className="space-y-6">
              <StockTickerInput tickers={tickers} onTickersChange={setTickers} />
              <DateRangeSelector dateRange={dateRange} onDateRangeChange={setDateRange} />
              
              <div className="space-y-4">
                <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4" />
                  <span>Risk-Free Rate (%)</span>
                </Label>
                <Input
                  type="number"
                  step="0.01"
                  min="0"
                  max="10"
                  value={optimizationParams.risk_free_rate * 100}
                  onChange={(e) => setOptimizationParams(prev => ({
                    ...prev,
                    risk_free_rate: parseFloat(e.target.value) / 100
                  }))}
                  className="w-full"
                />
              </div>
            </TabsContent>
            
            <TabsContent value="strategy" className="space-y-6">
              <div className="space-y-4">
                <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
                  <Brain className="w-4 h-4" />
                  <span>Optimization Strategy</span>
                </Label>
                <Select value={strategy} onValueChange={setStrategy}>
                  <SelectTrigger className="w-full h-12">
                    <SelectValue placeholder="Select advanced strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(strategies).map(([key, meta]) => (
                      <SelectItem key={key} value={key}>
                        <div className="flex flex-col py-1">
                          <span className="font-semibold">{meta.name}</span>
                          <span className="text-sm text-slate-500">{meta.description}</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {meta.features.map((feature, idx) => (
                              <span key={idx} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                {feature}
                              </span>
                            ))}
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              {strategy === 'multi_objective' && (
                <div className="space-y-4 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <Label className="text-sm font-semibold">Multi-Objective Configuration</Label>
                  
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3">
                      <Checkbox 
                        checked={optimizationParams.objectives?.includes('sharpe')}
                        onCheckedChange={(checked) => {
                          const objectives = optimizationParams.objectives || [];
                          if (checked) {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: [...objectives, 'sharpe']
                            }));
                          } else {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: objectives.filter(obj => obj !== 'sharpe')
                            }));
                          }
                        }}
                      />
                      <span>Sharpe Ratio</span>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <Checkbox 
                        checked={optimizationParams.objectives?.includes('sortino')}
                        onCheckedChange={(checked) => {
                          const objectives = optimizationParams.objectives || [];
                          if (checked) {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: [...objectives, 'sortino']
                            }));
                          } else {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: objectives.filter(obj => obj !== 'sortino')
                            }));
                          }
                        }}
                      />
                      <span>Sortino Ratio</span>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <Checkbox 
                        checked={optimizationParams.objectives?.includes('max_div')}
                        onCheckedChange={(checked) => {
                          const objectives = optimizationParams.objectives || [];
                          if (checked) {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: [...objectives, 'max_div']
                            }));
                          } else {
                            setOptimizationParams(prev => ({
                              ...prev,
                              objectives: objectives.filter(obj => obj !== 'max_div')
                            }));
                          }
                        }}
                      />
                      <span>Maximum Diversification</span>
                    </div>
                  </div>
                  
                  {optimizationParams.objectives && optimizationParams.objectives.length > 1 && (
                    <div className="space-y-3">
                      <Label className="text-sm">Objective Weights</Label>
                      {optimizationParams.objectives.map((objective, idx) => (
                        <div key={objective} className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-sm capitalize">{objective.replace('_', ' ')}</span>
                            <span className="text-sm">{Math.round((optimizationParams.objective_weights?.[idx] || 0) * 100)}%</span>
                          </div>
                          <Slider
                            value={[Math.round((optimizationParams.objective_weights?.[idx] || 0) * 100)]}
                            onValueChange={([value]) => updateObjectiveWeight(idx, value)}
                            min={0}
                            max={100}
                            step={5}
                            className="w-full"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="constraints" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
                    <Shield className="w-4 h-4" />
                    <span>Position Limits</span>
                  </Label>
                  
                  <div className="space-y-3">
                    <div>
                      <Label className="text-sm">Minimum Weight (%)</Label>
                      <Input
                        type="number"
                        step="0.01"
                        min="0"
                        max="50"
                        value={(constraints.min_weights || 0) * 100}
                        onChange={(e) => setConstraints(prev => ({
                          ...prev,
                          min_weights: parseFloat(e.target.value) / 100
                        }))}
                      />
                    </div>
                    
                    <div>
                      <Label className="text-sm">Maximum Weight (%)</Label>
                      <Input
                        type="number"
                        step="0.01"
                        min="0"
                        max="100"
                        value={(constraints.max_weights || 0) * 100}
                        onChange={(e) => setConstraints(prev => ({
                          ...prev,
                          max_weights: parseFloat(e.target.value) / 100
                        }))}
                      />
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <Label className="text-base font-semibold text-slate-700 dark:text-slate-300">
                    Target Return (Optional)
                  </Label>
                  <Input
                    type="number"
                    step="0.01"
                    min="0"
                    max="50"
                    placeholder="e.g., 12 for 12%"
                    value={constraints.target_return ? constraints.target_return * 100 : ''}
                    onChange={(e) => setConstraints(prev => ({
                      ...prev,
                      target_return: e.target.value ? parseFloat(e.target.value) / 100 : undefined
                    }))}
                  />
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="advanced" className="space-y-6">
              <div className="space-y-4">
                <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
                  <Settings className="w-4 h-4" />
                  <span>Analysis Options</span>
                </Label>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center space-x-3">
                    <Checkbox 
                      checked={includeOptions.monte_carlo}
                      onCheckedChange={(checked) => setIncludeOptions(prev => ({
                        ...prev,
                        monte_carlo: !!checked
                      }))}
                    />
                    <span>Monte Carlo Simulation</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Checkbox 
                      checked={includeOptions.risk_decomposition}
                      onCheckedChange={(checked) => setIncludeOptions(prev => ({
                        ...prev,
                        risk_decomposition: !!checked
                      }))}
                    />
                    <span>Risk Decomposition</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Checkbox 
                      checked={includeOptions.efficient_frontier}
                      onCheckedChange={(checked) => setIncludeOptions(prev => ({
                        ...prev,
                        efficient_frontier: !!checked
                      }))}
                    />
                    <span>Enhanced Efficient Frontier</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Checkbox 
                      checked={includeOptions.forecasting}
                      onCheckedChange={(checked) => setIncludeOptions(prev => ({
                        ...prev,
                        forecasting: !!checked
                      }))}
                    />
                    <span>AI Forecasting</span>
                  </div>
                </div>
              </div>
              
              {includeOptions.monte_carlo && (
                <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg space-y-4">
                  <Label className="text-sm font-semibold">Monte Carlo Parameters</Label>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-sm">Number of Simulations</Label>
                      <Select 
                        value={optimizationParams.monte_carlo?.num_simulations.toString()}
                        onValueChange={(value) => setOptimizationParams(prev => ({
                          ...prev,
                          monte_carlo: {
                            ...prev.monte_carlo!,
                            num_simulations: parseInt(value)
                          }
                        }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1000">1,000</SelectItem>
                          <SelectItem value="5000">5,000</SelectItem>
                          <SelectItem value="10000">10,000</SelectItem>
                          <SelectItem value="50000">50,000</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div>
                      <Label className="text-sm">Time Horizon (Days)</Label>
                      <Select 
                        value={optimizationParams.monte_carlo?.time_horizon.toString()}
                        onValueChange={(value) => setOptimizationParams(prev => ({
                          ...prev,
                          monte_carlo: {
                            ...prev.monte_carlo!,
                            time_horizon: parseInt(value)
                          }
                        }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="30">1 Month</SelectItem>
                          <SelectItem value="90">3 Months</SelectItem>
                          <SelectItem value="252">1 Year</SelectItem>
                          <SelectItem value="504">2 Years</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
          
          <Button
            type="submit"
            disabled={isSubmitting}
            className="w-full h-12 bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:scale-105 text-white font-semibold shadow-lg transition"
          >
            {isSubmitting ? (
              <div className="flex items-center space-x-3">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Advanced Optimization Running...</span>
                <Sparkles className="w-4 h-4 animate-pulse" />
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Brain className="w-5 h-5" />
                <span>Run Advanced Optimization</span>
                <Sparkles className="w-4 h-4" />
              </div>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default AdvancedPortfolioForm;