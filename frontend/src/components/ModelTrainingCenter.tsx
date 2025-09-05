import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import TickerAutocomplete from "@/components/TickerAutocomplete";
import { 
  Brain, 
  Zap, 
  Play, 
  Square, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Database,
  TrendingUp,
  AlertCircle,
  Settings,
  Download,
  Upload
} from "lucide-react";

interface TrainingJob {
  id: string;
  ticker: string;
  models: string[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: string;
  endTime?: string;
  results?: {
    xgboost?: {
      test_r2: number;
      test_mae: number;
      training_samples: number;
    };
    lstm?: {
      test_r2: number;
      test_mae: number;
      epochs_trained: number;
    };
  };
  error?: string;
}

interface ModelStatus {
  ticker: string;
  xgboost_available: boolean;
  lstm_available: boolean;
  xgboost_last_trained?: string;
  lstm_last_trained?: string;
  xgboost_performance?: number;
  lstm_performance?: number;
}

export const ModelTrainingCenter: React.FC = () => {
  const [activeTab, setActiveTab] = useState('train');
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [modelStatuses, setModelStatuses] = useState<ModelStatus[]>([]);
  const [selectedTicker, setSelectedTicker] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost']);
  const [trainingPeriod, setTrainingPeriod] = useState('1y');
  const [isTraining, setIsTraining] = useState(false);
  const [popularTickers] = useState([
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 
    'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'NFLX', 'CRM', 'ADBE', 'PYPL'
  ]);
  
  // Batch training state
  const [batchTickers, setBatchTickers] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL']);
  const [batchModels, setBatchModels] = useState<string[]>(['xgboost']);
  const [batchPeriod, setBatchPeriod] = useState('1y');
  const [isBatchTraining, setIsBatchTraining] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ completed: 0, total: 0 });

  useEffect(() => {
    fetchModelStatuses();
    // Poll for training job updates every 5 seconds
    const interval = setInterval(updateTrainingJobs, 5000);
    return () => clearInterval(interval);
  }, [fetchModelStatuses]);

  const fetchModelStatuses = useCallback(async () => {
    try {
      // This would be a real API call to check model availability
      const mockStatuses: ModelStatus[] = popularTickers.map(ticker => ({
        ticker,
        xgboost_available: Math.random() > 0.7,
        lstm_available: Math.random() > 0.8,
        xgboost_last_trained: Math.random() > 0.5 ? '2024-08-01T10:00:00Z' : undefined,
        lstm_last_trained: Math.random() > 0.5 ? '2024-08-01T10:00:00Z' : undefined,
        xgboost_performance: Math.random() * 0.8,
        lstm_performance: Math.random() * 0.7
      }));
      setModelStatuses(mockStatuses);
    } catch (error) {
      console.error('Failed to fetch model statuses:', error);
    }
  }, [popularTickers]);

  const updateTrainingJobs = () => {
    setTrainingJobs(jobs => 
      jobs.map(job => {
        if (job.status === 'running') {
          const newProgress = Math.min(job.progress + Math.random() * 10, 100);
          return {
            ...job,
            progress: newProgress,
            status: newProgress >= 100 ? 'completed' : 'running',
            endTime: newProgress >= 100 ? new Date().toISOString() : job.endTime,
            results: newProgress >= 100 ? {
              xgboost: job.models.includes('xgboost') ? {
                test_r2: 0.65 + Math.random() * 0.2,
                test_mae: 2.5 + Math.random() * 1.5,
                training_samples: 200 + Math.floor(Math.random() * 300)
              } : undefined,
              lstm: job.models.includes('lstm') ? {
                test_r2: 0.60 + Math.random() * 0.25,
                test_mae: 3.0 + Math.random() * 1.5,
                epochs_trained: 45 + Math.floor(Math.random() * 30)
              } : undefined
            } : job.results
          };
        }
        return job;
      })
    );
  };

  const startTraining = async () => {
    if (!selectedTicker || selectedModels.length === 0) return;

    setIsTraining(true);
    
    const newJob: TrainingJob = {
      id: `job_${Date.now()}`,
      ticker: selectedTicker.toUpperCase(),
      models: selectedModels,
      status: 'running',
      progress: 0,
      startTime: new Date().toISOString()
    };

    setTrainingJobs(prev => [newJob, ...prev]);

    try {
      // Call the training API
      const response = await fetch(`/api/train/${selectedTicker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          models: selectedModels,
          period: trainingPeriod
        })
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.status}`);
      }

      const result = await response.json();
      
      // Update job with results
      setTrainingJobs(prev => prev.map(job => 
        job.id === newJob.id ? {
          ...job,
          status: 'completed',
          progress: 100,
          endTime: new Date().toISOString(),
          results: result.training_results
        } : job
      ));

      // Refresh model statuses
      fetchModelStatuses();

    } catch (error) {
      console.error('Training error:', error);
      setTrainingJobs(prev => prev.map(job => 
        job.id === newJob.id ? {
          ...job,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Training failed'
        } : job
      ));
    } finally {
      setIsTraining(false);
    }
  };

  const trainPopularStocks = async () => {
    const untrained = modelStatuses.filter(status => 
      !status.xgboost_available || !status.lstm_available
    ).slice(0, 5); // Train top 5 untrained stocks

    for (const stock of untrained) {
      const modelsToTrain = [];
      if (!stock.xgboost_available) modelsToTrain.push('xgboost');
      if (!stock.lstm_available) modelsToTrain.push('lstm');

      const job: TrainingJob = {
        id: `batch_${stock.ticker}_${Date.now()}`,
        ticker: stock.ticker,
        models: modelsToTrain,
        status: 'running',
        progress: 0,
        startTime: new Date().toISOString()
      };

      setTrainingJobs(prev => [job, ...prev]);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Clock className="w-4 h-4 text-blue-600 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-600" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getModelIcon = (model: string) => {
    switch (model) {
      case 'xgboost': return <Zap className="w-4 h-4" />;
      case 'lstm': return <Brain className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  const formatDuration = (start: string, end?: string) => {
    const startTime = new Date(start);
    const endTime = end ? new Date(end) : new Date();
    const duration = Math.floor((endTime.getTime() - startTime.getTime()) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const startBatchTraining = async () => {
    if (batchTickers.length === 0 || batchModels.length === 0) return;

    setIsBatchTraining(true);
    setBatchProgress({ completed: 0, total: batchTickers.length });
    
    const batchJobs: TrainingJob[] = batchTickers.map(ticker => ({
      id: `batch_${ticker}_${Date.now()}`,
      ticker: ticker.toUpperCase(),
      models: batchModels,
      status: 'pending',
      progress: 0,
      startTime: new Date().toISOString()
    }));

    setTrainingJobs(prev => [...batchJobs, ...prev]);

    try {
      for (let i = 0; i < batchTickers.length; i++) {
        const ticker = batchTickers[i];
        const jobId = batchJobs[i].id;

        // Update job status to running
        setTrainingJobs(prev => 
          prev.map(job => 
            job.id === jobId ? { ...job, status: 'running' as const } : job
          )
        );

        try {
          const response = await fetch(`/api/train/${ticker}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              models: batchModels,
              period: batchPeriod
            })
          });

          if (!response.ok) {
            throw new Error(`Training failed for ${ticker}: ${response.status}`);
          }

          const result = await response.json();
          
          // Update job with success
          setTrainingJobs(prev => 
            prev.map(job => 
              job.id === jobId 
                ? {
                    ...job,
                    status: 'completed' as const,
                    progress: 100,
                    endTime: new Date().toISOString(),
                    results: result.training_results
                  }
                : job
            )
          );

        } catch (error: unknown) {
          // Update job with error
          setTrainingJobs(prev => 
            prev.map(job => 
              job.id === jobId 
                ? {
                    ...job,
                    status: 'failed' as const,
                    progress: 0,
                    endTime: new Date().toISOString(),
                    error: error.message
                  }
                : job
            )
          );
        }

        // Update batch progress
        setBatchProgress({ completed: i + 1, total: batchTickers.length });
      }

    } finally {
      setIsBatchTraining(false);
      setBatchProgress({ completed: 0, total: 0 });
      fetchModelStatuses();
    }
  };

  const addTickerToBatch = (ticker: string) => {
    const symbol = ticker.toUpperCase().trim();
    if (symbol && !batchTickers.includes(symbol)) {
      setBatchTickers([...batchTickers, symbol]);
    }
  };

  const removeTickerFromBatch = (ticker: string) => {
    setBatchTickers(batchTickers.filter(t => t !== ticker));
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <Card className="border-0 shadow-sm bg-white overflow-hidden">
        <CardHeader className="pb-8 bg-gradient-to-r from-purple-50 to-indigo-50">
          <CardTitle className="flex items-center gap-3 text-2xl font-bold text-gray-900">
            <div className="p-3 bg-purple-100 rounded-xl">
              <Brain className="w-6 h-6 text-purple-600" />
            </div>
            AI Model Training Center
          </CardTitle>
          <CardDescription className="text-gray-600 mt-3 text-base leading-relaxed">
            Train XGBoost and LSTM models for accurate stock predictions using advanced machine learning algorithms
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-100 p-1 rounded-xl h-14">
          <TabsTrigger 
            value="train"
            className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm py-3"
          >
            Single Training
          </TabsTrigger>
          <TabsTrigger 
            value="batch"
            className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm py-3"
          >
            Batch Training
          </TabsTrigger>
          <TabsTrigger 
            value="status"
            className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm py-3"
          >
            Model Status
          </TabsTrigger>
          <TabsTrigger 
            value="jobs"
            className="rounded-lg font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm py-3"
          >
            Training Jobs
          </TabsTrigger>
        </TabsList>

        {/* Training Tab */}
        <TabsContent value="train" className="space-y-8 mt-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Manual Training */}
            <Card className="border-0 shadow-sm bg-white overflow-hidden">
              <CardHeader className="pb-6 bg-gradient-to-r from-blue-50 to-cyan-50">
                <CardTitle className="flex items-center gap-3 text-lg font-semibold text-gray-900">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Play className="w-5 h-5 text-blue-600" />
                  </div>
                  Train Individual Stock
                </CardTitle>
                <CardDescription className="text-gray-600 mt-2">
                  Train models for a specific ticker symbol
                </CardDescription>
              </CardHeader>
              <CardContent className="p-8 space-y-6">
                <div>
                  <Label htmlFor="ticker">Stock Ticker</Label>
                  <div className="mt-1">
                    <TickerAutocomplete
                      value={selectedTicker}
                      onChange={setSelectedTicker}
                      onSelect={(suggestion) => setSelectedTicker(suggestion.symbol)}
                      placeholder="AAPL"
                      showPopular={true}
                    />
                  </div>
                </div>

                <div>
                  <Label>Models to Train</Label>
                  <div className="mt-2 space-y-2">
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes('xgboost')}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels(prev => [...prev, 'xgboost']);
                          } else {
                            setSelectedModels(prev => prev.filter(m => m !== 'xgboost'));
                          }
                        }}
                      />
                      <Zap className="w-4 h-4 text-orange-600" />
                      <span>XGBoost (Fast, Technical Analysis)</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes('lstm')}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels(prev => [...prev, 'lstm']);
                          } else {
                            setSelectedModels(prev => prev.filter(m => m !== 'lstm'));
                          }
                        }}
                      />
                      <Brain className="w-4 h-4 text-purple-600" />
                      <span>LSTM (Slow, Deep Learning)</span>
                    </label>
                  </div>
                </div>

                <div>
                  <Label>Training Period</Label>
                  <Select value={trainingPeriod} onValueChange={setTrainingPeriod}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="6mo">6 Months</SelectItem>
                      <SelectItem value="1y">1 Year</SelectItem>
                      <SelectItem value="2y">2 Years</SelectItem>
                      <SelectItem value="5y">5 Years</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button 
                  onClick={startTraining}
                  disabled={!selectedTicker || selectedModels.length === 0 || isTraining}
                  className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium"
                  size="lg"
                >
                  {isTraining ? (
                    <>
                      <Clock className="w-5 h-5 mr-2 animate-spin" />
                      Training in Progress...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 mr-2" />
                      Start Training
                    </>
                  )}
                </Button>

                <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-blue-100 rounded-lg flex-shrink-0">
                      <AlertCircle className="h-4 w-4 text-blue-600" />
                    </div>
                    <div className="text-sm text-blue-800 leading-relaxed">
                      <p className="font-medium mb-1">Training Time Estimates</p>
                      <p>XGBoost: 2-5 minutes • LSTM: 10-20 minutes</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Batch Training */}
            <Card className="border-0 shadow-sm bg-white overflow-hidden">
              <CardHeader className="pb-6 bg-gradient-to-r from-green-50 to-emerald-50">
                <CardTitle className="flex items-center gap-3 text-lg font-semibold text-gray-900">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <Database className="w-5 h-5 text-green-600" />
                  </div>
                  Quick Batch Training
                </CardTitle>
                <CardDescription className="text-gray-600 mt-2">
                  Train models for popular stocks automatically
                </CardDescription>
              </CardHeader>
              <CardContent className="p-8 space-y-6">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p>Popular tickers that need training:</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {modelStatuses
                      .filter(s => !s.xgboost_available || !s.lstm_available)
                      .slice(0, 8)
                      .map(status => (
                        <Badge key={status.ticker} variant="outline">
                          {status.ticker}
                        </Badge>
                      ))}
                  </div>
                </div>

                <Button 
                  onClick={trainPopularStocks}
                  variant="outline"
                  className="w-full py-3 rounded-xl border-2 border-green-200 text-green-700 hover:bg-green-50 font-medium"
                  size="lg"
                >
                  <TrendingUp className="w-5 h-5 mr-2" />
                  Train Popular Stocks
                </Button>

                <div className="p-4 bg-green-50 rounded-xl border border-green-200">
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-green-100 rounded-lg flex-shrink-0">
                      <AlertCircle className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="text-sm text-green-800 leading-relaxed">
                      This will automatically train missing models for the top 5 untrained popular stocks.
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Batch Training Tab */}
        <TabsContent value="batch" className="space-y-8 mt-8">
          <Card className="border-0 shadow-sm bg-white overflow-hidden">
            <CardHeader className="pb-8 bg-gradient-to-r from-purple-50 to-violet-50">
              <CardTitle className="flex items-center gap-3 text-xl font-semibold text-gray-900">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Database className="w-5 h-5 text-purple-600" />
                </div>
                Custom Batch Training
              </CardTitle>
              <CardDescription className="text-gray-600 mt-2 text-base">
                Train models for multiple stocks at once with custom settings
              </CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-8">
              {/* Stock Selection */}
              <div>
                <Label className="text-sm font-medium mb-2 block">Select Stocks for Batch Training</Label>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <TickerAutocomplete
                      value=""
                      onChange={addTickerToBatch}
                      onSelect={(suggestion) => addTickerToBatch(suggestion.symbol)}
                      placeholder="Add ticker to batch (e.g., AAPL)"
                      showPopular={false}
                      className="flex-1"
                    />
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setBatchTickers([...batchTickers, ...popularTickers.slice(0, 5)])}
                    >
                      Add Top 5
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setBatchTickers([])}
                    >
                      Clear All
                    </Button>
                  </div>
                  
                  {/* Selected Tickers */}
                  <div className="flex flex-wrap gap-2 min-h-[40px] p-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800">
                    {batchTickers.length === 0 ? (
                      <span className="text-gray-500 text-sm">No stocks selected</span>
                    ) : (
                      batchTickers.map(ticker => (
                        <Badge key={ticker} variant="secondary" className="px-3 py-1">
                          {ticker}
                          <button 
                            onClick={() => removeTickerFromBatch(ticker)} 
                            className="ml-2 hover:text-red-600"
                          >
                            ×
                          </button>
                        </Badge>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* Model Selection */}
              <div>
                <Label className="text-sm font-medium mb-2 block">Models to Train</Label>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={batchModels.includes('xgboost')}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setBatchModels([...batchModels, 'xgboost']);
                        } else {
                          setBatchModels(batchModels.filter(m => m !== 'xgboost'));
                        }
                      }}
                      className="rounded"
                    />
                    <Zap className="w-4 h-4 text-yellow-600" />
                    <span>XGBoost (Fast, ~2-5 min per stock)</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={batchModels.includes('lstm')}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setBatchModels([...batchModels, 'lstm']);
                        } else {
                          setBatchModels(batchModels.filter(m => m !== 'lstm'));
                        }
                      }}
                      className="rounded"
                    />
                    <Brain className="w-4 h-4 text-purple-600" />
                    <span>LSTM (Deep learning, ~10-20 min per stock)</span>
                  </label>
                </div>
              </div>

              {/* Training Period */}
              <div>
                <Label className="text-sm font-medium mb-2 block">Training Period</Label>
                <Select value={batchPeriod} onValueChange={setBatchPeriod}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="6mo">6 Months (Faster)</SelectItem>
                    <SelectItem value="1y">1 Year (Recommended)</SelectItem>
                    <SelectItem value="2y">2 Years (More Data)</SelectItem>
                    <SelectItem value="5y">5 Years (Max History)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Batch Progress */}
              {isBatchTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Batch Training Progress</span>
                    <span>{batchProgress.completed} / {batchProgress.total}</span>
                  </div>
                  <Progress 
                    value={(batchProgress.completed / batchProgress.total) * 100} 
                    className="w-full"
                  />
                </div>
              )}

              {/* Training Controls */}
              <div className="flex space-x-3">
                <Button 
                  onClick={startBatchTraining}
                  disabled={batchTickers.length === 0 || batchModels.length === 0 || isBatchTraining}
                  className="flex-1"
                  size="lg"
                >
                  {isBatchTraining ? (
                    <>
                      <Clock className="w-4 h-4 mr-2 animate-spin" />
                      Training Batch...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Batch Training
                    </>
                  )}
                </Button>
                
                <div className="text-center">
                  <div className="text-xs text-gray-500">
                    Est. Time: {batchTickers.length} × {batchModels.includes('lstm') ? '15' : '3'} min
                  </div>
                  <div className="text-xs text-gray-500">
                    ~{Math.ceil(batchTickers.length * (batchModels.includes('lstm') ? 15 : 3) / 60)}h total
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="border-t pt-4">
                <Label className="text-sm font-medium mb-2 block">Quick Batch Presets</Label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setBatchTickers(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']);
                      setBatchModels(['xgboost']);
                      setBatchPeriod('1y');
                    }}
                  >
                    Tech Giants (XGB)
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setBatchTickers(['SPY', 'QQQ', 'VTI', 'IWM', 'EFA']);
                      setBatchModels(['xgboost']);
                      setBatchPeriod('2y');
                    }}
                  >
                    Popular ETFs (XGB)
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setBatchTickers(popularTickers.slice(0, 10));
                      setBatchModels(['xgboost', 'lstm']);
                      setBatchPeriod('1y');
                    }}
                  >
                    Top 10 (Both Models)
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Model Status Tab */}
        <TabsContent value="status" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Availability Status</CardTitle>
              <CardDescription>
                Check which models are trained and available for predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {modelStatuses.map(status => (
                  <div key={status.ticker} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <span className="font-medium text-lg">{status.ticker}</span>
                      <div className="flex items-center space-x-2">
                        <div className="flex items-center space-x-1">
                          <Zap className="w-4 h-4 text-orange-600" />
                          {status.xgboost_available ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <XCircle className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                        <div className="flex items-center space-x-1">
                          <Brain className="w-4 h-4 text-purple-600" />
                          {status.lstm_available ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <XCircle className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {status.xgboost_last_trained && (
                        <div>XGB: {new Date(status.xgboost_last_trained).toLocaleDateString()}</div>
                      )}
                      {status.lstm_last_trained && (
                        <div>LSTM: {new Date(status.lstm_last_trained).toLocaleDateString()}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Training Jobs Tab */}
        <TabsContent value="jobs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Training Jobs History</CardTitle>
              <CardDescription>
                Monitor current and past training jobs
              </CardDescription>
            </CardHeader>
            <CardContent>
              {trainingJobs.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No training jobs yet. Start training models to see progress here.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {trainingJobs.map(job => (
                    <div key={job.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(job.status)}
                          <div>
                            <span className="font-medium text-lg">{job.ticker}</span>
                            <div className="flex items-center space-x-2 mt-1">
                              {job.models.map(model => (
                                <Badge key={model} variant="secondary" className="flex items-center space-x-1">
                                  {getModelIcon(model)}
                                  <span>{model.toUpperCase()}</span>
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {job.startTime && formatDuration(job.startTime, job.endTime)}
                        </div>
                      </div>

                      {job.status === 'running' && (
                        <div className="mb-3">
                          <div className="flex justify-between text-sm mb-1">
                            <span>Training Progress</span>
                            <span>{job.progress.toFixed(0)}%</span>
                          </div>
                          <Progress value={job.progress} className="h-2" />
                        </div>
                      )}

                      {job.status === 'completed' && job.results && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                          {job.results.xgboost && (
                            <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                              <div className="font-medium text-orange-800 dark:text-orange-200">XGBoost Results</div>
                              <div className="text-sm text-orange-700 dark:text-orange-300">
                                R²: {job.results.xgboost.test_metrics?.r2?.toFixed(3) || 'N/A'} | 
                                MAE: ${job.results.xgboost.test_metrics?.mae?.toFixed(2) || 'N/A'} |
                                Samples: {job.results.xgboost.training_samples || 'N/A'}
                              </div>
                            </div>
                          )}
                          {job.results.lstm && (
                            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                              <div className="font-medium text-purple-800 dark:text-purple-200">LSTM Results</div>
                              <div className="text-sm text-purple-700 dark:text-purple-300">
                                R²: {job.results.lstm.test_metrics?.r2?.toFixed(3) || 'N/A'} | 
                                MAE: ${job.results.lstm.test_metrics?.mae?.toFixed(2) || 'N/A'} |
                                Epochs: {job.results.lstm.training_history?.epochs_trained || 'N/A'}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {job.status === 'failed' && job.error && (
                        <Card className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-red-200 dark:border-red-700">
                          <CardContent className="p-4">
                            <div className="flex items-start space-x-3">
                              <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
                                <XCircle className="w-5 h-5 text-white" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <h4 className="font-semibold text-red-800 dark:text-red-300 mb-1">Training Failed</h4>
                                <p className="text-sm text-red-600 dark:text-red-400 mb-3">{job.error}</p>
                                <div className="flex space-x-2">
                                  <Button 
                                    size="sm" 
                                    onClick={() => startSingleTraining(job.ticker, job.models, 'instant')}
                                    className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white"
                                  >
                                    Retry Training
                                  </Button>
                                  <Button 
                                    size="sm" 
                                    variant="outline"
                                    onClick={() => setTrainingJobs(prev => prev.filter(j => j.id !== job.id))}
                                    className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-600 dark:text-red-400"
                                  >
                                    Dismiss
                                  </Button>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ModelTrainingCenter;