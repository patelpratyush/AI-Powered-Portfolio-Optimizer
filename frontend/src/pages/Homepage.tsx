import { Link } from 'react-router-dom';
import { useDarkMode } from '@/hooks/useDarkMode';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  Brain, 
  Sparkles, 
  Target, 
  BarChart3, 
  Shield, 
  Zap, 
  ArrowRight,
  Moon,
  Sun,
  CheckCircle,
  Users,
  Building,
  GraduationCap
} from "lucide-react";

const Homepage = () => {
  const { isDarkMode, toggleDarkMode } = useDarkMode();

  const basicFeatures = [
    "Modern Portfolio Theory",
    "Sharpe Ratio Optimization", 
    "Risk Parity Strategies",
    "Efficient Frontier Analysis",
    "Basic Risk Metrics",
    "Portfolio Forecasting"
  ];

  const advancedFeatures = [
    "Black-Litterman Optimization",
    "Hierarchical Risk Parity",
    "Multi-Objective Optimization", 
    "Monte Carlo Simulation",
    "Risk Decomposition Analysis",
    "Advanced Constraints",
    "ML-Enhanced Strategies",
    "Professional Risk Metrics"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-indigo-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-gradient-to-br from-emerald-400/10 to-cyan-600/10 rounded-full blur-2xl animate-pulse delay-1000"></div>
      </div>

      {/* Header */}
      <header className="relative z-20 px-4 py-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
              AI Portfolio Optimizer
            </h1>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleDarkMode}
            className="hover:bg-white/20 dark:hover:bg-slate-800/50 backdrop-blur-sm"
          >
            {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </Button>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        {/* Hero Section */}
        <section className="text-center mb-16 pt-8">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-slate-900 via-blue-900 to-indigo-900 dark:from-slate-100 dark:via-blue-300 dark:to-indigo-300 bg-clip-text text-transparent leading-tight">
              Professional Portfolio Optimization
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-400 mb-8 leading-relaxed">
              Choose the perfect optimization tool for your investment needs. From beginner-friendly strategies 
              to institutional-grade algorithms powered by machine learning and advanced financial mathematics.
            </p>
            <div className="flex flex-wrap justify-center gap-6 text-sm text-slate-500 dark:text-slate-400">
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Users className="w-4 h-4 text-green-600" />
                <span>10,000+ Portfolios Optimized</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Building className="w-4 h-4 text-blue-600" />
                <span>Institutional-Grade Algorithms</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <GraduationCap className="w-4 h-4 text-purple-600" />
                <span>Research-Backed Methods</span>
              </div>
            </div>
          </div>
        </section>

        {/* Optimizer Selection */}
        <section className="mb-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Basic Optimizer */}
            <Card className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 group">
              <div className="absolute top-4 right-4">
                <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                  Beginner Friendly
                </Badge>
              </div>
              
              <CardHeader className="pb-6">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
                    <Target className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl bg-gradient-to-r from-slate-900 to-green-900 dark:from-slate-100 dark:to-green-300 bg-clip-text text-transparent">
                      Smart Optimizer
                    </CardTitle>
                    <CardDescription className="text-base">
                      Perfect for individual investors and beginners
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <p className="text-slate-600 dark:text-slate-400">
                  Easy-to-use portfolio optimization with proven strategies. Get started quickly with 
                  modern portfolio theory, risk analysis, and intelligent forecasting.
                </p>
                
                <div className="space-y-3">
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    <span>Key Features</span>
                  </h4>
                  <div className="grid grid-cols-1 gap-2">
                    {basicFeatures.map((feature, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-400">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="pt-4">
                  <Link to="/basic">
                    <Button className="w-full h-12 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold shadow-lg transition-all group-hover:scale-105">
                      <Target className="w-5 h-5 mr-2" />
                      Start Basic Optimization
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                </div>
                
                <div className="text-xs text-slate-500 dark:text-slate-400 text-center">
                  Best for: Individual investors, beginners, simple portfolios
                </div>
              </CardContent>
            </Card>

            {/* Advanced Optimizer */}
            <Card className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 group border-2 border-purple-200 dark:border-purple-700">
              <div className="absolute top-4 right-4">
                <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                  Professional
                </Badge>
              </div>
              
              <div className="absolute -top-2 -right-2">
                <div className="w-6 h-6 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-full flex items-center justify-center">
                  <Sparkles className="w-3 h-3 text-white" />
                </div>
              </div>
              
              <CardHeader className="pb-6">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent">
                      AI Pro Optimizer
                    </CardTitle>
                    <CardDescription className="text-base">
                      Institutional-grade algorithms and machine learning
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <p className="text-slate-600 dark:text-slate-400">
                  Professional portfolio optimization with advanced algorithms used by institutional investors. 
                  Features machine learning, Monte Carlo simulation, and sophisticated risk management.
                </p>
                
                <div className="space-y-3">
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center space-x-2">
                    <Zap className="w-4 h-4 text-purple-600" />
                    <span>Advanced Features</span>
                  </h4>
                  <div className="grid grid-cols-1 gap-2">
                    {advancedFeatures.map((feature, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-400">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="pt-4">
                  <Link to="/advanced">
                    <Button className="w-full h-12 bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-700 hover:via-indigo-700 hover:to-blue-700 text-white font-semibold shadow-lg transition-all group-hover:scale-105">
                      <Brain className="w-5 h-5 mr-2" />
                      Launch AI Pro Optimizer
                      <Sparkles className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                </div>
                
                <div className="text-xs text-slate-500 dark:text-slate-400 text-center">
                  Best for: Professional traders, institutions, complex portfolios
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* AI Hub Section */}
        <section className="mb-16">
          <Card className="relative bg-gradient-to-br from-cyan-50 via-blue-50 to-indigo-100 dark:from-cyan-900/20 dark:via-blue-900/20 dark:to-indigo-900/20 backdrop-blur-xl border-cyan-200 dark:border-cyan-700 shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full blur-2xl"></div>
            <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-indigo-400/20 to-purple-600/20 rounded-full blur-xl"></div>
            
            <CardHeader className="relative z-10 text-center pb-6">
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <Brain className="w-8 h-8 text-white" />
                </div>
              </div>
              <CardTitle className="text-3xl bg-gradient-to-r from-cyan-900 via-blue-900 to-indigo-900 dark:from-cyan-200 dark:via-blue-200 dark:to-indigo-200 bg-clip-text text-transparent mb-2">
                AI Investment Hub
              </CardTitle>
              <CardDescription className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
                Advanced AI-powered stock predictions, portfolio analysis, and machine learning models. 
                Get professional-grade forecasts using XGBoost, LSTM, and Prophet algorithms.
              </CardDescription>
              <Badge className="mt-4 bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300 text-sm">
                <Sparkles className="w-3 h-3 mr-1" />
                New AI Features
              </Badge>
            </CardHeader>
            
            <CardContent className="relative z-10">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="text-center p-4 bg-white/50 dark:bg-slate-800/50 rounded-xl backdrop-blur-sm">
                  <TrendingUp className="w-8 h-8 mx-auto mb-3 text-blue-600" />
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">AI Predictions</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    XGBoost, LSTM & Prophet ensemble forecasts
                  </p>
                </div>
                
                <div className="text-center p-4 bg-white/50 dark:bg-slate-800/50 rounded-xl backdrop-blur-sm">
                  <BarChart3 className="w-8 h-8 mx-auto mb-3 text-green-600" />
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">Portfolio Import</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    CSV upload & live market data analysis
                  </p>
                </div>
                
                <div className="text-center p-4 bg-white/50 dark:bg-slate-800/50 rounded-xl backdrop-blur-sm">
                  <Brain className="w-8 h-8 mx-auto mb-3 text-purple-600" />
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">Model Training</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Train AI models for any stock ticker
                  </p>
                </div>
                
                <div className="text-center p-4 bg-white/50 dark:bg-slate-800/50 rounded-xl backdrop-blur-sm">
                  <Target className="w-8 h-8 mx-auto mb-3 text-orange-600" />
                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">Smart Signals</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    BUY/SELL recommendations with confidence
                  </p>
                </div>
              </div>
              
              <div className="text-center">
                <Link to="/ai-hub">
                  <Button size="lg" className="h-14 px-8 bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-700 hover:via-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg transition-all hover:scale-105">
                    <Brain className="w-5 h-5 mr-3" />
                    Explore AI Investment Hub
                    <Sparkles className="w-4 h-4 ml-3" />
                  </Button>
                </Link>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-3">
                  Features machine learning models, advanced technical analysis, and professional-grade forecasting
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Comparison Table */}
        <section className="mb-16">
          <Card className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
            <CardHeader>
              <CardTitle className="text-2xl text-center bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
                Feature Comparison
              </CardTitle>
              <CardDescription className="text-center text-base">
                Choose the right optimizer for your needs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-700">
                      <th className="text-left p-4 font-semibold">Feature</th>
                      <th className="text-center p-4 font-semibold text-green-600">Smart Optimizer</th>
                      <th className="text-center p-4 font-semibold text-purple-600">AI Pro Optimizer</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {[
                      ["Optimization Strategies", "5 Core Strategies", "8+ Advanced Algorithms"],
                      ["Risk Analysis", "Basic Metrics", "20+ Professional Metrics"],
                      ["Portfolio Constraints", "Simple Limits", "Advanced Constraints"],
                      ["Visualization", "Standard Charts", "Interactive Dashboards"],
                      ["Monte Carlo Simulation", "Basic", "Up to 50,000 Simulations"],
                      ["Machine Learning", "❌", "✅ Full ML Suite"],
                      ["Black-Litterman", "❌", "✅ Market Equilibrium"],
                      ["Risk Decomposition", "❌", "✅ Component Analysis"],
                      ["Real-time Updates", "❌", "✅ Live Data"],
                      ["Complexity Level", "Beginner", "Professional"]
                    ].map(([feature, basic, advanced], idx) => (
                      <tr key={idx} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700/50">
                        <td className="p-4 font-medium">{feature}</td>
                        <td className="p-4 text-center text-slate-600 dark:text-slate-400">{basic}</td>
                        <td className="p-4 text-center text-slate-600 dark:text-slate-400">{advanced}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Quick Stats */}
        <section className="text-center">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <TrendingUp className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">99.9%</div>
              <div className="text-slate-600 dark:text-slate-400">Uptime</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <BarChart3 className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">15%</div>
              <div className="text-slate-600 dark:text-slate-400">Avg. Return Improvement</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">30%</div>
              <div className="text-slate-600 dark:text-slate-400">Risk Reduction</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">AI</div>
              <div className="text-slate-600 dark:text-slate-400">Powered Insights</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Homepage;