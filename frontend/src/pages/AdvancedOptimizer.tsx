import { Link } from 'react-router-dom';
import { useDarkMode } from '@/hooks/useDarkMode';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Sparkles, Target, TrendingUp, Zap, Shield, BarChart3, ArrowLeft, Moon, Sun } from "lucide-react";
import AdvancedPortfolioForm from "../components/AdvancedPortfolioForm";

const AdvancedOptimizer = () => {
  const { isDarkMode, toggleDarkMode } = useDarkMode();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-purple-50 to-indigo-100 dark:from-slate-900 dark:via-purple-900/20 dark:to-indigo-900/20 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-purple-400/20 to-indigo-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-gradient-to-br from-indigo-400/10 to-purple-600/10 rounded-full blur-2xl animate-pulse delay-1000"></div>
      </div>

      {/* Header */}
      <header className="relative z-20 px-4 py-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent">
              AI Pro Portfolio Optimizer
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/">
              <Button 
                variant="outline" 
                className="bg-white/20 dark:bg-slate-800/20 backdrop-blur-sm border-white/30 dark:border-slate-600/30"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Home
              </Button>
            </Link>
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleDarkMode}
              className="hover:bg-white/20 dark:hover:bg-slate-800/50 backdrop-blur-sm"
            >
              {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </Button>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        {/* Hero Section */}
        <section className="text-center mb-16">
          <div className="max-w-4xl mx-auto">
            <div className="flex justify-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
            </div>
            
            <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-slate-900 via-purple-900 to-indigo-900 dark:from-slate-100 dark:via-purple-300 dark:to-indigo-300 bg-clip-text text-transparent leading-tight">
              Advanced AI Portfolio Optimizer
            </h1>
            
            <p className="text-xl text-slate-600 dark:text-slate-400 mb-8 leading-relaxed">
              Professional-grade portfolio optimization powered by machine learning, advanced risk models, 
              and cutting-edge financial algorithms. Experience the future of portfolio management.
            </p>

            <div className="flex flex-wrap justify-center gap-6 text-sm text-slate-500 dark:text-slate-400">
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Brain className="w-4 h-4 text-purple-600" />
                <span>Black-Litterman Optimization</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Target className="w-4 h-4 text-indigo-600" />
                <span>Monte Carlo Simulation</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Shield className="w-4 h-4 text-blue-600" />
                <span>Advanced Risk Management</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Zap className="w-4 h-4 text-green-600" />
                <span>ML-Enhanced Strategies</span>
              </div>
            </div>
          </div>
        </section>

        {/* Enhanced Features Overview */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent mb-4">
              Professional-Grade Features
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Advanced algorithms and risk management tools used by institutional investors worldwide
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="group bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-indigo-500/5 to-transparent group-hover:from-purple-500/20 group-hover:via-indigo-500/10 transition-all duration-300"></div>
              <CardHeader className="relative pb-4">
                <CardTitle className="flex items-center space-x-3 text-xl">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-purple-700 to-indigo-700 dark:from-purple-300 dark:to-indigo-300 bg-clip-text text-transparent">
                    Advanced Algorithms
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="relative">
                <p className="text-slate-600 dark:text-slate-400 mb-6 leading-relaxed">
                  Black-Litterman, Hierarchical Risk Parity, and Multi-Objective optimization with ML enhancement.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                    <div className="w-3 h-3 bg-purple-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Market equilibrium modeling</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-indigo-50 dark:bg-indigo-900/20">
                    <div className="w-3 h-3 bg-indigo-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Investor views incorporation</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                    <div className="w-3 h-3 bg-blue-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Machine learning clustering</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="group bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 via-emerald-500/5 to-transparent group-hover:from-green-500/20 group-hover:via-emerald-500/10 transition-all duration-300"></div>
              <CardHeader className="relative pb-4">
                <CardTitle className="flex items-center space-x-3 text-xl">
                  <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-green-700 to-emerald-700 dark:from-green-300 dark:to-emerald-300 bg-clip-text text-transparent">
                    Analytics Suite
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="relative">
                <p className="text-slate-600 dark:text-slate-400 mb-6 leading-relaxed">
                  In-depth risk decomposition, Monte Carlo simulations, and interactive visualizations.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-green-50 dark:bg-green-900/20">
                    <div className="w-3 h-3 bg-green-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Risk contribution analysis</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-emerald-50 dark:bg-emerald-900/20">
                    <div className="w-3 h-3 bg-emerald-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">VaR & CVaR calculations</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-teal-50 dark:bg-teal-900/20">
                    <div className="w-3 h-3 bg-teal-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Stress testing scenarios</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="group bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 via-red-500/5 to-transparent group-hover:from-orange-500/20 group-hover:via-red-500/10 transition-all duration-300"></div>
              <CardHeader className="relative pb-4">
                <CardTitle className="flex items-center space-x-3 text-xl">
                  <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
                    <Shield className="w-6 h-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-orange-700 to-red-700 dark:from-orange-300 dark:to-red-300 bg-clip-text text-transparent">
                    Risk Controls
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="relative">
                <p className="text-slate-600 dark:text-slate-400 mb-6 leading-relaxed">
                  Sophisticated constraints, sector limits, and institutional-grade risk management.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-orange-50 dark:bg-orange-900/20">
                    <div className="w-3 h-3 bg-orange-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Position size limits</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-red-50 dark:bg-red-900/20">
                    <div className="w-3 h-3 bg-red-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">Sector diversification</span>
                  </div>
                  <div className="flex items-center space-x-3 text-sm p-2 rounded-lg bg-pink-50 dark:bg-pink-900/20">
                    <div className="w-3 h-3 bg-pink-500 rounded-full flex-shrink-0"></div>
                    <span className="font-medium">ESG compliance</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Main Optimizer Form */}
        <section className="max-w-4xl mx-auto">
          <AdvancedPortfolioForm />
        </section>

        {/* Additional Features */}
        <section className="mt-16 text-center">
          <Card className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
            <CardHeader className="pb-4">
              <CardTitle className="text-2xl bg-gradient-to-r from-slate-900 to-purple-900 dark:from-slate-100 dark:to-purple-300 bg-clip-text text-transparent">
                What Makes This Advanced?
              </CardTitle>
              <CardDescription className="text-base text-slate-600 dark:text-slate-400">
                Professional features used by institutional investors
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 text-sm text-slate-600 dark:text-slate-400">
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <TrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">5+</div>
                  <div>Optimization Strategies</div>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">20+</div>
                  <div>Risk Metrics</div>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-violet-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-1">AI</div>
                  <div>Enhanced Forecasting</div>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <Sparkles className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-1">∞</div>
                  <div>Monte Carlo Sims</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
};

export default AdvancedOptimizer;