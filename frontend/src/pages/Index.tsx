
import EfficientFrontierChart from "@/components/EfficientFrontierChart";
import PortfolioForm from "@/components/PortfolioForm";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3, Moon, Sparkles, Sun, Target, TrendingUp, ArrowLeft } from "lucide-react";
import { Link, useLocation } from 'react-router-dom';
import { useDarkMode } from '@/hooks/useDarkMode';


const Index = () => {
  const { isDarkMode, toggleDarkMode } = useDarkMode();
  const location = useLocation();
  const result = location.state?.result;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900 transition-colors duration-500">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-indigo-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-gradient-to-br from-emerald-400/10 to-cyan-600/10 rounded-full blur-2xl animate-pulse delay-1000"></div>
      </div>

      {/* Header with dark mode toggle */}
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
        <section className="text-center mb-16 pt-8">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-slate-900 via-blue-900 to-indigo-900 dark:from-slate-100 dark:via-blue-300 dark:to-indigo-300 bg-clip-text text-transparent leading-tight">
              Optimize Your Portfolio with AI
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-400 mb-8 leading-relaxed">
              Leverage advanced AI algorithms and financial optimization techniques to build your ideal investment portfolio. 
              Get intelligent target return guidance, visualize the efficient frontier, simulate future growth with forecasting, and compare strategies like Sharpe Maximization, Risk Parity, and Target Return Optimization—all powered by real-time stock data.
            </p>

            <div className="flex flex-wrap justify-center gap-6 text-sm text-slate-500 dark:text-slate-400">
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Target className="w-4 h-4 text-blue-600" />
                <span>Mean Variance Optimization</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <BarChart3 className="w-4 h-4 text-indigo-600" />
                <span>Risk Analysis</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-sm px-4 py-2 rounded-full">
                <Sparkles className="w-4 h-4 text-purple-600" />
                <span>AI-Powered Insights</span>
              </div>
            </div>
          </div>
        </section>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {/* Portfolio Form */}
          <div className="space-y-6">
            <PortfolioForm />
            
            {/* Feature Cards */}
            {/* <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Card className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-4 h-4 text-white" />
                    </div>
                    <h3 className="font-semibold text-slate-900 dark:text-slate-100">Smart Allocation</h3>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Automatically calculate optimal asset weights based on your risk tolerance
                  </p>
                </CardContent>
              </Card>
              
              <Card className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center">
                      <BarChart3 className="w-4 h-4 text-white" />
                    </div>
                    <h3 className="font-semibold text-slate-900 dark:text-slate-100">Risk Analytics</h3>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Advanced risk metrics including Sharpe ratio and volatility analysis
                  </p>
                </CardContent>
              </Card>
            </div> */}
          </div>

          {/* Efficient Frontier Chart */}
          <div className="lg:pl-4">
            <EfficientFrontierChart/>
          </div>
        </div>

        {/* Bottom Section */}
        {/* <section className="text-center">
          <Card className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl">
            <CardHeader className="pb-4">
              <CardTitle className="text-2xl bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
                Ready to Optimize Your Investments?
              </CardTitle>
              <CardDescription className="text-base text-slate-600 dark:text-slate-400">
                Join thousands of investors who trust our AI-powered portfolio optimization
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="flex flex-wrap justify-center gap-8 text-sm text-slate-500 dark:text-slate-400">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">99.9%</div>
                  <div>Uptime</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">10k+</div>
                  <div>Portfolios Optimized</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">15%</div>
                  <div>Avg. Return Improvement</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section> */}
      </main>
    </div>
  );
};

export default Index;
