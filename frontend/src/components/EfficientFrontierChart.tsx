
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Target, TrendingUp } from "lucide-react";
import { useState, useEffect } from 'react';

const EfficientFrontierChart = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check if dark mode is active
    const checkDarkMode = () => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    };

    checkDarkMode();
    
    // Listen for changes to the dark class
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  return (
    <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border-white/20 dark:border-slate-700/20 shadow-xl hover:shadow-2xl transition-all duration-300">
      <CardHeader className="pb-6">
        <CardTitle className="flex items-center space-x-3 text-2xl">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
            <LineChart className="w-5 h-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-slate-900 to-blue-900 dark:from-slate-100 dark:to-blue-300 bg-clip-text text-transparent">
            Efficient Frontier
          </span>
        </CardTitle>
        <CardDescription className="text-base text-slate-600 dark:text-slate-400">
          Risk vs Return optimization curve with your portfolio position
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 bg-gradient-to-br from-slate-50/80 to-blue-50/80 dark:from-slate-800/50 dark:to-slate-700/50 rounded-2xl border border-white/40 dark:border-slate-600/40 flex items-center justify-center relative overflow-hidden backdrop-blur-sm">
          {/* Enhanced background grid */}
          <div className="absolute inset-0 opacity-20">
            <div className={`absolute bottom-6 left-6 right-6 h-0.5 bg-gradient-to-r from-transparent ${isDarkMode ? 'via-slate-600' : 'via-slate-400'} to-transparent`}></div>
            <div className={`absolute bottom-6 left-6 top-6 w-0.5 bg-gradient-to-b from-transparent ${isDarkMode ? 'via-slate-600' : 'via-slate-400'} to-transparent`}></div>
            {/* Grid lines */}
            <div className="absolute inset-0">
              <div className={`absolute bottom-1/4 left-6 right-6 h-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
              <div className={`absolute bottom-2/4 left-6 right-6 h-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
              <div className={`absolute bottom-3/4 left-6 right-6 h-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
              <div className={`absolute bottom-6 left-1/4 top-6 w-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
              <div className={`absolute bottom-6 left-2/4 top-6 w-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
              <div className={`absolute bottom-6 left-3/4 top-6 w-px ${isDarkMode ? 'bg-slate-600' : 'bg-slate-300'}`}></div>
            </div>
          </div>
          
          {/* Enhanced efficient frontier curve */}
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 300 200">
            <defs>
              <linearGradient id="curveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3B82F6" />
                <stop offset="50%" stopColor="#6366F1" />
                <stop offset="100%" stopColor="#8B5CF6" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge> 
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            <path
              d="M 50 150 Q 100 50 250 80"
              stroke="url(#curveGradient)"
              strokeWidth="4"
              fill="none"
              opacity="0.8"
              filter="url(#glow)"
            />
            <circle cx="150" cy="100" r="8" fill="#EF4444" opacity="0.9" className="animate-pulse" />
            <circle cx="150" cy="100" r="12" fill="none" stroke="#EF4444" strokeWidth="2" opacity="0.6" className="animate-ping" />
          </svg>
          
          <div className={`text-center z-10 ${isDarkMode ? 'bg-slate-800/60' : 'bg-white/60'} backdrop-blur-sm rounded-2xl p-6 border ${isDarkMode ? 'border-slate-600/40' : 'border-white/40'}`}>
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Target className="w-6 h-6 text-white" />
            </div>
            <p className={`text-base font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'} mb-2`}>
              Interactive Chart Loading
            </p>
            <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
              Chart will display when portfolio is optimized
            </p>
          </div>
        </div>
        
        <div className="mt-6 flex items-center justify-between">
          <div className={`flex items-center space-x-2 text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
            <TrendingUp className="w-4 h-4" />
            <span>Risk (Volatility)</span>
          </div>
          <div className={`flex items-center space-x-2 text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
            <span>Expected Return</span>
            <TrendingUp className="w-4 h-4" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EfficientFrontierChart;
