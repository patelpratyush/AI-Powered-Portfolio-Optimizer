import React, { Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { LoadingSpinner } from "./components/LoadingSpinner";

// Lazy load components for better performance
const Homepage = React.lazy(() => import("./pages/Homepage"));
const Index = React.lazy(() => import("./pages/Index"));
const Results = React.lazy(() => import("./pages/Results"));
const NotFound = React.lazy(() => import("./pages/NotFound"));
const CurrentPortfolioAnalyzer = React.lazy(() => import("./pages/CurrentPortfolioAnalyzer"));
const AdvancedOptimizer = React.lazy(() => import("./pages/AdvancedOptimizer"));
const AdvancedResults = React.lazy(() => import("./pages/AdvancedResults"));
const AIHub = React.lazy(() => import("./pages/AIHub"));

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<Homepage />} />
              <Route path="/basic" element={<Index />} />
              <Route path="/results" element={<Results />} />
              <Route path="/analyze" element={<CurrentPortfolioAnalyzer />} />
              <Route path="/advanced" element={<AdvancedOptimizer />} />
              <Route path="/advanced-results" element={<AdvancedResults />} />
              <Route path="/ai-hub" element={<AIHub />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
