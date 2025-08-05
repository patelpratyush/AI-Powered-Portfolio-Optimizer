
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Homepage from "./pages/Homepage";
import Index from "./pages/Index";
import Results from "./pages/Results";
import NotFound from "./pages/NotFound";
import CurrentPortfolioAnalyzer from "./pages/CurrentPortfolioAnalyzer";
import AdvancedOptimizer from "./pages/AdvancedOptimizer";
import AdvancedResults from "./pages/AdvancedResults";
import AIHub from "./pages/AIHub";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
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
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
