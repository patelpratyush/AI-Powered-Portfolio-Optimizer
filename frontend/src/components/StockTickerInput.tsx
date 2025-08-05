import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, X } from "lucide-react";
import { useEffect, useRef, useState } from 'react';

interface StockTickerInputProps {
  tickers: string[];
  onTickersChange: (tickers: string[]) => void;
}

interface Suggestion {
  symbol: string;
  name: string;
}

const StockTickerInput = ({ tickers, onTickersChange }: StockTickerInputProps) => {
  const [newTicker, setNewTicker] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [searchTimeout, setSearchTimeout] = useState<NodeJS.Timeout | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
  const handleClickOutside = (event: MouseEvent) => {
    if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
      setSuggestions([]);
    }
  };

  document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const fetchSuggestions = async (query: string) => {
    if (!query) return setSuggestions([]);
    try {
      const res = await fetch(`/api/autocomplete?q=${query}`);
      const data = await res.json();
      setSuggestions(data);
    } catch (err) {
      console.error("Error fetching suggestions:", err);
      setSuggestions([]);
    }
  };

  const addTicker = (ticker: string) => {
    const symbol = ticker.toUpperCase().trim();
    if (symbol && !tickers.includes(symbol)) {
      onTickersChange([...tickers, symbol]);
      setNewTicker('');
      setSuggestions([]);
    }
  };

  const removeTicker = (tickerToRemove: string) => {
    onTickersChange(tickers.filter(ticker => ticker !== tickerToRemove));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      addTicker(newTicker);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="ticker-input" className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
          <div className="w-2 h-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"></div>
          <span>Stock Tickers</span>
        </Label>
        <div className="flex space-x-2 mt-2 relative" ref={wrapperRef}>
          <Input
            id="ticker-input"
            placeholder="Enter ticker (e.g., AAPL)"
            value={newTicker}
            onChange={(e) => {
              const val = e.target.value.toUpperCase();
              setNewTicker(val);
              if (searchTimeout) clearTimeout(searchTimeout);
              const timeout = setTimeout(() => fetchSuggestions(val), 300); // debounce
              setSearchTimeout(timeout);
            }}
            onKeyPress={handleKeyPress}
            className="flex-1"
          />
          <Button onClick={() => addTicker(newTicker)} size="sm" variant="outline">
            <Plus className="w-4 h-4" />
          </Button>

          {suggestions.length > 0 && (
            <ul className="absolute z-10 top-full mt-1 w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded shadow max-h-60 overflow-y-auto text-sm">
              {suggestions.map((sugg) => (
                <li
                  key={sugg.symbol}
                  onClick={() => addTicker(sugg.symbol)}
                  className="px-4 py-2 hover:bg-blue-100 dark:hover:bg-slate-700 cursor-pointer"
                >
                  <span className="font-semibold">{sugg.symbol}</span> - {sugg.name}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {tickers.map((ticker) => (
          <Badge key={ticker} variant="secondary" className="px-3 py-1">
            {ticker}
            <button onClick={() => removeTicker(ticker)} className="ml-2">
              <X className="w-3 h-3" />
            </button>
          </Badge>
        ))}
      </div>
    </div>
  );
};

export default StockTickerInput;
