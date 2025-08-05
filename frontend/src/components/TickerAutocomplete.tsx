import React, { useState, useEffect, useRef } from 'react';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Search, TrendingUp, X, Loader2 } from "lucide-react";

interface TickerSuggestion {
  symbol: string;
  name: string;
}

interface TickerAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onSelect?: (ticker: TickerSuggestion) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  showPopular?: boolean;
}

// Popular stocks for quick selection when no input
const POPULAR_TICKERS: TickerSuggestion[] = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
  { symbol: 'JNJ', name: 'Johnson & Johnson' },
  { symbol: 'V', name: 'Visa Inc.' },
];

export const TickerAutocomplete: React.FC<TickerAutocompleteProps> = ({
  value,
  onChange,
  onSelect,
  placeholder = "Enter ticker symbol (e.g., AAPL)",
  className = "",
  disabled = false,
  showPopular = true
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<TickerSuggestion[]>([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch suggestions from backend API
  useEffect(() => {
    if (value.length === 0) {
      setSuggestions(showPopular ? POPULAR_TICKERS : []);
      setIsLoading(false);
      return;
    }

    if (value.length >= 1) {
      setIsLoading(true);
      
      // Debounce API calls
      const timeoutId = setTimeout(async () => {
        try {
          const response = await fetch(`/api/autocomplete?q=${encodeURIComponent(value)}`);
          if (response.ok) {
            const data = await response.json();
            setSuggestions(data || []);
          } else {
            console.error('Autocomplete API error:', response.status);
            setSuggestions([]);
          }
        } catch (error) {
          console.error('Autocomplete fetch error:', error);
          setSuggestions([]);
        } finally {
          setIsLoading(false);
        }
      }, 200); // 200ms debounce

      return () => clearTimeout(timeoutId);
    }
  }, [value, showPopular]);

  // Handle clicks outside dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) &&
          inputRef.current && !inputRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.toUpperCase();
    onChange(newValue);
    setIsOpen(true);
    setHighlightedIndex(-1);
  };

  const handleSuggestionClick = (suggestion: TickerSuggestion) => {
    onChange(suggestion.symbol);
    onSelect?.(suggestion);
    setIsOpen(false);
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightedIndex(prev => prev > 0 ? prev - 1 : prev);
        break;
      case 'Enter':
        e.preventDefault();
        if (highlightedIndex >= 0) {
          handleSuggestionClick(suggestions[highlightedIndex]);
        } else if (suggestions.length > 0) {
          handleSuggestionClick(suggestions[0]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        inputRef.current?.blur();
        break;
    }
  };


  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
        <Input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className={`pl-10 pr-10 ${value ? 'font-mono text-sm' : ''}`}
        />
        {value && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => {
              onChange('');
              inputRef.current?.focus();
            }}
          >
            <X className="w-3 h-3" />
          </Button>
        )}
      </div>

      {isOpen && suggestions.length > 0 && (
        <Card 
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 shadow-lg border border-gray-200 dark:border-gray-700 max-h-80 overflow-y-auto"
        >
          <CardContent className="p-0">
            {value.length === 0 && showPopular && (
              <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2 flex items-center">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Popular Stocks
                </p>
              </div>
            )}
            
            {isLoading && (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <Loader2 className="w-6 h-6 mx-auto mb-2 animate-spin" />
                <p className="text-sm">Searching...</p>
              </div>
            )}
            
            {!isLoading && suggestions.map((suggestion, index) => (
              <div
                key={suggestion.symbol}
                className={`p-3 cursor-pointer border-b border-gray-100 dark:border-gray-800 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors ${
                  index === highlightedIndex ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                }`}
                onClick={() => handleSuggestionClick(suggestion)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-mono font-semibold text-sm text-blue-600 dark:text-blue-400">
                        {suggestion.symbol}
                      </span>
                      <Badge variant="outline" className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        STOCK
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                      {suggestion.name}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            
            {!isLoading && value.length > 0 && suggestions.length === 0 && (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No matches found for "{value}"</p>
                <p className="text-xs mt-1">Try a different ticker symbol</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default TickerAutocomplete;