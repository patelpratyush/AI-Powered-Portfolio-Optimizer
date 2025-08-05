import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import TickerAutocomplete from "@/components/TickerAutocomplete";
import { 
  Upload, 
  Plus, 
  Trash2, 
  FileText, 
  CheckCircle, 
  AlertCircle,
  Download,
  Edit3
} from "lucide-react";
import { useDropzone } from 'react-dropzone';

interface PortfolioHolding {
  id: string;
  ticker: string;
  shares: number;
  avgPrice: number;
  currentValue?: number;
  marketPrice?: number;
  gainLoss?: number;
  gainLossPercent?: number;
}

interface PortfolioUploaderProps {
  onPortfolioImported: (holdings: PortfolioHolding[]) => void;
  existingHoldings?: PortfolioHolding[];
}

export const PortfolioUploader: React.FC<PortfolioUploaderProps> = ({ 
  onPortfolioImported, 
  existingHoldings = [] 
}) => {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>(existingHoldings);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResults, setValidationResults] = useState<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  } | null>(null);
  const [csvData, setCsvData] = useState<string>('');
  const [activeTab, setActiveTab] = useState('upload');

  // CSV Upload and Parsing
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csv = e.target?.result as string;
        setCsvData(csv);
        parseCSV(csv);
      };
      reader.readAsText(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxFiles: 1
  });

  const parseCSV = (csvText: string) => {
    try {
      const lines = csvText.split('\n').filter(line => line.trim());
      const headers = lines[0].toLowerCase().split(',').map(h => h.trim());
      
      // Validate headers
      const requiredHeaders = ['ticker', 'shares', 'price'];
      const hasRequiredHeaders = requiredHeaders.every(header => 
        headers.some(h => h.includes(header) || h.includes(header.replace('price', 'avg')))
      );

      if (!hasRequiredHeaders) {
        throw new Error('CSV must contain columns: ticker, shares, price/avgPrice');
      }

      const parsedHoldings: PortfolioHolding[] = [];
      
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        if (values.length < 3) continue;

        const tickerIndex = headers.findIndex(h => h.includes('ticker') || h.includes('symbol'));
        const sharesIndex = headers.findIndex(h => h.includes('shares') || h.includes('quantity'));
        const priceIndex = headers.findIndex(h => h.includes('price') || h.includes('avg'));

        if (tickerIndex >= 0 && sharesIndex >= 0 && priceIndex >= 0) {
          parsedHoldings.push({
            id: `${Date.now()}-${i}`,
            ticker: values[tickerIndex].toUpperCase(),
            shares: parseFloat(values[sharesIndex]) || 0,
            avgPrice: parseFloat(values[priceIndex]) || 0
          });
        }
      }

      setHoldings(parsedHoldings);
      setActiveTab('manual');
    } catch (error) {
      console.error('CSV parsing error:', error);
      setValidationResults({
        valid: false,
        errors: [`CSV parsing failed: ${error.message}`],
        warnings: []
      });
    }
  };

  // Manual Entry Functions
  const addNewHolding = () => {
    const newHolding: PortfolioHolding = {
      id: `manual-${Date.now()}`,
      ticker: '',
      shares: 0,
      avgPrice: 0
    };
    setHoldings([...holdings, newHolding]);
  };

  const updateHolding = (id: string, field: keyof PortfolioHolding, value: string | number) => {
    setHoldings(holdings.map(holding => 
      holding.id === id ? { ...holding, [field]: value } : holding
    ));
  };

  const deleteHolding = (id: string) => {
    setHoldings(holdings.filter(holding => holding.id !== id));
  };

  // Validation
  const validatePortfolio = async () => {
    setIsValidating(true);
    const errors: string[] = [];
    const warnings: string[] = [];

    // Basic validation
    holdings.forEach((holding, index) => {
      if (!holding.ticker) {
        errors.push(`Row ${index + 1}: Ticker is required`);
      }
      if (holding.shares <= 0) {
        errors.push(`Row ${index + 1}: Shares must be greater than 0`);
      }
      if (holding.avgPrice <= 0) {
        errors.push(`Row ${index + 1}: Average price must be greater than 0`);
      }
    });

    // Check for duplicates
    const tickers = holdings.map(h => h.ticker);
    const duplicates = tickers.filter((ticker, index) => tickers.indexOf(ticker) !== index);
    if (duplicates.length > 0) {
      warnings.push(`Duplicate tickers found: ${[...new Set(duplicates)].join(', ')}`);
    }

    // Validate tickers against API (simulate API call)
    try {
      const response = await fetch('/api/validate-portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ holdings })
      });
      
      if (response.ok) {
        const validationData = await response.json();
        if (validationData.invalid_tickers?.length > 0) {
          errors.push(`Invalid tickers: ${validationData.invalid_tickers.join(', ')}`);
        }
        
        // Add market data
        const enrichedHoldings = holdings.map(holding => {
          const marketData = validationData.market_data?.[holding.ticker];
          if (marketData) {
            const currentValue = holding.shares * marketData.price;
            const costBasis = holding.shares * holding.avgPrice;
            return {
              ...holding,
              marketPrice: marketData.price,
              currentValue,
              gainLoss: currentValue - costBasis,
              gainLossPercent: ((currentValue - costBasis) / costBasis) * 100
            };
          }
          return holding;
        });
        setHoldings(enrichedHoldings);
      }
    } catch (error) {
      warnings.push('Could not validate tickers against market data');
    }

    setValidationResults({
      valid: errors.length === 0,
      errors,
      warnings
    });
    setIsValidating(false);
  };

  const handleImport = () => {
    if (validationResults?.valid && holdings.length > 0) {
      onPortfolioImported(holdings);
    }
  };

  // CSV Template Download
  const downloadTemplate = () => {
    const template = 'ticker,shares,avgPrice\nAAPL,100,150.00\nGOOGL,50,2500.00\nMSFT,75,300.00';
    const blob = new Blob([template], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'portfolio_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const totalValue = holdings.reduce((sum, holding) => 
    sum + (holding.currentValue || holding.shares * holding.avgPrice), 0
  );

  const totalGainLoss = holdings.reduce((sum, holding) => sum + (holding.gainLoss || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Upload className="w-5 h-5 text-blue-600" />
            <span>Import Your Portfolio</span>
          </CardTitle>
          <CardDescription>
            Upload a CSV file or manually enter your current holdings to get personalized AI insights
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload">CSV Upload</TabsTrigger>
          <TabsTrigger value="manual">Manual Entry</TabsTrigger>
        </TabsList>

        {/* CSV Upload Tab */}
        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Upload CSV File</CardTitle>
              <CardDescription>
                Supports CSV, XLS, and XLSX files with ticker, shares, and price columns
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive 
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                {isDragActive ? (
                  <p className="text-blue-600">Drop your file here...</p>
                ) : (
                  <div>
                    <p className="text-lg font-medium mb-2">Drag & drop your portfolio file</p>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">or click to browse</p>
                    <Button variant="outline" size="sm">
                      Choose File
                    </Button>
                  </div>
                )}
              </div>

              {/* Template Download */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-gray-600" />
                  <div>
                    <p className="font-medium">Need a template?</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Download our CSV template to get started
                    </p>
                  </div>
                </div>
                <Button variant="outline" size="sm" onClick={downloadTemplate}>
                  <Download className="w-4 h-4 mr-2" />
                  Template
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Manual Entry Tab */}
        <TabsContent value="manual" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg">Portfolio Holdings</CardTitle>
                <CardDescription>
                  Add your individual stock positions manually
                </CardDescription>
              </div>
              <Button onClick={addNewHolding} size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Add Position
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {holdings.length === 0 ? (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    <Edit3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No holdings added yet. Click "Add Position" to start.</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {holdings.map((holding, index) => (
                      <div key={holding.id} className="grid grid-cols-12 gap-4 items-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="col-span-3">
                          <Label htmlFor={`ticker-${holding.id}`} className="text-xs font-medium">
                            Ticker
                          </Label>
                          <div className="mt-1">
                            <TickerAutocomplete
                              value={holding.ticker}
                              onChange={(value) => updateHolding(holding.id, 'ticker', value)}
                              onSelect={(suggestion) => updateHolding(holding.id, 'ticker', suggestion.symbol)}
                              placeholder="AAPL"
                              showPopular={false}
                              className="w-full"
                            />
                          </div>
                        </div>
                        <div className="col-span-2">
                          <Label htmlFor={`shares-${holding.id}`} className="text-xs font-medium">
                            Shares
                          </Label>
                          <Input
                            id={`shares-${holding.id}`}
                            type="number"
                            value={holding.shares || ''}
                            onChange={(e) => updateHolding(holding.id, 'shares', parseFloat(e.target.value) || 0)}
                            placeholder="100"
                            className="mt-1"
                          />
                        </div>
                        <div className="col-span-2">
                          <Label htmlFor={`price-${holding.id}`} className="text-xs font-medium">
                            Avg Price
                          </Label>
                          <Input
                            id={`price-${holding.id}`}
                            type="number"
                            step="0.01"
                            value={holding.avgPrice || ''}
                            onChange={(e) => updateHolding(holding.id, 'avgPrice', parseFloat(e.target.value) || 0)}
                            placeholder="150.00"
                            className="mt-1"
                          />
                        </div>
                        <div className="col-span-2">
                          <Label className="text-xs font-medium">Current Value</Label>
                          <div className="mt-1 p-2 bg-white dark:bg-gray-700 rounded border text-sm">
                            ${(holding.currentValue || holding.shares * holding.avgPrice).toLocaleString()}
                          </div>
                        </div>
                        <div className="col-span-2">
                          <Label className="text-xs font-medium">Gain/Loss</Label>
                          <div className="mt-1 p-2 bg-white dark:bg-gray-700 rounded border text-sm">
                            {holding.gainLoss !== undefined ? (
                              <span className={holding.gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}>
                                ${holding.gainLoss.toFixed(2)}
                              </span>
                            ) : (
                              <span className="text-gray-400">—</span>
                            )}
                          </div>
                        </div>
                        <div className="col-span-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => deleteHolding(holding.id)}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Portfolio Summary */}
      {holdings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Portfolio Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="text-sm text-blue-700 dark:text-blue-300">Total Positions</div>
                <div className="text-2xl font-bold text-blue-800 dark:text-blue-200">
                  {holdings.length}
                </div>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="text-sm text-green-700 dark:text-green-300">Total Value</div>
                <div className="text-2xl font-bold text-green-800 dark:text-green-200">
                  ${totalValue.toLocaleString()}
                </div>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="text-sm text-purple-700 dark:text-purple-300">Total Gain/Loss</div>
                <div className={`text-2xl font-bold ${totalGainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${totalGainLoss.toFixed(2)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Validation Results */}
      {validationResults && (
        <div className="space-y-2">
          {validationResults.errors.map((error, index) => (
            <Alert key={index} variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ))}
          {validationResults.warnings.map((warning, index) => (
            <Alert key={index}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{warning}</AlertDescription>
            </Alert>
          ))}
          {validationResults.valid && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>Portfolio validation successful! Ready to import.</AlertDescription>
            </Alert>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <Button 
          onClick={validatePortfolio} 
          disabled={holdings.length === 0 || isValidating}
          variant="outline"
        >
          {isValidating ? 'Validating...' : 'Validate Portfolio'}
        </Button>
        <Button 
          onClick={handleImport}
          disabled={!validationResults?.valid || holdings.length === 0}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <CheckCircle className="w-4 h-4 mr-2" />
          Import Portfolio
        </Button>
      </div>
    </div>
  );
};

export default PortfolioUploader;