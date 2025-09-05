// src/pages/CurrentPortfolioAnalyzer.tsx

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, RefreshCw } from "lucide-react";
import axios from "axios";
import { ArcElement, BubbleController, Chart, Legend, LinearScale, PointElement, Tooltip } from "chart.js";
import { useState } from "react";
import { Bubble, Pie } from "react-chartjs-2";

Chart.register(ArcElement, Tooltip, Legend, BubbleController, LinearScale, PointElement);

export default function CurrentPortfolioAnalyzer() {
  const [tickers, setTickers] = useState(["AAPL", "GOOGL"]);
  const [shares, setShares] = useState([5, 3]);
  const [dateRange, setDateRange] = useState({ start: "2023-01-01", end: "2024-01-01" });
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post("http://localhost:5000/api/portfolio/analyze", {
        tickers,
        shares,
        start: dateRange.start,
        end: dateRange.end,
      });
      setResult(res.data);
    } catch (err: unknown) {
      console.error(err);
      const error = err as { response?: { data?: { message?: string } }; message?: string };
      setError(error.response?.data?.message || error.message || "Failed to analyze portfolio");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="p-6 space-y-6">
      <CardHeader>
        <CardTitle>📊 Analyze Your Current Portfolio</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {tickers.map((ticker, i) => (
          <div key={i} className="flex space-x-4 items-center">
            <Input
              value={ticker}
              onChange={(e) => {
                const newT = [...tickers];
                newT[i] = e.target.value.toUpperCase();
                setTickers(newT);
              }}
              placeholder="Ticker"
            />
            <Input
              type="number"
              value={shares[i]}
              onChange={(e) => {
                const newS = [...shares];
                newS[i] = Number(e.target.value);
                setShares(newS);
              }}
              placeholder="Shares"
            />
          </div>
        ))}
        <Button
          variant="secondary"
          onClick={() => {
            setTickers([...tickers, ""]);
            setShares([...shares, 0]);
          }}
        >
          ➕ Add Stock
        </Button>
        <div className="flex space-x-4">
          <Input
            type="date"
            value={dateRange.start}
            onChange={(e) => setDateRange((d) => ({ ...d, start: e.target.value }))}
          />
          <Input
            type="date"
            value={dateRange.end}
            onChange={(e) => setDateRange((d) => ({ ...d, end: e.target.value }))}
          />
        </div>
        <Button onClick={handleAnalyze} disabled={loading} className="w-full">
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Analyzing Portfolio...
            </>
          ) : (
            "Analyze Portfolio"
          )}
        </Button>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleAnalyze}
                className="ml-4"
              >
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {result && (
          <>
            <div className="text-xl font-semibold">Your Portfolio Metrics</div>
            <ul className="list-disc list-inside">
              <li>Expected Return: {(result.expected_return * 100).toFixed(2)}%</li>
              <li>Volatility: {(result.volatility * 100).toFixed(2)}%</li>
              <li>Sharpe Ratio: {result.sharpe_ratio}</li>
            </ul>

            <div className="grid md:grid-cols-2 gap-6 mt-6">
              <div>
                <div className="text-md font-medium mb-2">📈 Holdings Breakdown (Pie)</div>
                <Pie
                  data={{
                    labels: (result.details as Array<{ ticker: string }>).map((d) => d.ticker),
                    datasets: [
                      {
                        label: "Value ($)",
                        data: (result.details as Array<{ value: number }>).map((d) => d.value),
                        backgroundColor: ["#60a5fa", "#818cf8", "#c084fc", "#f472b6", "#facc15"],
                      },
                    ],
                  }}
                />
              </div>

              <div>
                <div className="text-md font-medium mb-2">📉 Sharpe vs Volatility (Bubble)</div>
                <Bubble
                  data={{
                    datasets: [
                      {
                        label: "Your Portfolio",
                        data: [{ x: result.volatility, y: result.sharpe_ratio, r: 10 }],
                        backgroundColor: "#22d3ee",
                      },
                      ...Object.entries(result.comparison || {}).map(([key, data]: [string, { volatility: number; sharpe_ratio: number }]) => ({
                        label: key,
                        data: [{ x: data.volatility, y: data.sharpe_ratio, r: 10 }],
                        backgroundColor: key === "max_sharpe" ? "#4ade80" : "#fbbf24",
                      })),
                    ],
                  }}
                  options={{
                    scales: {
                      x: { title: { display: true, text: "Volatility" } },
                      y: { title: { display: true, text: "Sharpe Ratio" } },
                    },
                  }}
                />
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
