// src/pages/CurrentPortfolioAnalyzer.tsx

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import axios from "axios";
import { ArcElement, BubbleController, Chart, Legend, LinearScale, PointElement, Tooltip } from "chart.js";
import { useState } from "react";
import { Bubble, Pie } from "react-chartjs-2";

Chart.register(ArcElement, Tooltip, Legend, BubbleController, LinearScale, PointElement);

export default function CurrentPortfolioAnalyzer() {
  const [tickers, setTickers] = useState(["AAPL", "GOOGL"]);
  const [shares, setShares] = useState([5, 3]);
  const [dateRange, setDateRange] = useState({ start: "2023-01-01", end: "2024-01-01" });
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/portfolio/analyze", {
        tickers,
        shares,
        start: dateRange.start,
        end: dateRange.end,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze portfolio");
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
        <Button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Portfolio"}
        </Button>

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
                    labels: result.details.map((d: any) => d.ticker),
                    datasets: [
                      {
                        label: "Value ($)",
                        data: result.details.map((d: any) => d.value),
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
                      ...Object.entries(result.comparison || {}).map(([key, data]: any) => ({
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
