# AI-Powered Portfolio Optimizer

A full-stack web application that uses modern portfolio theory, AI-powered insights, and financial data to help users optimize their investment portfolios. Users can input stock tickers, select optimization strategies, set target returns, and visualize results such as the efficient frontier and future forecast.

## 🚀 Features

- 📈 Portfolio optimization using Sharpe Ratio, Risk Parity, and Target Return.
- 💡 AI-generated target return suggestions (conservative, moderate, aggressive).
- 📊 Efficient Frontier visualization and expected growth forecasting.
- 🧠 Real-time stock data integration (Yahoo Finance).
- 🎯 Dark mode support and responsive UI with Tailwind + React.
- 🔁 Flask-based API with endpoints for optimization, forecasting, and autocomplete.

---

## 📂 Project Structure

```bash
.
├── backend
│   ├── app.py
│   ├── models
│   ├── requirements.txt
│   ├── routes
│   │   ├── autocomplete.py
│   │   ├── clean_tickers.csv
│   │   └── optimize.py
│   └── utils
├── frontend
│   ├── bun.lockb
│   ├── components.json
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.js
│   ├── public
│   │   └── robots.txt
│   ├── README.md
│   ├── src
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── components
│   │   │   ├── DateRangeSelector.tsx
│   │   │   ├── EfficientFrontierChart.tsx
│   │   │   ├── PortfolioComposition.tsx
│   │   │   ├── PortfolioForm.tsx
│   │   │   ├── PortfolioMetrics.tsx
│   │   │   └── StockTickerInput.tsx
│   │   ├── hooks
│   │   │   ├── use-mobile.tsx
│   │   │   └── use-toast.ts
│   │   ├── index.css
│   │   ├── lib
│   │   │   └── utils.ts
│   │   ├── main.tsx
│   │   ├── pages
│   │   │   ├── CurrentPortfolioAnalyzer.tsx
│   │   │   ├── Index.tsx
│   │   │   ├── NotFound.tsx
│   │   │   └── Results.tsx
│   │   └── vite-env.d.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
└── README.md
```

---

## 🧪 Getting Started

### 🔧 Backend (Flask API)

1. **Navigate to the backend folder:**

    ```bash
    cd backend
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip3 install -r requirements.txt
    ```

4. **Run the Flask app:**

    ```bash
    python3 app.py
    ```

The API will run at [http://localhost:5000](http://localhost:5000)

### 🌐 Frontend (React + Vite)

1. **Navigate to the frontend folder:**

    ```bash
    cd frontend
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Run the development server:**

    ```bash
    npm run dev
    ```

The app will be available at [http://localhost:5173](http://localhost:5173) and will connect to the backend on port 5000.

## 🧠 Technologies Used

- **Frontend:** React, Vite, Tailwind CSS, TypeScript, Recharts

- **Backend:** Python, Flask, Prophet, NumPy, SciPy, Pandas

- **APIs:** Yahoo Finance (via yfinance)
