# AI Model Training Guide

## 🚀 Quick Start (Recommended)

To get started quickly with the most popular stocks:

```bash
cd backend
python quick_train.py
```

This will train XGBoost models for 10 essential stocks (~5-10 minutes).

## 📋 Training Requirements

### Dependencies
Make sure you have installed the ML dependencies:
```bash
pip install xgboost tensorflow keras ta joblib
```

### Models Available
1. **XGBoost** - Fast, lightweight, great for technical analysis
   - Training time: ~2-5 minutes per stock
   - Memory usage: Low
   - Best for: Short-term predictions (1-30 days)

2. **LSTM** - Deep learning, pattern recognition
   - Training time: ~10-20 minutes per stock  
   - Memory usage: High
   - Best for: Complex pattern recognition

3. **Prophet** - No training required, works out-of-the-box
   - Used for baseline predictions
   - Best for: Long-term trends

## 🎯 Training Options

### Option 1: Web Interface (Coming Soon)
Use the ModelTrainingCenter component in the frontend.

### Option 2: API Training
Train individual stocks via API:

```bash
# Train XGBoost only (fast)
curl -X POST http://localhost:5000/api/train/AAPL \
  -H "Content-Type: application/json" \
  -d '{"models": ["xgboost"], "period": "1y"}'

# Train both models (slow)
curl -X POST http://localhost:5000/api/train/AAPL \
  -H "Content-Type: application/json" \
  -d '{"models": ["xgboost", "lstm"], "period": "2y"}'
```

### Option 3: Batch Training Script
Train multiple stocks at once:

```bash
# Train XGBoost for top 10 stocks (parallel)
python train_popular_stocks.py --models xgboost --tickers AAPL MSFT GOOGL AMZN TSLA --parallel

# Train both models for specific stocks (sequential)
python train_popular_stocks.py --models xgboost lstm --tickers AAPL MSFT

# Train all popular stocks (30+ stocks)
python train_popular_stocks.py --models xgboost --parallel --max-workers 3
```

## 📊 Model Performance

### Expected Performance Ranges
- **XGBoost R² Score**: 0.40 - 0.80 (higher is better)
- **LSTM R² Score**: 0.35 - 0.75 (higher is better)
- **MAE (Mean Absolute Error)**: $1-5 for most stocks

### Performance Factors
- **Data Quality**: More recent, higher volume stocks perform better
- **Market Conditions**: Volatile periods are harder to predict
- **Training Period**: 1-2 years optimal for most models
- **Stock Characteristics**: Large-cap stocks generally more predictable

## 🛠️ Troubleshooting

### Common Issues

1. **"Insufficient data" Error**
   - Stock may be too new or have gaps in data
   - Try a longer training period or different stock

2. **"SVD did not converge" Error**  
   - Usually happens with very volatile stocks
   - Try XGBoost instead of LSTM

3. **Out of Memory Error**
   - Reduce batch size or train models sequentially
   - Close other applications
   - Train fewer stocks at once

4. **Model Loading Error**
   - Model files may be corrupted
   - Delete old model files and retrain

### Model File Locations
Models are saved in `backend/models/saved/`:
- XGBoost: `xgb_{ticker}.joblib`
- LSTM: `lstm_{ticker}.h5` 
- Scalers: `*_scaler_{ticker}.joblib`

To reset models, delete these files and retrain.

## 📈 Using Trained Models

### Check Model Availability
```bash
curl http://localhost:5000/api/models/available
```

### Make Predictions
```bash
# Use specific model
curl "http://localhost:5000/api/predict/AAPL?models=xgboost&days=10"

# Use ensemble (requires multiple trained models)
curl "http://localhost:5000/api/predict/AAPL?models=all&days=10"
```

### Model Status Check
```bash
# This endpoint doesn't exist yet, but could be added
curl http://localhost:5000/api/models/status/AAPL
```

## ⚡ Performance Tips

### For Fast Training (XGBoost only)
- Use `--parallel` flag with 2-4 workers
- Stick to 1-year training period
- Train popular, liquid stocks first

### For Best Accuracy (Both models)
- Use 2-year training period
- Train sequentially to avoid memory issues
- Include both XGBoost and LSTM for ensemble

### Memory Management
- **XGBoost**: Can train 3-4 stocks in parallel
- **LSTM**: Train one at a time
- **Mixed**: Train XGBoost in parallel, then LSTM sequentially

## 🎯 Recommended Training Strategy

### Phase 1: Essential Stocks (XGBoost)
```bash
python quick_train.py
```
Trains: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, V, PG

### Phase 2: Popular Stocks (XGBoost)
```bash
python train_popular_stocks.py --models xgboost --parallel
```

### Phase 3: Deep Learning (LSTM) - Optional
```bash
python train_popular_stocks.py --models lstm --tickers AAPL MSFT GOOGL AMZN TSLA
```

## 📝 Training Logs

Training results are logged to:
- Console output during training
- `training_results.json` (if using batch script)
- Model performance metrics in API responses

## 🚨 Important Notes

1. **Training Time**: Budget 2-5 minutes per stock for XGBoost, 10-20 for LSTM
2. **Storage**: Each model takes ~1-5 MB disk space
3. **Updates**: Retrain models monthly for best performance
4. **Legal**: Only use for educational/personal purposes
5. **Risk**: No guarantee of prediction accuracy - do not use for real trading without proper validation

## 🆘 Need Help?

If you encounter issues:
1. Check the console logs for detailed error messages
2. Try training with fewer stocks or smaller time periods
3. Ensure all dependencies are installed correctly
4. Check available disk space and memory

Happy training! 🎉