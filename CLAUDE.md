# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Common Development Commands

### Quick Start (Recommended)
```bash
# Start complete development environment with Docker
./scripts/start-dev.sh

# Or manually with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Backend (Flask API)
```bash
# Start Flask development server (local)
cd backend
python app.py

# Install Python dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train ML models (quick start)
python quick_train.py

# Train models for popular stocks
python train_popular_stocks.py --models xgboost --parallel

# Train specific stock
python train_popular_stocks.py --models xgboost --tickers AAPL MSFT

# Database operations
python -c "from models.database import init_database; init_database()"
```

### Frontend (React + TypeScript)
```bash
# Start development server
cd frontend
npm run dev

# Build for production
npm run build

# Build for development
npm run build:dev

# Lint code (fix TypeScript issues)
npm run lint

# Preview production build
npm run preview

# Install dependencies
npm install
```

### Docker Operations
```bash
# View logs
docker-compose logs -f [service_name]

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh

# Stop services
docker-compose down

# Rebuild containers
docker-compose build --no-cache
```

## 🏗 Architecture Overview

### Full-Stack Structure
- **Frontend**: React 18 + TypeScript + Vite + shadcn/ui + Tailwind CSS
- **Backend**: Flask + Python 3.12 with modular blueprint architecture + WebSocket support
- **Database**: PostgreSQL with SQLAlchemy ORM and Redis caching
- **ML Pipeline**: Ensemble of XGBoost + LSTM + Prophet models
- **Authentication**: JWT-based user authentication with role management
- **Real-time**: WebSocket integration for live data streaming
- **Data Source**: Yahoo Finance API via yfinance library

### Backend Architecture
The Flask application (`backend/app.py`) uses an application factory pattern with blueprint-based modular structure:

**Core System Routes:**
- **`routes/auth.py`**: JWT authentication, user registration/login, profile management
- **`routes/dashboard.py`**: Personalized dashboard with customizable widgets
- **`routes/notifications.py`**: Email notifications, price alerts, in-app messaging
- **`routes/analytics.py`**: Advanced portfolio analytics, risk metrics, performance analysis
- **`routes/websocket.py`**: Real-time WebSocket connections for live data streaming

**Financial Services Routes:**
- **`routes/optimize.py`**: Basic portfolio optimization using modern portfolio theory
- **`routes/advanced_optimize.py`**: Advanced optimization strategies (Sharpe Ratio, Risk Parity, Target Return)
- **`routes/predict.py`**: ML model predictions and ensemble forecasting
- **`routes/import_portfolio.py`**: CSV portfolio import/export functionality
- **`routes/autocomplete.py`**: Stock ticker search and validation

**Configuration & Infrastructure:**
- **`config.py`**: Environment-based configuration management (dev/staging/production)
- **`models/database.py`**: SQLAlchemy models and database operations
- **`utils/error_handlers.py`**: Centralized error handling and custom exceptions
- **`utils/cache.py`**: Redis caching layer with fallback strategies
- **`schemas.py`**: Pydantic validation schemas for API requests/responses

### ML Model Pipeline
1. **Data Ingestion**: Real-time market data from Yahoo Finance
2. **Feature Engineering**: 67 technical indicators (RSI, MACD, Bollinger Bands, volume metrics)
3. **Model Ensemble**: Weighted combination of three models:
   - **XGBoost**: Fast technical analysis predictor (1-30 day forecasts)
   - **LSTM**: Deep learning pattern recognition (sequential data)
   - **Prophet**: Time series forecasting with seasonality
4. **Trading Recommendations**: BuySellAdvisor engine with confidence scoring

### Frontend Architecture
**Authentication & User Management:**
- **`contexts/AuthContext.tsx`**: Global authentication state and JWT token management
- **`components/AuthModal.tsx`**: Login/registration modal with investment preferences

**Core Application Pages:**
- **`pages/Homepage.tsx`**: Landing page with feature overview
- **`pages/AIHub.tsx`**: Main AI analysis interface with prediction tools
- **`pages/AdvancedOptimizer.tsx`**: Portfolio optimization interface
- **`pages/Results.tsx`**: Basic optimization results display
- **`pages/AdvancedResults.tsx`**: Advanced analytics and performance metrics

**Key Components:**
- **`StockForecastChart.tsx`**: ML prediction visualization with confidence intervals
- **`EnhancedRecommendations.tsx`**: AI trading advice with detailed reasoning
- **`ModelTrainingCenter.tsx`**: ML model management and training interface
- **`PortfolioUploader.tsx`**: CSV import interface with validation
- **`EnhancedCharts.tsx`**: Interactive financial visualizations
- **`ErrorBoundary.tsx`**: React error boundary for graceful error handling

**Infrastructure:**
- **`lib/api-client.ts`**: Centralized API client with retry logic and error handling
- **`utils/accessibility.ts`**: WCAG 2.1 AA compliance utilities for 2025 legal requirements
- **`utils/pwa.ts`**: Progressive Web App features with service worker management
- **`utils/preloader.ts`**: Component preloading system for performance optimization
- **State Management**: TanStack Query for server state, React Context for auth, React hooks for local state
- **Styling**: Tailwind CSS with shadcn/ui component library
- **Real-time**: WebSocket integration for live data updates

## 📊 Model Training System

### Quick Start
```bash
cd backend
python quick_train.py  # Trains XGBoost for 10 popular stocks (~5-10 minutes)
```

### Model Types
- **XGBoost**: 2-5 minutes/stock, low memory, best for short-term predictions
- **LSTM**: 10-20 minutes/stock, high memory, best for pattern recognition  
- **Prophet**: No training required, used for baseline predictions

### Model Storage
Models are saved in `backend/models/saved/`:
- XGBoost: `xgb_{ticker}.joblib`
- LSTM: `lstm_{ticker}.h5`
- Scalers: `*_scaler_{ticker}.joblib`

### Performance Expectations
- XGBoost R² Score: 0.40 - 0.80
- LSTM R² Score: 0.35 - 0.75
- MAE: $1-5 for most stocks

## 🔧 Development Workflow

### Testing Commands
```bash
# Backend tests
cd backend
pytest tests/ -v                    # Run all tests with verbose output
pytest tests/test_api.py -v         # Run specific test file
pytest tests/ -k "test_health"      # Run tests matching pattern

# Frontend linting and type checking
cd frontend
npm run lint                        # ESLint TypeScript checking (strict mode)
npm run build                       # Build with TypeScript type checking
```

### Adding New Features
1. **Backend Routes**: Create new blueprints in `backend/routes/` with proper error handling using `@safe_api_call` decorator
2. **Database Models**: Add SQLAlchemy models in `backend/models/database.py` with proper relationships
3. **API Schemas**: Define Pydantic schemas in `backend/schemas.py` for request/response validation
4. **Frontend Components**: Add components in `src/components/` or pages in `src/pages/`
5. **API Integration**: Use centralized `apiClient` from `lib/api-client.ts` for all API calls
6. **Accessibility**: Ensure all new UI components meet WCAG 2.1 AA compliance using `utils/accessibility.ts`
7. **Real-time Features**: Integrate WebSocket events in `routes/websocket.py` for live updates

### Code Conventions
- **Backend**: Python Flask with modular blueprints, Pydantic validation, comprehensive error handling
- **Database**: SQLAlchemy ORM with proper migrations and indexing
- **Authentication**: JWT tokens with proper expiration and refresh logic
- **Caching**: Redis for high-performance caching with intelligent invalidation
- **Frontend**: TypeScript strict mode (no `any` types), functional components with hooks, error boundaries
- **Accessibility**: WCAG 2.1 AA compliance required for all UI components
- **PWA**: Progressive Web App features with service worker caching and offline support
- **Performance**: Code splitting, lazy loading, and component preloading for optimal performance
- **Styling**: Tailwind CSS utilities, shadcn/ui components for consistency
- **API**: RESTful endpoints under `/api` prefix with proper HTTP status codes
- **WebSocket**: Real-time features use Socket.IO with proper connection management

### Key Libraries & Dependencies
**Backend:**
- **Core**: Flask, Flask-CORS, Flask-JWT-Extended, Flask-SocketIO
- **Database**: SQLAlchemy, PostgreSQL, Redis
- **ML**: scikit-learn, xgboost, tensorflow, prophet, pyportfolioopt
- **Validation**: Pydantic, Marshmallow
- **Finance**: yfinance, ta (technical analysis)
- **Testing**: pytest, pytest-flask

**Frontend:**
- **Core**: React, TypeScript, Vite
- **UI**: shadcn/ui, Tailwind CSS, Lucide React
- **State**: TanStack Query, React Context
- **API**: Custom API client with Axios
- **Real-time**: Socket.IO client
- **Charts**: Recharts for financial visualizations

## 🎯 Important Implementation Notes

### Authentication & Security
- JWT-based authentication with access/refresh token rotation
- Password hashing using bcrypt with salt rounds
- Rate limiting (100 requests/minute default) with Flask-Limiter
- Input validation using Pydantic schemas on all endpoints
- CORS configured for specific origins, not wildcard
- SQL injection prevention through SQLAlchemy ORM
- XSS protection with proper response headers

### Database Architecture
- PostgreSQL for production with SQLAlchemy ORM
- Redis for caching with intelligent TTL and invalidation
- Database models include: User, Portfolio, Notification, PriceAlert, PredictionHistory, ModelMetadata
- Proper foreign key relationships and cascading deletes
- Database connection pooling and transaction management
- Automatic database initialization and migration support

### Real-time Features
- WebSocket connections using Flask-SocketIO
- JWT authentication for WebSocket connections
- Stock price streaming with smart subscription management
- Background tasks for price fetching and alert monitoring
- Connection cleanup and graceful degradation

### Portfolio Optimization
- Uses PyPortfolioOpt library for efficient frontier calculations
- Supports multiple strategies: Sharpe Ratio maximization, Risk Parity, Target Return
- Real-time data integration with Yahoo Finance
- CSV import/export functionality with comprehensive data validation
- Portfolio persistence and historical tracking

### ML Prediction System
- Ensemble approach combines three different model types (Prophet, XGBoost, LSTM)
- Feature engineering includes 67+ technical indicators
- Model training can run in parallel (XGBoost) or sequentially (LSTM)
- Model metadata tracking with performance metrics
- Confidence intervals and uncertainty quantification
- Automatic model retraining recommendations

### Notification System
- Multi-channel notifications (email, WebSocket, in-app)
- Background email processing with queue system
- Price alerts with customizable conditions
- SMTP integration for email notifications
- Notification history and read status tracking

### Performance & Caching
- Redis caching with multiple TTL strategies and intelligent cache invalidation
- Database connection pooling with SQLAlchemy for efficiency
- Yahoo Finance API caching layer to reduce external API calls
- Frontend lazy loading and code splitting with React.lazy()
- Component preloading system for anticipated user interactions
- Efficient re-renders with React memo and useMemo hooks
- WebSocket connection pooling and cleanup
- ML model lazy loading and caching to improve response times

### Error Handling
- Centralized error handling with custom exception classes
- Comprehensive logging with structured formats
- Graceful degradation for external service failures
- Frontend error boundaries for crash prevention
- Automatic retry logic for transient failures

## 📁 Key Files to Know

### Backend Entry Points
- `backend/app.py`: Flask application factory with all extensions initialized
- `backend/config.py`: Environment-based configuration management
- `backend/requirements.txt`: Python dependencies with version pinning
- `backend/quick_train.py`: Fast ML model training script for development

### Frontend Entry Points  
- `frontend/src/main.tsx`: React application entry point with providers
- `frontend/src/contexts/AuthContext.tsx`: Global authentication state management
- `frontend/src/lib/api-client.ts`: Centralized API client with error handling
- `frontend/package.json`: Node.js dependencies and build scripts

### Configuration & Infrastructure Files
- `backend/.env.example`: Environment variables template with all required settings
- `docker-compose.yml`: Production container orchestration
- `docker-compose.dev.yml`: Development environment overrides
- `backend/Dockerfile`: Multi-stage production container build
- `frontend/Dockerfile`: Nginx-based frontend container
- `.github/workflows/ci.yml`: Complete CI/CD pipeline
- `scripts/start-dev.sh`: Development environment startup script

### Key Database & Schema Files
- `backend/models/database.py`: SQLAlchemy models and database utilities
- `backend/schemas.py`: Pydantic validation schemas for all API endpoints
- `backend/utils/error_handlers.py`: Centralized error handling system
- `backend/utils/cache.py`: Redis caching utilities with fallback strategies

### Development & Testing
- `backend/tests/`: Pytest test suite with fixtures and mocks (run with `pytest tests/ -v`)
- `frontend/tsconfig.json`: TypeScript configuration with strict mode (no `any` types allowed)
- `frontend/tailwind.config.ts`: Tailwind CSS configuration with custom design tokens
- `frontend/eslint.config.js`: ESLint rules for TypeScript strict mode and code quality

## 🚀 Quick Reference

### API Endpoints Structure
```
/api/auth/*           - Authentication (login, register, profile)
/api/analytics/*      - Advanced portfolio analytics and risk metrics
/api/notifications/*  - Notification management and price alerts
/api/dashboard        - Personalized dashboard and widgets
/api/watchlist        - User stock watchlists
/api/predict/*        - ML predictions and model training
/api/optimize         - Portfolio optimization
/api/advanced-optimize - Advanced optimization strategies
/health               - System health check
```

### Environment Variables (Critical)
```bash
SECRET_KEY=           # Flask secret key
JWT_SECRET_KEY=       # JWT token signing key
DATABASE_URL=         # PostgreSQL connection string
REDIS_URL=            # Redis connection string
SMTP_USERNAME=        # Email service credentials
SMTP_PASSWORD=        # Email service password
```

### Docker Services
- **backend**: Flask API with ML models, authentication, and WebSocket support
- **frontend**: React app served by Nginx with PWA capabilities
- **db**: PostgreSQL database with connection pooling
- **redis**: Redis cache and session store with intelligent TTL management
- **nginx**: Reverse proxy for production deployments (optional, use `--profile production`)

## 🎯 Recent Improvements & Current State

### Performance Optimizations (Completed)
- **ML Model Lazy Loading**: Models are loaded on-demand to reduce memory usage and startup time
- **Database Connection Pooling**: SQLAlchemy connection pooling for improved database performance
- **API Caching Layer**: Yahoo Finance API responses cached with Redis to reduce external API calls
- **Frontend Code Splitting**: React.lazy() implementation for optimal bundle sizes

### Accessibility & PWA Features (Completed)
- **WCAG 2.1 AA Compliance**: Full accessibility support for 2025 legal requirements
- **Progressive Web App**: Service worker implementation with offline support and caching
- **Loading States**: Comprehensive loading spinners and error boundaries
- **Component Preloading**: Intelligent component preloading for better perceived performance

### Code Quality & TypeScript (Completed)
- **TypeScript Strict Mode**: Zero `any` types allowed, full type safety enforced
- **ESLint Configuration**: Comprehensive TypeScript linting rules
- **Error Boundaries**: React error boundaries for graceful failure handling
- **API Client**: Centralized API client with retry logic and proper error handling

### Pending Enhancements
- **ESG Integration**: Environmental, Social, and Governance factors in portfolio analysis
- **Advanced Risk Modeling**: Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations

### Key Files Added/Enhanced
- `frontend/src/utils/accessibility.ts`: WCAG 2.1 AA compliance utilities
- `frontend/src/utils/pwa.ts`: Progressive Web App management system  
- `frontend/src/utils/preloader.ts`: Component preloading optimization
- `frontend/src/components/AccessibleChart.tsx`: Accessible chart component
- `frontend/src/components/LoadingSpinner.tsx`: Loading state components
- `frontend/src/lib/api-client.ts`: Enhanced API client with full TypeScript typing