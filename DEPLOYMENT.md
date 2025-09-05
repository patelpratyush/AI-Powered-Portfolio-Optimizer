# 🚀 Deployment Guide

This guide covers different deployment options for the AI-Powered Portfolio Optimizer.

## 📋 Prerequisites

- Docker and Docker Compose
- Git
- Basic knowledge of environment variables
- SSL certificates (for production HTTPS)

## 🏃‍♂️ Quick Start with Docker

### Development Environment

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Powered-Portfolio-Optimizer
   ```

2. **Set up environment variables:**
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. **Start development services:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - API Health Check: http://localhost:5000/health

### Production Environment

1. **Set up production environment variables:**
   ```bash
   export SECRET_KEY="your-super-secret-key-here"
   export JWT_SECRET_KEY="your-jwt-secret-key-here"
   ```

2. **Start production services:**
   ```bash
   docker-compose --profile production up -d
   ```

3. **Set up SSL certificates** (place in `./ssl/` directory):
   - `cert.pem` - SSL certificate
   - `key.pem` - Private key

## 🌐 Cloud Deployment Options

### Option 1: Heroku

1. **Install Heroku CLI and login:**
   ```bash
   heroku login
   ```

2. **Create applications:**
   ```bash
   # Backend
   heroku create your-app-backend
   heroku addons:create heroku-postgresql:hobby-dev -a your-app-backend
   heroku addons:create heroku-redis:hobby-dev -a your-app-backend
   
   # Frontend
   heroku create your-app-frontend
   ```

3. **Deploy backend:**
   ```bash
   cd backend
   git init
   heroku git:remote -a your-app-backend
   
   # Set environment variables
   heroku config:set FLASK_ENV=production
   heroku config:set SECRET_KEY=your-secret-key
   heroku config:set JWT_SECRET_KEY=your-jwt-secret
   
   git add .
   git commit -m "Deploy backend"
   git push heroku main
   ```

4. **Deploy frontend:**
   ```bash
   cd ../frontend
   # Update API base URL in config
   echo "VITE_API_URL=https://your-app-backend.herokuapp.com" > .env.production
   
   git init
   heroku git:remote -a your-app-frontend
   heroku buildpacks:set https://github.com/heroku/heroku-buildpack-static.git
   
   git add .
   git commit -m "Deploy frontend"
   git push heroku main
   ```

### Option 2: AWS ECS with Fargate

1. **Build and push Docker images:**
   ```bash
   # Build images
   docker build -t your-registry/portfolio-backend ./backend
   docker build -t your-registry/portfolio-frontend ./frontend
   
   # Push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   docker push your-registry/portfolio-backend
   docker push your-registry/portfolio-frontend
   ```

2. **Create ECS service definitions** (use AWS Console or Terraform)

3. **Set up Application Load Balancer** for routing

### Option 3: DigitalOcean App Platform

1. **Connect your GitHub repository**

2. **Configure app spec** (create `.do/app.yaml`):
   ```yaml
   name: portfolio-optimizer
   services:
   - name: backend
     source_dir: /backend
     github:
       repo: your-username/AI-Powered-Portfolio-Optimizer
       branch: main
     run_command: gunicorn --bind 0.0.0.0:$PORT app:app
     environment_slug: python
     instance_count: 1
     instance_size_slug: basic-xxs
     envs:
     - key: FLASK_ENV
       value: production
   
   - name: frontend
     source_dir: /frontend
     github:
       repo: your-username/AI-Powered-Portfolio-Optimizer
       branch: main
     run_command: npm run build && npm run preview
     environment_slug: node-js
     instance_count: 1
     instance_size_slug: basic-xxs
   
   databases:
   - name: portfolio-db
     engine: PG
     version: "13"
     size_slug: db-s-1vcpu-1gb
   ```

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```bash
# Required
FLASK_ENV=production
SECRET_KEY=your-super-secret-key
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db

# Optional
JWT_SECRET_KEY=your-jwt-secret
RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
```

**Frontend:**
```bash
VITE_API_URL=https://your-backend-url.com
```

### Database Migration

```bash
# Run after deployment
docker-compose exec backend python -c "
from models.database import init_database
init_database()
print('Database initialized')
"
```

## 🚦 Health Checks and Monitoring

### Health Check Endpoints

- **Backend:** `GET /health`
- **Frontend:** `GET /health` (via Nginx)

### Monitoring Setup

1. **Application Performance:**
   ```bash
   # Add to backend requirements.txt
   flask-monitoring-dashboard
   prometheus-flask-exporter
   ```

2. **Log Aggregation:**
   - Use ELK stack (Elasticsearch, Logstash, Kibana)
   - Or cloud solutions (CloudWatch, Datadog, New Relic)

3. **Error Tracking:**
   ```bash
   pip install sentry-sdk[flask]
   ```

## 🔒 Security Considerations

### Production Checklist

- [ ] Use HTTPS everywhere
- [ ] Set strong SECRET_KEY and JWT_SECRET_KEY
- [ ] Enable rate limiting
- [ ] Set up proper CORS origins
- [ ] Use environment variables for secrets
- [ ] Enable request logging
- [ ] Set up monitoring and alerting
- [ ] Regular security updates
- [ ] Backup strategy for database and models

### SSL/TLS Setup

1. **Let's Encrypt (Free):**
   ```bash
   certbot certonly --webroot -w /var/www/html -d your-domain.com
   ```

2. **Update Nginx configuration** to use SSL certificates

## 📊 Performance Optimization

### Backend

- Use Redis for caching API responses
- Implement connection pooling for database
- Use background jobs for ML model training
- Enable gzip compression

### Frontend

- Enable asset caching
- Use CDN for static assets
- Implement code splitting
- Optimize bundle size

### Database

- Create proper indexes
- Regular vacuuming (PostgreSQL)
- Monitor query performance
- Set up read replicas if needed

## 🔄 CI/CD Pipeline

The included GitHub Actions workflow (`.github/workflows/ci.yml`) provides:

- Automated testing
- Security scanning
- Docker image building
- Deployment automation

### Customize for Your Deployment

1. **Update deployment step** in CI pipeline
2. **Add environment-specific secrets** to GitHub
3. **Configure branch protection** rules

## 🆘 Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using the port
   lsof -i :5000
   # Change ports in docker-compose.yml
   ```

2. **Memory issues with ML models:**
   ```bash
   # Increase Docker memory limits
   docker-compose up --scale backend=1 --memory=2g
   ```

3. **Database connection errors:**
   ```bash
   # Check database logs
   docker-compose logs db
   # Verify connection string
   ```

### Log Access

```bash
# Application logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Database logs
docker-compose logs -f db

# All services
docker-compose logs -f
```

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer:** Add nginx/haproxy
2. **Multiple Backend Instances:**
   ```bash
   docker-compose up --scale backend=3
   ```
3. **Database Read Replicas**
4. **Redis Cluster** for caching

### Vertical Scaling

- Increase container memory/CPU limits
- Use larger database instances
- Optimize ML model memory usage

## 📝 Maintenance

### Regular Tasks

1. **Update dependencies** monthly
2. **Retrain ML models** monthly
3. **Database maintenance** (vacuum, analyze)
4. **Log rotation and cleanup**
5. **Security updates**
6. **Backup verification**

### Backup Strategy

```bash
# Database backup
docker-compose exec db pg_dump -U portfolio_user portfolio_optimizer > backup.sql

# Model files backup
tar -czf models_backup.tar.gz backend/models/saved/

# Automated backups (add to cron)
0 2 * * * /path/to/backup-script.sh
```