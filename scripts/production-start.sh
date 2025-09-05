#!/bin/bash

# Production startup script for Portfolio Optimizer
# This script handles all production deployment tasks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="portfolio-optimizer"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
LOG_DIR="/var/log/portfolio-optimizer"
PID_DIR="/var/run/portfolio-optimizer"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$PID_DIR"
    sudo mkdir -p "/var/lib/portfolio-optimizer"
    
    # Set ownership and permissions
    sudo chown -R $(whoami):$(whoami) "$LOG_DIR"
    sudo chown -R $(whoami):$(whoami) "$PID_DIR"
    
    chmod 755 "$LOG_DIR"
    chmod 755 "$PID_DIR"
}

# Function to validate environment
validate_environment() {
    print_status "Validating production environment..."
    
    # Check required commands
    local required_commands=("python3" "pip" "node" "npm" "docker" "docker-compose")
    
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            print_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check environment variables
    if [ ! -f ".env" ]; then
        print_error ".env file not found. Copy .env.production.example to .env and configure it."
        exit 1
    fi
    
    # Source environment variables
    source .env
    
    # Check required environment variables
    local required_vars=("SECRET_KEY" "JWT_SECRET_KEY" "DATABASE_URL" "FLASK_ENV")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_error "Required environment variable not set: $var"
            exit 1
        fi
    done
    
    # Validate FLASK_ENV
    if [ "$FLASK_ENV" != "production" ]; then
        print_warning "FLASK_ENV is not set to 'production'. Current value: $FLASK_ENV"
    fi
    
    print_status "Environment validation passed"
}

# Function to run deployment validation
run_deployment_validation() {
    print_status "Running deployment validation checks..."
    
    cd "$BACKEND_DIR"
    
    # Run validation script
    python3 -c "
from utils.deployment_helpers import DeploymentValidator
validator = DeploymentValidator()
results = validator.run_all_validations()

if not results['overall_passed']:
    print('Deployment validation failed:')
    for category, data in results.items():
        if category != 'overall_passed' and category != 'timestamp':
            if 'errors' in data and data['errors']:
                for error in data['errors']:
                    print(f'ERROR: {error}')
    exit(1)
else:
    print('All deployment validations passed!')
"
    
    cd ..
    print_status "Deployment validation completed"
}

# Function to backup database
backup_database() {
    print_status "Creating database backup..."
    
    cd "$BACKEND_DIR"
    
    python3 -c "
from utils.deployment_helpers import DatabaseMigrationManager
import os

db_url = os.getenv('DATABASE_URL')
if db_url:
    migrator = DatabaseMigrationManager(db_url)
    try:
        backup_file = migrator.create_backup()
        print(f'Database backup created: {backup_file}')
    except Exception as e:
        print(f'Backup failed: {e}')
        exit(1)
else:
    print('No DATABASE_URL configured, skipping backup')
"
    
    cd ..
}

# Function to build and deploy with Docker
deploy_with_docker() {
    print_status "Deploying with Docker Compose..."
    
    # Build images
    print_status "Building Docker images..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
    
    # Start services
    print_status "Starting production services..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    
    # Wait for services
    wait_for_service "Backend API" "http://localhost:5000/health"
    wait_for_service "Frontend" "http://localhost:80/health" || print_warning "Frontend health check failed (may be normal)"
}

# Function to deploy without Docker (manual)
deploy_manual() {
    print_status "Deploying manually (without Docker)..."
    
    # Install backend dependencies
    print_status "Installing backend dependencies..."
    cd "$BACKEND_DIR"
    pip install -r requirements.txt
    
    # Run database migrations
    print_status "Running database migrations..."
    python3 -c "from models.database import init_database; init_database()"
    
    # Start backend with Gunicorn
    print_status "Starting backend with Gunicorn..."
    gunicorn \
        --bind 0.0.0.0:5000 \
        --workers ${GUNICORN_WORKERS:-4} \
        --threads ${GUNICORN_THREADS:-2} \
        --timeout ${GUNICORN_TIMEOUT:-120} \
        --keepalive ${GUNICORN_KEEPALIVE:-5} \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --preload \
        --log-level info \
        --log-file "$LOG_DIR/gunicorn.log" \
        --access-logfile "$LOG_DIR/gunicorn-access.log" \
        --pid "$PID_DIR/gunicorn.pid" \
        --daemon \
        app:app
    
    cd ..
    
    # Build and serve frontend
    print_status "Building and serving frontend..."
    cd "$FRONTEND_DIR"
    npm ci --production
    npm run build
    
    # Serve with a simple HTTP server (for production, use Nginx)
    print_warning "Serving frontend with simple HTTP server. For production, configure Nginx."
    cd dist
    python3 -m http.server 8080 &
    echo $! > "$PID_DIR/frontend.pid"
    
    cd ../..
    
    # Wait for services
    wait_for_service "Backend API" "http://localhost:5000/health"
    wait_for_service "Frontend" "http://localhost:8080/"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    print_status "Running post-deployment tests..."
    
    cd "$BACKEND_DIR"
    
    # Run basic load test
    python3 -c "
from utils.deployment_helpers import LoadTestRunner
import os

base_url = 'http://localhost:5000'
load_tester = LoadTestRunner(base_url)
results = load_tester.run_basic_load_test(concurrent_users=5, duration_seconds=30)

print(f'Load test results:')
print(f'Total requests: {results[\"total_requests\"]}')
print(f'Successful requests: {results[\"successful_requests\"]}')
print(f'Failed requests: {results[\"failed_requests\"]}')
print(f'Average response time: {results[\"average_response_time\"]:.2f}ms')
print(f'Requests per second: {results[\"requests_per_second\"]:.2f}')

if results['failed_requests'] > results['total_requests'] * 0.1:  # More than 10% failures
    print('WARNING: High failure rate in load test!')
    exit(1)
"
    
    cd ..
    print_status "Post-deployment tests completed"
}

# Function to display deployment summary
show_deployment_summary() {
    print_status "Deployment Summary"
    echo "=================="
    
    echo "Services Status:"
    
    # Check backend
    if curl -s "http://localhost:5000/health" >/dev/null 2>&1; then
        echo "✓ Backend API: Running (http://localhost:5000)"
    else
        echo "✗ Backend API: Not responding"
    fi
    
    # Check frontend
    if curl -s "http://localhost:80/" >/dev/null 2>&1 || curl -s "http://localhost:8080/" >/dev/null 2>&1; then
        echo "✓ Frontend: Running"
    else
        echo "✗ Frontend: Not responding"
    fi
    
    # Check database
    cd "$BACKEND_DIR"
    python3 -c "
import os
from sqlalchemy import create_engine

try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('✓ Database: Connected')
except Exception as e:
    print(f'✗ Database: Connection failed - {e}')
" 2>/dev/null
    cd ..
    
    echo ""
    echo "Log Files:"
    echo "- Application logs: $LOG_DIR/"
    echo "- PID files: $PID_DIR/"
    echo ""
    echo "Useful Commands:"
    echo "- View logs: tail -f $LOG_DIR/*.log"
    echo "- Check processes: ps aux | grep -E '(gunicorn|python.*app)'"
    echo "- Stop services: ./scripts/production-stop.sh"
    echo "- Health check: curl http://localhost:5000/health/detailed"
}

# Function to handle cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Deployment failed! Check logs for details."
        
        # Show recent error logs if available
        if [ -f "$LOG_DIR/gunicorn.log" ]; then
            print_error "Recent backend logs:"
            tail -20 "$LOG_DIR/gunicorn.log"
        fi
    fi
}

# Main deployment function
main() {
    print_status "Starting Portfolio Optimizer Production Deployment"
    
    # Set up cleanup on exit
    trap cleanup EXIT
    
    # Check deployment mode
    USE_DOCKER=${USE_DOCKER:-"true"}
    SKIP_VALIDATION=${SKIP_VALIDATION:-"false"}
    SKIP_BACKUP=${SKIP_BACKUP:-"false"}
    SKIP_TESTS=${SKIP_TESTS:-"false"}
    
    # Create directories
    create_directories
    
    # Validate environment
    validate_environment
    
    # Run deployment validation
    if [ "$SKIP_VALIDATION" != "true" ]; then
        run_deployment_validation
    else
        print_warning "Skipping deployment validation"
    fi
    
    # Backup database
    if [ "$SKIP_BACKUP" != "true" ]; then
        backup_database
    else
        print_warning "Skipping database backup"
    fi
    
    # Deploy application
    if [ "$USE_DOCKER" = "true" ]; then
        deploy_with_docker
    else
        deploy_manual
    fi
    
    # Run post-deployment tests
    if [ "$SKIP_TESTS" != "true" ]; then
        run_post_deployment_tests
    else
        print_warning "Skipping post-deployment tests"
    fi
    
    # Show summary
    show_deployment_summary
    
    print_status "Production deployment completed successfully!"
}

# Show help
show_help() {
    echo "Portfolio Optimizer Production Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --no-docker             Deploy without Docker (manual deployment)"
    echo "  --skip-validation       Skip deployment validation checks"
    echo "  --skip-backup          Skip database backup"
    echo "  --skip-tests           Skip post-deployment tests"
    echo ""
    echo "Environment Variables:"
    echo "  USE_DOCKER             Use Docker deployment (default: true)"
    echo "  SKIP_VALIDATION        Skip validation (default: false)"
    echo "  SKIP_BACKUP            Skip backup (default: false)"
    echo "  SKIP_TESTS             Skip tests (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full Docker deployment"
    echo "  $0 --no-docker          # Manual deployment"
    echo "  $0 --skip-validation    # Skip validation checks"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-docker)
            USE_DOCKER="false"
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION="true"
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main