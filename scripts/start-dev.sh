#!/bin/bash

# Development startup script for AI-Powered Portfolio Optimizer

echo "🚀 Starting AI-Powered Portfolio Optimizer Development Environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Set Docker Compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

print_status "Using Docker Compose: $DOCKER_COMPOSE"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p backend/logs
mkdir -p backend/models/saved
mkdir -p ssl

# Check if .env file exists in backend
if [ ! -f "backend/.env" ]; then
    print_warning ".env file not found in backend directory"
    if [ -f "backend/.env.example" ]; then
        print_status "Copying .env.example to .env..."
        cp backend/.env.example backend/.env
        print_warning "Please edit backend/.env with your configuration before continuing"
        print_warning "Press Enter to continue after editing .env file..."
        read -p ""
    else
        print_error "No .env.example file found. Please create backend/.env manually."
        exit 1
    fi
fi

# Start services with Docker Compose
print_status "Starting development services..."

if [ "$1" = "--build" ]; then
    print_status "Building images from scratch..."
    $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml build --no-cache
fi

# Start the services
$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml up -d

# Check if services are running
print_status "Checking service health..."
sleep 10

# Check backend health
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    print_success "Backend API is healthy"
else
    print_warning "Backend API health check failed"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is accessible"
else
    print_warning "Frontend health check failed"
fi

# Check database
if $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml exec -T db pg_isready -U portfolio_user -d portfolio_optimizer_dev > /dev/null 2>&1; then
    print_success "Database is ready"
else
    print_warning "Database health check failed"
fi

# Check Redis
if $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis health check failed"
fi

echo ""
print_success "Development environment started successfully!"
echo ""
print_status "Services available at:"
echo "  • Frontend:    http://localhost:3000"
echo "  • Backend API: http://localhost:5000"
echo "  • Health:      http://localhost:5000/health"
echo ""
print_status "To view logs:"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml logs -f [service]"
echo ""
print_status "To stop services:"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml down"
echo ""
print_status "To train ML models (after services are running):"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.dev.yml exec backend python quick_train.py"
echo ""
print_warning "Note: Initial model training may take 5-10 minutes for optimal performance"