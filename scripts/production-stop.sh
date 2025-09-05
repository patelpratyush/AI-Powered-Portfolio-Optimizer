#!/bin/bash

# Production stop script for Portfolio Optimizer
# This script gracefully stops all production services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="portfolio-optimizer"
PID_DIR="/var/run/portfolio-optimizer"
LOG_DIR="/var/log/portfolio-optimizer"

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

# Function to stop Docker services
stop_docker_services() {
    print_status "Stopping Docker services..."
    
    if command -v docker-compose >/dev/null 2>&1; then
        if [ -f "docker-compose.yml" ]; then
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
            print_status "Docker services stopped"
        else
            print_warning "docker-compose.yml not found"
        fi
    else
        print_warning "docker-compose not available"
    fi
}

# Function to stop manual services
stop_manual_services() {
    print_status "Stopping manual services..."
    
    # Stop Gunicorn backend
    if [ -f "$PID_DIR/gunicorn.pid" ]; then
        local pid=$(cat "$PID_DIR/gunicorn.pid")
        if ps -p $pid >/dev/null 2>&1; then
            print_status "Stopping Gunicorn (PID: $pid)..."
            kill -TERM $pid
            
            # Wait for graceful shutdown
            local count=0
            while ps -p $pid >/dev/null 2>&1 && [ $count -lt 30 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if ps -p $pid >/dev/null 2>&1; then
                print_warning "Forcing Gunicorn shutdown..."
                kill -KILL $pid
            fi
            
            rm -f "$PID_DIR/gunicorn.pid"
            print_status "Gunicorn stopped"
        else
            print_warning "Gunicorn process not running (PID: $pid)"
            rm -f "$PID_DIR/gunicorn.pid"
        fi
    else
        print_warning "No Gunicorn PID file found"
    fi
    
    # Stop frontend server
    if [ -f "$PID_DIR/frontend.pid" ]; then
        local pid=$(cat "$PID_DIR/frontend.pid")
        if ps -p $pid >/dev/null 2>&1; then
            print_status "Stopping frontend server (PID: $pid)..."
            kill -TERM $pid
            rm -f "$PID_DIR/frontend.pid"
            print_status "Frontend server stopped"
        else
            print_warning "Frontend process not running (PID: $pid)"
            rm -f "$PID_DIR/frontend.pid"
        fi
    else
        print_warning "No frontend PID file found"
    fi
    
    # Find and stop any remaining Python processes
    local python_pids=$(pgrep -f "python.*app.py\|gunicorn.*app:app" || true)
    if [ -n "$python_pids" ]; then
        print_status "Stopping remaining Python processes..."
        echo "$python_pids" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        echo "$python_pids" | xargs kill -KILL 2>/dev/null || true
    fi
    
    # Find and stop any remaining HTTP servers
    local http_pids=$(pgrep -f "python.*http.server" || true)
    if [ -n "$http_pids" ]; then
        print_status "Stopping HTTP servers..."
        echo "$http_pids" | xargs kill -TERM 2>/dev/null || true
    fi
}

# Function to cleanup resources
cleanup_resources() {
    print_status "Cleaning up resources..."
    
    # Remove stale PID files
    if [ -d "$PID_DIR" ]; then
        find "$PID_DIR" -name "*.pid" -type f -delete 2>/dev/null || true
    fi
    
    # Cleanup temporary files
    find /tmp -name "*portfolio-optimizer*" -type f -mtime +1 -delete 2>/dev/null || true
    
    # Clear any stuck Redis connections (if Redis is available)
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli ping >/dev/null 2>&1 && {
            print_status "Clearing Redis connections..."
            redis-cli --scan --pattern "*portfolio_optimizer*" | xargs -r redis-cli del 2>/dev/null || true
        }
    fi
    
    print_status "Cleanup completed"
}

# Function to check if any services are still running
check_remaining_processes() {
    print_status "Checking for remaining processes..."
    
    local remaining_processes=$(ps aux | grep -E "(gunicorn|python.*app\.py|python.*http\.server)" | grep -v grep | grep -E "(portfolio|optimizer)" || true)
    
    if [ -n "$remaining_processes" ]; then
        print_warning "Some processes may still be running:"
        echo "$remaining_processes"
        echo ""
        print_warning "You may need to manually stop these processes with:"
        print_warning "kill -9 <PID>"
        return 1
    else
        print_status "No remaining processes found"
        return 0
    fi
}

# Function to show stop summary
show_stop_summary() {
    print_status "Stop Summary"
    echo "============"
    
    # Check if Docker services are stopped
    if command -v docker-compose >/dev/null 2>&1 && [ -f "docker-compose.yml" ]; then
        local running_containers=$(docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps -q 2>/dev/null | wc -l)
        if [ "$running_containers" -eq 0 ]; then
            echo "✓ Docker services: Stopped"
        else
            echo "✗ Docker services: Some containers still running"
        fi
    fi
    
    # Check manual services
    if [ ! -f "$PID_DIR/gunicorn.pid" ] || ! ps -p $(cat "$PID_DIR/gunicorn.pid" 2>/dev/null) >/dev/null 2>&1; then
        echo "✓ Backend API: Stopped"
    else
        echo "✗ Backend API: Still running"
    fi
    
    if [ ! -f "$PID_DIR/frontend.pid" ] || ! ps -p $(cat "$PID_DIR/frontend.pid" 2>/dev/null) >/dev/null 2>&1; then
        echo "✓ Frontend: Stopped"
    else
        echo "✗ Frontend: Still running"
    fi
    
    echo ""
    echo "Log files preserved in: $LOG_DIR"
    echo ""
    echo "To restart services, run: ./scripts/production-start.sh"
}

# Function to force stop all services
force_stop() {
    print_warning "Force stopping all services..."
    
    # Force stop Docker
    if command -v docker >/dev/null 2>&1; then
        docker stop $(docker ps -q --filter "label=com.docker.compose.project=$PROJECT_NAME") 2>/dev/null || true
        docker kill $(docker ps -q --filter "label=com.docker.compose.project=$PROJECT_NAME") 2>/dev/null || true
    fi
    
    # Force kill all related processes
    pkill -9 -f "gunicorn" || true
    pkill -9 -f "python.*app\.py" || true
    pkill -9 -f "python.*http\.server" || true
    pkill -9 -f "portfolio.optimizer" || true
    
    # Clean up PID files
    rm -rf "$PID_DIR"/*.pid 2>/dev/null || true
    
    print_status "Force stop completed"
}

# Main stop function
main() {
    local FORCE_STOP=${1:-"false"}
    
    if [ "$FORCE_STOP" = "--force" ] || [ "$FORCE_STOP" = "-f" ]; then
        force_stop
    else
        print_status "Starting graceful shutdown of Portfolio Optimizer services..."
        
        # Try to determine deployment mode
        if [ -f "docker-compose.yml" ] && command -v docker-compose >/dev/null 2>&1; then
            local running_containers=$(docker-compose -f docker-compose.yml ps -q 2>/dev/null | wc -l)
            if [ "$running_containers" -gt 0 ]; then
                stop_docker_services
            else
                stop_manual_services
            fi
        else
            stop_manual_services
        fi
        
        # Cleanup resources
        cleanup_resources
        
        # Check for remaining processes
        if ! check_remaining_processes; then
            print_warning "Some processes may still be running. Use --force to force stop."
        fi
    fi
    
    # Show summary
    show_stop_summary
    
    print_status "Portfolio Optimizer services stopped"
}

# Show help
show_help() {
    echo "Portfolio Optimizer Production Stop Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -f, --force             Force stop all services (kill -9)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Graceful shutdown"
    echo "  $0 --force              # Force stop all services"
}

# Parse command line arguments
case ${1:-} in
    -h|--help)
        show_help
        exit 0
        ;;
    -f|--force)
        main "--force"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac