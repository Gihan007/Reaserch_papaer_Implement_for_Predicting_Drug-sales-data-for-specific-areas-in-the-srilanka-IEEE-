#!/bin/bash

# Drug Sales Prediction System - Setup and Deployment Script
# This script handles complete project setup, testing, and deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        log_success "Python $PYTHON_VERSION detected"
    else
        log_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed."
        exit 1
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        log_error "Git is required but not installed."
        exit 1
    fi

    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker detected - container deployment available"
    else
        log_warning "Docker not detected - container deployment unavailable"
    fi
}

# Setup Python virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi

    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"

    # Upgrade pip
    pip install --upgrade pip
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    # Install main dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Main dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi

    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        log_success "Development dependencies installed"
    else
        log_warning "requirements-dev.txt not found - skipping dev dependencies"
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit run --all-files
        log_success "Pre-commit hooks configured"
    else
        log_warning "pre-commit not available - skipping hook setup"
    fi
}

# Run code quality checks
run_quality_checks() {
    log_info "Running code quality checks..."

    # Format code with Black
    if command -v black &> /dev/null; then
        black .
        log_success "Code formatted with Black"
    fi

    # Sort imports with isort
    if command -v isort &> /dev/null; then
        isort .
        log_success "Imports sorted with isort"
    fi

    # Lint with flake8
    if command -v flake8 &> /dev/null; then
        flake8 . --max-line-length=88 --extend-ignore=E203,W503
        log_success "Linting passed"
    fi

    # Type check with mypy
    if command -v mypy &> /dev/null; then
        mypy src/ --ignore-missing-imports
        log_success "Type checking passed"
    fi
}

# Run tests
run_tests() {
    log_info "Running test suite..."

    if [ -d "tests" ]; then
        # Run pytest with coverage
        if command -v pytest &> /dev/null; then
            pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
            log_success "Tests completed"
        else
            log_warning "pytest not available - running basic tests"
            python -m unittest discover tests/
        fi
    else
        log_warning "No tests directory found"
    fi
}

# Setup data directory
setup_data() {
    log_info "Setting up data directory..."

    mkdir -p data/
    mkdir -p models_/
    mkdir -p logs/

    # Check for data files
    DATA_FILES=("C1.csv" "C2.csv" "C3.csv" "C4.csv" "C5.csv" "C6.csv" "C7.csv" "C8.csv")
    MISSING_FILES=()

    for file in "${DATA_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            MISSING_FILES+=("$file")
        fi
    done

    if [ ${#MISSING_FILES[@]} -eq 0 ]; then
        log_success "All data files present"
    else
        log_warning "Missing data files: ${MISSING_FILES[*]}"
        log_info "Please ensure all CSV files are in the project root"
    fi
}

# Build Docker image
build_docker() {
    log_info "Building Docker image..."

    if command -v docker &> /dev/null; then
        docker build -t drug-sales-prediction:latest .
        log_success "Docker image built successfully"
    else
        log_error "Docker not available"
        exit 1
    fi
}

# Deploy locally
deploy_local() {
    log_info "Starting local deployment..."

    # Check if port 5000 is available
    if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
        log_warning "Port 5000 is already in use"
        read -p "Kill existing process? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:5000 | xargs kill -9
            log_info "Killed existing process on port 5000"
        fi
    fi

    # Start the application
    source venv/bin/activate
    python app.py &
    APP_PID=$!

    log_success "Application started with PID: $APP_PID"
    log_info "Application running at http://localhost:5000"

    # Wait for user input
    read -p "Press Enter to stop the application..."
    kill $APP_PID
    log_success "Application stopped"
}

# Deploy to cloud (Heroku example)
deploy_cloud() {
    log_info "Preparing for cloud deployment..."

    # Check for Heroku CLI
    if ! command -v heroku &> /dev/null; then
        log_error "Heroku CLI not installed. Install from: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi

    # Create Procfile if it doesn't exist
    if [ ! -f "Procfile" ]; then
        echo "web: python app.py" > Procfile
        log_success "Procfile created"
    fi

    # Create runtime.txt if it doesn't exist
    if [ ! -f "runtime.txt" ]; then
        echo "python-3.9.7" > runtime.txt
        log_success "runtime.txt created"
    fi

    log_info "Ready for Heroku deployment. Run:"
    echo "  heroku create your-app-name"
    echo "  git push heroku main"
}

# Main menu
show_menu() {
    echo
    echo "========================================"
    echo " Drug Sales Prediction System Setup"
    echo "========================================"
    echo "1. Complete Setup (Recommended)"
    echo "2. Install Dependencies Only"
    echo "3. Run Tests"
    echo "4. Code Quality Checks"
    echo "5. Build Docker Image"
    echo "6. Deploy Locally"
    echo "7. Prepare Cloud Deployment"
    echo "8. Exit"
    echo "========================================"
    read -p "Choose an option (1-8): " choice
}

# Main setup function
complete_setup() {
    log_info "Starting complete project setup..."

    check_requirements
    setup_venv
    install_dependencies
    setup_precommit
    run_quality_checks
    run_tests
    setup_data

    log_success "Complete setup finished!"
    log_info "You can now:"
    echo "  - Run 'python app.py' to start the web application"
    echo "  - Run 'pytest tests/' to run tests"
    echo "  - Run './setup.sh' again for other options"
}

# Main script logic
main() {
    cd "$(dirname "$0")"  # Change to script directory

    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            case $choice in
                1)
                    complete_setup
                    ;;
                2)
                    check_requirements
                    setup_venv
                    install_dependencies
                    ;;
                3)
                    source venv/bin/activate 2>/dev/null || true
                    run_tests
                    ;;
                4)
                    source venv/bin/activate 2>/dev/null || true
                    run_quality_checks
                    ;;
                5)
                    build_docker
                    ;;
                6)
                    deploy_local
                    ;;
                7)
                    deploy_cloud
                    ;;
                8)
                    log_info "Goodbye!"
                    exit 0
                    ;;
                *)
                    log_error "Invalid option. Please choose 1-8."
                    ;;
            esac
            echo
        done
    else
        # Command line mode
        case $1 in
            "setup")
                complete_setup
                ;;
            "install")
                check_requirements
                setup_venv
                install_dependencies
                ;;
            "test")
                source venv/bin/activate 2>/dev/null || true
                run_tests
                ;;
            "quality")
                source venv/bin/activate 2>/dev/null || true
                run_quality_checks
                ;;
            "docker")
                build_docker
                ;;
            "local")
                deploy_local
                ;;
            "cloud")
                deploy_cloud
                ;;
            *)
                log_error "Invalid argument. Use: setup, install, test, quality, docker, local, cloud"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"