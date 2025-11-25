# Drug Sales Prediction System - Deployment Guide

## 🌐 International Deployment Guide

This guide provides comprehensive instructions for deploying the Drug Sales Prediction System at international standards, suitable for research institutions, healthcare organizations, and cloud platforms.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Setup](#production-setup)
- [Monitoring & Observability](#monitoring--observability)
- [Performance Optimization](#performance-optimization)
- [Security Considerations](#security-considerations)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### One-Command Setup
```bash
# Clone and setup everything
git clone https://github.com/Gihan007/drug-sales-prediction.git
cd drug-sales-prediction
chmod +x setup.sh && ./setup.sh

# Start the application
docker-compose up -d
```

### Verify Installation
```bash
# Check if services are running
docker-compose ps

# Test the application
curl http://localhost/health

# Run benchmarks
python benchmark_performance.py
```

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11 with WSL2
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection

### Software Dependencies
```bash
# Python 3.8+
python --version

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version

# Optional: CUDA for GPU acceleration
nvidia-smi  # Check GPU availability
```

### Cloud Platform Accounts (Optional)
- **AWS**: EC2, ECS, S3, CloudWatch
- **Google Cloud**: GCE, GKE, Cloud Storage, Cloud Monitoring
- **Azure**: VMs, AKS, Blob Storage, Application Insights
- **Heroku**: For simple deployment

---

## 💻 Local Development

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Gihan007/drug-sales-prediction.git
cd drug-sales-prediction

# Run setup script
./setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Data Preparation
```bash
# Place data files in project root
# C1.csv, C2.csv, ..., C8.csv

# Verify data integrity
python -c "import pandas as pd; print(pd.read_csv('C1.csv').head())"
```

### 3. Model Training
```bash
# Train all models
python -c "from src.pipeline.train_all_models import train_all; train_all()"

# Or train specific model
python -c "from src.models.transformer_model import train_transformer; train_transformer('C1')"
```

### 4. Start Development Server
```bash
# Flask development server
python app.py

# Or with Docker
docker-compose --profile dev up

# Access application
open http://localhost:5000
```

---

## 🐳 Docker Deployment

### Single Container Deployment
```bash
# Build and run
docker build -t drug-sales-prediction .
docker run -p 5000:5000 -v $(pwd)/data:/app/data:ro drug-sales-prediction

# Or use docker-compose
docker-compose up -d app
```

### Full Stack Deployment
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale services
docker-compose up -d --scale app=3
```

### Development with Docker
```bash
# Start development environment
docker-compose --profile dev up

# Access Jupyter Lab
open http://localhost:8888
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (t3.medium or larger)
# AMI: Ubuntu 20.04 LTS
# Security Group: Allow 22, 80, 443, 5000

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

#### 2. Deploy Application
```bash
# Clone and deploy
git clone https://github.com/Gihan007/drug-sales-prediction.git
cd drug-sales-prediction
docker-compose up -d

# Setup SSL with Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

#### 3. Load Balancing (Optional)
```bash
# Install nginx
sudo apt install nginx

# Configure nginx as reverse proxy
sudo cp nginx.conf /etc/nginx/sites-available/drug-prediction
sudo ln -s /etc/nginx/sites-available/drug-prediction /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Heroku Deployment
```bash
# Install Heroku CLI
npm install -g heroku

# Login and create app
heroku login
heroku create your-app-name

# Deploy
git push heroku main

# Scale dynos
heroku ps:scale web=1
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/drug-prediction
gcloud run deploy --image gcr.io/PROJECT-ID/drug-prediction --platform managed
```

---

## 🏭 Production Setup

### 1. Environment Configuration
```bash
# Create production environment file
cp .env.example .env.production

# Edit with production values
nano .env.production
```

### 2. Database Setup (Optional)
```bash
# For PostgreSQL
docker-compose up -d postgres

# Run migrations (if implemented)
python -c "from src.database.migrations import run_migrations; run_migrations()"
```

### 3. SSL/TLS Configuration
```bash
# Using certbot
sudo certbot --nginx -d your-domain.com

# Or manual certificate installation
sudo cp fullchain.pem /etc/ssl/certs/
sudo cp privkey.pem /etc/ssl/private/
```

### 4. Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# AWS Security Groups
# Allow: 22 (SSH), 80 (HTTP), 443 (HTTPS), 5000 (App)
```

### 5. Process Management
```bash
# Using systemd
sudo cp deployment/drug-prediction.service /etc/systemd/system/
sudo systemctl enable drug-prediction
sudo systemctl start drug-prediction

# Using supervisor
sudo apt install supervisor
sudo cp deployment/supervisor.conf /etc/supervisor/conf.d/
sudo supervisorctl reread && sudo supervisorctl update
```

---

## 📊 Monitoring & Observability

### Application Monitoring
```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

### Log Management
```bash
# View application logs
docker-compose logs -f app

# Setup log rotation
sudo cp deployment/logrotate.conf /etc/logrotate.d/drug-prediction
sudo logrotate /etc/logrotate.d/drug-prediction
```

### Health Checks
```bash
# Application health
curl http://localhost/health

# Database health (if used)
curl http://localhost:5432/health

# System resources
htop
df -h
free -h
```

---

## ⚡ Performance Optimization

### Model Optimization
```bash
# Run performance benchmarks
python benchmark_performance.py

# Optimize models
python -c "from src.optimization.model_optimization import optimize_all_models; optimize_all_models()"

# Quantize models for production
python -c "from src.optimization.quantization import quantize_models; quantize_models()"
```

### Infrastructure Optimization
```bash
# Enable gzip compression
# Configure in nginx.conf

# Setup caching
docker-compose up -d redis

# CDN integration (CloudFlare, AWS CloudFront)
# Configure DNS and CDN settings
```

### Database Optimization (if used)
```bash
# Create indexes
python -c "from src.database.optimize import create_indexes; create_indexes()"

# Query optimization
python -c "from src.database.optimize import optimize_queries; optimize_queries()"
```

---

## 🔒 Security Considerations

### Application Security
```bash
# Update dependencies regularly
pip install --upgrade -r requirements.txt

# Security scanning
docker run --rm -v $(pwd):/app clair-scanner --ip localhost drug-sales-prediction

# Secret management
# Use environment variables for secrets
# Never commit secrets to git
```

### Network Security
```bash
# Setup firewall
sudo ufw enable

# SSL/TLS enforcement
# Redirect HTTP to HTTPS in nginx

# Rate limiting
# Configure in nginx.conf
```

### Data Security
```bash
# Encrypt sensitive data
# Use HTTPS for all communications
# Implement proper authentication
# Regular security audits
```

---

## 💾 Backup & Recovery

### Automated Backups
```bash
# Database backup (if used)
docker exec postgres pg_dump -U prediction_user drug_prediction > backup_$(date +%Y%m%d).sql

# Model backups
tar -czf models_backup_$(date +%Y%m%d).tar.gz models_/

# Configuration backup
cp .env.production .env.production.backup
```

### Backup Scripts
```bash
# Setup automated backups
crontab -e
# Add: 0 2 * * * /path/to/backup-script.sh

# Test backup restoration
./deployment/restore-backup.sh backup_20231201.tar.gz
```

### Disaster Recovery
```bash
# Create recovery plan
cp deployment/disaster-recovery-plan.md .

# Test failover
docker-compose up -d --scale app=2
docker stop app_app_1  # Test failover
```

---

## 🔧 Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker-compose logs app

# Check dependencies
python -c "import flask, torch, pandas; print('Dependencies OK')"

# Check port availability
netstat -tlnp | grep 5000
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats

# Check system resources
free -h
vmstat 1

# Optimize memory usage
python -c "from src.optimization.memory_optimize import optimize_memory; optimize_memory()"
```

#### Model Loading Errors
```bash
# Check model files
ls -la models_/

# Verify model integrity
python -c "from src.models.model_loader import verify_models; verify_models()"

# Rebuild models
python -c "from src.pipeline.rebuild_models import rebuild_all; rebuild_all()"
```

#### Database Connection Issues
```bash
# Test database connection
docker exec postgres pg_isready -U prediction_user -d drug_prediction

# Check connection string
grep DATABASE_URL .env.production

# Reset database
docker-compose down postgres && docker-compose up -d postgres
```

### Performance Issues
```bash
# Run diagnostics
python deployment/diagnostics.py

# Profile application
python -m cProfile -s time app.py

# Check system bottlenecks
iotop
iostat -x 1
```

### Getting Help
```bash
# Check documentation
open docs/index.html

# View logs
tail -f logs/app.log

# Run diagnostics
python deployment/health-check.py

# Community support
# GitHub Issues: https://github.com/Gihan007/drug-sales-prediction/issues
# Email: gihan.lakmal@research.ucsc.cmb.ac.lk
```

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
```bash
# Weekly
# Update dependencies
pip install --upgrade -r requirements.txt

# Run tests
pytest tests/

# Backup data
./deployment/backup.sh

# Monthly
# Security updates
sudo apt update && sudo apt upgrade

# Performance monitoring
python benchmark_performance.py

# Log rotation
sudo logrotate /etc/logrotate.d/drug-prediction
```

### Monitoring Alerts
- Set up alerts for:
  - High CPU/memory usage
  - Application downtime
  - Failed predictions
  - Security vulnerabilities

### Documentation Updates
- Keep deployment documentation current
- Update runbooks for new procedures
- Document troubleshooting procedures

---

## 🎯 Success Metrics

### Technical Metrics
- **Uptime**: >99.9%
- **Response Time**: <500ms for predictions
- **Accuracy**: MAE < 3.0 across categories
- **Throughput**: >50 requests/second

### Business Metrics
- **User Adoption**: Number of active users
- **Prediction Usage**: Daily prediction requests
- **Accuracy Feedback**: User-reported accuracy scores

### Research Metrics
- **Publication Count**: Target 2-3 IEEE papers
- **Citation Count**: Track academic citations
- **Conference Presentations**: Present at 2+ international conferences

---

## 📈 Scaling Strategy

### Vertical Scaling
```bash
# Increase instance size
# AWS: t3.medium → t3.large → t3.xlarge
# GCP: e2-medium → e2-standard-4 → e2-highmem-4

# Add more memory
# Current: 8GB → Target: 32GB for large datasets
```

### Horizontal Scaling
```bash
# Load balancer setup
docker-compose up -d nginx

# Auto-scaling (AWS)
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name drug-prediction-asg \
  --launch-template LaunchTemplateId=lt-1234567890abcdef0 \
  --min-size 1 --max-size 10 --desired-capacity 3

# Kubernetes deployment (advanced)
kubectl apply -f k8s/deployment.yaml
```

### Global Distribution
```bash
# Multi-region deployment
# AWS: us-east-1, eu-west-1, ap-south-1
# GCP: us-central1, europe-west1, asia-south1

# CDN integration
# CloudFlare or AWS CloudFront for static assets
```

---

*This deployment guide ensures your Drug Sales Prediction System meets international standards for research and production deployment. For additional support, please contact the development team.*