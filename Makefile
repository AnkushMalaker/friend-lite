# ========================================
# Friend-Lite Configuration Management
# ========================================
# This Makefile generates all service-specific configuration files from the master .env
# Configuration is data-driven via config.yaml - no hardcoded variable names!

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env | grep -v '^\s*$$' | grep -v '^\s*\#')
endif

# Load configuration definitions
include config.env
# Export all variables from config.env
export $(shell sed 's/=.*//' config.env | grep -v '^\s*$$' | grep -v '^\s*\#')

.PHONY: help config config-docker config-k8s config-all clean deploy deploy-docker deploy-k8s deploy-k8s-full deploy-infrastructure deploy-apps check-infrastructure check-apps build-backend up-backend down-backend

help: ## Show this help message
	@echo "Friend-Lite Configuration Management"
	@echo "===================================="
	@echo
	@echo "📝 Configuration targets:"
	@echo "  config               Generate all configuration files (Docker + K8s)"
	@echo "  config-docker        Generate Docker Compose .env files"
	@echo "  config-k8s           Generate Kubernetes files (Skaffold env + ConfigMap/Secret)"
	@echo
	@echo "🚀 Simple deployment workflow:"
	@echo "  1. Edit config.env with your settings"
	@echo "  2. Run: make config-k8s"
	@echo "  3. Run: skaffold run -p advanced-backend --default-repo=your-registry"
	@echo
	@echo "🧹 Utility targets:"
	@echo "  clean                Clean up generated configuration files"
	@echo
	@echo "💡 Key improvement:"
	@echo "  - No more manual setValueTemplates in skaffold.yaml!"
	@echo "  - All environment variables automatically available via ConfigMap/Secret"
	@echo "  - Add new env vars to config.env, run 'make config', done!"
	@echo
	@echo "Current configuration:"
	@echo "  DOMAIN: $(DOMAIN)"
	@echo "  DEPLOYMENT_MODE: $(DEPLOYMENT_MODE)"
	@echo "  INFRASTRUCTURE_NAMESPACE: $(INFRASTRUCTURE_NAMESPACE)"
	@echo "  APPLICATION_NAMESPACE: $(APPLICATION_NAMESPACE)"

config: config-all ## Generate all configuration files

config-docker: ## Generate Docker Compose configuration files
	@echo "🐳 Generating Docker Compose configuration files..."
	@python3 scripts/generate-docker-configs.py
	@echo "✅ Docker Compose configuration files generated"

config-k8s: ## Generate Kubernetes configuration files (Skaffold env + ConfigMap/Secret)
	@echo "☸️  Generating Kubernetes configuration files..."
	@python3 scripts/generate-docker-configs.py
	@python3 scripts/generate-k8s-configs.py
	@echo "📦 Applying ConfigMap and Secret to Kubernetes..."
	@kubectl apply -f k8s-manifests/configmap.yaml -n $(APPLICATION_NAMESPACE) 2>/dev/null || echo "⚠️  ConfigMap not applied (cluster not available?)"
	@kubectl apply -f k8s-manifests/secrets.yaml -n $(APPLICATION_NAMESPACE) 2>/dev/null || echo "⚠️  Secret not applied (cluster not available?)"
	@echo "📦 Copying ConfigMap and Secret to speech namespace..."
	@kubectl get configmap friend-lite-config -n $(APPLICATION_NAMESPACE) -o yaml | \
		sed -e '/namespace:/d' -e '/resourceVersion:/d' -e '/uid:/d' -e '/creationTimestamp:/d' | \
		kubectl apply -n speech -f - 2>/dev/null || echo "⚠️  ConfigMap not copied to speech namespace"
	@kubectl get secret friend-lite-secrets -n $(APPLICATION_NAMESPACE) -o yaml | \
		sed -e '/namespace:/d' -e '/resourceVersion:/d' -e '/uid:/d' -e '/creationTimestamp:/d' | \
		kubectl apply -n speech -f - 2>/dev/null || echo "⚠️  Secret not copied to speech namespace"
	@echo "✅ Kubernetes configuration files generated"

config-all: config-docker config-k8s ## Generate all configuration files
	@echo "✅ All configuration files generated"

clean-config: ## Clean up generated configuration files
	@echo "🧹 Cleaning up generated configuration files..."
	@rm -f backends/advanced/.env
	@rm -f extras/speaker-recognition/.env
	@rm -f extras/openmemory-mcp/.env
	@rm -f extras/asr-services/.env
	@rm -f extras/havpe-relay/.env
	@rm -f backends/simple/.env
	@rm -f backends/other-backends/omi-webhook-compatible/.env
	@rm -f skaffold.env
	@rm -f backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "✅ Generated files cleaned"

# ========================================
# DEPLOYMENT TARGETS
# ========================================

deploy: ## Deploy using configured deployment mode
	@echo "🚀 Deploying using $(DEPLOYMENT_MODE) mode..."
ifeq ($(DEPLOYMENT_MODE),docker-compose)
	@$(MAKE) deploy-docker
else ifeq ($(DEPLOYMENT_MODE),kubernetes)
	@$(MAKE) deploy-k8s
else
	@echo "❌ Unknown deployment mode: $(DEPLOYMENT_MODE)"
	@exit 1
endif

deploy-docker: config-docker ## Deploy using Docker Compose
	@echo "🐳 Deploying with Docker Compose..."
	@cd backends/advanced && docker-compose up -d
	@echo "✅ Docker Compose deployment completed"

deploy-k8s: config-skaffold ## Deploy to Kubernetes using Skaffold
	@echo "☸️  Deploying to Kubernetes with Skaffold..."
	@set -a; source skaffold.env; set +a; skaffold run --profile=advanced-backend --default-repo=anubis:32000
	@echo "✅ Kubernetes deployment completed"

deploy-k8s-full: deploy-infrastructure deploy-apps ## Deploy infrastructure + applications to Kubernetes
	@echo "✅ Full Kubernetes deployment completed"

deploy-infrastructure: ## Deploy infrastructure services to Kubernetes
	@echo "🏗️  Deploying infrastructure services..."
	@kubectl apply -f k8s-manifests/
	@echo "✅ Infrastructure deployment completed"

deploy-apps: config-skaffold ## Deploy application services to Kubernetes
	@echo "📱 Deploying application services..."
	@set -a; source skaffold.env; set +a; skaffold run --profile=advanced-backend --default-repo=anubis:32000
	@echo "✅ Application deployment completed"

# ========================================
# UTILITY TARGETS
# ========================================

check-infrastructure: ## Check if infrastructure services are running
	@echo "🔍 Checking infrastructure services..."
	@kubectl get pods -n $(INFRASTRUCTURE_NAMESPACE) || echo "❌ Infrastructure namespace not found"
	@kubectl get services -n $(INFRASTRUCTURE_NAMESPACE) || echo "❌ Infrastructure services not found"

check-apps: ## Check if application services are running
	@echo "🔍 Checking application services..."
	@kubectl get pods -n $(APPLICATION_NAMESPACE) || echo "❌ Application namespace not found"
	@kubectl get services -n $(APPLICATION_NAMESPACE) || echo "❌ Application services not found"

# ========================================
# DEVELOPMENT TARGETS
# ========================================

build-backend: ## Build backend Docker image
	@echo "🔨 Building backend Docker image..."
	@cd backends/advanced && docker build -t advanced-backend:latest .

up-backend: config-docker ## Start backend services
	@echo "🚀 Starting backend services..."
	@cd backends/advanced && docker-compose up -d

down-backend: ## Stop backend services
	@echo "🛑 Stopping backend services..."
	@cd backends/advanced && docker-compose down
