# ========================================
# Friend-Lite Configuration Management
# ========================================
# This Makefile generates all service-specific configuration files from the master .env

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env | grep -v '^\s*$$' | grep -v '^\s*\#')
endif

# Default values
DOMAIN ?= localhost
BACKEND_PORT ?= 8000
WEBUI_PORT ?= 5173
SPEAKER_PORT ?= 8085
MONGODB_PORT ?= 27017
QDRANT_PORT ?= 6333
INFRASTRUCTURE_NAMESPACE ?= infrastructure
APPLICATION_NAMESPACE ?= friend-lite
DEPLOYMENT_MODE ?= docker-compose

# Computed values
BACKEND_HOST = ${DOMAIN_PREFIX}.$(DOMAIN)
WEBUI_HOST = ${DOMAIN_PREFIX}.$(DOMAIN)
SPEAKER_HOST = speaker.$(DOMAIN)
BACKEND_URL = http://${DOMAIN_PREFIX}.$(DOMAIN):$(BACKEND_PORT)
WEBUI_URL = http://${DOMAIN_PREFIX}.$(DOMAIN):$(WEBUI_PORT)
SPEAKER_SERVICE_URL = http://$(SPEAKER_HOST):$(SPEAKER_PORT)
CORS_ORIGINS = http://$(DOMAIN):$(WEBUI_PORT),http://$(DOMAIN):3000,http://localhost:$(WEBUI_PORT),http://localhost:3000
VITE_ALLOWED_HOSTS = localhost 127.0.0.1 ${DOMAIN_PREFIX}.$(DOMAIN) ${EXTERNAL_DOMAIN} $(SPEAKER_HOST)
CHAT_LLM_MODEL = $(OPENAI_MODEL)

# MongoDB URIs
MONGODB_URI = mongodb://mongo:$(MONGODB_PORT)
MONGODB_K8S_URI = mongodb://mongodb.$(INFRASTRUCTURE_NAMESPACE).svc.cluster.local:27017/friend

# Qdrant URLs
QDRANT_BASE_URL = qdrant
QDRANT_K8S_URL = qdrant.$(INFRASTRUCTURE_NAMESPACE).svc.cluster.local

.PHONY: help config config-docker config-k8s config-skaffold clean deploy deploy-docker deploy-k8s deploy-k8s-full deploy-infrastructure deploy-apps check-infrastructure check-apps build-backend up-backend down-backend

help: ## Show this help message
	@echo "Friend-Lite Configuration Management"
	@echo "===================================="
	@echo
	@echo "Configuration targets:"
	@echo "  config               Generate all configuration files"
	@echo "  config-docker        Generate Docker Compose configuration files"
	@echo "  config-k8s           Generate Kubernetes Helm values files (for direct Helm deployments, not Skaffold)" 
	@echo "  config-skaffold      Generate Skaffold configuration with environment variables"
	@echo "  clean-config         Clean up generated configuration files"
	@echo
	@echo "Deployment targets:"
	@echo "  deploy               Deploy using the configured deployment mode"
	@echo "  deploy-docker        Deploy using Docker Compose"
	@echo "  deploy-k8s           Deploy to Kubernetes (smart - checks infrastructure first)"
	@echo "  deploy-k8s-full      Deploy everything to Kubernetes (including infrastructure)"
	@echo "  deploy-infrastructure Deploy infrastructure to Kubernetes (MongoDB, Qdrant)"
	@echo "  deploy-apps          Deploy applications to Kubernetes (Backend, WebUI)"
	@echo
	@echo "Utility targets:"
	@echo "  show-config          Show current configuration values"
	@echo "  status               Show deployment status"
	@echo "  check-infrastructure Check if infrastructure is already deployed"
	@echo "  check-apps           Check if applications are already deployed"
	@echo "  help                 Show this help message"
	@echo
	@echo "Legacy Docker targets (deprecated - use config + deploy):"
	@echo "  build-backend        Build Docker Compose services from backends/advanced"
	@echo "  up-backend           Start Docker Compose services from backends/advanced"
	@echo "  down-backend         Stop Docker Compose services from backends/advanced"

config: config-docker config-k8s config-skaffold ## Generate all configuration files

config-docker: ## Generate Docker Compose configuration files
	@echo "🐳 Generating Docker Compose configuration files..."
	@mkdir -p backends/advanced
	@echo "# Auto-generated from master .env - DO NOT EDIT DIRECTLY" > backends/advanced/.env
	@echo "# Edit the root .env file and run 'make config' to regenerate" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Authentication" >> backends/advanced/.env
	@echo "AUTH_SECRET_KEY=$(AUTH_SECRET_KEY)" >> backends/advanced/.env
	@echo "ADMIN_PASSWORD=$(ADMIN_PASSWORD)" >> backends/advanced/.env
	@echo "ADMIN_EMAIL=$(ADMIN_EMAIL)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# LLM Configuration" >> backends/advanced/.env
	@echo "LLM_PROVIDER=$(LLM_PROVIDER)" >> backends/advanced/.env
	@echo "OPENAI_API_KEY=$(OPENAI_API_KEY)" >> backends/advanced/.env
	@echo "OPENAI_BASE_URL=$(OPENAI_BASE_URL)" >> backends/advanced/.env
	@echo "OPENAI_MODEL=$(OPENAI_MODEL)" >> backends/advanced/.env
	@echo "CHAT_LLM_MODEL=$(CHAT_LLM_MODEL)" >> backends/advanced/.env
	@echo "CHAT_TEMPERATURE=$(CHAT_TEMPERATURE)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Speech-to-Text" >> backends/advanced/.env
	@echo "TRANSCRIPTION_PROVIDER=$(TRANSCRIPTION_PROVIDER)" >> backends/advanced/.env
	@echo "DEEPGRAM_API_KEY=$(DEEPGRAM_API_KEY)" >> backends/advanced/.env
	@echo "MISTRAL_API_KEY=$(MISTRAL_API_KEY)" >> backends/advanced/.env
	@echo "MISTRAL_MODEL=$(MISTRAL_MODEL)" >> backends/advanced/.env
	@echo "PARAKEET_ASR_URL=$(PARAKEET_ASR_URL)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Database" >> backends/advanced/.env
	@echo "MONGODB_URI=$(MONGODB_URI)" >> backends/advanced/.env
	@echo "QDRANT_BASE_URL=$(QDRANT_BASE_URL)" >> backends/advanced/.env
	@echo "NEO4J_HOST=$(NEO4J_HOST)" >> backends/advanced/.env
	@echo "NEO4J_USER=$(NEO4J_USER)" >> backends/advanced/.env
	@echo "NEO4J_PASSWORD=$(NEO4J_PASSWORD)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Memory Provider" >> backends/advanced/.env
	@echo "MEMORY_PROVIDER=$(MEMORY_PROVIDER)" >> backends/advanced/.env
	@echo "OPENMEMORY_MCP_URL=$(OPENMEMORY_MCP_URL)" >> backends/advanced/.env
	@echo "OPENMEMORY_CLIENT_NAME=$(OPENMEMORY_CLIENT_NAME)" >> backends/advanced/.env
	@echo "OPENMEMORY_USER_ID=$(OPENMEMORY_USER_ID)" >> backends/advanced/.env
	@echo "OPENMEMORY_TIMEOUT=$(OPENMEMORY_TIMEOUT)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Optional Services" >> backends/advanced/.env
	@echo "SPEAKER_SERVICE_URL=http://speaker-service:$(SPEAKER_PORT)" >> backends/advanced/.env
	@echo "HF_TOKEN=$(HF_TOKEN)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Audio Processing" >> backends/advanced/.env
	@echo "NEW_CONVERSATION_TIMEOUT_MINUTES=$(NEW_CONVERSATION_TIMEOUT_MINUTES)" >> backends/advanced/.env
	@echo "AUDIO_CROPPING_ENABLED=$(AUDIO_CROPPING_ENABLED)" >> backends/advanced/.env
	@echo "MIN_SPEECH_SEGMENT_DURATION=$(MIN_SPEECH_SEGMENT_DURATION)" >> backends/advanced/.env
	@echo "CROPPING_CONTEXT_PADDING=$(CROPPING_CONTEXT_PADDING)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Network Configuration" >> backends/advanced/.env
	@echo "HOST_IP=$(DOMAIN)" >> backends/advanced/.env
	@echo "BACKEND_PUBLIC_PORT=$(BACKEND_PORT)" >> backends/advanced/.env
	@echo "WEBUI_PORT=$(WEBUI_PORT)" >> backends/advanced/.env
	@echo "CORS_ORIGINS=$(CORS_ORIGINS)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Debug" >> backends/advanced/.env
	@echo "DEBUG_DIR=$(DEBUG_DIR)" >> backends/advanced/.env
	@echo "MEM0_TELEMETRY=$(MEM0_TELEMETRY)" >> backends/advanced/.env
	@echo "" >> backends/advanced/.env
	@echo "# Langfuse" >> backends/advanced/.env
	@echo "LANGFUSE_PUBLIC_KEY=$(LANGFUSE_PUBLIC_KEY)" >> backends/advanced/.env
	@echo "LANGFUSE_SECRET_KEY=$(LANGFUSE_SECRET_KEY)" >> backends/advanced/.env
	@echo "LANGFUSE_HOST=$(LANGFUSE_HOST)" >> backends/advanced/.env
	@echo "LANGFUSE_ENABLE_TELEMETRY=$(LANGFUSE_ENABLE_TELEMETRY)" >> backends/advanced/.env
	
	@mkdir -p extras/speaker-recognition
	@echo "# Auto-generated from master .env - DO NOT EDIT DIRECTLY" > extras/speaker-recognition/.env
	@echo "# Edit the root .env file and run 'make config' to regenerate" >> extras/speaker-recognition/.env
	@echo "" >> extras/speaker-recognition/.env
	@echo "HF_TOKEN=$(HF_TOKEN)" >> extras/speaker-recognition/.env
	@echo "COMPUTE_MODE=$(COMPUTE_MODE)" >> extras/speaker-recognition/.env
	@echo "SIMILARITY_THRESHOLD=$(SIMILARITY_THRESHOLD)" >> extras/speaker-recognition/.env
	@echo "" >> extras/speaker-recognition/.env
	@echo "# Service Configuration" >> extras/speaker-recognition/.env
	@echo "SPEAKER_SERVICE_HOST=0.0.0.0" >> extras/speaker-recognition/.env
	@echo "SPEAKER_SERVICE_PORT=$(SPEAKER_PORT)" >> extras/speaker-recognition/.env
	@echo "SPEAKER_SERVICE_URL=http://speaker-service:$(SPEAKER_PORT)" >> extras/speaker-recognition/.env
	@echo "" >> extras/speaker-recognition/.env
	@echo "# React Web UI Configuration" >> extras/speaker-recognition/.env
	@echo "REACT_UI_HOST=$(REACT_UI_HOST)" >> extras/speaker-recognition/.env
	@echo "REACT_UI_PORT=$(REACT_UI_PORT)" >> extras/speaker-recognition/.env
	@echo "REACT_UI_HTTPS=$(REACT_UI_HTTPS)" >> extras/speaker-recognition/.env
	@echo "" >> extras/speaker-recognition/.env
	@echo "# External Services" >> extras/speaker-recognition/.env
	@echo "DEEPGRAM_API_KEY=$(DEEPGRAM_API_KEY)" >> extras/speaker-recognition/.env
	@echo "GROQ_API_KEY=$(GROQ_API_KEY)" >> extras/speaker-recognition/.env
	@echo "" >> extras/speaker-recognition/.env
	@echo "# Additional" >> extras/speaker-recognition/.env
	@echo "WEBUI_CORS_ORIGIN=*" >> extras/speaker-recognition/.env
	@echo "VITE_ALLOWED_HOSTS=$(VITE_ALLOWED_HOSTS)" >> extras/speaker-recognition/.env
	@echo "SPEAKER_SERVICE_TEST_PORT=8086" >> extras/speaker-recognition/.env
	
	@echo "✅ Docker Compose configuration generated"

config-k8s: ## Generate Kubernetes Helm values files
	@echo "☸️  Generating Kubernetes ConfigMap and Secret files..."
	@mkdir -p backends/charts/advanced-backend/templates
	
	# Generate ConfigMap
	@echo "apiVersion: v1" > backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "kind: ConfigMap" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "metadata:" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  name: advanced-backend-env" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  labels:" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "    {{- include \"advanced-backend.labels\" . | nindent 4 }}" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "data:" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  LLM_PROVIDER: \"$(LLM_PROVIDER)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENAI_BASE_URL: \"$(OPENAI_BASE_URL)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENAI_MODEL: \"$(OPENAI_MODEL)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  TRANSCRIPTION_PROVIDER: \"$(TRANSCRIPTION_PROVIDER)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  MEMORY_PROVIDER: \"$(MEMORY_PROVIDER)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENMEMORY_CLIENT_NAME: \"$(OPENMEMORY_CLIENT_NAME)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENMEMORY_USER_ID: \"$(OPENMEMORY_USER_ID)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENMEMORY_TIMEOUT: \"$(OPENMEMORY_TIMEOUT)\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  # Kubernetes-specific values (set by Skaffold)" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  MONGODB_URI: \"{{ .Values.env.MONGODB_URI | default \"mongo\" }}\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  QDRANT_BASE_URL: \"{{ .Values.env.QDRANT_BASE_URL | default \"qdrant.root.svc.cluster.local\" }}\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  CORS_ORIGINS: \"{{ .Values.env.CORS_ORIGINS }}\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	@echo "  OPENMEMORY_MCP_URL: \"{{ .Values.env.OPENMEMORY_MCP_URL }}\"" >> backends/charts/advanced-backend/templates/env-configmap.yaml
	
	# Generate Secret
	@echo "apiVersion: v1" > backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "kind: Secret" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "metadata:" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "  name: advanced-backend-secrets" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "  labels:" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "    {{- include \"advanced-backend.labels\" . | nindent 4 }}" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "type: Opaque" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "data:" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@printf "  AUTH_SECRET_KEY: %s\n" "$$(printf '%s' '$(AUTH_SECRET_KEY)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@printf "  ADMIN_PASSWORD: %s\n" "$$(printf '%s' '$(ADMIN_PASSWORD)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml  
	@printf "  ADMIN_EMAIL: %s\n" "$$(printf '%s' '$(ADMIN_EMAIL)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@printf "  DEEPGRAM_API_KEY: %s\n" "$$(printf '%s' '$(DEEPGRAM_API_KEY)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@printf "  OPENAI_API_KEY: %s\n" "$$(printf '%s' '$(OPENAI_API_KEY)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@printf "  HF_TOKEN: %s\n" "$$(printf '%s' '$(HF_TOKEN)' | base64)" >> backends/charts/advanced-backend/templates/env-secret.yaml
	@echo "✅ Kubernetes ConfigMap and Secret files generated"

config-skaffold: ## Generate Skaffold configuration with environment variables
	@echo "🚀 Generating Skaffold environment file..."
	@echo "# Auto-generated from master .env - DO NOT EDIT DIRECTLY" > skaffold.env
	@echo "# Edit the root .env file and run 'make config' to regenerate" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# Network Configuration" >> skaffold.env
	@echo "BACKEND_HOST=$(BACKEND_HOST)" >> skaffold.env
	@echo "WEBUI_HOST=$(WEBUI_HOST)" >> skaffold.env
	@echo "SPEAKER_HOST=$(SPEAKER_HOST)" >> skaffold.env
	@echo "EXTERNAL_DOMAIN=$(EXTERNAL_DOMAIN)" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# Namespaces" >> skaffold.env
	@echo "INFRASTRUCTURE_NAMESPACE=$(INFRASTRUCTURE_NAMESPACE)" >> skaffold.env
	@echo "APPLICATION_NAMESPACE=$(APPLICATION_NAMESPACE)" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# Node Ports" >> skaffold.env
	@echo "BACKEND_NODEPORT=$(BACKEND_NODEPORT)" >> skaffold.env
	@echo "WEBUI_NODEPORT=$(WEBUI_NODEPORT)" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# API Keys and Secrets" >> skaffold.env
	@echo "HF_TOKEN=$(HF_TOKEN)" >> skaffold.env
	@echo "DEEPGRAM_API_KEY=$(DEEPGRAM_API_KEY)" >> skaffold.env
	@echo "OPENAI_API_KEY=$(OPENAI_API_KEY)" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# Speaker Recognition" >> skaffold.env
	@echo "SIMILARITY_THRESHOLD=$(SIMILARITY_THRESHOLD)" >> skaffold.env
	@echo "COMPUTE_MODE=$(COMPUTE_MODE)" >> skaffold.env
	@echo "SPEAKER_SERVICE_HOST=0.0.0.0" >> skaffold.env
	@echo "SPEAKER_SERVICE_PORT=$(SPEAKER_PORT)" >> skaffold.env
	@echo "SPEAKER_SERVICE_URL=http://speaker-recognition-speaker.speech.svc.cluster.local:$(SPEAKER_PORT)" >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# React UI Configuration" >> skaffold.env
	@echo "REACT_UI_HOST=$(REACT_UI_HOST)" >> skaffold.env
	@echo "REACT_UI_PORT=$(REACT_UI_PORT)" >> skaffold.env
	@echo "REACT_UI_HTTPS=$(REACT_UI_HTTPS)" >> skaffold.env
	@echo "WEBUI_CORS_ORIGIN=*" >> skaffold.env
	@echo 'VITE_ALLOWED_HOSTS="$(VITE_ALLOWED_HOSTS)"' >> skaffold.env
	@echo "" >> skaffold.env
	@echo "# Memory Provider Configuration" >> skaffold.env
	@echo "MEMORY_PROVIDER=$(MEMORY_PROVIDER)" >> skaffold.env
	@echo "OPENMEMORY_MCP_URL=$(OPENMEMORY_MCP_URL)" >> skaffold.env
	@echo "✅ Skaffold environment file generated"

clean-config: ## Clean up generated configuration files
	@echo "🧹 Cleaning up generated configuration files..."
	@rm -f backends/advanced/.env
	@rm -f extras/speaker-recognition/.env
	@rm -f skaffold.env
	@echo "✅ Generated files cleaned"

deploy-docker: config-docker ## Deploy using Docker Compose
	@echo "🐳 Deploying with Docker Compose..."
	@cd backends/advanced && docker compose up --build -d
	@echo "✅ Docker Compose deployment complete"
	@echo "   Backend: $(BACKEND_URL)"
	@echo "   WebUI:   $(WEBUI_URL)"

deploy-infrastructure: config-skaffold ## Deploy infrastructure services (MongoDB, Qdrant)
	@echo "📦 Deploying infrastructure to namespace: $(INFRASTRUCTURE_NAMESPACE)"
	@if [ ! -f skaffold.env ]; then echo "❌ skaffold.env not found. Run 'make config-skaffold' first."; exit 1; fi
	@set -a; source skaffold.env; set +a; skaffold run --profile=infrastructure
	@echo "⏳ Waiting for infrastructure to be ready..."
	@kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mongodb -n $(INFRASTRUCTURE_NAMESPACE) --timeout=300s || true
	@kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant -n $(INFRASTRUCTURE_NAMESPACE) --timeout=300s || true
	@echo "✅ Infrastructure deployment complete"

check-apps: ## Check if applications are already deployed
	@echo "🔍 Checking application status in namespace: $(APPLICATION_NAMESPACE)"
	@if kubectl get namespace $(APPLICATION_NAMESPACE) >/dev/null 2>&1; then \
		echo "  ✅ Namespace '$(APPLICATION_NAMESPACE)' exists"; \
		if kubectl get deployment advanced-backend -n $(APPLICATION_NAMESPACE) >/dev/null 2>&1; then \
			echo "  ✅ Advanced Backend is deployed ($(shell kubectl get deployment advanced-backend -n $(APPLICATION_NAMESPACE) -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo '0/0') replicas)"; \
		else \
			echo "  ❌ Advanced Backend not found"; \
		fi; \
		if kubectl get deployment webui -n $(APPLICATION_NAMESPACE) >/dev/null 2>&1; then \
			echo "  ✅ WebUI is deployed ($(shell kubectl get deployment webui -n $(APPLICATION_NAMESPACE) -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo '0/0') replicas)"; \
		else \
			echo "  ❌ WebUI not found"; \
		fi; \
	else \
		echo "  ❌ Namespace '$(APPLICATION_NAMESPACE)' does not exist"; \
	fi

deploy-apps: config-k8s config-skaffold ## Deploy application services (Backend, WebUI)
	@echo "🚀 Deploying/updating applications to namespace: $(APPLICATION_NAMESPACE)"
	@$(MAKE) check-apps
	@echo
	@if [ ! -f skaffold.env ]; then echo "❌ skaffold.env not found. Run 'make config-skaffold' first."; exit 1; fi
	@echo "📦 Running Skaffold deployment (will update existing deployments)..."
	@set -a; source skaffold.env; set +a; skaffold run --profile=advanced-backend --default-repo=$(CONTAINER_REGISTRY) --force=false
	@echo "✅ Application deployment complete"

check-infrastructure: ## Check if infrastructure is already deployed
	@echo "🔍 Checking infrastructure status in namespace: $(INFRASTRUCTURE_NAMESPACE)"
	@if kubectl get namespace $(INFRASTRUCTURE_NAMESPACE) >/dev/null 2>&1; then \
		echo "  ✅ Namespace '$(INFRASTRUCTURE_NAMESPACE)' exists"; \
		if kubectl get deployment mongodb -n $(INFRASTRUCTURE_NAMESPACE) >/dev/null 2>&1; then \
			echo "  ✅ MongoDB is deployed"; \
		else \
			echo "  ❌ MongoDB not found"; \
		fi; \
		if kubectl get deployment qdrant -n $(INFRASTRUCTURE_NAMESPACE) >/dev/null 2>&1; then \
			echo "  ✅ Qdrant is deployed"; \
		else \
			echo "  ❌ Qdrant not found"; \
		fi; \
	else \
		echo "  ❌ Namespace '$(INFRASTRUCTURE_NAMESPACE)' does not exist"; \
	fi

deploy-k8s: config-k8s config-skaffold ## Deploy to Kubernetes (smart - checks infrastructure first)
	@echo "☸️  Smart Kubernetes Deployment"
	@echo "================================"
	@$(MAKE) check-infrastructure
	@echo
	@if kubectl get deployment mongodb -n $(INFRASTRUCTURE_NAMESPACE) >/dev/null 2>&1 && \
	   kubectl get deployment qdrant -n $(INFRASTRUCTURE_NAMESPACE) >/dev/null 2>&1; then \
		echo "📋 Infrastructure already exists, skipping infrastructure deployment"; \
		echo "   Use 'make deploy-infrastructure' to force infrastructure redeploy"; \
	else \
		echo "📦 Infrastructure missing, deploying infrastructure first..."; \
		$(MAKE) deploy-infrastructure; \
	fi
	@echo
	@echo "🚀 Deploying applications..."
	@$(MAKE) deploy-apps

deploy-k8s-full: deploy-infrastructure deploy-apps ## Deploy everything to Kubernetes (including infrastructure)

deploy: ## Deploy using the configured deployment mode
ifeq ($(DEPLOYMENT_MODE),kubernetes)
	@$(MAKE) deploy-k8s
else
	@$(MAKE) deploy-docker
endif

status: ## Show deployment status
	@echo "📊 Friend-Lite Deployment Status"
	@echo "================================"
	@echo "Deployment Mode: $(DEPLOYMENT_MODE)"
	@echo "Domain: $(DOMAIN)"
	@echo
	@if [ "$(DEPLOYMENT_MODE)" = "kubernetes" ]; then \
		echo "Kubernetes Pods:"; \
		kubectl get pods -n $(INFRASTRUCTURE_NAMESPACE) 2>/dev/null || echo "  No infrastructure pods found"; \
		kubectl get pods -n $(APPLICATION_NAMESPACE) 2>/dev/null || echo "  No application pods found"; \
	else \
		echo "Docker Containers:"; \
		cd backends/advanced && docker compose ps 2>/dev/null || echo "  No containers running"; \
	fi

show-config: ## Show current configuration values
	@echo "🔧 Current Configuration"
	@echo "======================="
	@echo "DEPLOYMENT_MODE:     $(DEPLOYMENT_MODE)"
	@echo "DOMAIN:              $(DOMAIN)"
	@echo "BACKEND_URL:         $(BACKEND_URL)"
	@echo "WEBUI_URL:           $(WEBUI_URL)"
	@echo "SPEAKER_SERVICE_URL: $(SPEAKER_SERVICE_URL)"
	@echo "LLM_PROVIDER:        $(LLM_PROVIDER)"
	@echo "TRANSCRIPTION_PROVIDER: $(TRANSCRIPTION_PROVIDER)"
	@echo "MEMORY_PROVIDER:     $(MEMORY_PROVIDER)"
	@echo
	@echo "Infrastructure Namespace: $(INFRASTRUCTURE_NAMESPACE)"
	@echo "Application Namespace:    $(APPLICATION_NAMESPACE)"

# ========================================
# LEGACY TARGETS (Deprecated)
# ========================================

build-backend: ## Legacy: Build Docker Compose services (use 'make deploy-docker' instead)
	@echo "⚠️  WARNING: build-backend is deprecated. Use 'make deploy-docker' instead."
	@echo "Building Docker Compose services from backends/advanced directory..."
	@cd backends/advanced && docker compose build

up-backend: ## Legacy: Start Docker Compose services (use 'make deploy-docker' instead)  
	@echo "⚠️  WARNING: up-backend is deprecated. Use 'make deploy-docker' instead."
	@echo "Starting Docker Compose services from backends/advanced directory..."
	@cd backends/advanced && docker compose up -d

down-backend: ## Legacy: Stop Docker Compose services
	@echo "Stopping Docker Compose services from backends/advanced directory..."
	@cd backends/advanced && docker compose down
