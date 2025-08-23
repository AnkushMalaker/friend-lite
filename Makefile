# Makefile for building Docker Compose from backends/advanced directory

.PHONY: all help build-backend build-no-cache up-backend down-backend logs clean build-langfuse build-asr-services up-langfuse down-langfuse up-asr-services down-asr-services

# Build all Docker Compose services
all: build-backend build-langfuse build-asr-services
	@echo "All Docker Compose services have been built successfully!"

# Default target
help:
	@echo "Available targets:"
	@echo "  all          - Build all Docker Compose services from all directories"
	@echo "  build-backend - Build Docker Compose services from backends/advanced"
	@echo "  up-backend   - Start Docker Compose services from backends/advanced"
	@echo "  down-backend - Stop Docker Compose services from backends/advanced"
	@echo "  logs         - Show Docker Compose logs from backends/advanced"
	@echo "  clean        - Remove containers, networks, and images from backends/advanced"
	@echo "  build-langfuse - Build Docker Compose services from extras/langfuse"
	@echo "  up-langfuse   - Start Docker Compose services from extras/langfuse"
	@echo "  down-langfuse - Stop Docker Compose services from extras/langfuse"
	@echo "  build-asr-services - Build Docker Compose services from extras/asr-services"
	@echo "  up-asr-services   - Start Docker Compose services from extras/asr-services"
	@echo "  down-asr-services - Stop Docker Compose services from extras/asr-services"
	@echo "  help         - Show this help message"

# Build Docker Compose services
build-backend:
	@echo "Building Docker Compose services from backends/advanced directory..."
	cd backends/advanced && docker-compose build

# Start Docker Compose services
up-backend:
	@echo "Starting Docker Compose services from backends/advanced directory..."
	cd backends/advanced && docker-compose up -d

# Stop Docker Compose services
down-backend:
	@echo "Stopping Docker Compose services from backends/advanced directory..."
	cd backends/advanced && docker-compose down

# Show Docker Compose logs
logs:
	@echo "Showing Docker Compose logs from backends/advanced directory..."
	cd backends/advanced && docker-compose logs -f

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources from backends/advanced directory..."
	cd backends/advanced && docker-compose down --rmi all --volumes --remove-orphans

# Build Langfuse Docker Compose services
build-langfuse:
	@echo "Building Langfuse Docker Compose services from extras/langfuse directory..."
	cd extras/langfuse && docker-compose build

# Build ASR Services Docker Compose services
build-asr-services:
	@echo "Building ASR Services Docker Compose services from extras/asr-services directory..."
	cd extras/asr-services && docker-compose build

# Start Langfuse Docker Compose services
up-langfuse:
	@echo "Starting Langfuse Docker Compose services from extras/langfuse directory..."
	cd extras/langfuse && docker-compose up -d

# Stop Langfuse Docker Compose services
down-langfuse:
	@echo "Stopping Langfuse Docker Compose services from extras/langfuse directory..."
	cd extras/langfuse && docker-compose down

# Start ASR Services Docker Compose services
up-asr-services:
	@echo "Starting ASR Services Docker Compose services from extras/asr-services directory..."
	cd extras/asr-services && docker-compose up -d

# Stop ASR Services Docker Compose services
down-asr-services:
	@echo "Stopping ASR Services Docker Compose services from extras/asr-services directory..."
	cd extras/asr-services && docker-compose down
