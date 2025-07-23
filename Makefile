# Makefile for running multiple docker-compose services

# Directories
BACKEND_DIR = backends/advanced-backend
ASR_DIR = extras/asr-services

# Default target
.PHONY: up down help

# Run both services
up:
	@echo "Starting backend services..."
	cd $(BACKEND_DIR) && docker-compose up -d
	@echo "Starting ASR services..."
	cd $(ASR_DIR) && docker-compose up -d
	@echo "All services started!"

# Stop both services
down:
	@echo "Stopping ASR services..."
	cd $(ASR_DIR) && docker-compose down
	@echo "Stopping backend services..."
	cd $(BACKEND_DIR) && docker-compose down
	@echo "All services stopped!"

# Build and run both services
build:
	@echo "Building and starting backend services..."
	cd $(BACKEND_DIR) && docker-compose up -d --build
	@echo "Building and starting ASR services..."
	cd $(ASR_DIR) && docker-compose up -d --build
	@echo "All services built and started!"


# Help
help:
	@echo "Available targets:"
	@echo "  up           - Start both backend and ASR services"
	@echo "  down         - Stop both services"
	@echo "  build        - Build and start both services"
