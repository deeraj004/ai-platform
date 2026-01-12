#!/bin/bash
# Quick start script for Docker container

set -e

echo "🚀 Banking Agentic RAG System - Docker Quick Start"
echo "=================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "📝 Please create .env file with required environment variables"
    echo "   See README.md or DOCKER_GUIDE.md for details"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "   Please start Docker Desktop or Docker daemon"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "✅ Using docker-compose"
    echo ""
    echo "Building and starting container..."
    docker-compose up -d --build
    
    echo ""
    echo "⏳ Waiting for container to be healthy..."
    sleep 5
    
    echo ""
    echo "📊 Container Status:"
    docker-compose ps
    
    echo ""
    echo "📝 View logs with: docker-compose logs -f"
    echo "🛑 Stop with: docker-compose down"
    echo ""
    echo "🌐 Application available at:"
    echo "   - API: http://localhost:8000"
    echo "   - Swagger: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
    
elif command -v docker &> /dev/null; then
    echo "✅ Using Docker CLI"
    echo ""
    
    # Check if container already exists and remove it
    if docker ps -a --format '{{.Names}}' | grep -q '^banking-rag$'; then
        echo "⚠️  Container 'banking-rag' already exists"
        echo "🛑 Stopping and removing existing container..."
        docker stop banking-rag > /dev/null 2>&1 || true
        docker rm banking-rag > /dev/null 2>&1 || true
        echo "✅ Existing container removed"
    fi
    
    # Check if port 8000 is already in use
    if docker ps --format '{{.Ports}}' | grep -q ':8000->'; then
        echo "⚠️  Warning: Port 8000 is already in use by another container"
        echo "   You may need to stop the other container first"
    fi
    
    echo ""
    echo "Building image..."
    docker build -t banking-agentic-rag:latest .
    
    echo ""
    echo "Starting container..."
    docker run -d \
        --name banking-rag \
        -p 8000:8000 \
        --env-file .env \
        -v "$(pwd)/src/core/tools:/app/src/core/tools" \
        --restart unless-stopped \
        banking-agentic-rag:latest
    
    echo ""
    echo "⏳ Waiting for container to start..."
    sleep 5
    
    echo ""
    echo "📊 Container Status:"
    docker ps | grep banking-rag
    
    echo ""
    echo "📝 View logs with: docker logs -f banking-rag"
    echo "🛑 Stop with: docker stop banking-rag && docker rm banking-rag"
    echo ""
    echo "🌐 Application available at:"
    echo "   - API: http://localhost:8000"
    echo "   - Swagger: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
else
    echo "❌ Error: Docker is not installed!"
    echo "   Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

echo ""
echo "✅ Done! Container is running."

