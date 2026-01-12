@echo off
REM Quick start script for Docker container (Windows)
REM Usage: .\docker-start.bat (PowerShell) or docker-start.bat (CMD)

echo.
echo ==================================================
echo   Banking Agentic RAG System - Docker Quick Start
echo ==================================================
echo.

REM Check if .env file exists
if not exist .env (
    echo ⚠️  Warning: .env file not found!
    echo 📝 Please create .env file with required environment variables
    echo    See README.md or DOCKER_GUIDE.md for details
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Docker is not running!
    echo    Please start Docker Desktop
    pause
    exit /b 1
)

REM Check if docker-compose is available
where docker-compose >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Using docker-compose
    echo.
    echo Building and starting container...
    docker-compose up -d --build
    
    echo.
    echo ⏳ Waiting for container to be healthy...
    timeout /t 5 /nobreak >nul
    
    echo.
    echo 📊 Container Status:
    docker-compose ps
    
    echo.
    echo 📝 View logs with: docker-compose logs -f
    echo 🛑 Stop with: docker-compose down
) else (
    REM Fallback to Docker CLI
    echo ✅ Using Docker CLI
    echo.
    echo Building image...
    docker build -t banking-agentic-rag:latest .
    
    echo.
    echo Starting container...
    docker run -d ^
        --name banking-rag ^
        -p 8000:8000 ^
        --env-file .env ^
        -v "%CD%\src\core\tools:/app/src/core/tools" ^
        --restart unless-stopped ^
        banking-agentic-rag:latest
    
    echo.
    echo ⏳ Waiting for container to start...
    timeout /t 5 /nobreak >nul
    
    echo.
    echo 📊 Container Status:
    docker ps | findstr banking-rag
    
    echo.
    echo 📝 View logs with: docker logs -f banking-rag
    echo 🛑 Stop with: docker stop banking-rag ^&^& docker rm banking-rag
)

echo.
echo 🌐 Application available at:
echo    - API: http://localhost:8000
echo    - Swagger: http://localhost:8000/docs
echo    - Health: http://localhost:8000/health
echo.
echo ✅ Done! Container is running.
pause

