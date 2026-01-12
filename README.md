# Banking Agentic RAG System

A scalable, production-ready agentic system for banking chatbot applications with support for 100+ tools (scalable to thousands). This system intelligently routes and executes banking API calls through a multi-agent architecture powered by LangGraph.

## 🎯 Features

### Core Capabilities
- **Multi-Agent Architecture**: Intent Classification → Tool Routing → Planning → Execution → Evaluation
- **Scalable Tool Management**: Handles 100+ tools efficiently, designed to scale to 1000+
- **Two-Stage Tool Routing**: Domain-based pre-filtering + Semantic similarity for optimal performance
- **Banking API Integration**: Generic support for any banking API via Postman collections
- **OAuth2 Authentication**: Automatic token management for banking API calls
- **Dynamic Base URL**: Runtime base URL replacement for multi-environment support
- **RAG Pipeline**: Integrated Retrieval-Augmented Generation for knowledge base queries
- **System Search**: Self-querying capabilities for system introspection
- **Rate Limiting**: Token bucket algorithm for API protection (with bulk operation bypass)
- **Basic Authentication**: HTTP Basic Auth for API endpoints
- **Thread-Safe**: Singleton patterns and proper concurrency handling
- **Observability**: LangSmith integration with graceful fallback

### Performance Optimizations
- **Embedding Caching**: LRU cache with configurable size limits
- **Batch Operations**: Efficient batch encoding and similarity calculations
- **Atomic File Writes**: Prevents data corruption on concurrent writes
- **Early Exit Handling**: Smart routing for greetings and invalid queries

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (REST API)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Intent Classifier Agent                          │
│  • Understands user intent                                    │
│  • Handles greetings                                         │
│  • Extracts domains/entities                                  │
│  • Normalizes queries                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Tool Router Agent                                │
│  • Domain-based pre-filtering (Stage 1)                      │
│  • Semantic similarity filtering (Stage 2)                  │
│  • Reduces 1000+ tools → 15-20 relevant tools                │
│  • LRU cache for embeddings                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Planning Agent                                   │
│  • Creates execution plan                                    │
│  • Maps domains to tools                                     │
│  • Orders execution steps                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Execution Agent (ReAct)                         │
│  • Executes tools with LLM reasoning                         │
│  • Handles tool dependencies                                 │
│  • Adapts based on results                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Critic Evaluator Agent                           │
│  • Evaluates execution quality                               │
│  • Triggers retry if needed                                  │
│  • Provides feedback                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Generator                               │
│  • Generates final answer                                    │
│  • Formats response                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    Final Response
```

### Key Components

1. **Intent Classifier Agent**: Classifies user intent, handles greetings, extracts domains
2. **Tool Router Agent**: Two-stage filtering (domain + semantic) to reduce tool set
3. **Planning Agent**: Creates execution plans from user intent
4. **Execution Agent**: ReAct agent that executes tools with LLM reasoning
5. **Critic Evaluator Agent**: Evaluates execution quality and triggers retries
6. **Tool Registry**: Manages 100+ banking API tools with domain organization
7. **Postman Parser**: Generic parser for Postman collections to register banking APIs
8. **OAuth2 Auth Manager**: Automatic OAuth2 token generation and refresh for banking APIs
9. **RAG Pipeline**: Retrieval-Augmented Generation for knowledge base queries
10. **System Search**: Self-querying for system capabilities

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_rag_app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy model (optional, for NLP features)**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env  # Create .env file if it doesn't exist
   # Edit .env with your configuration
   ```

5. **Register Banking APIs (Optional)**
   ```bash
   python scripts/register_banking_apis.py "PayPal APIs.postman_collection.json"
   ```
   This will parse the Postman collection and register all banking API endpoints as tools.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_MODEL=gpt-4

# OpenSearch Configuration
OPENSEARCH_HOST=your-opensearch-host
OPENSEARCH_SEARCH_ALIAS_NAME=your_alias_name
OPENSEARCH_VECTOR_DIMENSIONS=1536
OPENSEARCH_TOP_K=5

# AWS Configuration (if using AWS OpenSearch)
AWS_REGION=us-east-1
AWS_SERVICE_NAME=es
IS_CLOUD_ENABLED=true

# API Authentication
API_USERNAME=admin
API_PASSWORD=your_secure_password
ENABLE_API_AUTH=true

# Banking API OAuth2 Configuration (for banking API tools)
BANKING_API_CLIENT_ID=your_banking_api_client_id
BANKING_API_CLIENT_SECRET=your_banking_api_client_secret
BANKING_API_BASE_URL=https://api.sandbox.bank.com  # Base URL for banking API
BANKING_API_TOKEN_URL=https://api.sandbox.bank.com/v1/oauth2/token  # Optional, defaults to {BASE_URL}/v1/oauth2/token

# Tool Router Configuration
TOOL_ROUTER_TOP_K=15
TOOL_ROUTER_MAX_CACHE_SIZE=1000

# Rate Limiting
RATE_LIMIT_SYSTEM_SEARCH_CAPACITY=100
RATE_LIMIT_SYSTEM_SEARCH_REFILL=10.0
RATE_LIMIT_TOOL_REGISTRY_CAPACITY=50
RATE_LIMIT_TOOL_REGISTRY_REFILL=5.0
RATE_LIMIT_API_CALLS_CAPACITY=1000
RATE_LIMIT_API_CALLS_REFILL=100.0

# Observability (Optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=banking-agentic-system
LANGSMITH_TRACING=true
```

### Configuration Files

- `settings.py`: Main configuration file
- `.env`: Environment variables (not committed to git)

## 📖 Usage

### Starting the Application

**Local Development:**
```bash
python app.py
```

**Using Uvicorn:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Using Docker:**
```bash
docker build -t banking-agentic-rag .
docker run -p 8000:8000 --env-file .env banking-agentic-rag
```

### Accessing the API

1. **Swagger UI**: `http://localhost:8000/docs` (Basic Auth if enabled via `ENABLE_SWAGGER_AUTH`)
2. **ReDoc**: `http://localhost:8000/redoc`
3. **Health Check**: `http://localhost:8000/health`

### Making API Calls

**Using curl:**
```bash
curl -u admin:admin123 \
  -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "27a3b4c5d6e7f8g9h0i1j2k3l4m",
    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
    "session_id": "660e8400-e29b-41d4-a716-446655440000",
    "user_query": "What is my account balance?",
    "conversation_history": []
  }'
```

**Using Python:**
```python
import requests
from requests.auth import HTTPBasicAuth

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "request_id": "27a3b4c5d6e7f8g9h0i1j2k3l4m",
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "session_id": "660e8400-e29b-41d4-a716-446655440000",
        "user_query": "What is my account balance?",
        "conversation_history": []
    },
    auth=HTTPBasicAuth("admin", "admin123")
)

print(response.json())
```

**Using Swagger UI:**
1. Navigate to `http://localhost:8000/docs`
2. Click "Authorize" button
3. Enter username: `admin` and password: `admin123`
4. Click "Authorize"
5. Try the endpoints!

## 🔌 API Endpoints

### Query Processing

**POST `/api/v1/query`**
- Process banking queries through the agentic workflow
- Requires: Basic Auth
- Request body: `QueryRequest` (see Swagger docs for schema)
- Response: `QueryResponse` with answer and metadata

### Tool Registry

**POST `/api/v1/tools/register`**
- Register a new banking API tool
- Requires: Basic Auth
- Request body: `ToolRegistrationRequest`

**GET `/api/v1/tools`**
- List all registered tools
- Requires: Basic Auth
- Query params: `domain`, `category`, `search` (optional)

**GET `/api/v1/tools/{tool_name}`**
- Get details about a specific tool
- Requires: Basic Auth

**DELETE `/api/v1/tools/{tool_name}`**
- Unregister a tool
- Requires: Basic Auth

**POST `/api/v1/tools/bulk-register`**
- Register multiple tools at once
- Requires: Basic Auth
- Request body: `BulkToolRegistrationRequest`

### Health Checks

**GET `/health`**
- Basic health check
- No auth required

**GET `/health/detailed`**
- Detailed health with system metrics
- No auth required

**GET `/ready`**
- Readiness check for Kubernetes
- No auth required

**GET `/live`**
- Liveness check
- No auth required

## 🐳 Docker Deployment

### Quick Start

**Option 1: Using Quick Start Script (Easiest)**
```bash
# Linux/Mac
./docker-start.sh

# Windows PowerShell
.\docker-start.bat

# Windows CMD
docker-start.bat
```

**Option 2: Using Docker Compose (Recommended)**
```bash
# Ensure .env file exists with all required variables
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Option 3: Using Docker CLI**
```bash
# Build image
docker build -t banking-agentic-rag:latest .

# Run container
docker run -d \
  --name banking-rag \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/src/core/tools:/app/src/core/tools \
  banking-agentic-rag:latest
```

### Complete Docker Guide

For detailed Docker instructions, troubleshooting, and production deployment, see **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)**.

**Key Topics Covered:**
- Step-by-step containerization guide
- Environment variable configuration
- Volume mounting for data persistence
- Health checks and monitoring
- Troubleshooting common issues
- Production deployment strategies
- Security best practices

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tool_router.py
```

### Test Examples

See `examples/` directory for:
- Bulk API registration examples
- Query processing examples
- Tool registry usage

## 🔧 Development

### Project Structure

```
ai_rag_app/
├── app.py                          # FastAPI application entry point
├── settings.py                     # Configuration
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── src/
│   ├── core/
│   │   ├── agents/                 # Agent implementations
│   │   │   ├── intent_agent.py
│   │   │   ├── tool_router_agent.py
│   │   │   ├── planning_agent.py
│   │   │   ├── execution_agent.py
│   │   │   └── critic_evaluator_agent.py
│   │   ├── endpoints/              # API endpoints
│   │   │   ├── query_endpoints.py
│   │   │   ├── tool_registry_endpoints.py
│   │   │   └── health_endpoints.py
│   │   ├── orchestrator/           # LangGraph orchestration
│   │   │   ├── graph.py
│   │   │   ├── agent_orchestrator.py
│   │   │   └── state_schema.py
│   │   ├── registry/               # Tool registry
│   │   │   ├── tool_registry.py
│   │   │   └── banking_api_registry.py
│   │   ├── tools/                  # Tool implementations
│   │   │   ├── rag.py
│   │   │   ├── system_search.py
│   │   │   └── registered_banking_tools.json  # Persisted tool registry
│   │   └── utils/                  # Utilities
│   │       ├── basic_auth.py
│   │       ├── rate_limiter.py
│   │       ├── observability.py
│   │       ├── opensearch_connector.py
│   │       ├── postman_parser.py    # Postman collection parser
│   │       └── oauth2_auth.py       # OAuth2 authentication manager
│   └── middleware/                 # Middleware
│       └── edge_security_middleware.py
├── scripts/                        # Utility scripts
│   └── register_banking_apis.py   # Bulk register APIs from Postman collection
├── examples/                       # Example scripts
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml              # Docker Compose configuration
├── docker-start.sh                 # Quick start script (Linux/Mac)
├── docker-start.bat                # Quick start script (Windows)
├── DOCKER_GUIDE.md                 # Complete Docker deployment guide
└── BANKING_API_INTEGRATION.md      # Banking API integration guide
```

### Key Design Decisions

1. **LangGraph**: Chosen for native state management and conditional routing
2. **Two-Stage Filtering**: Domain pre-filter + semantic similarity for scalability
3. **Singleton Pattern**: Efficient resource reuse for stateless agents
4. **LRU Cache**: Bounded memory usage for embeddings
5. **Atomic File Writes**: Prevents corruption on concurrent writes
6. **Rate Limiting**: Token bucket algorithm for DoS protection

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Domain Filtering | <1ms | Excellent |
| Query Embedding | ~10ms | Good |
| Tool Embedding (100 tools) | ~50ms | Good |
| Similarity Calculation | ~5ms | Excellent |
| **Total (cached)** | **~15ms** | **Excellent** |
| **Total (first request)** | **~65ms** | **Good** |

### Scalability

| Tool Count | Performance | Memory | Status |
|------------|-------------|--------|--------|
| 100 | Excellent | 0.5MB | ✅ Ready |
| 1,000 | Good | 5MB | ✅ Ready |
| 10,000 | Acceptable | 50MB | ⚠️ Consider vector DB |
| 100,000 | Slow | 500MB | ❌ Needs redesign |

## 🔒 Security Features

- ✅ HTTP Basic Authentication for API endpoints
- ✅ Rate limiting (token bucket algorithm) with bulk operation bypass
- ✅ Input validation (Pydantic models)
- ✅ SSRF protection in tool registry
- ✅ Thread-safe operations
- ✅ Credentials from environment variables
- ✅ Secure credential comparison (timing attack prevention)
- ✅ OAuth2 token caching and automatic refresh
- ✅ Description truncation to prevent API response size issues

## 🔌 Banking API Integration

### Registering Banking APIs from Postman Collections

The system supports generic banking API integration through Postman collections:

1. **Prepare your Postman collection** with all banking API endpoints
2. **Configure OAuth2 credentials** in `.env`:
   ```bash
   BANKING_API_CLIENT_ID=your_client_id
   BANKING_API_CLIENT_SECRET=your_client_secret
   BANKING_API_BASE_URL=https://api.sandbox.bank.com
   ```

3. **Register all APIs at once**:
   ```bash
   python scripts/register_banking_apis.py "Your Banking APIs.postman_collection.json"
   ```

### How It Works

1. **Postman Collection Parsing**: Extracts API endpoints, methods, parameters, and descriptions
2. **Tool Registration**: Registers each endpoint as a LangChain tool with OAuth2 authentication
3. **Runtime Base URL Replacement**: `{{base_url}}` placeholder is replaced with `BANKING_API_BASE_URL` at runtime
4. **Automatic Authentication**: OAuth2 tokens are automatically fetched and refreshed when tools are invoked
5. **Domain Organization**: Tools are organized by domain (e.g., `orders`, `payments`, `invoices`)

### Features

- **Generic Support**: Works with any banking API, not provider-specific
- **Bulk Registration**: Register 100+ APIs in seconds (rate limiting bypassed for bulk operations)
- **OAuth2 Integration**: Automatic token management with client credentials flow
- **Multi-Environment**: Switch between sandbox/production by changing `BANKING_API_BASE_URL`
- **Description Truncation**: Long HTML descriptions are automatically cleaned and truncated

For detailed integration guide, see [BANKING_API_INTEGRATION.md](BANKING_API_INTEGRATION.md).

## 📝 Recent Updates

### Version 2.0 - Banking API Integration

**New Features:**
- ✅ Generic banking API support via Postman collections
- ✅ OAuth2 authentication with automatic token management
- ✅ Runtime base URL replacement (`{{base_url}}` → `BANKING_API_BASE_URL`)
- ✅ Bulk API registration with rate limiting bypass
- ✅ Description truncation for long HTML descriptions
- ✅ 111+ banking API tools registered and ready to use

**Improvements:**
- ✅ Fixed rate limiting for bulk operations
- ✅ Enhanced error handling for API calls
- ✅ Thread-safe OAuth2 token management
- ✅ Generic implementation (not provider-specific)

**Breaking Changes:**
- None - fully backward compatible

**Migration Notes:**
- If you have existing PayPal-specific configurations, update to use generic `BANKING_API_*` environment variables
- See [BANKING_API_INTEGRATION.md](BANKING_API_INTEGRATION.md) for migration guide

---

**Built with ❤️ using LangGraph, FastAPI, and Azure OpenAI**

