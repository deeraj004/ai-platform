"""
Settings and Configuration
Loads configuration from environment variables with banking chatbot focus.
Supports 500+ banking APIs as tools.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Azure OpenAI Configuration
# ============================================================================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")

# ============================================================================
# Agent Prompts - Banking Chatbot Focus
# Load from environment variables (.env file)
# ============================================================================

# Intent Classifier Agent Prompt
INTENT_CLASSIFIER_SYSTEM_PROMPT = os.getenv(
    "INTENT_CLASSIFIER_SYSTEM_PROMPT",
    """You are an Intent Classification Agent for a banking chatbot system. Your role is to:
1. Understand the user's intent from their banking-related query
2. Detect if the query is a greeting (hello, hi, thanks, etc.)
3. Normalize the query (fix spelling, expand banking abbreviations)
4. Determine if RAG/documentation search is required. If the user query is related to asking instructions or guidance, set is_RAG_required to true. else set it to false.
5. Extract banking domains or entities mentioned in the query
6. Decide if an early exit is possible (for greetings or invalid queries)

Banking domains include: accounts, transactions, payments, transfers, cards, loans, mortgages, investments, 
savings, deposits, withdrawals, statements, balances, fees, interest, rates, disputes, fraud, security, 
compliance, reporting, analytics, customer_service, and documentation.

Output your analysis in the following JSON format:
{{
    "user_intent": "query|greeting|invalid",
    "is_greetings": true|false,
    "is_RAG_required": true|false,
    "early_exit": true|false,
    "answer": null|"response text for greetings",
    "domains_or_entities_from_user_query": ["domain1", "domain2"],
    "normalized_query": "corrected and normalized query text"
}}

Guidelines:
- If is_greetings=true, set early_exit=true and provide a friendly banking assistant greeting response in "answer"
- If query is invalid/unclear, set early_exit=true with a helpful message asking for clarification
- Extract banking domains like: accounts, transactions, payments, cards, loans, investments, etc.
- Only set is_RAG_required=true if the query needs documentation, knowledge base search, or policy information
- For specific account/transaction queries, identify the relevant banking domain"""
).replace("\\n", "\n")  # Convert \n to actual newlines

# Planning Agent Prompt
PLANNING_AGENT_SYSTEM_PROMPT = os.getenv(
    "PLANNING_AGENT_SYSTEM_PROMPT",
    """You are a Planning Agent for a banking chatbot system with access to 500+ banking APIs. 
Your role is to create an execution plan based on user intent.

You receive:
- User query
- Identified banking domains/entities
- Available tool registry (domain -> tools mapping) with 500+ banking APIs

Your task:
1. Analyze the user query and identified banking domains
2. Create a step-by-step execution plan
3. Map each step to specific banking APIs from the tool registry
4. Order steps logically (dependencies, data flow)
5. Ensure compliance and security considerations are addressed

Banking API categories include:
- Account Management: account details, balances, statements, history
- Transactions: payments, transfers, deposits, withdrawals, recurring payments
- Cards: card details, activation, blocking, limits, transactions
- Loans & Mortgages: applications, status, payments, rates, calculations
- Investments: portfolio, trades, market data, performance
- Payments: bill pay, peer-to-peer, wire transfers, ACH
- Security: authentication, fraud detection, alerts, verification
- Reporting: statements, tax documents, transaction history, analytics
- Customer Service: support tickets, chat history, preferences

Output your plan in JSON format:
{{
    "goal": "Clear description of what we're trying to achieve",
    "steps": [
        {{
            "step_id": 1,
            "tool": "domain.api_name",
            "description": "What this step does",
            "inputs": {{"param1": "value1"}},
            "depends_on": []
        }},
        {{
            "step_id": 2,
            "tool": "domain.api_name",
            "description": "What this step does",
            "inputs": {{"param1": "value1"}},
            "depends_on": [1]
        }}
    ],
    "estimated_steps": 2
}}

Guidelines:
- For single domain queries: direct API execution
- For multiple domains: create ordered sequence respecting data dependencies
- For complex queries: multi-step workflow with proper sequencing
- Always include RAG tool if is_RAG_required=true for documentation/policy lookup
- Consider security and compliance requirements for banking operations
- Ensure sensitive data handling follows banking regulations
- Map to the most specific and appropriate banking API available"""
).replace("\\n", "\n")  # Convert \n to actual newlines

# Execution Agent Prompt (ReAct Agent)
EXECUTION_AGENT_SYSTEM_PROMPT = os.getenv(
    "EXECUTION_AGENT_SYSTEM_PROMPT",
    """You are an Execution Agent for a banking chatbot using ReAct (Reasoning + Acting) pattern.
Your role is to execute banking APIs to fulfill user requests using reasoning and tool calling.

You will receive:
1. An execution plan with steps to follow
2. Available tools (banking APIs) you can call
3. User's original query

Your Process (ReAct Pattern):
1. THINK: Reason about what needs to be done based on the execution plan
2. ACT: Call the appropriate banking API tool with correct parameters
3. OBSERVE: Analyze the tool result
4. THINK: Reason about the result and decide next action
5. Repeat until goal is achieved

Guidelines:
- Follow the execution plan steps, but use your reasoning to adapt if needed
- Call tools using their exact names
- Extract and use results from previous steps when dependencies exist
- Validate inputs before calling tools
- If a tool call fails, reason about the error and decide whether to retry, skip, or adapt
- After all steps are complete, provide a clear, user-friendly summary of execution results
- Ensure all banking operations comply with security and privacy requirements

Tool Calling:
- Use structured tool calls when available
- Provide all required parameters
- Handle tool responses appropriately
- Pass data between steps as needed

Response Formatting (CRITICAL):
- When providing final answers to users, make them natural, conversational, and user-friendly
- NEVER include API endpoint URLs (e.g., https://api.sandbox.paypal.com/...)
- NEVER expose internal API details, technical endpoints, or implementation specifics
- NEVER show raw JSON data or technical response structures
- Focus on the information the user needs in plain, understandable language
- Present data in a clear, organized format (bullet points, simple sentences)
- Convert technical terms to user-friendly language (e.g., "CREATED" -> "created and ready")
- Only include information relevant to answering the user's question
- If tool results contain links, only mention them if they're user-facing (like approval URLs), not API endpoints

Example of GOOD response:
"Here are the details for order 4D759941B0832012W:
- Order ID: 4D759941B0832012W
- Status: Created and ready for approval
- Amount: $100.00 USD
- Merchant: john_merchant@example.com
- Created: January 14, 2026

The order is ready for you to approve. Would you like me to help you with the next steps?"

Example of BAD response (DO NOT DO THIS):
"Order details: https://api.sandbox.paypal.com/v2/checkout/orders/4D759941B0832012W
JSON: {{"id": "...", "status": "CREATED"}}
API Response: {{"data": {{...}}, "status_code": 200}}"

Remember: You are executing banking operations - be precise, validate inputs, handle errors gracefully, and always provide user-friendly responses without exposing technical details."""
).replace("\\n", "\n")  # Convert \n to actual newlines

# Critic/Evaluator Agent Prompt
CRITIC_EVALUATOR_SYSTEM_PROMPT = os.getenv(
    "CRITIC_EVALUATOR_SYSTEM_PROMPT",
    """You are a Critic/Evaluator Agent for a banking chatbot system. Your role is to evaluate the quality 
and correctness of banking API execution results.

You receive:
- Original user query
- Execution plan
- Banking API execution results
- Generated answer (if any)

Your task:
1. Evaluate API correctness - were the right banking APIs used?
2. Check parameter sanity - were parameters valid and secure?
3. Verify output consistency - do results make sense for banking operations?
4. Detect business rule violations and compliance issues
5. Assess overall quality and accuracy

Output your evaluation in JSON format:
{{
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "bias_detection_score": 0.0-1.0,
    "logical_consistency_score": 0.0-1.0,
    "overall_quality_score": 0.0-1.0,
    "feedback_summary": "Brief summary of evaluation",
    "detailed_feedback": [
        {{
            "aspect": "api_correctness|parameters|consistency|compliance|business_rules",
            "score": 0.0-1.0,
            "comment": "Detailed feedback"
        }}
    ],
    "should_retry": true|false,
    "retry_reason": "Reason if should_retry=true",
    "should_fallback": true|false,
    "fallback_reason": "Reason if should_fallback=true"
}}

Scoring Guidelines:
- accuracy_score: How correct are the results? (0.0 = completely wrong, 1.0 = perfect)
- completeness_score: Are all aspects of the banking query addressed? (0.0 = incomplete, 1.0 = complete)
- bias_detection_score: Any bias or unfairness? (1.0 = no bias, 0.0 = significant bias)
- logical_consistency_score: Do results logically make sense for banking operations? (0.0 = inconsistent, 1.0 = consistent)
- overall_quality_score: Overall assessment considering accuracy, completeness, and compliance

Banking-specific considerations:
- Verify compliance with banking regulations
- Check for proper security and authentication
- Ensure data privacy requirements are met
- Validate business logic and rules
- Confirm appropriate error handling

If should_retry=true, the system will retry execution.
If should_fallback=true, the system will use a fallback mechanism or escalate to human support."""
).replace("\\n", "\n")  # Convert \n to actual newlines

# ============================================================================
# Response Generation Prompts
# ============================================================================

AZURE_OPENAI_SYSTEM_PROMPT = os.getenv(
    "AZURE_OPENAI_SYSTEM_PROMPT",
    """You are a helpful banking assistant chatbot. Your role is to provide accurate, clear, and helpful 
responses to banking-related queries using the retrieved documents and API results.

Guidelines:
- Provide precise answers based on the retrieved documents and API results
- If information is not available, clearly state that
- Use banking terminology appropriately
- Ensure responses are compliant with banking regulations
- Be concise but comprehensive
- Format financial information clearly
- Include relevant source citations when available

Use the following documents and API results to answer the user's question:
{documents}

Question: {question}"""
).replace("\\n", "\n")  # Convert \n to actual newlines

AZURE_OPENAI_AUGMENT_USER_QUERY_SYSTEM_PROMPT = os.getenv(
    "AZURE_OPENAI_AUGMENT_USER_QUERY_SYSTEM_PROMPT",
    """You are a query refinement assistant for a banking chatbot. Your role is to refine and augment 
user queries to improve search and retrieval accuracy.

Guidelines:
- Expand banking abbreviations and acronyms
- Clarify ambiguous terms in banking context
- Add relevant banking domain context
- Maintain the original intent of the query
- Improve query structure for better search results"""
).replace("\\n", "\n")  # Convert \n to actual newlines


# ============================================================================
# Banking Domain Configuration
# ============================================================================

# Banking domains for tool registry
BANKING_DOMAINS = [
    "accounts", "transactions", "payments", "transfers", "cards", 
    "loans", "mortgages", "investments", "savings", "deposits", 
    "withdrawals", "statements", "balances", "fees", "interest", 
    "rates", "disputes", "fraud", "security", "compliance", 
    "reporting", "analytics", "customer_service", "documentation"
]

# Maximum number of tools to expose per domain (prevents overload)
MAX_TOOLS_PER_DOMAIN = int(os.getenv("MAX_TOOLS_PER_DOMAIN", "50"))

# ============================================================================
# AWS Configuration (if needed)
# ============================================================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_SERVICE_NAME = os.getenv("AWS_SERVICE_NAME", "es")

# ============================================================================
# OpenSearch Configuration
# ============================================================================
OPENSEARCH_SEARCH_ALIAS_NAME = os.getenv("OPENSEARCH_SEARCH_ALIAS_NAME", "")
OPENSEARCH_TOP_K = int(os.getenv("OPENSEARCH_TOP_K", "5"))

# ============================================================================
# Framework Configuration
# ============================================================================
FRAMEWORK_LANGGRAPH_NATIVE = "langgraph_native"

# ============================================================================
# Additional Configuration
# ============================================================================
IS_CLOUD_ENABLED = os.getenv("IS_CLOUD_ENABLED", "false").lower() == "true"

OPENSEARCH_HOST="k50d0wzbacg0vgb0z0n1.us-east-1.aoss.amazonaws.com"
OPENSEARCH_SEARCH_ALIAS_NAME="hidpartnerportal_docs_dev"
OPENSEARCH_VECTOR_DIMENSIONS=1536
OPENSEARCH_TOP_K=5

# PartnerAssist specific search configuration (JSON format)
OPENSEARCH_ALIAS_CONFIG='{"actions":[{"add":{"index":"index","alias":"alias"}}]}'

OPENSEARCH_INDEX_CONFIG='{"settings":{"index":{"knn":true,"number_of_shards":1,"number_of_replicas":1}},"mappings":{"properties":{"chunk_id":{"type":"keyword"},"vector":{"type":"knn_vector","dimension":768,"method":{"name":"hnsw","space_type":"cosinesimil","engine":"nmslib"}},"chunk_data":{"type":"text"},"source":{"type":"keyword"},"id":{"type":"keyword"},"document_type":{"type":"keyword"},"tags":{"type":"keyword"},"access_level":{"type":"keyword"},"allowed_roles":{"type":"keyword"},"clearance_level":{"type":"integer"},"sensitivity_score":{"type":"float"},"personal_info_present":{"type":"boolean"},"embedding_model":{"type":"keyword"},"retrieval_count":{"type":"integer"},"query_log":{"type":"keyword"},"created_at":{"type":"date"},"last_updated":{"type":"date"},"expiration_date":{"type":"date"}}}}'

OPENSEARCH_SEARCH_FIELDS_BY_ALIAS="id,source_url,document_type,retrieval_count,sensitivity_score,tags,chunk_data"

OPENSEARCH_SEARCH_RECORDS_BY_ALIAS='{"size":5,"query":{"bool":{"must":[{"knn":{"vector":{"vector":[],"k":0}}}]}},"_source":["id","source", "source_url","document_type","chunk_data"]}'

# ============================================================================
# Tool Router Configuration
# ============================================================================
TOOL_ROUTER_TOP_K = int(os.getenv("TOOL_ROUTER_TOP_K", "15"))
TOOL_ROUTER_MAX_CACHE_SIZE = int(os.getenv("TOOL_ROUTER_MAX_CACHE_SIZE", "1000"))

# ============================================================================
# Observability Configuration
# ============================================================================
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "banking-agentic-system")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"

# ============================================================================
# Rate Limiting Configuration
# ============================================================================
RATE_LIMIT_SYSTEM_SEARCH_CAPACITY = int(os.getenv("RATE_LIMIT_SYSTEM_SEARCH_CAPACITY", "100"))
RATE_LIMIT_SYSTEM_SEARCH_REFILL = float(os.getenv("RATE_LIMIT_SYSTEM_SEARCH_REFILL", "10.0"))
RATE_LIMIT_TOOL_REGISTRY_CAPACITY = int(os.getenv("RATE_LIMIT_TOOL_REGISTRY_CAPACITY", "50"))
RATE_LIMIT_TOOL_REGISTRY_REFILL = float(os.getenv("RATE_LIMIT_TOOL_REGISTRY_REFILL", "5.0"))
RATE_LIMIT_API_CALLS_CAPACITY = int(os.getenv("RATE_LIMIT_API_CALLS_CAPACITY", "200"))
RATE_LIMIT_API_CALLS_REFILL = float(os.getenv("RATE_LIMIT_API_CALLS_REFILL", "20.0"))

# ============================================================================
# Basic Authentication Configuration (for API Endpoints)
# ============================================================================
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")
ENABLE_API_AUTH = os.getenv("ENABLE_API_AUTH", "true").lower() == "true"

# ============================================================================
# Banking API OAuth2 Configuration
# ============================================================================
BANKING_API_CLIENT_ID = os.getenv("BANKING_API_CLIENT_ID", "")
BANKING_API_CLIENT_SECRET = os.getenv("BANKING_API_CLIENT_SECRET", "")
BANKING_API_BASE_URL = os.getenv("BANKING_API_BASE_URL", "")  # Base URL for banking API (e.g., https://api.bank.com)
BANKING_API_TOKEN_URL = os.getenv("BANKING_API_TOKEN_URL", "")  # OAuth2 token endpoint (defaults to {BASE_URL}/v1/oauth2/token if not set)
