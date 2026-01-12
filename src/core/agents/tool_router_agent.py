"""
Tool Router Agent - Pre-filters tools using semantic similarity before planning.
Reduces tool set from 100+ to 10-20 most relevant tools.
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from langchain_core.tools import BaseTool
from settings import logger
import hashlib
import json
import threading
from collections import OrderedDict

# Import numpy for type hints only
if TYPE_CHECKING:
    import numpy as np

# Try to import embedding model, fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Set to None if not available to avoid NameError
    np = None
    SentenceTransformer = None
    cosine_similarity = None
    logger.warning("sentence-transformers or scikit-learn not available. Tool router will use domain-based filtering only.")


from src.core.orchestrator.state_schema import AgenticState
from src.core.utils.observability import trace_agent


class LRUCache:
    """LRU Cache for tool embeddings with size limit."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str):
        """Get item and move to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Add item, evict oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)


class ToolRouterAgent:
    """
    Tool Router Agent - Uses embeddings to pre-filter tools before planning.
    
    Purpose:
    - Reduce tool set from 100+ to 10-20 most relevant tools
    - Use semantic similarity to find relevant tools
    - Prevents planning agent from being overwhelmed with too many tools
    
    Strategy:
    1. Pre-compute and cache tool embeddings (tools don't change often)
    2. Get domains from intent classification
    3. Collect candidate tools from those domains
    4. Use cached embeddings for fast similarity search
    5. Return top-k most relevant tools
    
    Performance Optimizations:
    - Caches tool embeddings to avoid re-encoding on every request
    - Uses batch similarity calculation for efficiency
    - Only encodes query on each request (fast)
    - Singleton pattern for efficient resource reuse
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ToolRouterAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize only once (singleton pattern)."""
        if self._initialized:
            # Update config if provided, but don't reinitialize
            if config:
                self.config.update(config)
            return
        
        with self._lock:
            if self._initialized:
                if config:
                    self.config.update(config)
                return
        self.config = config or {}
        self.embedding_model = None
        
        # Thread-safe cache for tool embeddings with LRU eviction
        self._cache_lock = threading.Lock()  # ✅ FIX: Initialize lock
        max_cache_size = self.config.get("max_cache_size", 1000)
        self._tool_embeddings_cache: LRUCache = LRUCache(max_size=max_cache_size)
        self._tool_signatures: Dict[str, str] = {}
        
        self._initialize_embeddings()
        ToolRouterAgent._initialized = True
    
    def _initialize_embeddings(self):
        """Initialize embedding model for semantic search"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Tool Router: Embedding libraries not available, using domain-based filtering")
            return
        
        try:
            # Use a lightweight model for fast embeddings
            if SentenceTransformer is not None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Tool Router: Embedding model initialized")
            else:
                self.embedding_model = None
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            self.embedding_model = None
    
    def _get_tool_text(self, tool: BaseTool) -> str:
        """Generate text representation of tool for embedding"""
        text = f"{tool.name} {tool.description}"
        
        # Add parameter information if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                schema = tool.args_schema.schema()
                params = schema.get("properties", {})
                param_text = " ".join([
                    f"{k} {v.get('description', '')}" 
                    for k, v in params.items()
                ])
                text += f" {param_text}"
            except Exception:
                pass
        
        return text
    
    def _get_tool_signature(self, tool: BaseTool) -> str:
        """Generate a signature for a tool to detect changes"""
        text = self._get_tool_text(tool)
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_tool_embeddings(self, tools: List[BaseTool]) -> Dict[str, Any]:
        """
        Get embeddings for tools with caching.
        Only generates embeddings for new/changed tools.
        Thread-safe with LRU eviction.
        """
        tool_embeddings = {}
        
        if not self.embedding_model:
            return tool_embeddings
        
        try:
            # Collect tools that need embedding (new or changed)
            tools_to_encode = []
            tool_texts = []
            
            for tool in tools:
                tool_name = tool.name
                signature = self._get_tool_signature(tool)
                
                # Check if we have cached embedding and signature matches (thread-safe)
                with self._cache_lock:
                    cached_embedding = self._tool_embeddings_cache.get(tool_name)
                    cached_signature = self._tool_signatures.get(tool_name)
                    has_cache = (cached_embedding is not None and 
                                cached_signature == signature)
                
                if has_cache:
                    # Use cached embedding
                    with self._cache_lock:
                        tool_embeddings[tool_name] = cached_embedding
                else:
                    # Need to generate new embedding
                    tools_to_encode.append(tool)
                    tool_texts.append(self._get_tool_text(tool))
            
            # Batch encode new/changed tools (much faster than one-by-one)
            if tools_to_encode:
                logger.debug(f"Tool Router: Encoding {len(tools_to_encode)} new/changed tools")
                # Batch encoding is more efficient
                new_embeddings = self.embedding_model.encode(
                    tool_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Cache the new embeddings (thread-safe with LRU eviction)
                with self._cache_lock:
                    for tool, embedding in zip(tools_to_encode, new_embeddings):
                        tool_name = tool.name
                        signature = self._get_tool_signature(tool)
                        tool_embeddings[tool_name] = embedding
                        self._tool_embeddings_cache.put(tool_name, embedding)
                        self._tool_signatures[tool_name] = signature
            
            logger.debug(
                f"Tool Router: Using {len(tool_embeddings)} embeddings "
                f"({len(tools_to_encode)} new, {len(tool_embeddings) - len(tools_to_encode)} cached)"
            )
            
        except Exception as e:
            logger.error(f"Error generating tool embeddings: {e}", exc_info=True)
        
        return tool_embeddings
    
    def route_tools(
        self,
        query: str,
        domains: List[str],
        all_tools: Dict[str, List[BaseTool]],
        top_k: Optional[int] = None  # ✅ FIX: Make configurable
    ) -> List[BaseTool]:
        """
        Two-stage tool routing for optimal performance:
        
        Stage 1: Domain/Entity Filtering (fast, reduces candidate set)
        - Filter tools based on identified domains/entities from user query
        - This dramatically reduces the number of tools to process
        
        Stage 2: Semantic Similarity Filtering (accurate, on filtered set)
        - Embed user query once
        - Embed only the domain-filtered tools (not all tools!)
        - Calculate similarity between query and filtered tools
        - Return top-k most relevant tools
        
        This approach is much more efficient than embedding all tools:
        - With 1000 tools across 10 domains, domain filtering reduces to ~100 tools
        - Only need to embed 100 tools instead of 1000 (10x reduction)
        - Query embedding is done once regardless
        
        Args:
            query: User query
            domains: Identified domains from intent classification
            all_tools: All available tools organized by domain
            top_k: Maximum number of tools to return (defaults to config or 15)
        
        Returns:
            Filtered list of most relevant tools
        """
        # ✅ FIX: Get top_k from config if not provided
        if top_k is None:
            top_k = self.config.get("top_k", 15)
        
        try:
            # ============================================================
            # STAGE 1: Domain/Entity-Based Filtering (Fast Pre-filter)
            # ============================================================
            logger.info(f"Tool Router Stage 1: Domain filtering from {len(domains)} domains")
            
            # Collect tools from identified domains only
            candidate_tools = []
            domain_tool_counts = {}
            
            for domain in domains:
                if domain in all_tools:
                    domain_tools = all_tools[domain]
                    candidate_tools.extend(domain_tools)
                    domain_tool_counts[domain] = len(domain_tools)
            
            # Always include RAG and system tools (they're always useful)
            if "rag" in all_tools:
                for tool in all_tools["rag"]:
                    if tool not in candidate_tools:
                        candidate_tools.append(tool)
                domain_tool_counts["rag"] = len([t for t in all_tools["rag"] if t not in candidate_tools])
            
            if "system" in all_tools:
                for tool in all_tools["system"]:
                    if tool not in candidate_tools:
                        candidate_tools.append(tool)
                domain_tool_counts["system"] = len([t for t in all_tools["system"] if t not in candidate_tools])
            
            total_tools_before_filtering = sum(len(tools) for tools in all_tools.values())
            logger.info(
                f"Tool Router Stage 1: Filtered {total_tools_before_filtering} tools → {len(candidate_tools)} candidates "
                f"(reduction: {((1 - len(candidate_tools)/total_tools_before_filtering) * 100):.1f}%) "
                f"from domains: {domain_tool_counts}"
            )
            
            if not candidate_tools:
                logger.warning("Tool Router: No candidate tools found after domain filtering")
                return []
            
            # Early return if we have few tools (no need for embedding)
            if len(candidate_tools) <= top_k:
                logger.info(
                    f"Tool Router: Returning all {len(candidate_tools)} domain-filtered tools "
                    f"(within limit of {top_k}, skipping embedding stage)"
                )
                return candidate_tools
            
            # ============================================================
            # STAGE 2: Semantic Similarity Filtering (Accurate, on filtered set)
            # ============================================================
            if self.embedding_model and EMBEDDINGS_AVAILABLE:
                try:
                    logger.info(
                        f"Tool Router Stage 2: Semantic similarity on {len(candidate_tools)} domain-filtered tools"
                    )
                    
                    # Embed user query (done once, regardless of tool count)
                    query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
                    logger.debug("Tool Router: Query embedding generated")
                    
                    # Embed ONLY the domain-filtered tools (not all tools!)
                    # This is the key optimization: we only embed the reduced candidate set
                    tool_embeddings = self._get_tool_embeddings(candidate_tools)
                    
                    if not tool_embeddings:
                        # Fallback if embeddings failed
                        logger.warning(
                            "Tool Router Stage 2: Embedding generation failed, "
                            f"using domain-based filtering (returning top {top_k})"
                        )
                        return candidate_tools[:top_k]
                    
                    # Batch calculate similarities (much faster than one-by-one)
                    # Stack all tool embeddings into a matrix
                    tool_names_list = list(tool_embeddings.keys())
                    tool_embeddings_matrix = np.vstack([
                        tool_embeddings[name] for name in tool_names_list
                    ])
                    
                    # Batch cosine similarity calculation (single matrix operation)
                    # This is efficient: O(1) matrix operation instead of O(n) loop
                    similarities_batch = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        tool_embeddings_matrix
                    )[0]
                    
                    # Create similarity dict
                    similarities = {
                        tool_name: float(sim) 
                        for tool_name, sim in zip(tool_names_list, similarities_batch)
                    }
                    
                    # Sort by similarity and get top-k
                    sorted_tools = sorted(
                        candidate_tools,
                        key=lambda t: similarities.get(t.name, 0.0),
                        reverse=True
                    )
                    
                    selected_tools = sorted_tools[:top_k]
                    
                    # Log top selections with similarity scores
                    top_selections = [
                        (t.name, similarities.get(t.name, 0.0)) 
                        for t in selected_tools[:5]
                    ]
                    logger.info(
                        f"Tool Router Stage 2: Selected {len(selected_tools)}/{len(candidate_tools)} tools "
                        f"using semantic similarity. Top 5: {top_selections}"
                    )
                    logger.info(
                        f"Tool Router: Final result - {len(selected_tools)} tools selected from "
                        f"{total_tools_before_filtering} total tools "
                        f"(domain filter: {len(candidate_tools)}, similarity filter: {len(selected_tools)})"
                    )
                    
                    return selected_tools
                
                except Exception as e:
                    logger.error(f"Tool Router Stage 2: Error in embedding-based routing: {e}", exc_info=True)
                    # Fallback to domain-based (Stage 1 result)
                    logger.warning(
                        f"Tool Router: Falling back to domain-based filtering, "
                        f"returning top {top_k} from {len(candidate_tools)} candidates"
                    )
                    return candidate_tools[:top_k]
            
            # Fallback: embeddings not available, use domain-based filtering only
            logger.info(
                f"Tool Router: Embeddings not available, using Stage 1 (domain-based) only. "
                f"Returning top {top_k} from {len(candidate_tools)} candidates"
            )
            return candidate_tools[:top_k]
        
        except Exception as e:
            logger.error(f"Tool Router error: {e}", exc_info=True)
            # Fallback: return tools from identified domains
            candidate_tools = []
            for domain in domains:
                if domain in all_tools:
                    candidate_tools.extend(all_tools[domain])
            return candidate_tools[:top_k] if candidate_tools else []
    
    @trace_agent("tool_router")
    def process(self, state: AgenticState, tool_registry: Dict[str, List[BaseTool]]) -> AgenticState:
        """
        Process state and route tools.
        Stores filtered tools in state for planning agent.
        
        Args:
            state: AgenticState with intent classification
            tool_registry: All available tools organized by domain
        
        Returns:
            Updated AgenticState with routed tools in metadata
        """
        try:
            logger.info("Tool Router: Starting tool routing")
            
            # Get domains from intent classification
            intent_data = state.metadata.get("intent_classification", {})
            domains = intent_data.get("domains_or_entities", [])
            
            if not domains:
                # Default to all domains if none identified (fallback)
                # Get all available domains from tool registry
                from src.core.registry.tool_registry import tool_registry
                domains = list(tool_registry.list_domains())
                logger.warning(f"Tool Router: No domains identified, using all {len(domains)} available domains")
            
            # ✅ FIX: Get top_k from config or state metadata
            top_k = self.config.get("top_k", 15)
            if "tool_router_top_k" in state.metadata:
                top_k = state.metadata["tool_router_top_k"]
            
            # Route tools
            filtered_tools = self.route_tools(
                query=state.normalized_query or state.user_query,
                domains=domains,
                all_tools=tool_registry,
                top_k=top_k
            )
            
            # Store filtered tools in state metadata
            state.metadata["routed_tools"] = {
                "tools": [tool.name for tool in filtered_tools],
                "count": len(filtered_tools),
                "domains": domains,
                "routing_method": "embeddings" if self.embedding_model else "domain_based",
                "top_k_used": top_k,
                "cache_size": len(self._tool_embeddings_cache)
            }
            
            # Store actual tool objects for planning agent (as a list of tool names for now)
            # Planning agent will reconstruct from registry
            state.metadata["filtered_tool_names"] = [tool.name for tool in filtered_tools]
            
            logger.info(
                f"Tool Router: Routed {len(filtered_tools)} tools from {len(domains)} domains "
                f"(method: {state.metadata['routed_tools']['routing_method']}, top_k={top_k}, cache_size={len(self._tool_embeddings_cache)})"
            )
            
            return state
        
        except Exception as e:
            logger.error(f"Tool Router error: {e}", exc_info=True)
            # Fallback: continue without filtering
            state.metadata["routed_tools"] = {
                "error": str(e),
                "count": 0,
                "routing_method": "error"
            }
            return state


def tool_router_node(state: AgenticState, tool_registry: Dict[str, List[BaseTool]]) -> AgenticState:
    """
    LangGraph node function for Tool Router Agent.
    
    Args:
        state: AgenticState
        tool_registry: All available tools organized by domain
    
    Returns:
        Updated AgenticState with routed tools
    """
    agent = ToolRouterAgent()
    return agent.process(state, tool_registry)

