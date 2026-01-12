import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.tools import tool
from settings import TRANSFORMER_EMBEDDING_MODEL, AZURE_OPENAI_EMBEDDINGS_MODEL, AZURE_OPENAI_EMBEDDINGS_ENDPOINT, AZURE_OPENAI_EMBEDDINGS_API_KEY, AZURE_OPENAI_EMBEDDINGS_API_VERSION, IS_AZURE_OPENAI_EMBEDDINGS_ENABLED, AZURE_OPENAI_EMBEDDINGS_DIMENSIONS, logger

# Lazy loading: Initialize the model only when needed
_transformer_model = None

def get_transformer_model():
    """Get the transformer model, initializing it if necessary."""
    global _transformer_model
    if _transformer_model is None:
        try:
            logger.info(f"Initializing transformer model: {TRANSFORMER_EMBEDDING_MODEL}")
            _transformer_model = SentenceTransformer(TRANSFORMER_EMBEDDING_MODEL)
            logger.info("Transformer model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            # Log specific guidance for common issues
            if "Permission denied" in str(e) or "mkdir" in str(e):
                logger.error("Cache permission issue detected. Ensure HuggingFace cache directories are writable.")
                logger.error("Consider setting HF_HOME environment variable to a writable location.")
            raise
    return _transformer_model

def generate_openai_embeddings(chunk):
    """Generate embeddings using Azure OpenAI."""
    logger.info("Generating Azure OpenAI embeddings for a chunk.")
    
    embeddings = AzureOpenAIEmbeddings(
        model=AZURE_OPENAI_EMBEDDINGS_MODEL,
        dimensions=AZURE_OPENAI_EMBEDDINGS_DIMENSIONS,
        azure_endpoint=AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
        api_key=AZURE_OPENAI_EMBEDDINGS_API_KEY,
        openai_api_version=AZURE_OPENAI_EMBEDDINGS_API_VERSION,
    )
    
    embedding = embeddings.embed_query(chunk)
    logger.info("Generated Azure OpenAI embeddings successfully.")
    return embedding

def generate_transformer_embeddings(chunk): 
    """Generate embeddings using the transformer model."""
    logger.info("Generating transformer embeddings for a chunk.")
    if not isinstance(chunk, str):
        logger.error(f"Expected a string, but got {type(chunk)}: {chunk}")
        raise ValueError(f"Expected a string, but got {type(chunk)}: {chunk}")
    
    # Use lazy loading to get the model
    model = get_transformer_model()
    embedding = model.encode(chunk)
    logger.info("Generated transformer embeddings successfully.")
    return embedding

def chunk_text_using_transformer_model(file_text, max_tokens_per_chunk=510, overlap=50):
    logger.info("Chunking text using transformer model.")
    model_name = "bert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer {model_name}: {e}")
        if "Permission denied" in str(e) or "mkdir" in str(e):
            logger.error("Cache permission issue detected. Ensure HuggingFace cache directories are writable.")
        raise

    if isinstance(file_text, list):
        combined_text = " ".join(file_text)
    else:
        combined_text = file_text

    tokens = tokenizer(combined_text, add_special_tokens=False)["input_ids"]

    token_split_texts = []
    for i in range(0, len(tokens), max_tokens_per_chunk - overlap):
        chunk_tokens = tokens[i : i + max_tokens_per_chunk]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        token_split_texts.append(chunk_text)
    
    logger.info("Text chunking completed.")
    return token_split_texts

@tool(description="Generates embeddings for user query chunks.")
def get_chunk_embeddings(user_query):
    """Generate embeddings for user query."""
    logger.info("Generating embeddings for user query chunks.")
    try:
        if IS_AZURE_OPENAI_EMBEDDINGS_ENABLED:
            embedding = generate_openai_embeddings(user_query)
        else:
            embedding = generate_transformer_embeddings(user_query)
        
        if isinstance(embedding, list):
            query_embedding_list = embedding
        else:
            query_embedding_list = embedding.tolist()
        
        # Convert numpy float32 values to regular Python floats for JSON serialization
        query_embedding_list = [float(x) for x in query_embedding_list]
        
        logger.info("Generated embeddings for all chunks successfully.")
        return query_embedding_list
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise ValueError(f"Error generating embeddings for chunks: {str(e)}")
      
get_chunk_embeddings = get_chunk_embeddings