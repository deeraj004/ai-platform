from settings import logger, ENABLE_RERANKING, TRANSFORMER_RERANKING_MODEL
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from langchain_core.tools import tool
cross_encoder = CrossEncoder(TRANSFORMER_RERANKING_MODEL) if ENABLE_RERANKING and TRANSFORMER_RERANKING_MODEL else None

@tool(
    description=(
        "Reranks retrieved text chunks using CrossEncoder. "
        "Takes 'query' and 'documents' from state, filters out chunks "
        "with score < 0.3, and returns reranked 'documents' and 'scores'."
    ),
)
def rerank_chunks_with_crossencoder(query, source_urls, chunk_data, scores, query_filter_attributes, document_metadata=None) -> Dict[str, Any]:
    """
    LangGraph tool wrapper for CrossEncoder ranking.
    Now also reorders document_metadata to match the reranked results.
    """


    for i, (chunk, score) in enumerate(zip(chunk_data[:5], scores[:5])):
        logger.info(f"Before rerank [{i}] score={score:.4f} text={chunk[:100]}...")
    
    if not chunk_data:
        return [], [], [], []
    
    # Step 2: Rerank with cross-encoder
    if not ENABLE_RERANKING or not TRANSFORMER_RERANKING_MODEL:
        logger.warning("Reranking is disabled or model is not specified, skipping reranking.")
        return source_urls, chunk_data, scores, document_metadata or []

    cross_encoder_inputs = [(query, passage) for passage in chunk_data]
    cross_scores = [float(s) for s in cross_encoder.predict(cross_encoder_inputs)]

    # Min-Max normalization
    min_s, max_s = float(min(cross_scores)), float(max(cross_scores))
    if max_s > min_s:
        norm_scores = [(s - min_s) / (max_s - min_s) for s in cross_scores]
    else:
        # All scores same → assign neutral value
        norm_scores = [0.5] * len(cross_scores)

    logger.info(f"Reranked top scores (normalized [0,1]): {norm_scores[:5]}")

    # Step 3: Sort by reranked scores (include document_metadata if available)
    if document_metadata and len(document_metadata) == len(chunk_data):
        reranked = sorted(
            zip(source_urls, chunk_data, norm_scores, document_metadata),
            key=lambda x: x[2],
            reverse=True
        )
        reranked_source_urls, reranked_chunk_data, reranked_scores, reranked_metadata = zip(*reranked)
        reranked_metadata = list(reranked_metadata)
    else:
        reranked = sorted(
            zip(source_urls, chunk_data, norm_scores),
            key=lambda x: x[2],
            reverse=True
        )
        reranked_source_urls, reranked_chunk_data, reranked_scores = zip(*reranked)
        reranked_metadata = []
    
    # Step 4: Logging after reranking
    for i, (chunk, score) in enumerate(zip(reranked_chunk_data[:5], reranked_scores[:5])):
        logger.info(f"After rerank [{i}] score={score:.4f} text={chunk[:100]}...")
    
    if reranked_metadata:
        logger.info(f"Reranked {len(reranked_metadata)} document metadata entries to match reranked results")
    
    return list(reranked_source_urls), list(reranked_chunk_data), list(reranked_scores), reranked_metadata

cross_encoder_rerank_tool = rerank_chunks_with_crossencoder