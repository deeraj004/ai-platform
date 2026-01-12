# src/core/tools/base/llm/azure_openai_response_builder.py

import sys
import os
import time
import datetime
from typing import Dict, Any, List, Optional

import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Add project root for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_MODEL,
    AZURE_OPENAI_AUGMENT_USER_QUERY_SYSTEM_PROMPT,
    AZURE_OPENAI_SYSTEM_PROMPT,
    AZURE_OPENAI_DATE_MESSAGE_TEMPLATE,
    logger
)


class AzureOpenAI:
    """
    Handles augmentation, refinement, and response generation using Azure ChatOpenAI.
    Exposes specific methods as LangGraph tools.
    """

    _client = None

    def __init__(self):
        logger.info("Initializing AzureOpenAIResponseBuilder.")

    @classmethod
    def get_client(cls) -> AzureChatOpenAI:
        """Get or create the Azure OpenAI client (singleton)."""
        if cls._client is None:
            logger.info("Creating AzureChatOpenAI client instance.")
            cls._client = AzureChatOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                model=AZURE_OPENAI_MODEL,
                temperature=0.7
            )
        return cls._client

    def _build_date_message(self) -> str:
        """Helper to build the date message string."""

        now = datetime.datetime.now()
        date_ddmmyyyy = now.strftime("%d/%m/%Y")
        date_written = now.strftime("%B %d, %Y")
        current_date = date_written  # Use written format as current_date
        
        # Support multiple placeholder formats in the template
        try:
            return AZURE_OPENAI_DATE_MESSAGE_TEMPLATE.format(
                current_date=current_date,
                date_ddmmyyyy=date_ddmmyyyy, 
                date_written=date_written
            )
        except KeyError as e:
            # If template has unexpected placeholders, log and return simple date
            logger.warning(f"Date message template has unexpected placeholder {e}, using simple format")
            return f"Current date: {current_date}"

    def refine_and_augment_user_query(
        self,
        query: str,
        precheck_summary: dict,
        conversation_history: list[dict],
        template_kwargs: dict = None
    ) -> Dict[str, Any]:
        """
        Refine and augment a user query using AzureChatOpenAI.

        Args:
            query (str): Raw user query.
            precheck_summary (dict): Metadata including temporal or contextual flags.
            conversation_history (list[dict]): Past conversation turns.
            template_kwargs (dict, optional): Additional dynamic parameters.

        Returns:
            Dict[str, Any]: Refined query or error message.
        """
        try:
            logger.info("Refining and augmenting user query using AzureChatOpenAI.")

            # Base system prompt
            system_prompt = AZURE_OPENAI_AUGMENT_USER_QUERY_SYSTEM_PROMPT

            # Check for temporal context
            has_temporal = (
                precheck_summary.get("has_temporal", False)
                if isinstance(precheck_summary, dict)
                else getattr(precheck_summary, "has_temporal", False)
            )

            if has_temporal:
                system_prompt += self._build_date_message()

            # Construct LLM message payload
            messages = [
                ("system", system_prompt),
                ("user", f"User Query: {query}\nConversation History: {conversation_history}")
            ]

            # Invoke Azure LLM
            start = time.time()
            response = self.get_client().invoke(messages)
            duration = time.time() - start

            logger.info(f"User query refined and augmented successfully in {duration:.2f}s.")
            return {"refined_query": response.content.strip()}

        except Exception as e:
            logger.error(f"Error refining and augmenting query: {e}")
            return {"error": str(e)}

    def generate_response(self, query: str, context: str, ranked_search_results: tuple[list[str], list[str]], template_kwargs: dict = None, custom_system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        LangGraph Tool: Generates a response using AzureChatOpenAI.
        Expects state with:
            - query: str
            - context: list
            - ranked_search_results: tuple[list[str], list[str]]
            - template_kwargs (optional): dict
            - custom_system_prompt (optional): str - Override default system prompt
        Returns dict with 'final_response'.
        """
        try:

            logger.info("Generating final response using AzureChatOpenAI.")
            # Use custom prompt if provided (for TSL use case), otherwise use default
            system_prompt = custom_system_prompt if custom_system_prompt else AZURE_OPENAI_SYSTEM_PROMPT
            if custom_system_prompt:
                logger.info("Using custom system prompt for response generation")
            date_message = self._build_date_message()

            # Extract search results
            source_urls, chunk_data = ranked_search_results

            # Format documents
            if chunk_data:
                formatted_documents_list = []
                for i, chunk in enumerate(chunk_data):
                    doc_text = f"Document {i+1}:\n{chunk}"
                    if source_urls and i < len(source_urls) and source_urls[i]:
                        doc_text += f"\nSource: {source_urls[i]}"
                    formatted_documents_list.append(doc_text)
                formatted_documents = "\n\n".join(formatted_documents_list)
            else:
                formatted_documents = "No relevant documents found."

            # Handle prompt with placeholders
            if "{documents}" in system_prompt and "{question}" in system_prompt:
                formatted_system_prompt = system_prompt.format(
                    documents=formatted_documents, question=query
                )
                messages = [
                    ("system", f"{formatted_system_prompt}\n{date_message}"),
                    ("user", query)
                ]
            else:
                messages = [
                    ("system", f"{system_prompt}\n{date_message}"),
                    ("user", f"Query: {query} Context: {context}\n\nReference Data: {ranked_search_results}")
                ]

            start = time.time()
            response = self.get_client().invoke(messages)
            end = time.time()

            logger.info(f"Final response generated successfully in {end-start:.2f}s.")
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return str(e)


# ✅ Singleton instance
azure_chatopenai = AzureOpenAI()

# ✅ Exposed tools - standalone functions that call instance methods
@tool(description="Refines and augments the user query based on conversation history and context.")
def refine_and_augment_tool(query: str, precheck_summary: dict, conversation_history: list[dict], template_kwargs: dict = None) -> Dict[str, Any]:
    """
    Refines and augments a user query using AzureChatOpenAI.
    - query: str
    - precheck_summary: dict
    - conversation_history: list[dict]
    - template_kwargs (optional): dict
    Returns dict with 'refined_query'.
    """
    return azure_chatopenai.refine_and_augment_user_query(query, precheck_summary, conversation_history, template_kwargs)

@tool(description="Generates the final response using AzureChatOpenAI and retrieved context.")
def generate_response_tool(query: str, context: str, ranked_search_results: tuple[list[str], list[str]], template_kwargs: dict = None, custom_system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a response using AzureChatOpenAI.
    - custom_system_prompt: Optional custom prompt override for specific use cases (e.g., TSL)
    """
    return azure_chatopenai.generate_response(query, context, ranked_search_results, template_kwargs, custom_system_prompt)
