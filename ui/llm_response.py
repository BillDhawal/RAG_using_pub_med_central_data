from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from supabase import Client
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging
from langchain.tools.retriever import create_retriever_tool
from rag.rag_pipeline import PubMedRAG

# Configure logging - only essential logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Global RAG pipeline instance
_rag_instance = None

def get_rag_instance(supabase_client: Client) -> PubMedRAG:
    global _rag_instance

    if _rag_instance is None:
        logger.debug("Initializing new RAG instance")

        ## TODO: Remove this is only for testing, Initialize the supabase client. only once and pass the retriver for langchain RAG.
        _rag_instance = PubMedRAG(supabase_client)
        _rag_instance.initialize()

    return _rag_instance


def format_chat_history(messages: List[Dict[str, str]]) -> List:
    formatted_messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant that specializes in medical research. "
                "Provide informative and relevant responses based on PubMed research "
                "papers and the conversation context. Always cite your sources with "
                "PMID when possible."
            )
        )
    ]

    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=content))

    return formatted_messages


# RAG instance is sent as a parameter to this function, such that RAG is initialized only once.
def get_llm_response(
    user_input: str, rag_instance: PubMedRAG, chat_history: List[Dict[str, str]] = None
):
    try:
        logger.debug("Processing user input for LLM response")
        # Initialize chat history if needed
        if chat_history is None:
            chat_history = []

        # Stream the response from the RAG instance
        for chunk in rag_instance.query_stream(user_input):
            yield chunk  # Yield each chunk as it is received
    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}", exc_info=True)
        yield f"Error generating response: {str(e)}"
