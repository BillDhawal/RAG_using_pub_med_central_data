import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from data_loader.data_loader import load_pubmed_data
from .llm_integration import create_llama_chat_model, create_rag_chain
from langchain_community.retrievers import WikipediaRetriever
from langgraph.prebuilt import create_react_agent
from langchain.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import SupabaseVectorStore
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instance
supabase_vs_retriever = None

# Create an adapter class that provides the methods LangChain expects, this adapter is created to handle runtime errors.
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        global supabase_vs_retriever
        return supabase_vs_retriever.get_relevant_documents(query)

class PubMedRAG:

    def __init__(self, supabase_client):
        # Initialize components as None
        self.llm = None
        self.rag_chain = None
        self.supabase_client = supabase_client

    def initialize(self):
        global supabase_vs_retriever
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize the vectorstore and retriever
        vectorstore = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=embedding_model,
            table_name="pubmed_documents",
            query_name="match_documents",
        )
        supabase_vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        retriever = MyRetriever()

        retrieval_tool = create_retriever_tool(
            retriever=retriever,
            name="my_retriever",
            description="A tool to retrieve documents from my knowledge base created from PubMedCentral Database.",
        )
        logger.debug("Initializing RAG pipeline")
        system_message = "You are a medical research assistant with expertise in analyzing PubMed papers. Use the following pieces of context from research papers to answer the user's question. If you don't know the answer, just say you don't know. Don't try to make up an answer But help the user with the information that you got from the retrieval"

        wiki_retriever = create_retriever_tool(
            retriever=WikipediaRetriever(),
            name="wikipedia_retriever",
            description="Wikipedia retriever to search for medical terms and helping for information which is not in the knowledge base.",
        )
        # Initialize LLM
        if self.llm is None:
            self._initialize_llm()

        # If check for self.rag_chain
        if self.rag_chain is None:
            self.rag_chain = create_react_agent(
                model=self.llm,
                tools=[retrieval_tool, wiki_retriever],
                prompt=system_message,
            )
            logger.debug("RAG pipeline initialization complete")

    def _initialize_llm(self):
        self.llm = create_llama_chat_model()

    def query(self, question: str) -> str:

        logger.debug(f"Processing RAG query: {question[:50]}...")

        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            messages = self.rag_chain.invoke({"messages": [("human", question)]})
            return messages["messages"][-1].content
        except Exception as e:
            logger.error(f"3.E2 Error in RAG query: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"
