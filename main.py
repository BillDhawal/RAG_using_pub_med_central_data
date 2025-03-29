import streamlit as st
import json
import os
import logging
import textwrap

# Configure logging - simplified
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from data_loader.data_loader import (
    load_pubmed_data,
)
from supabase import create_client, Client
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from ui.llm_response import get_llm_response, get_rag_instance

# Define the path for storing chat history
CHAT_HISTORY_FILE = "chat_history.json"


def chunk_text(text, chunk_size=5000, overlap=100):
    chunks = []
    text_length = len(text)

    # Iterate through the text with the specified chunk size and overlap
    for i in range(0, text_length, chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk.strip():  # Check if the chunk is not empty after stripping whitespace
            chunks.append(chunk)

    # Debug log: Print the size of the chunks list and an example chunk
    logger.debug(f"Total chunks created: {len(chunks)}")
    if chunks:
        logger.debug(f"Example chunk: {chunks[0]}")

    return chunks

def load_chat_history():
    """Load chat history from file or initialize if not exists."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []


def save_chat_history(messages):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(messages, f)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")


def display_message(message):
    """Display a single message."""
    with st.chat_message(message["role"]):
        st.write(message["content"])


def initialize_session():
    """Initialize session state variables."""
    # Check if chat history exists in session state, if not, load it
    #
    if "chat_history" not in st.session_state:
        logger.debug("Loading chat history")
        st.session_state.chat_history = load_chat_history()
    if "rag_initialized" not in st.session_state:
        logger.debug("RAG system not initialized")
        st.session_state.rag_initialized = False
    if "rag_instance" not in st.session_state:
        logger.debug("Setting RAG instance to None initially")
        st.session_state.rag_instance = None
    if "supabase_client" not in st.session_state:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        logger.debug("Supabase create client")
        st.session_state.supabase_client = create_client(supabase_url, supabase_key)


def create_new_vector_store(
    supabase_instance, start_idx: int = 500, num_samples: int = 100
):
    try:
        documents = load_pubmed_data(start_idx, num_samples)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        data = []
        # Add a check out of all documents which have PMID value.
        for doc in documents:
            if not doc.get("PMID"):
                print(f"Document with ID {doc['id']} does not have a PMID value.")
                continue
            chunks = chunk_text(doc["content"])
            logger.debug(f"Total chunks for document ID {doc['id']}: {len(chunks)}")
            for idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                record = {
                    "title": doc["title"],
                    "pmid": str(doc["PMID"]),
                    "content": chunk,  # Store the chunked content
                    "chunk_id": idx,  # Track chunk order
                    "metadata": {"original_id": doc["id"], "total_chunks": len(chunks)},
                    "embedding": embedding,
                }
                data.append(record)
        logger.debug(f"Total records prepared for insertion: {len(data)}")
        batch_size = 50
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}, Batch size: {len(batch)}")
            try:
                result = supabase_instance.table("pubmed_documents").upsert(batch).execute()
                print(f"Inserted batch {i//batch_size + 1}")
            except Exception as e:
                print(f"Error inserting batch {i//batch_size + 1}: {e}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")

def main():
    # TODO : Add Streaming of text
    # TODO : Add thinking chain of RAG
    # TODO : Add LangSmith for Observability
    logger.debug("Starting main application")
    st.set_page_config(
        page_title="PubMed Chatbot", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("PubMed Chatbot")

    # Initialize session
    # Entry point for the app
    # Load chat history and initialize session state of rag_initialized
    ## TODO: Enhancement : Rename this function to load_session_state_chat_history()
    initialize_session()

    # Initialize RAG instance once
    # Second hero function here, which initializes the RAG instance.
    supabase_instance = st.session_state.supabase_client
    if st.session_state.rag_instance is None:
        logger.debug("Getting RAG instance for the first time")
        st.session_state.rag_instance = get_rag_instance(supabase_instance)
        if st.session_state.rag_instance:
            st.session_state.rag_initialized = True
    else:
        logger.debug("Using existing RAG instance from session state")
        rag_instance = st.session_state.rag_instance
        st.session_state.rag_initialized = True

    # Add sidebar controls
    # Any streamlit command inside this block will be rendered in the sidebar
    with st.sidebar:
        st.header("Controls")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()

        # Add a divider for better UI
        st.divider()

        # Display text in header formatting.
        st.header("RAG Settings")

        # Initialize RAG system if not already done
        if not st.session_state.rag_initialized:
            with st.spinner("Initializing RAG system... (this might take a minute)"):
                try:
                    # This will create the vectorstore if it doesn't exist
                    ## TODO : Add return value for get_rag_instance() better logging,
                    ## Also initialze the rag_initialized session state accordingly.
                    rag_instance = get_rag_instance(supabase_instance)
                    if rag_instance:
                        st.session_state.rag_initialized = True
                        st.success("RAG system initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing RAG system: {str(e)}")
        else:
            st.success("RAG system ready!")

        # TODO : There is a bug ig I uncomment it.
        # if st.button("Force Vector Store Creation for test data"):
        #     with st.spinner("Force Vector Store Creation, Downloading test data, Creating Embeddings, Storing to Supabase..."):
        #         try:
        #             # Force creation of embeddings and storage in Supabase
        #             # This is a placeholder for the actual implementation

        #             create_new_vector_store(supabase_instance)
        #         except Exception as e:
        #             st.error(f"Error rebuilding RAG index: {str(e)}")

    # Create a container for chat history
    chat_container = st.container()

    # Handle user input
    if user_input := st.chat_input(
        "What would you like to know about medical research?"
    ):
        logger.debug(f"Received user input: {user_input[:50]}...")
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get assistant response
        # get_llm_response is the hero function here, Which gets the response from the RAG pipeline.
        # Takes the RAG instance
        ## TODO: Enhancement : Add text Streaming
        with st.spinner("Researching PubMed articles..."):
            response_content = ""  # Initialize an empty string to accumulate the response
            response_placeholder = st.empty()  # Create a placeholder for streaming response

            # Stream the response and update the placeholder
            for chunk in get_llm_response(
                user_input,
                st.session_state.rag_instance,
                st.session_state.chat_history[:-1],
            ):
                response_content += chunk  # Accumulate the streamed content
                response_placeholder.text(response_content)  # Update the placeholder with the current content

            # Add the final response to the chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})

        # Save updated chat history to file
        save_chat_history(st.session_state.chat_history)

    # Display all messages in the chat container
    with chat_container:
        for message in st.session_state.chat_history:
            display_message(message)


if __name__ == "__main__":
    main()
