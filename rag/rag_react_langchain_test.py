import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from rich.console import Console
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain import hub
from langgraph.errors import GraphRecursionError
from langchain_community.retrievers import WikipediaRetriever

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()


rich = Console()
RECURSION_LIMIT = 2 * 3 + 1

LLM_APIKEY = os.getenv("LLM_APIKEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://llm-api.cyverse.ai/v1")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

print(f"API Key: {LLM_APIKEY is not None}, API Base: {LLM_API_BASE}")


# Create an adapter class that provides the methods LangChain expects, this adapter is created to handle runtime errors.
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()


supabase_client = create_client(supabase_url, supabase_key)


def create_llama_chat_model():
    return ChatOpenAI(
        model="Llama-3.2-11B-Vision-Instruct",
        openai_api_key=LLM_APIKEY,
        openai_api_base=LLM_API_BASE,
    )


llm = create_llama_chat_model()

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        print(f"Document metadata: {doc.metadata}")
        title = doc.metadata.get("title", "No title available")
        # Check both uppercase and lowercase versions of PMID
        pmid = doc.metadata.get("PMID", doc.metadata.get("pmid", "No PMID available"))
        formatted_docs.append(
            f"Document Title: {title}\nPMID: {pmid}\nContent: {doc.page_content}"
        )
    return "\n\n".join(formatted_docs)


st_model = SentenceTransformerEmbeddings()

vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=st_model,
    table_name="pubmed_documents",
    query_name="match_documents",
)
supabase_vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return supabase_vs_retriever.get_relevant_documents(query)


retriever = MyRetriever()

retrieval_tool = create_retriever_tool(
    retriever=retriever,
    name="my_retriever",
    description="A tool to retrieve documents from my knowledge base created on OSHO discourse .",
)

# print(retrieval_tool.invoke("Lysosomal hydrolases of the epidermis , why is it used?"))

system_message = "You are an research assistant with expertise in going through the RAG trained on osho discourse. Use the following pieces of context from discource to answer the user's question. If you don't know the answer, just say you don't know. Don't try to make up an answer But help the user with the information that you got from the retrieval"

wiki_retriever = create_retriever_tool(
    retriever=WikipediaRetriever(),
    name="wikipedia_retriever",
    description="Wikipedia retriever to search for terms and helping for information which is not in the knowledge base.",
)

langgraph_agent = create_react_agent(
    model=llm, tools=[retrieval_tool, wiki_retriever], prompt=system_message
)


# query = (
#     "What was the main objective of the study comparing lorazepam and pentobarbital?"
# )
# messages = langgraph_agent.invoke({"messages": [("human", query)]})
# # {
# #     "input": query,
# #     "output": messages["messages"][-1].content,
# # }
# print(messages)

# try:
#     for chunk in langgraph_agent.stream(
#         {"messages": [("human", query)]},
#         {"recursion_limit": RECURSION_LIMIT},
#         stream_mode="values",
#     ):
#         print(chunk["messages"][-1])
# except GraphRecursionError:
#     print({"input": query, "output": "Agent stopped due to max iterations."})


def process_chunks(chunk):
    """
    Processes a chunk from the agent and displays detailed information about tool calls,
    including their outputs.
    """

    if "agent" in chunk:
        for message in chunk["agent"]["messages"]:
            if "tool_calls" in message.additional_kwargs:
                tool_calls = message.additional_kwargs["tool_calls"]

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    tool_query = tool_arguments["query"]

                    rich.print(
                        f"\nThe tool called: [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] "
                        f"with the query: [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]."
                        f" Please wait for the tool's response...",
                        style="deep_sky_blue1",
                    )

                    # Capture tool response
                    tool_response = retrieval_tool.invoke({"query": tool_query})

                    rich.print(
                        f"\n[green]Tool Response:[/green]\n{tool_response}\n",
                        style="green",
                    )

            else:
                agent_answer = message.content
                rich.print(f"\nAgent:\n{agent_answer}", style="black on white")


# Loop until the user chooses to quit the chat
while True:
    # Get the user's question and display it in the terminal
    user_question = input("\nUser:\n")

    # Check if the user wants to quit the chat
    if user_question.lower() == "quit":
        rich.print("\nAgent:\nHave a nice day! :wave:\n", style="black on white")
        break

    # Use the stream method of the LangGraph agent to get the agent's answer
    for chunk in langgraph_agent.stream(
        {"messages": [HumanMessage(content=user_question)]}
    ):
        # Process the chunks from the agent
        process_chunks(chunk)


# # What was the main objective of the study comparing lorazepam and pentobarbital?
# # Which drug provided greater sedation and antianxiety effects?
# # What were the dosages of lorazepam and pentobarbital used in the study?
