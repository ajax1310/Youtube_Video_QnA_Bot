from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessageGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Define the ChatState to hold all conversation messages
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_app(transcript_path: str):
    """Create a LangGraph app for conversational QA given a transcript file path."""
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    SYSTEM_PROMPT = f"""You are a helpful assistant having a conversation about the content of a YouTube video.

Use the following transcript content as your only source of truth:
{transcript_text}

Answer questions and maintain a friendly, conversational tone.
If a question is unrelated to the video content, politely say so.
"""

    # Prompt for conversational chat
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])

    # Chat model setup
    llm = ChatGroq(api_key=groq_api_key, model_name="Llama-3.3-70b-Versatile")
    chain: Runnable = prompt | llm

    # Node function for LangGraph
    def conversational_node(state: ChatState) -> ChatState:
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph
    graph = StateGraph(ChatState)
    graph.add_node("conversational", conversational_node)
    graph.set_entry_point("conversational")
    graph.set_finish_point("conversational")
    return graph.compile()
