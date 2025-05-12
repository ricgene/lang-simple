from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict

# Load environment variables from .env file
load_dotenv()

# Check if the API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

# Initialize the LLM - will use OPENAI_API_KEY from loaded env vars
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the state type for the graph
class GraphState(TypedDict):
    messages: List

# Define node functions
def call_llm(state):
    """Call the LLM with the current messages."""
    response = llm.invoke(state["messages"])
    updated_messages = state["messages"] + [response]
        
    return {"messages": updated_messages}

# Build the graph - provide the state schema
builder = StateGraph(state_schema=GraphState)

# Add nodes
builder.add_node("llm", call_llm)

# Set the entry point
builder.set_entry_point("llm")

# Add edges - direct to END for one-turn agent
builder.add_edge("llm", END)

# Compile the graph
graph = builder.compile()

# Run the agent - for local testing
def run_agent(query):
    result = graph.invoke({
        "messages": [HumanMessage(content=query)]
    })
    return result["messages"]

# Example usage
if __name__ == "__main__":
    messages = run_agent("Research the top 5 machine learning frameworks and compare them.")
    for message in messages:
        print(f"{message.type}: {message.content}\n")