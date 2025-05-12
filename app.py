from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import os

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define node functions
def call_llm(state):
    """Call the LLM with the current messages."""
    response = llm.invoke(state["messages"])
    updated_messages = state["messages"] + [response]
    
    # Try to extract next steps from the response
    try:
        # Simple approach: assume the LLM returns JSON-like text with steps
        next_steps_text = response.content.split("Next steps:")[1].strip()
        next_steps = json.loads(next_steps_text)
    except:
        next_steps = []
        
    return {"messages": updated_messages, "next_steps": next_steps}

# Build the graph - using dict-based state
builder = StateGraph()

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