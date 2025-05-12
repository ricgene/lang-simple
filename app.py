from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# Define our state
class AgentState:
    messages: list
    next_steps: list = []

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define node functions
def call_llm(state: AgentState):
    """Call the LLM with the current messages."""
    response = llm.invoke(state.messages)
    state.messages.append(response)
    
    # Try to extract next steps from the response
    try:
        # Simple approach: assume the LLM returns JSON-like text with steps
        next_steps_text = response.content.split("Next steps:")[1].strip()
        state.next_steps = json.loads(next_steps_text)
    except:
        state.next_steps = []
        
    return state

def determine_if_done(state: AgentState):
    """Determine if we're done or need to continue."""
    if not state.next_steps or "COMPLETE" in state.next_steps:
        return "end"
    else:
        return "continue"

def execute_steps(state: AgentState):
    """Execute the next steps suggested by the LLM."""
    # In a real agent, you would execute the steps here
    # For this simple example, we'll just acknowledge them
    step_results = []
    for step in state.next_steps:
        step_results.append(f"Executed: {step}")
    
    # Add the results to messages
    state.messages.append(HumanMessage(content=f"Step results: {', '.join(step_results)}"))
    state.next_steps = []
    return state

# Build the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("llm", call_llm)
builder.add_node("execute", execute_steps)

# Add edges
builder.add_edge("llm", END)

builder.add_edge("execute", "llm")

# Compile the graph
graph = builder.compile()

# Run the agent
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