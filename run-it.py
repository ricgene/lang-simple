from langgraph_sdk import get_sync_client
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LANGCHAIN_API_KEY"):
    raise ValueError("LANGCHAIN_API_KEY not found in environment variables. Please add it to your .env file.")

# Use the correct API key
client = get_sync_client(
    url="https://lang-simple-deployd-e8f006c74dd75124b0d880d30816a4d7.us.langgraph.app", 
    api_key="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"
)

# Stream the response with the correct graph name
for chunk in client.runs.stream(
    None,  # None means threadless run
    "simple_agent",  # Use the correct graph name from the error message
    input={
        "messages": [
            {"type": "human", "content": "What is LangGraph?"}
        ]
    },
    stream_mode="updates"
):
    print(chunk.data)