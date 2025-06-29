from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

CHROMA_DB_PATH = "./chromadb"
COLLECTION_NAME = "jd_collection"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH) if CHROMA_DB_PATH else chromadb.Client()

# Create the vector store. This will set its dimension based on the first embedding added.
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize conversation memory with summary buffer
# This will keep recent messages and summarize older ones
conversation_memory = {}  # Store memory per session/user

def estimate_tokens(text):
    """Rough estimate of tokens (1 token ≈ 4 characters for English)"""
    return len(text) // 4

def get_or_create_memory(session_id="default"):
    """Get or create conversation memory for a session"""
    if session_id not in conversation_memory:
        # Simple conversation history with summarization
        conversation_memory[session_id] = {
            "messages": [],
            "summary": "",
            "max_messages": 8,
            "max_summary_length": 1500  # Limit summary length
        }
    return conversation_memory[session_id]

def add_to_memory(session_id, human_message, ai_message):
    """Add conversation turn to memory with automatic summarization"""
    memory = get_or_create_memory(session_id)
    
    # Truncate long messages to reduce tokens
    max_message_length = 1000  # characters
    human_message = human_message[:max_message_length] + "..." if len(human_message) > max_message_length else human_message
    ai_message = ai_message[:max_message_length] + "..." if len(ai_message) > max_message_length else ai_message
    
    # Add new messages
    memory["messages"].append({"role": "human", "content": human_message})
    memory["messages"].append({"role": "ai", "content": ai_message})
    
    # If we exceed max messages, summarize older ones
    if len(memory["messages"]) > memory["max_messages"]:
        # Take the first half of messages to summarize
        messages_to_summarize = memory["messages"][:memory["max_messages"]//2]
        
        # Create summary prompt
        conversation_text = ""
        for msg in messages_to_summarize:
            role = "Human" if msg["role"] == "human" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Generate summary using LLM with token limits
        summary_prompt = f"""Please provide a very brief summary of this conversation in 2-3 sentences maximum (under 50 words):

{conversation_text}

Brief Summary:"""
        
        try:
            new_summary = llm.invoke(summary_prompt).content
            
            # Update memory: replace old summary with new one (don't accumulate)
            memory["summary"] = new_summary[:memory["max_summary_length"]]  # Truncate if too long
            
            # Keep only the recent messages
            memory["messages"] = memory["messages"][memory["max_messages"]//2:]
            
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    return memory

def get_conversation_context(session_id):
    """Get the full conversation context including summary"""
    memory = get_or_create_memory(session_id)
    
    context = ""
    if memory["summary"]:
        context += f"Previous conversation summary:\n{memory['summary']}\n\n"
    
    context += "Recent conversation:\n"
    for msg in memory["messages"]:
        role = "Human" if msg["role"] == "human" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context

retriever = vectorstore.as_retriever()

# Updated RAG chain that includes conversation context
def create_rag_response(question, session_id="default"):
    """Create RAG response with conversation context"""
    # Get conversation context
    conversation_context = get_conversation_context(session_id)
    
    # Get relevant documents with limit
    relevant_docs = retriever.get_relevant_documents(question, k=20)  # Limit to most relevant docs
    
    # Truncate document content to reduce tokens
    max_doc_length = 3000  # characters per document
    doc_content = []
    for doc in relevant_docs:
        content = doc.page_content[:max_doc_length]
        if len(doc.page_content) > max_doc_length:
            content += "..."
        doc_content.append(content)
    
    # Create more concise prompt
    enhanced_prompt = f"""Answer the question using the context provided. Be concise.

Context: {conversation_context[:1000]}...

Documents: {' '.join(doc_content)}

Question: {question}

Answer:"""

    # Get response from LLM
    response = llm.invoke(enhanced_prompt)
    return response.content

# Create a Flask application instance
app = Flask(__name__)

# Define a route for the home page or root URL
@app.route('/')
def home():
    """
    Handles requests to the root URL.
    Returns a simple greeting message.
    """
    return "Hello from your simple Flask server!"

# Define another route that takes a name as a query parameter
@app.route('/greet')
def greet():
    """
    Handles requests to /greet.
    Takes an optional 'name' query parameter and returns a personalized greeting.
    Example: /greet?name=Alice
    """
    # Get the 'name' query parameter from the request URL
    user_name = request.args.get('name', 'Guest') # Default to 'Guest' if no name is provided
    return f"Hello, {user_name}! Welcome to the Flask server."

# Define a route that demonstrates a simple JSON response
# This route will now only accept POST requests
@app.route('/rest/message', methods=['POST']) # Changed to only accept POST requests
def api_data():
    """
    Handles POST requests to /api/data.
    Expects a JSON payload in the request body.
    Returns a JSON object with a confirmation message.
    """
    if request.is_json: # Check if the incoming request has a JSON content type
        received_data = request.get_json() # Get the JSON payload
        user_message = received_data.get("message", "")
        session_id = received_data.get("session_id", "default")  # Allow different sessions
        
        message = f"Received POST request for session {session_id} with JSON data: {received_data}"
        print(message) # Print to console for debugging/logging
        
        # Create RAG response with conversation context
        reply = create_rag_response(user_message, session_id)
        
        # Add to conversation memory
        add_to_memory(session_id, user_message, reply)
        
        response_data = {
            "status": "success",
            "message": reply,
            "session_id": session_id
        }
        return jsonify(response_data), 200 # Return 200 OK status
    else:
        # If not JSON, return an error
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

# Combined route for memory operations
@app.route('/rest/memory/<session_id>', methods=['GET', 'DELETE'])
def memory_operations(session_id):
    """Combined route for memory operations - GET to view, DELETE to clear"""
    if request.method == 'GET':
        # Get memory data (previously debug_memory)
        memory = get_or_create_memory(session_id)
        
        # Calculate token estimates
        total_message_tokens = sum(estimate_tokens(msg["content"]) for msg in memory["messages"])
        summary_tokens = estimate_tokens(memory["summary"])
        total_tokens = total_message_tokens + summary_tokens
        
        return jsonify({
            "session_id": session_id,
            "summary": memory["summary"],
            "messages": memory["messages"],
            "message_count": len(memory["messages"]),
            "estimated_tokens": {
                "messages": total_message_tokens,
                "summary": summary_tokens,
                "total": total_tokens
            }
        })
    
    elif request.method == 'DELETE':
        # Clear memory data (previously clear_memory)
        if session_id in conversation_memory:
            del conversation_memory[session_id]
            return jsonify({"status": "success", "message": f"Memory cleared for session {session_id}"})
        else:
            return jsonify({"status": "info", "message": f"No memory found for session {session_id}"})

# Run the Flask application
if __name__ == '__main__':
    # app.run() starts the development server.
    # debug=True enables debug mode (reloads on code changes, provides debugger).
    # host='0.0.0.0' makes the server accessible from other machines on the network.
    app.run(debug=True, host='0.0.0.0', port=5000)
