from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_chroma import Chroma
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
print(f"ChromaDB vectorstore '{COLLECTION_NAME}' initialized.")

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini")

retriever = vectorstore.as_retriever()
rag_chain = (
  {
    "context": retriever, 
    "question": RunnablePassthrough()
  }
  | prompt
  | llm
  | StrOutputParser()
)

while True:
  user_input = input("Your question or 'exit'> ").strip()
  
  if user_input.lower() == 'exit':
    print("Goodbye!")
    break
  
  reply = rag_chain.invoke(user_input)
  print(reply)
