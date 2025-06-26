from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain import hub
from langsmith import traceable
import chromadb
from langchain_chroma import Chroma
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
reply = rag_chain.invoke("For which job definitions are maintenance.")
print(reply)

# retriever = vectorstore.as_retriever()
# context = retriever.invoke(prompt)

# # Adding context to our prompt
# template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
# prompt_with_context = template.invoke({"query": prompt, "context": context})

# @traceable
# def reply_to_query(prompt_with_context):
#     llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
#     results = llm.invoke(prompt_with_context)
#     return results.content

# print(reply_to_query(prompt_with_context))