from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import Union, Dict, Any 
import xml.etree.ElementTree as ET 
import chromadb
import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

RMJ_URL = os.getenv("RMJ_URL")
RMJ_REST_API_KEY = os.getenv("RMJ_REST_API_KEY")

JD_DETAILS_URL = f"{RMJ_URL}/api-extension/External/REDWOOD/Redwood_RestService/rest/v1/JobDefinition/"
GET_JDS_URL = f"{RMJ_URL}/api-rest/list/JobDefinition?check=View"
USERNAME = "admin"
PASSWORD = "admin"

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

def _xml_to_dict(element: ET.Element) -> Dict[str, Any]:
  """
  Helper function to convert a simple XML element and its children to a dictionary.
  Handles basic elements with text and attributes.
  Does not handle complex mixed content, namespaces, or repeated tags perfectly.
  """
  result = {}
  if element.attrib:
    result.update(element.attrib) # Include attributes
  
  children = list(element)
  if children:
    for child in children:
      if child.tag in result:
        # If tag repeats, convert to list of dicts
        if not isinstance(result[child.tag], list):
          result[child.tag] = [result[child.tag]]
        result[child.tag].append(_xml_to_dict(child))
      else:
        result[child.tag] = _xml_to_dict(child)
  elif element.text and element.text.strip():
    # If no children but has text, take text content
    return element.text.strip()
  return result

def get_rest_service_data(url: str, params: dict = None, headers: dict = None, auth: tuple = None) -> Union[Dict[str, Any], str]:
  try:
    response = requests.get(url, params=params, headers=headers, auth=auth)
    response.raise_for_status() 

    # print(f"Status Code: {response.status_code}", file=sys.stderr)
    # print(f"Response Headers: {response.headers}", file=sys.stderr)

    content_type = response.headers.get('Content-Type', '').lower()

    if 'application/json' in content_type:
      return response.json()
    else:
      try:
        root = ET.fromstring(response.text)
        return _xml_to_dict(root) # Convert XML to dictionary
      except ET.ParseError as xml_err:
        print("Warning: Response is neither JSON nor XML. Returning raw text.", file=sys.stderr)
        return {"raw_response": response.text, "status_code": response.status_code}
  
  except requests.exceptions.HTTPError as http_err:
      print(f"HTTP error occurred: {http_err}", file=sys.stderr)
      return {"error": "HTTP Error", "details": str(http_err), "status_code": http_err.response.status_code}
  except requests.exceptions.ConnectionError as conn_err:
      print(f"Connection error occurred: {conn_err}", file=sys.stderr)
      return {"error": "Connection Error", "details": str(conn_err)}
  except requests.exceptions.Timeout as timeout_err:
      print(f"Timeout error occurred: {timeout_err}", file=sys.stderr)
      return {"error": "Timeout Error", "details": str(timeout_err)}
  except requests.exceptions.RequestException as req_err:
      print(f"An unexpected request error occurred: {req_err}", file=sys.stderr)
      return {"error": "Request Error", "details": str(req_err)}
  except json.JSONDecodeError as json_err:
      print(f"JSON decoding error: {json_err}. Response content: {response.text}", file=sys.stderr)
      return {"error": "JSON Decode Error", "details": str(json_err), "raw_response": response.text}
  except Exception as e:
      print(f"An unexpected error occurred: {e}", file=sys.stderr)
      return {"error": "Unexpected Error", "details": str(e)}
  

r = get_rest_service_data(GET_JDS_URL, auth=(USERNAME, PASSWORD))
for jd in r.get("rest-object", []):

  jdName = jd.get('business-key').removeprefix("JobDefinition:")

  print(f"Job Definition: {jdName},  -  {jd.get('description')}")
  jd_details = get_rest_service_data(f"{JD_DETAILS_URL}{jdName}", headers={"X-API-KEY": RMJ_REST_API_KEY})
  if isinstance(jd_details, dict):
    vectorstore.add_documents([Document(
        page_content=json.dumps(jd_details, indent=2),
        metadata={"source": jdName}
    )])
  else:
    print(f"Failed to fetch details for {jdName}: {jd_details}")

print(f"Store has {len(vectorstore._collection.get()['documents'])} documents.")