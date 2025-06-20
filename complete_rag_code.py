# Complete the missing functions in data_process.py
import json
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_query_metadata(self, file_path: str) -> List[Dict[str, str]]:
    """Extract queries from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    queries.append(item)
                else:
                    queries.append({'query': str(item), 'metadata': {}})
        elif isinstance(data, dict):
            if 'queries' in data:
                queries = data['queries']
            else:
                queries = [data]
        
        return queries
    except Exception as e:
        logger.error(f"Error extracting queries from {file_path}: {e}")
        return []

# Complete the missing functions in llm_fun.py
def get_llm_response_with_context(context: str, query: str) -> str:
    """Get LLM response with provided context (RAG)"""
    prompt = f"""
    You are a helpful medical assistant. Answer the following question based on the provided context.
    
    Context:
    {context}
    
    Question: {query}
    
    Instructions:
    - Use only the information provided in the context
    - If the context doesn't contain enough information, mention that
    - Be precise and accurate in your response
    - Cite relevant parts of the context when answering
    """
    
    sys_prompt = "You are a knowledgeable medical assistant. Provide accurate and helpful responses based on the given context."
    response = get_llm_response(prompt, sys_prompt)
    
    if response:
        return response
    else:
        logger.error("Error in getting answer from llm : Fun(get_llm_response_with_context)")
        return "I apologize, but I'm unable to provide an answer at this time due to a technical issue."

def get_llm_response_without_context(query: str) -> str:
    """Get LLM response without context (baseline)"""
    prompt = f"""
    You are a helpful medical assistant. Answer the following question based on your general knowledge.
    
    Question: {query}
    
    Instructions:
    - Provide a comprehensive answer based on your medical knowledge
    - Be accurate and informative
    - If you're uncertain about something, mention it
    - Keep the response focused and relevant
    """
    
    sys_prompt = "You are a knowledgeable medical assistant with expertise in healthcare topics."
    response = get_llm_response(prompt, sys_prompt)
    
    if response:
        return response
    else:
        logger.error("Error in getting answer from llm : Fun(get_llm_response_without_context)")
        return "I apologize, but I'm unable to provide an answer at this time due to a technical issue."

def get_metadata_from_llm(query: str) -> Dict[str, str]:
    """Extract metadata from query using LLM"""
    prompt = f"""
    Analyze the following medical query and extract metadata information.
    
    Query: {query}
    
    Extract the following metadata (use "unknown" if not specified):
    - Gender: Which gender does this query relate to? (male/female/both/unknown)
    - Age group: What age group is this about? (infant/child/adolescent/adult/elderly/unknown)
    - Keywords: List 5-10 relevant medical keywords (comma-separated)
    - Summary: Brief summary of what the query is asking (max 40 words)
    - Location: Any specific body part or location mentioned (or "unknown")
    
    Format your response as JSON:
    {{
        "gender": "...",
        "age_group": "...",
        "keywords": "...",
        "summary": "...",
        "location": "..."
    }}
    """
    
    sys_prompt = "You are a medical expert specializing in analyzing medical queries and extracting relevant metadata."
    response = get_llm_response(prompt, sys_prompt)
    
    if response:
        try:
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                return metadata
            else:
                logger.warning("No JSON found in LLM response, using default metadata")
                return {
                    "gender": "unknown",
                    "age_group": "unknown", 
                    "keywords": "medical query",
                    "summary": "General medical question",
                    "location": "unknown"
                }
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            return {
                "gender": "unknown",
                "age_group": "unknown",
                "keywords": "medical query", 
                "summary": "General medical question",
                "location": "unknown"
            }
    else:
        logger.error("Error in getting metadata from llm")
        return {
            "gender": "unknown",
            "age_group": "unknown",
            "keywords": "medical query",
            "summary": "General medical question", 
            "location": "unknown"
        }

# Create the missing embedding.py file
# embedding.py
import numpy as np
import requests
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def compute_embeddings(text: str, embedding_dim: int = 512) -> np.ndarray:
    """
    Compute embeddings for given text using an embedding service or model
    This is a placeholder - replace with your actual embedding service
    """
    try:
        # Option 1: Using a REST API service (replace with your actual service)
        # payload = {
        #     "text": text,
        #     "model": "text-embedding-model",
        #     "dimensions": embedding_dim
        # }
        # response = requests.post("YOUR_EMBEDDING_SERVICE_URL", json=payload)
        # if response.status_code == 200:
        #     return np.array(response.json()["embedding"])
        
        # Option 2: Placeholder random embeddings (REPLACE THIS WITH ACTUAL EMBEDDINGS)
        # This is just for demonstration - you need to use a real embedding model
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        logger.info(f"Generated embedding of dimension {embedding_dim} for text length {len(text)}")
        return embedding
        
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        # Return zero vector as fallback
        return np.zeros(embedding_dim, dtype=np.float32)

def normalize_embeddings(embeddings: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Normalize embeddings to unit length"""
    if isinstance(embeddings, list):
        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append(emb / norm)
            else:
                normalized.append(emb)
        return normalized
    else:
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            return embeddings / norm
        else:
            return embeddings

def batch_compute_embeddings(texts: List[str], embedding_dim: int = 512) -> List[np.ndarray]:
    """Compute embeddings for a batch of texts"""
    embeddings = []
    for text in texts:
        embedding = compute_embeddings(text, embedding_dim)
        embeddings.append(embedding)
    return embeddings

# Create the missing config.py file
# config.py
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_FOLDER = os.path.join(BASE_DIR, "data")
    QUERY_FOLDER = os.path.join(BASE_DIR, "queries") 
    LLM_CHUNK = os.path.join(BASE_DIR, "llm_chunks")
    QUERY = os.path.join(BASE_DIR, "query_files")
    INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
    
    # Embedding dimensions
    EMBEDDING_DIM = 512
    CHUNK_EMBEDDING_DIM = 512
    METADATA_EMBEDDING_DIM = 128
    
    # LLM settings
    LLM_API_URL = ""  # Add your LLM API URL here
    LLM_MODEL = "gemini-1.5-flash"
    LLM_MAX_TOKENS = 10000
    LLM_TEMPERATURE = 0.5
    
    # FAISS settings
    FAISS_NUM_CLUSTERS = 50
    FAISS_TOP_K_CLUSTERS = 3
    FAISS_HNSW_M = 16
    FAISS_HNSW_EF_CONSTRUCTION = 64
    
    # Chunking settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

config = Config()

# Fix the XML parsing error in llm_fun.py
def parse_response(response: str) -> List[Dict]:
    """Parse LLM response containing XML chunks"""
    cleaned_response = clean_response(response)
    try:
        root = ET.fromstring(f"<root>{cleaned_response}</root>")  # Fixed: fromstring instead of formstring
        chunks_data = []
        
        for chunk_elem in root.findall('chunk'):
            chunk_id = chunk_elem.get('id', 'unknown')

            text_elem = chunk_elem.find('text')
            chunk_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""

            metadata_elem = chunk_elem.find('metadata')
            metadata = {}
            if metadata_elem is not None:
                for meta_child in metadata_elem:
                    metadata[meta_child.tag] = meta_child.text.strip() if meta_child.text else ""  # Fixed: meta_child.tag instead of meta_child
             
            # Extract queries 
            queries_elem = chunk_elem.find('queries')
            chunk_queries = []
             
            if queries_elem is not None:
                for query_elem in queries_elem.findall('query'):
                    query_id = query_elem.get('id', 'unknown')
                    query_text = query_elem.text.strip() if query_elem.text else ""
                    
                    # Extract query metadata
                    query_metadata_elem = query_elem.find('metadata')
                    query_metadata = {}
                    if query_metadata_elem is not None:
                        for meta_child in query_metadata_elem:
                            query_metadata[meta_child.tag] = meta_child.text.strip() if meta_child.text else ""
                    
                    if query_text:
                        query_data = {
                            'id': query_id,
                            'text': query_text,
                            'chunk_id': chunk_id,
                            'metadata': query_metadata if query_metadata else metadata.copy()
                        }
                        chunk_queries.append(query_data)
             
            # Create chunk data structure 
            chunk_data = {
                'id': chunk_id,
                'text': chunk_text,
                'metadata': metadata,
                'queries': chunk_queries
            }
             
            chunks_data.append(chunk_data)
         
        return chunks_data
         
    except ET.ParseError as e:
        logger.error(f"XML response parsing error: {e}")
        return []

# Fix the main.py issues
def process_queries(faiss_index, queries):
    """Process queries and evaluate RAG performance"""
    if not queries:
        logger.warning("No queries.")
        return
 
    logger.info("Processing queries with provided metadata")
 
    correct_retrievals = 0
    total_queries = len(queries)
    
    for i, query_item in enumerate(queries):
        try:
            if isinstance(query_item, dict):
                query_text = query_item.get('text', query_item.get('query', ''))
                provided_metadata = query_item.get('metadata', {})
                chunk_id = query_item.get('chunk_id', '')
            else:
                query_text = str(query_item)
                provided_metadata = {}
                chunk_id = ''
         
            if not query_text:
                logger.warning(f"Empty query at index {i}")
                continue
                 
            query_embedding = compute_embeddings(query_text, config.CHUNK_EMBEDDING_DIM)
            query_embedding = normalize_embeddings(query_embedding)
 
            if provided_metadata:
                query_metadata = provided_metadata
                logger.info("Using provided metadata for query")
            else:
                logger.info("No metadata provided, generating it from LLM.")
                query_metadata = get_metadata_from_llm(query_text)
         
            metadata_embedding = compute_embeddings(
                metadata_embedding_text(query_metadata), 
                config.METADATA_EMBEDDING_DIM
            )
 
            results = faiss_index.search(
                query_text, query_embedding, query_metadata, metadata_embedding
            )
         
            context_parts = []
            retrieved_chunk_ids = []
            
            logger.info(f"Found {len(results)} relevant chunks:")
            for score, local_idx, metadata, text in results:
                context_parts.append(f"[Score: {score:.3f}] {text}")
                # Extract chunk ID from metadata or use local_idx
                result_chunk_id = metadata.get('chunk_id', f'chunk_{local_idx}')
                retrieved_chunk_ids.append(result_chunk_id)
         
            context = "\n\n".join(context_parts)
 
            # Check if correct chunk was retrieved
            if chunk_id and chunk_id in retrieved_chunk_ids:
                correct_retrievals += 1
         
            ans_rag = get_llm_response_with_context(context, query_text)
            ans_no_rag = get_llm_response_without_context(query_text)
 
            print(f"Question {i+1}: \n{query_text}\n")
            print(f"Query Metadata: \n{query_metadata}\n")
            print(f"Context Extracted: \n{context}\n")
            print(f"Answer with RAG: \n{ans_rag}\n")
            print(f"Answer without RAG: \n{ans_no_rag}\n")
            print("-" * 150)
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
            continue
    
    # Calculate and print retrieval accuracy
    if total_queries > 0:
        accuracy = correct_retrievals / total_queries
        print(f"\nRetrieval Accuracy: {correct_retrievals}/{total_queries} = {accuracy:.2%}")

# Fix the load_queries function in main.py to avoid conflict
def load_queries_from_file(query_file):
    """Load queries from file - renamed to avoid conflict"""
    try:
        if not os.path.exists(query_file):
            logger.error(f"Query file not found: {query_file}")
            return []
            
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
         
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'queries' in data:
            return data['queries']
        else:
            logger.error("Unsupported query file format")
            return []
     
    except json.JSONDecodeError:
        logger.info("JSON parsing failed, trying plain text format")
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                return [{'text': line, 'metadata': {}} for line in lines]
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            return []
     
    except Exception as e:
        logger.error(f"Error loading queries: {e}")
        return []