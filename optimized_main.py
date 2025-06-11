import logging
import os
import json
from typing import Dict
from embedding import compute_embeddings, normalize_embeddings
from data_process import load_documents, load_queries
from llm_fun import get_chunk_from_llm, get_metadata_from_llm, get_llm_response
from config import config
from optimized_faiss_index import OptimizedHierarchicalFAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


def build_optimized_index_from_documents():
    """Build and save the optimized hierarchical index from documents"""
    # Use reduced metadata dimension for efficiency
    faiss_index = OptimizedHierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=64  # Reduced from full embedding dimension
    )

    logger.info("Loading and preprocessing documents...")
    documents = load_documents(config.DATA_FOLDER)

    if not documents:
        logger.error("No documents loaded. Please check the data folder.")
        return None
    
    os.makedirs(config.LLM_CHUNK, exist_ok=True)
    all_chunks = []
    all_metadata = []
    
    for doc in documents:
        logger.info(f"Processing document: {doc['filename']}")
        
        chunks = get_chunk_from_llm(doc['content'])
        filename = os.path.splitext(doc['filename'])[0]
        out_file = os.path.join(config.LLM_CHUNK, f"{filename}.txt")

        with open(out_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                metadata = get_metadata_from_llm(chunk)

                all_chunks.append(chunk)
                all_metadata.append(metadata)

                f.write(f"-- Chunk {i+1} -- \n")
                f.write(f"Text: \n{chunk}\n\n")
                f.write(f"Metadata:\n{json.dumps(metadata, indent=2)}\n\n")
        
        logger.info(f"Created {len(chunks)} chunks from {doc['filename']}")
    
    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Compute embeddings for chunks (content only)
    logger.info("Computing content embeddings for chunks...")
    chunk_embeddings = []
    for i, chunk in enumerate(all_chunks):
        if i % 100 == 0:
            logger.info(f"Processing chunk {i+1}/{len(all_chunks)}")
        
        chunk_embedding = compute_embeddings(chunk)
        chunk_embedding = normalize_embeddings(chunk_embedding)
        chunk_embeddings.append(chunk_embedding)

    # Build optimized hierarchical FAISS index
    logger.info("Building optimized hierarchical FAISS index...")
    faiss_index.build_hierarchical_index(
        content_embeddings=chunk_embeddings,
        metadata_list=all_metadata,
        chunk_list=all_chunks
    )
    
    # Save the index
    if not os.path.exists(config.INDEX_PATH):
        os.makedirs(config.INDEX_PATH)
    faiss_index.save_index(config.INDEX_PATH)
    
    logger.info("Optimized index built and saved successfully!")
    return faiss_index


def load_optimized_index():
    """Load pre-built optimized hierarchical index"""
    faiss_index = OptimizedHierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=64
    )
    
    try:
        faiss_index.load_index(config.INDEX_PATH)
        logger.info("Successfully loaded existing optimized index!")
        return faiss_index
    except Exception as e:
        logger.error(f"Failed to load existing optimized index: {e}")
        return None


def process_queries_with_optimized_index(faiss_index, queries_with_metadata):
    """Process queries using optimized hierarchical index"""
    
    if not queries_with_metadata:
        logger.warning("No queries with metadata found.")
        return

    logger.info("Processing queries with optimized hierarchical index...")

    for i, query_item in enumerate(queries_with_metadata):
        # Extract query text and metadata
        if isinstance(query_item, dict):
            query = query_item.get('query', '')
            provided_metadata = query_item.get('metadata', {})
        else:
            # Fallback: assume it's just a string query
            query = str(query_item)
            provided_metadata = {}
        
        logger.info(f"\nQuery {i+1}: {query}")
        logger.info(f"Provided metadata: {provided_metadata}")
        
        # Compute query embeddings (content only)
        query_embedding = compute_embeddings(query)
        query_embedding = normalize_embeddings(query_embedding)

        # Use provided metadata or get from LLM as fallback
        if provided_metadata:
            query_metadata = provided_metadata
            logger.info("Using provided metadata for query")
        else:
            logger.info("No metadata provided, generating from LLM...")
            query_metadata = get_metadata_from_llm(query)
        
        # Search the optimized hierarchical index
        # Note: No need for metadata embedding - the system handles it internally
        results = faiss_index.search(
            query_text=query,
            query_embedding=query_embedding,
            query_metadata=query_metadata,
            k=getattr(config, 'TOP_K', 5),
            num_clusters=getattr(config, 'NUM_CLUSTERS', 5)
        )
        
        # Extract context from results
        context_parts = []
        logger.info(f"Found {len(results)} relevant chunks:")
        for score, local_idx, metadata, text in results:
            context_parts.append(f"[Score: {score:.3f}] {text}")
            logger.info(f"  Score: {score:.3f}, Metadata: {metadata}")
        
        context = "\n\n".join(context_parts)
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Get LLM response
        answer = get_llm_response(full_prompt)
        print(f"Question: {query}")
        print(f"Query Metadata: {query_metadata}")
        print(f"Context: {context}")
        print(f"Answer: {answer}\n")
        print("-" * 80)


def load_queries_with_metadata(query_file):
    """Load queries with optional metadata from JSON file"""
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'queries' in data:
            return data['queries']
        else:
            logger.error("Unsupported query file format")
            return []
    
    except json.JSONDecodeError:
        # Fallback: try to read as plain text queries
        logger.info("JSON parsing failed, trying plain text format...")
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                return [{'query': line, 'metadata': {}} for line in lines]
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            return []
    
    except Exception as e:
        logger.error(f"Error loading queries: {e}")
        return []


def check_optimized_index_exists():
    """Check if a pre-built optimized index exists"""
    index_files = [
        os.path.join(config.INDEX_PATH, "optimized_metadata.pkl")
    ]
    return all(os.path.exists(f) for f in index_files)


def main():
    """Main function with optimized hierarchical index"""
    
    # Check if optimized index already exists
    if check_optimized_index_exists():
        logger.info("Found existing optimized index. Loading...")
        faiss_index = load_optimized_index()
        
        if faiss_index is not None:
            # Load queries with metadata
            queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
            
            if queries_with_metadata:
                process_queries_with_optimized_index(faiss_index, queries_with_metadata)
            else:
                logger.warning("No queries loaded. Please check your query file.")
        else:
            logger.error("Failed to load existing optimized index. Rebuilding...")
            faiss_index = build_optimized_index_from_documents()
            if faiss_index:
                queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
                if queries_with_metadata:
                    process_queries_with_optimized_index(faiss_index, queries_with_metadata)
    else:
        logger.info("No existing optimized index found. Building new index...")
        faiss_index = build_optimized_index_from_documents()
        if faiss_index:
            queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
            if queries_with_metadata:
                process_queries_with