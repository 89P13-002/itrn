# main.py
import logging
import os
import json
from typing import Dict, List
from embedding import compute_embeddings, normalize_embeddings
from data_process import load_documents, load_queries
from llm_fun import get_chunk_from_llm, get_metadata_from_llm, get_llm_response, get_chunk_metadata_from_llm, get_llm_response_with_context, get_llm_response_without_context
from config import config
from faiss_index import HierarchicalFAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

import sys
log_file = open("output.log", "w")
sys.stdout = log_file

# Global counters for unique IDs
chunk_counter = 0
query_counter = 0

def reset_counters():
    """Reset global counters (useful for testing)"""
    global chunk_counter, query_counter
    chunk_counter = 0
    query_counter = 0

def get_next_chunk_id():
    """Get next unique chunk ID"""
    global chunk_counter
    chunk_counter += 1
    return f"chunk_{chunk_counter}"

def get_next_query_id():
    """Get next unique query ID"""
    global query_counter
    query_counter += 1
    return f"query_{query_counter}"

def metadata_embedding_text(metadata: Dict[str, str]) -> str:
    text_parts = []
    
    if metadata.get('gender'):
        text_parts.append(f"{metadata['gender']}")
    if metadata.get('age_group'):
        text_parts.append(f"{metadata['age_group']}")
    if metadata.get('age_range'):
        text_parts.append(f"{metadata['age_range']}")  
    if metadata.get('keywords'):
        text_parts.append(f"{metadata['keywords']}")
    if metadata.get('summary'):
        summary = metadata['summary'][:200] if len(metadata['summary']) > 200 else metadata['summary']
        text_parts.append(f"{summary}")
    if metadata.get('location'):
        text_parts.append(f"{metadata['location']}")   
    return " | ".join(text_parts) if text_parts else "general content"


def build_index_from_documents():
    faiss_index = HierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=config.METADATA_EMBEDDING_DIM 
    )

    logger.info("Loading and preprocessing documents.")
    documents = load_documents(config.DATA_FOLDER)

    if not documents:
        logger.error("No documents ? Please check the data folder.")
        return None
    
    os.makedirs(config.LLM_CHUNK, exist_ok=True)
    os.makedirs(config.QUERY, exist_ok=True)
    all_chunks = []
    all_metadata = []
    all_chunk_ids = []  # Store chunk IDs for tracking
    
    for doc in documents:
        logger.info(f"Processing document: {doc['filename']}")
        
        chunks_and_data = get_chunk_metadata_from_llm(doc['content'])
        filename = os.path.splitext(doc['filename'])[0]
        chunk_metadata_file = os.path.join(config.LLM_CHUNK, f"{filename}.txt")
        query_file = os.path.join(config.LLM_CHUNK, f"{filename}.json")

        chunks = []
        queries = []
        for chk in chunks_and_data:
            chunk_id = get_next_chunk_id()  # Generate unique chunk ID
            
            chk_mtd = {
                'chunk_id': chunk_id,
                'chunk_text': chk['text'],
                'chunk_metadata': chk['metadata']
            }
            chunks.append(chk_mtd)
            all_chunks.append(chk['text'])
            all_metadata.append(chk['metadata'])
            all_chunk_ids.append(chunk_id)

            qry = chk['queries']
            for q in qry:
                query_id = get_next_query_id()  # Generate unique query ID
                query_item = {
                    'query_id': query_id,
                    'query': q,
                    'chunk_id': chunk_id,  # Reference to the source chunk
                    'metadata': chk['metadata']
                }
                queries.append(query_item)
            
        with open(chunk_metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        with open(query_file, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(chunks_and_data)} chunks from {doc['filename']}")
    
    logger.info(f"Total chunks created: {len(all_chunks)}")

    logger.info("Computing embeddings for chunks.")
    chunk_embeddings = []
    for chunk in all_chunks:
        chunk_embedding = compute_embeddings(chunk, config.CHUNK_EMBEDDING_DIM)
        chunk_embeddings.append(chunk_embedding)

    logger.info("Computing metadata embeddings")
    metadata_embeddings = []
    for metadata in all_metadata:
        metadata_embedding = compute_embeddings(metadata_embedding_text(metadata), config.METADATA_EMBEDDING_DIM)
        metadata_embeddings.append(metadata_embedding)

    logger.info("Building hierarchical FAISS index.")
    faiss_index.build_hierarchical_index(chunk_embeddings, metadata_embeddings, all_chunks, all_metadata, all_chunk_ids)
    
    if not os.path.exists(config.INDEX_PATH):
        os.makedirs(config.INDEX_PATH)
    faiss_index.save_index(config.INDEX_PATH)
    
    logger.info("Indexing done /--\ ")
    return faiss_index


def load_existing_index():
    faiss_index = HierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=config.METADATA_EMBEDDING_DIM
    )
    
    try:
        faiss_index.load_index(config.INDEX_PATH)
        return faiss_index
    except Exception as e:
        logger.error(f"Failed to load existing index: {e}")
        return None


def check_index_exists():
    index_files = [
        os.path.join(config.INDEX_PATH, "metadata_clusters.faiss"),
        os.path.join(config.INDEX_PATH, "metadata_index.pkl")
    ]
    return all(os.path.exists(f) for f in index_files)

def process_queries(faiss_index, queries):
    if not queries:
        logger.warning("No queries.")
        return

    logger.info("Processing queries with provided metadata")

    correct_retrievals = 0
    total_queries = len(queries)
    
    for i, query_item in enumerate(queries):
        if isinstance(query_item, dict):
            query_text = query_item.get('query', '')
            provided_metadata = query_item.get('metadata', {})
            expected_chunk_id = query_item.get('chunk_id', '')
            query_id = query_item.get('query_id', f'query_{i+1}')
        else:
            query_text = str(query_item)
            provided_metadata = {}
            expected_chunk_id = ''
            query_id = f'query_{i+1}'
        
        query_embedding = compute_embeddings(query_text, 512)
        query_embedding = normalize_embeddings(query_embedding)

        if provided_metadata:
            query_metadata = provided_metadata
            logger.info("Using provided metadata for query")
        else:
            logger.info("No metadata provided, generating it from LLM.")
            query_metadata = get_metadata_from_llm(query_text)
        
        metadata_embedding = compute_embeddings(metadata_embedding_text(query_metadata), config.METADATA_EMBEDDING_DIM)

        results = faiss_index.search(query_text, query_embedding, query_metadata, metadata_embedding)
        
        context_parts = []
        retrieved_chunk_ids = []
        logger.info(f"Found {len(results)} relevant chunks:")
        
        for score, local_idx, metadata, text, chunk_id in results:
            context_parts.append(f"[Score: {score:.3f}] {text}")
            retrieved_chunk_ids.append(chunk_id)
        
        context = "\n\n".join(context_parts)

        # Check if the expected chunk ID is in the retrieved results
        retrieval_score = 1 if expected_chunk_id in retrieved_chunk_ids else 0
        correct_retrievals += retrieval_score

        ans_rag = get_llm_response_with_context(context, query_text)
        ans_no_rag = get_llm_response_without_context(query_text)

        print(f"Query ID: {query_id}")
        print(f"Question Asked: \n{query_text}\n")
        print(f"Expected Chunk ID: {expected_chunk_id}")
        print(f"Retrieved Chunk IDs: {retrieved_chunk_ids}")
        print(f"Retrieval Score: {retrieval_score}")
        print(f"Query Metadata: \n{query_metadata}\n")
        print(f"Context Extracted: \n{context}\n")
        print(f"Answer with RAG: \n{ans_rag}\n")
        print(f"Answer without RAG: \n{ans_no_rag}\n\n")
        print("-" * 150)

    # Calculate and print final accuracy
    accuracy = correct_retrievals / total_queries if total_queries > 0 else 0
    print(f"\nRetrieval Accuracy: {correct_retrievals}/{total_queries} = {accuracy:.3f}")
    logger.info(f"Retrieval Accuracy: {correct_retrievals}/{total_queries} = {accuracy:.3f}")


def load_all_queries():
    """Load queries from all files returned by data_process load_queries function"""
    try:
        # Get all query files from data_process load_queries function
        query_files = load_queries()  # This should return list of query files
        
        all_queries = []
        
        for query_file in query_files:
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    all_queries.extend(data)
                elif isinstance(data, dict) and 'queries' in data:
                    all_queries.extend(data['queries'])
                else:
                    logger.warning(f"Unsupported query file format in {query_file}")
                    
            except json.JSONDecodeError:
                logger.info(f"JSON parsing failed for {query_file}, trying plain text format")
                try:
                    with open(query_file, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        text_queries = [{'query': line, 'metadata': {}} for line in lines]
                        all_queries.extend(text_queries)
                except Exception as e:
                    logger.error(f"Failed to load queries from {query_file}: {e}")
            
            except Exception as e:
                logger.error(f"Error loading queries from {query_file}: {e}")
        
        logger.info(f"Loaded {len(all_queries)} queries from {len(query_files)} files")
        return all_queries
    
    except Exception as e:
        logger.error(f"Error in load_all_queries: {e}")
        return []


def main():
    if check_index_exists():
        logger.info("Found existing index. Loading")
        faiss_index = load_existing_index()
        
        if faiss_index is not None:
            query_doc = load_all_queries()
            
            if query_doc:
                process_queries(faiss_index, query_doc)
            else:
                logger.warning("No queries loaded. Please check your query folder.")
        else:
            logger.error("Failed to load existing index. Rebuilding")
            faiss_index = build_index_from_documents()
            if faiss_index:
                query_doc = load_all_queries()
                if query_doc:
                    process_queries(faiss_index, query_doc)
    else:
        logger.info("No existing index found. Building new index.")
        faiss_index = build_index_from_documents()
        if faiss_index:
            query_doc = load_all_queries()
            if query_doc:
                process_queries(faiss_index, query_doc)
    
    logger.info("\nRAG with hierarchical metadata clustering completed\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical RAG System')
    parser.add_argument('--rebuild', action='store_true', 
                       help='Force rebuild index even if it exists')
    parser.add_argument('--query-only', action='store_true',
                       help='Only run queries (requires existing index)')
    parser.add_argument('--build-only', action='store_true',
                       help='Only build index (skip queries)')
    
    args = parser.parse_args()
    
    if args.query_only:
        if check_index_exists():
            faiss_index = load_existing_index()
            if faiss_index:
                query_doc = load_all_queries()
                if query_doc:
                    process_queries(faiss_index, query_doc)
                else:
                    logger.error("No queries loaded!")
            else:
                logger.error("Failed to load existing index!")
        else:
            logger.error("No existing index found! Use --rebuild to create one.")
    
    elif args.build_only:
        logger.info("Building index only...")
        build_index_from_documents()
    
    elif args.rebuild:
        logger.info("Force rebuilding index...")
        faiss_index = build_index_from_documents()
        if faiss_index:
            queries_with_metadata = load_all_queries()
            if queries_with_metadata:
                process_queries(faiss_index, queries_with_metadata)
    
    else:
        main()
