def process_queries_with_metadata(faiss_index, queries_with_metadata):
    """Process queries with provided metadata using existing index"""
    
    if not queries_with_metadata:
        logger.warning("No queries with metadata found.")
        return

    logger.info("Processing queries with provided metadata...")

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
        
        # Compute query embeddings
        query_embedding = compute_embeddings(query)
        query_embedding = normalize_embeddings(query_embedding)

        # Use provided metadata or get from LLM as fallback
        if provided_metadata:
            query_metadata = provided_metadata
            logger.info("Using provided metadata for query")
        else:
            logger.info("No metadata provided, generating from LLM...")
            query_metadata = get_metadata_from_llm(query)
        
        # Compute metadata embedding
        metadata_embedding = compute_embeddings(metadata_embedding_text(query_metadata))
        metadata_embedding = normalize_embeddings(metadata_embedding)

        # Search the hierarchical index
        results = faiss_index.search(query, query_embedding, query_metadata, metadata_embedding)
        
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


def main():
    """Main function with enhanced query metadata handling"""
    
    # Check if index already exists
    if check_index_exists():
        logger.info("Found existing index. Loading...")
        faiss_index = load_existing_index()
        
        if faiss_index is not None:
            # Load queries with metadata
            queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
            
            if queries_with_metadata:
                process_queries_with_metadata(faiss_index, queries_with_metadata)
            else:
                logger.warning("No queries loaded. Please check your query file.")
        else:
            logger.error("Failed to load existing index. Rebuilding...")
            faiss_index = build_index_from_documents()
            if faiss_index:
                queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
                if queries_with_metadata:
                    process_queries_with_metadata(faiss_index, queries_with_metadata)
    else:
        logger.info("No existing index found. Building new index...")
        faiss_index = build_index_from_documents()
        if faiss_index:
            queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
            if queries_with_metadata:
                process_queries_with_metadata(faiss_index, queries_with_metadata)
    
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
        # Only run queries with existing index
        if check_index_exists():
            faiss_index = load_existing_index()
            if faiss_index:
                queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
                if queries_with_metadata:
                    process_queries_with_metadata(faiss_index, queries_with_metadata)
                else:
                    logger.error("No queries loaded!")
            else:
                logger.error("Failed to load existing index!")
        else:
            logger.error("No existing index found! Use --rebuild to create one.")
    
    elif args.build_only:
        # Only build index
        logger.info("Building index only...")
        build_index_from_documents()
    
    elif args.rebuild:
        # Force rebuild
        logger.info("Force rebuilding index...")
        faiss_index = build_index_from_documents()
        if faiss_index:
            queries_with_metadata = load_queries_with_metadata(config.QUERY_FILE)
            if queries_with_metadata:
                process_queries_with_metadata(faiss_index, queries_with_metadata)
    
    else:
        # Default behavior: smart reuse
        main()