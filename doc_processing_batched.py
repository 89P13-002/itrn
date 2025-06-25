from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken  # For accurate token counting

def get_token_count(text, model="gpt-3.5-turbo"):
    """
    Get accurate token count for a given text and model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to rough estimation if tiktoken fails
        return len(text) // 4

def create_document_splitter(chunk_size=6000, chunk_overlap=200, model="gpt-3.5-turbo"):
    """
    Create a RecursiveCharacterTextSplitter optimized for LLM processing.
    
    Args:
        chunk_size (int): Maximum tokens per chunk
        chunk_overlap (int): Number of tokens to overlap between chunks
        model (str): Model name for accurate token counting
    
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    
    # Custom length function that counts tokens instead of characters
    def token_len_function(text):
        return get_token_count(text, model)
    
    # Separators optimized for document processing
    separators = [
        "\n\n\n",  # Multiple newlines (section breaks)
        "\n\n",    # Double newlines (paragraph breaks)
        "\n",      # Single newlines
        ". ",      # Sentence endings
        "! ",      # Exclamation sentences
        "? ",      # Question sentences
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Spaces
        ""         # Character level (last resort)
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len_function,
        separators=separators,
        keep_separator=True,  # Keep separators to maintain context
        add_start_index=True  # Track original position in document
    )
    
    return splitter

def process_document_with_langchain(doc_content, filename, batch_processor_func, 
                                   max_chunk_tokens=6000, overlap_tokens=200):
    """
    Process a document using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        doc_content (str): Full document content
        filename (str): Document filename for logging
        batch_processor_func: Function to process each chunk
        max_chunk_tokens (int): Maximum tokens per chunk
        overlap_tokens (int): Overlap between chunks
    
    Returns:
        list: Combined chunks and data from all document parts
    """
    
    # Create text splitter
    splitter = create_document_splitter(
        chunk_size=max_chunk_tokens,
        chunk_overlap=overlap_tokens
    )
    
    # Split the document
    try:
        # Create a Document object for LangChain
        from langchain.schema import Document
        doc = Document(page_content=doc_content, metadata={"source": filename})
        
        # Split the document
        split_docs = splitter.split_documents([doc])
        logger.info(f"Split document '{filename}' into {len(split_docs)} chunks using LangChain")
        
    except ImportError:
        # Fallback to text splitting if Document import fails
        split_texts = splitter.split_text(doc_content)
        logger.info(f"Split document '{filename}' into {len(split_texts)} chunks using LangChain (text mode)")
        
        # Convert to document-like structure
        split_docs = []
        for i, text in enumerate(split_texts):
            split_docs.append({
                'page_content': text,
                'metadata': {'source': filename, 'chunk_index': i}
            })
    
    all_chunks_data = []
    
    for i, doc_chunk in enumerate(split_docs):
        logger.info(f"Processing chunk {i+1}/{len(split_docs)} for {filename}")
        
        # Extract content based on whether it's a Document object or dict
        if hasattr(doc_chunk, 'page_content'):
            chunk_content = doc_chunk.page_content
            chunk_metadata = doc_chunk.metadata
        else:
            chunk_content = doc_chunk['page_content']
            chunk_metadata = doc_chunk['metadata']
        
        try:
            # Process the chunk through LLM
            chunk_data = batch_processor_func(chunk_content)
            
            # Enhance metadata for each chunk
            for chunk_item in chunk_data:
                if 'metadata' not in chunk_item:
                    chunk_item['metadata'] = {}
                
                # Add LangChain splitting metadata
                chunk_item['metadata'].update({
                    'langchain_chunk_index': i,
                    'total_langchain_chunks': len(split_docs),
                    'source_document': filename,
                    'chunk_token_count': get_token_count(chunk_content),
                    'original_start_index': chunk_metadata.get('start_index', None)
                })
                
                # Preserve any existing metadata from LangChain
                if 'start_index' in chunk_metadata:
                    chunk_item['metadata']['document_start_index'] = chunk_metadata['start_index']
            
            all_chunks_data.extend(chunk_data)
            
        except Exception as e:
            logger.error(f"Error processing LangChain chunk {i+1} of {filename}: {e}")
            logger.error(f"Chunk content preview: {chunk_content[:200]}...")
            # Continue with other chunks even if one fails
            continue
    
    return all_chunks_data

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
    
    # Configuration for document splitting
    MAX_CHUNK_TOKENS = getattr(config, 'MAX_LLM_TOKENS', 6000)  # Configurable max tokens
    OVERLAP_TOKENS = getattr(config, 'CHUNK_OVERLAP_TOKENS', 200)  # Configurable overlap
    
    for doc in documents:
        logger.info(f"Processing document: {doc['filename']}")
        
        # Process document using LangChain recursive splitter
        chunks_and_data = process_document_with_langchain(
            doc['content'], 
            doc['filename'], 
            get_chunk_metadata_from_llm,
            max_chunk_tokens=MAX_CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS
        )
        
        filename = os.path.splitext(doc['filename'])[0]
        chunk_metadata_file = os.path.join(config.LLM_CHUNK, f"{filename}.txt")
        query_file = os.path.join(config.QUERY, f"{filename}.json")
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
            
            qry = chk.get('queries', [])  # Handle missing queries gracefully
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
    
    # Process embeddings in batches to avoid memory issues
    chunk_embeddings = []
    batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 50)  # Configurable batch size
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        logger.info(f"Computing embeddings for chunk batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
        
        for chunk in batch_chunks:
            try:
                chunk_embedding = compute_embeddings(chunk, config.CHUNK_EMBEDDING_DIM)
                chunk_embeddings.append(chunk_embedding)
            except Exception as e:
                logger.error(f"Error computing embedding for chunk: {e}")
                # Add zero vector as fallback
                zero_embedding = [0.0] * config.CHUNK_EMBEDDING_DIM
                chunk_embeddings.append(zero_embedding)
    
    logger.info("Computing metadata embeddings")
    metadata_embeddings = []
    
    for i in range(0, len(all_metadata), batch_size):
        batch_metadata = all_metadata[i:i + batch_size]
        logger.info(f"Computing metadata embeddings for batch {i//batch_size + 1}/{(len(all_metadata) + batch_size - 1)//batch_size}")
        
        for metadata in batch_metadata:
            try:
                metadata_embedding = compute_embeddings(
                    metadata_embedding_text(metadata), 
                    config.METADATA_EMBEDDING_DIM
                )
                metadata_embeddings.append(metadata_embedding)
            except Exception as e:
                logger.error(f"Error computing metadata embedding: {e}")
                # Add zero vector as fallback
                zero_embedding = [0.0] * config.METADATA_EMBEDDING_DIM
                metadata_embeddings.append(zero_embedding)
    
    logger.info("Building hierarchical FAISS index.")
    faiss_index.build_hierarchical_index(
        chunk_embeddings, 
        metadata_embeddings, 
        all_chunks, 
        all_metadata, 
        all_chunk_ids
    )
    
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