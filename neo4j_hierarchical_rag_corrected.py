import json
from typing import List, Dict, Optional, Any, Tuple
import requests
from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import util

class EmbeddingFunction:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "texts": texts,
            "model": "text-embedding-005",
            "config": {
                "task_type": "RETRIEVAL_QUERY",
                "auto_truncate": True,
                "dimension": 768
            }
        }
        
        response = requests.post(url=self.api_url, json=payload, verify=False)
        
        if response.status_code == 200:
            embeddings = response.json().get("embeddings", [])
            return [[float(x) for x in emb] for emb in embeddings]
        raise Exception(f"Embedding error: {response.status_code}")
    
    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

class MetadataExtractor:
    def __init__(self, llm_inference_api_url: str):
        self.llm_inference_api_url = llm_inference_api_url
    
    def extract_metadata(self, text_chunk: str) -> Dict[str, Any]:
        """Extract structured metadata from text chunk using LLM"""
        system_prompt = """You are a metadata extraction assistant. Analyze the given text and extract structured metadata.
        Return a JSON object with the following fields:
        - topic: main topic/subject (string)
        - category: document category (e.g., "technical", "business", "legal", "academic", etc.)
        - keywords: list of important keywords (list of strings, max 5)
        - domain: subject domain (e.g., "technology", "finance", "healthcare", etc.)
        - document_type: type of document (e.g., "report", "article", "manual", "research", etc.)
        - summary: brief summary in 1-2 sentences (string)
        - tags: additional descriptive tags (list of strings, max 3)
        
        Return only valid JSON without any additional text."""
        
        user_prompt = f"Extract metadata from this text:\n\n{text_chunk[:2000]}..."  # Limit text length
        
        payload = {
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            },
            "model": {
                "name": "gemini-1.5-flash",
                "top_p": 0.1,
                "temperature": 0.3,
                "max_tokens": 500,
                "do_sample": True,
                "repetition_penalty": 0,
                "json_mode": True
            }
        }
        
        try:
            response = requests.post(url=self.llm_inference_api_url, json=payload, verify=False)
            if response.status_code == 200:
                metadata_str = response.json().get("response", "{}")
                return json.loads(metadata_str)
        except Exception as e:
            print(f"Metadata extraction error: {e}")
        
        # Fallback metadata
        return {
            "topic": "unknown",
            "category": "general",
            "keywords": [],
            "domain": "general",
            "document_type": "document",
            "summary": "No summary available",
            "tags": []
        }
    
    def metadata_to_text(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata dictionary to text for embedding"""
        text_parts = []
        
        if metadata.get('topic'):
            text_parts.append(f"Topic: {metadata['topic']}")
        
        if metadata.get('category'):
            text_parts.append(f"Category: {metadata['category']}")
        
        if metadata.get('domain'):
            text_parts.append(f"Domain: {metadata['domain']}")
        
        if metadata.get('document_type'):
            text_parts.append(f"Type: {metadata['document_type']}")
        
        if metadata.get('keywords'):
            text_parts.append(f"Keywords: {', '.join(metadata['keywords'])}")
        
        if metadata.get('tags'):
            text_parts.append(f"Tags: {', '.join(metadata['tags'])}")
        
        if metadata.get('summary'):
            text_parts.append(f"Summary: {metadata['summary']}")
        
        return ". ".join(text_parts)

class Neo4jVectorDB:
    def __init__(self, uri: str, username: str, password: str, embedding_function: EmbeddingFunction):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_function = embedding_function
        self._create_indexes()
    
    def close(self):
        self.driver.close()
    
    def _create_indexes(self):
        """Create necessary indexes and constraints"""
        with self.driver.session() as session:
            try:
                # Check Neo4j version to determine vector index syntax
                version_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
                version_record = version_result.single()
                neo4j_version = version_record['version'] if version_record else "5.0"
                
                # For Neo4j 5.x+, use the new vector index syntax
                if version_record and float(neo4j_version.split('.')[0]) >= 5:
                    # Create vector index for document content embeddings
                    session.run("""
                        CREATE VECTOR INDEX document_content_embeddings IF NOT EXISTS
                        FOR (d:Document) ON (d.content_embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    
                    # Create vector index for cluster metadata embeddings
                    session.run("""
                        CREATE VECTOR INDEX cluster_metadata_embeddings IF NOT EXISTS
                        FOR (c:Cluster) ON (c.metadata_embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                else:
                    # For older versions, create point indexes (fallback)
                    print("Warning: Using fallback indexes for older Neo4j version")
                    session.run("CREATE INDEX document_content_idx IF NOT EXISTS FOR (d:Document) ON (d.content)")
                    session.run("CREATE INDEX cluster_metadata_idx IF NOT EXISTS FOR (c:Cluster) ON (c.metadata_text)")
                
                # Create regular indexes
                session.run("CREATE INDEX cluster_topic_idx IF NOT EXISTS FOR (c:Cluster) ON (c.topic)")
                session.run("CREATE INDEX cluster_category_idx IF NOT EXISTS FOR (c:Cluster) ON (c.category)")
                session.run("CREATE INDEX cluster_domain_idx IF NOT EXISTS FOR (c:Cluster) ON (c.domain)")
                session.run("CREATE INDEX document_chunk_id_idx IF NOT EXISTS FOR (d:Document) ON (d.chunk_id)")
                session.run("CREATE INDEX document_cluster_id_idx IF NOT EXISTS FOR (d:Document) ON (d.cluster_id)")
                
            except Exception as e:
                print(f"Index creation warning: {e}")

    def _vector_similarity_search(self, session, index_name: str, query_embedding: List[float], 
                                top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search with fallback for different Neo4j versions"""
        try:
            # Try modern vector search first
            if index_name == 'document_content_embeddings':
                cypher = """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node, score
                    ORDER BY score DESC
                """
            else:  # cluster_metadata_embeddings
                cypher = """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node, score
                    ORDER BY score DESC
                """
            
            result = session.run(cypher, 
                               index_name=index_name, 
                               top_k=top_k, 
                               query_embedding=query_embedding)
            
            return [(record['node'], record['score']) for record in result]
            
        except Exception as e:
            print(f"Vector search failed, using fallback cosine similarity: {e}")
            # Fallback to manual calculation
            return self._manual_cosine_similarity(session, query_embedding, top_k, index_name)
    
    def _manual_cosine_similarity(self, session, query_embedding: List[float], top_k: int, index_type: str):
        """Manual cosine similarity calculation as fallback"""
        if index_type == 'document_content_embeddings':
            cypher = "MATCH (n:Document) WHERE n.content_embedding IS NOT NULL RETURN n, n.content_embedding as embedding"
        else:
            cypher = "MATCH (n:Cluster) WHERE n.metadata_embedding IS NOT NULL RETURN n, n.metadata_embedding as embedding"
        
        result = session.run(cypher)
        
        similarities = []
        for record in result:
            node = record['n']
            embedding = record['embedding']
            if embedding:
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in embedding) ** 0.5
                
                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    similarities.append((node, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def create_or_get_cluster(self, cluster_metadata: Dict[str, Any], metadata_extractor: MetadataExtractor) -> str:
        """Create a cluster node based on metadata with embedding"""
        # Create a more unique cluster ID to avoid conflicts
        cluster_id = f"{cluster_metadata['category']}_{cluster_metadata['domain']}_{cluster_metadata['topic']}".replace(" ", "_").lower()
        
        # Convert metadata to text and create embedding
        metadata_text = metadata_extractor.metadata_to_text(cluster_metadata)
        metadata_embedding = self.embedding_function.embed_query(metadata_text)
        
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Cluster {id: $cluster_id})
                SET c.topic = $topic,
                    c.category = $category,
                    c.domain = $domain,
                    c.keywords = $keywords,
                    c.document_type = $document_type,
                    c.tags = $tags,
                    c.metadata_text = $metadata_text,
                    c.metadata_embedding = $metadata_embedding,
                    c.document_count = COALESCE(c.document_count, 0),
                    c.created_at = datetime(),
                    c.updated_at = datetime()
            """, 
            cluster_id=cluster_id,
            topic=cluster_metadata['topic'],
            category=cluster_metadata['category'],
            domain=cluster_metadata['domain'],
            keywords=cluster_metadata['keywords'],
            document_type=cluster_metadata['document_type'],
            tags=cluster_metadata.get('tags', []),
            metadata_text=metadata_text,
            metadata_embedding=metadata_embedding
            )
        
        return cluster_id
    
    def find_similar_clusters(self, document_metadata: Dict[str, Any], metadata_extractor: MetadataExtractor, 
                            max_clusters: int = 3, similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find clusters similar to document metadata using vector similarity"""
        
        # Convert document metadata to embedding
        doc_metadata_text = metadata_extractor.metadata_to_text(document_metadata)
        doc_metadata_embedding = self.embedding_function.embed_query(doc_metadata_text)
        
        with self.driver.session() as session:
            # Use vector similarity search to find similar clusters
            similar_nodes = self._vector_similarity_search(
                session, 
                'cluster_metadata_embeddings', 
                doc_metadata_embedding, 
                max_clusters * 2  # Get more candidates to filter by threshold
            )
            
            similar_clusters = []
            for node, score in similar_nodes:
                if score >= similarity_threshold and len(similar_clusters) < max_clusters:
                    # Extract cluster_id from node properties
                    cluster_id = node.get('id', node.get('cluster_id', ''))
                    if cluster_id:
                        similar_clusters.append((cluster_id, score))
            
            return similar_clusters
    
    def add_document_to_clusters(self, cluster_ids: List[str], chunk_text: str, chunk_metadata: Dict[str, Any], 
                               chunk_id: int, similarity_scores: List[float] = None):
        """Add a document chunk to multiple clusters"""
        content_embedding = self.embedding_function.embed_query(chunk_text)
        
        if similarity_scores is None:
            similarity_scores = [1.0] * len(cluster_ids)
        
        with self.driver.session() as session:
            for i, cluster_id in enumerate(cluster_ids):
                similarity_score = similarity_scores[i] if i < len(similarity_scores) else 1.0
                
                # Create unique document ID to avoid conflicts
                doc_id = f"doc_{chunk_id}_{cluster_id}"
                
                session.run("""
                    MATCH (c:Cluster {id: $cluster_id})
                    CREATE (d:Document {
                        id: $doc_id,
                        chunk_id: $chunk_id,
                        cluster_id: $cluster_id,
                        content: $content,
                        content_embedding: $content_embedding,
                        topic: $topic,
                        category: $category,
                        domain: $domain,
                        keywords: $keywords,
                        document_type: $document_type,
                        summary: $summary,
                        tags: $tags,
                        created_at: datetime()
                    })
                    CREATE (d)-[:BELONGS_TO {similarity_score: $similarity_score, created_at: datetime()}]->(c)
                    SET c.document_count = c.document_count + 1,
                        c.updated_at = datetime()
                """,
                doc_id=doc_id,
                cluster_id=cluster_id,
                chunk_id=chunk_id,
                content=chunk_text,
                content_embedding=content_embedding,
                topic=chunk_metadata['topic'],
                category=chunk_metadata['category'],
                domain=chunk_metadata['domain'],
                keywords=chunk_metadata['keywords'],
                document_type=chunk_metadata['document_type'],
                summary=chunk_metadata['summary'],
                tags=chunk_metadata.get('tags', []),
                similarity_score=similarity_score
                )
    
    def find_relevant_clusters_by_embedding(self, query_metadata: Dict[str, Any], metadata_extractor: MetadataExtractor, 
                                          top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant clusters using metadata embedding similarity"""
        
        # Convert query metadata to embedding
        query_metadata_text = metadata_extractor.metadata_to_text(query_metadata)
        query_metadata_embedding = self.embedding_function.embed_query(query_metadata_text)
        
        with self.driver.session() as session:
            similar_nodes = self._vector_similarity_search(
                session, 
                'cluster_metadata_embeddings', 
                query_metadata_embedding, 
                top_k
            )
            
            clusters = []
            for node, score in similar_nodes:
                clusters.append({
                    'cluster_id': node.get('id', ''),
                    'topic': node.get('topic', ''),
                    'category': node.get('category', ''),
                    'domain': node.get('domain', ''),
                    'keywords': node.get('keywords', []),
                    'tags': node.get('tags', []),
                    'document_count': node.get('document_count', 0),
                    'metadata_text': node.get('metadata_text', ''),
                    'similarity_score': score
                })
            
            return clusters
    
    def search_within_cluster(self, cluster_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents within a specific cluster using content embeddings"""
        with self.driver.session() as session:
            # Get all documents in the cluster first
            cluster_docs_result = session.run("""
                MATCH (d:Document)-[r:BELONGS_TO]->(c:Cluster {id: $cluster_id})
                RETURN d, r.similarity_score as cluster_similarity
            """, cluster_id=cluster_id)
            
            # Calculate similarities manually if vector search fails
            documents = []
            for record in cluster_docs_result:
                doc = record['d']
                doc_embedding = doc.get('content_embedding', [])
                
                if doc_embedding and query_embedding:
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    norm_a = sum(a * a for a in query_embedding) ** 0.5
                    norm_b = sum(b * b for b in doc_embedding) ** 0.5
                    
                    content_similarity = 0.0
                    if norm_a > 0 and norm_b > 0:
                        content_similarity = dot_product / (norm_a * norm_b)
                    
                    cluster_similarity = record['cluster_similarity']
                    combined_score = cluster_similarity * 0.3 + content_similarity * 0.7
                    
                    documents.append({
                        'chunk_id': doc.get('chunk_id', 0),
                        'content': doc.get('content', ''),
                        'topic': doc.get('topic', ''),
                        'category': doc.get('category', ''),
                        'domain': doc.get('domain', ''),
                        'summary': doc.get('summary', ''),
                        'tags': doc.get('tags', []),
                        'cluster_similarity': cluster_similarity,
                        'content_similarity': content_similarity,
                        'combined_score': combined_score
                    })
            
            # Sort by combined score and return top k
            documents.sort(key=lambda x: x['combined_score'], reverse=True)
            return documents[:top_k]

class HierarchicalRAGPipeline:
    def __init__(self, embedding_api_url: str, llm_inference_api_url: str, 
                 neo4j_uri: str, neo4j_username: str, neo4j_password: str,
                 max_clusters_per_document: int = 2, cluster_similarity_threshold: float = 0.7):
        """
        Initialize hierarchical RAG pipeline
        
        Args:
            max_clusters_per_document: Maximum number of clusters a document can belong to
            cluster_similarity_threshold: Minimum similarity score for cluster assignment
        """
        self.embedding_function = EmbeddingFunction(embedding_api_url)
        self.metadata_extractor = MetadataExtractor(llm_inference_api_url)
        self.vector_db = Neo4jVectorDB(neo4j_uri, neo4j_username, neo4j_password, self.embedding_function)
        self.llm_inference_api_url = llm_inference_api_url
        
        # Hyperparameters
        self.max_clusters_per_document = max_clusters_per_document
        self.cluster_similarity_threshold = cluster_similarity_threshold
    
    def ingest_documents(self, file_path: str):
        """Ingest documents with multi-cluster assignment based on metadata embeddings"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self._split_text_into_chunks(text)
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Extract metadata for the chunk
            chunk_metadata = self.metadata_extractor.extract_metadata(chunk)
            
            # Create cluster if it doesn't exist
            primary_cluster_id = self.vector_db.create_or_get_cluster(chunk_metadata, self.metadata_extractor)
            
            # Find similar existing clusters for multi-cluster assignment
            similar_clusters = self.vector_db.find_similar_clusters(
                chunk_metadata, 
                self.metadata_extractor,
                max_clusters=self.max_clusters_per_document,
                similarity_threshold=self.cluster_similarity_threshold
            )
            
            # Ensure primary cluster is included
            cluster_assignments = [(primary_cluster_id, 1.0)]  # Primary cluster gets max score
            
            # Add similar clusters (avoiding duplicates)
            for cluster_id, similarity_score in similar_clusters:
                if cluster_id != primary_cluster_id and len(cluster_assignments) < self.max_clusters_per_document:
                    cluster_assignments.append((cluster_id, similarity_score))
            
            # Add document to selected clusters
            cluster_ids = [ca[0] for ca in cluster_assignments]
            similarity_scores = [ca[1] for ca in cluster_assignments]
            
            self.vector_db.add_document_to_clusters(
                cluster_ids, chunk, chunk_metadata, i, similarity_scores
            )
            
            print(f"  Assigned to {len(cluster_ids)} clusters: {cluster_ids}")
        
        print(f"Successfully ingested {len(chunks)} chunks with multi-cluster assignment.")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= end:  # Fixed infinite loop condition
                break
        return chunks
    
    def query(self, question: str, top_clusters: int = 3, docs_per_cluster: int = 3) -> Dict[str, Any]:
        """Hierarchical retrieval using metadata embeddings for cluster selection"""
        
        # Extract metadata from the query
        query_metadata = self.metadata_extractor.extract_metadata(question)
        print(f"Query metadata: {query_metadata}")
        
        # Find most relevant clusters using metadata embedding similarity
        relevant_clusters = self.vector_db.find_relevant_clusters_by_embedding(
            query_metadata, self.metadata_extractor, top_clusters
        )
        print(f"Found {len(relevant_clusters)} relevant clusters using metadata embeddings")
        
        # Get query content embedding for document search
        query_content_embedding = self.embedding_function.embed_query(question)
        
        # Search within each relevant cluster
        all_documents = []
        for cluster in relevant_clusters:
            print(f"Searching in cluster: {cluster['topic']} (similarity: {cluster['similarity_score']:.3f})")
            
            cluster_docs = self.vector_db.search_within_cluster(
                cluster['cluster_id'], query_content_embedding, docs_per_cluster
            )
            
            # Add cluster info to each document
            for doc in cluster_docs:
                doc['cluster_info'] = {
                    'cluster_id': cluster['cluster_id'],
                    'cluster_topic': cluster['topic'],
                    'cluster_category': cluster['category'],
                    'cluster_domain': cluster['domain'],
                    'cluster_metadata_similarity': cluster['similarity_score']
                }
                # Calculate final score combining cluster and content similarity
                doc['final_score'] = (
                    cluster['similarity_score'] * 0.4 +  # Cluster relevance weight
                    doc['combined_score'] * 0.6          # Content relevance weight
                )
            
            all_documents.extend(cluster_docs)
        
        # Sort all documents by final combined score
        all_documents.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Generate answer using top documents
        context_docs = all_documents[:5]  # Use top 5 documents for context
        context = "\n\n".join([
            f"[Cluster: {doc['cluster_info']['cluster_topic']} | Score: {doc['final_score']:.3f}]\n{doc['content']}" 
            for doc in context_docs
        ])
        
        answer = self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "query_metadata": query_metadata,
            "relevant_clusters": relevant_clusters,
            "source_documents": all_documents,
            "context_used": context_docs,
            "retrieval_stats": {
                "clusters_searched": len(relevant_clusters),
                "documents_found": len(all_documents),
                "max_clusters_per_doc": self.max_clusters_per_document,
                "similarity_threshold": self.cluster_similarity_threshold
            }
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        system_prompt = """You are a helpful assistant. Answer the question based only on the provided context. 
        If the context doesn't contain enough information to answer the question, say so clearly.
        The context includes relevance scores - prioritize information from higher-scoring sources."""
        
        user_prompt = f"""Context:\n{context}\n\nQuestion: {question}"""
        
        payload = {
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            },
            "model": {
                "name": "gemini-1.5-flash",
                "top_p": 0.7,
                "temperature": 0.3,
                "max_tokens": 1000,
                "do_sample": True,
                "repetition_penalty": 0,
                "json_mode": False
            }
        }
        
        try:
            response = requests.post(url=self.llm_inference_api_url, json=payload, verify=False)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            return f"Error generating answer: {e}"
        
        return "Unable to generate answer."
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about the clusters and document assignments"""
        with self.vector_db.driver.session() as session:
            # Get cluster statistics
            cluster_result = session.run("""
                MATCH (c:Cluster)
                OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
                RETURN c.id as cluster_id, c.topic, c.category, c.domain, 
                       c.keywords, c.tags, count(d) as document_count
                ORDER BY document_count DESC
            """)
            
            clusters = []
            for record in cluster_result:
                clusters.append({
                    'cluster_id': record['cluster_id'],
                    'topic': record['c.topic'],
                    'category': record['c.category'],
                    'domain': record['c.domain'],
                    'keywords': record['c.keywords'],
                    'tags': record['c.tags'],
                    'document_count': record['document_count']
                })
            
            # Get multi-cluster assignment statistics
            multi_cluster_result = session.run("""
                MATCH (d:Document)-[:BELONGS_TO]->(c:Cluster)
                WITH d.chunk_id as chunk_id, count(c) as cluster_count
                RETURN cluster_count, count(*) as documents_with_this_count
                ORDER BY cluster_count
            """)
            
            multi_cluster_stats = {}
            for record in multi_cluster_result:
                multi_cluster_stats[record['cluster_count']] = record['documents_with_this_count']
            
            return {
                'total_clusters': len(clusters),
                'clusters': clusters,
                'total_documents': sum(c['document_count'] for c in clusters),
                'unique_document_chunks': len(set(str(c['cluster_id']).split('_')[0] for c in clusters if c['cluster_id'])),
                'multi_cluster_assignment_stats': multi_cluster_stats,
                'hyperparameters': {
                    'max_clusters_per_document': self.max_clusters_per_document,
                    'cluster_similarity_threshold': self.cluster_similarity_threshold
                }
            }
    
    def update_hyperparameters(self, max_clusters_per_document: int = None, 
                             cluster_similarity_threshold: float = None):
        """Update hyperparameters for cluster assignment"""
        if max_clusters_per_document is not None:
            self.max_clusters_per_document = max_clusters_per_document
            print(f"Updated max_clusters_per_document to: {max_clusters_per_document}")
        
        if cluster_similarity_threshold is not None:
            self.cluster_similarity_threshold = cluster_similarity_threshold
            print(f"Updated cluster_similarity_threshold to: {cluster_similarity_threshold}")
    
    def close(self):
        """Close database connection"""
        self.vector_db.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    embedding_api_url = "YOUR_EMBEDDING_API_URL"
    llm_inference_api_url = "YOUR_LLM_API_URL"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "your_password"
    
    # Initialize the hierarchical RAG pipeline with hyperparameters
    rag = HierarchicalRAGPipeline(
        embedding_api_url=embedding_api_url,
        llm_inference_api_url=llm_inference_api_url,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        max_clusters_per_document=2,        # Allow documents in up to 2 clusters
        cluster_similarity_threshold=0.75   # Minimum similarity for cluster assignment
    )
    
    try:
        # Ingest documents with multi-cluster assignment
        rag.ingest_documents("test.txt")