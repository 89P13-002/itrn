import json
from typing import List, Dict, Optional, Any, Tuple
import requests
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

class CustomEmbeddings(Embeddings):
    """Custom embedding class compatible with LangChain"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
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
        """Embed a single query"""
        if not text:
            return []
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

class SimpleMetadataExtractor:
    """Simplified metadata extractor focusing on topic and category only"""
    
    def __init__(self, llm_inference_api_url: str):
        self.llm_inference_api_url = llm_inference_api_url
    
    def extract_metadata(self, text_chunk: str) -> Dict[str, str]:
        """Extract simple metadata: topic and category only"""
        system_prompt = """You are a metadata extraction assistant. Analyze the given text and extract simple metadata.
        Return a JSON object with only these fields:
        - topic: main topic/subject (string, max 3 words)
        - category: document category (one of: "technical", "business", "academic", "general")
        
        Return only valid JSON without any additional text."""
        
        user_prompt = f"Extract metadata from this text:\n\n{text_chunk[:1500]}..."
        
        payload = {
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            },
            "model": {
                "name": "gemini-1.5-flash",
                "top_p": 0.1,
                "temperature": 0.3,
                "max_tokens": 100,
                "do_sample": True,
                "repetition_penalty": 0,
                "json_mode": True
            }
        }
        
        try:
            response = requests.post(url=self.llm_inference_api_url, json=payload, verify=False)
            if response.status_code == 200:
                metadata_str = response.json().get("response", "{}")
                metadata = json.loads(metadata_str)
                # Ensure we have the required fields
                return {
                    "topic": metadata.get("topic", "unknown"),
                    "category": metadata.get("category", "general")
                }
        except Exception as e:
            print(f"Metadata extraction error: {e}")
        
        # Fallback metadata
        return {
            "topic": "unknown",
            "category": "general"
        }

class HierarchicalRAGWithLangChain:
    """Hierarchical RAG implementation using LangChain Neo4j integration"""
    
    def __init__(self, embedding_api_url: str, llm_inference_api_url: str, 
                 neo4j_url: str, neo4j_username: str, neo4j_password: str):
        
        self.embeddings = CustomEmbeddings(embedding_api_url)
        self.metadata_extractor = SimpleMetadataExtractor(llm_inference_api_url)
        self.llm_inference_api_url = llm_inference_api_url
        
        # Initialize Neo4j graph connection
        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Create clusters dictionary to track vector stores
        self.cluster_stores = {}
        
        # Setup Neo4j constraints and indexes
        self._setup_neo4j_schema()
    
    def _setup_neo4j_schema(self):
        """Setup Neo4j schema for hierarchical structure"""
        try:
            # Create constraints and indexes
            self.graph.query("""
                CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:Cluster) REQUIRE c.id IS UNIQUE
            """)
            
            self.graph.query("""
                CREATE INDEX cluster_topic IF NOT EXISTS FOR (c:Cluster) ON (c.topic)
            """)
            
            self.graph.query("""
                CREATE INDEX cluster_category IF NOT EXISTS FOR (c:Cluster) ON (c.category)
            """)
            
            print("Neo4j schema setup completed")
            
        except Exception as e:
            print(f"Schema setup warning: {e}")
    
    def _get_cluster_id(self, topic: str, category: str) -> str:
        """Generate cluster ID from metadata"""
        return f"{category}_{topic}".replace(" ", "_").lower()
    
    def _create_or_get_cluster(self, topic: str, category: str) -> str:
        """Create cluster node in Neo4j if it doesn't exist"""
        cluster_id = self._get_cluster_id(topic, category)
        
        # Check if cluster exists, if not create it
        result = self.graph.query("""
            MERGE (c:Cluster {id: $cluster_id})
            SET c.topic = $topic,
                c.category = $category,
                c.document_count = COALESCE(c.document_count, 0)
            RETURN c.id as cluster_id
        """, {"cluster_id": cluster_id, "topic": topic, "category": category})
        
        return cluster_id
    
    def _get_or_create_vector_store(self, cluster_id: str) -> Neo4jVector:
        """Get or create vector store for a specific cluster"""
        if cluster_id not in self.cluster_stores:
            # Create vector store for this cluster with a unique index name
            index_name = f"vector_index_{cluster_id}"
            
            self.cluster_stores[cluster_id] = Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=self.graph._driver.uri,
                username=self.graph._driver._auth[0],
                password=self.graph._driver._auth[1],
                index_name=index_name,
                node_label="Document",
                text_node_property="content",
                embedding_node_property="embedding",
                create_id_index=True,
            )
        
        return self.cluster_stores[cluster_id]
    
    def ingest_documents(self, file_path: str):
        """Ingest documents with hierarchical clustering"""
        # Read the document
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        print(f"Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(chunk)
            topic = metadata['topic']
            category = metadata['category']
            
            # Create or get cluster
            cluster_id = self._create_or_get_cluster(topic, category)
            
            # Create LangChain document with metadata
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "topic": topic,
                    "category": category,
                    "cluster_id": cluster_id
                }
            )
            
            # Get vector store for this cluster
            vector_store = self._get_or_create_vector_store(cluster_id)
            
            # Add document to vector store
            vector_store.add_documents([doc])
            
            # Update cluster document count
            self.graph.query("""
                MATCH (c:Cluster {id: $cluster_id})
                SET c.document_count = c.document_count + 1
            """, {"cluster_id": cluster_id})
            
            print(f"  Added to cluster: {cluster_id} (topic: {topic}, category: {category})")
        
        print(f"Successfully ingested {len(chunks)} chunks into {len(self.cluster_stores)} clusters.")
    
    def _find_relevant_clusters(self, query_metadata: Dict[str, str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant clusters based on query metadata"""
        query_topic = query_metadata['topic']
        query_category = query_metadata['category']
        
        # Simple matching strategy: exact category match gets priority, then topic similarity
        result = self.graph.query("""
            MATCH (c:Cluster)
            WITH c,
                 CASE WHEN c.category = $category THEN 1.0 ELSE 0.3 END as category_score,
                 CASE WHEN c.topic CONTAINS $topic OR $topic CONTAINS c.topic THEN 0.8 ELSE 0.2 END as topic_score
            WITH c, category_score + topic_score as total_score
            WHERE total_score > 0.5
            RETURN c.id as cluster_id, c.topic, c.category, c.document_count, total_score
            ORDER BY total_score DESC
            LIMIT $top_k
        """, {
            "category": query_category,
            "topic": query_topic,
            "top_k": top_k
        })
        
        clusters = []
        for record in result:
            clusters.append({
                "cluster_id": record["cluster_id"],
                "topic": record["c.topic"],
                "category": record["c.category"],
                "document_count": record["c.document_count"],
                "relevance_score": record["total_score"]
            })
        
        # If no clusters found, return all clusters
        if not clusters:
            result = self.graph.query("""
                MATCH (c:Cluster)
                RETURN c.id as cluster_id, c.topic, c.category, c.document_count, 0.5 as total_score
                ORDER BY c.document_count DESC
                LIMIT $top_k
            """, {"top_k": top_k})
            
            clusters = [{
                "cluster_id": record["cluster_id"],
                "topic": record["c.topic"],
                "category": record["c.category"],
                "document_count": record["c.document_count"],
                "relevance_score": record["total_score"]
            } for record in result]
        
        return clusters
    
    def query(self, question: str, top_clusters: int = 2, docs_per_cluster: int = 3) -> Dict[str, Any]:
        """Hierarchical query with cluster selection and document retrieval"""
        
        # Extract metadata from query
        query_metadata = self.metadata_extractor.extract_metadata(question)
        print(f"Query metadata: {query_metadata}")
        
        # Find relevant clusters
        relevant_clusters = self._find_relevant_clusters(query_metadata, top_clusters)
        print(f"Found {len(relevant_clusters)} relevant clusters")
        
        # Search within each cluster
        all_results = []
        
        for cluster in relevant_clusters:
            cluster_id = cluster["cluster_id"]
            print(f"Searching in cluster: {cluster['topic']} ({cluster['category']})")
            
            if cluster_id in self.cluster_stores:
                # Perform similarity search within the cluster
                vector_store = self.cluster_stores[cluster_id]
                docs = vector_store.similarity_search_with_score(question, k=docs_per_cluster)
                
                for doc, score in docs:
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score,
                        "cluster_info": cluster,
                        "final_score": cluster["relevance_score"] * 0.4 + (1 - score) * 0.6  # Combine scores
                    }
                    all_results.append(result)
        
        # Sort by final score
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Generate answer using top results
        top_results = all_results[:5]
        context = "\n\n".join([
            f"[Cluster: {result['cluster_info']['topic']} | Score: {result['final_score']:.3f}]\n{result['content']}"
            for result in top_results
        ])
        
        answer = self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "query_metadata": query_metadata,
            "relevant_clusters": relevant_clusters,
            "retrieved_documents": all_results,
            "context_used": top_results,
            "stats": {
                "clusters_searched": len(relevant_clusters),
                "documents_retrieved": len(all_results)
            }
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        system_prompt = """You are a helpful assistant. Answer the question based only on the provided context.
        If the context doesn't contain enough information, say so clearly.
        Prioritize information from higher-scoring sources."""
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        
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
                return response.json().get("response", "Unable to generate answer.")
        except Exception as e:
            return f"Error generating answer: {e}"
        
        return "Unable to generate answer."
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about clusters"""
        result = self.graph.query("""
            MATCH (c:Cluster)
            RETURN c.id as cluster_id, c.topic, c.category, c.document_count
            ORDER BY c.document_count DESC
        """)
        
        clusters = []
        total_docs = 0
        
        for record in result:
            cluster_info = {
                "cluster_id": record["cluster_id"],
                "topic": record["c.topic"],
                "category": record["c.category"],
                "document_count": record["c.document_count"]
            }
            clusters.append(cluster_info)
            total_docs += record["c.document_count"]
        
        return {
            "total_clusters": len(clusters),
            "total_documents": total_docs,
            "clusters": clusters
        }
    
    def close(self):
        """Close connections"""
        self.graph._driver.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    embedding_api_url = "YOUR_EMBEDDING_API_URL"
    llm_inference_api_url = "YOUR_LLM_API_URL"
    neo4j_url = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "your_password"
    
    # Initialize the hierarchical RAG pipeline
    rag = HierarchicalRAGWithLangChain(
        embedding_api_url=embedding_api_url,
        llm_inference_api_url=llm_inference_api_url,
        neo4j_url=neo4j_url,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password
    )
    
    try:
        # Ingest documents
        rag.ingest_documents("test.txt")
        
        # Get cluster statistics
        stats = rag.get_cluster_stats()
        print(f"\nCluster Statistics:")
        print(f"Total Clusters: {stats['total_clusters']}")
        print(f"Total Documents: {stats['total_documents']}")
        
        for cluster in stats['clusters']:
            print(f"- {cluster['topic']} ({cluster['category']}): {cluster['document_count']} docs")
        
        # Query the system
        question = "What is the main topic of the document?"
        result = rag.query(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Query metadata: {result['query_metadata']}")
        print(f"Clusters searched: {len(result['relevant_clusters'])}")
        print(f"Documents retrieved: {len(result['retrieved_documents'])}")
        
    finally:
        rag.close()
