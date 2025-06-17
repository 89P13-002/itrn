# faiss_index_enhanced.py - Enhanced version with query context weighting and no chunk filtering
import logging
import numpy as np
import faiss
import pickle
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
import threading
from config import config
from llm_fun import get_chunk_and_metadata_from_llm, compute_embeddings, normalize_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

class EnhancedScalableHierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 384, metadata_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        
        # Hierarchical structure
        self.metadata_cluster_centroids = None
        self.metadata_cluster_index = None
        
        # Fine-grained content indices within each metadata cluster
        self.content_indices = {}             # cluster_id -> content FAISS indices
        self.cluster_content_embeddings = {} # cluster_id -> content embeddings
        self.cluster_texts = {}               # cluster_id -> chunk texts
        self.cluster_queries = {}             # cluster_id -> related queries
        self.cluster_chunk_info = {}          # cluster_id -> chunk info
        self.cluster_assignments = {}         # chunk_hash -> list of assigned cluster_ids
        
        # Content deduplication
        self.content_hashes = set()           # Track unique content hashes
        self.chunk_to_hash = {}               # chunk_index -> content_hash
        self.hash_to_clusters = {}            # content_hash -> list of cluster_ids it belongs to
        
        self.next_chunk_id = 0
        self.lock = threading.Lock()
        self.metadata_pca = None
    
    def _generate_content_hash(self, text: str) -> str:
        """Generate hash for content deduplication"""
        # Normalize text for better deduplication
        normalized = ' '.join(text.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _build_metadata_clusters(self, all_chunk_data: List[Dict], num_clusters: int = 50):
        """Build coarse clusters using metadata embeddings"""
        logger.info("Building metadata-based coarse clusters...")
        
        if not all_chunk_data:
            logger.error("No chunk data provided for clustering")
            return {}
        
        # Extract metadata embeddings
        metadata_embeddings = [item['metadata_embedding'] for item in all_chunk_data]
        metadata_embeddings = np.array(metadata_embeddings)
        
        if len(metadata_embeddings.shape) == 3:
            metadata_embeddings = metadata_embeddings.squeeze(1)
        
        # Reduce metadata embedding dimension
        reduced_metadata_embeddings = self._reduce_metadata_embeddings(metadata_embeddings)
        
        # Determine optimal number of clusters
        n_samples = len(reduced_metadata_embeddings)
        optimal_clusters = min(num_clusters, max(5, n_samples // 20))
        
        logger.info(f"Performing metadata K-means clustering with {optimal_clusters} clusters on {n_samples} samples")
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=optimal_clusters, 
            random_state=42, 
            n_init=5,
            max_iter=300,
            algorithm='lloyd'
        )
        cluster_labels = kmeans.fit_predict(reduced_metadata_embeddings)
        
        # Store cluster centroids and create index
        self.metadata_cluster_centroids = kmeans.cluster_centers_
        self.metadata_cluster_index = faiss.IndexFlatIP(self.metadata_cluster_centroids.shape[1])
        self.metadata_cluster_index.add(self.metadata_cluster_centroids.astype(np.float32))
        
        # Create metadata cluster assignments
        cluster_assignments = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}"
            cluster_assignments[cluster_id].append(i)
        
        logger.info(f"Created {len(cluster_assignments)} metadata clusters")
        return dict(cluster_assignments)
    
    def _reduce_metadata_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce metadata embedding dimensions using PCA"""
        if embeddings.shape[1] <= self.metadata_dim:
            return embeddings
        
        if self.metadata_pca is None:
            from sklearn.decomposition import PCA
            self.metadata_pca = PCA(n_components=self.metadata_dim, random_state=42)
            reduced = self.metadata_pca.fit_transform(embeddings)
        else:
            reduced = self.metadata_pca.transform(embeddings)
        
        return reduced
    
    def _assign_chunks_to_clusters(self, all_chunk_data: List[Dict], 
                                  cluster_assignments: Dict[str, List[int]],
                                  top_k_clusters: int = 3):
        """Assign chunks to top-k most similar metadata clusters - GENERALIZED VERSION"""
        logger.info(f"Assigning chunks to top-{top_k_clusters} metadata clusters (generalized)...")
        
        cluster_data = defaultdict(lambda: {
            'content_embeddings': [], 
            'texts': [], 
            'queries': [],
            'chunk_info': [],
            'content_hashes': set()
        })
        
        # Process each chunk - NO RELEVANCE FILTERING
        for chunk_idx, chunk_data in all_chunk_data:
            content_hash = self._generate_content_hash(chunk_data['chunk_text'])
            
            # Skip if content already exists globally
            if content_hash in self.content_hashes:
                logger.debug(f"Skipping duplicate content at index {chunk_idx}")
                continue
            
            # NO RELEVANCE CHECK - ACCEPT ALL CHUNKS
            # Get metadata embedding for cluster assignment
            metadata_emb = chunk_data['metadata_embedding']
            if len(metadata_emb.shape) == 2:
                metadata_emb = metadata_emb.squeeze(0)
            
            reduced_metadata_emb = self._reduce_metadata_embeddings(
                metadata_emb.reshape(1, -1)
            ).flatten()
            
            # Find top-k similar clusters
            search_k = min(top_k_clusters, len(cluster_assignments))
            scores, indices = self.metadata_cluster_index.search(
                reduced_metadata_emb.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # Assign to top clusters
            cluster_ids = list(cluster_assignments.keys())
            assigned_clusters = []
            
            for score, cluster_idx in zip(scores[0], indices[0]):
                if cluster_idx != -1 and cluster_idx < len(cluster_ids):
                    cluster_id = cluster_ids[cluster_idx]
                    
                    # Check for duplicate content within cluster
                    if content_hash not in cluster_data[cluster_id]['content_hashes']:
                        cluster_data[cluster_id]['content_embeddings'].append(chunk_data['content_embedding'])
                        cluster_data[cluster_id]['texts'].append(chunk_data['chunk_text'])
                        cluster_data[cluster_id]['queries'].append(chunk_data['queries'])
                        
                        chunk_info = {
                            'chunk_idx': chunk_idx,
                            'queries': chunk_data['queries'],
                            'content_hash': content_hash,
                            'metadata': chunk_data.get('metadata', {}),
                            'filename': chunk_data.get('filename', 'unknown')
                        }
                        cluster_data[cluster_id]['chunk_info'].append(chunk_info)
                        cluster_data[cluster_id]['content_hashes'].add(content_hash)
                        
                        assigned_clusters.append(cluster_id)
            
            if assigned_clusters:
                self.content_hashes.add(content_hash)
                self.cluster_assignments[content_hash] = assigned_clusters
                self.chunk_to_hash[chunk_idx] = content_hash
                self.hash_to_clusters[content_hash] = assigned_clusters
            
            if chunk_idx % 1000 == 0:
                logger.info(f"Processed {chunk_idx + 1} chunks")
        
        logger.info(f"Completed chunk assignment. Active clusters: {len([k for k, v in cluster_data.items() if v['texts']])}")
        return cluster_data
    
    def _create_hnsw_index(self, content_embs_np: np.ndarray, cluster_size: int) -> faiss.Index:
        """Create HNSW index with appropriate parameters"""
        if cluster_size <= 20:
            M, efConstruction = 8, 40
        elif cluster_size <= 100:
            M, efConstruction = 16, 64
        elif cluster_size <= 500:
            M, efConstruction = 24, 128
        elif cluster_size <= 2000:
            M, efConstruction = 32, 200
        else:
            M, efConstruction = 48, 256
        
        try:
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            index.hnsw.efConstruction = efConstruction
            index.add(content_embs_np.astype(np.float32))
            logger.debug(f"Created HNSW index with M={M}, efConstruction={efConstruction}")
            return index
        except Exception as e:
            logger.error(f"HNSW creation failed: {e}, falling back to flat index")
            fallback_index = faiss.IndexFlatIP(self.embedding_dim)
            fallback_index.add(content_embs_np.astype(np.float32))
            return fallback_index
    
    def build_hierarchical_index(self, all_chunk_data: List[Dict]):
        """Build hierarchical index from processed chunk data"""
        logger.info("Building enhanced hierarchical FAISS index...")
        
        if not all_chunk_data:
            logger.error("No chunk data provided")
            return
        
        # Step 1: Build metadata clusters
        cluster_assignments = self._build_metadata_clusters(all_chunk_data)
        
        # Step 2: Assign chunks to clusters - NO FILTERING
        cluster_data = self._assign_chunks_to_clusters(all_chunk_data, cluster_assignments)
        
        # Step 3: Build HNSW indices for each cluster
        logger.info("Building HNSW content indices...")
        
        for cluster_id, data in cluster_data.items():
            if not data['texts']:
                continue
            
            content_embs_np = np.array(data['content_embeddings'])
            if len(content_embs_np.shape) == 3:
                content_embs_np = content_embs_np.squeeze(1)
            
            cluster_size = len(content_embs_np)
            logger.info(f"Creating HNSW index for {cluster_id} with {cluster_size} vectors")
            
            # Create HNSW index
            index = self._create_hnsw_index(content_embs_np, cluster_size)
            
            # Store cluster data
            self.content_indices[cluster_id] = index
            self.cluster_content_embeddings[cluster_id] = content_embs_np
            self.cluster_texts[cluster_id] = data['texts']
            self.cluster_queries[cluster_id] = data['queries']
            self.cluster_chunk_info[cluster_id] = data['chunk_info']
        
        logger.info(f"Built hierarchical index with {len(self.content_indices)} clusters")
    
    def print_cluster_contents(self, output_path: str = None):
        """Print entire chunks for each metadata cluster"""
        logger.info("Printing cluster contents...")
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        
        for cluster_id in sorted(self.content_indices.keys()):
            print(f"\n{'='*80}")
            print(f"CLUSTER: {cluster_id}")
            print(f"Total chunks: {len(self.cluster_texts[cluster_id])}")
            print(f"{'='*80}")
            
            texts = self.cluster_texts[cluster_id]
            chunk_info = self.cluster_chunk_info[cluster_id]
            
            cluster_content = []
            
            for i, (text, info) in enumerate(zip(texts, chunk_info)):
                chunk_display = f"""
CHUNK {i+1} (Original Index: {info['chunk_idx']}):
Source: {info.get('filename', 'unknown')}
Content Hash: {info['content_hash'][:8]}...
Metadata: {json.dumps(info.get('metadata', {}), indent=2)}
Queries: {info['queries']}

TEXT:
{text}
{'-'*60}
"""
                print(chunk_display)
                cluster_content.append({
                    'local_index': i,
                    'original_index': info['chunk_idx'],
                    'filename': info.get('filename', 'unknown'),
                    'content_hash': info['content_hash'],
                    'metadata': info.get('metadata', {}),
                    'queries': info['queries'],
                    'text': text
                })
            
            # Save to file if output path provided
            if output_path:
                cluster_file = os.path.join(output_path, f"{cluster_id}_full_contents.json")
                with open(cluster_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'cluster_id': cluster_id,
                        'total_chunks': len(texts),
                        'chunks': cluster_content
                    }, f, indent=2, ensure_ascii=False)
        
        logger.info("Completed printing cluster contents")
    
    def search_with_query_context_weighting(self, query_text: str, query_embedding: np.ndarray, 
                                          query_metadata: Dict, metadata_embedding: np.ndarray,
                                          query_context_weight: float = 2.0,
                                          k: int = 5, num_metadata_clusters: int = 5) -> List[Tuple[float, int, Dict, str, List[str]]]:
        """Enhanced search with QUERY CONTEXT weighting and duplicate handling"""
        
        # Find relevant metadata clusters
        relevant_clusters = self._find_relevant_metadata_clusters(
            metadata_embedding, num_metadata_clusters
        )
        
        if not relevant_clusters:
            logger.warning("No relevant metadata clusters found")
            return []
        
        # Search within clusters
        all_results = []
        seen_hashes = set()  # Track seen content hashes to avoid duplicates
        results_per_cluster = max(1, (k * 3) // len(relevant_clusters))  # Get more results initially
        
        for cluster_id in relevant_clusters:
            cluster_results = self._search_content_cluster(
                cluster_id, query_embedding, results_per_cluster
            )
            
            # Add results with query context weighting
            for score, local_idx, text in cluster_results:
                if cluster_id in self.cluster_chunk_info:
                    chunk_info = self.cluster_chunk_info[cluster_id][local_idx]
                    content_hash = chunk_info['content_hash']
                    
                    # Skip if we've already seen this content
                    if content_hash in seen_hashes:
                        continue
                    
                    seen_hashes.add(content_hash)
                    
                    # Calculate query context similarity
                    query_context_score = self._calculate_query_context_similarity(
                        query_text, chunk_info['queries']
                    )
                    
                    # Apply query context weighting to the retrieval score
                    weighted_score = score * (1 + query_context_weight * query_context_score)
                    
                    all_results.append((
                        weighted_score, 
                        chunk_info['chunk_idx'], 
                        chunk_info.get('metadata', {}), 
                        text, 
                        chunk_info['queries'],
                        {
                            'original_score': score,
                            'query_context_score': query_context_score,
                            'weighted_score': weighted_score,
                            'cluster_id': cluster_id,
                            'content_hash': content_hash,
                            'filename': chunk_info.get('filename', 'unknown')
                        }
                    ))
        
        # Sort by weighted score and return top k unique results
        all_results.sort(key=lambda x: x[0], reverse=True)
        return all_results[:k]
    
    def _calculate_query_context_similarity(self, query_text: str, chunk_queries: List[str]) -> float:
        """Calculate similarity between search query and chunk's stored queries"""
        if not chunk_queries:
            return 0.0
        
        query_words = set(query_text.lower().split())
        
        max_similarity = 0.0
        for chunk_query in chunk_queries:
            chunk_query_words = set(chunk_query.lower().split())
            
            # Calculate Jaccard similarity
            if len(query_words.union(chunk_query_words)) > 0:
                similarity = len(query_words.intersection(chunk_query_words)) / len(query_words.union(chunk_query_words))
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _find_relevant_metadata_clusters(self, metadata_embedding: np.ndarray, 
                                       num_clusters: int = 5) -> List[str]:
        """Find most relevant metadata clusters for the query"""
        if self.metadata_cluster_index is None:
            logger.warning("No metadata clusters available")
            return []
        
        if len(metadata_embedding.shape) == 2:
            metadata_embedding = metadata_embedding.squeeze(0)
        
        query_metadata_reduced = self._reduce_metadata_embeddings(
            metadata_embedding.reshape(1, -1)
        ).flatten()
        
        # Normalize
        norm = np.linalg.norm(query_metadata_reduced)
        if norm > 0:
            query_metadata_reduced = query_metadata_reduced / norm
        
        search_k = min(num_clusters, len(self.content_indices))
        
        try:
            scores, indices = self.metadata_cluster_index.search(
                query_metadata_reduced.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            cluster_ids = list(self.content_indices.keys())
            relevant_clusters = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(cluster_ids):
                    cluster_id = cluster_ids[idx]
                    if cluster_id in self.content_indices:
                        relevant_clusters.append(cluster_id)
            
            return relevant_clusters
        except Exception as e:
            logger.error(f"Error in metadata cluster search: {e}")
            return list(self.content_indices.keys())[:search_k]
    
    def _search_content_cluster(self, cluster_id: str, query_embedding: np.ndarray, 
                              k: int) -> List[Tuple[float, int, str]]:
        """Search within a specific content cluster"""
        if cluster_id not in self.content_indices:
            return []
        
        index = self.content_indices[cluster_id]
        texts_list = self.cluster_texts[cluster_id]
        
        # Set HNSW search parameters
        if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
            cluster_size = len(texts_list)
            index.hnsw.efSearch = max(k * 2, min(cluster_size, 64))
        
        try:
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding.squeeze(0)
            
            search_k = min(k, len(texts_list))
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(texts_list):
                    results.append((float(score), idx, texts_list[idx]))
            
            return results
        except Exception as e:
            logger.error(f"Error searching cluster {cluster_id}: {e}")
            return []
    
    def save_index(self, path: str):
        """Save the enhanced hierarchical index to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save metadata cluster index
        if self.metadata_cluster_index is not None:
            faiss.write_index(self.metadata_cluster_index, os.path.join(path, "metadata_clusters.faiss"))
        
        # Save content indices
        for cluster_id, index in self.content_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"content_{safe_name}.index"))
        
        # Save all metadata and mappings
        metadata_file = os.path.join(path, "enhanced_index_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'cluster_content_embeddings': self.cluster_content_embeddings,
                'cluster_texts': self.cluster_texts,
                'cluster_queries': self.cluster_queries,
                'cluster_chunk_info': self.cluster_chunk_info,
                'cluster_assignments': self.cluster_assignments,
                'content_hashes': self.content_hashes,
                'chunk_to_hash': self.chunk_to_hash,
                'hash_to_clusters': self.hash_to_clusters,
                'next_chunk_id': self.next_chunk_id,
                'embedding_dim': self.embedding_dim,
                'metadata_dim': self.metadata_dim,
                'metadata_pca': self.metadata_pca,
                'metadata_cluster_centroids': self.metadata_cluster_centroids
            }, f)
        
        logger.info(f"Saved enhanced hierarchical index to {path}")
    
    def load_index(self, path: str):
        """Load the enhanced hierarchical index from disk"""
        # Load metadata cluster index
        metadata_index_path = os.path.join(path, "metadata_clusters.faiss")
        if os.path.exists(metadata_index_path):
            self.metadata_cluster_index = faiss.read_index(metadata_index_path)
        
        # Load metadata and mappings
        metadata_file = os.path.join(path, "enhanced_index_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.cluster_content_embeddings = data.get('cluster_content_embeddings', {})
                self.cluster_texts = data.get('cluster_texts', {})
                self.cluster_queries = data.get('cluster_queries', {})
                self.cluster_chunk_info = data.get('cluster_chunk_info', {})
                self.cluster_assignments = data.get('cluster_assignments', {})
                self.content_hashes = data.get('content_hashes', set())
                self.chunk_to_hash = data.get('chunk_to_hash', {})
                self.hash_to_clusters = data.get('hash_to_clusters', {})
                self.next_chunk_id = data.get('next_chunk_id', 0)
                self.embedding_dim = data.get('embedding_dim', 384)
                self.metadata_dim = data.get('metadata_dim', 128)
                self.metadata_pca = data.get('metadata_pca')
                self.metadata_cluster_centroids = data.get('metadata_cluster_centroids')
        
        # Load content indices
        for cluster_id in self.cluster_texts.keys():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"content_{safe_name}.index")
            if os.path.exists(index_path):
                self.content_indices[cluster_id] = faiss.read_index(index_path)
        
        logger.info(f"Loaded enhanced hierarchical index from {path}")
    
    def get_index_stats(self):
        """Get comprehensive index statistics"""
        total_chunks = sum(len(texts) for texts in self.cluster_texts.values())
        total_queries = sum(len(queries) for queries in self.cluster_queries.values())
        
        stats = {
            'total_clusters': len(self.content_indices),
            'total_unique_chunks': len(self.content_hashes),
            'total_chunks_assigned': total_chunks,
            'total_queries': total_queries,
            'avg_chunks_per_cluster': total_chunks / len(self.content_indices) if self.content_indices else 0,
            'avg_queries_per_cluster': total_queries / len(self.cluster_queries) if self.cluster_queries else 0,
            'deduplication_ratio': len(self.content_hashes) / max(self.next_chunk_id, 1),
            'embedding_dim': self.embedding_dim,
            'metadata_dim': self.metadata_dim,
            'hnsw_clusters': sum(1 for idx in self.content_indices.values() if hasattr(idx, 'hnsw')),
            'clusters_per_hash': {
                h: len(clusters) for h, clusters in self.hash_to_clusters.items()
            }
        }
        
        return stats


def process_documents_enhanced():
    """Process documents with enhanced generalized approach"""
    logger.info("Starting enhanced document processing...")
    
    # Initialize enhanced index
    faiss_index = EnhancedScalableHierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=128
    )
    
    # Load documents
    documents = load_documents(config.DATA_FOLDER)
    if not documents:
        logger.error("No documents loaded")
        return None
    
    # Process all documents - NO FILTERING
    all_chunk_data = []
    
    for doc in documents:
        logger.info(f"Processing document: {doc['filename']}")
        
        # Get chunks, metadata, and queries simultaneously
        processed_chunks = get_chunk_and_metadata_from_llm(doc['content'])
        
        for chunk_data in processed_chunks:
            # Compute embeddings
            content_embedding = normalize_embeddings(
                compute_embeddings(chunk_data['chunk_text'])
            )
            metadata_embedding = normalize_embeddings(
                compute_embeddings(chunk_data['metadata_text'])
            )
            
            enhanced_chunk_data = {
                'chunk_text': chunk_data['chunk_text'],
                'metadata': chunk_data['metadata'],
                'queries': chunk_data['queries'],
                'content_embedding': content_embedding,
                'metadata_embedding': metadata_embedding,
                'metadata_text': chunk_data['metadata_text'],
                'filename': doc['filename']
            }
            
            all_chunk_data.append(enhanced_chunk_data)
            faiss_index.next_chunk_id += 1
    
    logger.info(f"Processed {len(all_chunk_data)} total chunks (no filtering applied)")
    
    # Build hierarchical index
    faiss_index.build_hierarchical_index(all_chunk_data)
    
    # Print cluster contents
    cluster_output_path = os.path.join(config.OUTPUT_PATH, "cluster_contents")
    faiss_index.print_cluster_contents(cluster_output_path)
    
    # Print statistics
    stats = faiss_index.get_index_stats()
    logger.info(f"Final Index Statistics: {json.dumps(stats, indent=2)}")
    
    # Save index
    os.makedirs(config.INDEX_PATH, exist_ok=True)
    faiss_index.save_index(config.INDEX_PATH)
    
    logger.info("Enhanced generalized index completed!")
    return faiss_index


def search_with_context_weighting(faiss_index, query_text: str, query_metadata: Dict = None, 
                                query_context_weight: float = 2.0, k: int = 5):
    """Search function with query context weighting"""
    
    # Compute query embeddings
    query_embedding = normalize_embeddings(compute_embeddings(query_text))
    
    if query_metadata is None:
        query_metadata = {'topic': 'general', 'category': 'query'}
    
    metadata_text = ' '.join([f"{k}: {v}" for k, v in query_metadata.items()])
    metadata_embedding = normalize_embeddings(compute_embeddings(metadata_text))
    
    # Search with context weighting
    results = faiss_index.search_with_query_context_weighting(
        query_text=query_text,
        query_embedding=query_embedding,
        query_metadata=query_metadata,
        metadata_embedding=metadata_embedding,
        query_context_weight=query_context_weight,
        k=k
    )
    
    # Display results
    print(f"\nSearch Results for: '{query_text}'")
    print(f"Query Context Weight: {query_context_weight}")
    print("="*80)
    
    for i, (weighted_score, chunk_idx, metadata, text, queries, debug_info) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Weighted Score: {weighted_score:.4f}")
        print(f"Original Score: {debug_info['original_score']:.4f}")
        print(f"Query Context Score: {debug_info['query_context_score']:.4f}")
        print(f"Cluster: {debug_info['cluster_id']}")
        print(f"Source: {debug_info['filename']}")
        print(f"Content Hash: {debug_info['content_hash'][:8]}...")
        print(f"Chunk Queries: {queries}")
        print(f"Text: {text[:200]}...")
        print("-"*60)
    
    return results


# Example usage
if __name__ == "__main__":
    # Process documents
    index = process_documents_enhanced()
    
    if index:
        # Example search with different context weights
        search_with_context_weighting(
            index, 
            "What is machine learning?", 
            query_context_weight=2.0, 
            k=5
        )
