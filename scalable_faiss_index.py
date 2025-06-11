# faiss_index.py - Scalable metadata-only clustering version
import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
import threading
from config import config
from llm_fun import get_metadata_from_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

class ScalableHierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 384, metadata_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim  # Reduced dimension for metadata clustering
        
        # Hierarchical structure - metadata-based coarse clustering
        self.metadata_clusters = {}           # metadata_cluster_id -> cluster info
        self.metadata_cluster_centroids = None  # centroids of metadata clusters
        self.metadata_cluster_index = None    # FAISS index for metadata clusters
        
        # Fine-grained content indices within each metadata cluster
        self.content_indices = {}             # metadata_cluster_id -> content FAISS indices
        self.cluster_metadata = {}            # metadata_cluster_id -> list of chunk metadata
        self.cluster_content_embeddings = {} # metadata_cluster_id -> content embeddings
        self.cluster_texts = {}               # metadata_cluster_id -> chunk texts
        self.cluster_assignments = {}         # chunk_idx -> list of assigned metadata_cluster_ids
        
        self.next_chunk_id = 0
        self.lock = threading.Lock()
        
        # PCA for metadata dimension reduction
        self.metadata_pca = None
    
    def _reduce_metadata_embeddings(self, metadata_embeddings: np.ndarray) -> np.ndarray:
        """Reduce metadata embedding dimension using PCA for faster clustering"""
        from sklearn.decomposition import PCA
        
        if metadata_embeddings.shape[1] <= self.metadata_dim:
            return metadata_embeddings
        
        if self.metadata_pca is None:
            logger.info(f"Reducing metadata embedding dimension from {metadata_embeddings.shape[1]} to {self.metadata_dim}")
            self.metadata_pca = PCA(n_components=self.metadata_dim, random_state=42)
            reduced_embeddings = self.metadata_pca.fit_transform(metadata_embeddings)
        else:
            reduced_embeddings = self.metadata_pca.transform(metadata_embeddings)
        
        # Normalize after PCA
        norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return reduced_embeddings / norms
    
    def _build_metadata_clusters(self, metadata_embeddings: List[np.ndarray], 
                                metadata_list: List[Dict[str, str]], 
                                num_clusters: int = 50):
        """Build coarse clusters using ONLY metadata embeddings for scalability"""
        logger.info("Building metadata-based coarse clusters...")
        
        # Convert to numpy array and handle shape
        metadata_embeddings = np.array(metadata_embeddings)
        if len(metadata_embeddings.shape) == 3:
            metadata_embeddings = metadata_embeddings.squeeze(1)
        
        # Reduce metadata embedding dimension for faster clustering
        reduced_metadata_embeddings = self._reduce_metadata_embeddings(metadata_embeddings)
        
        # Determine optimal number of clusters for scalability
        n_samples = len(reduced_metadata_embeddings)
        optimal_clusters = min(num_clusters, max(5, n_samples // 20))  # More conservative clustering
        
        logger.info(f"Performing metadata K-means clustering with {optimal_clusters} clusters on {n_samples} samples")
        logger.info(f"Using reduced metadata dimension: {reduced_metadata_embeddings.shape[1]}")
        
        # Perform clustering on reduced metadata embeddings
        kmeans = KMeans(
            n_clusters=optimal_clusters, 
            random_state=42, 
            n_init=5,  # Reduced for faster processing
            max_iter=300,  # Reduced iterations
            algorithm='lloyd'  # Most stable for large datasets
        )
        cluster_labels = kmeans.fit_predict(reduced_metadata_embeddings)
        
        # Store cluster centroids and create index
        self.metadata_cluster_centroids = kmeans.cluster_centers_
        self.metadata_cluster_index = faiss.IndexFlatIP(self.metadata_cluster_centroids.shape[1])
        self.metadata_cluster_index.add(self.metadata_cluster_centroids.astype(np.float32))
        
        # Create metadata cluster structure
        metadata_cluster_data = defaultdict(lambda: {'indices': [], 'metadata': []})
        for i, label in enumerate(cluster_labels):
            cluster_id = f"meta_cluster_{label}"
            metadata_cluster_data[cluster_id]['indices'].append(i)
            metadata_cluster_data[cluster_id]['metadata'].append(metadata_list[i])
        
        self.metadata_clusters = dict(metadata_cluster_data)
        
        logger.info(f"Created {len(metadata_cluster_data)} metadata clusters")
        return metadata_cluster_data
    
    def _assign_chunks_to_clusters(self, chunk_list: List[str], 
                                  content_embeddings: List[np.ndarray],
                                  metadata_embeddings: List[np.ndarray],
                                  metadata_list: List[Dict[str, str]],
                                  top_k_clusters: int = 3):
        """Assign each chunk to top-k most similar metadata clusters"""
        logger.info(f"Assigning chunks to top-{top_k_clusters} metadata clusters...")
        
        # Reduce metadata embeddings for cluster assignment
        metadata_embeddings_np = np.array(metadata_embeddings)
        if len(metadata_embeddings_np.shape) == 3:
            metadata_embeddings_np = metadata_embeddings_np.squeeze(1)
        
        reduced_metadata_embeddings = self._reduce_metadata_embeddings(metadata_embeddings_np)
        
        # Initialize cluster data structures
        cluster_data = defaultdict(lambda: {
            'content_embeddings': [], 
            'texts': [], 
            'metadata': [],
            'chunk_indices': []
        })
        
        # For each chunk, find top-k most similar metadata clusters
        for chunk_idx, (chunk_text, content_emb, metadata_emb, metadata) in enumerate(zip(
            chunk_list, content_embeddings, metadata_embeddings, metadata_list
        )):
            
            # Search for most similar metadata clusters
            scores, indices = self.metadata_cluster_index.search(
                reduced_metadata_embeddings[chunk_idx:chunk_idx+1].astype(np.float32),
                min(top_k_clusters, len(self.metadata_clusters))
            )
            
            # Assign chunk to top-k clusters
            assigned_clusters = []
            cluster_ids = list(self.metadata_clusters.keys())
            
            for score, cluster_idx in zip(scores[0], indices[0]):
                if cluster_idx != -1 and cluster_idx < len(cluster_ids):
                    cluster_id = cluster_ids[cluster_idx]
                    assigned_clusters.append(cluster_id)
                    
                    # Add chunk to this cluster
                    cluster_data[cluster_id]['content_embeddings'].append(content_emb)
                    cluster_data[cluster_id]['texts'].append(chunk_text)
                    cluster_data[cluster_id]['metadata'].append(metadata)
                    cluster_data[cluster_id]['chunk_indices'].append(chunk_idx)
            
            # Store cluster assignments for this chunk
            self.cluster_assignments[chunk_idx] = assigned_clusters
            
            if chunk_idx % 1000 == 0:
                logger.info(f"Assigned {chunk_idx + 1} chunks to clusters")
        
        logger.info(f"Completed chunk assignment. Clusters with content: {len([k for k, v in cluster_data.items() if v['texts']])}")
        return cluster_data
    
    def build_hierarchical_index(self, content_embeddings: List[np.ndarray], 
                                metadata_embeddings: List[np.ndarray], 
                                chunk_list: List[str], 
                                metadata_list: List[Dict[str, str]]):
        """Build scalable hierarchical index with metadata-only coarse clustering"""
        logger.info("Building scalable hierarchical FAISS index...")
        
        # Step 1: Build metadata-only coarse clusters
        self._build_metadata_clusters(metadata_embeddings, metadata_list)
        
        # Step 2: Assign chunks to top-k most similar metadata clusters
        cluster_data = self._assign_chunks_to_clusters(
            chunk_list, content_embeddings, metadata_embeddings, metadata_list
        )
        
        # Step 3: Build content-based fine indices within each cluster
        logger.info("Building content indices within metadata clusters...")
        
        for cluster_id, data in cluster_data.items():
            if not data['texts']:  # Skip empty clusters
                continue
                
            content_embs = data['content_embeddings']
            texts = data['texts']
            metadata = data['metadata']
            
            # Convert content embeddings to numpy array
            content_embs_np = np.array(content_embs)
            if len(content_embs_np.shape) == 3:
                content_embs_np = content_embs_np.squeeze(1)
            
            # Create appropriate FAISS index based on cluster size
            cluster_size = len(content_embs_np)
            
            if cluster_size > 200:  # Use IVF for larger clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = min(max(cluster_size // 30, 4), 64)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
                if cluster_size >= nlist * 2:  # Ensure sufficient training data
                    index.train(content_embs_np.astype(np.float32))
                    index.add(content_embs_np.astype(np.float32))
                else:
                    # Fallback to flat index
                    index = faiss.IndexFlatIP(self.embedding_dim)
                    index.add(content_embs_np.astype(np.float32))
            else:
                # Use flat index for smaller clusters
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(content_embs_np.astype(np.float32))
            
            # Store cluster data
            self.content_indices[cluster_id] = index
            self.cluster_metadata[cluster_id] = metadata
            self.cluster_content_embeddings[cluster_id] = content_embs_np
            self.cluster_texts[cluster_id] = texts
        
        logger.info(f"Built hierarchical index with {len(self.content_indices)} content clusters")
    
    def _find_relevant_metadata_clusters(self, metadata_embedding: np.ndarray, 
                                       num_clusters: int = 5) -> List[str]:
        """Find most relevant metadata clusters for the query"""
        if self.metadata_cluster_index is None or len(self.metadata_clusters) == 0:
            logger.warning("No metadata clusters available")
            return []
        
        # Handle embedding shape and reduce dimension
        if len(metadata_embedding.shape) == 2:
            metadata_embedding = metadata_embedding.squeeze(0)
        
        # Reduce query metadata embedding dimension
        query_metadata_reduced = self._reduce_metadata_embeddings(
            metadata_embedding.reshape(1, -1)
        ).flatten()
        
        # Normalize
        norm = np.linalg.norm(query_metadata_reduced)
        if norm > 0:
            query_metadata_reduced = query_metadata_reduced / norm
        
        # Search metadata cluster index
        search_k = min(num_clusters, len(self.metadata_clusters))
        
        try:
            scores, indices = self.metadata_cluster_index.search(
                query_metadata_reduced.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # Return relevant cluster IDs
            cluster_ids = list(self.metadata_clusters.keys())
            relevant_clusters = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(cluster_ids):
                    cluster_id = cluster_ids[idx]
                    # Only include clusters that have content indices
                    if cluster_id in self.content_indices:
                        relevant_clusters.append(cluster_id)
                        logger.debug(f"Selected metadata cluster {cluster_id} with score {score:.3f}")
            
            return relevant_clusters
            
        except Exception as e:
            logger.error(f"Error in metadata cluster search: {e}")
            return [cid for cid in list(self.metadata_clusters.keys())[:search_k] 
                   if cid in self.content_indices]
    
    def _search_content_cluster(self, cluster_id: str, query_embedding: np.ndarray, 
                              k: int) -> List[Tuple[float, str, int, Dict, str]]:
        """Search within a specific content cluster"""
        if cluster_id not in self.content_indices:
            return []
        
        index = self.content_indices[cluster_id]
        metadata_list = self.cluster_metadata[cluster_id]
        texts_list = self.cluster_texts[cluster_id]
        
        # Set nprobe for IVF indices
        if hasattr(index, 'nprobe'):
            index.nprobe = min(getattr(config, 'NPROBE', 10), getattr(index, 'nlist', 10))
        
        try:
            # Handle query embedding shape
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding.squeeze(0)
            
            # Search within this cluster using content embedding
            search_k = min(k, len(metadata_list))
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # Collect results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(metadata_list):
                    results.append((
                        float(score),
                        cluster_id,
                        idx,
                        metadata_list[idx],
                        texts_list[idx]
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching content cluster {cluster_id}: {e}")
            return []
    
    def search(self, query_text: str, query_embedding: np.ndarray, 
              query_metadata: Dict, metadata_embedding: np.ndarray, 
              k: int = 5, num_metadata_clusters: int = 5) -> List[Tuple[float, int, Dict, str]]:
        """
        Scalable hierarchical search:
        1. Find relevant metadata clusters using query metadata
        2. Search content within selected clusters using query content embedding
        """
        # Step 1: Find relevant metadata clusters
        relevant_clusters = self._find_relevant_metadata_clusters(
            metadata_embedding, num_metadata_clusters
        )
        
        if not relevant_clusters:
            logger.warning("No relevant metadata clusters found")
            return []
        
        logger.debug(f"Found {len(relevant_clusters)} relevant metadata clusters")
        
        # Step 2: Search content within relevant clusters
        all_results = []
        results_per_cluster = max(1, (k * 2) // len(relevant_clusters))
        
        # Parallel search across relevant clusters
        with ThreadPoolExecutor(max_workers=min(len(relevant_clusters), 6)) as executor:
            future_to_cluster = {
                executor.submit(self._search_content_cluster, cluster_id, query_embedding, results_per_cluster): cluster_id
                for cluster_id in relevant_clusters
            }
            
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    cluster_results = future.result(timeout=15)
                    all_results.extend(cluster_results)
                    logger.debug(f"Found {len(cluster_results)} results in cluster {cluster_id}")
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
        
        # Sort by similarity score and return top k
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Total results from {len(relevant_clusters)} clusters: {len(all_results)}, returning top {k}")
        
        # Format results for compatibility
        final_results = []
        for score, cluster_id, local_idx, metadata, text in all_results[:k]:
            final_results.append((score, local_idx, metadata, text))
        
        return final_results
    
    def save_index(self, path: str):
        """Save the scalable hierarchical index to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save metadata cluster index
        if self.metadata_cluster_index is not None:
            faiss.write_index(self.metadata_cluster_index, os.path.join(path, "metadata_clusters.faiss"))
        
        # Save content indices
        for cluster_id, index in self.content_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"content_{safe_name}.index"))
        
        # Save all metadata and mappings
        metadata_file = os.path.join(path, "scalable_index_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata_clusters': self.metadata_clusters,
                'cluster_metadata': self.cluster_metadata,
                'cluster_texts': self.cluster_texts,
                'cluster_assignments': self.cluster_assignments,
                'next_chunk_id': self.next_chunk_id,
                'embedding_dim': self.embedding_dim,
                'metadata_dim': self.metadata_dim,
                'metadata_pca': self.metadata_pca,
                'metadata_cluster_centroids': self.metadata_cluster_centroids
            }, f)
        
        logger.info(f"Saved scalable hierarchical index to {path}")
    
    def load_index(self, path: str):
        """Load the scalable hierarchical index from disk"""
        # Load metadata cluster index
        metadata_index_path = os.path.join(path, "metadata_clusters.faiss")
        if os.path.exists(metadata_index_path):
            self.metadata_cluster_index = faiss.read_index(metadata_index_path)
        
        # Load metadata and mappings
        metadata_file = os.path.join(path, "scalable_index_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata_clusters = data['metadata_clusters']
                self.cluster_metadata = data['cluster_metadata']
                self.cluster_texts = data['cluster_texts']
                self.cluster_assignments = data['cluster_assignments']
                self.next_chunk_id = data['next_chunk_id']
                self.embedding_dim = data['embedding_dim']
                self.metadata_dim = data.get('metadata_dim', 128)
                self.metadata_pca = data.get('metadata_pca')
                self.metadata_cluster_centroids = data.get('metadata_cluster_centroids')
        
        # Load content indices
        for cluster_id in self.metadata_clusters.keys():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"content_{safe_name}.index")
            if os.path.exists(index_path):
                self.content_indices[cluster_id] = faiss.read_index(index_path)
        
        logger.info(f"Loaded scalable hierarchical index from {path}")

    def get_index_stats(self):
        """Get statistics about the index"""
        stats = {
            'metadata_clusters': len(self.metadata_clusters),
            'content_clusters': len(self.content_indices),
            'total_chunks': self.next_chunk_id,
            'avg_chunks_per_cluster': 0,
            'metadata_dim': self.metadata_dim,
            'content_dim': self.embedding_dim
        }
        
        if self.cluster_texts:
            total_texts = sum(len(texts) for texts in self.cluster_texts.values())
            stats['avg_chunks_per_cluster'] = total_texts / len(self.cluster_texts) if self.cluster_texts else 0
        
        return stats


# Update the main processing functions to use the new scalable index
def build_index_from_documents():
    """Build and save the scalable hierarchical index from documents"""
    # Use the new scalable index with reduced metadata dimension
    faiss_index = ScalableHierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=128  # Reduced dimension for faster clustering
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

    # Compute embeddings for chunks
    logger.info("Computing content embeddings for chunks...")
    chunk_embeddings = []
    for chunk in all_chunks:
        chunk_embedding = compute_embeddings(chunk)
        chunk_embedding = normalize_embeddings(chunk_embedding)
        chunk_embeddings.append(chunk_embedding)

    logger.info("Computing metadata embeddings for chunks...")
    metadata_embeddings = []
    for metadata in all_metadata:
        metadata_embedding = compute_embeddings(metadata_embedding_text(metadata))
        metadata_embedding = normalize_embeddings(metadata_embedding)
        metadata_embeddings.append(metadata_embedding)

    # Build scalable hierarchical FAISS index
    logger.info("Building scalable hierarchical FAISS index...")
    faiss_index.build_hierarchical_index(chunk_embeddings, metadata_embeddings, all_chunks, all_metadata)
    
    # Print index statistics
    stats = faiss_index.get_index_stats()
    logger.info(f"Index Statistics: {stats}")
    
    # Save the index
    if not os.path.exists(config.INDEX_PATH):
        os.makedirs(config.INDEX_PATH)
    faiss_index.save_index(config.INDEX_PATH)
    
    logger.info("Scalable index built and saved successfully!")
    return faiss_index


def load_existing_index():
    """Load pre-built scalable hierarchical index"""
    faiss_index = ScalableHierarchicalFAISSIndex(
        embedding_dim=config.EMBEDDING_DIM,
        metadata_dim=128
    )
    
    try:
        faiss_index.load_index(config.INDEX_PATH)
        stats = faiss_index.get_index_stats()
        logger.info(f"Successfully loaded existing index! Stats: {stats}")
        return faiss_index
    except Exception as e:
        logger.error(f"Failed to load existing index: {e}")
        return None


def check_index_exists():
    """Check if a pre-built scalable index exists"""
    index_files = [
        os.path.join(config.INDEX_PATH, "metadata_clusters.faiss"),
        os.path.join(config.INDEX_PATH, "scalable_index_metadata.pkl")
    ]
    return all(os.path.exists(f) for f in index_files)