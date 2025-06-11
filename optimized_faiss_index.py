import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import threading
from config import config
from llm_fun import get_metadata_from_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


class SkipListNode:
    """Node for skip list implementation"""
    def __init__(self, score: float, data: any, level: int):
        self.score = score
        self.data = data
        self.forward = [None] * (level + 1)


class SkipList:
    """Skip list for efficient range queries and insertions"""
    def __init__(self, max_level: int = 16):
        self.max_level = max_level
        self.header = SkipListNode(-float('inf'), None, max_level)
        self.level = 0
    
    def _random_level(self) -> int:
        """Generate random level for new node"""
        level = 0
        while np.random.random() < 0.5 and level < self.max_level:
            level += 1
        return level
    
    def insert(self, score: float, data: any):
        """Insert element with given score"""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find position to insert
        for i in range(self.level, -1, -1):
            while (current.forward[i] is not None and 
                   current.forward[i].score > score):  # Descending order
                current = current.forward[i]
            update[i] = current
        
        # Generate random level for new node
        new_level = self._random_level()
        
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level
        
        # Create and insert new node
        new_node = SkipListNode(score, data, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def get_top_k(self, k: int) -> List[Tuple[float, any]]:
        """Get top k elements (highest scores)"""
        results = []
        current = self.header.forward[0]
        count = 0
        
        while current is not None and count < k:
            results.append((current.score, current.data))
            current = current.forward[0]
            count += 1
        
        return results


class OptimizedHierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 384, metadata_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim  # Reduced dimension for metadata
        
        # Metadata clustering components
        self.metadata_clusters = {}         # cluster_id -> cluster info
        self.metadata_cluster_centroids = None
        self.metadata_pca = None           # PCA for dimension reduction
        self.metadata_kmeans = None        # Mini-batch K-means for metadata
        
        # Text embedding indices with skip lists
        self.cluster_indices = {}          # cluster_id -> FAISS index
        self.cluster_skip_lists = {}       # cluster_id -> SkipList for fast retrieval
        self.cluster_metadata = {}         # cluster_id -> list of chunk metadata
        self.cluster_texts = {}            # cluster_id -> list of chunk texts
        self.cluster_embeddings = {}       # cluster_id -> chunk embeddings
        
        # Chunk to cluster assignments
        self.chunk_to_clusters = {}        # chunk_idx -> list of assigned cluster_ids
        self.global_chunk_count = 0
        
        self.lock = threading.Lock()
    
    def _prepare_metadata_features(self, metadata_list: List[Dict[str, str]]) -> np.ndarray:
        """Convert metadata to numerical features and reduce dimensions"""
        logger.info("Preparing metadata features...")
        
        # Extract and encode metadata features
        features = []
        for metadata in metadata_list:
            feature_vec = []
            
            # Gender encoding (one-hot)
            gender = metadata.get('gender', 'unknown').lower()
            feature_vec.extend([
                1 if gender == 'male' else 0,
                1 if gender == 'female' else 0,
                1 if gender == 'any' or gender == 'unknown' else 0
            ])
            
            # Age group encoding (one-hot)
            age_group = metadata.get('age_group', 'unknown').lower()
            feature_vec.extend([
                1 if age_group == 'child' else 0,
                1 if age_group == 'teenager' else 0,
                1 if age_group == 'young_adult' else 0,
                1 if age_group == 'middle_aged' else 0,
                1 if age_group == 'senior' else 0,
                1 if age_group == 'any' or age_group == 'unknown' else 0
            ])
            
            # Location encoding (simple categorical)
            location = metadata.get('location', 'general').lower()
            feature_vec.extend([
                1 if 'urban' in location else 0,
                1 if 'rural' in location else 0,
                1 if 'cold' in location or 'winter' in location else 0,
                1 if 'hot' in location or 'sunny' in location else 0,
                1 if location == 'general' else 0
            ])
            
            # Keywords count (simple numerical features)
            keywords = metadata.get('keywords', '')
            keyword_count = len(keywords.split(',')) if keywords else 0
            feature_vec.append(min(keyword_count / 10.0, 1.0))  # Normalized
            
            # Summary length (normalized)
            summary = metadata.get('summary', '')
            summary_length = min(len(summary) / 200.0, 1.0)  # Normalized
            feature_vec.append(summary_length)
            
            features.append(feature_vec)
        
        features = np.array(features, dtype=np.float32)
        
        # Apply PCA for dimension reduction
        if features.shape[1] > self.metadata_dim:
            self.metadata_pca = PCA(n_components=self.metadata_dim, random_state=42)
            features = self.metadata_pca.fit_transform(features)
        else:
            # Pad if needed
            if features.shape[1] < self.metadata_dim:
                padding = np.zeros((features.shape[0], self.metadata_dim - features.shape[1]))
                features = np.hstack([features, padding])
        
        logger.info(f"Metadata features shape: {features.shape}")
        return features.astype(np.float32)
    
    def _cluster_metadata(self, metadata_features: np.ndarray, num_clusters: int = 20) -> np.ndarray:
        """Cluster metadata using Mini-batch K-means"""
        logger.info(f"Clustering metadata into {num_clusters} clusters...")
        
        # Use Mini-batch K-means for scalability
        n_samples = len(metadata_features)
        batch_size = min(1000, max(100, n_samples // 10))
        
        self.metadata_kmeans = MiniBatchKMeans(
            n_clusters=min(num_clusters, n_samples),
            batch_size=batch_size,
            random_state=42,
            max_iter=100,
            n_init=3
        )
        
        cluster_labels = self.metadata_kmeans.fit_predict(metadata_features)
        self.metadata_cluster_centroids = self.metadata_kmeans.cluster_centers_
        
        logger.info(f"Created {len(np.unique(cluster_labels))} metadata clusters")
        return cluster_labels
    
    def _assign_chunks_to_clusters(self, content_embeddings: List[np.ndarray],
                                  metadata_features: np.ndarray,
                                  chunk_list: List[str],
                                  metadata_list: List[Dict[str, str]],
                                  top_k_clusters: int = 3):
        """Assign each chunk to top-k most similar metadata clusters"""
        logger.info("Assigning chunks to metadata clusters...")
        
        # Cluster metadata
        cluster_labels = self._cluster_metadata(metadata_features)
        
        # Initialize cluster storage
        for cluster_id in np.unique(cluster_labels):
            cluster_key = f"meta_cluster_{cluster_id}"
            self.cluster_indices[cluster_key] = None
            self.cluster_skip_lists[cluster_key] = SkipList()
            self.cluster_metadata[cluster_key] = []
            self.cluster_texts[cluster_key] = []
            self.cluster_embeddings[cluster_key] = []
        
        # For each chunk, find top-k most similar metadata clusters
        for chunk_idx, (content_emb, metadata_feat, text, metadata) in enumerate(
            zip(content_embeddings, metadata_features, chunk_list, metadata_list)):
            
            # Calculate distances to all cluster centroids
            distances = np.linalg.norm(
                self.metadata_cluster_centroids - metadata_feat.reshape(1, -1), 
                axis=1
            )
            
            # Get top-k closest clusters
            top_clusters_idx = np.argsort(distances)[:top_k_clusters]
            assigned_clusters = []
            
            for cluster_idx in top_clusters_idx:
                cluster_key = f"meta_cluster_{cluster_idx}"
                assigned_clusters.append(cluster_key)
                
                # Add chunk to this cluster
                self.cluster_metadata[cluster_key].append(metadata)
                self.cluster_texts[cluster_key].append(text)
                self.cluster_embeddings[cluster_key].append(content_emb)
            
            # Store chunk to cluster mapping
            self.chunk_to_clusters[chunk_idx] = assigned_clusters
            self.global_chunk_count += 1
        
        logger.info(f"Assigned {len(chunk_list)} chunks to clusters")
    
    def _build_cluster_indices(self):
        """Build FAISS indices and skip lists for each cluster"""
        logger.info("Building FAISS indices for clusters...")
        
        for cluster_key, embeddings_list in self.cluster_embeddings.items():
            if not embeddings_list:
                continue
            
            # Convert embeddings to numpy array
            embeddings = np.array(embeddings_list)
            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(1)
            
            cluster_size = len(embeddings)
            logger.info(f"Building index for {cluster_key} with {cluster_size} chunks")
            
            # Choose appropriate FAISS index based on cluster size
            if cluster_size > 1000:
                # Use HNSW for large clusters
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                index.hnsw.efConstruction = 64
                index.hnsw.efSearch = 32
                index.add(embeddings.astype(np.float32))
            elif cluster_size > 100:
                # Use IVF for medium clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = min(max(cluster_size // 20, 4), 20)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                index.train(embeddings.astype(np.float32))
                index.add(embeddings.astype(np.float32))
                index.nprobe = min(5, nlist)
            else:
                # Use flat index for small clusters
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(embeddings.astype(np.float32))
            
            self.cluster_indices[cluster_key] = index
    
    def build_hierarchical_index(self, content_embeddings: List[np.ndarray],
                                metadata_list: List[Dict[str, str]],
                                chunk_list: List[str]):
        """Build optimized hierarchical index"""
        logger.info("Building optimized hierarchical FAISS index...")
        
        # Prepare metadata features
        metadata_features = self._prepare_metadata_features(metadata_list)
        
        # Assign chunks to clusters
        self._assign_chunks_to_clusters(
            content_embeddings, metadata_features, chunk_list, metadata_list
        )
        
        # Build FAISS indices for each cluster
        self._build_cluster_indices()
        
        logger.info(f"Built hierarchical index with {len(self.cluster_indices)} clusters")
    
    def _find_relevant_clusters(self, query_metadata: Dict[str, str], top_k: int = 5) -> List[str]:
        """Find most relevant metadata clusters for query"""
        if self.metadata_pca is None or self.metadata_kmeans is None:
            return list(self.cluster_indices.keys())[:top_k]
        
        # Convert query metadata to features
        query_features = self._prepare_metadata_features([query_metadata])
        
        # Transform using same PCA
        if hasattr(self.metadata_pca, 'transform'):
            query_features = self.metadata_pca.transform(query_features)
        
        # Find distances to cluster centroids
        distances = np.linalg.norm(
            self.metadata_cluster_centroids - query_features.reshape(1, -1),
            axis=1
        )
        
        # Get top-k closest clusters
        top_clusters_idx = np.argsort(distances)[:top_k]
        relevant_clusters = [f"meta_cluster_{idx}" for idx in top_clusters_idx]
        
        logger.debug(f"Selected clusters: {relevant_clusters}")
        return relevant_clusters
    
    def _search_cluster_with_skiplist(self, cluster_key: str, query_embedding: np.ndarray, k: int) -> List[Tuple[float, int, Dict, str]]:
        """Search within cluster using FAISS + skip list optimization"""
        if cluster_key not in self.cluster_indices:
            return []
        
        index = self.cluster_indices[cluster_key]
        metadata_list = self.cluster_metadata[cluster_key]
        texts_list = self.cluster_texts[cluster_key]
        
        if len(metadata_list) == 0:
            return []
        
        try:
            # Handle query embedding shape
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding.squeeze(0)
            
            # Search FAISS index
            search_k = min(k * 2, len(metadata_list))  # Get more candidates
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # Use skip list for final ranking (optional optimization)
            skip_list = SkipList()
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(metadata_list):
                    skip_list.insert(float(score), (idx, metadata_list[idx], texts_list[idx]))
            
            # Get top-k from skip list
            top_results = skip_list.get_top_k(k)
            
            results = []
            for score, (idx, metadata, text) in top_results:
                results.append((score, idx, metadata, text))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching cluster {cluster_key}: {e}")
            return []
    
    def search(self, query_text: str, query_embedding: np.ndarray,
              query_metadata: Dict, k: int = 5, num_clusters: int = 5) -> List[Tuple[float, int, Dict, str]]:
        """Optimized hierarchical search"""
        logger.debug(f"Searching with query: {query_text[:50]}...")
        
        # Find relevant metadata clusters
        relevant_clusters = self._find_relevant_clusters(query_metadata, num_clusters)
        
        # Search within relevant clusters in parallel
        all_results = []
        results_per_cluster = max(1, k // len(relevant_clusters)) if relevant_clusters else k
        
        with ThreadPoolExecutor(max_workers=min(len(relevant_clusters), 4)) as executor:
            future_to_cluster = {
                executor.submit(
                    self._search_cluster_with_skiplist, 
                    cluster_key, 
                    query_embedding, 
                    results_per_cluster + 2
                ): cluster_key
                for cluster_key in relevant_clusters
            }
            
            for future in as_completed(future_to_cluster):
                cluster_key = future_to_cluster[future]
                try:
                    cluster_results = future.result(timeout=10)
                    all_results.extend(cluster_results)
                    logger.debug(f"Found {len(cluster_results)} results in {cluster_key}")
                except Exception as e:
                    logger.error(f"Error in parallel search for {cluster_key}: {e}")
        
        # Final ranking using skip list
        final_skip_list = SkipList()
        for score, idx, metadata, text in all_results:
            final_skip_list.insert(score, (idx, metadata, text))
        
        # Get top-k results
        top_results = final_skip_list.get_top_k(k)
        final_results = []
        for score, (idx, metadata, text) in top_results:
            final_results.append((score, idx, metadata, text))
        
        logger.debug(f"Returning {len(final_results)} final results")
        return final_results
    
    def save_index(self, path: str):
        """Save the optimized hierarchical index"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS indices
        for cluster_key, index in self.cluster_indices.items():
            if index is not None:
                safe_name = cluster_key.replace('/', '_').replace('\\', '_')
                faiss.write_index(index, os.path.join(path, f"{safe_name}.index"))
        
        # Save metadata and cluster info
        metadata_file = os.path.join(path, "optimized_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata_clusters': self.metadata_clusters,
                'metadata_cluster_centroids': self.metadata_cluster_centroids,
                'metadata_pca': self.metadata_pca,
                'metadata_kmeans': self.metadata_kmeans,
                'cluster_metadata': self.cluster_metadata,
                'cluster_texts': self.cluster_texts,
                'chunk_to_clusters': self.chunk_to_clusters,
                'global_chunk_count': self.global_chunk_count,
                'embedding_dim': self.embedding_dim,
                'metadata_dim': self.metadata_dim
            }, f)
        
        logger.info(f"Saved optimized hierarchical index to {path}")
    
    def load_index(self, path: str):
        """Load the optimized hierarchical index"""
        # Load metadata and cluster info
        metadata_file = os.path.join(path, "optimized_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata_clusters = data['metadata_clusters']
                self.metadata_cluster_centroids = data['metadata_cluster_centroids']
                self.metadata_pca = data['metadata_pca']
                self.metadata_kmeans = data['metadata_kmeans']
                self.cluster_metadata = data['cluster_metadata']
                self.cluster_texts = data['cluster_texts']
                self.chunk_to_clusters = data['chunk_to_clusters']
                self.global_chunk_count = data['global_chunk_count']
                self.embedding_dim = data['embedding_dim']
                self.metadata_dim = data['metadata_dim']
        
        # Load FAISS indices
        for cluster_key in self.cluster_metadata.keys():
            safe_name = cluster_key.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"{safe_name}.index")
            if os.path.exists(index_path):
                self.cluster_indices[cluster_key] = faiss.read_index(index_path)
            
            # Reinitialize skip lists
            self.cluster_skip_lists[cluster_key] = SkipList()
        
        logger.info(f"Loaded optimized hierarchical index from {path}")