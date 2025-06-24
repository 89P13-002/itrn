# faiss_index.py
import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

class HierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 512, metadata_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim

        self.metadata_clusters = {}           # metadata_cluster_id -> cluster info
        self.metadata_cluster_centroids = None# centroids of metadata clusters
        self.metadata_cluster_index = None    # FAISS index for metadata clusters
        
        self.content_indices = {}             # metadata_cluster_id -> content FAISS indices
        self.cluster_metadata = {}            # metadata_cluster_id -> list of chunk metadata
        self.cluster_content_embeddings = {}  # metadata_cluster_id -> content embeddings
        self.cluster_texts = {}               # metadata_cluster_id -> chunk texts
        self.cluster_chunk_ids = {}           # metadata_cluster_id -> chunk IDs
        self.cluster_assignments = {}         # chunk_idx -> list of assigned metadata_cluster_ids
        
        self.lock = threading.Lock()
    
    def _build_metadata_clusters(self, metadata_embeddings: List[np.ndarray], 
                                metadata_list: List[Dict[str, str]], 
                                num_clusters: int = 50):
        logger.info("Building metadata-based coarse clusters...")
        
        metadata_embeddings = np.array(metadata_embeddings)
        if len(metadata_embeddings.shape) == 3:
            metadata_embeddings = metadata_embeddings.squeeze(1)
        
        n_samples = len(metadata_embeddings)
        optimal_clusters = min(num_clusters, max(5, n_samples // 20))
        
        logger.info(f"Performing metadata K-means clustering with {optimal_clusters} clusters on {n_samples} samples")
        
        kmeans = KMeans(
            n_clusters=optimal_clusters, 
            random_state=42, 
            n_init=5, 
            max_iter=5000,
            algorithm='lloyd' 
        )
        cluster_labels = kmeans.fit_predict(metadata_embeddings)
        
        self.metadata_cluster_centroids = kmeans.cluster_centers_
        self.metadata_cluster_index = faiss.IndexFlatIP(self.metadata_cluster_centroids.shape[1])
        self.metadata_cluster_index.add(self.metadata_cluster_centroids.astype(np.float32))
        
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
                                  chunk_ids: List[str],
                                  top_k_clusters: int = 3):
        logger.info(f"Assigning chunks to top-{top_k_clusters} metadata clusters")
        
        metadata_embeddings_np = np.array(metadata_embeddings)
        if len(metadata_embeddings_np.shape) == 3:
            metadata_embeddings_np = metadata_embeddings_np.squeeze(1)
        
        cluster_data = defaultdict(lambda: {
            'content_embeddings': [], 
            'texts': [], 
            'metadata': [],
            'chunk_ids': [],
            'chunk_indices': []
        })
        
        # For each chunk, find top-k most similar metadata clusters assign them to it Fuzzy assignment
        for chunk_idx, (chunk_text, content_emb, metadata_emb, metadata, chunk_id) in enumerate(zip(
            chunk_list, content_embeddings, metadata_embeddings, metadata_list, chunk_ids
        )):
            
            # Search for most similar metadata clusters
            scores, indices = self.metadata_cluster_index.search(
                metadata_embeddings_np[chunk_idx:chunk_idx+1].astype(np.float32),
                min(top_k_clusters, len(self.metadata_clusters))
            )
            
            # Assign chunk to top-k clusters
            assigned_clusters = []
            cluster_ids = list(self.metadata_clusters.keys())
            
            for score, cluster_idx in zip(scores[0], indices[0]):
                if cluster_idx != -1 and cluster_idx < len(cluster_ids):
                    cluster_id = cluster_ids[cluster_idx]
                    assigned_clusters.append(cluster_id)
                    
                    # Adding chunk to this cluster
                    cluster_data[cluster_id]['content_embeddings'].append(content_emb)
                    cluster_data[cluster_id]['texts'].append(chunk_text)
                    cluster_data[cluster_id]['metadata'].append(metadata)
                    cluster_data[cluster_id]['chunk_ids'].append(chunk_id)
                    cluster_data[cluster_id]['chunk_indices'].append(chunk_idx)
            
            self.cluster_assignments[chunk_idx] = assigned_clusters
                    
        return cluster_data

    def _create_hnsw_index(self, content_embs_np: np.ndarray, cluster_size: int):      
        if cluster_size <= 20:
            M = 8
            efConstruction = 40
        elif cluster_size <= 100:
            M = 16
            efConstruction = 64
        elif cluster_size <= 500:
            M = 24
            efConstruction = 128
        elif cluster_size <= 2000:
            M = 32
            efConstruction = 200
        else:
            M = 40
            efConstruction = 256

        logger.info(f"Creating HNSW indexing with M = {M} for cluster size {cluster_size} and efConstruction {efConstruction}")
        
        try:
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            index.hnsw.efConstruction = efConstruction
            index.add(content_embs_np.astype(np.float32))
            
            logger.info("Done indexing with HNSW")
            return index
        except Exception as e:
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(content_embs_np.astype(np.float32))

            logger.info(f"Done indexing with FLAT due to error : {e}")
            return index

    def build_hierarchical_index(self, content_embeddings: List[np.ndarray], 
                                metadata_embeddings: List[np.ndarray], 
                                chunk_list: List[str], 
                                metadata_list: List[Dict[str, str]],
                                chunk_ids: List[str]):
        logger.info("Building scalable hierarchical FAISS index...")
        
        # Step 1: Building coarse metadata clusters
        self._build_metadata_clusters(metadata_embeddings, metadata_list)
        
        # Step 2: Assign chunks to top-k most similar metadata clusters
        cluster_data = self._assign_chunks_to_clusters(
            chunk_list, content_embeddings, metadata_embeddings, metadata_list, chunk_ids
        )
        
        # Step 3: Building fine grained cluster w.r.t chunks
        logger.info("Building chunk based finer cluster.")
        
        for cluster_id, data in cluster_data.items():
            if not data['texts']: 
                continue
                
            content_embs = data['content_embeddings']
            texts = data['texts']
            metadata = data['metadata']
            chunk_ids_list = data['chunk_ids']
            
            content_embs_np = np.array(content_embs)
            if len(content_embs_np.shape) == 3:
                content_embs_np = content_embs_np.squeeze(1)
            
            cluster_size = len(content_embs_np)

            # Indexing by HNSW
            index = self._create_hnsw_index(content_embs_np, cluster_size)

            self.content_indices[cluster_id] = index
            self.cluster_metadata[cluster_id] = metadata
            self.cluster_content_embeddings[cluster_id] = content_embs_np
            self.cluster_texts[cluster_id] = texts
            self.cluster_chunk_ids[cluster_id] = chunk_ids_list
        
        logger.info(f"Built hierarchical index with {len(self.content_indices)} content clusters")
    
    def _find_relevant_metadata_clusters(self, metadata_embedding: np.ndarray, 
                                       num_clusters: int = 5) -> List[str]:
        if self.metadata_cluster_index is None or len(self.metadata_clusters) == 0:
            logger.warning("No metadata clusters available")
            return []
        
        if len(metadata_embedding.shape) == 2:
            metadata_embedding = metadata_embedding.squeeze(0)
        
        norm = np.linalg.norm(metadata_embedding)
        if norm > 0:
            metadata_embedding = metadata_embedding / norm
        
        search_k = min(num_clusters, len(self.metadata_clusters))
        
        try:
            scores, indices = self.metadata_cluster_index.search(
                metadata_embedding.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # get relevant metadata cluster 
            cluster_ids = list(self.metadata_clusters.keys())
            relevant_clusters = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(cluster_ids):
                    cluster_id = cluster_ids[idx]
                    # Only include clusters that have content indices
                    if cluster_id in self.content_indices:
                        relevant_clusters.append(cluster_id)
                        logger.info(f"Selected metadata cluster {cluster_id} with score {score:.3f}")
            
            return relevant_clusters
            
        except Exception as e:
            logger.error(f"Error in metadata cluster search: {e}")
            return [cid for cid in list(self.metadata_clusters.keys())[:search_k] 
                   if cid in self.content_indices]
    
    def _search_content_cluster(self, cluster_id: str, query_embedding: np.ndarray, 
                              k: int, ef_search: Optional[int] = None) -> List[Tuple[float, str, int, Dict, str, str]]:
        if cluster_id not in self.content_indices:
            return []
        
        index = self.content_indices[cluster_id]
        metadata_list = self.cluster_metadata[cluster_id]
        texts_list = self.cluster_texts[cluster_id]
        chunk_ids_list = self.cluster_chunk_ids[cluster_id]
        
        # HNSW indexing case
        if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
            cluster_size = len(metadata_list)

            if ef_search is not None:
                index.hnsw.efSearch = ef_search
            else:
                if cluster_size <= 50:
                    index.hnsw.efSearch = max(k*4, 32)
                elif cluster_size <= 200:
                    index.hnsw.efSearch = max(k*3, 64)
                elif cluster_size <= 1000:
                    index.hnsw.efSearch = max(k*2, 80)
                else:
                    index.hnsw.efSearch = max(2*k, 100)
                
                index.hnsw.efSearch = min(index.hnsw.efSearch, min(cluster_size, 512))
        
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
                    results.append((
                        float(score),
                        cluster_id,
                        idx,
                        metadata_list[idx],
                        texts_list[idx],
                        chunk_ids_list[idx]  # Include chunk ID
                    ))
            
            return results

        except Exception as e:
            logger.error(f"Error in searching content cluster {cluster_id}: {e}")
            return []

    def search(self, query_text: str, query_embedding: np.ndarray, 
              query_metadata: Dict, metadata_embedding: np.ndarray, 
              k: int = 5, num_metadata_clusters: int = 5, ef_search: Optional[int] = None) -> List[Tuple[float, int, Dict, str, str]]:
        """
        Hierarchical FAISS:
        1. First find relevant metadata clusters using query metadata
        2. Then search content within selected metadata clusters using query content embedding
        Returns: List of (score, local_idx, metadata, text, chunk_id)
        """
        # Step 1: Finding relevant metadata clusters
        relevant_clusters = self._find_relevant_metadata_clusters(
            metadata_embedding, num_metadata_clusters
        )
        
        if not relevant_clusters:
            logger.warning("No relevant metadata clusters found")
            return []
        
        logger.info(f"Found {len(relevant_clusters)} relevant metadata clusters")
        
        # Step 2: Now search for the content within relevant clusters
        all_results = []
        results_per_cluster = max(1, (k * 2) // len(relevant_clusters))
        
        with ThreadPoolExecutor(max_workers=min(len(relevant_clusters), 6)) as executor:
            future_to_cluster = {
                executor.submit(self._search_content_cluster, cluster_id, query_embedding, results_per_cluster, ef_search): cluster_id
                for cluster_id in relevant_clusters
            }
            
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    cluster_results = future.result(timeout=15)
                    all_results.extend(cluster_results)
                    logger.info(f"Found {len(cluster_results)} results in cluster {cluster_id}")
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
        
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Total results from {len(relevant_clusters)} clusters: {len(all_results)}, and returning top {k}")
        
        final_results = []
        for score, cluster_id, local_idx, metadata, text, chunk_id in all_results[:k]:
            final_results.append((score, local_idx, metadata, text, chunk_id))
        
        return final_results
    
    def save_index(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        if self.metadata_cluster_index is not None:
            faiss.write_index(self.metadata_cluster_index, os.path.join(path, "metadata_clusters.faiss"))
        
        for cluster_id, index in self.content_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"content_{safe_name}.index"))
        
        metadata_file = os.path.join(path, "metadata_index.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata_clusters': self.metadata_clusters,
                'cluster_metadata': self.cluster_metadata,
                'cluster_texts': self.cluster_texts,
                'cluster_chunk_ids': self.cluster_chunk_ids,
                'cluster_assignments': self.cluster_assignments,
                'embedding_dim': self.embedding_dim,
                'metadata_dim': self.metadata_dim,
                'metadata_cluster_centroids': self.metadata_cluster_centroids
            }, f)
        
        logger.info(f"Saved hierarchical faiss index to {path}")
    
    def load_index(self, path: str):
        metadata_index_path = os.path.join(path, "metadata_clusters.faiss")
        if os.path.exists(metadata_index_path):
            self.metadata_cluster_index = faiss.read_index(metadata_index_path)
        
        metadata_file = os.path.join(path, "metadata_index.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata_clusters = data['metadata_clusters']
                self.cluster_metadata = data['cluster_metadata']
                self.cluster_texts = data['cluster_texts']
                self.cluster_chunk_ids = data.get('cluster_chunk_ids', {})
                self.cluster_assignments = data['cluster_assignments']
                self.embedding_dim = data['embedding_dim']
                self.metadata_dim = data.get('metadata_dim', 128)
                self.metadata_cluster_centroids = data.get('metadata_cluster_centroids')
        
        for cluster_id in self.metadata_clusters.keys():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"content_{safe_name}.index")
            if os.path.exists(index_path):
                self.content_indices[cluster_id] = faiss.read_index(index_path)
        
        logger.info(f"Loaded hierarchical index from {path}")
