# faiss_index.py
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

class HierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
        # Hierarchical structure
        self.coarse_clusters = {}           # coarse_cluster_id -> cluster info
        self.coarse_cluster_embeddings = None  # embeddings of coarse cluster centroids
        self.coarse_cluster_index = None    # FAISS index for coarse clusters
        
        self.fine_cluster_indices = {}      # coarse_cluster_id -> fine FAISS indices
        self.fine_cluster_metadata = {}     # coarse_cluster_id -> list of chunk metadata
        self.fine_cluster_embeddings = {}   # coarse_cluster_id -> chunk embeddings
        self.fine_cluster_texts = {}        # coarse_cluster_id -> chunk texts
        
        self.global_to_local = {}           # global_id -> (coarse_id, fine_id)
        self.next_global_id = 0
        self.lock = threading.Lock()
    
    def _build_coarse_clusters(self, content_embeddings: List[np.ndarray], 
                              metadata_embeddings: List[np.ndarray], 
                              chunk_list: List[str], 
                              metadata_list: List[Dict[str, str]], 
                              num_coarse_clusters: int = 100):
        """Build coarse clusters using weighted combination of content and metadata embeddings"""
        logger.info("Building coarse clusters from combined content and metadata...")
        
        content_weight = 0.35
        metadata_weight = 0.65

        # Convert to numpy arrays and ensure proper shape
        content_embeddings = np.array(content_embeddings)
        metadata_embeddings = np.array(metadata_embeddings)
        
        # Handle different embedding shapes
        if len(content_embeddings.shape) == 3:
            content_embeddings = content_embeddings.squeeze(1)
        if len(metadata_embeddings.shape) == 3:
            metadata_embeddings = metadata_embeddings.squeeze(1)

        # Ensure both embeddings have the same dimensionality
        if content_embeddings.shape[1] != metadata_embeddings.shape[1]:
            logger.error(f"Content embeddings shape: {content_embeddings.shape}, Metadata embeddings shape: {metadata_embeddings.shape}")
            raise ValueError("Content and metadata embeddings must have the same dimensionality.")

        # Combine embeddings with weights
        combined_embeddings = (content_embeddings * content_weight + 
                              metadata_embeddings * metadata_weight)

        # Normalize combined embeddings
        norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        combined_embeddings = combined_embeddings / norms
        
        # Determine optimal number of clusters
        n_samples = len(combined_embeddings)
        optimal_clusters = min(num_coarse_clusters, max(2, n_samples // 10))
        
        logger.info(f"Performing K-means clustering with {optimal_clusters} clusters on {n_samples} samples")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10, max_iter=500)
        cluster_labels = kmeans.fit_predict(combined_embeddings)
        
        # Create coarse cluster structure
        coarse_cluster_data = defaultdict(lambda: {'indices': [], 'texts': [], 'metadata': []})
        for i, label in enumerate(cluster_labels):
            cluster_id = f"coarse_{label}"
            coarse_cluster_data[cluster_id]['indices'].append(i)
            coarse_cluster_data[cluster_id]['texts'].append(chunk_list[i])
            coarse_cluster_data[cluster_id]['metadata'].append(metadata_list[i])
        
        # Store coarse cluster centroids and create index
        coarse_centroids = kmeans.cluster_centers_
        self.coarse_cluster_embeddings = coarse_centroids
        self.coarse_clusters = dict(coarse_cluster_data)
        
        # Create FAISS index for coarse clusters
        self.coarse_cluster_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
        self.coarse_cluster_index.add(coarse_centroids.astype(np.float32))
        
        logger.info(f"Created {len(coarse_cluster_data)} coarse clusters")
        return coarse_cluster_data
  
    def build_hierarchical_index(self, content_embeddings: List[np.ndarray], 
                                metadata_embeddings: List[np.ndarray], 
                                chunk_list: List[str], 
                                metadata_list: List[Dict[str, str]]):
        """Build hierarchical FAISS index with coarse and fine-grained clustering"""
        logger.info("Building hierarchical FAISS index...")

        # Step 1: Build coarse clusters based on combined similarity
        coarse_cluster_data = self._build_coarse_clusters(
            content_embeddings, metadata_embeddings, chunk_list, metadata_list
        )
        
        # Step 2: Build fine-grained FAISS indices within each coarse cluster
        logger.info("Building fine-grained indices within coarse clusters...")
        
        for coarse_cluster_id, cluster_info in coarse_cluster_data.items():
            indices = cluster_info['indices']
            texts = cluster_info['texts']
            metadata = cluster_info['metadata']
            
            if len(indices) < 1:
                logger.warning(f"Skipping empty cluster {coarse_cluster_id}")
                continue
                
            # Extract embeddings for this cluster
            cluster_content_embeddings = np.array([content_embeddings[i] for i in indices])
            
            # Handle embedding shape
            if len(cluster_content_embeddings.shape) == 3:
                cluster_content_embeddings = cluster_content_embeddings.squeeze(1)
            
            # Create appropriate FAISS index based on cluster size
            if len(indices) > 100:
                # Use IVF for larger clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = min(max(len(indices) // 20, 4), 50)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
                # Train the index if we have enough data
                if len(cluster_content_embeddings) >= nlist:
                    index.train(cluster_content_embeddings.astype(np.float32))
                    index.add(cluster_content_embeddings.astype(np.float32))
                else:
                    # Fallback to flat index
                    index = faiss.IndexFlatIP(self.embedding_dim)
                    index.add(cluster_content_embeddings.astype(np.float32))
            else:
                # Use flat index for smaller clusters
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(cluster_content_embeddings.astype(np.float32))
            
            # Store fine cluster data
            self.fine_cluster_indices[coarse_cluster_id] = index
            self.fine_cluster_metadata[coarse_cluster_id] = metadata
            self.fine_cluster_embeddings[coarse_cluster_id] = cluster_content_embeddings
            self.fine_cluster_texts[coarse_cluster_id] = texts
            
            # Update global to local mapping
            for local_id, global_idx in enumerate(indices):
                self.global_to_local[self.next_global_id] = (coarse_cluster_id, local_id)
                self.next_global_id += 1
        
        logger.info(f"Built hierarchical index with {len(self.fine_cluster_indices)} fine clusters")
    
    def _find_similar_coarse_clusters(self, query_embedding: np.ndarray, 
                                     metadata_embedding: np.ndarray, 
                                     num_clusters: int = 3) -> List[str]:
        """Find most similar coarse clusters using combined query embedding"""
        if self.coarse_cluster_index is None or len(self.coarse_clusters) == 0:
            logger.warning("No coarse clusters available, returning empty list")
            return []

        content_weight = 0.35
        metadata_weight = 0.65
        
        # Handle embedding shapes
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding.squeeze(0)
        if len(metadata_embedding.shape) == 2:
            metadata_embedding = metadata_embedding.squeeze(0)
        
        # Combine embeddings with same weights used during clustering
        combined_query = (query_embedding * content_weight + 
                         metadata_embedding * metadata_weight)
        
        # Normalize combined query
        norm = np.linalg.norm(combined_query)
        if norm > 0:
            combined_query = combined_query / norm
        
        # Limit search to available clusters
        search_k = min(num_clusters, len(self.coarse_clusters))
        
        # Search coarse cluster index
        try:
            scores, indices = self.coarse_cluster_index.search(
                combined_query.reshape(1, -1).astype(np.float32), 
                search_k
            )
            
            # Return cluster IDs for the most similar clusters
            cluster_ids = list(self.coarse_clusters.keys())
            selected_clusters = []
            
            logger.debug(f"Coarse cluster search scores: {scores[0]}")
            logger.debug(f"Coarse cluster search indices: {indices[0]}")
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(cluster_ids):
                    selected_clusters.append(cluster_ids[idx])
                    logger.debug(f"Selected cluster {cluster_ids[idx]} with score {score}")
            
            if not selected_clusters:
                logger.warning("No valid clusters found, using first available clusters")
                selected_clusters = cluster_ids[:search_k]
            
            return selected_clusters
            
        except Exception as e:
            logger.error(f"Error in coarse cluster search: {e}")
            # Fallback to first few clusters
            return list(self.coarse_clusters.keys())[:search_k]
    
    def _search_cluster_parallel(self, cluster_id: str, query_embedding: np.ndarray, k: int) -> List[Tuple[float, str, int, Dict, str]]:
        """Search within a specific cluster in parallel"""
        if cluster_id not in self.fine_cluster_indices:
            return []
        
        index = self.fine_cluster_indices[cluster_id]
        metadata_list = self.fine_cluster_metadata[cluster_id]
        texts_list = self.fine_cluster_texts[cluster_id]
        
        # Set nprobe for IVF indices
        if hasattr(index, 'nprobe'):
            index.nprobe = min(getattr(config, 'NPROBE', 10), getattr(index, 'nlist', 10))
        
        try:
            # Handle query embedding shape
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding.squeeze(0)
            
            # Search within this cluster using content embedding
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                min(k, len(metadata_list))
            )
            
            # Collect results with cluster info
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
            logger.error(f"Error searching cluster {cluster_id}: {e}")
            return []
    
    def search(self, query_text: str, query_embedding: np.ndarray, 
              query_metadata: Dict, metadata_embedding: np.ndarray, 
              k: int = 5, num_coarse_clusters: int = 3) -> List[Tuple[float, int, Dict, str]]:
        """
        Hierarchical search: first find similar coarse clusters, then search within them
        """
        # Step 1: Find most similar coarse clusters using combined similarity
        selected_clusters = self._find_similar_coarse_clusters(
            query_embedding, metadata_embedding, num_coarse_clusters
        )

        logger.debug(f"Selected {len(selected_clusters)} coarse clusters: {selected_clusters}")
        
        # Step 2: Search within selected clusters only (hierarchical approach)
        all_results = []
        
        # Calculate how many results to get from each cluster
        results_per_cluster = max(1, k // len(selected_clusters)) if selected_clusters else k
        
        with ThreadPoolExecutor(max_workers=min(len(selected_clusters), 4)) as executor:
            # Submit parallel search tasks - only for selected clusters
            future_to_cluster = {
                executor.submit(self._search_cluster_parallel, cluster_id, query_embedding, results_per_cluster + 2): cluster_id
                for cluster_id in selected_clusters
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    cluster_results = future.result(timeout=10)
                    all_results.extend(cluster_results)
                    logger.debug(f"Found {len(cluster_results)} results in cluster {cluster_id}")
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
        
        # Sort by similarity score (descending) and return top k
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Total results from {len(selected_clusters)} clusters: {len(all_results)}, returning top {k}")
        
        # Format results for compatibility
        final_results = []
        for score, cluster_id, local_idx, metadata, text in all_results[:k]:
            final_results.append((score, local_idx, metadata, text))
        
        return final_results
    
    def save_index(self, path: str):
        """Save the hierarchical index to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save coarse cluster index
        if self.coarse_cluster_index is not None:
            faiss.write_index(self.coarse_cluster_index, os.path.join(path, "coarse_index.faiss"))
        
        # Save fine cluster FAISS indices
        for cluster_id, index in self.fine_cluster_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"fine_{safe_name}.index"))
        
        # Save metadata and mappings
        metadata_file = os.path.join(path, "hierarchical_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'coarse_clusters': self.coarse_clusters,
                'fine_cluster_metadata': self.fine_cluster_metadata,
                'fine_cluster_texts': self.fine_cluster_texts,
                'global_to_local': self.global_to_local,
                'next_global_id': self.next_global_id,
                'embedding_dim': self.embedding_dim
            }, f)
        
        logger.info(f"Saved hierarchical index to {path}")
    
    def load_index(self, path: str):
        """Load the hierarchical index from disk"""
        # Load coarse cluster index
        coarse_index_path = os.path.join(path, "coarse_index.faiss")
        if os.path.exists(coarse_index_path):
            self.coarse_cluster_index = faiss.read_index(coarse_index_path)
        
        # Load metadata and mappings
        metadata_file = os.path.join(path, "hierarchical_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.coarse_clusters = data['coarse_clusters']
                self.fine_cluster_metadata = data['fine_cluster_metadata']
                self.fine_cluster_texts = data['fine_cluster_texts']
                self.global_to_local = data['global_to_local']
                self.next_global_id = data['next_global_id']
                self.embedding_dim = data['embedding_dim']
        
        # Load fine cluster indices
        for cluster_id in self.coarse_clusters.keys():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"fine_{safe_name}.index")
            if os.path.exists(index_path):
                self.fine_cluster_indices[cluster_id] = faiss.read_index(index_path)
        
        logger.info(f"Loaded hierarchical index from {path}")


# Main processing script corrections
import logging
import os
import json
from typing import Dict
from embedding import compute_embeddings, normalize_embeddings
from data_process import load_documents, load_queries
from llm_fun import get_chunk_from_llm, get_metadata_from_llm, get_llm_response
from config import config
from faiss_index import HierarchicalFAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


def metadata_embedding_text(metadata: Dict[str, str]) -> str:
    """Convert metadata dictionary to text for embedding"""
    text_parts = []
    
    if metadata.get('gender'):
        text_parts.append(f"Gender: {metadata['gender']}")
    if metadata.get('age_group'):
        text_parts.append(f"Age_group: {metadata['age_group']}")
    if metadata.get('age_range'):
        text_parts.append(f"Age_range: {metadata['age_range']}")  
    if metadata.get('keywords'):
        text_parts.append(f"Keywords: {metadata['keywords']}")
    if metadata.get('summary'):
        summary = metadata['summary'][:200] if len(metadata['summary']) > 200 else metadata['summary']
        text_parts.append(f"Summary: {summary}")
    if metadata.get('location'):
        text_parts.append(f"Location: {metadata['location']}")   
    return " | ".join(text_parts) if text_parts else "general content"


def main():
    faiss_index = HierarchicalFAISSIndex(config.EMBEDDING_DIM)

    logger.info("Loading and preprocessing documents...")
    documents = load_documents(config.DATA_FOLDER)

    if not documents:
        logger.error("No documents loaded. Please check the data folder.")
        return
    
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

    # Build hierarchical FAISS index
    logger.info("Building hierarchical FAISS index...")
    faiss_index.build_hierarchical_index(chunk_embeddings, metadata_embeddings, all_chunks, all_metadata)
    
    # Save the index
    if not os.path.exists(config.INDEX_PATH):
        os.makedirs(config.INDEX_PATH)
    faiss_index.save_index(config.INDEX_PATH)

    # Process queries
    queries = load_queries(config.QUERY_FILE)

    if queries:
        logger.info("Processing queries...")

        for i, query in enumerate(queries):
            logger.info(f"\nQuery {i+1}: {query}")
            
            # Compute query embeddings
            query_embedding = compute_embeddings(query)
            query_embedding = normalize_embeddings(query_embedding)

            # Get query metadata
            query_metadata = get_metadata_from_llm(query)
            
            # Compute metadata embedding
            metadata_embedding = compute_embeddings(metadata_embedding_text(query_metadata))
            metadata_embedding = normalize_embeddings(metadata_embedding)

            # Search the hierarchical index
            results = faiss_index.search(query, query_embedding, query_metadata, metadata_embedding)
            
            # Extract context from results
            context_parts = []
            for score, local_idx, metadata, text in results:
                context_parts.append(f"[Score: {score:.3f}] {text}")
            
            context = "\n\n".join(context_parts)
            full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

            # Get LLM response
            answer = get_llm_response(full_prompt)
            print(f"Question: {query}\n\nContext: {context}\n\nAnswer: {answer}\n\n")
    
    logger.info("\nRAG with hierarchical metadata clustering completed\n")    


if __name__ == '__main__':
    main()