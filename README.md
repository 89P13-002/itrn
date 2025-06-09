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
        
        self.global_to_local = {}           # global_id -> (coarse_id, fine_id)
        self.next_global_id = 0
        self.lock = threading.Lock()
        
        # Store metadata embeddings for similarity search
        self.chunk_metadata_embeddings = None  # All chunk metadata embeddings
        self.chunk_metadata_texts = []         # All chunk metadata texts
    
    
    
    def _build_coarse_clusters(self, content_embeddings: List[np.ndarray], metadata_embeddings: List[np.ndarray], chunk_list: List[str], metadata_list: List[Dict[str, str]], 
                            num_coarse_clusters: int = 100):
        logger.info("Building coarse clusters from combined content and metadata...")
        
        content_weight = 0.35
        metadata_weight = 0.65

        # Convert lists of embeddings to NumPy arrays
        content_embeddings = np.array(content_embeddings)
        metadata_embeddings = np.array(metadata_embeddings)

        # Ensure both embeddings have the same shape
        if content_embeddings.shape[1] != metadata_embeddings.shape[1]:
            logger.error("Content and metadata embeddings must have the same dimensionality.")
            raise ValueError("Content and metadata embeddings must have the same dimensionality.")

        # Combine embeddings with weights
        combined_embeddings = (content_embeddings * content_weight + 
                            metadata_embeddings * metadata_weight)

        # Normalize combined embeddings
        combined_embeddings = combined_embeddings / np.linalg.norm(
            combined_embeddings, axis=1, keepdims=True
        )
        
        # print(combined_embeddings.shape)
        cmd = []
        for it in combined_embeddings:
            cmd.append(it[0])

        # print(cmd.shape)
        logger.info(f"Performing K-means clustering with {num_coarse_clusters} clusters")
        kmeans = KMeans(n_clusters=min(num_coarse_clusters,len(cmd)), random_state=42, n_init=10, max_iter=500)
        cluster_labels = kmeans.fit_predict(cmd)
        
        # Create coarse cluster structure
        coarse_cluster_data = defaultdict(lambda: {'indices': [], 'texts': []})
        for i, label in enumerate(cluster_labels):
            coarse_cluster_data[f"coarse_{label}"]['indices'].append(i)
            coarse_cluster_data[f"coarse_{label}"]['texts'].append(chunk_list[i])
        
        # Store coarse cluster centroids for later search
        coarse_centroids = kmeans.cluster_centers_
        self.coarse_cluster_embeddings = coarse_centroids
        
        # Create FAISS index for coarse clusters using the combined embedding dimension
        self.coarse_cluster_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
        self.coarse_cluster_index.add(coarse_centroids.astype(np.float32))
        
        logger.info(f"Created {len(coarse_cluster_data)} coarse clusters")
        return coarse_cluster_data
  
    def build_hierarchical_index(self, content_embeddings: List[np.ndarray], metadata_embeddings: List[np.ndarray], chunk_list:List[str], metadata_list: List[Dict[str, str]]):
        logger.info("Building hierarchical FAISS indexing")

        # Step 1: Build coarse clusters based on combined similarity
        coarse_cluster_data = self._build_coarse_clusters(content_embeddings, metadata_embeddings, chunk_list, metadata_list)
        
        # Step 2: Build fine-grained FAISS indices within each coarse cluster
        logger.info("Building fine-grained indices within coarse clusters...")
        
        for coarse_cluster_id, cluster_info in coarse_cluster_data.items():
            indices = cluster_info['indices']
            texts = cluster_info['texts']
            
            if len(indices) < 2:
                logger.warning(f"Skipping cluster {coarse_cluster_id} with only {len(indices)} items")
                continue
                
            # Use content embeddings for fine-grained search (more precise for actual retrieval)
            cluster_content_embeddings = content_embeddings[indices]
            cluster_metadata = [metadata_list[i] for i in indices] 
            
            # Create appropriate FAISS index based on cluster size
            if len(indices) > 100:
                # Use IVF for larger clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = min(max(len(indices) // 20, 4), 50)  # Adaptive nlist
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
                # Train the index
                if len(cluster_content_embeddings) >= nlist:
                    index.train(cluster_content_embeddings.astype(np.float32))
                else:
                    # Not enough data to train IVF, use flat index
                    index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                # Use flat index for smaller clusters
                index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Add embeddings to index
            index.add(cluster_content_embeddings.astype(np.float32))
            
            # Store fine cluster data
            self.fine_cluster_indices[coarse_cluster_id] = index
            self.fine_cluster_metadata[coarse_cluster_id] = cluster_metadata
            self.fine_cluster_embeddings[coarse_cluster_id] = cluster_content_embeddings
            
            # Update global to local mapping
            for local_id, global_idx in enumerate(indices):
                self.global_to_local[self.next_global_id] = (coarse_cluster_id, local_id)
                self.next_global_id += 1
        
        logger.info(f"Built hierarchical index with {len(self.fine_cluster_indices)} fine clusters")
    
    def _find_similar_coarse_clusters(self, query_embedding: np.ndarray, query_metadata: Dict[str, str], metadata_embedding: np.ndarray, num_clusters: int = 3) -> List[str]:
        if self.coarse_cluster_index is None:
            # Fallback: return all clusters
            return list(self.coarse_clusters.keys())[:num_clusters]
        

        content_weight = 0.35
        metadata_weight = 0.65
        
        if query_embedding.shape[1] != metadata_embedding.shape[1]:
            # Different dimensions - concatenate
            combined_query = np.concatenate([
                query_embedding * content_weight,
                metadata_embedding * metadata_weight
            ], axis=1)
        else:
            # Same dimensions - weighted average
            combined_query = (query_embedding * content_weight + 
                            metadata_embedding * metadata_weight)
        
        # Normalize combined query
        combined_query = combined_query / np.linalg.norm(combined_query, axis=1, keepdims=True)
        
        # Search coarse cluster index
        try:
            scores, indices = self.coarse_cluster_index.search(
                combined_query.astype(np.float32), 
                min(num_clusters, len(self.coarse_clusters))
            )
            
            # Return cluster IDs
            selected_clusters = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.coarse_clusters):
                    clusetr_ids = list(self.coarse_clusters.keys())
                    if idx < len(cluster_ids):
                        selected_clusters.append(cluster_ids[idx])
            
            return selected_clusters if selected_clusters else list(self.coarse_clusters.keys())[:num_clusters]
            
        except Exception as e:
            logger.error(f"Error in coarse cluster search: {e}")
            return list(self.coarse_clusters.keys())[:num_clusters]
    
    def _search_cluster_parallel(self, cluster_id: str, query_embedding: np.ndarray, k: int) -> List[Tuple[float, str, int, Dict]]:
        if cluster_id not in self.fine_cluster_indices:
            return []
        
        index = self.fine_cluster_indices[cluster_id]
        metadata_list = self.fine_cluster_metadata[cluster_id]
        
        # Set nprobe for IVF indices
        if hasattr(index, 'nprobe'):
            index.nprobe = min(config.NPROBE, getattr(index, 'nlist', 10))
        
        try:
            # Search within this cluster using content embedding
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                min(k, len(metadata_list))
            )
            
            # Collect results with cluster info
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(metadata_list):
                    results.append((float(score), cluster_id, idx, metadata_list[idx]))
            
            return results
        except Exception as e:
            logger.error(f"Error searching cluster {cluster_id}: {e}")
            return []
    
    def search(self, query_text:str, query_embedding: np.ndarray, query_metadata: List[Dict], metadata_embedding: np.ndarray, k: int = 5, 
              num_coarse_clusters: int = 3) -> List[Tuple[float, int, Dict]]:
        
        # Step 1: Find most similar coarse clusters using combined similarity
        selected_clusters = self._find_similar_coarse_clusters(
            query_embedding, query_metadata, metadata_embedding, num_coarse_clusters
        )

        logger.debug(f"Selected coarse clusters: {selected_clusters}")
        
        # Step 2: Search selected clusters in parallel using content embeddings
        all_results = []
        
        with ThreadPoolExecutor(max_workers=min(len(selected_clusters), 4)) as executor:
            # Submit parallel search tasks
            future_to_cluster = {
                executor.submit(self._search_cluster_parallel, cluster_id, query_embedding, k * 2): cluster_id
                for cluster_id in selected_clusters
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    cluster_results = future.result(timeout=10)
                    all_results.extend(cluster_results)
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
        
        # sort fun for returning only toppmost few most similar context
        
        
        final_results = []
        for context in all_results[:k]:
            final_results.append(context)
        
        return final_results
    

    def save_index(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        # Save coarse cluster index
        if self.coarse_cluster_index is not None:
            faiss.write_index(self.coarse_cluster_index, os.path.join(path, "coarse_index.faiss"))
        
        # Save fine cluster FAISS indices
        for cluster_id, index in self.fine_cluster_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"fine_{safe_name}.index"))
        
        logger.info(f"Saved hierarchical index to {path}")


import logging
import os
import json
from typing import Dict
from embedding import compute_embeddings, normalize_embeddings
from data_process import load_documents,load_queries
from llm_fun import get_chunk_from_llm, get_metadata_from_llm, get_llm_response
from config import config
from faiss_index import HierarchicalFAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


def metadata_embedding_text(metadata: Dict[str, str]) -> str:
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
        logger.error("No documents loaded. Please check the lecture folder.")
        return
    
    os.makedirs(config.LLM_CHUNK, exist_ok=True)
    all_chunks = []
    all_metadata = []
    for doc in documents:
        logger.info(f"Processing document: {doc['filename']}")
        
        chunks = get_chunk_from_llm(doc['content'])
        filename = os.path.splitext(doc['filename'])[0]
        out_file = os.path.join(config.LLM_CHUNK,f"{filename}.txt")

        with open(out_file,"w",encoding="utf-8") as f:
            for i,chunk in enumerate(chunks):
                metadata = get_metadata_from_llm(chunk)

                all_chunks.append(chunk)
                all_metadata.append(metadata)

                f.write(f"-- Chunk {i+1} -- \n")
                f.write(f"Text : \n {chunk}\n\n")
                f.write(f"Metadata \n{json.dumps(metadata,indent=2)}\n\n")
        
        logger.info(f"Created {len(chunks)} chunks from {doc['filename']}")
    
    logger.info(f"Total chunks created: {len(all_chunks)}")


    # Compute embeddings for chunks
    logger.info("Computing content embeddings for chunks...")
    chunk_embeddings = []
    for chk in all_chunks:
        chk_embedding = compute_embeddings(chk)
        chk_embedding = normalize_embeddings(chk_embedding)
        chunk_embeddings.append(chk_embedding)

    metadata_embeddings = []
    for mtd in all_metadata:
        mtd_embedding = compute_embeddings(metadata_embedding_text(mtd))
        mtd_embedding = normalize_embeddings(mtd_embedding)
        metadata_embeddings.append(mtd_embedding)

    # Build hierarchical FAISS index with similarity-based metadata clustering
    logger.info("Building hierarchical FAISS index with metadata similarity clustering...")
    faiss_index.build_hierarchical_index(chunk_embeddings, metadata_embeddings, all_chunks, all_metadata)
    
    # Save the index
    if not os.path.exists(config.INDEX_PATH):
        os.makedirs(config.INDEX_PATH)
    faiss_index.save_index(config.INDEX_PATH)


    queries = load_queries(config.QUERY_FILE)

    if queries:
        logger.info("Processing queries...")

        for query in queries:
            logger.info(f"\nQuery {i+1}: {query}")
            query_emb = compute_embeddings(query)
            query_emb = normalize_embeddings(query_emb)

            metadata = get_metadata_from_llm(query)

            
            metadata_emd = compute_embeddings(metadata_embedding_text(metadata))
            metadata_emd = normalize_embeddings(metadata_emd)

            results = faiss_index.search(query, query_emb,metadata, metadata_emd)
            
            context = "\n\n".join(results)
            full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

            answer = get_llm_response(full_prompt)
            print(f"Question : {query}\n\nContext : {context}\n\nAnswer: {answer}\n\n")
    
    logger.info("\nRAG with metadata done\n")    

if __name__ == '__main__':
    main()
