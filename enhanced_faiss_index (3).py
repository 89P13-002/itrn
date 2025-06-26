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
from datetime import datetime
import json

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
        self.chunk_to_clusters = {}           # chunk_id -> list of metadata_cluster_ids
        
        # Enhanced mappings for better tracking
        self.chunk_id_to_metadata_cluster = {}  # chunk_id -> primary metadata cluster
        self.chunk_id_to_all_clusters = {}      # chunk_id -> all assigned clusters with scores
        self.metadata_cluster_stats = {}       # cluster_id -> statistics
        self.global_chunk_registry = {}        # chunk_id -> complete chunk info
        
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
        
        # Initialize cluster statistics
        for cluster_id in self.metadata_clusters.keys():
            self.metadata_cluster_stats[cluster_id] = {
                'chunk_count': 0,
                'primary_assignments': 0,
                'secondary_assignments': 0,
                'created_at': datetime.now().isoformat()
            }
        
        logger.info(f"Created {len(metadata_cluster_data)} metadata clusters")
        return metadata_cluster_data
    
    def _assign_chunks_to_clusters(self, chunk_list: List[str], 
                                  content_embeddings: List[np.ndarray],
                                  metadata_embeddings: List[np.ndarray],
                                  metadata_list: List[Dict[str, str]],
                                  chunk_ids: List[str],
                                  top_k_clusters: int = 3,
                                  output_dir: str = "cluster_outputs"):
        logger.info(f"Assigning chunks to top-{top_k_clusters} metadata clusters")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
            assigned_clusters_with_scores = []
            cluster_ids = list(self.metadata_clusters.keys())
            
            for rank, (score, cluster_idx) in enumerate(zip(scores[0], indices[0])):
                if cluster_idx != -1 and cluster_idx < len(cluster_ids):
                    cluster_id = cluster_ids[cluster_idx]
                    assigned_clusters.append(cluster_id)
                    assigned_clusters_with_scores.append({
                        'cluster_id': cluster_id,
                        'score': float(score),
                        'rank': rank + 1,
                        'is_primary': rank == 0
                    })
                    
                    # Adding chunk to this cluster
                    cluster_data[cluster_id]['content_embeddings'].append(content_emb)
                    cluster_data[cluster_id]['texts'].append(chunk_text)
                    cluster_data[cluster_id]['metadata'].append(metadata)
                    cluster_data[cluster_id]['chunk_ids'].append(chunk_id)
                    cluster_data[cluster_id]['chunk_indices'].append(chunk_idx)
                    
                    # Update statistics
                    if rank == 0:
                        self.metadata_cluster_stats[cluster_id]['primary_assignments'] += 1
                    else:
                        self.metadata_cluster_stats[cluster_id]['secondary_assignments'] += 1
                    
                    self.metadata_cluster_stats[cluster_id]['chunk_count'] += 1
            
            self.cluster_assignments[chunk_idx] = assigned_clusters
            
            # Enhanced chunk tracking
            if assigned_clusters:
                primary_cluster = assigned_clusters[0]  # Highest scoring cluster
                self.chunk_id_to_metadata_cluster[chunk_id] = primary_cluster
                self.chunk_id_to_all_clusters[chunk_id] = assigned_clusters_with_scores
                
                # Register chunk globally
                self.global_chunk_registry[chunk_id] = {
                    'text': chunk_text,
                    'metadata': metadata,
                    'primary_cluster': primary_cluster,
                    'all_clusters': assigned_clusters,
                    'cluster_scores': assigned_clusters_with_scores,
                    'chunk_index': chunk_idx
                }
            
            # Also track chunk_id to cluster mapping (backward compatibility)
            if chunk_id not in self.chunk_to_clusters:
                self.chunk_to_clusters[chunk_id] = []
            self.chunk_to_clusters[chunk_id].extend(assigned_clusters)
        
        # Write enhanced chunk information to files
        self._write_cluster_files(cluster_data, output_dir)
        self._write_chunk_mappings(output_dir)
        self._write_cluster_statistics(output_dir)
        
        logger.info(f"Finished assigning chunks to clusters with enhanced tracking")
        return cluster_data
    
    def _write_cluster_files(self, cluster_data, output_dir):
        """Write chunk texts to separate files for each metadata cluster"""
        logger.info(f"Writing chunk texts to files in {output_dir}")
        logger.info(f"Total clusters to write: {len(cluster_data)}")
        
        for cluster_id, data in cluster_data.items():
            logger.info(f"Processing cluster {cluster_id} with {len(data['texts'])} texts")
            
            if data['texts']:  # Only write if there are texts in the cluster
                # Create safe filename from cluster_id
                safe_cluster_name = cluster_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                output_file = os.path.join(output_dir, f"{safe_cluster_name}_chunks.txt")
                
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== Metadata Cluster: {cluster_id} ===\n")
                        f.write(f"Total chunks: {len(data['texts'])}\n")
                        f.write(f"Created at: {datetime.now().isoformat()}\n")
                        f.write("=" * 50 + "\n\n")
                        
                        # Write all chunks in this cluster
                        for i in range(len(data['texts'])):
                            chunk_text = data['texts'][i]
                            chunk_id = data['chunk_ids'][i]
                            metadata = data['metadata'][i]
                            
                            f.write(f"--- Chunk {i+1} ---\n")
                            f.write(f"Chunk ID: {chunk_id}\n")
                            f.write(f"Primary Metadata Cluster: {self.chunk_id_to_metadata_cluster.get(chunk_id, 'Unknown')}\n")
                            f.write(f"Current Cluster: {cluster_id}\n")
                            f.write(f"All Assigned Clusters: {self.chunk_to_clusters.get(chunk_id, [])}\n")
                            f.write(f"Metadata: {str(metadata)}\n")
                            f.write(f"Text:\n{chunk_text}\n")
                            f.write("-" * 30 + "\n\n")
                            
                            # Force flush to ensure data is written
                            f.flush()
                    
                    # Verify file was written
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        logger.info(f"Successfully written {len(data['texts'])} chunks to {output_file} (size: {file_size} bytes)")
                    else:
                        logger.error(f"File {output_file} was not created!")
                    
                except Exception as e:
                    logger.error(f"Error writing to file {output_file}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"Cluster {cluster_id} has no texts to write")
    
    def _write_chunk_mappings(self, output_dir):
        """Write detailed chunk-to-cluster mappings"""
        # Basic chunk-to-cluster mapping file
        chunk_mapping_file = os.path.join(output_dir, "chunk_to_clusters_mapping.txt")
        try:
            with open(chunk_mapping_file, 'w', encoding='utf-8') as mapping_f:
                mapping_f.write("=== Chunk ID to Metadata Clusters Mapping ===\n")
                mapping_f.write(f"Total chunks: {len(self.chunk_to_clusters)}\n")
                mapping_f.write(f"Generated at: {datetime.now().isoformat()}\n")
                mapping_f.write("=" * 60 + "\n\n")
                
                for chunk_id, cluster_list in self.chunk_to_clusters.items():
                    mapping_f.write(f"Chunk ID: {chunk_id}\n")
                    mapping_f.write(f"Primary cluster: {self.chunk_id_to_metadata_cluster.get(chunk_id, 'Unknown')}\n")
                    mapping_f.write(f"All assigned clusters: {cluster_list}\n")
                    mapping_f.write("-" * 40 + "\n")
                
            logger.info(f"Created chunk-to-cluster mapping file: {chunk_mapping_file}")
        except Exception as e:
            logger.error(f"Error creating chunk mapping file: {e}")
        
        # Detailed chunk registry with scores
        detailed_mapping_file = os.path.join(output_dir, "detailed_chunk_mappings.json")
        try:
            with open(detailed_mapping_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunk_id_to_metadata_cluster': self.chunk_id_to_metadata_cluster,
                    'chunk_id_to_all_clusters': self.chunk_id_to_all_clusters,
                    'global_chunk_registry': {k: {**v, 'text': v['text'][:200] + '...' if len(v['text']) > 200 else v['text']} 
                                            for k, v in self.global_chunk_registry.items()},
                    'generation_timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created detailed chunk mappings JSON: {detailed_mapping_file}")
        except Exception as e:
            logger.error(f"Error creating detailed mappings file: {e}")
    
    def _write_cluster_statistics(self, output_dir):
        """Write cluster statistics to file"""
        stats_file = os.path.join(output_dir, "cluster_statistics.json")
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata_cluster_stats': self.metadata_cluster_stats,
                    'total_chunks': len(self.global_chunk_registry),
                    'total_clusters': len(self.metadata_clusters),
                    'generation_timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Created cluster statistics file: {stats_file}")
        except Exception as e:
            logger.error(f"Error creating statistics file: {e}")

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
                                chunk_ids: List[str],
                                output_dir: str = "cluster_outputs"):
        logger.info("Building scalable hierarchical FAISS index...")
        
        # Step 1: Building coarse metadata clusters
        self._build_metadata_clusters(metadata_embeddings, metadata_list)
        
        # Step 2: Assign chunks to top-k most similar metadata clusters
        cluster_data = self._assign_chunks_to_clusters(
            chunk_list, content_embeddings, metadata_embeddings, metadata_list, chunk_ids,
            output_dir=output_dir
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
                                       num_clusters: int = 5) -> List[Tuple[str, float]]:
        """Find relevant metadata clusters and return them with scores"""
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
            
            # get relevant metadata cluster with scores
            cluster_ids = list(self.metadata_clusters.keys())
            relevant_clusters = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(cluster_ids):
                    cluster_id = cluster_ids[idx]
                    # Only include clusters that have content indices
                    if cluster_id in self.content_indices:
                        relevant_clusters.append((cluster_id, float(score)))
                        logger.info(f"Selected metadata cluster {cluster_id} with score {score:.3f}")
            
            return relevant_clusters
            
        except Exception as e:
            logger.error(f"Error in metadata cluster search: {e}")
            return [(cid, 0.0) for cid in list(self.metadata_clusters.keys())[:search_k] 
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
              k: int = 5, num_metadata_clusters: int = 5, ef_search: Optional[int] = None,
              debug_output_dir: str = None) -> List[Tuple[float, int, Dict, str, str]]:
        """
        Enhanced Hierarchical FAISS:
        1. First find relevant metadata clusters using query metadata
        2. Then search content within selected metadata clusters using query content embedding
        3. Provide detailed routing information for debugging
        Returns: List of (score, local_idx, metadata, text, chunk_id)
        """
        logger.info(f"=== ENHANCED SEARCH DEBUG INFO ===")
        logger.info(f"Query: {query_text[:100]}...")
        logger.info(f"Query metadata: {query_metadata}")
        logger.info(f"Searching for top {k} results from {num_metadata_clusters} metadata clusters")
        
        # Step 1: Finding relevant metadata clusters with scores
        relevant_clusters_with_scores = self._find_relevant_metadata_clusters(
            metadata_embedding, num_metadata_clusters
        )
        
        if not relevant_clusters_with_scores:
            logger.warning("No relevant metadata clusters found")
            return []
        
        relevant_clusters = [cluster_id for cluster_id, _ in relevant_clusters_with_scores]
        logger.info(f"✓ Selected metadata clusters with scores:")
        for cluster_id, score in relevant_clusters_with_scores:
            logger.info(f"  - {cluster_id}: score={score:.4f}")
        
        # Enhanced cluster routing verification
        logger.info(f"\n=== CLUSTER ROUTING VERIFICATION ===")
        self._verify_cluster_routing(query_text, query_metadata, relevant_clusters_with_scores)
        
        # Log which chunks are in the selected clusters for debugging
        total_chunks_in_selected_clusters = 0
        chunks_by_cluster = {}
        for cluster_id in relevant_clusters:
            if cluster_id in self.cluster_chunk_ids:
                chunk_count = len(self.cluster_chunk_ids[cluster_id])
                chunks_by_cluster[cluster_id] = self.cluster_chunk_ids[cluster_id]
                total_chunks_in_selected_clusters += chunk_count
                logger.info(f"  - {cluster_id}: {chunk_count} chunks")
                
                # Show sample chunk IDs in this cluster
                sample_chunks = self.cluster_chunk_ids[cluster_id][:3]
                logger.info(f"    Sample chunks: {sample_chunks}")
        
        logger.info(f"Total chunks available for search: {total_chunks_in_selected_clusters}")
        
        # Step 2: Now search for the content within relevant clusters
        all_results = []
        results_per_cluster = max(1, (k * 2) // len(relevant_clusters))
        
        logger.info(f"\n=== CONTENT SEARCH WITHIN CLUSTERS ===")
        logger.info(f"Searching {results_per_cluster} results per cluster...")
        
        cluster_search_results = {}
        
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
                    cluster_search_results[cluster_id] = cluster_results
                    logger.info(f"✓ Found {len(cluster_results)} results in cluster {cluster_id}")
                    
                    # Log the top result from each cluster for debugging
                    if cluster_results:
                        top_result = cluster_results[0]
                        score, _, _, _, text, chunk_id = top_result
                        logger.info(f"  Top result: chunk_id={chunk_id}, score={score:.3f}")
                        logger.info(f"  Preview: '{text[:80]}...'")
                        
                        # Verify routing correctness
                        if chunk_id in self.chunk_id_to_metadata_cluster:
                            primary_cluster = self.chunk_id_to_metadata_cluster[chunk_id]
                            all_clusters = self.chunk_to_clusters.get(chunk_id, [])
                            logger.info(f"  Routing check - Primary cluster: {primary_cluster}, Found in: {cluster_id}")
                            logger.info(f"  All assigned clusters: {all_clusters}")
                            
                            if cluster_id not in all_clusters:
                                logger.warning(f"  ⚠️  ROUTING ISSUE: Chunk {chunk_id} found in {cluster_id} but not assigned to it!")
                        
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
                    cluster_search_results[cluster_id] = []
        
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"\n=== SEARCH RESULTS SUMMARY ===")
        logger.info(f"Total results from {len(relevant_clusters)} clusters: {len(all_results)}")
        logger.info(f"Returning top {k} results")
        
        final_results = []
        for i, (score, cluster_id, local_idx, metadata, text, chunk_id) in enumerate(all_results[:k]):
            final_results.append((score, local_idx, metadata, text, chunk_id))
            logger.info(f"\nResult {i+1}:")
            logger.info(f"  Chunk ID: {chunk_id}")
            logger.info(f"  Found in cluster: {cluster_id}")
            logger.info(f"  Content score: {score:.4f}")
            logger.info(f"  Text preview: '{text[:100]}...'")
            
            # Enhanced routing verification for final results
            if chunk_id in self.chunk_id_to_all_clusters:
                cluster_info = self.chunk_id_to_all_clusters[chunk_id]
                logger.info(f"  Cluster assignments:")
                for cluster_assignment in cluster_info:
                    is_found_cluster = cluster_assignment['cluster_id'] == cluster_id
                    marker = "✓ FOUND HERE" if is_found_cluster else ""
                    logger.info(f"    - {cluster_assignment['cluster_id']}: "
                              f"rank={cluster_assignment['rank']}, "
                              f"metadata_score={cluster_assignment['score']:.4f}, "
                              f"primary={cluster_assignment['is_primary']} {marker}")
        
        # Optionally save search debug info to file
        if debug_output_dir:
            self._save_enhanced_search_debug_info(
                query_text, query_metadata, relevant_clusters_with_scores, 
                cluster_search_results, all_results[:k], debug_output_dir
            )
        
        logger.info(f"\n=== END ENHANCED SEARCH DEBUG ===")
        return final_results
    
    def _verify_cluster_routing(self, query_text: str, query_metadata: Dict, 
                               relevant_clusters_with_scores: List[Tuple[str, float]]):
        """Verify if the query is being routed to the correct metadata clusters"""
        logger.info("Verifying cluster routing correctness...")
        
        for cluster_id, score in relevant_clusters_with_scores:
            if cluster_id in self.metadata_clusters:
                # Get sample metadata from this cluster
                cluster_info = self.metadata_clusters[cluster_id]
                sample_metadata = cluster_info['metadata'][:3] if cluster_info['metadata'] else []
                
                logger.info(f"Cluster {cluster_id} (score: {score:.4f}):")
                logger.info(f"  Sample metadata from this cluster:")
                for i, meta in enumerate(sample_metadata):
                    logger.info(f"    {i+1}. {meta}")
                
                # Check if query metadata is similar to cluster metadata
                self._compare_metadata_similarity(query_metadata, sample_metadata,