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
from sentence_transformers import SentenceTransformer
import threading
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalFAISSIndex:
    def __init__(self, embedding_dim: int = 384, metadata_embedding_model: SentenceTransformer = None):
        self.embedding_dim = embedding_dim
        self.metadata_embedding_model = metadata_embedding_model
        
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
        
    def _create_metadata_embedding_text(self, metadata: Dict[str, str]) -> str:
        """Create a comprehensive text representation of metadata for embedding"""
        text_parts = []
        
        # Prioritize key fields with proper formatting
        if metadata.get('topic'):
            text_parts.append(f"Topic: {metadata['topic']}")
        if metadata.get('category'):
            text_parts.append(f"Category: {metadata['category']}")
        if metadata.get('difficulty'):
            text_parts.append(f"Difficulty: {metadata['difficulty']}")
        if metadata.get('keywords'):
            text_parts.append(f"Keywords: {metadata['keywords']}")
        if metadata.get('summary'):
            # Truncate summary to avoid overwhelming the embedding
            summary = metadata['summary'][:200] if len(metadata['summary']) > 200 else metadata['summary']
            text_parts.append(f"Summary: {summary}")
        if metadata.get('source_file'):
            text_parts.append(f"Source: {metadata['source_file']}")
            
        return " | ".join(text_parts) if text_parts else "general content"
    
    def _extract_query_metadata(self, query: str) -> Dict[str, str]:
        """Extract implied metadata from query text using simple heuristics"""
        query_lower = query.lower()
        
        # Simple keyword-based metadata extraction
        metadata = {
            'topic': 'general',
            'category': 'query',
            'difficulty': 'intermediate',
            'keywords': '',
            'summary': query[:100] + ('...' if len(query) > 100 else '')
        }
        
        # Topic inference based on keywords
        topic_keywords = {
            'machine learning': ['ml', 'machine learning', 'algorithm', 'model', 'training', 'prediction'],
            'data science': ['data', 'analysis', 'statistics', 'dataset', 'visualization'],
            'programming': ['code', 'programming', 'function', 'variable', 'syntax', 'debug'],
            'mathematics': ['math', 'equation', 'formula', 'calculate', 'theorem', 'proof'],
            'database': ['database', 'sql', 'query', 'table', 'schema', 'relationship'],
            'networking': ['network', 'protocol', 'tcp', 'ip', 'router', 'switch'],
            'security': ['security', 'encryption', 'authentication', 'vulnerability', 'attack']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                metadata['topic'] = topic
                break
        
        # Difficulty inference
        if any(word in query_lower for word in ['basic', 'simple', 'introduction', 'beginner']):
            metadata['difficulty'] = 'beginner'
        elif any(word in query_lower for word in ['advanced', 'complex', 'detailed', 'expert']):
            metadata['difficulty'] = 'advanced'
        
        # Extract potential keywords (simple approach)
        import re
        words = re.findall(r'\b\w+\b', query_lower)
        # Filter out common words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        metadata['keywords'] = ', '.join(keywords[:5])  # Top 5 keywords
        
        return metadata
    
    def _compute_metadata_embeddings(self, metadata_list: List[Dict[str, str]]) -> np.ndarray:
        """Compute embeddings for all metadata"""
        if not self.metadata_embedding_model:
            logger.warning("No metadata embedding model provided")
            return None
            
        metadata_texts = [self._create_metadata_embedding_text(meta) for meta in metadata_list]
        self.chunk_metadata_texts = metadata_texts
        
        logger.info("Computing metadata embeddings...")
        metadata_embeddings = self.metadata_embedding_model.encode(
            metadata_texts, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Normalize metadata embeddings
        norms = np.linalg.norm(metadata_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        metadata_embeddings = metadata_embeddings / norms
        
        return metadata_embeddings
    
    def _build_coarse_clusters(self, content_embeddings: np.ndarray, metadata_list: List[Dict[str, str]], 
                              num_coarse_clusters: int = None):
        """Build coarse clusters based on combined content and metadata similarity"""
        logger.info("Building coarse clusters from combined content and metadata...")
        
        # Compute metadata embeddings
        metadata_embeddings = self._compute_metadata_embeddings(metadata_list)
        self.chunk_metadata_embeddings = metadata_embeddings
        
        if metadata_embeddings is None:
            return self._fallback_clustering(metadata_list)
        
        # Combine content and metadata embeddings with weights
        content_weight = 0.7  # Higher weight for content
        metadata_weight = 0.3  # Lower weight for metadata
        
        # Ensure both embedding types have the same number of samples
        assert content_embeddings.shape[0] == metadata_embeddings.shape[0], \
            "Content and metadata embeddings must have the same number of samples"
        
        # If embedding dimensions are different, we need to handle this
        if content_embeddings.shape[1] != metadata_embeddings.shape[1]:
            logger.info(f"Content dim: {content_embeddings.shape[1]}, Metadata dim: {metadata_embeddings.shape[1]}")
            # Simple concatenation approach
            combined_embeddings = np.concatenate([
                content_embeddings * content_weight,
                metadata_embeddings * metadata_weight
            ], axis=1)
        else:
            # Same dimensions - weighted average
            combined_embeddings = (content_embeddings * content_weight + 
                                 metadata_embeddings * metadata_weight)
        
        # Normalize combined embeddings
        combined_embeddings = combined_embeddings / np.linalg.norm(
            combined_embeddings, axis=1, keepdims=True
        )
        
        # Determine optimal number of coarse clusters
        if num_coarse_clusters is None:
            num_coarse_clusters = min(max(len(metadata_list) // 25, 3), 12)
        
        logger.info(f"Performing K-means clustering with {num_coarse_clusters} clusters")
        kmeans = KMeans(n_clusters=num_coarse_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(combined_embeddings)
        
        # Create coarse cluster structure
        coarse_cluster_data = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            coarse_cluster_data[f"coarse_{label}"].append(i)
        
        # Store coarse cluster centroids for later search
        coarse_centroids = kmeans.cluster_centers_
        self.coarse_cluster_embeddings = coarse_centroids
        
        # Create FAISS index for coarse clusters using the combined embedding dimension
        self.coarse_cluster_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
        self.coarse_cluster_index.add(coarse_centroids.astype(np.float32))
        
        # Store coarse cluster metadata and statistics
        for cluster_id, indices in coarse_cluster_data.items():
            cluster_metadata = [metadata_list[i] for i in indices]
            self.coarse_clusters[cluster_id] = {
                'indices': indices,
                'metadata': cluster_metadata,
                'centroid_text': self._create_cluster_summary(cluster_metadata),
                'size': len(indices),
                'content_embeddings': content_embeddings[indices],
                'metadata_embeddings': metadata_embeddings[indices] if metadata_embeddings is not None else None
            }
        
        logger.info(f"Created {len(coarse_cluster_data)} coarse clusters")
        return coarse_cluster_data
    
    def _fallback_clustering(self, metadata_list: List[Dict[str, str]]):
        """Fallback clustering when no embedding model is available"""
        clusters = defaultdict(list)
        for i, metadata in enumerate(metadata_list):
            # Create cluster key based on topic and category
            topic = metadata.get('topic', 'general')
            category = metadata.get('category', 'doc')
            cluster_key = f"coarse_{topic}_{category}"
            clusters[cluster_key].append(i)
        
        # Ensure we don't have too many small clusters
        min_cluster_size = max(len(metadata_list) // 20, 2)
        merged_clusters = defaultdict(list)
        other_cluster = []
        
        for cluster_id, indices in clusters.items():
            if len(indices) >= min_cluster_size:
                merged_clusters[cluster_id] = indices
            else:
                other_cluster.extend(indices)
        
        if other_cluster:
            merged_clusters['coarse_other'] = other_cluster
            
        return merged_clusters
    
    def _create_cluster_summary(self, cluster_metadata: List[Dict[str, str]]) -> str:
        """Create a comprehensive summary text for a cluster"""
        if not cluster_metadata:
            return "Empty cluster"
            
        # Analyze cluster composition
        topics = [m.get('topic', '') for m in cluster_metadata if m.get('topic')]
        categories = [m.get('category', '') for m in cluster_metadata if m.get('category')]
        difficulties = [m.get('difficulty', '') for m in cluster_metadata if m.get('difficulty')]
        
        # Get most common elements
        top_topics = Counter(topics).most_common(3)
        top_categories = Counter(categories).most_common(2)
        top_difficulties = Counter(difficulties).most_common(2)
        
        summary_parts = []
        if top_topics:
            topics_str = ', '.join([f"{t[0]}({t[1]})" for t in top_topics if t[0]])
            summary_parts.append(f"Topics: {topics_str}")
        
        if top_categories:
            categories_str = ', '.join([f"{c[0]}({c[1]})" for c in top_categories if c[0]])
            summary_parts.append(f"Categories: {categories_str}")
            
        if top_difficulties:
            difficulties_str = ', '.join([f"{d[0]}({d[1]})" for d in top_difficulties if d[0]])
            summary_parts.append(f"Difficulties: {difficulties_str}")
        
        summary_parts.append(f"Size: {len(cluster_metadata)} chunks")
        
        return " | ".join(summary_parts)
        
    def build_hierarchical_index(self, content_embeddings: np.ndarray, metadata_list: List[Dict[str, str]]):
        """Build hierarchical FAISS index with combined content and metadata similarity"""
        logger.info("Building hierarchical FAISS index with combined similarity...")
        
        if len(content_embeddings) != len(metadata_list):
            raise ValueError("Content embeddings and metadata must have the same length")
        
        # Step 1: Build coarse clusters based on combined similarity
        coarse_cluster_data = self._build_coarse_clusters(content_embeddings, metadata_list)
        
        # Step 2: Build fine-grained FAISS indices within each coarse cluster
        logger.info("Building fine-grained indices within coarse clusters...")
        
        for coarse_cluster_id, cluster_info in self.coarse_clusters.items():
            indices = cluster_info['indices']
            
            if len(indices) < 2:  # Skip clusters with too few items
                logger.warning(f"Skipping cluster {coarse_cluster_id} with only {len(indices)} items")
                continue
                
            # Use content embeddings for fine-grained search (more precise for actual retrieval)
            cluster_content_embeddings = content_embeddings[indices]
            cluster_metadata = cluster_info['metadata']
            
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
    
    def _find_similar_coarse_clusters(self, query_embedding: np.ndarray, query_metadata: Dict[str, str] = None, 
                                    num_clusters: int = 3) -> List[str]:
        """Find most similar coarse clusters using combined content and metadata similarity"""
        if self.coarse_cluster_index is None:
            # Fallback: return all clusters
            return list(self.coarse_clusters.keys())[:num_clusters]
        
        # Create combined query embedding
        if query_metadata and self.metadata_embedding_model:
            # Compute query metadata embedding
            query_metadata_text = self._create_metadata_embedding_text(query_metadata)
            query_metadata_embedding = self.metadata_embedding_model.encode(
                [query_metadata_text], convert_to_numpy=True
            )
            query_metadata_embedding = query_metadata_embedding / np.linalg.norm(
                query_metadata_embedding, axis=1, keepdims=True
            )
            
            # Combine query embeddings with same weights as during indexing
            content_weight = 0.7
            metadata_weight = 0.3
            
            if query_embedding.shape[0] != query_metadata_embedding.shape[1]:
                # Different dimensions - concatenate
                combined_query = np.concatenate([
                    query_embedding.reshape(1, -1) * content_weight,
                    query_metadata_embedding * metadata_weight
                ], axis=1)
            else:
                # Same dimensions - weighted average
                combined_query = (query_embedding.reshape(1, -1) * content_weight + 
                                query_metadata_embedding * metadata_weight)
            
            # Normalize combined query
            combined_query = combined_query / np.linalg.norm(combined_query, axis=1, keepdims=True)
        else:
            # Fallback: extend content embedding to match coarse cluster embedding dimension
            if hasattr(self, 'coarse_cluster_embeddings') and self.coarse_cluster_embeddings is not None:
                target_dim = self.coarse_cluster_embeddings.shape[1]
                current_dim = query_embedding.shape[0]
                
                if current_dim < target_dim:
                    # Pad with zeros
                    combined_query = np.zeros((1, target_dim))
                    combined_query[0, :current_dim] = query_embedding
                elif current_dim > target_dim:
                    # Truncate
                    combined_query = query_embedding[:target_dim].reshape(1, -1)
                else:
                    combined_query = query_embedding.reshape(1, -1)
            else:
                combined_query = query_embedding.reshape(1, -1)
        
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
                    cluster_ids = list(self.coarse_clusters.keys())
                    if idx < len(cluster_ids):
                        selected_clusters.append(cluster_ids[idx])
            
            return selected_clusters if selected_clusters else list(self.coarse_clusters.keys())[:num_clusters]
            
        except Exception as e:
            logger.error(f"Error in coarse cluster search: {e}")
            return list(self.coarse_clusters.keys())[:num_clusters]
    
    def _search_cluster_parallel(self, cluster_id: str, query_embedding: np.ndarray, k: int) -> List[Tuple[float, str, int, Dict]]:
        """Search a single cluster (for parallel execution) using content embeddings"""
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
    
    def search(self, query_embedding: np.ndarray, query_text: str = None, k: int = 5, 
              num_coarse_clusters: int = 3) -> List[Tuple[float, int, Dict]]:
        """
        Search across hierarchical index with query metadata integration
        
        Args:
            query_embedding: Content embedding of the query
            query_text: Original query text for metadata extraction
            k: Number of results to return
            num_coarse_clusters: Number of coarse clusters to search
        """
        # Extract metadata from query if provided
        query_metadata = None
        if query_text:
            query_metadata = self._extract_query_metadata(query_text)
            logger.debug(f"Extracted query metadata: {query_metadata}")
        
        # Step 1: Find most similar coarse clusters using combined similarity
        selected_clusters = self._find_similar_coarse_clusters(
            query_embedding, query_metadata, num_coarse_clusters
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
                    cluster_results = future.result(timeout=10)  # 10 second timeout
                    all_results.extend(cluster_results)
                except Exception as e:
                    logger.error(f"Error in parallel search for cluster {cluster_id}: {e}")
        
        # Step 3: Re-rank results considering metadata similarity if available
        if query_metadata and self.chunk_metadata_embeddings is not None:
            all_results = self._rerank_with_metadata(all_results, query_metadata)
        
        # Step 4: Sort by score and return top k
        all_results.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to expected format (score, global_idx, metadata)
        final_results = []
        for score, cluster_id, local_idx, metadata in all_results[:k]:
            # Create a more meaningful global index
            global_idx = hash(f"{cluster_id}_{local_idx}") % 1000000
            final_results.append((score, global_idx, metadata))
        
        return final_results
    
    def _rerank_with_metadata(self, results: List[Tuple[float, str, int, Dict]], 
                            query_metadata: Dict[str, str]) -> List[Tuple[float, str, int, Dict]]:
        """Re-rank results by combining content and metadata similarity"""
        if not self.metadata_embedding_model:
            return results
        
        try:
            # Compute query metadata embedding
            query_metadata_text = self._create_metadata_embedding_text(query_metadata)
            query_metadata_embedding = self.metadata_embedding_model.encode(
                [query_metadata_text], convert_to_numpy=True
            )
            query_metadata_embedding = query_metadata_embedding / np.linalg.norm(
                query_metadata_embedding, axis=1, keepdims=True
            )
            
            # Re-rank results
            reranked_results = []
            for content_score, cluster_id, local_idx, metadata in results:
                # Get metadata embedding for this result
                result_metadata_text = self._create_metadata_embedding_text(metadata)
                result_metadata_embedding = self.metadata_embedding_model.encode(
                    [result_metadata_text], convert_to_numpy=True
                )
                result_metadata_embedding = result_metadata_embedding / np.linalg.norm(
                    result_metadata_embedding, axis=1, keepdims=True
                )
                
                # Compute metadata similarity
                metadata_score = np.dot(query_metadata_embedding[0], result_metadata_embedding[0])
                
                # Combine scores (weighted average)
                combined_score = 0.7 * content_score + 0.3 * metadata_score
                reranked_results.append((combined_score, cluster_id, local_idx, metadata))
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in metadata re-ranking: {e}")
            return results
    
    def save_index(self, path: str):
        """Save hierarchical index to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save coarse cluster index
        if self.coarse_cluster_index is not None:
            faiss.write_index(self.coarse_cluster_index, os.path.join(path, "coarse_index.faiss"))
        
        # Save fine cluster FAISS indices
        for cluster_id, index in self.fine_cluster_indices.items():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(path, f"fine_{safe_name}.index"))
        
        # Save all metadata and mappings
        metadata_path = os.path.join(path, "hierarchical_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'coarse_clusters': self.coarse_clusters,
                'fine_cluster_metadata': self.fine_cluster_metadata,
                'coarse_cluster_embeddings': self.coarse_cluster_embeddings,
                'chunk_metadata_embeddings': self.chunk_metadata_embeddings,
                'chunk_metadata_texts': self.chunk_metadata_texts,
                'global_to_local': self.global_to_local,
                'embedding_dim': self.embedding_dim,
                'next_global_id': self.next_global_id
            }, f)
        
        logger.info(f"Saved hierarchical index to {path}")
    
    def load_index(self, path: str):
        """Load hierarchical index from disk"""
        # Load metadata and mappings
        metadata_path = os.path.join(path, "hierarchical_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.coarse_clusters = data.get('coarse_clusters', {})
            self.fine_cluster_metadata = data.get('fine_cluster_metadata', {})
            self.coarse_cluster_embeddings = data.get('coarse_cluster_embeddings', None)
            self.chunk_metadata_embeddings = data.get('chunk_metadata_embeddings', None)
            self.chunk_metadata_texts = data.get('chunk_metadata_texts', [])
            self.global_to_local = data.get('global_to_local', {})
            self.embedding_dim = data.get('embedding_dim', 384)
            self.next_global_id = data.get('next_global_id', 0)
        
        # Load coarse cluster index
        coarse_index_path = os.path.join(path, "coarse_index.faiss")
        if os.path.exists(coarse_index_path):
            self.coarse_cluster_index = faiss.read_index(coarse_index_path)
        
        # Load fine cluster FAISS indices
        self.fine_cluster_indices = {}
        for cluster_id in self.fine_cluster_metadata.keys():
            safe_name = cluster_id.replace('/', '_').replace('\\', '_')
            index_path = os.path.join(path, f"fine_{safe_name}.index")
            if os.path.exists(index_path):
                self.fine_cluster_indices[cluster_id] = faiss.read_index(index_path)
        
        logger.info(f"Loaded hierarchical index from {path}")

def evaluate_faiss_index(index: HierarchicalFAISSIndex, query_embeddings: np.ndarray, 
                        query_texts: List[str] = None, ground_truth: List[List[int]] = None) -> Dict[str, float]:
    """Evaluate FAISS index performance with metadata integration"""
    logger.info("Evaluating hierarchical FAISS index...")
    
    total_queries = len(query_embeddings)
    avg_recall = 0.0
    search_times = []
    cluster_distribution = defaultdict(int)
    
    import time
    for i, query_emb in enumerate(query_embeddings):
        query_text = query_texts[i] if query_texts and i < len(query_texts) else None
        
        start_time = time.time()
        results = index.search(query_emb, query_text=query_text, k=10, num_coarse_clusters=3)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Track cluster distribution
        for score, idx, metadata in results:
            cluster_key = f"{metadata.get('topic', 'unknown')}_{metadata.get('category', 'unknown')}"
            cluster_distribution[cluster_key] += 1
        
        if ground_truth and i < len(ground_truth):
            retrieved_ids = [r[1] for r in results]
            relevant_ids = ground_truth[i]
            if relevant_ids:
                recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
                avg_recall += recall
    
    metrics = {
        'avg_search_time': np.mean(search_times),
        'std_search_time': np.std(search_times),
        'avg_recall': avg_recall / total_queries if ground_truth else 0.0,
        'total_coarse_clusters': len(index.coarse_clusters),
        'total_fine_clusters': len(index.fine_cluster_indices),
        'cluster_distribution': dict(cluster_distribution)
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics