# config.py - Enhanced configuration for simultaneous processing

import os

class Config:
    # Data paths
    DATA_FOLDER = "data"
    INDEX_PATH = "enhanced_faiss_index"
    LLM_CHUNK = "llm_chunks"
    CLUSTER_ASSIGNMENTS_PATH = "cluster_assignments"
    
    # Embedding configuration
    EMBEDDING_DIM = 384  # Adjust based on your embedding model
    METADATA_DIM = 128   # Reduced dimension for metadata clustering
    
    # Clustering parameters
    NUM_METADATA_CLUSTERS = 50
    TOP_K_CLUSTER_ASSIGNMENT = 3  # Assign each chunk to top-k clusters
    
    # Search parameters
    DEFAULT_SEARCH_K = 5
    DEFAULT_METADATA_CLUSTERS = 5
    HNSW_EF_SEARCH = None  # Auto-adjust if None
    
    # Processing parameters
    MIN_CHUNK_LENGTH = 50    # Minimum characters for a chunk
    MAX_CHUNK_LENGTH = 800   # Maximum words for a chunk
    MIN_WORD_COUNT = 20      # Minimum words for relevance
    
    # Weight configuration
    QUERY_WEIGHTS = {
        'primary': 5,
        'secondary': 4, 
        'tertiary': 3,
        'additional': 1
    }
    
    # LLM configuration
    LLM_MODEL = "gpt-4"  # Or your preferred model
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 4000
    
    # Performance settings
    MAX_WORKERS = 6          # For parallel processing
    TIMEOUT_SECONDS = 15     # Timeout for cluster searches
    
    # Deduplication settings
    CONTENT_SIMILARITY_THRESHOLD = 0.95  # For near-duplicate detection
    
    # File output settings
    SAVE_DETAILED_ASSIGNMENTS = True
    SAVE_CLUSTER_STATISTICS = True
    OUTPUT_FORMAT = "json"   # json or txt
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "enhanced_faiss.log"
    
    # Evaluation settings
    RELEVANCE_THRESHOLD = 0.5  # Minimum relevance score to include chunk
    ENABLE_LLM_RELEVANCE_CHECK = True  # Use LLM for relevance evaluation
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_FOLDER,
            cls.INDEX_PATH, 
            cls.LLM_CHUNK,
            cls.CLUSTER_ASSIGNMENTS_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_cluster_assignment_file(cls, cluster_id: str) -> str:
        """Get file path for cluster assignment"""
        safe_name = cluster_id.replace('/', '_').replace('\\', '_')
        return os.path.join(cls.CLUSTER_ASSIGNMENTS_PATH, f"{safe_name}_assignments.json")
    
    @classmethod
    def get_summary_file(cls) -> str:
        """Get path for cluster summary file"""
        return os.path.join(cls.CLUSTER_ASSIGNMENTS_PATH, "cluster_summary.json")

# Global config instance
config = Config()

# Initialize directories on import
config.create_directories()