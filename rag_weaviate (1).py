from typing import List, Dict, Optional, Any
import requests
import numpy as np
import weaviate
import weaviate.classes as wvc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain_community.document_loaders import WikipediaLoader


class CustomEmbeddingFunction:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using your API"""
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
            return response.json().get("embeddings", [])
        raise Exception(f"Embedding error: {response.status_code}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using your API"""
        return self.embed_documents([text])[0]


class WeaviateVectorDatabase:
    def __init__(self, embedding_function: CustomEmbeddingFunction, 
                 weaviate_url: str = "http://localhost:8080", 
                 class_name: str = "Document"):
        self.embedding_function = embedding_function
        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Weaviate instance"""
        try:
            # Weaviate v4 client initialization
            self.client = weaviate.connect_to_local(
                host=self.weaviate_url.replace("http://", "").replace("https://", "")
            )
            
            # Create collection if it doesn't exist
            if not self.client.collections.exists(self.class_name):
                self._create_collection()
                
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise
    
    def _create_collection(self):
        """Create Weaviate collection for documents"""
        try:
            self.client.collections.create(
                name=self.class_name,
                description="A document chunk for RAG retrieval",
                properties=[
                    wvc.config.Property(
                        name="content",
                        data_type=wvc.config.DataType.TEXT,
                        description="The text content of the document chunk"
                    ),
                    wvc.config.Property(
                        name="chunk_id",
                        data_type=wvc.config.DataType.INT,
                        description="The ID of the chunk"
                    ),
                    wvc.config.Property(
                        name="source",
                        data_type=wvc.config.DataType.TEXT,
                        description="The source of the document"
                    ),
                    wvc.config.Property(
                        name="url",
                        data_type=wvc.config.DataType.TEXT,
                        description="The URL of the document source"
                    )
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.none()
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to Weaviate with embeddings"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings for all texts
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Get collection
        collection = self.client.collections.get(self.class_name)
        
        # Prepare data objects
        objects = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            obj = wvc.data.DataObject(
                properties={
                    "content": text,
                    "chunk_id": metadata.get("chunk_id", i),
                    "source": metadata.get("source", "unknown"),
                    "url": metadata.get("url", "unknown")
                },
                vector=embedding
            )
            objects.append(obj)
        
        # Batch insert
        collection.data.insert_many(objects)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        query_embedding = self.embedding_function.embed_query(query)
        
        # Get collection
        collection = self.client.collections.get(self.class_name)
        
        # Perform vector search
        result = collection.query.near_vector(
            near_vector=query_embedding,
            limit=k,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        documents = []
        for obj in result.objects:
            doc = Document(
                page_content=obj.properties["content"],
                metadata={
                    "chunk_id": obj.properties["chunk_id"],
                    "source": obj.properties["source"],
                    "url": obj.properties.get("url", "unknown"),
                    "distance": obj.metadata.distance if obj.metadata else None
                }
            )
            documents.append(doc)
        
        return documents
    
    def delete_all(self):
        """Delete all documents from the collection"""
        try:
            if self.client.collections.exists(self.class_name):
                self.client.collections.delete(self.class_name)
                self._create_collection()
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def close(self):
        """Close the Weaviate client connection"""
        if self.client:
            self.client.close()


class RAGPipeline:
    def __init__(self, embedding_api_url: str, llm_api_url: str, 
                 weaviate_url: str = "http://localhost:8080"):
        self.embedding_function = CustomEmbeddingFunction(embedding_api_url)
        self.vector_db = WeaviateVectorDatabase(self.embedding_function, weaviate_url)
        self.llm_api_url = llm_api_url
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.generator = self._setup_generator()
    
    def _setup_generator(self):
        """Setup the generator component with prompt template"""
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        def generate_response(inputs: Dict[str, str]) -> str:
            """Wrapper function to call LLM API"""
            context = inputs["context"]
            question = inputs["question"]
            full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            return self._call_llm_api(full_prompt)
        
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | generate_response
        )
    
    def _call_llm_api(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Call your LLM API"""
        payload = {
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": prompt
            },
            "model": {
                "name": "gemini-1.5-flash",
                "top_p": 1,
                "temperature": 1,
                "max_tokens": 1000,
                "do_sample": True,
                "repetition_penalty": 0,
                "json_mode": False
            }
        }
        
        response = requests.post(url=self.llm_api_url, json=payload, verify=False)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        raise Exception(f"LLM API error: {response.status_code}")
    
    def load_yemen_attack_data(self):
        """Load Wikipedia data about March-May 2025 US attacks in Yemen"""
        try:
            # Load Wikipedia article using WikipediaLoader
            loader = WikipediaLoader(
                query="March–May 2025 United States attacks in Yemen",
                load_max_docs=1,
                doc_content_chars_max=10000
            )
            
            documents = loader.load()
            
            if documents:
                print(f"Successfully loaded Wikipedia article: {documents[0].metadata.get('title', 'Unknown')}")
                
                # Process each document
                for doc in documents:
                    chunks = self.text_splitter.split_text(doc.page_content)
                    metadatas = [
                        {
                            "chunk_id": i, 
                            "source": f"Wikipedia: {doc.metadata.get('title', 'Unknown')}",
                            "url": doc.metadata.get('source', 'Unknown')
                        } 
                        for i in range(len(chunks))
                    ]
                    
                    self.vector_db.add_documents(chunks, metadatas)
                    print(f"Added {len(chunks)} chunks from Wikipedia article")
            else:
                print("No Wikipedia article found. Loading fallback content...")
                self._load_fallback_yemen_data()
                
        except Exception as e:
            print(f"Error loading from Wikipedia: {e}")
            print("Loading fallback content...")
            self._load_fallback_yemen_data()
    
    def _load_fallback_yemen_data(self):
        """Fallback content if Wikipedia loader fails"""
        fallback_text = """
        March–May 2025 United States attacks in Yemen

        In March 2025, the United States launched a large campaign of air and naval strikes against Houthi targets in Yemen. Codenamed Operation Rough Rider, it has been the largest U.S. military operation in the Middle East of President Donald Trump's second term. The strikes began on March 15, targeting radar systems, air defenses, and ballistic and drone launch sites used by the Houthis to attack commercial ships and naval vessels in the Red Sea and Gulf of Aden.

        Background
        The Houthi group began targeting international shipping in the Red Sea in response to Israeli operations in Gaza. These attacks disrupted global trade routes, forcing many vessels to take longer routes around the Cape of Good Hope, adding significant costs and delays to international commerce.

        Operation Details
        The operation, designated as Operation Rough Rider, represented the most significant U.S. military engagement in the region during Trump's second presidency. The strikes targeted critical Houthi infrastructure including radar and surveillance systems, air defense installations, ballistic missile launch sites, drone facilities and launch platforms, and command and control centers.

        Timeline
        March 15, 2025: Initial wave of strikes begins targeting Houthi military infrastructure
        March 16-31, 2025: Continued operations against radar systems and launch sites
        April 1-29, 2025: Sustained campaign targeting air defenses and missile capabilities
        April 30, 2025: United Kingdom joins the United States in conducting strikes on Houthi targets
        May 1-31, 2025: Joint US-UK operations continue against remaining Houthi capabilities

        International Response
        On April 30, 2025, the United Kingdom joined the United States in conducting strikes on Houthi targets, marking an expansion of the coalition effort. The joint operations aimed to degrade Houthi capabilities to attack Red Sea shipping.

        Impact
        The strikes significantly impacted Houthi military capabilities, though the group continued to pose threats to international shipping. The operation aimed to protect international shipping lanes and restore stability to Red Sea commerce.
        """
        
        self.ingest_text(fallback_text, source="Fallback: March-May 2025 US attacks in Yemen")
    
    def ingest_documents(self, file_path: str):
        """Process documents from file and store embeddings"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.ingest_text(text, source=file_path)
    
    def ingest_text(self, text: str, source: str = "unknown"):
        """Process text and store embeddings using RecursiveCharacterTextSplitter"""
        chunks = self.text_splitter.split_text(text)
        metadatas = [{"chunk_id": i, "source": source} for i in range(len(chunks))]
        
        self.vector_db.add_documents(chunks, metadatas)
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        docs = self.vector_db.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        answer = self.generator.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "source_documents": docs,
            "context": context
        }
    
    def clear_database(self):
        """Clear all data from the vector database"""
        self.vector_db.delete_all()
    
    def close(self):
        """Close database connections"""
        self.vector_db.close()


# Example Usage
if __name__ == "__main__":
    # Initialize with your API URLs
    embedding_api_url = "YOUR_EMBEDDING_API_URL"
    llm_api_url = "YOUR_LLM_API_URL"
    weaviate_url = "http://localhost:8080"  # Default Weaviate URL
    
    # Create pipeline
    rag = RAGPipeline(embedding_api_url, llm_api_url, weaviate_url)
    
    # Load Yemen attack data from Wikipedia
    print("Loading Yemen attack data from Wikipedia...")
    rag.load_yemen_attack_data()
    
    # Optional: Load additional documents from file
    # rag.ingest_documents("additional_document.txt")
    
    # Query the system
    questions = [
        "What was Operation Rough Rider?",
        "When did the US attacks in Yemen begin in 2025?",
        "Which countries participated in the strikes against Yemen?",
        "What were the main targets of the US strikes?",
        "How many casualties resulted from the strikes?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['source_documents'])} documents")
        for i, doc in enumerate(result['source_documents']):
            distance = doc.metadata.get('distance', 'N/A')
            print(f"  Source {i+1}: {doc.metadata['source']} (Chunk {doc.metadata['chunk_id']}, Distance: {distance})")
        print("-" * 50)
    
    # Close connections when done
    rag.close()
