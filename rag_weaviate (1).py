import re
import json
from typing import List, Dict, Any
from xml.etree import ElementTree as ET

def get_chunk_and_metadata_from_llm(document_content: str, llm_client=None) -> List[Dict]:
    """
    Simultaneously extract chunks, metadata, and related queries using XML format
    Returns list of dictionaries with chunk data
    """
    
    xml_prompt = f"""
<task>
Analyze the following document and extract meaningful chunks with their metadata and related queries.
Each chunk should contain substantial information and avoid duplication.
Return the response in a structured JSON format for easy parsing.
</task>

<document>
{document_content}
</document>

<instructions>
1. Break the document into meaningful, coherent chunks (200-800 words each)
2. For each chunk, extract comprehensive metadata
3. Generate 3-5 related queries that this chunk can answer
4. Skip chunks that don't add new information or are too short
5. Ensure chunks are self-contained and meaningful
6. Make sure query metadata is different from chunk metadata (queries should be questions/search terms)
7. Return the entire response as a valid JSON array
</instructions>

<output_format>
Return a JSON array where each element is a chunk object with this structure:
{{
    "chunk_id": "1",
    "text": "Full chunk text content here...",
    "metadata": {{
        "topic": "Main topic/subject of this chunk",
        "category": "Document category (e.g., technical, legal, medical, etc.)",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "summary": "Brief 1-2 sentence summary of chunk content",
        "document_type": "Type of document (e.g., manual, report, article, etc.)",
        "domain": "Domain/field (e.g., healthcare, finance, technology, etc.)",
        "word_count": "Approximate word count",
        "key_concepts": ["concept1", "concept2", "concept3"]
    }},
    "queries": [
        {{
            "query": "What is the main topic discussed in this section?",
            "priority": 5,
            "query_type": "factual"
        }},
        {{
            "query": "How does this relate to the broader document context?",
            "priority": 4,
            "query_type": "analytical"
        }},
        {{
            "query": "What are the key points mentioned here?",
            "priority": 3,
            "query_type": "summarization"
        }}
    ]
}}

IMPORTANT: 
- Return ONLY the JSON array, no additional text
- Ensure all JSON is properly formatted and valid
- Each chunk should be completely self-contained
- Queries should be actual questions that users might ask about the content
- Make sure the entire document is covered without significant gaps
</output_format>
"""

    try:
        # Call LLM (you'll need to replace this with your actual LLM client call)
        if llm_client:
            response = llm_client.generate(xml_prompt)
        else:
            # Placeholder - replace with your actual LLM call
            raise ValueError("LLM client not provided. Please pass your LLM client instance.")
        
        # Parse the JSON response
        chunks_data = parse_llm_response(response)
        
        # Validate and clean the data
        validated_chunks = validate_and_clean_chunks(chunks_data)
        
        return validated_chunks
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        # Fallback to simple chunking if LLM fails
        return fallback_chunking(document_content)

def parse_llm_response(response_text: str) -> List[Dict]:
    """
    Parse LLM response - handles both JSON and XML formats
    """
    response_text = response_text.strip()
    
    # Try parsing as JSON first
    try:
        # Remove any markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Try to parse as JSON
        chunks_data = json.loads(response_text)
        if isinstance(chunks_data, list):
            return chunks_data
        else:
            return [chunks_data] if isinstance(chunks_data, dict) else []
            
    except json.JSONDecodeError:
        # If JSON parsing fails, try XML parsing
        return parse_xml_response(response_text)

def parse_xml_response(xml_text: str) -> List[Dict]:
    """
    Parse XML format response as fallback
    """
    chunks = []
    
    try:
        # Find all chunk blocks using regex
        chunk_pattern = r'<chunk id="(\d+)">(.*?)</chunk>'
        chunk_matches = re.findall(chunk_pattern, xml_text, re.DOTALL)
        
        for chunk_id, chunk_content in chunk_matches:
            chunk_dict = {"chunk_id": chunk_id}
            
            # Extract text
            text_match = re.search(r'<text>(.*?)</text>', chunk_content, re.DOTALL)
            if text_match:
                chunk_dict["text"] = text_match.group(1).strip()
            
            # Extract metadata
            metadata = {}
            metadata_match = re.search(r'<metadata>(.*?)</metadata>', chunk_content, re.DOTALL)
            if metadata_match:
                metadata_content = metadata_match.group(1)
                
                # Parse individual metadata fields
                for field in ['topic', 'category', 'summary', 'document_type', 'domain']:
                    field_match = re.search(f'<{field}>(.*?)</{field}>', metadata_content, re.DOTALL)
                    if field_match:
                        metadata[field] = field_match.group(1).strip()
                
                # Parse keywords
                keywords_match = re.search(r'<keywords>(.*?)</keywords>', metadata_content, re.DOTALL)
                if keywords_match:
                    keywords_text = keywords_match.group(1).strip()
                    metadata["keywords"] = [k.strip() for k in keywords_text.split(',')]
                
            chunk_dict["metadata"] = metadata
            
            # Extract queries
            queries = []
            queries_match = re.search(r'<queries>(.*?)</queries>', chunk_content, re.DOTALL)
            if queries_match:
                queries_content = queries_match.group(1)
                query_matches = re.findall(r'<query priority="(\d+)">(.*?)</query>', queries_content)
                
                for priority, query_text in query_matches:
                    queries.append({
                        "query": query_text.strip(),
                        "priority": int(priority),
                        "query_type": "general"
                    })
            
            chunk_dict["queries"] = queries
            chunks.append(chunk_dict)
            
    except Exception as e:
        print(f"Error parsing XML response: {str(e)}")
        
    return chunks

def validate_and_clean_chunks(chunks_data: List[Dict]) -> List[Dict]:
    """
    Validate and clean chunk data
    """
    validated_chunks = []
    
    for i, chunk in enumerate(chunks_data):
        try:
            # Ensure required fields exist
            if not chunk.get("text") or len(chunk["text"].strip()) < 50:
                continue  # Skip chunks that are too short
            
            validated_chunk = {
                "chunk_id": chunk.get("chunk_id", str(i + 1)),
                "text": chunk["text"].strip(),
                "metadata": chunk.get("metadata", {}),
                "queries": chunk.get("queries", [])
            }
            
            # Ensure metadata has required fields
            metadata = validated_chunk["metadata"]
            required_metadata_fields = ["topic", "category", "summary", "document_type", "domain"]
            for field in required_metadata_fields:
                if field not in metadata:
                    metadata[field] = "Unknown"
            
            # Ensure keywords is a list
            if "keywords" not in metadata or not isinstance(metadata["keywords"], list):
                metadata["keywords"] = []
            
            # Add word count
            metadata["word_count"] = len(validated_chunk["text"].split())
            
            # Ensure queries is a list with proper structure
            queries = validated_chunk["queries"]
            if not isinstance(queries, list):
                queries = []
            
            # Validate each query
            validated_queries = []
            for query in queries:
                if isinstance(query, dict) and "query" in query:
                    validated_queries.append({
                        "query": query["query"],
                        "priority": query.get("priority", 1),
                        "query_type": query.get("query_type", "general")
                    })
                elif isinstance(query, str):
                    validated_queries.append({
                        "query": query,
                        "priority": 1,
                        "query_type": "general"
                    })
            
            validated_chunk["queries"] = validated_queries
            validated_chunks.append(validated_chunk)
            
        except Exception as e:
            print(f"Error validating chunk {i}: {str(e)}")
            continue
    
    return validated_chunks

def fallback_chunking(document_content: str) -> List[Dict]:
    """
    Fallback chunking method if LLM processing fails
    """
    # Simple sentence-based chunking
    sentences = re.split(r'[.!?]+', document_content)
    chunks = []
    current_chunk = ""
    chunk_id = 1
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) > 600 and current_chunk:
            # Create chunk
            chunks.append({
                "chunk_id": str(chunk_id),
                "text": current_chunk.strip(),
                "metadata": {
                    "topic": "Unknown",
                    "category": "General",
                    "keywords": [],
                    "summary": current_chunk[:100] + "...",
                    "document_type": "Unknown",
                    "domain": "General",
                    "word_count": len(current_chunk.split())
                },
                "queries": [{
                    "query": "What information is contained in this section?",
                    "priority": 3,
                    "query_type": "general"
                }]
            })
            current_chunk = sentence
            chunk_id += 1
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_id": str(chunk_id),
            "text": current_chunk.strip(),
            "metadata": {
                "topic": "Unknown",
                "category": "General", 
                "keywords": [],
                "summary": current_chunk[:100] + "...",
                "document_type": "Unknown",
                "domain": "General",
                "word_count": len(current_chunk.split())
            },
            "queries": [{
                "query": "What information is contained in this section?",
                "priority": 3,
                "query_type": "general"
            }]
        })
    
    return chunks

# Example usage:
if __name__ == "__main__":
    # Example document
    sample_doc = """
    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms 
    that can learn from and make predictions or decisions based on data. Unlike traditional programming, 
    where explicit instructions are given to solve a problem, machine learning algorithms build mathematical 
    models based on training data to make predictions or decisions without being explicitly programmed for 
    every possible scenario.
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and 
    reinforcement learning. Supervised learning uses labeled training data to learn a mapping from 
    inputs to outputs. Common examples include classification and regression tasks.
    """
    
    # You would call it like this with your LLM client:
    # chunks = get_chunk_and_metadata_from_llm(sample_doc, your_llm_client)
    
    # For demonstration, showing the fallback:
    chunks = fallback_chunking(sample_doc)
    print(json.dumps(chunks, indent=2))
