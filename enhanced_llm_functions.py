# llm_fun.py - Enhanced LLM functions with XML format and simultaneous processing

import json
import re
import logging
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def get_chunk_and_metadata_from_llm(document_content: str) -> List[Dict]:
    """
    Simultaneously extract chunks, metadata, and related queries using XML format
    Returns list of dictionaries with chunk data
    """
    
    xml_prompt = f"""
<task>
Analyze the following document and extract meaningful chunks with their metadata and related queries.
Each chunk should contain substantial information and avoid duplication.
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

Format your response as follows for each chunk:
</instructions>

<output_format>
<chunk id="1">
    <text>
    [Chunk text content here]
    </text>
    
    <metadata>
        <topic>[Main topic/subject]</topic>
        <category>[Document category]</category>
        <keywords>[Comma-separated keywords]</keywords>
        <summary>[Brief summary]</summary>
        <document_type>[Type of document]</document_type>
        <domain>[Domain/field]</domain>
    </metadata>
    
    <queries>
        <query priority="5">[Most important query this chunk answers]</query>
        <query priority="4">[Second most important query]</query>
        <query priority="3">[Third most important query]</query>
        <query priority="1">[Additional relevant query]</query>
        <query priority="1">[Another additional query if applicable]</query>
    </queries>
</chunk>

<chunk id="2">
    [Next chunk following same format]
</chunk>
...
</output_format>

Begin processing:
"""

    try:
        # Call your LLM service here (replace with actual LLM call)
        response = call_llm_service(xml_prompt)
        
        # Parse the XML response
        chunks_data = parse_xml_chunks_response(response)
        
        logger.info(f"Extracted {len(chunks_data)} chunks with metadata and queries")
        return chunks_data
        
    except Exception as e:
        logger.error(f"Error in simultaneous chunk/metadata extraction: {e}")
        # Fallback to basic chunking
        return fallback_basic_chunking(document_content)


def parse_xml_chunks_response(xml_response: str) -> List[Dict]:
    """Parse XML response containing chunks, metadata, and queries"""
    chunks_data = []
    
    try:
        # Clean the response and extract chunk sections
        chunk_pattern = r'<chunk id="(\d+)">(.*?)</chunk>'
        chunk_matches = re.findall(chunk_pattern, xml_response, re.DOTALL)
        
        for chunk_id, chunk_content in chunk_matches:
            chunk_data = parse_single_chunk(chunk_content)
            if chunk_data:
                chunk_data['chunk_id'] = chunk_id
                chunks_data.append(chunk_data)
    
    except Exception as e:
        logger.error(f"Error parsing XML chunks response: {e}")
        # Try alternative parsing
        chunks_data = parse_xml_alternative(xml_response)
    
    return chunks_data


def parse_single_chunk(chunk_content: str) -> Dict:
    """Parse a single chunk's content including text, metadata, and queries"""
    try:
        # Extract text
        text_match = re.search(r'<text>(.*?)</text>', chunk_content, re.DOTALL)
        chunk_text = text_match.group(1).strip() if text_match else ""
        
        if len(chunk_text) < 50:  # Skip very short chunks
            return None
        
        # Extract metadata
        metadata = {}
        metadata_section = re.search(r'<metadata>(.*?)</metadata>', chunk_content, re.DOTALL)
        if metadata_section:
            metadata_content = metadata_section.group(1)
            
            # Parse individual metadata fields
            metadata_fields = [
                'topic', 'category', 'keywords', 'summary', 
                'document_type', 'domain'
            ]
            
            for field in metadata_fields:
                field_match = re.search(f'<{field}>(.*?)</{field}>', metadata_content, re.DOTALL)
                if field_match:
                    metadata[field] = field_match.group(1).strip()
        
        # Extract queries with priorities
        queries = []
        query_weights = []
        queries_section = re.search(r'<queries>(.*?)</queries>', chunk_content, re.DOTALL)
        
        if queries_section:
            queries_content = queries_section.group(1)
            query_matches = re.findall(r'<query priority="(\d+)">(.*?)</query>', queries_content, re.DOTALL)
            
            # Sort by priority (5, 4, 3, 1, 1...)
            query_matches.sort(key=lambda x: int(x[0]), reverse=True)
            
            for priority, query_text in query_matches:
                queries.append(query_text.strip())
                query_weights.append(int(priority))
        
        # Create metadata text for embedding
        metadata_text = create_metadata_embedding_text(metadata)
        
        return {
            'chunk_text': chunk_text,
            'metadata': metadata,
            'metadata_text': metadata_text,
            'queries': queries,
            'query_weights': query_weights
        }
        
    except Exception as e:
        logger.error(f"Error parsing single chunk: {e}")
        return None


def parse_xml_alternative(xml_response: str) -> List[Dict]:
    """Alternative XML parsing method using ElementTree"""
    chunks_data = []
    
    try:
        # Wrap response in root element if needed
        if not xml_response.strip().startswith('<root>'):
            xml_response = f"<root>{xml_response}</root>"
        
        root = ET.fromstring(xml_response)
        
        for chunk_elem in root.findall('.//chunk'):
            chunk_data = {}
            
            # Extract text
            text_elem = chunk_elem.find('text')
            if text_elem is not None and text_elem.text:
                chunk_text = text_elem.text.strip()
                if len(chunk_text) < 50:
                    continue
                chunk_data['chunk_text'] = chunk_text
            
            # Extract metadata
            metadata = {}
            metadata_elem = chunk_elem.find('metadata')
            if metadata_elem is not None:
                for field in ['topic', 'category', 'keywords', 'summary', 'document_type', 'domain']:
                    field_elem = metadata_elem.find(field)
                    if field_elem is not None and field_elem.text:
                        metadata[field] = field_elem.text.strip()
            
            chunk_data['metadata'] = metadata
            chunk_data['metadata_text'] = create_metadata_embedding_text(metadata)
            
            # Extract queries with priorities
            queries = []
            query_weights = []
            queries_elem = chunk_elem.find('queries')
            if queries_elem is not None:
                query_elems = queries_elem.findall('query')
                
                # Sort queries by priority
                query_data = []
                for query_elem in query_elems:
                    if query_elem.text:
                        priority = int(query_elem.get('priority', '1'))
                        query_data.append((priority, query_elem.text.strip()))
                
                query_data.sort(key=lambda x: x[0], reverse=True)
                
                for priority, query_text in query_data:
                    queries.append(query_text)
                    query_weights.append(priority)
            
            chunk_data['queries'] = queries
            chunk_data['query_weights'] = query_weights
            chunk_data['chunk_id'] = chunk_elem.get('id', str(len(chunks_data)))
            
            if chunk_data.get('chunk_text'):
                chunks_data.append(chunk_data)
    
    except Exception as e:
        logger.error(f"Error in alternative XML parsing: {e}")
    
    return chunks_data


def create_metadata_embedding_text(metadata: Dict[str, str]) -> str:
    """Create text representation of metadata for embedding"""
    metadata_parts = []
    
    # Prioritize important fields
    priority_fields = ['topic', 'summary', 'category', 'domain']
    for field in priority_fields:
        if field in metadata and metadata[field]:
            metadata_parts.append(f"{field}: {metadata[field]}")
    
    # Add other fields
    for key, value in metadata.items():
        if key not in priority_fields and value:
            metadata_parts.append(f"{key}: {value}")
    
    return ". ".join(metadata_parts)


def fallback_basic_chunking(document_content: str) -> List[Dict]:
    """Fallback basic chunking if XML parsing fails"""
    logger.warning("Using fallback basic chunking")
    
    # Simple sentence-based chunking
    sentences = re.split(r'[.!?]+', document_content)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if current_length + len(sentence) > 600 and current_chunk:
            # Create chunk
            chunk_text = '. '.join(current_chunk) + '.'
            chunk_data = {
                'chunk_text': chunk_text,
                'metadata': {
                    'topic': 'General',
                    'category': 'Document',
                    'summary': chunk_text[:100] + '...',
                    'keywords': '',
                    'document_type': 'Text',
                    'domain': 'General'
                },
                'queries': [
                    f"What information is provided about {chunk_text.split()[0:3]}?",
                    f"What are the main points discussed?",
                    f"What details are mentioned?"
                ],
                'query_weights': [5, 4, 3],
                'chunk_id': str(len(chunks))
            }
            chunk_data['metadata_text'] = create_metadata_embedding_text(chunk_data['metadata'])
            chunks.append(chunk_data)
            
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    # Add final chunk if exists
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunk_data = {
            'chunk_text': chunk_text,
            'metadata': {
                'topic': 'General',
                'category': 'Document',
                'summary': chunk_text[:100] + '...',
                'keywords': '',
                'document_type': 'Text',
                'domain': 'General'
            },
            'queries': [
                f"What information is provided about {chunk_text.split()[0:3]}?",
                "What are the main points discussed?",
                "What details are mentioned?"
            ],
            'query_weights': [5, 4, 3],
            'chunk_id': str(len(chunks))
        }
        chunk_data['metadata_text'] = create_metadata_embedding_text(chunk_data['metadata'])
        chunks.append(chunk_data)
    
    return chunks


def call_llm_service(prompt: str) -> str:
    """
    Call your LLM service with the XML prompt
    Replace this with your actual LLM API call
    """
    try:
        # Example implementation - replace with your LLM service
        import openai  # or your preferred LLM client
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are an expert document analyzer. Follow the XML format exactly as specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error calling LLM service: {e}")
        raise


def get_metadata_from_llm(chunk_text: str) -> Dict[str, str]:
    """
    Legacy function for backward compatibility
    Extract metadata from a single chunk using XML format
    """
    xml_prompt = f"""
<task>
Extract comprehensive metadata for the following text chunk.
</task>

<chunk>
{chunk_text}
</chunk>

<instructions>
Analyze the chunk and provide structured metadata.
</instructions>

<output_format>
<metadata>
    <topic>[Main topic/subject]</topic>
    <category>[Document category]</category>
    <keywords>[Comma-separated keywords]</keywords>
    <summary>[Brief summary]</summary>
    <document_type>[Type of document]</document_type>
    <domain>[Domain/field]</domain>
</metadata>
</output_format>
"""
    
    try:
        response = call_llm_service(xml_prompt)
        
        # Parse metadata from XML response
        metadata = {}
        metadata_match = re.search(r'<metadata>(.*?)</metadata>', response, re.DOTALL)
        
        if metadata_match:
            metadata_content = metadata_match.group(1)
            fields = ['topic', 'category', 'keywords', 'summary', 'document_type', 'domain']
            
            for field in fields:
                field_match = re.search(f'<{field}>(.*?)</{field}>', metadata_content, re.DOTALL)
                if field_match:
                    metadata[field] = field_match.group(1).strip()
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {
            'topic': 'General',
            'category': 'Document',
            'keywords': '',
            'summary': chunk_text[:100] + '...',
            'document_type': 'Text',
            'domain': 'General'
        }


def evaluate_chunk_relevance(chunk_text: str, metadata: Dict, queries: List[str]) -> Tuple[bool, float]:
    """
    Evaluate if a chunk is relevant and calculate relevance score
    Returns (is_relevant, relevance_score)
    """
    
    xml_prompt = f"""
<task>
Evaluate the relevance and information value of this text chunk.
</task>

<chunk>
{chunk_text}
</chunk>

<metadata>
{json.dumps(metadata, indent=2)}
</metadata>

<queries>
{json.dumps(queries, indent=2)}
</queries>

<instructions>
1. Determine if this chunk contains substantial, unique information
2. Check if it answers the provided queries meaningfully
3. Assess if it would be valuable in a knowledge base
</instructions>

<output_format>
<evaluation>
    <is_relevant>[true/false]</is_relevant>
    <relevance_score>[0.0 to 1.0]</relevance_score>
    <reasoning>[Brief explanation]</reasoning>
</evaluation>
</output_format>
"""
    
    try:
        response = call_llm_service(xml_prompt)
        
        # Parse evaluation
        eval_match = re.search(r'<evaluation>(.*?)</evaluation>', response, re.DOTALL)
        if eval_match:
            eval_content = eval_match.group(1)
            
            relevant_match = re.search(r'<is_relevant>(.*?)</is_relevant>', eval_content)
            score_match = re.search(r'<relevance_score>(.*?)</relevance_score>', eval_content)
            
            is_relevant = relevant_match.group(1).strip().lower() == 'true' if relevant_match else True
            relevance_score = float(score_match.group(1).strip()) if score_match else 0.5
            
            return is_relevant, relevance_score
    
    except Exception as e:
        logger.error(f"Error evaluating chunk relevance: {e}")
    
    # Default evaluation based on simple heuristics
    word_count = len(chunk_text.split())
    if word_count < 20:
        return False, 0.0
    elif word_count > 50:
        return True, 0.8
    else:
        return True, 0.6


# Additional utility functions
def compute_embeddings(text: str):
    """Compute embeddings for text - implement based on your embedding model"""
    # Replace with your actual embedding computation
    # Example using sentence transformers:
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # return model.encode(text)
    
    # Placeholder implementation
    import numpy as np
    return np.random.random(384)  # Replace with actual embedding


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors"""
    import numpy as np
    norm = np.linalg.norm(embeddings)
    if norm > 0:
        return embeddings / norm
    return embeddings


def load_documents(data_folder: str) -> List[Dict]:
    """Load documents from data folder"""
    import os
    documents = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(('.txt', '.md', '.doc', '.docx', '.pdf')):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append({
                    'filename': filename,
                    'content': content
                })
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    
    return documents