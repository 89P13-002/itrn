import xml.etree.ElementTree as ET
import json
import os
from typing import List, Dict
import re

def parse_llm_response(llm_response: str, output_dir: str = "output") -> List[Dict]:
    """
    Parse LLM response in XML format and extract chunks, metadata, and queries.
    
    Args:
        llm_response (str): The XML response from the LLM
        output_dir (str): Directory to save output files
    
    Returns:
        List[Dict]: List of dictionaries containing chunk data and metadata
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean and prepare the XML response
    cleaned_response = clean_xml_response(llm_response)
    
    try:
        # Parse the XML
        root = ET.fromstring(f"<root>{cleaned_response}</root>")
        chunks_data = []
        all_queries = []
        
        # Extract each chunk
        for chunk_elem in root.findall('chunk'):
            chunk_id = chunk_elem.get('id', 'unknown')
            
            # Extract text content
            text_elem = chunk_elem.find('text')
            chunk_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
            
            # Extract metadata
            metadata_elem = chunk_elem.find('metadata')
            metadata = {}
            
            if metadata_elem is not None:
                for meta_child in metadata_elem:
                    metadata[meta_child.tag] = meta_child.text.strip() if meta_child.text else ""
            
            # Extract queries
            queries_elem = chunk_elem.find('queries')
            chunk_queries = []
            
            if queries_elem is not None:
                for query_elem in queries_elem.findall('query'):
                    query_id = query_elem.get('id', 'unknown')
                    query_text = query_elem.text.strip() if query_elem.text else ""
                    if query_text:
                        query_data = {
                            'id': query_id,
                            'text': query_text,
                            'chunk_id': chunk_id,
                            'metadata': metadata.copy()  # Add metadata to each query
                        }
                        chunk_queries.append(query_data)
                        all_queries.append(query_data)
            
            # Create chunk data structure
            chunk_data = {
                'id': chunk_id,
                'text': chunk_text,
                'metadata': metadata,
                'queries': chunk_queries
            }
            
            chunks_data.append(chunk_data)
        
        # Write chunks and metadata to file
        chunks_file_path = os.path.join(output_dir, "chunks_and_metadata.json")
        with open(chunks_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Write queries with metadata to JSON file
        queries_file_path = os.path.join(output_dir, "queries_with_metadata.json")
        with open(queries_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_queries, f, indent=2, ensure_ascii=False)
        
        # Also write a readable text format
        write_readable_format(chunks_data, all_queries, output_dir)
        
        print(f"Successfully parsed {len(chunks_data)} chunks with {len(all_queries)} total queries")
        print(f"Files saved to: {output_dir}/")
        
        # Return in the requested format: list of dicts with chunk text and metadata
        return_data = []
        for chunk in chunks_data:
            return_data.append({
                'chunk_text': chunk['text'],
                'chunk_metadata': chunk['metadata']
            })
        
        return return_data
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        print("Attempting fallback parsing...")
        return fallback_parse(llm_response, output_dir)
    
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return []

def clean_xml_response(response: str) -> str:
    """
    Clean the LLM response to ensure valid XML format.
    """
    # Remove any text before the first <chunk> tag
    start_match = re.search(r'<chunk', response)
    if start_match:
        response = response[start_match.start():]
    
    # Remove any text after the last </chunk> tag
    end_matches = list(re.finditer(r'</chunk>', response))
    if end_matches:
        last_match = end_matches[-1]
        response = response[:last_match.end()]
    
    # Escape special characters in text content
    response = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', response)
    
    return response

def fallback_parse(response: str, output_dir: str) -> List[Dict]:
    """
    Fallback parser using regex when XML parsing fails.
    """
    chunks_data = []
    all_queries = []
    
    # Find all chunk blocks
    chunk_pattern = r'<chunk id="([^"]*)">(.*?)</chunk>'
    chunk_matches = re.findall(chunk_pattern, response, re.DOTALL)
    
    for chunk_id, chunk_content in chunk_matches:
        # Extract text
        text_match = re.search(r'<text>\s*(.*?)\s*</text>', chunk_content, re.DOTALL)
        chunk_text = text_match.group(1).strip() if text_match else ""
        
        # Extract metadata
        metadata = {}
        metadata_match = re.search(r'<metadata>(.*?)</metadata>', chunk_content, re.DOTALL)
        if metadata_match:
            metadata_content = metadata_match.group(1)
            # Extract individual metadata fields
            for field in ['topic', 'category', 'keywords', 'summary', 'document_type', 'domain']:
                field_match = re.search(f'<{field}>(.*?)</{field}>', metadata_content, re.DOTALL)
                if field_match:
                    metadata[field] = field_match.group(1).strip()
        
        # Extract queries
        chunk_queries = []
        queries_match = re.search(r'<queries>(.*?)</queries>', chunk_content, re.DOTALL)
        if queries_match:
            queries_content = queries_match.group(1)
            query_pattern = r'<query id="([^"]*)">(.*?)</query>'
            query_matches = re.findall(query_pattern, queries_content, re.DOTALL)
            
            for query_id, query_text in query_matches:
                query_data = {
                    'id': query_id,
                    'text': query_text.strip(),
                    'chunk_id': chunk_id,
                    'metadata': metadata.copy()  # Add metadata to each query
                }
                chunk_queries.append(query_data)
                all_queries.append(query_data)
        
        chunk_data = {
            'id': chunk_id,
            'text': chunk_text,
            'metadata': metadata,
            'queries': chunk_queries
        }
        
        chunks_data.append(chunk_data)
    
    # Write files
    if chunks_data:
        chunks_file_path = os.path.join(output_dir, "chunks_and_metadata.json")
        with open(chunks_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        queries_file_path = os.path.join(output_dir, "queries_with_metadata.json")
        with open(queries_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_queries, f, indent=2, ensure_ascii=False)
        
        write_readable_format(chunks_data, all_queries, output_dir)
        
        print(f"Fallback parsing: {len(chunks_data)} chunks with {len(all_queries)} total queries")
    
    # Return in the requested format: list of dicts with chunk text and metadata
    return_data = []
    for chunk in chunks_data:
        return_data.append({
            'chunk_text': chunk['text'],
            'chunk_metadata': chunk['metadata']
        })
    
    return return_data

def write_readable_format(chunks_data: List[Dict], all_queries: List[Dict], output_dir: str):
    """
    Write chunks and queries in a human-readable text format.
    """
    # Write chunks in readable format
    chunks_readable_path = os.path.join(output_dir, "chunks_readable.txt")
    with open(chunks_readable_path, 'w', encoding='utf-8') as f:
        f.write("DOCUMENT CHUNKS AND METADATA\n")
        f.write("=" * 50 + "\n\n")
        
        for chunk in chunks_data:
            f.write(f"CHUNK ID: {chunk['id']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"TEXT:\n{chunk['text']}\n\n")
            
            f.write("METADATA:\n")
            for key, value in chunk['metadata'].items():
                f.write(f"  {key.upper()}: {value}\n")
            
            f.write("\nQUERIES:\n")
            for query in chunk['queries']:
                f.write(f"  - {query['text']}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    # Write queries in readable format
    queries_readable_path = os.path.join(output_dir, "queries_readable.txt")
    with open(queries_readable_path, 'w', encoding='utf-8') as f:
        f.write("ALL QUERIES WITH METADATA\n")
        f.write("=" * 40 + "\n\n")
        
        for i, query in enumerate(all_queries, 1):
            f.write(f"{i}. QUERY: {query['text']}\n")
            f.write(f"   CHUNK ID: {query['chunk_id']}\n")
            f.write("   METADATA:\n")
            for key, value in query.get('metadata', {}).items():
                f.write(f"     {key.upper()}: {value}\n")
            f.write("\n")

# Example usage and testing
def test_parser():
    """
    Test the parser with sample XML response.
    """
    sample_response = '''
    <chunk id="1">
        <text>
        This is the first chunk of content that discusses machine learning fundamentals.
        It covers basic concepts and provides an introduction to the field.
        </text>
        
        <metadata>
            <topic>Machine Learning Fundamentals</topic>
            <category>Technology</category>
            <keywords>machine learning, AI, fundamentals, introduction</keywords>
            <summary>Introduction to machine learning concepts</summary>
            <document_type>Educational</document_type>
            <domain>Computer Science</domain>
        </metadata>
        
        <queries>
            <query id="1">What are the fundamentals of machine learning?</query>
            <query id="2">How is machine learning introduced in this document?</query>
        </queries>
    </chunk>
    
    <chunk id="2">
        <text>
        Advanced machine learning techniques including deep learning and neural networks.
        This section explores more complex algorithms and their applications.
        </text>
        
        <metadata>
            <topic>Advanced Machine Learning</topic>
            <category>Technology</category>
            <keywords>deep learning, neural networks, algorithms, advanced</keywords>
            <summary>Advanced ML techniques and applications</summary>
            <document_type>Educational</document_type>
            <domain>Computer Science</domain>
        </metadata>
        
        <queries>
            <query id="3">What are advanced machine learning techniques?</query>
            <query id="4">How do neural networks work?</query>
            <query id="5">What are the applications of deep learning?</query>
        </queries>
    </chunk>
    '''
    
    result = parse_llm_response(sample_response, "test_output")
    return result

if __name__ == "__main__":
    # Run test
    test_result = test_parser()
    print(f"Test completed. Parsed {len(test_result)} chunks.")
