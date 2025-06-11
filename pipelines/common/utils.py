import json
import base64
import numpy as np
from typing import List, Dict, Any, Union

def deserialize_from_pubsub(msg_bytes: bytes) -> Dict[str, Any]:
    """
    Deserialize a Pub/Sub message from bytes to a dictionary.
    
    Args:
        msg_bytes: Raw bytes from Pub/Sub
        
    Returns:
        A dictionary containing the deserialized message
    """
    # Decode from base64 (Pub/Sub default encoding)
    decoded = base64.b64decode(msg_bytes)
    
    # Parse JSON
    data = json.loads(decoded)
    
    return data

def serialize_to_pubsub(data: Dict[str, Any]) -> bytes:
    """
    Serialize a dictionary to bytes for Pub/Sub.
    
    Args:
        data: A dictionary to serialize
        
    Returns:
        Base64-encoded bytes ready for Pub/Sub
    """
    # Convert to JSON and encode as bytes
    json_data = json.dumps(data).encode('utf-8')
    
    return json_data

def parse_vector(msg_bytes: bytes) -> List[float]:
    """
    Parse a vector from a Pub/Sub message.
    
    Args:
        msg_bytes: Raw bytes from Pub/Sub
        
    Returns:
        A numeric vector (list of floats)
    """
    # Deserialize the message
    data = deserialize_from_pubsub(msg_bytes)
    
    # Extract vector from the message
    # Assuming the vector is stored under the 'vector' key
    return data.get('vector', [])

def encode_vector(vector: List[float], timestamp: Union[int, float] = None) -> bytes:
    """
    Encode a vector for publishing to Pub/Sub.
    
    Args:
        vector: A numeric vector (list of floats)
        timestamp: Optional timestamp to include in the message
        
    Returns:
        Base64-encoded bytes ready for Pub/Sub
    """
    # Create a message with the vector
    message = {
        'vector': vector
    }
    
    # Add timestamp if provided
    if timestamp is not None:
        message['timestamp'] = timestamp
    
    # Serialize the message
    return serialize_to_pubsub(message)

def parse_matrix(msg_bytes: bytes) -> List[List[float]]:
    """
    Parse a matrix from a Pub/Sub message.
    
    Args:
        msg_bytes: Raw bytes from Pub/Sub
        
    Returns:
        A numeric matrix (list of lists of floats)
    """
    # Deserialize the message
    data = deserialize_from_pubsub(msg_bytes)
    
    # Extract matrix from the message
    # Assuming the matrix is stored under the 'matrix' key
    return data.get('matrix', [])

def encode_matrix(matrix: List[List[float]], timestamp: Union[int, float] = None) -> bytes:
    """
    Encode a matrix for publishing to Pub/Sub.
    
    Args:
        matrix: A numeric matrix (list of lists of floats)
        timestamp: Optional timestamp to include in the message
        
    Returns:
        Base64-encoded bytes ready for Pub/Sub
    """
    # Create a message with the matrix
    message = {
        'matrix': matrix
    }
    
    # Add timestamp if provided
    if timestamp is not None:
        message['timestamp'] = timestamp
    
    # Serialize the message
    return serialize_to_pubsub(message)

def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to have unit length.
    
    Args:
        vector: A numeric vector (list of floats)
        
    Returns:
        A normalized vector
    """
    # Convert to numpy array
    arr = np.array(vector)
    
    # Compute the norm
    norm = np.linalg.norm(arr)
    
    # Normalize the vector
    if norm > 0:
        arr = arr / norm
    
    # Convert back to list
    return arr.tolist()

def normalize_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Normalize a matrix to have unit Frobenius norm.
    
    Args:
        matrix: A numeric matrix (list of lists of floats)
        
    Returns:
        A normalized matrix
    """
    # Convert to numpy array
    arr = np.array(matrix)
    
    # Compute the Frobenius norm
    norm = np.linalg.norm(arr)
    
    # Normalize the matrix
    if norm > 0:
        arr = arr / norm
    
    # Convert back to list of lists
    return arr.tolist()
