import base64
import json
import numpy as np
from google.cloud import pubsub_v1

# Configuration
PROJECT_ID = "YOUR-PROJECT"
TOPIC_5S = "M.latent.5s"
TOPIC_15S = "M.latent.15s"

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
topic_path_5s = publisher.topic_path(PROJECT_ID, TOPIC_5S)
topic_path_15s = publisher.topic_path(PROJECT_ID, TOPIC_15S)

def matrix_solver_fn(previous_vector, current_vector):
    """
    Solve for the latent matrix M given two consecutive w-vectors.
    
    This is a simplified implementation. In a real-world scenario,
    this would implement a more sophisticated algorithm.
    
    Args:
        previous_vector: The previous w-vector
        current_vector: The current w-vector
        
    Returns:
        A matrix representing the latent state
    """
    # Convert inputs to numpy arrays for easier manipulation
    prev = np.array(previous_vector)
    curr = np.array(current_vector)
    
    # Simple implementation: create a matrix from the outer product
    # In a real implementation, this would be a more sophisticated algorithm
    matrix = np.outer(prev, curr)
    
    # Apply some normalization or transformation
    # This is just an example - actual implementation would depend on the specific requirements
    matrix = matrix / np.linalg.norm(matrix)
    
    return matrix.tolist()

def process_pubsub_message(event, context):
    """
    Cloud Function entry point for processing Pub/Sub messages.
    
    Args:
        event (dict): The dictionary with data specific to this type of event.
                      The `data` field contains the PubsubMessage message data.
        context (google.cloud.functions.Context): The Cloud Functions event
                                                 metadata.
    Returns:
        None; the output is written to Pub/Sub.
    """
    # Extract the message data
    if 'data' in event:
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        message_data = json.loads(pubsub_message)
        
        # Get the vector from the message
        current_vector = message_data.get('vector')
        
        # Get the previous vector from the state
        # In a real implementation, this would be retrieved from a state store
        # For simplicity, we're using a global variable here
        global previous_vector
        
        if hasattr(process_pubsub_message, 'previous_vector') and process_pubsub_message.previous_vector is not None:
            # Call the solver function
            matrix = matrix_solver_fn(process_pubsub_message.previous_vector, current_vector)
            
            # Prepare the output message
            output_message = {
                'matrix': matrix,
                'timestamp': message_data.get('timestamp')
            }
            
            # Publish the result to the appropriate topic
            # Determine which topic to publish to based on the subscription
            if 'w.rollup.5s' in context.resource.get('name', ''):
                topic_path = topic_path_5s
            else:
                topic_path = topic_path_15s
            
            # Publish the message
            data = json.dumps(output_message).encode('utf-8')
            publisher.publish(topic_path, data=data)
        
        # Update the previous vector for the next invocation
        process_pubsub_message.previous_vector = current_vector
    
    return 'OK'
