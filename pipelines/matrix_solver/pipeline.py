import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import PickleCoder
import json
import base64
import numpy as np

# Deserialize Pub/Sub message to a numeric vector
def parse_vector(msg_bytes):
    """
    Deserialize a Pub/Sub message into a numeric vector.
    
    Args:
        msg_bytes: Raw bytes from Pub/Sub
        
    Returns:
        A numeric vector (list of floats)
    """
    # Decode from base64 (Pub/Sub default encoding)
    decoded = base64.b64decode(msg_bytes)
    
    # Parse JSON
    data = json.loads(decoded)
    
    # Extract vector from the message
    # Assuming the vector is stored under the 'vector' key
    return data['vector']

# Encode a matrix for Pub/Sub
def encode_matrix(matrix):
    """
    Encode a matrix for publishing to Pub/Sub.
    
    Args:
        matrix: A numeric matrix (list of lists of floats)
        
    Returns:
        Base64-encoded bytes ready for Pub/Sub
    """
    # Create a message with the matrix
    message = {
        'matrix': matrix,
        'timestamp': beam.window.TimestampedValue.get_timestamp(matrix)
    }
    
    # Convert to JSON and encode as bytes
    json_data = json.dumps(message).encode('utf-8')
    
    return json_data

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

class PairwiseSolver(beam.DoFn):
    """
    A DoFn that maintains state to process pairs of consecutive vectors.
    """
    # Define a state cell to store the last vector
    LAST = ReadModifyWriteStateSpec('last', PickleCoder())
    
    def process(self, element, state=beam.DoFn.StateParam(LAST)):
        """
        Process a vector, comparing it with the previous vector to generate a matrix.
        
        Args:
            element: A tuple of (key, vector)
            state: The state cell containing the previous vector
            
        Yields:
            A matrix generated from the previous and current vectors
        """
        _, current = element
        previous = state.read()
        
        if previous is not None:
            # Call the solver function
            result = matrix_solver_fn(previous, current)
            yield result
        
        # Update the state with the current vector
        state.write(current)

def run(argv=None):
    """Main entry point; defines and runs the pipeline."""
    
    pipeline_options = PipelineOptions(
        argv,
        streaming=True,
        save_main_session=True,
        runner='DataflowRunner',
        project='YOUR-PROJECT',
        region='YOUR-REGION',
        job_name='matrix-solver',
        temp_location='gs://YOUR-BUCKET/temp',
        staging_location='gs://YOUR-BUCKET/staging'
    )
    
    with beam.Pipeline(options=pipeline_options) as p:
        # Process 5-second rollups
        (p
         | 'Read5s' >> beam.io.ReadFromPubSub(
             topic='projects/YOUR-PROJECT/topics/w.rollup.5s')
         | 'Parse5s' >> beam.Map(parse_vector)
         | 'KeyBy5s' >> beam.Map(lambda v: ('__singleton__', v))
         | 'SolvePairs5s' >> beam.ParDo(PairwiseSolver())
         | 'Encode5s' >> beam.Map(encode_matrix)
         | 'WriteLatent5s' >> beam.io.WriteToPubSub(
             topic='projects/YOUR-PROJECT/topics/M.latent.5s')
        )
        
        # Process 15-second rollups
        (p
         | 'Read15s' >> beam.io.ReadFromPubSub(
             topic='projects/YOUR-PROJECT/topics/w.rollup.15s')
         | 'Parse15s' >> beam.Map(parse_vector)
         | 'KeyBy15s' >> beam.Map(lambda v: ('__singleton__', v))
         | 'SolvePairs15s' >> beam.ParDo(PairwiseSolver())
         | 'Encode15s' >> beam.Map(encode_matrix)
         | 'WriteLatent15s' >> beam.io.WriteToPubSub(
             topic='projects/YOUR-PROJECT/topics/M.latent.15s')
        )

if __name__ == '__main__':
    run()
