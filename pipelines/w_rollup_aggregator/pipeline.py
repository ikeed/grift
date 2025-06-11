import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.window import FixedWindows
import numpy as np
import json
import base64

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

# Compute elementwise mean of a list of vectors
def average(vectors):
    """
    Compute the elementwise mean of a list of vectors.
    
    Args:
        vectors: List of numeric vectors (each a list of floats)
        
    Returns:
        A single vector representing the elementwise mean
    """
    return np.mean(vectors, axis=0).tolist()

# Encode a vector for Pub/Sub
def encode_vector(vector):
    """
    Encode a vector for publishing to Pub/Sub.
    
    Args:
        vector: A numeric vector (list of floats)
        
    Returns:
        Base64-encoded bytes ready for Pub/Sub
    """
    # Create a message with the vector
    message = {
        'vector': vector,
        'timestamp': beam.window.TimestampedValue.get_timestamp(vector)
    }
    
    # Convert to JSON and encode as bytes
    json_data = json.dumps(message).encode('utf-8')
    
    return json_data

def run(argv=None):
    """Main entry point; defines and runs the pipeline."""
    
    pipeline_options = PipelineOptions(
        argv,
        streaming=True,
        save_main_session=True,
        runner='DataflowRunner',
        project='YOUR-PROJECT',
        region='YOUR-REGION',
        job_name='w-rollup-aggregator',
        temp_location='gs://YOUR-BUCKET/temp',
        staging_location='gs://YOUR-BUCKET/staging'
    )
    
    with beam.Pipeline(options=pipeline_options) as p:
        # Read raw w-vectors from Pub/Sub
        raw = (
            p
            | 'ReadIn' >> beam.io.ReadFromPubSub(
                topic='projects/YOUR-PROJECT/topics/w.decoupled.raw')
            | 'Parse' >> beam.Map(parse_vector)
        )
        
        # 5-second fixed windows → emits at 5, 10, 15, ...
        (raw
            | 'Window5s' >> beam.WindowInto(FixedWindows(5))
            | 'Avg5s' >> beam.CombineGlobally(average).without_defaults()
            | 'Encode5s' >> beam.Map(encode_vector)
            | 'To5sPub' >> beam.io.WriteToPubSub(
                topic='projects/YOUR-PROJECT/topics/w.rollup.5s')
        )
        
        # 15-second fixed windows → emits at 15, 30, ...
        (raw
            | 'Window15s' >> beam.WindowInto(FixedWindows(15))
            | 'Avg15s' >> beam.CombineGlobally(average).without_defaults()
            | 'Encode15s' >> beam.Map(encode_vector)
            | 'To15sPub' >> beam.io.WriteToPubSub(
                topic='projects/YOUR-PROJECT/topics/w.rollup.15s')
        )

if __name__ == '__main__':
    run()
