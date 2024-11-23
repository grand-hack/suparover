import vecs
import replicate
import base64
import os
from dotenv import load_dotenv


def initialize_database():
    """
    Initialize the vector store client and documents collection.
    Requires SUPABASE_DB_URI in .env file.

    Returns:
        vecs.Collection: Initialized vector store collection
    """
    # Load environment variables
    load_dotenv()

    # Get database URI from environment variables
    db_uri = os.getenv("SUPABASE_DB_URI")
    if not db_uri:
        raise ValueError("SUPABASE_DB_URI not found in environment variables")

    # Create vector store client
    vx = vecs.create_client(db_uri)

    # Create or get existing collection of vectors with dimension 1024
    docs = vx.get_or_create_collection(name="documents", dimension=1024)

    return docs


def embed_image(image_path):
    """
    Convert an image to an embedding using Replicate's imagebind model.
    Requires REPLICATE_API_TOKEN in .env file.

    Args:
        image_path (str): Path to the image file

    Returns:
        list: Image embedding with dimension 1024

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If REPLICATE_API_TOKEN is not set
    """
    # Check if REPLICATE_API_TOKEN is set
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN not found in environment variables")

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Convert image to base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Get embedding from Replicate
    embedding = replicate.run(
        "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
        input={"input": f"data:image/png;base64,{image_base64}"},
    )

    return embedding


def get_most_similar_record(image_path, threshold=0.8, docs=None):
    """
    Find the most similar record in the database for a given image.

    Args:
        image_path (str): Path to the query image file
        threshold (float, optional): Maximum cosine distance threshold (0 to 2).
                                   Lower values mean more similarity required.
                                   Default is 0.3 (fairly strict matching)
        docs (vecs.Collection, optional): The vector store collection object.
                                        If None, will initialize a new connection.

    Returns:
        tuple: (name, content, distance) of the most similar record, or (None, None, None) if no results
               or if similarity is below threshold

    Raises:
        ValueError: If database connection fails or threshold is invalid
    """
    # Validate threshold
    if not 0 <= threshold <= 2:
        raise ValueError("Threshold must be between 0 and 2 (cosine distance)")

    # Initialize database connection if not provided
    if docs is None:
        try:
            docs = initialize_database()
        except Exception as e:
            raise ValueError(f"Failed to initialize database: {str(e)}")

    # Get embedding for the query image
    query_embedding = embed_image(image_path)

    # Query the collection for the most similar record
    results = docs.query(
        data=query_embedding,
        limit=1,
        include_value=True,  # Include cosine distance in results
        include_metadata=True,
        measure="cosine_distance",
    )

    # Check if we got any results
    if not results:
        return None, None, None

    # Results format is [('id', distance, {'name': 'Name', 'content': 'Content'})]
    record_id, distance, metadata = results[0]

    # Check if the distance is within our threshold
    if distance > threshold:
        return None, None, None

    return metadata["name"], metadata["content"], distance
