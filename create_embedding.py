from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

# Step 1: Preprocess the sentences (e.g., lowercasing, removing punctuation)
def preprocess(sentences):
    import re
    return [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]

# Step 2: Convert sentences to embeddings using a pre-trained model
def sentences_to_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any other model from SentenceTransformers
    embeddings = model.encode(sentences)
    return embeddings

# Step 3: Connect to Milvus instance
def connect_milvus(host='localhost', port='19530'):
    connections.connect("default", host=host, port=port)

# Step 4: Create a collection in Milvus and store the embeddings
def create_and_store_embeddings(collection_name, embeddings):
    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
    ]

    # Create a schema
    schema = CollectionSchema(fields, "Sentence Embeddings Collection")

    # Create the collection
    collection = Collection(name=collection_name, schema=schema)

    # Insert embeddings
    data = [embeddings.tolist()]  # embeddings need to be a list of lists
    collection.insert(data)

if __name__ == "__main__":
    sentences = [
        "This is the first sentence.",
        "Here is the second one.",
        "This makes it three.",
        # ... add up to 20 sentences
    ]

    # Preprocess the sentences
    processed_sentences = preprocess(sentences)

    # Convert sentences to embeddings
    embeddings = sentences_to_embeddings(processed_sentences)

    # Connect to Milvus
    connect_milvus()

    # Create a collection and store embeddings
    collection_name = "sentence_embeddings"
    create_and_store_embeddings(collection_name, embeddings)

    print("Embeddings have been successfully stored in Milvus.")
