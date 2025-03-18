import numpy as np
import faiss
import pickle
import os
from src.indexer import generate_embedding
from config import FAISS_INDEX_PATH, SIMILARITY_MODE


def load_faiss_index():
    """Load FAISS index from file."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("❌ FAISS index file not found.")
        return None, None

    try:
        with open(FAISS_INDEX_PATH, "rb") as f:
            index, text_chunks = pickle.load(f)
        print("✅ FAISS index loaded successfully.")
        return index, text_chunks
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None, None


def search_semantically(query, threshold_euclidean=8.0, threshold_cosine=0.2, top_k=5):
    """Perform semantic search in FAISS index."""
    index, text_chunks = load_faiss_index()
    if index is None:
        print("❌ No FAISS index found.")
        return []

    # Generate query embedding
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        print("❌ Query embedding is None. Check TogetherAI response.")
        return []

    query_vector = np.array([query_embedding], dtype="float32")

    # Normalize query vector (crucial for cosine similarity)
    if SIMILARITY_MODE == "cosine":
        faiss.normalize_L2(query_vector)

    print(f"Query vector shape: {query_vector.shape}")
    print(f"FAISS index dimension: {index.d}")

    # if query_vector.shape[1] != index.d:
    #     print(f"❌ Mismatch: Query dim ({query_vector.shape[1]}) vs Index dim ({index.d})")
    #     return []

    # Perform FAISS search
    D, I = index.search(query_vector, top_k)

    print(f"FAISS Search Distances: {D}")
    print(f"FAISS Search Indices: {I}")

    # Process results based on the similarity mode
    if SIMILARITY_MODE == "cosine":
        # Cosine similarity is in [-1,1], higher is better
        results = [
            (text_chunks[i], float(D[0][idx]))  # Directly use FAISS cosine similarity score
            for idx, i in enumerate(I[0])
            if D[0][idx] > threshold_cosine  # Cosine similarity (higher is better)
        ]

    else:
        # Euclidean distance is in [0,∞], lower is better
        results = [
            (text_chunks[i], float(D[0][idx]))
            for idx, i in enumerate(I[0])
            if D[0][idx] < threshold_euclidean
        ]

    print(f"Search Results: {results}")

    return results

