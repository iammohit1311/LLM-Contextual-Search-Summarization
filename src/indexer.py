import requests
import numpy as np
import faiss
import pickle

from config import FAISS_INDEX_PATH, TOGETHERAI_API_KEY, EMBEDDING_MODE, TOGETHERAI_EMBEDDING_MODEL, SIMILARITY_MODE
from langchain_community.embeddings import HuggingFaceEmbeddings

if EMBEDDING_MODE == "huggingface":
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def generate_embedding(text):
    """Generate embeddings using TogetherAI or Hugging Face."""
    if EMBEDDING_MODE == "huggingface":
        embedding = embedding_model.embed_query(text)
    elif EMBEDDING_MODE == "togetherai":
        headers = {"Authorization": f"Bearer {TOGETHERAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": TOGETHERAI_EMBEDDING_MODEL, "input": text}

        try:
            response = requests.post("https://api.together.xyz/v1/embeddings", json=data, headers=headers)
            response_json = response.json()

            print("🛠️ TogetherAI Embedding API Response:", response_json)

            if "data" in response_json and len(response_json["data"]) > 0:
                embedding = response_json["data"][0]["embedding"]
            elif "error" in response_json:
                print(f"❌ TogetherAI API Error: {response_json['error']['message']}")
                return None
            else:
                print("❌ Unexpected API response format:", response_json)
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling TogetherAI Embedding API: {e}")
            return None

    else:
        raise ValueError("Invalid EMBEDDING_MODE. Choose 'huggingface' or 'togetherai'.")

    if not isinstance(embedding, list):
        print(f"❌ Invalid embedding type: {type(embedding)}. Expected list.")
        return None

    print(f"✅ Embedding shape: {len(embedding)}")

    if SIMILARITY_MODE == "cosine":
        # Normalize embedding (convert to unit vector)
        embedding = np.array(embedding, dtype='float32')
        embedding /= np.linalg.norm(embedding)

        return embedding.tolist()

    return embedding



def create_faiss_index(text_chunks):
    """Generate embeddings and create FAISS index."""
    vectors = [generate_embedding(chunk) for chunk in text_chunks]

    vectors = [v for v in vectors if v is not None]
    if not vectors:
        print("❌ No valid embeddings generated.")
        return None

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors, dtype="float32"))

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump((index, text_chunks), f)

    return index, text_chunks

def create_faiss_index_cosine(text_chunks):
    """Generate embeddings and create FAISS index with cosine similarity."""
    vectors = [generate_embedding(chunk) for chunk in text_chunks]
    vectors = [v for v in vectors if v is not None]

    if not vectors:
        print("❌ No valid embeddings generated.")
        return None

    vectors = np.array(vectors, dtype='float32')

    # Normalize all embeddings to unit vectors (for cosine similarity)
    faiss.normalize_L2(vectors)

    # Use IndexFlatIP for cosine similarity (inner product)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump((index, text_chunks), f)

    print("✅ FAISS index created successfully with cosine similarity.")
    return index, text_chunks
