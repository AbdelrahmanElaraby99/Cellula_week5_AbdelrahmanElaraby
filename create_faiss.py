
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import json
from openai import OpenAI
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
import faiss
import numpy as np

################################################## Setup ##################################################
# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")

# Hugging Face client for embeddings
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token,
)

# OpenRouter client for LLM calls
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

embedding_model_id = "BAAI/bge-code-v1"

################################################## Dataset ##################################################
class HFInferenceEmbeddings(Embeddings):
    """Embedding class wrapping Hugging Face Inference API."""

    def __init__(self, model_name: str, api_key: str):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        """Return a list of embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            emb = self.client.feature_extraction(text, model=self.model_name)
            # Flatten if shape = [1, D]
            if isinstance(emb, list) and isinstance(emb[0], list):
                emb = emb[0]
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        """Return a single embedding for a query."""
        emb = self.client.feature_extraction(text, model=self.model_name)
        if isinstance(emb, list) and isinstance(emb[0], list):
            emb = emb[0]
        return emb

# Load precomputed embeddings
with open("humaneval_with_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} documents with embeddings.")

# Prepare data
prompts = [item["prompt"] for item in data]
solutions = [item["prompt"] + "\n" + item['canonical_solution'] for item in data]
metadatas = [item.get("metadata", {}) for item in data]
embeddings = [np.array(item["embeddings"], dtype=np.float32) for item in data]
ids = [str(i) for i in range(len(data))]

# Setup persist directory
persist_directory = "./faiss_humaneval"
os.makedirs(persist_directory, exist_ok=True)

# Initialize embedding function for future queries
embedding_function = HFInferenceEmbeddings(
    model_name=embedding_model_id,
    api_key=hf_token
)

################################################## Build FAISS Index ##################################################
# --- Flatten embeddings ---
flat_embeddings = []
for emb in embeddings:
    emb = np.array(emb, dtype=np.float32)
    # Handle nested embeddings from HF inference (shape [1, D])
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]
    flat_embeddings.append(emb)

flat_embeddings = np.vstack(flat_embeddings).astype(np.float32)
faiss.normalize_L2(flat_embeddings)


# --- Create FAISS store from precomputed embeddings (no re-embedding) ---
vectorstore = FAISS.from_embeddings(
    text_embeddings=list(zip(solutions, flat_embeddings)),
    embedding=embedding_function,  # required for future queries
    metadatas=metadatas,
)

print(f"✅ Successfully built FAISS index with {len(flat_embeddings)} vectors")

# --- Save FAISS index to folder ---
persist_directory = "./faiss_humaneval"
os.makedirs(persist_directory, exist_ok=True)
vectorstore.save_local(persist_directory)
print(f"💾 FAISS index saved to: {persist_directory}")