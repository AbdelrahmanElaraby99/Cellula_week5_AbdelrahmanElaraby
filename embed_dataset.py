from copy import deepcopy
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import json
from openai import OpenAI
from datasets import load_dataset
import numpy as np
import tqdm

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
# Load the HumanEval dataset from the JSON file
with open("humaneval_processed.json", "r") as f:
    processed_data = json.load(f)

# Generate embeddings for each prompt+solution in the dataset, and use tqdm for progress bar
for item in tqdm.tqdm(processed_data):
    prompt_and_solution = item["prompt"] + "\n" + item["canonical_solution"]
    embeddings = hf_client.feature_extraction(prompt_and_solution, model=embedding_model_id)
    item["embeddings"] = np.array(embeddings).reshape(1, -1).tolist()  # Store as list for JSON serialization


# Save the updated data with embeddings to a new JSON file
with open("humaneval_with_embeddings.json", "w") as f:
    json.dump(processed_data, f, indent=2)

