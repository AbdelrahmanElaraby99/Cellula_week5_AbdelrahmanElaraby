from copy import deepcopy
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import json
from openai import OpenAI
from datasets import load_dataset

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

# Load the HumanEval dataset
dataset = load_dataset("openai/openai_humaneval")

# Extract required fields: task_id, prompt, canonical_solution
processed_data = [
    {
        "task_id": sample["task_id"],
        "prompt": sample["prompt"],
        "canonical_solution": sample["canonical_solution"]
    }
    for sample in dataset["test"]  # The dataset only has a "test" split
]

# save to JSON for later use
with open("humaneval_processed.json", "w") as f:
    json.dump(processed_data, f, indent=2)