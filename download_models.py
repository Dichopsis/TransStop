# download_models.py
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Starting pre-download of models and tokenizers...")

MODELS_TO_DOWNLOAD = {
    "NucTransformer": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
}

for model_key, model_name in MODELS_TO_DOWNLOAD.items():
    print(f"Downloading {model_key} ({model_name})...")
    try:
        # Download and cache the tokenizer
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  -> Tokenizer for {model_key} downloaded.")

        # Download and cache the model
        AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        print(f"  -> Model for {model_key} downloaded.")
        
        print(f"Successfully downloaded {model_key}.")
    except Exception as e:
        print(f"ERROR downloading {model_key}: {e}")

print("\nAll models have been pre-downloaded and cached.")