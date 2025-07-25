# download_models.py
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Début du pré-téléchargement des modèles et tokenizers...")

MODELS_TO_DOWNLOAD = {
    "NucTransformer": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
}

for model_key, model_name in MODELS_TO_DOWNLOAD.items():
    print(f"Téléchargement de {model_key} ({model_name})...")
    try:
        # Télécharge et met en cache le tokenizer
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  -> Tokenizer pour {model_key} téléchargé.")

        # Télécharge et met en cache le modèle
        AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        print(f"  -> Modèle pour {model_key} téléchargé.")
        
        print(f"Téléchargement de {model_key} terminé avec succès.")
    except Exception as e:
        print(f"ERREUR lors du téléchargement de {model_key}: {e}")

print("\nTous les modèles ont été pré-téléchargés et mis en cache.")