import os
import multimolecule
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

print("--- Début du pré-téléchargement des modèles et tokenizers ---")

# Dictionnaire unique contenant tous les modèles à télécharger
MODELS_TO_DOWNLOAD = {
    "NucTransformer": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "RNAErnie": "multimolecule/rnaernie",
}

for model_key, model_name in MODELS_TO_DOWNLOAD.items():
    print(f"\n--- Téléchargement de {model_key} ({model_name}) ---")
    try:
        # La logique de téléchargement est légèrement différente pour chaque modèle
        if model_key == "NucTransformer":
            # NucTransformer nécessite `trust_remote_code=True`
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"  -> Tokenizer pour {model_key} téléchargé.")
            AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            print(f"  -> Modèle pour {model_key} téléchargé.")

        elif model_key == "RNAErnie":
            # RNAErnie nécessite l'import de `multimolecule` pour s'enregistrer
            AutoTokenizer.from_pretrained(model_name)
            print(f"  -> Tokenizer pour {model_key} téléchargé.")
            AutoModel.from_pretrained(model_name)
            print(f"  -> Modèle pour {model_key} téléchargé.")

        print(f"Téléchargement de {model_key} terminé avec succès.")
    except Exception as e:
        print(f"ERREUR lors du téléchargement de {model_key}: {e}")

print("\n--- Tous les modèles ont été pré-téléchargés et mis en cache. ---")