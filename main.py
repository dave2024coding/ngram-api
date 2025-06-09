from fastapi import FastAPI, Query
from pydantic import BaseModel
import re
from model.loader import load_ngram_model

# Initialisation de l'API
app = FastAPI(title="API de Prédiction N-gram")

# Charger le modèle à l'initialisation
ngram_model_path = 'trigram_model.pkl'
n = 3  # Modifier selon le modèle
ngram_probabilities = load_ngram_model(ngram_model_path)

# Modèle de requête
class PredictionRequest(BaseModel):
    phrase: str
    top_k: int = 3

# Fonction de prédiction
def predict_next_word(phrase, n, ngram_probabilities, top_k=3):
    phrase = re.sub(r'[^\w\s]', '', phrase).lower()
    words = phrase.split()
    words = ['<s>'] * (n - 1) + words
    prefix_tuple = tuple(words[-(n - 1):])
    
    if prefix_tuple in ngram_probabilities:
        sorted_predictions = sorted(
            ngram_probabilities[prefix_tuple].items(),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_predictions[:top_k]
    else:
        return None

# Endpoint d'API
@app.post("/predict")
def get_prediction(request: PredictionRequest):
    result = predict_next_word(request.phrase, n, ngram_probabilities, top_k=request.top_k)
    
    if result:
        return {"predictions": [{"word": word, "probability": prob} for word, prob in result]}
    else:
        return {"message": "Aucune prédiction disponible pour la phrase donnée."}
