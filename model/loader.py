import pickle

def load_ngram_model(file_path):
    with open(file_path, 'rb') as file:
        ngram_probabilities = pickle.load(file)
    print(f"Modèle chargé depuis {file_path}")
    return ngram_probabilities
