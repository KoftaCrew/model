from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

scoring_model = CrossEncoder('../assets/cross-encoder/stsb-distilroberta-base', max_length=512)
embedder = SentenceTransformer('../assets/all-MiniLM-L6-v2')
