from sklearn.cluster import KMeans
# from model.sbert import embedder, model
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self):
        self.scoring_model = CrossEncoder('assets/stsb-distilroberta-base', max_length=512)
        self.embedder = SentenceTransformer('assets/all-MiniLM-L6-v2')

    def cluster_pairs(self, model_answer, student_answer):
        """
        Perform K-means clustering for model answer tokens student answer tokens
        instead of computing all possible combinations 
        """
        num_clusters = len(model_answer)

        corpus = model_answer + student_answer
        corpus_embeddings = self.embedder.encode(corpus)

        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        
        clustered_sentences = [[] for i in range(num_clusters)]

        for sentence_id, cluster_id in enumerate(clustering_model.labels_):
            clustered_sentences[cluster_id].append(corpus[sentence_id])
            
        return clustered_sentences

    def predict(self, request):
        ret = self.cluster_pairs(request.model_answer, request.student_answer)
        return ret, list(self.scoring_model.predict(ret))

