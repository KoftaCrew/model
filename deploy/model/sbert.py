from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

__version__ = "0.0.2"

class Model:
    def __init__(self):
        self.model = SentenceTransformer('assets/msmarco-distilbert-base-v4')

    def _cluster_pairs(self, model_answer, student_answer):
        """
        Perform K-means clustering for model answer tokens student answer tokens
        instead of computing all possible combinations 
        """
        num_clusters = len(model_answer)

        corpus = model_answer + student_answer
        corpus_embeddings = self.model.encode(corpus)

        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)

        clustered_sentences = [[] for i in range(num_clusters)]

        for sentence_id, cluster_id in enumerate(clustering_model.labels_):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        return clustered_sentences

    def _score_answers(self, pairs):
        scores = []
        for entry in pairs:
            e1 = self.model.encode(entry[0], convert_to_tensor=True)
            e2 = self.model.encode(entry[1], convert_to_tensor=True)
            scores.append(util.cos_sim(e1, e2))
        return scores

    def predict(self, request):
        ret = self._cluster_pairs(request.model_answer, request.student_answer)
        return ret, self._score_answers(ret)
