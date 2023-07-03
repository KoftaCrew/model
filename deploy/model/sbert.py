import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

__version__ = "0.0.3"

class Model:
    def __init__(self):
        self.model = SentenceTransformer('assets/msmarco-distilbert-base-v4')
        self.nlp = spacy.load('en_core_web_sm')

    def _cluster_pairs(self, model_answer, student_answer):
        pairs = []
        for answer in student_answer:
            sim_scores = util.cos_sim(
                self.model.encode(answer), self.model.encode(model_answer)
                )
            cls = np.argmax(sim_scores)
            pairs.append([model_answer[cls], answer])
        return pairs

    def _score_answers(self, pairs, scores_dict, answer_to_id_dict):
        confidence_list = []
        grade_scores = []
        for entry in pairs:
            e1 = self.model.encode(entry[0], convert_to_tensor=True)
            e2 = self.model.encode(entry[1], convert_to_tensor=True)
            sim = util.cos_sim(e1, e2)
            confidence_list.append(sim)
            mapped_id = answer_to_id_dict[entry[0]]
            if sim > 0.85:
                grade_scores.append(scores_dict[mapped_id] * sim)
            else:
                grade_scores.append(0)
        return confidence_list, grade_scores
    
    def segment_text(self, text, threshold=0.5):
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        metric = torch.nn.CosineSimilarity(dim=0)
        embeddings = torch.tensor(self.model.encode(sentences))

        current_embeddings_window = [embeddings[0]]
        last_cluster_end_index = 0
        current_clusters = 0
        sentences_clusters = []

        for i, cur_e in enumerate(embeddings[1:], start=1):
            all_sim_scores = np.mean(
                [metric(cur_e, ei) for ei in current_embeddings_window], axis=0
                )
            if all_sim_scores > threshold:
                current_embeddings_window.append(cur_e)
            else:
                current_clusters += 1
                sentences_clusters.append(sentences[last_cluster_end_index:i])
                current_embeddings_window = [cur_e]
                last_cluster_end_index = i

        def cluster_to_indicies(cluster):
            return [text.index(cluster[0]), text.index(cluster[-1]) + len(cluster[-1])]

        return [cluster_to_indicies(c) for c in sentences_clusters], sentences_clusters

    def predict(self, request):
        grade_dict = {k: v for k, v in zip(request.model_answer_ids, request.max_grades)}
        answer_to_id = {k: v for k, v in zip(request.model_answer, request.model_answer_ids)}

        indicies, seg_text = self.segment_text(request.student_answer)
        clustered_sentences = self._cluster_pairs(request.model_answer, seg_text)

        def transform_pairs_to_ids_and_indicies(pair, i):
            return answer_to_id[pair[0]], indicies[i]
            
        ret_model_answer_ids = []
        ret_segmenetd_student_answer_indicies = []

        for i, e in enumerate(clustered_sentences):
            md, si = transform_pairs_to_ids_and_indicies(e, i)
            ret_model_answer_ids.append(md)
            ret_segmenetd_student_answer_indicies.append(si)

        confidence, grades = self._score_answers(clustered_sentences, grade_dict, answer_to_id)
        return ret_model_answer_ids, ret_segmenetd_student_answer_indicies, confidence, grades
