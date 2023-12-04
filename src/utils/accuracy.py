import torch
from pytorch_metric_learning.utils import accuracy_calculator

class CustomAccuracyCalculator(accuracy_calculator.AccuracyCalculator):    
    def recall_at_k(self, knn_labels, query_labels, k, **kwargs):
        curr_knn_labels = knn_labels[:, :k]
        recall = 0
        for knn_label, query_label in zip(curr_knn_labels, query_labels):
            if query_label in knn_label[:k]:
                recall += 1
        return recall / (1. * len(curr_knn_labels))
    
    def calculate_recall_at_1(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 1, **kwargs)
    
    def calculate_recall_at_2(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 2, **kwargs)
    
    def calculate_recall_at_4(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 4, **kwargs)
    
    def calculate_recall_at_8(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 8, **kwargs)
    
    def calculate_recall_at_10(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 10, **kwargs)
    
    def calculate_recall_at_100(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels, 100, **kwargs)

    def requires_knn(self):
        return super().requires_knn() + [
            "recall_at_1",
            "recall_at_2",
            "recall_at_4",
            "recall_at_8",
            "recall_at_10",
            "recall_at_100",
        ] 