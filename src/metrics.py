"""
SENTINEL Metrics Module
Comprehensive evaluation metrics
"""

import numpy as np


class ComprehensiveEvaluator:
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def _precision_at_k(self, relevant, retrieved, k):
        if k == 0:
            return 0.0
        return len(relevant.intersection(retrieved[:k])) / k
    
    def _recall_at_k(self, relevant, retrieved, k):
        if not relevant:
            return 0.0
        return len(relevant.intersection(retrieved[:k])) / len(relevant)
    
    def _average_precision(self, relevant, retrieved):
        if not relevant:
            return 0.0
        hits = 0
        precision_sum = 0.0
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / idx
        return precision_sum / len(relevant)
    
    def _ndcg_at_k(self, relevant, retrieved, k):
        if not relevant:
            return 0.0
        dcg = 0.0
        for idx, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(idx + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, qrels, results, k_values):
        metrics = {
            "map": {"mean": 0.0},
        }
        for k in k_values:
            metrics[f"recall@{k}"] = {"mean": 0.0}
            metrics[f"precision@{k}"] = {"mean": 0.0}
            metrics[f"ndcg@{k}"] = {"mean": 0.0}
        
        query_ids = [qid for qid in results.keys() if qid in qrels]
        if not query_ids:
            return metrics
        
        map_scores = []
        for qid in query_ids:
            relevant = qrels[qid]
            retrieved = results.get(qid, [])
            for k in k_values:
                metrics[f"recall@{k}"]["mean"] += self._recall_at_k(relevant, retrieved, k)
                metrics[f"precision@{k}"]["mean"] += self._precision_at_k(relevant, retrieved, k)
                metrics[f"ndcg@{k}"]["mean"] += self._ndcg_at_k(relevant, retrieved, k)
            map_scores.append(self._average_precision(relevant, retrieved))
        
        n_queries = len(query_ids)
        for k in k_values:
            metrics[f"recall@{k}"]["mean"] /= n_queries
            metrics[f"precision@{k}"]["mean"] /= n_queries
            metrics[f"ndcg@{k}"]["mean"] /= n_queries
        
        metrics["map"]["mean"] = float(np.mean(map_scores)) if map_scores else 0.0
        return metrics


class RecallCalculator:
    @staticmethod
    def recall_at_k(relevant, retrieved, k):
        if not relevant:
            return 0.0
        return len(set(retrieved[:k]).intersection(relevant)) / len(relevant)


def calculate_network_impact(n_queries, k=10, vec_dim=1024):
    """
    Calculates the Backhaul Traffic Reduction for IEEE TMLCN.
    
    Compares:
    - Cloud baseline: Query vector (f32, 4096 bytes) + Top-K vectors (f32, 4096*K)
    - SENTINEL edge: Query text (500 bytes) + Top-K IDs (8*K)
    
    Args:
        n_queries: Number of queries
        k: Top-K results returned
        vec_dim: Vector dimensionality (1024 or 1536)
    
    Returns:
        Dictionary with network metrics
    """
    FLOAT32_BYTES = 4
    TEXT_BYTES = 500
    ID_BYTES = 8
    
    cloud_bytes = n_queries * (vec_dim * FLOAT32_BYTES + k * vec_dim * FLOAT32_BYTES)
    edge_bytes = n_queries * (TEXT_BYTES + k * ID_BYTES)
    
    cloud_gbps = (cloud_bytes * 8) / (1024**3)
    edge_gbps = (edge_bytes * 8) / (1024**3)
    
    return {
        "cloud_baseline_gbps": cloud_gbps,
        "sentinel_edge_gbps": edge_gbps,
        "bandwidth_saved_gbps": cloud_gbps - edge_gbps,
        "bandwidth_saved_percent": (1 - (edge_gbps / cloud_gbps)) * 100
    }


def calculate_network_load(n_queries, k=10, vec_dim=1536):
    FLOAT32_BYTES = 4
    TEXT_BYTES = 500
    cloud_bytes = n_queries * (vec_dim * FLOAT32_BYTES + k * vec_dim * FLOAT32_BYTES)
    edge_bytes = n_queries * TEXT_BYTES
    cloud_gbps = (cloud_bytes * 8) / (1024**3)
    edge_gbps = (edge_bytes * 8) / (1024**3)
    return cloud_gbps, edge_gbps


def compute_topological_integrity(qrels, query_id, retrieved_ids):
    """
    Computes recall for a single query given qrels and retrieved IDs.
    
    Args:
        qrels: {query_id: {doc_id: score}}
        query_id: Query ID
        retrieved_ids: List of retrieved doc IDs in rank order
    
    Returns:
        Tuple of (recall, is_hit)
    """
    if query_id not in qrels:
        return 0.0, False
    
    relevant = set(qrels[query_id].keys())
    retrieved = set(retrieved_ids[:10])
    
    recall = len(relevant & retrieved) / len(relevant) if relevant else 0.0
    is_hit = len(relevant & retrieved) > 0
    
    return recall, is_hit


def compute_fidelity_with_qrels(qrels, query_id, retrieved_ids):
    return compute_topological_integrity(qrels, query_id, retrieved_ids)
