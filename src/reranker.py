"""
Cross-Encoder Reranking Module
Improves retrieval precision by re-scoring candidates using query/document text
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation"""
    doc_id: str
    score: float


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving retrieval precision.
    
    Uses query-document pairs to re-score and reorder candidates from
    an initial retrieval stage. Significantly improves precision@k.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
        verbose: bool = False
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model identifier
            device: torch device ("cuda" or "cpu")
            batch_size: Batch size for reranking
            verbose: Print initialization messages
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        
        if verbose:
            logger.info(f"Loading cross-encoder model: {model_name}")
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            if verbose:
                logger.info(f"✅ Cross-encoder loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder {model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        points: Iterable,
        top_k: int = 10,
        payload_key: str = "text"
    ) -> List[RerankResult]:
        """
        Re-rank candidates using cross-encoder scoring.
        
        Args:
            query: Query text
            points: Qdrant points with payloads
            top_k: Number of candidates to collect for reranking
            payload_key: Key in point.payload for document text
            
        Returns:
            List of RerankResult sorted by score (descending)
        """
        candidates = []
        
        # Collect candidates with their text payloads
        for point in points:
            payload = point.payload or {}
            doc_text = payload.get(payload_key)
            
            if doc_text:
                candidates.append((str(point.id), doc_text))
            
            if len(candidates) >= top_k:
                break
        
        if not candidates:
            if self.verbose:
                logger.warning(f"No candidates with '{payload_key}' payload for reranking")
            return []
        
        # Create query-document pairs
        pairs = [(query, doc_text) for _, doc_text in candidates]
        
        # Score pairs with cross-encoder
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []
        
        # Rank results by score (descending)
        ranked = sorted(
            (
                RerankResult(doc_id=doc_id, score=float(score))
                for (doc_id, _), score in zip(candidates, scores)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        
        return ranked

    def get_model_info(self):
        """Get information about the reranker model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size
        }
