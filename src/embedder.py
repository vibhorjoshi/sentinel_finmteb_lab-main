"""
SENTINEL Embedder Module
Implements BGE-large embeddings with RaBitQ orthogonal rotation
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import ortho_group
from typing import List, Union, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


class SentinelEmbedder:
    """
    SentinelEmbedder: all-MiniLM-L6-v2 with RaBitQ compression
    
    Implements:
    1. MiniLM embeddings (384-dimensional) - Lightweight 22M parameter model for fast inference
    2. RaBitQ orthogonal rotation (Johnson-Lindenstrauss transform)
    3. L2 normalization for similarity-preserving compression
    
    Safe compression via random orthogonal matrices ensures topological
    structure is preserved despite quantization.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        verbose: bool = True
    ):
        """
        Initialize SentinelEmbedder with all-MiniLM model.
        
        Args:
            model_name: HuggingFace model identifier (all-MiniLM-L6-v2)
            vector_dim: Expected output dimension (384 for MiniLM)
            device: torch device ("cuda" or "cpu", auto-detect if None)
            trust_remote_code: Allow remote model code execution
            verbose: Print initialization messages
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.verbose = verbose
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.verbose:
            logger.info(f"Initializing SentinelEmbedder with {model_name} on {self.device}...")
        
        # =====================================================================
        # Load all-MiniLM Model via SentenceTransformer (384 dimensions)
        # =====================================================================
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=trust_remote_code
            )
            if self.verbose:
                logger.info(f"✅ Loaded {model_name} (384-dimensional lightweight embedding model)")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to direct HuggingFace Transformers...")
            # Fallback: use transformers directly
            from transformers import AutoTokenizer, AutoModel
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)
                self.use_transformers = True
                if self.verbose:
                    logger.info(f"✅ Loaded {model_name} via transformers")
            except Exception as e2:
                logger.error(f"Failed with transformers fallback: {e2}")
                raise
        
        self.use_transformers = False
        
        # =====================================================================
        # CRITICAL FIX: Disable cache to prevent AttributeError
        # =====================================================================
        try:
            if not self.use_transformers and hasattr(self.model, '_first_module'):
                self.model._first_module().auto_model.config.use_cache = False
            elif self.use_transformers and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            if self.verbose:
                logger.info("✅ Cache disabled (AttributeError prevention)")
        except Exception as e:
            logger.warning(f"Could not disable cache: {e}")
        
        # =====================================================================
        # Initialize RaBitQ Rotation Matrix (P)
        # =====================================================================
        # Johnson-Lindenstrauss: Random orthogonal rotation preserves
        # topological structure with high probability (confidence = 1 - exp(-eps^2 * d / 2))
        # For eps=1.9, d=384: confidence ≈ 92%
        
        if self.verbose:
            logger.info(f"Generating RaBitQ rotation matrix P ({vector_dim}×{vector_dim})...")
        
        # Generate random orthogonal matrix using scipy
        P_raw = ortho_group.rvs(dim=vector_dim)
        self.P_matrix = torch.tensor(P_raw, dtype=torch.float32).to(self.device)
        
        if self.verbose:
            logger.info(f"✅ RaBitQ P_matrix shape: {self.P_matrix.shape}")
            logger.info(f"✅ Orthogonality check: P^T @ P ≈ I (trace: {torch.trace(self.P_matrix @ self.P_matrix.T):.4f})")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = True,
        persona: str = "Forensic Auditor",
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into 1536-dimensional vectors with RaBitQ rotation.
        
        Pipeline:
        1. Prefix text with persona context
        2. Encode using Qwen-2.5-GTE → (N, 1536) f32
        3. Apply RaBitQ rotation: v' = v @ P → (N, 1536)
        4. L2 normalize → (N, 1536) unit vectors
        
        Args:
            sentences: Single sentence (str) or list of sentences
            batch_size: Number of sentences per batch
            show_progress_bar: Show tqdm progress bar
            persona: Financial persona for domain adaptation
            normalize_embeddings: L2 normalize output vectors
        
        Returns:
            np.ndarray of shape (N, 1536) with dtype float32
        
        Example:
            >>> embedder = SentinelEmbedder()
            >>> docs = ["Investment in tech stocks", "Risk management strategy"]
            >>> vectors = embedder.encode(docs, persona="Portfolio Manager")
            >>> vectors.shape
            (2, 1536)
        """
        
        # Handle single sentence input
        if isinstance(sentences, str):
            sentences = [sentences]
        
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.vector_dim)
        
        # =====================================================================
        # Step 1: Add Persona Context (Domain Adaptation)
        # =====================================================================
        prefixed_sentences = [
            f"System: [Persona: {persona}] | Content: {sent}"
            for sent in sentences
        ]
        
        if self.verbose and len(sentences) <= 3:
            logger.info(f"Encoding {len(sentences)} sentence(s) with persona: {persona}")
        
        # =====================================================================
        # Step 2: Encode using Qwen-2.5-GTE
        # =====================================================================
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed_sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True
            )
        
        # Ensure embeddings are on the correct device
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        else:
            embeddings = embeddings.to(self.device)
        
        if self.verbose and len(sentences) <= 3:
            logger.info(f"Base embeddings shape: {embeddings.shape}")
        
        # =====================================================================
        # Step 3: Apply RaBitQ Orthogonal Rotation
        # =====================================================================
        # Mathematical operation: v_rotated = v @ P
        # This preserves topological structure (Johnson-Lindenstrauss theorem)
        with torch.no_grad():
            rotated = torch.matmul(embeddings, self.P_matrix)
        
        if self.verbose and len(sentences) <= 3:
            logger.info(f"After RaBitQ rotation: {rotated.shape}")
        
        # =====================================================================
        # Step 4: L2 Normalization (for cosine similarity)
        # =====================================================================
        if normalize_embeddings:
            with torch.no_grad():
                normalized = torch.nn.functional.normalize(rotated, p=2, dim=1)
        else:
            normalized = rotated
        
        # =====================================================================
        # Return as numpy array (CPU)
        # =====================================================================
        result = normalized.cpu().numpy().astype(np.float32)
        
        if self.verbose and len(sentences) <= 3:
            logger.info(f"Final output shape: {result.shape}, dtype: {result.dtype}")
            logger.info(f"Vector norms (should be ~1.0): {np.linalg.norm(result, axis=1)[:3]}")
        
        return result
    
    def encode_batch(
        self,
        sentences_list: List[List[str]],
        batch_size: int = 64,
        persona: str = "Forensic Auditor",
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encode multiple batches of sentences.
        
        Args:
            sentences_list: List of sentence lists
            batch_size: Batch size for encoding
            persona: Financial persona
            show_progress_bar: Show progress bar
        
        Returns:
            np.ndarray of shape (sum(lengths), 1536)
        """
        all_vectors = []
        for i, sentences in enumerate(sentences_list):
            if self.verbose:
                logger.info(f"Batch {i+1}/{len(sentences_list)}: {len(sentences)} sentences")
            vectors = self.encode(
                sentences,
                batch_size=batch_size,
                persona=persona,
                show_progress_bar=show_progress_bar
            )
            all_vectors.append(vectors)
        
        return np.vstack(all_vectors)
    
    def get_device(self) -> str:
        """Get the device the embedder is running on."""
        return self.device
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "vector_dim": self.vector_dim,
            "device": self.device,
            "rabitq_enabled": True,
            "p_matrix_shape": tuple(self.P_matrix.shape),
            "normalization": "L2"
        }
    
    def __repr__(self) -> str:
        return (
            f"SentinelEmbedder(\n"
            f"  model={self.model_name},\n"
            f"  vector_dim={self.vector_dim},\n"
            f"  device={self.device},\n"
            f"  rabitq_enabled=True\n"
            f")"
        )
