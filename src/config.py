"""
SENTINEL Configuration Module
Centralized configuration for IEEE TMLCN Final Benchmark
"""

import os
import torch

# ============================================================================
# CORE SYSTEM PARAMETERS
# ============================================================================

# Target scale for IEEE paper
N_SAMPLES = 100000        # The 100K target for your paper
TARGET_DOCS = 1000        # IEEE final benchmark target
VECTOR_DIM = 384          # all-MiniLM-L6-v2 Dimension (Lightweight, fast inference)
RABITQ_EPSILON = 1.9      # 95% Confidence Bound for Rescoring
COMPRESSION_RATIO = 12.0  # 384 dims × 4 bytes / 128 bytes (12x compression with quantization)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
COLLECTION_NAME = "sentinel_100k_manifold"
GT_COLLECTION = f"{COLLECTION_NAME}_float32"
BQ_COLLECTION = COLLECTION_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINAL_RESULTS_FILE = "final_ieee_data.json"
RECALL_AT_K = 10
CLOUD_LOAD_GBPS = 160.0
SENTINEL_LOAD_GBPS = CLOUD_LOAD_GBPS / COMPRESSION_RATIO
BYTES_PER_FULL_VECTOR = VECTOR_DIM * 4
BYTES_PER_RABITQ_VECTOR = VECTOR_DIM / 8
DEFAULT_PERSONA = "Forensic Auditor"
CONCURRENT_NODES = 10000
EMBEDDING_BATCH_SIZE = 64

# ============================================================================
# CROSS-ENCODER RERANKING (Precision Improvement)
# ============================================================================

ENABLE_RERANKING = bool(os.getenv("SENTINEL_ENABLE_RERANKING", "True").lower() == "true")
RERANK_MODEL = os.getenv("SENTINEL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("SENTINEL_RERANK_TOP_K", "50"))  # Pool of candidates to rerank
RERANK_BATCH_SIZE = int(os.getenv("SENTINEL_RERANK_BATCH_SIZE", "16"))

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Qdrant collection configuration
COLLECTION_NAME = "sentinel_100k_manifold"
QDRANT_TIMEOUT = 30
QDRANT_PREFER_GRPC = False

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# FiQA corpus details
DATASET_NAME = "mteb/fiqa"
CORPUS_SPLIT = "corpus"
QUERIES_SPLIT = "queries"
QRELS_SPLIT = "qrels"

# Field names in FiQA dataset
CORPUS_TITLE_FIELD = "title"
CORPUS_TEXT_FIELD = "text"
QUERY_TEXT_FIELD = "text"

# ============================================================================
# BENCHMARK PARAMETERS
# ============================================================================

# Retrieval metrics
RECALL_AT_K = 10
RETRIEVAL_TOP_K = 20  # Retrieve more than needed for confidence-driven rescoring
CONFIDENCE_THRESHOLD = 0.5  # RaBitQ confidence threshold for rescoring

# Oversampling for fidelity analysis
OVERSAMPLING_FACTORS = [1, 2, 3, 4]  # For fidelity vs compression trade-off

# ============================================================================
# NETWORK LOAD CALCULATION
# ============================================================================

# Per-query network analysis
BYTES_PER_FULL_VECTOR = 1536 * 4  # f32: 6144 bytes per vector
BYTES_PER_RABITQ_VECTOR = 1536 * 0.125  # 1-bit: 192 bytes per vector
BYTES_PER_QUERY_RESULT = 200  # Verdict/answer only

# Baseline cloud-centric load (at 10K concurrent nodes)
CLOUD_LOAD_GBPS = 160.0
SENTINEL_LOAD_GBPS = 5.0

# ============================================================================
# FINANCIAL PERSONAS (Domain Adaptation)
# ============================================================================

FINANCIAL_PERSONAS = {
    "Forensic Auditor": "Specializes in detecting financial irregularities, fraud patterns, and regulatory violations.",
    "Risk Analyst": "Focuses on identifying market risks, credit risks, and operational vulnerabilities.",
    "Compliance Officer": "Ensures adherence to regulations, documentation standards, and audit trails.",
    "Portfolio Manager": "Analyzes investment opportunities, return metrics, and risk-adjusted performance.",
    "CFO": "Evaluates financial health, cash flows, capital allocation, and strategic decisions.",
}

DEFAULT_PERSONA = "Forensic Auditor"

# ============================================================================
# DEVICE & COMPUTE
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32
NUM_WORKERS = 4

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

LOG_LEVEL = "INFO"
VERBOSE = True
SAVE_INTERMEDIATE_RESULTS = True

# Output file names
FINAL_RESULTS_FILE = "final_ieee_data.json"
RESULTS_TABLE_FILE = "SENTINEL_RESULTS_TABLE.md"

# ============================================================================
# ADVANCED CONFIGURATION (Do not modify unless experienced)
# ============================================================================

# Qdrant on-disk storage (critical for 100K scale)
QDRANT_ON_DISK = True
QDRANT_BINARY_QUANTIZATION = True
QDRANT_ALWAYS_RAM = True

# Batch processing
MAX_BATCH_SIZE_ENCODE = 512
MAX_BATCH_SIZE_RETRIEVAL = 32

# RaBitQ implementation
RABITQ_USE_ORTHOGONAL = True  # Use scipy.stats.ortho_group for maximum robustness
RABITQ_NORMALIZE_OUTPUT = True  # L2 normalize after rotation

# ============================================================================
# PRINT CONFIGURATION ON IMPORT (if VERBOSE)
# ============================================================================

VERBOSE = False
if VERBOSE:
    print("=" * 70)
    print("SENTINEL CONFIGURATION LOADED")
    print("=" * 70)
    print(f"Target Scale (Paper): {N_SAMPLES:,} documents")
    print(f"Benchmark Scale: {TARGET_DOCS:,} documents with ground truth")
    print(f"Embedding Model: {EMBEDDING_MODEL} (Lightweight 2B Parameter)")
    print(f"Vector Dimension: {VECTOR_DIM}")
    print(f"Compression Ratio: {COMPRESSION_RATIO}x")
    print(f"Device: {DEVICE}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    print("=" * 70)
