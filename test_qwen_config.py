#!/usr/bin/env python3
"""
Quick test of Qwen model and configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("TESTING QWEN 1.5-2B CONFIGURATION")
print("=" * 70)

# Test 1: Import config
print("\n1. Checking configuration...")
from src.config import EMBEDDING_MODEL, VECTOR_DIM, COMPRESSION_RATIO, DEVICE
print(f"   ✅ Model: {EMBEDDING_MODEL}")
print(f"   ✅ Vector Dimension: {VECTOR_DIM}")
print(f"   ✅ Compression Ratio: {COMPRESSION_RATIO}x")
print(f"   ✅ Device: {DEVICE}")

# Test 2: Initialize embedder
print("\n2. Loading Qwen-1.5-2B embedder...")
try:
    from src.embedder import SentinelEmbedder
    embedder = SentinelEmbedder(verbose=True)
    print("   ✅ Embedder loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 3: Test encoding small batch
print("\n3. Testing embedding of sample text...")
try:
    sample_text = [
        "What are the financial implications of market volatility?",
        "Explain risk management strategies for investment portfolios."
    ]
    vectors = embedder.encode(sample_text, batch_size=4)
    print(f"   ✅ Encoded {len(vectors)} texts")
    print(f"   ✅ Vector shape: {vectors.shape}")
    print(f"   ✅ Vector dtype: {vectors.dtype}")
    print(f"   ✅ Vector norm (should be ~1.0): {(vectors**2).sum(axis=1).mean():.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load dataset
print("\n4. Loading FiQA dataset from HuggingFace...")
try:
    from src.dataset import SentinelDatasetManager
    manager = SentinelDatasetManager(verbose=True)
    corpus, queries, qrels = manager.load_smart_subset(target_docs=100)
    print(f"   ✅ Loaded corpus: {len(corpus)} docs")
    print(f"   ✅ Loaded queries: {len(queries)} queries")
    print(f"   ✅ Loaded qrels: {len(qrels)} query-answer pairs")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Ready to run benchmark")
print("=" * 70)
