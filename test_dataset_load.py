#!/usr/bin/env python3
"""
Test dataset loading before benchmark execution
"""
import sys
sys.path.insert(0, '/workspaces/sentinel_finmteb_lab')

from src.dataset import SentinelDatasetManager
from src.config import TARGET_DOCS

def test_dataset_load():
    """Test dataset loading"""
    print("=" * 80)
    print("TESTING DATASET LOAD")
    print("=" * 80)
    
    try:
        print(f"\n1. Loading FiQA dataset (target: {TARGET_DOCS} documents)...")
        manager = SentinelDatasetManager()
        corpus, queries, qrels = manager.load_smart_subset(target_docs=TARGET_DOCS)
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Corpus size: {len(corpus)} documents")
        print(f"   Queries: {len(queries)} queries")
        print(f"   Qrels entries: {len(qrels)} query-document pairs")
        
        # Validate data structure
        print("\n2. Validating data structure...")
        
        # Check corpus
        if corpus:
            sample_doc_id = list(corpus.keys())[0]
            print(f"   ✅ Corpus structure valid - Sample doc: {sample_doc_id[:20]}...")
            print(f"      Content: {corpus[sample_doc_id][:100]}...")
        
        # Check queries
        if queries:
            sample_query_id = list(queries.keys())[0]
            print(f"   ✅ Queries structure valid - Sample query: {sample_query_id}")
            print(f"      Content: {queries[sample_query_id][:100]}...")
        
        # Check qrels
        if qrels:
            sample_qid = list(qrels.keys())[0]
            print(f"   ✅ Qrels structure valid - Query: {sample_qid}")
            print(f"      Relevant docs: {qrels[sample_qid]}")
        
        # Validate answerable queries
        print("\n3. Validating answerable queries...")
        answerable_count = 0
        total_relevant = 0
        for qid, doc_ids in qrels.items():
            if len(doc_ids) > 0:
                answerable_count += 1
                total_relevant += len(doc_ids)
        
        print(f"   ✅ Answerable queries: {answerable_count}/{len(qrels)}")
        print(f"   ✅ Average relevant docs per query: {total_relevant/len(qrels):.2f}")
        
        # Check if selected queries have relevant docs in corpus
        print("\n4. Validating answer coverage...")
        valid_answers = 0
        for qid, doc_ids in qrels.items():
            for doc_id in doc_ids:
                if doc_id in corpus:
                    valid_answers += 1
        
        print(f"   ✅ Valid answer-document pairs: {valid_answers}/{total_relevant}")
        coverage = (valid_answers / total_relevant * 100) if total_relevant > 0 else 0
        print(f"   ✅ Coverage: {coverage:.1f}%")
        
        print("\n" + "=" * 80)
        print("✅ DATASET LOAD TEST PASSED - READY FOR BENCHMARK")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ DATASET LOAD TEST FAILED:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_load()
    sys.exit(0 if success else 1)
