"""
SENTINEL Dataset Manager
Smart subset loading from FiQA with guaranteed answerable queries
"""

import json
import logging
import os
from typing import Dict, Tuple, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


class SentinelDatasetManager:
    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True, verbose: bool = False):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.verbose = verbose

    def _cache_paths(self, target_docs: Optional[int]) -> Dict[str, str]:
        cache_key = f"fiqa_{target_docs}" if target_docs else "fiqa_all"
        return {
            "corpus": os.path.join(self.cache_dir, f"{cache_key}_corpus.json"),
            "queries": os.path.join(self.cache_dir, f"{cache_key}_queries.json"),
            "qrels": os.path.join(self.cache_dir, f"{cache_key}_qrels.json"),
        }

    def _load_cache(self, paths: Dict[str, str]) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, Dict[str, int]]]:
        with open(paths["corpus"], "r", encoding="utf-8") as f:
            corpus = json.load(f)
        with open(paths["queries"], "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(paths["qrels"], "r", encoding="utf-8") as f:
            qrels = json.load(f)
        return corpus, queries, qrels

    def _save_cache(
        self,
        paths: Dict[str, str],
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
    ) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

        with open(paths["corpus"], "w", encoding="utf-8") as f:
            json.dump(corpus, f)

        with open(paths["queries"], "w", encoding="utf-8") as f:
            json.dump(queries, f)

        with open(paths["qrels"], "w", encoding="utf-8") as f:
            json.dump(qrels, f)

    def load_smart_subset(
        self,
        target_docs: Optional[int],
        loading_method: str = "cached",
        qrels_split: str = "test",   # you can switch to "train" or "dev"
    ) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, Dict[str, int]]]:

        paths = self._cache_paths(target_docs)

        if (
            self.use_cache
            and loading_method == "cached"
            and all(os.path.exists(p) for p in paths.values())
        ):
            if self.verbose:
                print("   ✓ Loading dataset from cache")
            return self._load_cache(paths)

        if self.verbose:
            print("   -> Loading FiQA corpus, queries, and qrels from HuggingFace")

        # Correct HF dataset loading for mteb/fiqa
        corpus_ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
        queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
        qrels_ds = load_dataset("mteb/fiqa", "default", split=qrels_split)

        # Build qrels map and decide which docs to keep
        qrels_map: Dict[str, Dict[str, int]] = {}
        doc_ids_needed = []
        seen_docs = set()

        for row in qrels_ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])

            qrels_map.setdefault(qid, {})[did] = 1

            if did not in seen_docs:
                doc_ids_needed.append(did)
                seen_docs.add(did)

            if target_docs and len(doc_ids_needed) >= target_docs:
                break

        selected_doc_ids = set(doc_ids_needed)

        # Filter corpus to only selected docs
        corpus: Dict[str, Dict] = {}
        for row in corpus_ds:
            did = str(row["_id"])
            if did in selected_doc_ids:
                corpus[did] = {"title": row["title"], "text": row["text"]}
                if len(corpus) >= len(selected_doc_ids):
                    break

        # Filter queries to only those that exist in qrels_map
        queries: Dict[str, str] = {}
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in qrels_map:
                queries[qid] = row["text"]

        # Keep only qrels entries that point to docs we actually loaded
        # (important if target_docs truncates docs mid-way)
        filtered_qrels: Dict[str, Dict[str, int]] = {}
        for qid, rels in qrels_map.items():
            kept = {did: score for did, score in rels.items() if did in corpus}
            if kept:
                filtered_qrels[qid] = kept

        if self.use_cache:
            self._save_cache(paths, corpus, queries, filtered_qrels)

        return corpus, queries, filtered_qrels
