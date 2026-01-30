"""
SENTINEL Orchestration Pipeline
End-to-end pipeline orchestration for SENTINEL system
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import os

from src.dataset import SentinelDatasetManager
from src.embedder import SentinelEmbedder
from src.engine import SentinelEngine
from src.agents import MultiAgentOrchestrator, AgentPool

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE STAGE DEFINITIONS
# ============================================================================

class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    VECTORIZATION = "vectorization"
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    MULTI_AGENT = "multi_agent_analysis"
    CONSENSUS = "consensus_building"
    EVALUATION = "evaluation"
    EXPORT = "export"


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    stage: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    items_processed: int = 0
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# ORCHESTRATION PIPELINE
# ============================================================================

class SentinelOrchestrationPipeline:
    """
    Complete orchestration pipeline for SENTINEL system.
    
    Workflow:
    1. Data Loading (FiQA corpus)
    2. Preprocessing (filtering, validation)
    3. Vectorization (Qwen-2.5-GTE + RaBitQ)
    4. Indexing (Qdrant with 32x compression)
    5. Retrieval (search for query matches)
    6. Multi-Agent Analysis (5 specialized agents)
    7. Consensus Building (aggregate findings)
    8. Evaluation (calculate metrics)
    9. Export (save results)
    """
    
    def __init__(
        self,
        config_dict: Optional[Dict] = None,
        verbose: bool = True,
        use_cache: bool = True
    ):
        """
        Initialize orchestration pipeline.
        
        Args:
            config_dict: Configuration dictionary
            verbose: Print progress messages
            use_cache: Use disk caching
        """
        self.verbose = verbose
        self.use_cache = use_cache
        self.config = config_dict or self._default_config()
        
        # Stage tracking
        self.stage_metrics: Dict[PipelineStage, StageMetrics] = {}
        self.stage_results: Dict[PipelineStage, Dict] = {}
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        # Components (initialized on-demand)
        self.dataset_manager = None
        self.embedder = None
        self.engine = None
        self.agent_orchestrator = None
        
        if self.verbose:
            logger.info("Initialized SentinelOrchestrationPipeline")
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "dataset": {
                "name": "mteb/fiqa",
                "target_docs": 1000,
                "loading_method": "cached"
            },
            "embedding": {
                "model": "Alibaba-NLP/gte-Qwen2-1.5b-instruct",
                "vector_dim": 1536,
                "batch_size": 64
            },
            "indexing": {
                "collection_name": "sentinel_100k_manifold",
                "compression_ratio": 32.0
            },
            "retrieval": {
                "top_k": 10,
                "oversampling": 1
            },
            "agents": {
                "num_agents": 5,
                "consensus_method": "weighted_vote"
            },
            "evaluation": {
                "metric": "recall_at_k",
                "k": 10
            }
        }
    
    # ========================================================================
    # STAGE 1: DATA LOADING
    # ========================================================================
    
    def stage_data_loading(self) -> Dict:
        """
        Stage 1: Load FiQA dataset with smart subset selection.
        
        Returns:
            Dictionary with corpus, queries, qrels
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 1: DATA LOADING")
            logger.info("="*70)
        
        try:
            # Initialize dataset manager
            self.dataset_manager = SentinelDatasetManager(
                cache_dir="data/cache",
                use_cache=self.use_cache,
                verbose=self.verbose
            )
            
            # Load smart subset
            target_docs = self.config["dataset"]["target_docs"]
            corpus, queries, qrels = self.dataset_manager.load_smart_subset(
                target_docs=target_docs,
                loading_method=self.config["dataset"]["loading_method"]
            )
            
            # Calculate metrics
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.DATA_LOADING.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(corpus) + len(queries),
                success_rate=1.0,
                errors=[],
                metadata={
                    "num_documents": len(corpus),
                    "num_queries": len(queries),
                    "num_qrels": sum(len(v) for v in qrels.values())
                }
            )
            
            self.stage_metrics[PipelineStage.DATA_LOADING] = metrics
            self.stage_results[PipelineStage.DATA_LOADING] = {
                "corpus": corpus,
                "queries": queries,
                "qrels": qrels
            }
            
            if self.verbose:
                logger.info(f"✅ Data loading complete: {len(corpus)} docs, {len(queries)} queries")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return self.stage_results[PipelineStage.DATA_LOADING]
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 2: PREPROCESSING
    # ========================================================================
    
    def stage_preprocessing(self, corpus: Dict, queries: Dict, qrels: Dict) -> Dict:
        """
        Stage 2: Preprocess and validate data.
        
        Args:
            corpus: Corpus dictionary
            queries: Queries dictionary
            qrels: Qrels dictionary
        
        Returns:
            Preprocessed data
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 2: PREPROCESSING")
            logger.info("="*70)
        
        try:
            # Validate data
            validation_errors = []
            
            # Check corpus
            for doc_id, doc in corpus.items():
                if not doc.get("text", ""):
                    validation_errors.append(f"Empty document: {doc_id}")
            
            # Check queries
            for query_id, query_text in queries.items():
                if not query_text:
                    validation_errors.append(f"Empty query: {query_id}")
            
            # Check qrels
            for query_id, rels in qrels.items():
                if query_id not in queries:
                    validation_errors.append(f"Query in qrels not in queries: {query_id}")
                for doc_id in rels.keys():
                    if doc_id not in corpus:
                        validation_errors.append(f"Doc in qrels not in corpus: {doc_id}")
            
            # Clean qrels (remove invalid entries)
            qrels_clean = {}
            for query_id, rels in qrels.items():
                if query_id in queries:
                    clean_rels = {
                        doc_id: score for doc_id, score in rels.items()
                        if doc_id in corpus
                    }
                    if clean_rels:
                        qrels_clean[query_id] = clean_rels
            
            success_rate = 1.0 if not validation_errors else 0.95
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.PREPROCESSING.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(corpus) + len(queries),
                success_rate=success_rate,
                errors=validation_errors,
                metadata={
                    "validation_errors": len(validation_errors),
                    "qrels_after_cleaning": len(qrels_clean)
                }
            )
            
            self.stage_metrics[PipelineStage.PREPROCESSING] = metrics
            
            preprocessed = {
                "corpus": corpus,
                "queries": queries,
                "qrels": qrels_clean
            }
            self.stage_results[PipelineStage.PREPROCESSING] = preprocessed
            
            if self.verbose:
                logger.info(f"✅ Preprocessing complete")
                logger.info(f"   Validation errors: {len(validation_errors)}")
                logger.info(f"   Cleaned qrels: {len(qrels_clean)}")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 3: VECTORIZATION
    # ========================================================================
    
    def stage_vectorization(self, corpus: Dict, queries: Dict) -> Dict:
        """
        Stage 3: Vectorize documents and queries with Qwen-2.5 + RaBitQ.
        
        Args:
            corpus: Corpus dictionary
            queries: Queries dictionary
        
        Returns:
            Dictionary with document and query vectors
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 3: VECTORIZATION")
            logger.info("="*70)
        
        try:
            # Initialize embedder
            self.embedder = SentinelEmbedder(
                verbose=self.verbose
            )
            
            # Vectorize corpus
            if self.verbose:
                logger.info(f"Vectorizing {len(corpus)} documents...")
            
            doc_ids = list(corpus.keys())
            doc_texts = [f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}".strip() 
                         for doc_id in doc_ids]
            
            doc_vectors = self.embedder.encode(
                doc_texts,
                batch_size=self.config["embedding"]["batch_size"],
                show_progress_bar=self.verbose
            )
            
            # Vectorize queries
            if self.verbose:
                logger.info(f"Vectorizing {len(queries)} queries...")
            
            query_ids = list(queries.keys())
            query_texts = [queries[qid] for qid in query_ids]
            
            query_vectors = self.embedder.encode(
                query_texts,
                batch_size=self.config["embedding"]["batch_size"],
                show_progress_bar=self.verbose
            )
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.VECTORIZATION.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(doc_vectors) + len(query_vectors),
                success_rate=1.0,
                errors=[],
                metadata={
                    "doc_vectors_shape": str(doc_vectors.shape),
                    "query_vectors_shape": str(query_vectors.shape),
                    "time_per_doc": (end_time - start_time) / len(doc_vectors)
                }
            )
            
            self.stage_metrics[PipelineStage.VECTORIZATION] = metrics
            
            vectorized = {
                "doc_ids": doc_ids,
                "doc_vectors": doc_vectors,
                "query_ids": query_ids,
                "query_vectors": query_vectors
            }
            self.stage_results[PipelineStage.VECTORIZATION] = vectorized
            
            if self.verbose:
                logger.info(f"✅ Vectorization complete")
                logger.info(f"   Doc vectors: {doc_vectors.shape}")
                logger.info(f"   Query vectors: {query_vectors.shape}")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return vectorized
            
        except Exception as e:
            logger.error(f"Vectorization failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 4: INDEXING
    # ========================================================================
    
    def stage_indexing(self, doc_ids: List[str], doc_vectors: List) -> None:
        """
        Stage 4: Build Qdrant index with binary quantization (32x compression).
        
        Args:
            doc_ids: List of document IDs
            doc_vectors: Document vectors
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 4: INDEXING")
            logger.info("="*70)
        
        try:
            # Initialize engine
            self.engine = SentinelEngine(
                data_path="data/qdrant_storage",
                collection_name=self.config["indexing"]["collection_name"],
                verbose=self.verbose
            )
            
            # Create collection
            self.engine.init_collection()
            
            # Upsert vectors
            point_ids = [int(doc_id) for doc_id in doc_ids]
            self.engine.upsert_vectors(doc_vectors, point_ids, batch_size=128)
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.INDEXING.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(doc_vectors),
                success_rate=1.0,
                errors=[],
                metadata={
                    "num_vectors": len(doc_vectors),
                    "compression_ratio": self.config["indexing"]["compression_ratio"],
                    "memory_usage_mb": len(doc_vectors) * 192 / (1024 * 1024)
                }
            )
            
            self.stage_metrics[PipelineStage.INDEXING] = metrics
            
            if self.verbose:
                logger.info(f"✅ Indexing complete")
                logger.info(f"   Vectors indexed: {len(doc_vectors)}")
                logger.info(f"   Compression: {self.config['indexing']['compression_ratio']}x")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 5: RETRIEVAL
    # ========================================================================
    
    def stage_retrieval(
        self,
        query_ids: List[str],
        query_vectors: List,
        corpus: Dict
    ) -> Dict:
        """
        Stage 5: Retrieve top-k documents for each query.
        
        Args:
            query_ids: Query IDs
            query_vectors: Query vectors
            corpus: Corpus dictionary
        
        Returns:
            Dictionary with retrieval results
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 5: RETRIEVAL")
            logger.info("="*70)
        
        try:
            top_k = self.config["retrieval"]["top_k"]
            oversampling = self.config["retrieval"]["oversampling"]
            
            if self.verbose:
                logger.info(f"Retrieving top-{top_k} documents for {len(query_ids)} queries...")
            
            retrieval_results = {}
            errors = []
            
            for i, (query_id, query_vector) in enumerate(zip(query_ids, query_vectors)):
                try:
                    # Search
                    search_k = top_k * oversampling
                    results = self.engine.search(query_vector, top_k=search_k)
                    
                    # Filter to top_k
                    retrieval_results[query_id] = results[:top_k]
                    
                    if self.verbose and (i + 1) % 50 == 0:
                        logger.info(f"   Processed {i + 1}/{len(query_ids)} queries")
                
                except Exception as e:
                    errors.append(f"Query {query_id}: {str(e)}")
            
            success_rate = (len(retrieval_results) / len(query_ids)) if query_ids else 0.0
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.RETRIEVAL.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(retrieval_results),
                success_rate=success_rate,
                errors=errors,
                metadata={
                    "queries_processed": len(retrieval_results),
                    "avg_results_per_query": top_k
                }
            )
            
            self.stage_metrics[PipelineStage.RETRIEVAL] = metrics
            
            retrieval_dict = {
                "query_results": retrieval_results,
                "top_k": top_k
            }
            self.stage_results[PipelineStage.RETRIEVAL] = retrieval_dict
            
            if self.verbose:
                logger.info(f"✅ Retrieval complete")
                logger.info(f"   Queries processed: {len(retrieval_results)}")
                logger.info(f"   Success rate: {success_rate*100:.1f}%")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return retrieval_dict
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 6: MULTI-AGENT ANALYSIS
    # ========================================================================
    
    def stage_multi_agent_analysis(
        self,
        query_ids: List[str],
        retrieval_results: Dict,
        corpus: Dict
    ) -> Dict:
        """
        Stage 6: Run multi-agent analysis on retrieval results.
        
        Args:
            query_ids: List of query IDs
            retrieval_results: Dict of retrieval results per query
            corpus: Corpus dictionary
        
        Returns:
            Multi-agent analysis results
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 6: MULTI-AGENT ANALYSIS")
            logger.info("="*70)
        
        try:
            # Initialize orchestrator
            self.agent_orchestrator = MultiAgentOrchestrator(verbose=self.verbose)
            
            # Analyze each query with all agents
            agent_analyses = {}
            errors = []
            
            for i, query_id in enumerate(query_ids):
                try:
                    results = retrieval_results.get(query_id, [])
                    analysis = self.agent_orchestrator.analyze_query(
                        query_id,
                        results,
                        corpus,
                        consensus_method=self.config["agents"]["consensus_method"]
                    )
                    agent_analyses[query_id] = analysis
                    
                    if self.verbose and (i + 1) % 50 == 0:
                        logger.info(f"   Analyzed {i + 1}/{len(query_ids)} queries")
                
                except Exception as e:
                    errors.append(f"Query {query_id}: {str(e)}")
            
            success_rate = (len(agent_analyses) / len(query_ids)) if query_ids else 0.0
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.MULTI_AGENT.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(agent_analyses),
                success_rate=success_rate,
                errors=errors,
                metadata={
                    "queries_analyzed": len(agent_analyses),
                    "num_agents": 5
                }
            )
            
            self.stage_metrics[PipelineStage.MULTI_AGENT] = metrics
            
            agent_results = {
                "analyses": agent_analyses,
                "orchestrator_summary": self.agent_orchestrator.get_orchestrator_summary()
            }
            self.stage_results[PipelineStage.MULTI_AGENT] = agent_results
            
            if self.verbose:
                logger.info(f"✅ Multi-agent analysis complete")
                logger.info(f"   Queries analyzed: {len(agent_analyses)}")
                logger.info(f"   Success rate: {success_rate*100:.1f}%")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return agent_results
            
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 7: EVALUATION
    # ========================================================================
    
    def stage_evaluation(
        self,
        query_ids: List[str],
        retrieval_results: Dict,
        qrels: Dict
    ) -> Dict:
        """
        Stage 7: Evaluate retrieval quality using ground truth.
        
        Args:
            query_ids: List of query IDs
            retrieval_results: Dict of retrieval results
            qrels: Ground truth relevance dictionary
        
        Returns:
            Evaluation metrics
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 7: EVALUATION")
            logger.info("="*70)
        
        try:
            from src.metrics import RecallCalculator
            
            calculator = RecallCalculator()
            
            recalls = []
            for query_id in query_ids:
                if query_id not in qrels:
                    continue
                
                relevant_docs = set(qrels[query_id].keys())
                retrieved = set(str(doc_id) for doc_id, _ in retrieval_results.get(query_id, []))
                
                recall = calculator.recall_at_k(relevant_docs, retrieved, k=len(retrieved))
                recalls.append(recall)
            
            avg_recall = np.mean(recalls) if recalls else 0.0
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.EVALUATION.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=len(recalls),
                success_rate=1.0,
                errors=[],
                metadata={
                    "avg_recall": float(avg_recall),
                    "num_evaluated": len(recalls)
                }
            )
            
            self.stage_metrics[PipelineStage.EVALUATION] = metrics
            
            eval_results = {
                "avg_recall": float(avg_recall),
                "recalls": [float(r) for r in recalls],
                "num_queries_evaluated": len(recalls)
            }
            self.stage_results[PipelineStage.EVALUATION] = eval_results
            
            if self.verbose:
                logger.info(f"✅ Evaluation complete")
                logger.info(f"   Recall@10: {avg_recall:.4f}")
                logger.info(f"   Queries evaluated: {len(recalls)}")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # STAGE 9: EXPORT
    # ========================================================================
    
    def stage_export(self, output_file: str = "results/final_ieee_data.json") -> None:
        """
        Stage 9: Export results to JSON.
        
        Args:
            output_file: Output file path
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("STAGE 9: EXPORT")
            logger.info("="*70)
        
        try:
            # Prepare results
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_duration": self.pipeline_end_time - self.pipeline_start_time if self.pipeline_end_time else 0,
                "stage_metrics": {
                    stage.value: {
                        "duration": metrics.duration,
                        "items_processed": metrics.items_processed,
                        "success_rate": metrics.success_rate,
                        "errors": metrics.errors
                    }
                    for stage, metrics in self.stage_metrics.items()
                },
                "stage_results": {
                    stage.value: results
                    for stage, results in self.stage_results.items()
                    if stage != PipelineStage.DATA_LOADING  # Too large to export
                }
            }
            
            # Save to file
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            end_time = time.time()
            metrics = StageMetrics(
                stage=PipelineStage.EXPORT.value,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                items_processed=1,
                success_rate=1.0,
                errors=[],
                metadata={"output_file": output_file}
            )
            
            self.stage_metrics[PipelineStage.EXPORT] = metrics
            
            if self.verbose:
                logger.info(f"✅ Export complete")
                logger.info(f"   File: {output_file}")
                logger.info(f"   Time: {metrics.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # RUN COMPLETE PIPELINE
    # ========================================================================
    
    def run(self) -> Dict:
        """
        Run complete pipeline from data loading to export.
        
        Returns:
            Final results dictionary
        """
        self.pipeline_start_time = time.time()
        
        if self.verbose:
            logger.info("\n\n")
            logger.info("="*70)
            logger.info("SENTINEL ORCHESTRATION PIPELINE - STARTING")
            logger.info("="*70)
        
        try:
            # Stage 1: Data Loading
            data = self.stage_data_loading()
            corpus, queries, qrels = data["corpus"], data["queries"], data["qrels"]
            
            # Stage 2: Preprocessing
            preprocessed = self.stage_preprocessing(corpus, queries, qrels)
            corpus, queries, qrels = (
                preprocessed["corpus"],
                preprocessed["queries"],
                preprocessed["qrels"]
            )
            
            # Stage 3: Vectorization
            vectorized = self.stage_vectorization(corpus, queries)
            doc_ids, doc_vectors = vectorized["doc_ids"], vectorized["doc_vectors"]
            query_ids, query_vectors = vectorized["query_ids"], vectorized["query_vectors"]
            
            # Stage 4: Indexing
            self.stage_indexing(doc_ids, doc_vectors)
            
            # Stage 5: Retrieval
            retrieval_dict = self.stage_retrieval(query_ids, query_vectors, corpus)
            retrieval_results = retrieval_dict["query_results"]
            
            # Stage 6: Multi-Agent Analysis
            agent_results = self.stage_multi_agent_analysis(query_ids, retrieval_results, corpus)
            
            # Stage 7: Evaluation
            eval_results = self.stage_evaluation(query_ids, retrieval_results, qrels)
            
            # Stage 9: Export
            self.stage_export()
            
            self.pipeline_end_time = time.time()
            
            if self.verbose:
                logger.info("\n" + "="*70)
                logger.info("🌟 PIPELINE COMPLETE")
                logger.info("="*70)
                logger.info(f"Total time: {self.pipeline_end_time - self.pipeline_start_time:.1f}s")
            
            return {
                "success": True,
                "evaluation": eval_results,
                "agents": agent_results,
                "pipeline_duration": self.pipeline_end_time - self.pipeline_start_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.pipeline_end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "pipeline_duration": self.pipeline_end_time - self.pipeline_start_time
            }
    
    def get_pipeline_report(self) -> str:
        """Get formatted pipeline execution report."""
        report = "\n" + "="*70 + "\n"
        report += "SENTINEL PIPELINE EXECUTION REPORT\n"
        report += "="*70 + "\n\n"
        
        for stage, metrics in sorted(self.stage_metrics.items(), key=lambda x: x[1].start_time):
            report += f"{stage.value.upper()}\n"
            report += f"  Duration: {metrics.duration:.2f}s\n"
            report += f"  Items: {metrics.items_processed}\n"
            report += f"  Success rate: {metrics.success_rate*100:.1f}%\n"
            report += f"  Errors: {len(metrics.errors)}\n\n"
        
        if self.pipeline_end_time and self.pipeline_start_time:
            total = self.pipeline_end_time - self.pipeline_start_time
            report += f"TOTAL PIPELINE TIME: {total:.2f}s\n"
        
        report += "="*70 + "\n"
        
        return report
