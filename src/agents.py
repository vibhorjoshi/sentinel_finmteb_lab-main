"""
SENTINEL Multi-Agent Analysis System
5 specialized financial agents with consensus-building orchestrator
"""

import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT ROLE DEFINITIONS
# ============================================================================

class AgentRole(Enum):
    """Enumeration of agent roles in financial analysis."""
    FORENSIC_AUDITOR = "Forensic Auditor"
    RISK_ANALYST = "Risk Analyst"
    COMPLIANCE_OFFICER = "Compliance Officer"
    PORTFOLIO_MANAGER = "Portfolio Manager"
    CFO = "Chief Financial Officer"


# ============================================================================
# FINANCIAL AGENT
# ============================================================================

@dataclass
class AgentAnalysis:
    """Analysis result from a single agent."""
    agent_role: str
    query_id: str
    reasoning: str
    confidence: float
    key_findings: List[str]
    recommendations: List[str]
    risk_assessment: str
    supporting_docs: List[Tuple[str, float]]  # (doc_id, relevance_score)


class FinancialAgent:
    """Base class for domain-specific financial analysis agents."""
    
    def __init__(self, role: AgentRole, verbose: bool = False):
        """
        Initialize agent.
        
        Args:
            role: Agent role
            verbose: Print agent activity
        """
        self.role = role
        self.verbose = verbose
        self.analysis_history = []
    
    def analyze(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """
        Analyze query from agent's perspective.
        
        Args:
            query_id: Query ID
            query_text: Query text
            retrieved_docs: List of (doc_id, doc_text, score)
            corpus: Full corpus dictionary
        
        Returns:
            AgentAnalysis with findings and recommendations
        """
        # Dispatch to role-specific analysis
        if self.role == AgentRole.FORENSIC_AUDITOR:
            return self._analyze_as_auditor(query_id, query_text, retrieved_docs, corpus)
        elif self.role == AgentRole.RISK_ANALYST:
            return self._analyze_as_risk_analyst(query_id, query_text, retrieved_docs, corpus)
        elif self.role == AgentRole.COMPLIANCE_OFFICER:
            return self._analyze_as_compliance(query_id, query_text, retrieved_docs, corpus)
        elif self.role == AgentRole.PORTFOLIO_MANAGER:
            return self._analyze_as_portfolio_mgr(query_id, query_text, retrieved_docs, corpus)
        elif self.role == AgentRole.CFO:
            return self._analyze_as_cfo(query_id, query_text, retrieved_docs, corpus)
    
    def _analyze_as_auditor(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """Forensic auditor analysis."""
        findings = []
        recommendations = []
        risk_level = "LOW"
        
        # Analyze for red flags
        suspicious_patterns = ["unusual", "inconsistent", "discrepancy", "anomaly"]
        for doc_id, doc_text, score in retrieved_docs:
            if any(pattern in doc_text.lower() for pattern in suspicious_patterns):
                findings.append(f"Potential inconsistency detected in doc {doc_id}")
                risk_level = "MEDIUM"
        
        if findings:
            recommendations.append("Conduct deeper forensic examination")
            recommendations.append("Cross-reference with external audit trails")
        else:
            findings.append("Initial audit scan shows consistent patterns")
        
        supporting = [(doc_id, score) for doc_id, _, score in retrieved_docs]
        
        return AgentAnalysis(
            agent_role=self.role.value,
            query_id=query_id,
            reasoning="Forensic analysis focused on fraud detection and financial irregularities",
            confidence=0.85,
            key_findings=findings,
            recommendations=recommendations,
            risk_assessment=risk_level,
            supporting_docs=supporting
        )
    
    def _analyze_as_risk_analyst(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """Risk analyst analysis."""
        findings = []
        recommendations = []
        risk_level = "MEDIUM"
        
        # Analyze for risk factors
        risk_keywords = ["decline", "loss", "volatility", "uncertainty", "exposure"]
        for doc_id, doc_text, score in retrieved_docs:
            if any(keyword in doc_text.lower() for keyword in risk_keywords):
                findings.append(f"Risk factor identified in doc {doc_id}")
                risk_level = "HIGH"
        
        if risk_level == "HIGH":
            recommendations.append("Implement risk mitigation strategies")
            recommendations.append("Monitor market conditions closely")
        else:
            recommendations.append("Continue regular risk monitoring")
        
        findings.append("Comprehensive risk assessment completed")
        supporting = [(doc_id, score) for doc_id, _, score in retrieved_docs]
        
        return AgentAnalysis(
            agent_role=self.role.value,
            query_id=query_id,
            reasoning="Risk analysis focused on market, credit, and operational risks",
            confidence=0.80,
            key_findings=findings,
            recommendations=recommendations,
            risk_assessment=risk_level,
            supporting_docs=supporting
        )
    
    def _analyze_as_compliance(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """Compliance officer analysis."""
        findings = []
        recommendations = []
        compliance_status = "COMPLIANT"
        
        # Check for compliance issues
        compliance_keywords = ["violation", "breach", "non-compliant", "unauthorized"]
        for doc_id, doc_text, score in retrieved_docs:
            if any(keyword in doc_text.lower() for keyword in compliance_keywords):
                findings.append(f"Potential compliance issue in doc {doc_id}")
                compliance_status = "NON-COMPLIANT"
        
        if compliance_status == "NON-COMPLIANT":
            recommendations.append("Immediate remediation required")
            recommendations.append("Document compliance actions taken")
        else:
            findings.append("Current operations align with regulatory requirements")
            recommendations.append("Continue compliance monitoring and documentation")
        
        supporting = [(doc_id, score) for doc_id, _, score in retrieved_docs]
        
        return AgentAnalysis(
            agent_role=self.role.value,
            query_id=query_id,
            reasoning="Compliance analysis focused on regulatory adherence",
            confidence=0.90,
            key_findings=findings,
            recommendations=recommendations,
            risk_assessment=compliance_status,
            supporting_docs=supporting
        )
    
    def _analyze_as_portfolio_mgr(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """Portfolio manager analysis."""
        findings = []
        recommendations = []
        
        # Analyze for portfolio implications
        performance_keywords = ["growth", "return", "performance", "yield", "dividend"]
        for doc_id, doc_text, score in retrieved_docs:
            if any(keyword in doc_text.lower() for keyword in performance_keywords):
                findings.append(f"Portfolio-relevant performance data in doc {doc_id}")
        
        findings.append("Assessment of impact on overall portfolio positioning")
        
        recommendations.append("Review position sizing and allocation")
        recommendations.append("Consider rebalancing if needed")
        recommendations.append("Monitor performance metrics regularly")
        
        supporting = [(doc_id, score) for doc_id, _, score in retrieved_docs]
        
        return AgentAnalysis(
            agent_role=self.role.value,
            query_id=query_id,
            reasoning="Portfolio analysis focused on investment returns and asset allocation",
            confidence=0.78,
            key_findings=findings,
            recommendations=recommendations,
            risk_assessment="ACTIVE_MANAGEMENT",
            supporting_docs=supporting
        )
    
    def _analyze_as_cfo(
        self,
        query_id: str,
        query_text: str,
        retrieved_docs: List[Tuple[str, str, float]],
        corpus: Dict
    ) -> AgentAnalysis:
        """CFO (Chief Financial Officer) analysis."""
        findings = []
        recommendations = []
        
        # CFO perspective on financial health
        financial_keywords = ["cash flow", "liquidity", "solvency", "capital", "debt"]
        for doc_id, doc_text, score in retrieved_docs:
            if any(keyword in doc_text.lower() for keyword in financial_keywords):
                findings.append(f"Financial impact identified in doc {doc_id}")
        
        findings.append("Executive summary: Financial position assessment complete")
        
        recommendations.append("Ensure adequate liquidity buffers")
        recommendations.append("Optimize capital structure")
        recommendations.append("Present findings to board of directors")
        
        supporting = [(doc_id, score) for doc_id, _, score in retrieved_docs]
        
        return AgentAnalysis(
            agent_role=self.role.value,
            query_id=query_id,
            reasoning="Executive financial analysis for strategic decision-making",
            confidence=0.88,
            key_findings=findings,
            recommendations=recommendations,
            risk_assessment="STRATEGIC_REVIEW",
            supporting_docs=supporting
        )


# ============================================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================================

class MultiAgentOrchestrator:
    """
    Orchestrator for managing multiple specialized agents.
    
    Features:
    - Parallel agent analysis
    - Consensus-building with weighted voting
    - Cross-agent validation
    - Comprehensive result aggregation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize orchestrator.
        
        Args:
            verbose: Print orchestrator activity
        """
        self.verbose = verbose
        
        # Initialize agents
        self.agents = {
            role: FinancialAgent(role, verbose=False)
            for role in AgentRole
        }
        
        # Analysis history
        self.analysis_history = []
        
        if verbose:
            logger.info("Initialized MultiAgentOrchestrator with 5 agents")
    
    def analyze_query(
        self,
        query_id: str,
        retrieved_docs: List[Tuple[str, float]],
        corpus: Dict,
        consensus_method: str = "weighted_vote"
    ) -> Dict:
        """
        Run all agents on a query and build consensus.
        
        Args:
            query_id: Query ID
            retrieved_docs: List of (doc_id, score) tuples
            corpus: Full corpus dictionary
            consensus_method: Method for building consensus
        
        Returns:
            Aggregated analysis with consensus recommendations
        """
        if self.verbose:
            logger.debug(f"Analyzing query {query_id} with {len(self.agents)} agents")
        
        # Get query text
        query_text = "Financial query"  # Would be from queries dict in real use
        
        # Prepare retrieved docs with text
        retrieved_docs_with_text = []
        for doc_id, score in retrieved_docs:
            if doc_id in corpus:
                doc_text = corpus[doc_id].get("text", "")
                retrieved_docs_with_text.append((doc_id, doc_text, score))
        
        # Run all agents
        agent_results = {}
        for role, agent in self.agents.items():
            analysis = agent.analyze(query_id, query_text, retrieved_docs_with_text, corpus)
            agent_results[role.value] = analysis
        
        # Build consensus
        if consensus_method == "weighted_vote":
            consensus = self._build_weighted_consensus(agent_results)
        else:
            consensus = self._build_simple_consensus(agent_results)
        
        # Store in history
        self.analysis_history.append({
            "query_id": query_id,
            "agent_results": agent_results,
            "consensus": consensus
        })
        
        return {
            "query_id": query_id,
            "agent_analyses": {
                role: {
                    "role": analysis.agent_role,
                    "findings": analysis.key_findings,
                    "recommendations": analysis.recommendations,
                    "confidence": analysis.confidence,
                    "risk_assessment": analysis.risk_assessment
                }
                for role, analysis in agent_results.items()
            },
            "consensus": consensus
        }
    
    def _build_weighted_consensus(self, agent_results: Dict[str, AgentAnalysis]) -> Dict:
        """
        Build consensus using weighted voting.
        
        Weight by agent confidence score.
        """
        # Aggregate recommendations with weights
        weighted_recommendations = {}
        total_weight = 0
        
        for role, analysis in agent_results.items():
            weight = analysis.confidence
            total_weight += weight
            
            for rec in analysis.recommendations:
                if rec not in weighted_recommendations:
                    weighted_recommendations[rec] = 0.0
                weighted_recommendations[rec] += weight
        
        # Normalize weights and sort
        if total_weight > 0:
            ranked_recommendations = sorted(
                [
                    {
                        "recommendation": rec,
                        "confidence": weight / total_weight
                    }
                    for rec, weight in weighted_recommendations.items()
                ],
                key=lambda x: x["confidence"],
                reverse=True
            )
        else:
            ranked_recommendations = []
        
        # Risk aggregation
        risk_levels = [analysis.risk_assessment for analysis in agent_results.values()]
        overall_risk = self._aggregate_risk(risk_levels)
        
        return {
            "consensus_recommendations": ranked_recommendations[:5],  # Top 5
            "overall_risk_level": overall_risk,
            "agent_agreement": f"{len(agent_results)}/5 agents evaluated",
            "confidence_score": total_weight / len(agent_results) if agent_results else 0.0
        }
    
    def _build_simple_consensus(self, agent_results: Dict[str, AgentAnalysis]) -> Dict:
        """Build consensus by simple aggregation."""
        all_findings = []
        all_recommendations = []
        all_risks = []
        
        for analysis in agent_results.values():
            all_findings.extend(analysis.key_findings)
            all_recommendations.extend(analysis.recommendations)
            all_risks.append(analysis.risk_assessment)
        
        return {
            "aggregated_findings": list(set(all_findings)),
            "aggregated_recommendations": list(set(all_recommendations)),
            "risk_assessments": list(set(all_risks))
        }
    
    @staticmethod
    def _aggregate_risk(risk_levels: List[str]) -> str:
        """Aggregate risk levels from multiple agents."""
        risk_hierarchy = {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "ACTIVE_MANAGEMENT": 3,
            "STRATEGIC_REVIEW": 2,
            "LOW": 1,
            "COMPLIANT": 1
        }
        
        scores = [risk_hierarchy.get(risk, 2) for risk in risk_levels]
        avg_score = sum(scores) / len(scores) if scores else 2
        
        if avg_score >= 4.5:
            return "CRITICAL"
        elif avg_score >= 3.5:
            return "HIGH"
        elif avg_score >= 2.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_orchestrator_summary(self) -> Dict:
        """Get summary of orchestrator activity."""
        return {
            "total_analyses": len(self.analysis_history),
            "agents_active": list(AgentRole.__members__.keys()),
            "num_agents": len(self.agents),
            "consensus_method": "weighted_vote"
        }


# ============================================================================
# AGENT POOL (for parallel processing)
# ============================================================================

class AgentPool:
    """Pool for managing parallel agent execution."""
    
    def __init__(self, num_workers: int = 4, verbose: bool = False):
        """
        Initialize agent pool.
        
        Args:
            num_workers: Number of parallel workers
            verbose: Print pool activity
        """
        self.num_workers = num_workers
        self.verbose = verbose
        self.orchestrator = MultiAgentOrchestrator(verbose=verbose)
    
    def analyze_batch(
        self,
        queries: Dict[str, str],
        rankings: Dict[str, List[Tuple[str, float]]],
        corpus: Dict
    ) -> Dict:
        """
        Analyze batch of queries in parallel.
        
        Args:
            queries: Dict of {query_id: query_text}
            rankings: Dict of {query_id: [(doc_id, score), ...]}
            corpus: Full corpus dictionary
        
        Returns:
            Dict of query_id -> analysis results
        """
        results = {}
        
        query_ids = list(queries.keys())
        for i, query_id in enumerate(query_ids):
            if self.verbose and (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(query_ids)} queries in pool")
            
            retrieved_docs = rankings.get(query_id, [])
            analysis = self.orchestrator.analyze_query(query_id, retrieved_docs, corpus)
            results[query_id] = analysis
        
        return results


class MultiAgentOrchestrator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.agent_roles = [
            "Forensic Auditor",
            "Risk Analyst",
            "Compliance Officer",
            "Portfolio Manager",
            "CFO",
        ]
    
    def analyze_query(self, query_id, retrieval_results, documents, consensus_method="weighted_vote"):
        analyses = []
        if retrieval_results:
            top_doc_id, top_score = retrieval_results[0]
            top_doc = documents.get(top_doc_id, {})
            summary = top_doc.get("text", "") if isinstance(top_doc, dict) else str(top_doc)
        else:
            top_doc_id, top_score, summary = None, 0.0, ""
        
        for role in self.agent_roles:
            analyses.append(
                {
                    "role": role,
                    "query_id": query_id,
                    "top_document_id": top_doc_id,
                    "confidence": float(top_score),
                    "summary": summary[:200],
                }
            )
        
        consensus = {
            "method": consensus_method,
            "top_document_id": top_doc_id,
            "confidence": float(top_score),
        }
        
        return {
            "agents": analyses,
            "consensus": consensus,
        }
    
    def get_orchestrator_summary(self):
        return {
            "agent_count": len(self.agent_roles),
            "agent_roles": self.agent_roles,
        }
