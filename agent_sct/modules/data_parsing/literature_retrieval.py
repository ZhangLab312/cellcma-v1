"""
Literature Retrieval Module

Refactored from the existing PubMedSearcher, provides a unified literature retrieval interface.
Supports PubMed and web search for obtaining the latest bioinformatics methods.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class LiteratureRecord:
    """Literature record"""
    title: str
    authors: List[str]
    journal: str
    year: int
    abstract: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: str = "pubmed"  # pubmed, web, etc.
    relevance_score: float = 0.0
    
    def to_text(self) -> str:
        lines = [
            f"Title: {self.title}",
            f"Journal: {self.journal} ({self.year})",
        ]
        if self.abstract:
            lines.append(f"Abstract: {self.abstract[:500]}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'abstract': self.abstract,
            'pmid': self.pmid,
            'doi': self.doi,
            'url': self.url,
            'source': self.source,
            'relevance_score': self.relevance_score,
        }


class LiteratureRetriever:
    """
    Literature Retriever

    Integrates PubMed and web search to provide the latest domain knowledge for the Agent.
    """

    # Class-level retrieval keyword mapping
    RETRIEVAL_KEYWORDS_MAP = {
        "integration": [
            "single-cell multi-omics integration",
            "co-embedding",
            "cross-modality single-cell",
            "multimodal alignment",
            "network inference",
            "heterogeneous graph",
            "multi-omics batch effect correction"
        ],
        "grn": [
            "grn", "gene regulatory", "network inference",
            "multi-omics", "unpaired", "prior"
        ],
        "perturbation": [
            "perturbation", "drug response", "genetic perturbation", "chemical perturbations",
            "drug perturbation", "Cellular Responses", "drug discovery"
        ],
    }
    
    def __init__(
        self,
        enable_pubmed: bool = True,
        enable_web_search: bool = False,
        pubmed_email: Optional[str] = None,
        max_results: int = 20,
        min_year: Optional[int] = None,
    ):
        self.enable_pubmed = enable_pubmed
        self.enable_web_search = enable_web_search
        self.max_results = max_results
        self.min_year = min_year or (datetime.now().year - 5)
        
        self.pubmed_searcher = None
        if enable_pubmed:
            try:
                from Utils.pubmed_search import PubMedSearcher
                self.pubmed_searcher = PubMedSearcher(
                    email=pubmed_email or "user@example.com",
                    max_results=max_results
                )
                logger.info("PubMed searcher initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PubMed searcher: {e}")
    
    def should_retrieve(
        self,
        task_description: str,
        data_semantics: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine whether literature retrieval is needed.

        Args:
            task_description: Task description text
            data_semantics: Data semantic features
            task_type: Task type (integration, grn, perturbation)

        Returns:
            (whether retrieval is needed, reason)
        """
        # Task-agnostic general keywords
        general_keywords = ["single-cell", "scRNA-seq", "deep learning"]

        task_lower = task_description.lower().replace('_', ' ')
        task_type = task_type or ""

        # Get corresponding keywords based on task type (using class attribute)
        task_keywords = self.RETRIEVAL_KEYWORDS_MAP.get(task_type, general_keywords)
        matches = [kw for kw in task_keywords if kw.lower().replace('_', ' ') in task_lower]

        # If task description contains task type keywords, trigger directly
        if len(matches) >= 1:
            return True, f"{task_type.capitalize()} task detected with keywords: {matches}"

        # Fallback: if keywords didn't match but data_semantics is non-empty, still trigger retrieval
        if data_semantics and any(v is not None for v in data_semantics.values()):
            return True, f"Data available for {task_type or 'generic'} task, triggering literature retrieval"

        return False, "No specific retrieval triggers detected"
    
    def retrieve(
        self,
        query: str,
        task_type: Optional[str] = None,
        data_info: Optional[Dict[str, Any]] = None,
    ) -> List[LiteratureRecord]:
        """
        Retrieve literature

        Args:
            query: Base query term
            task_type: Task type (integration, grn, perturbation)
            data_info: Data information

        Returns:
            List of literature records
        """
        all_records = []
        
        # Build enhanced query
        enhanced_query = self._build_query(query, task_type, data_info)
        logger.info(f"Retrieving literature with query: {enhanced_query}")

        # PubMed retrieval
        if self.enable_pubmed and self.pubmed_searcher:
            try:
                pubmed_records = self._search_pubmed(enhanced_query, task_type)
                all_records.extend(pubmed_records)
                logger.info(f"Found {len(pubmed_records)} records from PubMed")
            except Exception as e:
                logger.warning(f"PubMed search failed: {e}")
        
        # Web search (optional)
        if self.enable_web_search:
            try:
                web_records = self._search_web(enhanced_query)
                all_records.extend(web_records)
                logger.info(f"Found {len(web_records)} records from web search")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Deduplicate and rank
        records = self._deduplicate_and_rank(all_records)
        
        return records[:self.max_results]
    
    def _build_query(
        self,
        base_query: str,
        task_type: Optional[str],
        data_info: Optional[Dict[str, Any]]
    ) -> str:
        """Build enhanced query"""
        parts = [base_query]
        
        # Use keywords from class attribute
        task_keywords = self.RETRIEVAL_KEYWORDS_MAP.get(task_type, [])
        parts.extend(task_keywords)
        
        # Add data features
        # if data_info:
        #     if data_info.get('modality') == 'multi':
        #         parts.append("scRNA-seq scATAC-seq")
        #     elif data_info.get('modality') == 'rna':
        #         parts.append("scRNA-seq")
        #     elif data_info.get('modality') == 'atac':
        #         parts.append("scATAC-seq")
        
        return " OR ".join(parts)
    
    def _search_pubmed(self, query: str, task_type: Optional[str] = None) -> List[LiteratureRecord]:
        """Search PubMed

        Args:
            query: Enhanced query term
            task_type: Task type (integration, grn, perturbation)
        """
        if not self.pubmed_searcher:
            return []

        try:
            # Dispatch to corresponding retrieval method based on task type
            if task_type == "integration":
                papers = self.pubmed_searcher.search_multi_omics_methods(
                    data_type=query,
                    year=self.min_year,
                    max_papers=self.max_results
                )
            elif task_type == "grn":
                papers = self.pubmed_searcher.search_grn_methods(
                    data_type=query,
                    year=self.min_year,
                    max_papers=self.max_results
                )
            elif task_type in ("perturbation", "genetic_perturbation"):
                papers = self.pubmed_searcher.search_perturbation_methods(
                    data_type=query,
                    year=self.min_year,
                    max_papers=self.max_results
                )
            else:
                # General search
                papers = self.pubmed_searcher.search(
                    query=query,
                    year=self.min_year
                )
            
            records = []
            for paper in papers:
                record = LiteratureRecord(
                    title=paper.get('title', 'Unknown'),
                    authors=paper.get('authors', []),
                    journal=paper.get('journal', 'Unknown'),
                    year=paper.get('year', 0),
                    abstract=paper.get('abstract', ''),
                    pmid=paper.get('pmid'),
                    doi=paper.get('doi'),
                    source="pubmed",
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.warning(f"PubMed search error: {e}")
            return []
    
    def _search_web(self, query: str) -> List[LiteratureRecord]:
        """Web search (placeholder)"""
        # Actual implementation can use search engine API
        logger.info("Web search not implemented yet")
        return []
    
    def _deduplicate_and_rank(
        self,
        records: List[LiteratureRecord]
    ) -> List[LiteratureRecord]:
        """Deduplicate and sort by relevance"""
        # Deduplicate based on PMID
        seen_pmids = set()
        unique_records = []
        
        for record in records:
            key = record.pmid or record.title.lower()
            if key not in seen_pmids:
                seen_pmids.add(key)
                unique_records.append(record)
        
        # Sort by year and relevance
        unique_records.sort(
            key=lambda r: (r.relevance_score, r.year),
            reverse=True
        )
        
        return unique_records
    
    def format_for_prompt(
        self,
        records: List[LiteratureRecord],
        max_records: int = 5
    ) -> str:
        """
        Format literature records as prompt text

        Args:
            records: List of literature records
            max_records: Maximum number of records to include

        Returns:
            Formatted text
        """
        if not records:
            return "No relevant literature found."
        
        lines = ["## Relevant Literature\n"]
        
        for i, record in enumerate(records[:max_records], 1):
            lines.append(f"### {i}. {record.title}")
            lines.append(record.to_text())
            lines.append("")
        
        return "\n".join(lines)
    
    def get_knowledge_context(
        self,
        task_description: str,
        data_semantics: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Get knowledge context (automatically determines whether retrieval is needed)

        Returns:
            Formatted knowledge context text
        """
        should_retrieve, reason = self.should_retrieve(task_description, data_semantics, task_type)
        
        if not should_retrieve:
            return ""
        
        logger.info(f"Literature retrieval triggered: {reason}")
        
        records = self.retrieve(
            query=task_description,
            task_type=task_type,
            data_info=data_semantics
        )
        
        return self.format_for_prompt(records)


# Global retriever instance
_default_retriever: Optional[LiteratureRetriever] = None


def get_literature_retriever(
    enable_pubmed: bool = True,
    **kwargs
) -> LiteratureRetriever:
    """Get global literature retriever instance"""
    global _default_retriever
    
    if _default_retriever is None:
        _default_retriever = LiteratureRetriever(
            enable_pubmed=enable_pubmed,
            **kwargs
        )
    
    return _default_retriever
