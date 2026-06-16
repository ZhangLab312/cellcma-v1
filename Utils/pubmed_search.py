"""
PubMed Literature Search Tool

Features:
1. Search PubMed database for relevant papers
2. Parse paper abstracts and key information
3. Extract model architecture and method information
4. Cache search results to avoid duplicate queries
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️  Biopython not installed. Install with: pip install biopython")

from Utils.logger import get_logger

logger = get_logger("PubMedSearch", log_dir="logs/pubmed")


class PubMedSearcher:
    """PubMed literature searcher."""

    def __init__(
        self,
        email: str = "your_email@example.com",  # Entrez requires an email
        api_key: Optional[str] = None,
        cache_dir: str = "cache/pubmed",
        max_results: int = 20,
        all_results: bool = False,
    ):
        """
        Initialize the PubMed searcher.

        Args:
            email: Email address required by the Entrez API
            api_key: Optional NCBI API key (to increase rate limits)
            cache_dir: Cache directory
            max_results: Maximum number of results to return (effective when all_results=False)
            all_results: If True, ignore max_results and fetch all matching results (may be thousands)
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("Biopython is required. Install with: pip install biopython")

        self.email = email
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.all_results = all_results

        # Configure Entrez
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        logger.info(
            "PubMedSearcher initialized",
            email=email,
            has_api_key=bool(api_key),
            cache_dir=str(cache_dir),
            all_results=all_results,
        )

    def _get_cache_key(self, query: str, year: Optional[int] = None) -> str:
        """Generate a cache key."""
        key_data = f"{query}_{year}_{self.max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _build_search_query(self, query: str, year: Optional[int] = None) -> str:
        """
        Build an optimized PubMed search query.

        Args:
            query: Original query
            year: Year filter

        Returns:
            Optimized search query
        """
        search_query = query.strip()
        search_query = re.sub(r'[+]', ' ', search_query)
        search_query = re.sub(r'\s+', ' ', search_query)

        if ' OR ' in search_query:
            search_query = f"({search_query})"

        if year:
            search_query += f" AND ({year}[PDAT] : {datetime.now().year}[PDAT])"

        search_query += " NOT review[pt]"

        return search_query

    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Check if cache has expired (7 days)
                cache_time = datetime.fromisoformat(data.get('timestamp', '2020-01-01'))
                if (datetime.now() - cache_time).days < 7:
                    logger.debug(f"Loaded from cache: {cache_key}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_cache(self, cache_key: str, data: Dict):
        """Save to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        data['timestamp'] = datetime.now().isoformat()
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def search(
        self,
        query: str,
        year: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed.

        Args:
            query: Search keywords
            year: Year filter (only retrieve papers from this year onwards)
            use_cache: Whether to use cache

        Returns:
            List of papers, each containing title, abstract, authors, citation, etc.
        """
        cache_key = self._get_cache_key(query, year)

        # Check cache
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(
                    f"Retrieved from cache: {len(cached['papers'])} papers",
                    query=query,
                    cache_key=cache_key
                )
                return cached['papers']

        logger.info(
            f"🔍 Searching PubMed: {query}",
            max_results=self.max_results,
            year=year
        )

        try:
            # Build search query (improved version)
            search_query = self._build_search_query(query, year)

            # Search
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=self.max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()

            id_list = search_results.get('IdList', [])
            total_count = int(search_results.get('Count', 0))

            # Determine retmax
            retmax = self.max_results
            if self.all_results:
                retmax = min(total_count, 5000)  # Reasonable upper limit to prevent memory overflow
                if total_count > 5000:
                    logger.warning(
                        f"Total {total_count} papers found, capping at 5000. "
                        "Consider narrowing your search query."
                    )

                logger.info(
                    f"Fetching ALL {retmax} papers (total matched: {total_count})",
                    query=query
                )

                # Fetch all IDs in batches
                all_ids = list(id_list)  # First batch already obtained
                batch_size = 500  # Entrez allows max 500 per batch
                for start in range(len(all_ids), retmax, batch_size):
                    try:
                        handle = Entrez.esearch(
                            db="pubmed",
                            term=search_query,
                            retstart=start,
                            retmax=batch_size,
                            sort="relevance"
                        )
                        batch_result = Entrez.read(handle)
                        handle.close()
                        batch_ids = batch_result.get('IdList', [])
                        all_ids.extend(batch_ids)
                        time.sleep(0.3)  # Avoid rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to fetch batch at {start}: {e}")
                        break

                id_list = all_ids[:retmax]

                logger.info(
                    f"Fetched {len(id_list)} total paper IDs",
                    query=query
                )

                if not id_list:
                    return []

            # Fetch abstracts
            papers = self._fetch_details(id_list)

            # Save to cache
            if use_cache and papers:
                self._save_cache(cache_key, {'papers': papers, 'query': query})

            return papers

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _fetch_details(self, id_list: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for papers."""
        papers = []
        batch_size = 100  # Entrez limit

        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]

            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_ids,
                    rettype="medline",
                    retmode="text"
                )
                records = handle.read()
                handle.close()

                # Parse MEDLINE format
                batch_papers = self._parse_medline(records)
                papers.extend(batch_papers)

                # Avoid too-fast requests
                if i + batch_size < len(id_list):
                    time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to fetch details for batch {i}: {e}")

        return papers

    def _parse_medline(self, medline_text: str) -> List[Dict[str, Any]]:
        papers = []
        current_paper = {}
        last_tag = None

        for line in medline_text.split('\n'):
            if not line:
                last_tag = None
                continue

            if len(line) >= 6 and line[4:6] == '- ':
                tag = line[:4].strip()
                value = line[6:].strip()

                if tag == 'PMID':
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = {'pmid': value}

                elif tag == 'TI':
                    current_paper['title'] = value

                elif tag == 'AB':
                    current_paper['abstract'] = value

                elif tag == 'AU':
                    current_paper.setdefault('authors', []).append(value)

                elif tag == 'DP':
                    current_paper['publication_date'] = value

                elif tag == 'TA':
                    current_paper['journal'] = value

                elif tag == 'JT':
                    current_paper['journal_title'] = value

                last_tag = tag

            elif line.startswith('      ') and last_tag == 'AB':
                current_paper['abstract'] += ' ' + line.strip()

            elif line.startswith('      ') and last_tag == 'TI':
                current_paper['title'] += ' ' + line.strip()

        if current_paper:
            papers.append(current_paper)

        return papers

    def search_multi_omics_methods(
        self,
        data_type: str = "single-cell multi-omics",
        integration_method: Optional[str] = None,
        year: Optional[int] = 2022,
        max_papers: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for multi-omics integration methods.

        Args:
            data_type: Data type description
            integration_method: Specific integration method (e.g., "VAE", "MMD")
            year: Starting year
            max_papers: Maximum number of papers

        Returns:
            List of relevant papers
        """
        query_parts = [
            "single-cell multi-omics integration",
            "scRNA-seq scATAC-seq integration",
            "unpaired single-cell integration",
            "cross-modality single-cell",
            "multimodal single-cell alignment",
            "single-cell co-embedding",
            "single-cell batch correction",
            "MultiVI",
            "StabMap",
            "MOIRA",
            "scMM",
            "GLUE single-cell",
            "single-cell vertical integration",
        ]

        if integration_method:
            query_parts.append(integration_method)

        query = " OR ".join([f'"{part}"' for part in query_parts])

        logger.info(
            f"Searching for multi-omics integration methods, data_type={data_type}, method={integration_method}"
        )

        papers = self.search(query, year=year)

        return papers[:max_papers]

    def search_grn_methods(
        self,
        data_type: str = "gene regulatory network",
        method_name: Optional[str] = None,
        year: Optional[int] = 2022,
        max_papers: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for gene regulatory network (GRN) inference methods.

        Args:
            data_type: Data type description
            method_name: Specific method name (e.g., "SCENIC", "PIDC")
            year: Starting year
            max_papers: Maximum number of papers

        Returns:
            List of relevant papers
        """
        query_parts = [
            "gene regulatory network inference",
            "single-cell GRN inference",
            "multi-omics GRN",
            "scRNA-seq scATAC-seq gene regulation",
            "peak-to-gene linkage",
            "TF target prediction single-cell",
            "chromatin accessibility gene regulation",
            "SCENIC",
            "CellOracle",
            "pySCENIC",
            "GRN single-cell multi-omics",
        ]

        if method_name:
            query_parts.append(method_name)

        query = " OR ".join([f'"{part}"' for part in query_parts])

        logger.info(
            f"Searching for GRN inference methods, data_type={data_type}, method={method_name}"
        )

        papers = self.search(query, year=year)

        return papers[:max_papers]

    def search_perturbation_methods(
        self,
        data_type: str = "single-cell perturbation",
        perturbation_type: Optional[str] = None,
        year: Optional[int] = 2022,
        max_papers: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for single-cell perturbation prediction methods.

        Args:
            data_type: Data type description (enhanced query from _build_query)
            perturbation_type: Perturbation type (e.g., "drug response", "CRISPR", "genetic")
            year: Starting year
            max_papers: Maximum number of papers

        Returns:
            List of relevant papers
        """
        # Use more short keywords to improve matching range
        query_parts = [
            "Cellular",
            "perturbation",   # Core keyword
            "drug perturbation",
            "genetic perturbation",
            "gene knockout",
            "CRISPR",
            "transcriptional responses",
            "chemical perturbations",
            "drug response",  # Short keyword combination
            "prediction",     # Short keyword
            "deep learning",  # Short keyword combination
        ]

        if perturbation_type:
            query_parts.append(perturbation_type)

        query = " OR ".join([f'"{part}"' for part in query_parts])

        logger.info(
            f"Searching for perturbation prediction methods, query={query}, perturbation_type={perturbation_type}"
        )

        papers = self.search(query, year=year)

        return papers[:max_papers]

    def extract_architecture_info(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract architecture information from papers.

        Args:
            papers: List of papers

        Returns:
            Summary of extracted architecture information
        """
        if not papers:
            return {
                "total_papers": 0,
                "architectures": [],
                "methods": [],
                "summary": "No papers found"
            }

        architectures = []
        methods_found = set()

        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')

            # Extract model/method name
            architecture = self._extract_model_name(title, abstract)
            if architecture:
                architectures.append({
                    'model': architecture,
                    'title': title,
                    'year': paper.get('publication_date', 'Unknown')[:4],
                    'pmid': paper.get('pmid', ''),
                    'journal': paper.get('journal', '')
                })
                methods_found.add(architecture)

        # Generate summary
        summary_parts = []
        if architectures:
            summary_parts.append(f"Found {len(papers)} relevant papers from PubMed")
            summary_parts.append(f"Identified {len(set(a['model'] for a in architectures))} unique architectures/methods:")
            for arch in architectures[:5]:
                summary_parts.append(f"  - {arch['model']} ({arch['year']})")

        return {
            "total_papers": len(papers),
            "architectures": architectures[:10],  # Return at most 10
            "methods": list(methods_found),
            "summary": "\n".join(summary_parts)
        }

    def _extract_model_name(self, title: str, abstract: str) -> Optional[str]:
        """Extract model name from title and abstract."""
        text = f"{title} {abstract}".lower()

        # Common multi-omics integration methods
        known_methods = [
            'scmo', 'multimodal', 'bindsc', 'totalvi', 'cobolt',
            'scmage', 'deepmap', 'unioncom', 'mofa', 'scvi',
            'vae', 'cvae', 'beta-vae', 'conditional vae',
            'gan', 'wgan', 'cycle gan', 'discriminator',
            'mmd', 'maximum mean discrepancy', 'wasserstein',
            'graph neural network', 'gcn', 'gat',
            'transformer', 'attention', 'bert',
            'autoint', 'bimnet', 'dcca', 'deepcca',
            'integrative nmf', 'constrained nmf'
        ]

        # Search for known methods
        for method in known_methods:
            if method in text:
                return method

        # Try to extract abbreviations (uppercase words) from the title
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', title)
        if acronyms:
            return acronyms[0]

        return None

    def format_for_prompt(
        self,
        papers: List[Dict[str, Any]],
        max_papers: Optional[int] = None
    ) -> str:
        """
        Format paper information into a prompt.

        Args:
            papers: List of papers
            max_papers: Maximum number of papers to include (None = no limit, return all)

        Returns:
            Formatted prompt text
        """
        if not papers:
            return "No relevant papers found in PubMed."

        # Use all papers when no limit is set
        selected = papers if max_papers is None else papers[:max_papers]

        prompt_parts = [
            f"## Relevant Literature from PubMed ({len(papers)} papers found):\n",
            "Use these papers as reference to derive model architectures, hyperparameters, and training strategies. "
            "After reading all papers, SELECT the TOP 10 most relevant to your task and explain why each was chosen.\n\n"
        ]

        for i, paper in enumerate(selected, 1):
            pmid = paper.get('pmid', 'N/A')
            title = paper.get('title', 'Unknown Title')
            year = paper.get('publication_date', 'Unknown')[:4]
            journal = paper.get('journal', 'Unknown')
            abstract = paper.get('abstract', '')

            prompt_parts.append(f"### Paper {i} [PMID:{pmid}]\n")
            prompt_parts.append(f"**Title:** {title}\n")
            prompt_parts.append(f"**Year:** {year}  **Journal:** {journal}\n")

            method = self._extract_model_name(title, abstract)
            if method:
                prompt_parts.append(f"**Method/Architecture:** {method}\n")

            if abstract:
                prompt_parts.append(f"**Abstract:** {abstract}\n")
            prompt_parts.append("\n")

        if max_papers is not None and len(papers) > max_papers:
            prompt_parts.append(
                f"\n*(Showing {max_papers}/{len(papers)} papers. Use all for architecture design.)*\n"
            )

        prompt_parts.append(
            "**Instruction:** Based on the literature above, derive a model architecture "
            "and hyperparameters that are most suitable for the current task. "
            "Cite specific papers (by PMID or title) when explaining your design choices.\n"
        )

        return "".join(prompt_parts)


# ================== Convenience Functions ==================

def search_pubmed_for_integration(
    query: str = "single-cell multi-omics integration",
    year: int = 2022,
    max_results: int = 10,
    email: str = "your_email@example.com"
) -> str:
    """
    Convenience function: search for multi-omics integration papers and return a formatted prompt.

    Args:
        query: Search keywords
        year: Starting year
        max_results: Maximum number of results
        email: Entrez email

    Returns:
        Formatted prompt text
    """
    try:
        searcher = PubMedSearcher(email=email, max_results=max_results)
        papers = searcher.search(query, year=year)
        architecture_info = searcher.extract_architecture_info(papers)

        result = searcher.format_for_prompt(papers, max_papers=5)
        result += f"\n{architecture_info['summary']}\n"

        return result

    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return f"Note: PubMed search encountered an error: {str(e)}. Proceeding with general knowledge."


if __name__ == "__main__":
    # Test code
    import sys

    print("="*60)
    print("PubMed Search Tool Test")
    print("="*60)

    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "single-cell multi-omics integration deep learning"

    print(f"\nSearching for: {query}\n")

    result = search_pubmed_for_integration(
        query=query,
        year=2022,
        max_results=5
    )

    print(result)
    print("\n" + "="*60)
