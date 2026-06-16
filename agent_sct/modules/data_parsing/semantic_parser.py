"""
Data Semantic Parser

Refactored from the existing PerturbationDataParser, extracts data semantic features and generates
textual descriptions for LLM to understand data characteristics without directly handling
high-dimensional numerical matrices.

New: Small sample display functionality, showing actual content of anndata's obs, var, and partial X.
New: Hardware resource information collection, helping architects design algorithms that fit resource constraints.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """Get system hardware resource information"""
    hw_info = {
        'cpu_count': os.cpu_count(),
        'memory_gb': None,
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_name': None,
        'gpu_memory_gb': None,
    }
    
    # Get memory information
    try:
        import psutil
        mem = psutil.virtual_memory()
        hw_info['memory_gb'] = round(mem.total / (1024**3), 2)
    except ImportError:
        try:
            # Linux system: get memory info via /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        hw_info['memory_gb'] = round(mem_kb / (1024**2), 2)
                        break
        except Exception:
            pass
    
    # Get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            hw_info['gpu_available'] = True
            hw_info['gpu_count'] = torch.cuda.device_count()
            hw_info['gpu_name'] = torch.cuda.get_device_name(0)
            # Get GPU memory (GB)
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            hw_info['gpu_memory_gb'] = round(mem_bytes / (1024**3), 2)
    except ImportError:
        # Try to get via nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    hw_info['gpu_available'] = True
                    hw_info['gpu_count'] = len(lines)
                    # Parse first line: GPU name, memory
                    parts = lines[0].split(',')
                    if len(parts) >= 2:
                        hw_info['gpu_name'] = parts[0].strip()
                        mem_str = parts[1].strip()
                        if 'MiB' in mem_str:
                            hw_info['gpu_memory_gb'] = round(int(mem_str.replace('MiB', '').strip()) / 1024, 2)
                        elif 'GiB' in mem_str:
                            hw_info['gpu_memory_gb'] = float(mem_str.replace('GiB', '').strip())
        except Exception:
            pass
    
    return hw_info


@dataclass
class DataSample:
    """Data sample container - small sample display"""
    # Cell metadata sample (obs)
    obs_sample: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Feature metadata sample (var)
    var_sample: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Expression matrix sample (X)
    expression_sample: Optional[np.ndarray] = None
    
    # Sample info
    n_obs_sampled: int = 0
    n_var_sampled: int = 0
    
    def to_text(self) -> str:
        """Convert sample to text description"""
        lines = []
        
        # Obs sample
        if not self.obs_sample.empty:
            lines.append("### Cell Metadata Sample (obs)")
            lines.append(f"Showing {self.n_obs_sampled} cells out of total:")
            lines.append(self.obs_sample.head(5).to_string())
            lines.append("")
            
            # Show obs column statistics
            lines.append("Cell Metadata Columns:")
            for col in self.obs_sample.columns:
                dtype = self.obs_sample[col].dtype
                n_unique = self.obs_sample[col].nunique()
                
                # Detect Categorical type
                is_categorical = hasattr(self.obs_sample[col], 'cat') or str(dtype) == 'category'
                cat_warning = " [CATEGORICAL - use .astype(str) before fillna]" if is_categorical else ""
                
                lines.append(f"  - {col}: {dtype}, {n_unique} unique values{cat_warning}")
                
                # Show some unique values
                if n_unique <= 10:
                    unique_vals = self.obs_sample[col].unique()
                    lines.append(f"    Values: {list(unique_vals)}")
                else:
                    unique_vals = self.obs_sample[col].unique()[:5]
                    lines.append(f"    Sample values: {list(unique_vals)}...")
            lines.append("")
        
        # Var sample
        if not self.var_sample.empty:
            lines.append("### Feature Metadata Sample (var)")
            lines.append(f"Showing {self.n_var_sampled} features out of total:")
            lines.append(self.var_sample.head(5).to_string())
            lines.append("")
            
            # Show var column statistics
            lines.append("Feature Metadata Columns:")
            for col in self.var_sample.columns:
                dtype = self.var_sample[col].dtype
                lines.append(f"  - {col}: {dtype}")
            lines.append("")
        
        # Expression matrix sample
        if self.expression_sample is not None:
            lines.append("### Expression Matrix Sample (X)")
            lines.append(f"Shape: {self.expression_sample.shape}")
            lines.append("First 5 cells × first 5 features:")
            
            # Convert to DataFrame for display
            sample_df = pd.DataFrame(
                self.expression_sample[:5, :5],
                index=[f"Cell_{i}" for i in range(min(5, self.expression_sample.shape[0]))],
                columns=[f"Feat_{i}" for i in range(min(5, self.expression_sample.shape[1]))]
            )
            lines.append(sample_df.to_string())
            lines.append("")
            
            # Expression value statistics
            lines.append("Expression Value Statistics:")
            lines.append(f"  Mean: {np.mean(self.expression_sample):.4f}")
            lines.append(f"  Std: {np.std(self.expression_sample):.4f}")
            lines.append(f"  Min: {np.min(self.expression_sample):.4f}")
            lines.append(f"  Max: {np.max(self.expression_sample):.4f}")
            lines.append(f"  Sparsity (zero ratio): {np.mean(self.expression_sample == 0):.2%}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class DataSemantics:
    """Data semantic feature container"""
    # Basic statistics
    n_cells: int
    n_features: int
    sparsity: float
    
    # RNA-specific
    n_genes: Optional[int] = None
    gene_detection_rate: Optional[float] = None
    
    # ATAC-specific
    n_peaks: Optional[int] = None
    peak_detection_rate: Optional[float] = None
    
    # Sequencing depth
    mean_sequencing_depth: float = 0.0
    std_sequencing_depth: float = 0.0
    
    # Batch information
    n_batches: Optional[int] = None
    batch_labels: Optional[List[str]] = None
    
    # Cell type
    n_cell_types: Optional[int] = None
    cell_type_labels: Optional[List[str]] = None
    cell_type_distribution: Optional[Dict[str, int]] = None
    
    # Perturbation information (perturbation tasks)
    n_conditions: Optional[int] = None
    condition_labels: Optional[List[str]] = None
    control_labels: Optional[List[str]] = None
    
    # Metadata
    data_signature: str = ""
    modality: str = "unknown"
    
    # Sample data
    data_sample: Optional[DataSample] = None
    
    # Hardware resource information
    hardware_info: Optional[Dict[str, Any]] = None
    
    def to_text_summary(self) -> str:
        """Generate natural language description"""
        lines = [
            f"Data Modality: {self.modality}",
            f"Dataset Size: {self.n_cells:,} cells × {self.n_features:,} features",
            f"Matrix Sparsity: {self.sparsity:.2%}",
            f"Mean Sequencing Depth: {self.mean_sequencing_depth:.1f} ± {self.std_sequencing_depth:.1f}",
        ]
        
        if self.n_genes:
            lines.append(f"Gene Count: {self.n_genes:,}")
        if self.gene_detection_rate:
            lines.append(f"Gene Detection Rate: {self.gene_detection_rate:.2%}")
        if self.n_peaks:
            lines.append(f"Peak Count: {self.n_peaks:,}")
        if self.peak_detection_rate:
            lines.append(f"Peak Detection Rate: {self.peak_detection_rate:.2%}")
        
        if self.n_cell_types:
            lines.append(f"Cell Types: {self.n_cell_types} ({', '.join(self.cell_type_labels[:5])}" + 
                        (f"... +{self.n_cell_types-5} more" if self.n_cell_types > 5 else ")"))
        
        if self.n_batches:
            lines.append(f"Batches: {self.n_batches}")
        
        if self.n_conditions:
            lines.append(f"Conditions: {self.n_conditions} ({', '.join(self.condition_labels[:5])}" +
                        (f"... +{self.n_conditions-5} more" if self.n_conditions > 5 else ")"))
            if self.control_labels:
                lines.append(f"Control Labels: {', '.join(self.control_labels)}")
        
        # Add hardware resource information
        if self.hardware_info:
            lines.append("")
            lines.append("=== Hardware Resources ===")
            hw = self.hardware_info
            if hw.get('cpu_count'):
                lines.append(f"CPU Cores: {hw['cpu_count']}")
            if hw.get('memory_gb'):
                lines.append(f"System Memory: {hw['memory_gb']} GB")
            if hw.get('gpu_available'):
                lines.append(f"GPU: {hw.get('gpu_name', 'Unknown')} ({hw.get('gpu_count', 1)}x)")
                if hw.get('gpu_memory_gb'):
                    lines.append(f"GPU Memory: {hw['gpu_memory_gb']} GB")
            else:
                lines.append("GPU: Not available")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'n_cells': self.n_cells,
            'n_features': self.n_features,
            'sparsity': self.sparsity,
            'n_genes': self.n_genes,
            'gene_detection_rate': self.gene_detection_rate,
            'n_peaks': self.n_peaks,
            'peak_detection_rate': self.peak_detection_rate,
            'mean_sequencing_depth': self.mean_sequencing_depth,
            'std_sequencing_depth': self.std_sequencing_depth,
            'n_batches': self.n_batches,
            'batch_labels': self.batch_labels,
            'n_cell_types': self.n_cell_types,
            'cell_type_labels': self.cell_type_labels,
            'cell_type_distribution': self.cell_type_distribution,
            'n_conditions': self.n_conditions,
            'condition_labels': self.condition_labels,
            'control_labels': self.control_labels,
            'data_signature': self.data_signature,
            'modality': self.modality,
            'hardware_info': self.hardware_info,
        }


class SemanticDataParser:
    """
    Data Semantic Parser

    Extracts semantic features from single-cell omics data and generates
    text descriptions understandable by LLM.
    Supports scRNA-seq, scATAC-seq, and multi-omics data.

    New: Small sample display functionality, allowing LLM to see actual data content.
    """
    
    # Column name candidates (for auto-detection)
    PERTURBATION_COL_CANDIDATES = [
        "perturbation", "drug", "treatment", "condition",
        "perturbed_gene", "sgRNA", "gene_perturbation",
        "pert_type", "perturbation_type", "pert", "gene",
        "perturbation_condition", "perturbation_label",
    ]
    
    CELLTYPE_COL_CANDIDATES = [
        "cell_type", "celltype", "cellType",
        "cell_line", "cell_label", "celltype_ontology_term_id",
        "cell_type_ontology", "cell_label",
    ]
    
    BATCH_COL_CANDIDATES = [
        "batch", "batch_id", "batch_label", "sample",
        "sample_id", "library_id", "sequencing_batch",
    ]
    
    CONTROL_LABEL_CANDIDATES = [
        "control", "ctrl", "vehicle", "dmso",
        "untreated", "baseline", "wt", "wildtype",
        "neg", "neg_control", "no_treatment",
    ]
    
    def __init__(
        self,
        data_path: str,
        modality: Literal["rna", "atac", "multi"] = "rna",
        verbose: bool = True,
        # Sample parameters
        sample_n_obs: int = 100,  # Number of cells to sample
        sample_n_vars: int = 50,  # Number of features to sample
    ):
        self.data_path = Path(data_path).resolve()
        self.modality = modality
        self.verbose = verbose
        self.sample_n_obs = sample_n_obs
        self.sample_n_vars = sample_n_vars
        
        # Load data
        self.adata = self._load_data()
        self._obs = self._load_obs()
        self._var = self._load_var()
        
        # Auto-detect columns
        self.pert_col = self._detect_column(self.PERTURBATION_COL_CANDIDATES, self._obs)
        self.celltype_col = self._detect_column(self.CELLTYPE_COL_CANDIDATES, self._obs)
        self.batch_col = self._detect_column(self.BATCH_COL_CANDIDATES, self._obs)
        
        if verbose:
            logger.info(f"Loaded {modality} data from {self.data_path}")
            logger.info(f"  Cells: {self.adata.n_obs:,}, Features: {self.adata.n_vars:,}")
    
    def _load_data(self) -> ad.AnnData:
        """Load data file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        suffix = self.data_path.suffix.lower()
        if suffix == '.h5ad':
            return ad.read_h5ad(self.data_path, backed="r")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_obs(self) -> pd.DataFrame:
        """Load obs metadata"""
        obs = self.adata.obs
        if hasattr(obs, "to_dataframe"):
            return obs.to_dataframe()
        return obs.copy()
    
    def _load_var(self) -> pd.DataFrame:
        """Load var metadata"""
        var = self.adata.var
        if hasattr(var, "to_dataframe"):
            return var.to_dataframe()
        return var.copy()
    
    def _detect_column(self, candidates: List[str], df: pd.DataFrame) -> Optional[str]:
        """Detect column name from DataFrame"""
        cols = set(df.columns)
        for col in candidates:
            if col in cols:
                return col
        return None
    
    def _sample_data(self) -> DataSample:
        n_obs = min(self.sample_n_obs, self.adata.n_obs)
        n_vars = min(self.sample_n_vars, self.adata.n_vars)

        if self.adata.n_obs > n_obs:
            obs_indices = np.random.choice(self.adata.n_obs, n_obs, replace=False)
            obs_indices = np.sort(obs_indices)
        else:
            obs_indices = np.arange(self.adata.n_obs)

        if self.adata.n_vars > n_vars:
            var_indices = np.random.choice(self.adata.n_vars, n_vars, replace=False)
            var_indices = np.sort(var_indices)
        else:
            var_indices = np.arange(self.adata.n_vars)

        obs_sample = self._obs.iloc[obs_indices].copy()
        var_sample = self._var.iloc[var_indices].copy()

        X_sample = self._load_x_sample(obs_indices, var_indices)

        return DataSample(
            obs_sample=obs_sample,
            var_sample=var_sample,
            expression_sample=X_sample,
            n_obs_sampled=n_obs,
            n_var_sampled=n_vars,
        )

    def _load_x_sample(
        self,
        obs_indices: np.ndarray,
        var_indices: np.ndarray,
    ) -> np.ndarray:
        X = self._slice_X(obs_indices)
        return X[:, var_indices]
    
    def parse(self, include_sample: bool = True) -> DataSemantics:
        """
        Parse data semantic features

        Args:
            include_sample: Whether to include small sample data

        Returns:
            DataSemantics: Data semantic feature object
        """
        # Basic statistics
        n_cells = self.adata.n_obs
        n_features = self.adata.n_vars
        
        # Calculate sparsity
        sparsity = self._calculate_sparsity()
        
        # Sequencing depth statistics
        depth_stats = self._calculate_sequencing_depth()
        
        # Sample data (small sample display)
        data_sample = None
        if include_sample:
            logger.info("Sampling data for preview...")
            data_sample = self._sample_data()
        
        # Build semantic object
        semantics = DataSemantics(
            n_cells=n_cells,
            n_features=n_features,
            sparsity=sparsity,
            mean_sequencing_depth=depth_stats['mean'],
            std_sequencing_depth=depth_stats['std'],
            modality=self.modality,
            data_signature=self._compute_signature(),
            data_sample=data_sample,
        )
        
        # Modality-specific features
        if self.modality in ["rna", "multi"]:
            semantics.n_genes = n_features
            semantics.gene_detection_rate = self._calculate_detection_rate()
        
        if self.modality in ["atac", "multi"]:
            semantics.n_peaks = n_features
            semantics.peak_detection_rate = self._calculate_detection_rate()
        
        # Cell type information
        if self.celltype_col:
            cell_types = self._get_unique_values(self.celltype_col, self._obs)
            semantics.n_cell_types = len(cell_types)
            semantics.cell_type_labels = cell_types
            semantics.cell_type_distribution = self._obs[self.celltype_col].value_counts().to_dict()
        
        # Batch information
        if self.batch_col:
            batches = self._get_unique_values(self.batch_col, self._obs)
            semantics.n_batches = len(batches)
            semantics.batch_labels = batches
        
        # Perturbation information
        if self.pert_col:
            conditions = self._get_unique_values(self.pert_col, self._obs)
            semantics.n_conditions = len(conditions)
            semantics.condition_labels = conditions
            semantics.control_labels = self._detect_control_labels()
        
        # Collect hardware resource information
        logger.info("Collecting hardware information...")
        semantics.hardware_info = get_hardware_info()
        if semantics.hardware_info.get('gpu_available'):
            logger.info(f"  GPU: {semantics.hardware_info.get('gpu_name')} ({semantics.hardware_info.get('gpu_memory_gb')} GB)")
        if semantics.hardware_info.get('memory_gb'):
            logger.info(f"  System Memory: {semantics.hardware_info.get('memory_gb')} GB")
        
        if self.verbose:
            logger.info("Data semantics parsed successfully")
            logger.info(f"  Sparsity: {sparsity:.2%}")
            if semantics.n_cell_types:
                logger.info(f"  Cell types: {semantics.n_cell_types}")
            if semantics.n_conditions:
                logger.info(f"  Conditions: {semantics.n_conditions}")
            if data_sample:
                logger.info(f"  Sampled: {data_sample.n_obs_sampled} cells × {data_sample.n_var_sampled} features")
        
        return semantics

    def _get_sparse_X(self, obs_indices: Optional[np.ndarray] = None):
        if obs_indices is not None:
            X = self.adata[obs_indices].X
        else:
            X = self.adata[::].X
        return X

    def _slice_X(self, obs_indices: Optional[np.ndarray] = None) -> np.ndarray:
        X = self._get_sparse_X(obs_indices)
        if hasattr(X, 'toarray'):
            X = X.toarray()
        arr = np.asarray(X)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        return arr

    def _calculate_sparsity(self) -> float:
        X = self._get_sparse_X()
        if hasattr(X, 'nnz'):
            nnz = X.nnz
        else:
            nnz = np.count_nonzero(X)
        total = X.shape[0] * X.shape[1]
        return 1.0 - (nnz / total) if total > 0 else 0.0
    
    def _calculate_detection_rate(self) -> float:
        if self.adata.n_obs > 10000:
            sample_idx = np.sort(np.random.choice(self.adata.n_obs, 10000, replace=False))
            X = self._get_sparse_X(sample_idx)
        else:
            X = self._get_sparse_X()

        if hasattr(X, 'toarray'):
            detected = (X > 0).sum(axis=0)
        else:
            detected = (X > 0).sum(axis=0)
        if hasattr(detected, 'A1'):
            detected = detected.A1
        detection_rate = float(np.mean(np.asarray(detected).ravel() > 0))

        return detection_rate
    
    def _calculate_sequencing_depth(self) -> Dict[str, float]:
        if self.adata.n_obs > 10000:
            sample_idx = np.sort(np.random.choice(self.adata.n_obs, 10000, replace=False))
            X = self._get_sparse_X(sample_idx)
        else:
            X = self._get_sparse_X()

        depth_per_cell = np.asarray(X.sum(axis=1)).ravel()
        
        return {
            'mean': float(np.mean(depth_per_cell)),
            'std': float(np.std(depth_per_cell)),
            'median': float(np.median(depth_per_cell)),
            'min': float(np.min(depth_per_cell)),
            'max': float(np.max(depth_per_cell)),
        }
    
    def _get_unique_values(self, col: str, df: pd.DataFrame) -> List[str]:
        """Get all unique values of a column"""
        return [str(v) for v in df[col].dropna().unique()]
    
    def _detect_control_labels(self) -> List[str]:
        """Auto-detect control labels"""
        if not self.pert_col:
            return []
        
        values = self._obs[self.pert_col].value_counts()
        
        # Strategy 1: keyword matching
        matched = []
        for label in values.index:
            label_lower = str(label).lower()
            if any(kw in label_lower for kw in self.CONTROL_LABEL_CANDIDATES):
                matched.append(str(label))
        
        if matched:
            return matched
        
        # Strategy 2: use the most frequent as control
        return [str(values.index[0])]
    
    def _compute_signature(self) -> str:
        """Compute data signature"""
        key = f"{self.data_path}|{self.adata.n_obs}|{self.adata.n_vars}|{self.modality}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def get_prompt_context(self, include_sample: bool = True) -> str:
        """
        Generate context for LLM prompts

        Includes statistics and actual data samples
        """
        semantics = self.parse(include_sample=include_sample)
        
        context_parts = []
        
        # 1. Data summary
        context_parts.append("## Data Summary")
        context_parts.append(semantics.to_text_summary())
        context_parts.append("")
        
        # 2. Small sample data display
        if include_sample and semantics.data_sample:
            context_parts.append("## Data Sample Preview")
            context_parts.append("Below is a sample of the actual data to help you understand its structure:")
            context_parts.append("")
            context_parts.append(semantics.data_sample.to_text())
        
        # 3. Technical details
        context_parts.append("## Technical Details")
        context_parts.append(f"- Data Signature: {semantics.data_signature}")
        context_parts.append(f"- File Path: {self.data_path}")
        context_parts.append("")
        
        # 4. Preprocessing suggestions
        suggestions = self._generate_suggestions(semantics)
        if suggestions:
            context_parts.append("## Preprocessing Suggestions")
            for s in suggestions:
                context_parts.append(f"- {s}")
        
        return "\n".join(context_parts)
    
    def _generate_suggestions(self, semantics: DataSemantics) -> List[str]:
        """Generate preprocessing suggestions based on data features"""
        suggestions = []
        
        if semantics.sparsity > 0.95:
            suggestions.append("High sparsity detected (>95%). Consider feature selection or dimensionality reduction.")
        
        if semantics.n_cells and semantics.n_cells > 50000:
            suggestions.append("Large dataset detected. Consider mini-batch training or sampling strategies.")
        
        if semantics.n_batches and semantics.n_batches > 1:
            suggestions.append("Multiple batches detected. Consider batch correction methods.")
        
        return suggestions


class MultiOmicsSemanticParser:
    """
    Multi-omics data semantic parser

    Simultaneously parses RNA and ATAC data, generating unified semantic descriptions.
    Includes small sample display functionality.
    """
    
    def __init__(
        self,
        rna_path: Optional[str] = None,
        atac_path: Optional[str] = None,
        verbose: bool = True,
        sample_n_obs: int = 100,
        sample_n_vars: int = 50,
    ):
        self.rna_parser = None
        self.atac_parser = None
        
        if rna_path:
            self.rna_parser = SemanticDataParser(
                rna_path, modality="rna", verbose=verbose,
                sample_n_obs=sample_n_obs, sample_n_vars=sample_n_vars
            )
        
        if atac_path:
            self.atac_parser = SemanticDataParser(
                atac_path, modality="atac", verbose=verbose,
                sample_n_obs=sample_n_obs, sample_n_vars=sample_n_vars
            )
    
    def parse(self, include_sample: bool = True) -> Dict[str, Any]:
        """Parse multi-omics data semantics"""
        result = {
            'modality': 'multi',
            'rna': None,
            'atac': None,
            'paired': False,
        }
        
        if self.rna_parser:
            rna_semantics = self.rna_parser.parse(include_sample=include_sample)
            result['rna'] = rna_semantics.to_dict()
            if rna_semantics.data_sample:
                result['rna_sample'] = rna_semantics.data_sample.to_text()
        
        if self.atac_parser:
            atac_semantics = self.atac_parser.parse(include_sample=include_sample)
            result['atac'] = atac_semantics.to_dict()
            if atac_semantics.data_sample:
                result['atac_sample'] = atac_semantics.data_sample.to_text()
        
        # Check if data is paired
        if self.rna_parser and self.atac_parser:
            result['paired'] = self._check_paired()

            rna_types = set()
            atac_types = set()
            rna_data = result.get('rna', {})
            atac_data = result.get('atac', {})
            if isinstance(rna_data, dict) and rna_data.get('cell_type_labels'):
                rna_types = set(rna_data['cell_type_labels'])
            if isinstance(atac_data, dict) and atac_data.get('cell_type_labels'):
                atac_types = set(atac_data['cell_type_labels'])
            overlap = rna_types & atac_types
            if overlap and len(overlap) / min(len(rna_types), len(atac_types), 1) > 0.3:
                result['metric_format'] = 'shared'
            else:
                result['metric_format'] = 'per_modality'
        
        return result
    
    def _check_paired(self) -> bool:
        """Check if RNA and ATAC data are paired"""
        if not (self.rna_parser and self.atac_parser):
            return False

        # Check 1: whether cell counts are the same
        if self.rna_parser.adata.n_obs != self.atac_parser.adata.n_obs:
            return False

        # Check 2: whether cell barcodes have overlap
        try:
            rna_barcodes = set(self.rna_parser.adata.obs_names.astype(str))
            atac_barcodes = set(self.atac_parser.adata.obs_names.astype(str))
            overlap = rna_barcodes & atac_barcodes
            if len(overlap) > 0.5 * min(len(rna_barcodes), len(atac_barcodes)):
                return True
        except Exception:
            pass

        # If cell counts are the same but no barcode overlap, still consider as possibly paired
        return self.rna_parser.adata.n_obs == self.atac_parser.adata.n_obs
    
    def get_prompt_context(self, include_sample: bool = True) -> str:
        """Generate context for LLM prompts"""
        context_parts = ["# Multi-Omics Data Analysis\n"]
        
        if self.rna_parser:
            context_parts.append("## scRNA-seq Data")
            context_parts.append(self.rna_parser.get_prompt_context(include_sample=include_sample))
        
        if self.atac_parser:
            context_parts.append("\n## scATAC-seq Data")
            context_parts.append(self.atac_parser.get_prompt_context(include_sample=include_sample))
        
        if self.rna_parser and self.atac_parser:
            paired = self._check_paired()
            context_parts.append(f"\n## Data Pairing")
            context_parts.append(f"Paired data: {paired}")
            if not paired:
                context_parts.append("Note: Unpaired data requires alignment or integration methods.")
        
        return "\n\n".join(context_parts)
