"""
GRN Inference Tasks - Gene Regulatory Network Inference Module

Provides GRN inference tasks:
1. task_grn.py - GRN inference task (supports BEELINE + ChIP-seq benchmark evaluation)

Usage:
```python
from agent_sct.tasks.grn_inference import run_grn_inference

# GRN task with ChIP-seq benchmark
result = await run_grn_inference(
    rna_path="data/rna.h5ad",
    atac_path="data/atac.h5ad",
    chipseq_ground_truth="data/dataset/chip_seq_TF/",
)
```

Launch commands:
```bash
# Single-omics GRN inference
python scripts/run_agent_sct.py --task grn --data data/rna_final.h5ad --chipseq data/dataset/chip_seq_TF/

# Multi-omics GRN inference (RNA + ATAC)
python scripts/run_agent_sct.py --task grn --rna data/rna.h5ad --atac data/atac.h5ad --chipseq data/dataset/chip_seq_TF/

# YAML config launch
python run_grn_multi_round.py --config grn.yaml
```
"""

from .task_grn import GRNInferenceTask, run_grn_inference

__all__ = [
    "GRNInferenceTask",
    "run_grn_inference",
]
