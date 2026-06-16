# CellCMA

**Collaborative Multi-agent Analysis for Single-cell Tasks**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="AnnData" src="https://img.shields.io/badge/Data-AnnData%20.h5ad-green">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-LLM%20Multi--Agent-purple">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Perturbation%20%7C%20Integration%20%7C%20GRN-lightgrey">
  <img alt="License" src="https://img.shields.io/badge/License-See%20LICENSE-orange">
</p>

> **Code availability statement.** This repository contains the source code, command-line entry point, and example workflows for **CellCMA**, the analysis framework accompanying the manuscript. The code is released to support reproducibility of the results reported in the paper and to enable extension of the method to new single-cell datasets and tasks.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Repository Contents](#repository-contents)
- [Installation](#installation)
- [LLM Configuration](#llm-configuration)
- [Usage](#usage)
- [Input Data Expectations](#input-data-expectations)
- [Output Artifacts](#output-artifacts)
- [Reproducibility](#reproducibility)
- [Evaluation Scope](#evaluation-scope)
- [Reporting & Code Availability](#reporting--code-availability)
- [Contact & Support](#contact--support)

---

## Overview

CellCMA is an autonomous multi-agent framework for single-cell analysis. It turns an analysis objective and one or more [AnnData](https://anndata.readthedocs.io/) inputs into a complete, executable workflow through **semantic data parsing**, **expert proposal debate**, **LLM-driven code generation**, **sandboxed execution**, **automatic repair**, **metric evaluation**, and **reusable memory**.

The goal is not to hard-code one best model for one benchmark. CellCMA is built as a *workflow construction layer*: it inspects the data, designs a task-specific plan, generates runnable Python, executes it against preloaded single-cell objects, evaluates the outcome, and iteratively improves the workflow.

| Challenge | CellCMA approach |
| --- | --- |
| High-dimensional and heterogeneous single-cell inputs | Converts AnnData matrices and metadata into compact semantic summaries for agent reasoning |
| Fragile prompt-to-code generation | Uses multi-expert debate before code generation and validation after execution |
| Long-running iterative workflows | Stores short-term execution state, episodic trajectories, and long-term successful patterns |
| Dataset-specific modeling choices | Selects architecture, losses, hyperparameters, and metrics based on parsed data context |
| Error-prone file loading in generated code | Injects data objects directly into the execution environment |
| Need for measurable outputs | Requires task-specific metrics and saves round-level results, blueprints, and generated code |

### Supported tasks

| Area | Supported workflow | Output |
| --- | --- | --- |
| Drug perturbation prediction | Predict transcriptional response under chemical or treatment perturbations | Predicted expression, fold changes, R²/MSE/PCC and DE-focused metrics |
| Genetic perturbation prediction | Model KO, KD, OE, CRISPR, guide-RNA, or combinatorial perturbation effects | Predicted expression response and perturbation-aware metrics |
| Multi-omics integration | Align scRNA-seq and scATAC-seq, including paired and unpaired settings | Shared latent embedding saved as `.h5ad` |
| GRN inference | Infer TF–target regulatory edges from RNA or RNA + ATAC evidence | Edge list CSV and optional ChIP-seq / BEELINE-style evaluation |
| Full pipeline | Compose integration, GRN inference, and perturbation workflows | Task-specific artifacts under `outputs/` |

---

## Key Features

CellCMA packages the modern single-cell analysis lifecycle into a reusable agent system built around four cooperating roles:

- **Architect** — designs candidate workflow blueprints from the parsed data context.
- **Scientist** — checks theoretical soundness and biological assumptions.
- **Engineer** — checks feasibility, runtime, and resource constraints.
- **Critic** — compares proposals against known method patterns and prior results.

Core modules:

1. **Semantic data parser** — samples cells and features and surfaces the *actual* `obs`, `var`, and `X` structure (not just statistics) to the LLM, together with hardware context.
2. **Multi-expert debate** — a four-role proposal-and-review loop with multi-round refinement.
3. **Hierarchical memory** — short-term execution state, long-term cross-session experience, and episodic experiment records.
4. **Data-injection execution** — AnnData objects are preloaded into the runtime so generated code focuses on analysis logic and avoids path/format loading bugs.

```text
User objective + AnnData inputs
        |
        v
Task Analysis
  - parses AnnData structure
  - detects cell type, batch, perturbation, and control labels
  - summarizes sparsity, dimensions, sequencing depth, and hardware context
        |
        v
Proposal Debate
  - Architect designs candidate workflow blueprints
  - Scientist checks theoretical soundness
  - Engineer checks feasibility and resource constraints
  - Critic compares against known method patterns and biological assumptions
        |
        v
Coding and Execution
  - generates Python analysis code
  - injects AnnData objects into the runtime
  - validates result completeness
  - repairs code using execution feedback
        |
        v
Memory and Optimization
  - records errors, metrics, parameters, and successful patterns
  - uses previous rounds to refine subsequent blueprints
        |
        v
Metrics, generated code, blueprints, integrated embeddings, GRN edge lists
```

---

## System Requirements

This section documents the hardware and software environment used to develop and test CellCMA. Matching it closely is the most reliable way to reproduce the reported results.

### Hardware requirements

CellCMA runs on a standard workstation. GPU acceleration is optional but recommended for the neural-network methods (e.g. scVI/scGLUE-style models) that the agent may generate for integration and GRN tasks.

| Component | Minimum | Recommended |
| --- | --- | --- |
| CPU | 4 cores, x86-64 | 8+ cores |
| RAM | 16 GB | 32 GB or more (depends on dataset size) |
| GPU | Not required (CPU-only PyTorch works) | NVIDIA GPU with ≥ 8 GB VRAM + CUDA 11.8/12.1 |
| Disk | ~10 GB (code + dependencies) | SSD, plus space for `.h5ad` datasets and `outputs/` |

> **Note.** The agent generates and executes analysis code at runtime. Peak memory is therefore determined by the dataset and the method the agent selects, not by CellCMA itself.

### Operating system

The development and test environment is **Linux x86-64** (Ubuntu-class). macOS and Windows are expected to work but are not the primary tested platform.

- **Linux** — x86-64, tested (this is the frozen environment)
- **macOS** — 13 (Ventura) and later, Intel and Apple silicon (expected to work)
- **Windows** — Windows 10 / 11, PowerShell (this repository's source machine)

### Software dependencies

CellCMA requires **Python 3.10 or newer** (developed and tested on **Python 3.12.3**). Dependencies split into two groups: the **framework core** that the CellCMA source imports directly, and the **single-cell analysis stack** that the LLM-generated analysis code calls at runtime. Both are needed to reproduce results. All versions below are the exact pinned versions from the frozen environment (see [`requirements.txt`](requirements.txt)).

#### Framework core (imported by the CellCMA source)

| Package | Version | Purpose |
| --- | --- | --- |
| `anndata` | 0.12.10 | `.h5ad` single-cell data structures |
| `numpy` | 2.1.3 | Numerical arrays |
| `pandas` | 2.3.3 | Metadata and result tables |
| `scipy` | 1.17.1 | Sparse matrices and statistics |
| `scikit-learn` | 1.8.0 | Clustering, metrics, dimensionality reduction |
| `imbalanced-learn` | 0.14.1 | Class-balanced sampling in generated workflows |
| `torch` | 2.7.0+cu128 | Backend for neural-network methods (CUDA 12.8 build) |
| `aiohttp` | 3.13.3 | Async LLM API calls |
| `requests` | 2.33.1 | PubMed / HTTP retrieval |
| `python-dotenv` | 1.2.2 | `.env` configuration loading |
| `biopython` | 1.86 | PubMed literature retrieval (Entrez) |
| `pyyaml` | 6.0.2 | YAML config parsing |
| `psutil` | 6.1.0 | Hardware / resource context for the parser |
| `matplotlib` | 3.9.2 | Plotting in generated analysis code |

#### Single-cell analysis stack (used by LLM-generated code)

The agent generates and executes analysis code at runtime; the packages below provide the methods (integration, GRN inference, perturbation, clustering, benchmarking) it may call.

| Package | Version | Purpose |
| --- | --- | --- |
| `scanpy` | 1.12 | scRNA-seq preprocessing and analysis |
| `scvi-tools` | 0.20.3 | scVI / multi-omics integration baselines |
| `scgpt` | 0.2.4 | scGPT foundation-model baseline |
| `scib` | 1.1.7 | Integration benchmark metrics |
| `mudata` | 0.3.3 | Multi-modal data containers |
| `leidenalg` | 0.11.0 | Graph clustering |
| `igraph` | 1.0.0 | Graph backend for clustering / GRN |
| `networkx` | 3.4.2 | Network construction and analysis |
| `umap-learn` | 0.5.12 | Embedding visualization |
| `rdkit` | 2026.3.1 | Drug / SMILES featurization (perturbation tasks) |
| `statsmodels` | 0.14.6 | Statistical testing in generated code |
| `seaborn` | 0.13.2 | Statistical plotting |
| `h5py` | 3.16.0 | HDF5 backend for `.h5ad` I/O |
| `pyarrow` | 24.0.0 | Efficient table I/O |
| `openpyxl` | 3.1.5 | Excel result export |

#### GPU / accelerator stack

The test environment uses **CUDA 12.8** with **cuDNN 9.7.1.26**. PyTorch is the `2.7.0+cu128` build; `jax`/`jaxlib` (0.10.0) and `flax` (0.12.7) back the scGPT path. A CPU-only PyTorch build runs without a GPU but is slower for deep-learning components.

> **Reproducibility tip.** [`requirements.txt`](requirements.txt) pins the exact versions above. For a byte-level record, also capture the full environment with `python -m pip freeze > requirements-frozen.txt` (or `conda env export --no-builds > environment.yml`) and store it alongside the run that produced your results.

---

## Repository Contents

```text
CellCMA/
|-- agent_sct/
|   |-- agents/                # Architect, Scientist, Engineer, Critic (base + expert agents)
|   |-- modules/
|   |   |-- data_parsing/      # AnnData semantic parser and literature retrieval
|   |   |-- debate/            # Multi-expert debate and multi-round refinement
|   |   |-- execution/         # Code generation and data-injection execution
|   |   `-- memory/            # Hierarchical memory system
|   `-- tasks/                 # Integration, GRN inference, perturbation tasks
|-- Agent/llm/                 # Unified async LLM client (OpenAI / Anthropic / Azure / local)
|-- scripts/run_agent_sct.py   # Main CLI entry point
|-- Utils/                     # Logging, PubMed search, benchmark utilities
|-- .env.example               # Example LLM configuration template
|-- requirements.txt           # Pinned dependencies (frozen environment)
|-- outputs/                   # Generated artifacts (ignored by git)
`-- README.md
```

---

## Installation

**1. Clone the repository.**

```bash
git clone https://github.com/ZhangLab312/cellcma-v1.git
cd cellcma-v1
```

**2. Create and activate a virtual environment.**

```bash
python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

**3. Upgrade pip and install PyTorch** matching your hardware (GPU optional).

```bash
python -m pip install -U pip setuptools wheel

# CUDA 12.8 build (matches the tested environment; Linux/Windows)
python -m pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# CPU-only alternative
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

**4. Install the remaining dependencies** from the pinned list.

```bash
python -m pip install -r requirements.txt
```

**5. Verify the installation.**

```bash
python scripts/run_agent_sct.py --help
```

For a different CUDA version, follow the official [PyTorch install selector](https://pytorch.org/get-started/locally/) and adjust the `torch` / `torchvision` build tags accordingly.

---

## LLM Configuration

CellCMA calls an LLM at runtime to debate, generate, and repair code. Create a local `.env` file from the example template:

```bash
cp .env.example .env   # Windows PowerShell: Copy-Item .env.example .env
```

Minimal configuration:

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
```

Supported provider modes include OpenAI-compatible APIs, Anthropic, Azure OpenAI, local vLLM / Ollama / LM Studio endpoints, and custom OpenAI-format services. See [`.env.example`](.env.example) for per-provider examples and programmatic model switching.

---

## Usage

All tasks share a single entry point, [`scripts/run_agent_sct.py`](scripts/run_agent_sct.py).

### Multi-omics integration

```bash
python scripts/run_agent_sct.py \
  --task integration \
  --rna data/rna.h5ad \
  --atac data/atac.h5ad \
  --output-dir outputs/integration
```

Expected artifact: `outputs/integration/integrated/`

### GRN inference from RNA

```bash
python scripts/run_agent_sct.py \
  --task grn \
  --data data/rna.h5ad \
  --chipseq data/dataset/chip_seq_TF/ \
  --output-dir outputs/grn
```

### GRN inference from RNA + ATAC

```bash
python scripts/run_agent_sct.py \
  --task grn \
  --rna data/rna.h5ad \
  --atac data/atac.h5ad \
  --prior data/prior_network.csv \
  --chipseq data/dataset/chip_seq_TF/ \
  --output-dir outputs/grn
```

Expected artifact: `outputs/grn_edge_list.csv`

### Drug perturbation prediction

```bash
python scripts/run_agent_sct.py \
  --task perturbation \
  --data data/drug_perturbation.h5ad \
  --mode drug \
  --output-dir outputs/perturbation
```

### Genetic perturbation prediction

```bash
python scripts/run_agent_sct.py \
  --task perturbation \
  --data data/genetic_perturbation.h5ad \
  --mode genetic \
  --output-dir outputs/genetic_perturbation
```

### Full pipeline

```bash
python scripts/run_agent_sct.py \
  --task full_pipeline \
  --rna data/rna.h5ad \
  --atac data/atac.h5ad \
  --data data/perturbation.h5ad \
  --output-dir outputs/full_pipeline
```

### Programmatic usage

```python
import asyncio

from agent_sct.tasks.integration.task_integration import run_integration
from agent_sct.tasks.grn_inference.task_grn import run_grn_inference
from agent_sct.tasks.perturbation.task_perturbation import run_perturbation


async def main():
    integration = await run_integration(
        rna_path="data/rna.h5ad",
        atac_path="data/atac.h5ad",
    )

    grn = await run_grn_inference(
        data_path="data/rna.h5ad",
        chipseq_ground_truth="data/dataset/chip_seq_TF/",
    )

    perturbation = await run_perturbation(
        data_path="data/drug_perturbation.h5ad",
        mode="drug",
    )

    print(integration.success, grn.success, perturbation.success)


asyncio.run(main())
```

---

## Input Data Expectations

CellCMA expects single-cell datasets in AnnData `.h5ad` format.

Recommended metadata fields:

| Concept | Common column names |
| --- | --- |
| Cell type | `cell_type`, `celltype`, `cellType`, `cell_line`, `cell_label` |
| Batch or sample | `batch`, `batch_id`, `sample`, `sample_id`, `library_id` |
| Perturbation | `perturbation`, `drug`, `treatment`, `condition`, `perturbed_gene`, `sgRNA`, `gene` |
| Control | labels containing `control`, `ctrl`, `vehicle`, `dmso`, `untreated`, `wt`, `neg_control` |

The parser attempts automatic detection. Clean and explicit metadata usually improves generated workflow quality.

---

## Output Artifacts

```text
outputs/
|-- blueprints/          # Selected workflow blueprints
|-- debate_outputs/      # Expert debate traces
|-- execution_results/   # Per-iteration execution summaries
|-- generated_code/      # Successful code plus iteration history
|-- integrated/          # Integrated AnnData embeddings
|-- round_results/       # Round-level metrics and feedback
|-- grn_edge_list.csv    # TF-target regulatory edge list
`-- grn_beeline_*.json   # Optional benchmark metrics
```

The memory system stores reusable experience outside the repository, by default at `~/.agent_sct/memory/`.

---

## Reproducibility

The following controls let you reproduce or ablate the workflows reported in the manuscript. Defaults match the paper's main runs.

```bash
# Increase outer experiment rounds
python scripts/run_agent_sct.py --task integration --rna data/rna.h5ad --atac data/atac.h5ad --max-rounds 10

# Increase inner code generation and repair attempts
python scripts/run_agent_sct.py --task grn --data data/rna.h5ad --max-iterations 8
```

Because the LLM is non-deterministic, set `LLM_TEMPERATURE=0` (or `--llm-temperature 0`) and pin the provider/model version for byte-level reproducibility. For all reported experiments, keep `.env`, the LLM model identifier, the dataset versions, and the captured `pip freeze` output alongside the saved run results.

---

## Evaluation Scope

CellCMA is designed for four major single-cell workflow families:

| Category | Example datasets | Representative baselines |
| --- | --- | --- |
| Drug perturbation | Sci-Plex, LINCS | scGen, CPA, chemCPA, Biolord, CondOT, scDPR, CellForge |
| Genetic perturbation | Adamson, Norman | scGen, CondOT, CPA, Biolord, scGPT, CellForge |
| Multi-omics integration | PBMC 10k, Buenrostro | Seurat v4, Harmony, scGPT, scVI |
| GRN inference | PBMC 10k, Buenrostro | scGLUE, LINGER, scMTNI, SCENIC, DeepSEM, scGPT, UnpairReg |

The framework evaluates generated workflows with task-specific metrics, including R², MSE, PCC, DE-focused perturbation metrics, NMI, ARI, ASW, kNN cross-modality mixing, AUROC, AUPRC, and F1.

---

## Reporting & Code Availability

- **Code availability.** All source code required to reproduce the analyses is provided in this repository at <https://github.com/ZhangLab312/cellcma-v1>. The main entry point is [`scripts/run_agent_sct.py`](scripts/run_agent_sct.py).
- **Reporting summary.** Hardware, software versions, and key hyperparameters are documented above in [System Requirements](#system-requirements) and [Reproducibility](#reproducibility). Please retain the `.env`, dataset versions, and `pip freeze` output used for each reported run.

---

## Contact & Support

- **Issues.** Please use the [GitHub issue tracker](https://github.com/ZhangLab312/cellcma-v1/issues) for bug reports and reproducibility questions.
- **Development notes.**
  - Keep `.env`, API keys, datasets, model weights, logs, and generated outputs out of git.
  - Place large `.h5ad` datasets under local `data/` paths rather than committing them.
  - Generated code and debate traces are useful for debugging but are run artifacts.
