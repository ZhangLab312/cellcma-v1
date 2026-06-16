# CellCMA

<p align="center">
  <b>Collaborative Multi-agent Analysis for Single-cell Tasks</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="AnnData" src="https://img.shields.io/badge/Data-AnnData%20.h5ad-green">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-LLM%20Multi--Agent-purple">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Perturbation%20%7C%20Integration%20%7C%20GRN-lightgrey">
</p>

CellCMA is an autonomous multi-agent framework for single-cell analysis. It turns an analysis objective and one or more AnnData inputs into a complete, executable workflow through semantic data parsing, expert proposal debate, LLM-driven code generation, sandboxed execution, automatic repair, metric evaluation, and reusable memory.

The goal is not to hard-code one best model for one benchmark. CellCMA is built as a workflow construction layer: it inspects the data, designs a task-specific plan, generates runnable Python, executes it against preloaded single-cell objects, evaluates the outcome, and iteratively improves the workflow.

## Why CellCMA

Modern single-cell projects often require more than a single script or a single model. Researchers need to inspect heterogeneous data, select a modeling strategy, handle noisy metadata, run iterative experiments, debug failures, and preserve lessons across tasks. CellCMA packages that process into a reusable agent system.

| Challenge | CellCMA approach |
| --- | --- |
| High-dimensional and heterogeneous single-cell inputs | Converts AnnData matrices and metadata into compact semantic summaries for agent reasoning |
| Fragile prompt-to-code generation | Uses multi-expert debate before code generation and validation after execution |
| Long-running iterative workflows | Stores short-term execution state, episodic trajectories, and long-term successful patterns |
| Dataset-specific modeling choices | Selects architecture, losses, hyperparameters, and metrics based on parsed data context |
| Error-prone file loading in generated code | Injects data objects directly into the execution environment |
| Need for measurable outputs | Requires task-specific metrics and saves round-level results, blueprints, and generated code |

## Core Capabilities

| Area | Supported workflow | Output |
| --- | --- | --- |
| Drug perturbation prediction | Predict transcriptional response under chemical or treatment perturbations | Predicted expression, fold changes, R2/MSE/PCC and DE-focused metrics |
| Genetic perturbation prediction | Model KO, KD, OE, CRISPR, guide-RNA, or combinatorial perturbation effects | Predicted expression response and perturbation-aware metrics |
| Multi-omics integration | Align scRNA-seq and scATAC-seq, including paired and unpaired settings | Shared latent embedding saved as `.h5ad` |
| GRN inference | Infer TF-target regulatory edges from RNA or RNA + ATAC evidence | Edge list CSV and optional ChIP-seq/BEELINE-style evaluation |
| Full pipeline | Compose integration, GRN inference, and perturbation workflows | Task-specific artifacts under `outputs/` |

## System Design

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

## Repository Layout

```text
CellCMA/
|-- agent_sct/
|   |-- agents/                # Architect, Scientist, Engineer, Critic
|   |-- modules/
|   |   |-- data_parsing/      # AnnData semantic parser and literature retrieval
|   |   |-- debate/            # Multi-expert debate and multi-round refinement
|   |   |-- execution/         # Code generation and data-injection execution
|   |   `-- memory/            # Hierarchical memory system
|   `-- tasks/                 # Integration, GRN inference, perturbation tasks
|-- Agent/llm/                 # Unified LLM client
|-- scripts/run_agent_sct.py   # Main CLI entry point
|-- Utils/                     # Logging, PubMed search, benchmark utilities
|-- examples/                  # Minimal usage examples
|-- experiments/               # Benchmarking, plotting, and experiment utilities
`-- outputs/                   # Generated artifacts, ignored by git
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ZhangLab312/cellcma-v1.git
cd cellcma-v1

python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install anndata numpy pandas scipy scikit-learn imbalanced-learn torch aiohttp python-dotenv biopython pyyaml psutil matplotlib
```

For GPU acceleration, install the PyTorch build that matches your CUDA environment.

## LLM Configuration

Create a local `.env` file from the example template:

```bash
cp .env.example .env
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

Supported provider modes include OpenAI-compatible APIs, Anthropic, Azure OpenAI, local vLLM/Ollama/LM Studio endpoints, and custom OpenAI-format services.

## Quick Start

### Multi-omics integration

```bash
python scripts/run_agent_sct.py \
  --task integration \
  --rna data/rna.h5ad \
  --atac data/atac.h5ad \
  --output-dir outputs/integration
```

Expected artifact:

```text
outputs/integration/integrated/
```

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

Expected artifact:

```text
outputs/grn_edge_list.csv
```

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

## Advanced Runtime Controls

```bash
# Increase outer experiment rounds
python scripts/run_agent_sct.py --task integration --rna data/rna.h5ad --atac data/atac.h5ad --max-rounds 10

# Increase inner code generation and repair attempts
python scripts/run_agent_sct.py --task grn --data data/rna.h5ad --max-iterations 8

# Run targeted ablations
python scripts/run_agent_sct.py --task perturbation --data data/demo.h5ad --no-debate
python scripts/run_agent_sct.py --task perturbation --data data/demo.h5ad --no-literature
python scripts/run_agent_sct.py --task perturbation --data data/demo.h5ad --no-memory

# Control the data preview exposed to the LLM
python scripts/run_agent_sct.py --task integration --rna data/rna.h5ad --sample-n-obs 200 --sample-n-vars 100
```

## Programmatic Usage

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

## Output Artifacts

```text
outputs/
|-- blueprints/          # Selected workflow blueprints
|-- debate_outputs/     # Expert debate traces
|-- execution_results/  # Per-iteration execution summaries
|-- generated_code/     # Successful code plus iteration history
|-- integrated/         # Integrated AnnData embeddings
|-- round_results/      # Round-level metrics and feedback
|-- grn_edge_list.csv   # TF-target regulatory edge list
`-- grn_beeline_*.json  # Optional benchmark metrics
```

The memory system stores reusable experience outside the repository:

```text
~/.agent_sct/memory/
```

## Evaluation Scope

CellCMA is designed for four major single-cell workflow families:

| Category | Example datasets | Representative baselines |
| --- | --- | --- |
| Drug perturbation | Sci-Plex, LINCS | scGen, CPA, chemCPA, Biolord, CondOT, scDPR, CellForge |
| Genetic perturbation | Adamson, Norman | scGen, CondOT, CPA, Biolord, scGPT, CellForge |
| Multi-omics integration | PBMC 10k, Buenrostro | Seurat v4, Harmony, scGPT, scVI |
| GRN inference | PBMC 10k, Buenrostro | scGLUE, LINGER, scMTNI, SCENIC, DeepSEM, scGPT, UnpairReg |

The framework evaluates generated workflows with task-specific metrics, including R2, MSE, PCC, DE-focused perturbation metrics, NMI, ARI, ASW, kNN cross-modality mixing, AUROC, AUPRC, and F1.

## Development Notes

- Keep `.env`, API keys, datasets, model weights, logs, and generated outputs out of git.
- Place large `.h5ad` datasets under local `data/` paths rather than committing them.
- Generated code and debate traces are useful for debugging, but they are usually run artifacts.
- Add a `LICENSE` file before public release if the repository is intended for external reuse.
