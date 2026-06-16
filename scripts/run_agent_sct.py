# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unified entry script supporting three main tasks:
- integration: Single-cell multi-omics integration
- grn: Gene regulatory network inference
- perturbation: Cell perturbation prediction
- full_pipeline: Full pipeline
Usage:
    python scripts/run_agent_sct.py --task integration --rna data_rna.h5ad --atac data_atac.h5ad
    python scripts/run_agent_sct.py --task grn --data data.h5ad
    python scripts/run_agent_sct.py --task perturbation --data data.h5ad --mode drug
"""


import sys
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file first (before all imports)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

import argparse
import asyncio
import logging
from typing import Optional

# Import Agent-SCT modules
from agent_sct.tasks.integration.task_integration import IntegrationTask, run_integration
from agent_sct.tasks.grn_inference.task_grn import GRNInferenceTask, run_grn_inference
from agent_sct.tasks.perturbation.task_perturbation import PerturbationTask, run_perturbation
from agent_sct.tasks.base_task import TaskConfig, TaskResult

# Import LLM configuration
from Agent.llm.llm_client import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Agent-SCT: AI Agent for Single-Cell Multi-omics Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-omics integration
  python scripts/run_agent_sct.py --task integration --rna data_rna.h5ad --atac data_atac.h5ad
  
  # GRN inference (single RNA data)
  python scripts/run_agent_sct.py --task grn --data data.h5ad
  
  # GRN inference (multi-omics: RNA + ATAC)
  python scripts/run_agent_sct.py --task grn --rna data_rna.h5ad --atac data_atac.h5ad
  
  # Perturbation prediction
  python scripts/run_agent_sct.py --task perturbation --data data.h5ad --mode drug
  
  # Full pipeline
  python scripts/run_agent_sct.py --task full_pipeline --rna data_rna.h5ad --atac data_atac.h5ad
        """
    )
    
    # Task type
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['integration', 'grn', 'perturbation', 'full_pipeline'],
        help='Task type to execute'
    )
    
    # Data paths
    parser.add_argument(
        '--rna',
        type=str,
        help='Path to scRNA-seq data (.h5ad)'
    )
    parser.add_argument(
        '--atac',
        type=str,
        help='Path to scATAC-seq data (.h5ad)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to single data file (.h5ad)'
    )
    
    # Task-specific parameters
    parser.add_argument(
        '--mode',
        type=str,
        default='drug',
        choices=['drug', 'genetic', 'compound'],
        help='Perturbation mode (for perturbation task)'
    )
    parser.add_argument(
        '--prior',
        type=str,
        help='Path to prior network file (for GRN task, used as prior knowledge)'
    )
    parser.add_argument(
        '--chipseq',
        type=str,
        help='Path to ChIP-seq ground truth network file (for GRN task, used for evaluation)'
    )
    
    # Configuration parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations for code execution (within each round)'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=5,
        help='Maximum rounds for multi-round experiment iteration'
    )
    parser.add_argument(
        '--no-debate',
        action='store_true',
        help='Disable multi-expert debate'
    )
    parser.add_argument(
        '--no-literature',
        action='store_true',
        help='Disable literature retrieval'
    )
    parser.add_argument(
        '--no-memory',
        action='store_true',
        help='Disable memory system'
    )
    
    # Data sampling parameters
    parser.add_argument(
        '--sample-n-obs',
        type=int,
        default=100,
        help='Number of cells to sample for preview'
    )
    parser.add_argument(
        '--sample-n-vars',
        type=int,
        default=50,
        help='Number of features to sample for preview'
    )
    
    # LLM parameters
    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='LLM model to use (overrides .env config)'
    )
    parser.add_argument(
        '--llm-temperature',
        type=float,
        default=0.7,
        help='LLM temperature'
    )
    
    return parser


def build_task_config(args) -> TaskConfig:
    """Build task configuration from command-line arguments"""
    # Load LLM configuration from environment variables
    llm_config = LLMConfig.from_env()
    # Override model configuration if provided via command-line arguments
    if args.llm_model:
        llm_config.model = args.llm_model
    if args.llm_temperature:
        llm_config.temperature = args.llm_temperature
    
    return TaskConfig(
        task_type=args.task,
        enable_debate=not args.no_debate,
        enable_literature_retrieval=not args.no_literature,
        enable_memory=not args.no_memory,
        max_iterations=args.max_iterations,
        max_rounds=args.max_rounds,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_config=llm_config,
        sample_n_obs=args.sample_n_obs,
        sample_n_vars=args.sample_n_vars,
    )


async def run_integration_task(args) -> TaskResult:
    """Run multi-omics integration task"""
    logger.info("="*60)
    logger.info("Running Multi-Omics Integration Task")
    logger.info("="*60)
    
    if not args.rna and not args.atac:
        raise ValueError("At least one of --rna or --atac must be provided")
    
    config = build_task_config(args)
    task = IntegrationTask(config)
    
    result = await task.execute(
        rna_path=args.rna,
        atac_path=args.atac,
    )
    
    return result


async def run_grn_task(args) -> TaskResult:
    """Run GRN inference task"""
    logger.info("="*60)
    logger.info("Running GRN Inference Task")
    logger.info("="*60)
    
    has_multi = args.rna and args.atac
    if not has_multi and not args.data:
        raise ValueError("At least one of --data or (--rna + --atac) must be provided for GRN task")
    
    config = build_task_config(args)
    task = GRNInferenceTask(config)
    
    result = await task.execute(
        rna_path=args.rna,
        atac_path=args.atac,
        data_path=args.data,
        prior_network=args.prior,
        chipseq_ground_truth=args.chipseq,
    )
    
    return result


async def run_perturbation_task(args) -> TaskResult:
    """Run perturbation prediction task"""
    logger.info("="*60)
    logger.info("Running Perturbation Prediction Task")
    logger.info("="*60)
    
    if not args.data:
        raise ValueError("--data must be provided for perturbation task")
    
    config = build_task_config(args)
    task = PerturbationTask(config)
    
    result = await task.execute(
        data_path=args.data,
        mode=args.mode,
    )
    
    return result


async def run_full_pipeline(args) -> list[TaskResult]:
    """Run the full pipeline"""
    logger.info("="*60)
    logger.info("Running Full Pipeline")
    logger.info("="*60)
    
    results = []
    
    # Step 1: Multi-omics integration
    if args.rna or args.atac:
        logger.info("\n>>> Step 1: Multi-omics Integration")
        result = await run_integration_task(args)
        results.append(result)
        
        if not result.success:
            logger.error("Integration failed, stopping pipeline")
            return results
    
    # Step 2: GRN inference (on integrated data or provided data)
    logger.info("\n>>> Step 2: GRN Inference")
    grn_args = argparse.Namespace(**vars(args))
    if not grn_args.data and results:
        # Use output from integration
        logger.info("Using integrated data for GRN inference")
    result = await run_grn_task(grn_args)
    results.append(result)
    
    # Step 3: Perturbation prediction
    logger.info("\n>>> Step 3: Perturbation Prediction")
    pert_args = argparse.Namespace(**vars(args))
    result = await run_perturbation_task(pert_args)
    results.append(result)
    
    return results


async def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Execute based on task type
        if args.task == 'integration':
            result = await run_integration_task(args)
            results = [result]
        elif args.task == 'grn':
            result = await run_grn_task(args)
            results = [result]
        elif args.task == 'perturbation':
            result = await run_perturbation_task(args)
            results = [result]
        elif args.task == 'full_pipeline':
            results = await run_full_pipeline(args)
        else:
            raise ValueError(f"Unknown task: {args.task}")
        
        # Print results summary
        logger.info("\n" + "="*60)
        logger.info("Task Execution Summary")
        logger.info("="*60)
        
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            logger.info(f"\nTask {i}: {result.task_type}")
            logger.info(f"  Status: {status}")
            logger.info(f"  Metrics: {result.metrics}")
            if result.error:
                logger.error(f"  Error: {result.error}")
        
        # Overall status
        all_success = all(r.success for r in results)
        if all_success:
            logger.info("\n🎉 All tasks completed successfully!")
            return 0
        else:
            logger.error("\n⚠️ Some tasks failed")
            return 1
            
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
