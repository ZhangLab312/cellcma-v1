"""
Base Task - Base Task Class

Base class for three main tasks (integration, GRN inference, perturbation prediction)
Provides a unified workflow interface using LLM to dynamically generate code
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad


def _make_serializable(obj: Any) -> Any:
    """Convert numpy/numpy-like types and special objects to JSON-serializable types"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, 'item'):  # numpy types like float32, int64
        return obj.item()
    if hasattr(obj, '__iter__') and not isinstance(obj, (bytes, memoryview)):
        try:
            return [_make_serializable(i) for i in obj]
        except (TypeError, AttributeError):
            pass
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    # fallback: convert to string
    return str(obj)

# Import core modules
from ..modules.data_parsing.semantic_parser import (
    SemanticDataParser, MultiOmicsSemanticParser, DataSemantics
)
from ..modules.data_parsing.literature_retrieval import (
    LiteratureRetriever, get_literature_retriever
)
from ..modules.debate.multi_expert_debate import (
    MultiExpertDebateSystem, DebateResult
)
from ..modules.memory.hierarchical_memory import (
    HierarchicalMemorySystem, get_memory_system, ShortTermState
)
from ..modules.execution.data_injection_executor import (
    DataInjectionExecutor, ExecutionResult
)
from ..modules.execution.llm_code_generator import (
    LLMCodeGenerator, CodeGenerationResult
)

logger = logging.getLogger(__name__)


def _check_embedding_quality(execution_results: List[ExecutionResult]) -> bool:
    """Check if the latent embeddings are valid (not collapsed).

    Returns False if embeddings show signs of latent collapse:
    - Near-zero variance across all dimensions
    - All identical values (degenerate embedding)
    """
    import numpy as np

    for r in execution_results:
        if not r.success:
            continue
        # Check return_value for latent_representation
        if r.return_value and isinstance(r.return_value, dict):
            latent = r.return_value.get('latent_representation')
            if latent is not None and isinstance(latent, np.ndarray) and latent.ndim == 2:
                var_per_dim = np.var(latent, axis=0)
                mean_var = np.mean(var_per_dim)
                if mean_var < 1e-4:
                    logger.warning(f"Latent collapse detected: mean variance = {mean_var:.6f}")
                    return False
        # Check variables dict
        if r.variables:
            result_var = r.variables.get('result')
            if isinstance(result_var, dict):
                latent = result_var.get('latent_representation')
                if latent is not None and isinstance(latent, np.ndarray) and latent.ndim == 2:
                    var_per_dim = np.var(latent, axis=0)
                    mean_var = np.mean(var_per_dim)
                    if mean_var < 1e-4:
                        logger.warning(f"Latent collapse detected: mean variance = {mean_var:.6f}")
                        return False
    return True


def _is_catastrophic_round(
    curr_metrics: Dict[str, Any],
    best_metrics: Dict[str, Any],
) -> bool:
    """Detect if a round's metrics represent a catastrophic failure.

    A round is catastrophic if:
    - ASW_celltype < 0 (embeddings are worse than random)
    - NMI < 0.5 (when best NMI > 0.8)
    - KNN_cross < 0.01 (almost no cross-modal mixing)
    - More than 50% drop from best metrics
    """
    computed = curr_metrics.get('computed', {}) if isinstance(curr_metrics, dict) else {}
    best_computed = best_metrics.get('computed', {}) if isinstance(best_metrics, dict) else best_metrics

    if not computed:
        return True

    asw_ct = computed.get('asw_celltype', 0) or 0
    nmi = computed.get('nmi', 0) or 0
    knn = computed.get('knn_cross', 0) or 0

    # Absolute thresholds
    if asw_ct < 0:
        return True
    if nmi < 0.3:
        return True
    if knn < 0.001 and (best_computed.get('knn_cross', 0) or 0) > 0.1:
        return True

    # Relative drop from best
    if best_computed:
        best_nmi = best_computed.get('nmi', 0) or 0
        best_ari = best_computed.get('ari', 0) or 0
        if best_nmi > 0.5 and nmi < best_nmi * 0.5:
            return True
        ari = computed.get('ari', 0) or 0
        if best_ari > 0.3 and ari < best_ari * 0.4:
            return True

    return False


@dataclass
class TaskResult:
    """Task execution result"""
    success: bool
    task_type: str
    output_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    debate_result: Optional[DebateResult] = None
    error: str = ""
    generated_codes: List[str] = field(default_factory=list)  # Record generated code
    round_results: List[Any] = field(default_factory=list)  # Multi-round iteration results
    best_round: int = 0  # Best round
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'task_type': self.task_type,
            'output_path': self.output_path,
            'metrics': self.metrics,
            'error': self.error,
            'n_code_iterations': len(self.generated_codes),
            'n_rounds': len(self.round_results),
            'best_round': self.best_round,
        }


@dataclass
class TaskConfig:
    """Task configuration"""
    task_type: str
    enable_debate: bool = True
    enable_literature_retrieval: bool = True
    enable_memory: bool = True
    # Convergence condition configuration
    max_iterations: int = 5  # Maximum iterations for code generation and correction
    max_rounds: int = 5  # Maximum rounds for multi-round experiment iteration (each round includes debate + code generation + execution)
    improvement_threshold: float = 0.05  # Relative improvement threshold (stop if improvement is below this value)
    metric_goals: Optional[Dict[str, float]] = None  # Absolute goals {"R2": 0.8, "MSE": 0.1}
    metric_directions: Optional[Dict[str, str]] = None  # Metric directions {"R2": "higher", "MSE": "lower"}
    
    output_dir: str = "./outputs"
    
    # Memory control configuration
    max_code_history: int = 10  # Keep the last N code entries, discard older ones
    max_result_history: int = 5  # Keep the last N round results (only keep detailed results for the best round)
    
    # LLM configuration
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_config: Optional[Any] = None
    
    # Data configuration
    sample_n_obs: int = 100
    sample_n_vars: int = 50


# Metric direction definitions (used for normalization during best-round selection)
_METRIC_DIRECTIONS = {
    'mse': 'lower', 'mse_de': 'lower', 'mae': 'lower', 'rmse': 'lower',
    'r2': 'higher', 'r2_de': 'higher',
    'pcc': 'higher', 'pcc_fc': 'higher', 'pcc_de': 'higher', 'pcc_logfc': 'higher',
    'nmi': 'higher', 'ari': 'higher', 'asw_celltype': 'higher',
    'asw_batch': 'higher', 'knn_cross': 'higher',
    'auroc': 'higher', 'auprc': 'higher', 'f1': 'higher',
}


def _compute_normalized_avg(metrics: Dict[str, Any]) -> float:
    """Normalized average of metrics for fair comparison across rounds.

    - higher-is-better metrics keep original value
    - lower-is-better metrics are negated (smaller MSE is better -> negated so larger is better)
    """
    vals = []
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        direction = _METRIC_DIRECTIONS.get(k, 'higher')
        vals.append(-v if direction == 'lower' else v)
    return sum(vals) / len(vals) if vals else -float('inf')


class BaseTask(ABC):
    """
    Base task class

    Unified task execution workflow:
    1. Data parsing (semantic)
    2. Literature retrieval (optional)
    3. Multi-expert debate (optional)
    4. LLM code generation
    5. Code execution
    6. LLM code correction on error
    7. Memory update
    """
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.task_type = config.task_type
        
        # Initialize core modules
        self.debate_system = MultiExpertDebateSystem(
            llm_config=config.llm_config,
            output_dir=self.config.output_dir,
            task_type=self.task_type,
        ) if config.enable_debate else None
        self.memory_system = get_memory_system() if config.enable_memory else None
        self.literature_retriever = get_literature_retriever() if config.enable_literature_retrieval else None
        self.executor = DataInjectionExecutor(timeout=600)
        self.code_generator = LLMCodeGenerator(llm_config=config.llm_config)  # LLM code generator
        
        # Task state
        self.data_info: Optional[Dict[str, Any]] = None
        self.semantics: Optional[DataSemantics] = None
        self.data_name: Optional[str] = None  # Dataset name (used for file naming)
        self.iteration: int = 0
        self.current_code: str = ""  # Current code
        
        logger.info(f"Task initialized: {self.task_type}")
    
    async def execute(self, **kwargs) -> TaskResult:
        """
        Execute main task workflow (supports multi-round iteration)

        Multi-round iteration workflow:
        1. Each round: debate -> code generation -> execution -> results fed back to architect
        2. Architect analyzes results and improves blueprint
        3. Repeat until max_rounds or convergence

        Returns:
            TaskResult: Task execution result
        """
        all_generated_codes = []
        # Record key metrics (avoid accumulating large objects)
        all_round_summaries = []  # Only save summary info
        final_success = False
        best_round = 0
        best_metrics = {}
        best_blueprint = None  # Track best blueprint for revert
        
        previous_rounds = []
        try:
            # Step 1: Parse data (execute only once)
            logger.info("="*60)
            logger.info("Step 1: Parsing data semantics...")
            logger.info("="*60)
            self._parse_data(**kwargs)
            
            # Step 2: Retrieve literature (execute only once)
            literature_context = ""
            if self.literature_retriever:
                logger.info("Step 2: Retrieving literature...")
                literature_context = self._retrieve_literature()
            
            # Initialize multi-round iteration system
            from ..modules.debate.multi_expert_debate import MultiRoundExperimentSystem, IterationRound
            
            multi_round_system = None
            current_blueprint = None
            
            # ===== Multi-round experiment iteration =====
            for round_num in range(1, self.config.max_rounds + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"EXPERIMENT ROUND {round_num}/{self.config.max_rounds}")
                logger.info(f"{'='*60}")
                
                self.code_generator.reset_fix_count()
                self.code_generator.reset_history()

                should_stop = False
                round_generated_codes = []
                round_execution_results = []
                
                # Step 3: Multi-expert debate (may re-debate or improve based on previous round)
                blueprint = None
                debate_result = None
                if self.debate_system:
                    if round_num == 1:
                        # First round: generate initial blueprint
                        logger.info(f"Round {round_num}: Running initial multi-expert debate...")
                        debate_result = await self._run_debate(literature_context)
                        blueprint = debate_result.final_blueprint
                        current_blueprint = blueprint
                    else:
                        # Subsequent rounds: architect improves blueprint based on previous results
                        logger.info(f"Round {round_num}: Architect improving blueprint based on previous results...")
                        if multi_round_system is None:
                            multi_round_system = MultiRoundExperimentSystem(
                                debate_system=self.debate_system,
                                llm_config=self.config.llm_config,
                                max_rounds=self.config.max_rounds,
                            )
                        
                        # Get results from the previous round
                        last_round = previous_rounds[-1]
                        improved_blueprint, architect_feedback = await multi_round_system.analyze_and_improve(
                            task_type=self.task_type,
                            data_info=self.data_info or {},
                            current_blueprint=current_blueprint,
                            execution_result={
                                'success': last_round.execution_result.get('success', False),
                                'metrics': last_round.metrics,
                                'output': last_round.execution_result.get('output', '')[:1000],
                                'error': last_round.execution_result.get('error', ''),
                            },
                            previous_rounds=previous_rounds[:-1],
                        )
                        blueprint = improved_blueprint
                        current_blueprint = improved_blueprint
                        logger.info(f"Blueprint improved for round {round_num}")
                
                # Step 4-6: LLM code generation, execution, correction
                logger.info(f"Round {round_num}: Generating, executing, and fixing code...")
                execution_results = await self._execute_with_llm_iterations(
                    blueprint=blueprint,
                    max_iterations=self.config.max_iterations,
                    previous_rounds=previous_rounds,
                )
                
                # Collect generated code (limit count to prevent memory accumulation)
                for result in execution_results:
                    if hasattr(result, 'code'):
                        round_generated_codes.append(result.code)
                        all_generated_codes.append(result.code)
                
                # Limit the size of all_generated_codes
                if len(all_generated_codes) > self.config.max_code_history:
                    all_generated_codes[:] = all_generated_codes[-self.config.max_code_history:]
                
                # Compute metrics for this round
                round_metrics = self._compute_metrics(execution_results)
                
                # Determine if this round succeeded
                # Check not only if code 'executed successfully' but also if metrics were computed
                # If code passes but is rejected by completeness validator (incomplete_execution=True), also mark as failure
                has_valid_result = any(
                    r.success and not r.incomplete_reason
                    for r in execution_results
                )
                round_success = has_valid_result
                
                # Record summary info for this round (don't save large objects)
                round_summary = {
                    'round_num': round_num,
                    'success': round_success,
                    'metrics': round_metrics.get('computed', {}),
                    'n_iterations': len(execution_results),
                    'incomplete_execution': round_metrics.get('incomplete_execution', False),
                    'incomplete_reason': round_metrics.get('incomplete_reason', ''),
                }
                all_round_summaries.append(round_summary)

                # ===== Stability Check 1: Embedding quality =====
                if round_success and not _check_embedding_quality(execution_results):
                    logger.warning(f"Round {round_num}: Latent collapse detected! Marking as failure.")
                    round_success = False
                    round_summary['success'] = False
                    round_summary['metrics']['_collapse_detected'] = True
                    # Revert to best blueprint if available
                    if best_blueprint is not None:
                        logger.info(f"Round {round_num}: Reverting to best blueprint from round {best_round}")
                        current_blueprint = best_blueprint.copy()

                # ===== Stability Check 2: Catastrophic round detection =====
                computed_metrics = round_metrics.get('computed', {})
                if round_success and computed_metrics and best_metrics:
                    if _is_catastrophic_round(round_metrics, best_metrics):
                        logger.warning(f"Round {round_num}: Catastrophic performance drop detected! Discarding round.")
                        round_summary['success'] = False
                        round_summary['metrics']['_catastrophic'] = True
                        # Revert blueprint
                        if best_blueprint is not None:
                            logger.info(f"Round {round_num}: Reverting to best blueprint from round {best_round}")
                            current_blueprint = best_blueprint.copy()

                # ===== Track best metrics and blueprint =====
                if round_success and computed_metrics:
                    curr_avg = _compute_normalized_avg(computed_metrics)
                    best_avg = -float('inf')
                    if best_metrics:
                        best_computed = best_metrics.get('computed', best_metrics)
                        best_avg = _compute_normalized_avg(best_computed)
                    if curr_avg > best_avg:
                            best_metrics = {'computed': computed_metrics.copy()}
                            best_round = round_num
                            if blueprint:
                                best_blueprint = blueprint.copy()
                            logger.info(f"Round {round_num}: New best metrics (avg={curr_avg:.4f})")

                round_result = IterationRound(
                    round_num=round_num,
                    blueprint=blueprint.copy() if blueprint else {},
                    execution_result={
                        'success': round_success,
                        'metrics': round_metrics,
                        'n_iterations': len(execution_results),
                        'incomplete_execution': round_metrics.get('incomplete_execution', False),
                        'incomplete_reason': round_metrics.get('incomplete_reason', ''),
                    },
                    metrics=round_metrics,
                    architect_feedback="Initial round" if round_num == 1 else "Improved based on previous results",
                )
                previous_rounds.append(round_result)
                
                if round_metrics.get('incomplete_execution'):
                    logger.warning(f"Round {round_num} completed: success=False (incomplete code detected: {round_metrics.get('incomplete_reason', '')[:100]})")
                else:
                    logger.info(f"Round {round_num} completed: success={round_success}, metrics={round_metrics.get('computed', {})}")
                
                # Save successful code from this round
                if round_success and round_generated_codes:
                    self._save_successful_code(round_generated_codes, execution_results, round_num)

                # Merge task-specific post-evaluation metrics (e.g., BEELINE) back into round_metrics
                extra_metrics = getattr(self, 'last_beeline_metrics', {})
                if extra_metrics:
                    if 'computed' not in round_metrics:
                        round_metrics['computed'] = {}
                    round_metrics['computed'].update(extra_metrics)
                    if all_round_summaries:
                        all_round_summaries[-1]['metrics'].update(extra_metrics)

                # Save blueprint for this round
                if blueprint:
                    self._save_blueprint(blueprint, round_num, round_metrics)
                
                # Run garbage collection periodically to prevent memory accumulation
                import gc
                gc.collect()
                
                # Initialize convergence check variables
                should_stop = False

                # Check convergence conditions (early stopping disabled)
                if round_num > 1 and multi_round_system:
                    prev_metrics = previous_rounds[-2].metrics.get('computed', {})
                    curr_metrics = round_metrics.get('computed', {})
                    improvement = multi_round_system.calculate_improvement(
                        curr_metrics, prev_metrics,
                        {k: v == 'higher' for k, v in self.config.metric_directions.items()} if self.config.metric_directions else None,
                    )
                    logger.info(f"Improvement from previous round: {improvement:.2%}")
                    # Early stopping disabled — always run all rounds
                    # if improvement < self.config.improvement_threshold:
                    #     logger.info(f"Improvement {improvement:.2%} below threshold {self.config.improvement_threshold:.2%}, stopping early.")
                    #     should_stop = True

                # Check if absolute metric goals are reached
                if self.config.metric_goals and round_metrics.get('computed'):
                    goals_met, goal_report = multi_round_system.check_metric_goals(
                        round_metrics['computed'],
                        self.config.metric_goals,
                        self.config.metric_directions,
                    )
                    if goals_met:
                        logger.info(f"Metric goals reached: {goal_report}")
                        should_stop = True

                if should_stop:
                    break

                # Update memory first (needs execution_results), then persist to file
                await self._update_memory(
                    execution_results, 
                    round_num=round_num, 
                    round_results=previous_rounds if round_num == self.config.max_rounds or should_stop else None,
                    blueprint=blueprint,
                    parameters={
                        'architecture_type': blueprint.get('architecture_type') if blueprint else None,
                        'latent_dim': blueprint.get('latent_dim') if blueprint else None,
                        'encoder_config': blueprint.get('encoder_config') if blueprint else None,
                        'decoder_config': blueprint.get('decoder_config') if blueprint else None,
                        'loss_functions': blueprint.get('loss_functions') if blueprint else None,
                        'batch_size': blueprint.get('batch_size_recommendation') if blueprint else None,
                    } if blueprint else {},
                )
                
                # Persist execution results to file and free memory
                self._save_execution_results(execution_results, round_num)
                self._save_round_result(round_result)
                
                # Check convergence conditions (early exit if any condition is met)
                if round_num > 1 and multi_round_system:
                    prev_metrics = previous_rounds[-2].metrics.get('computed', {})
                    curr_metrics = round_metrics.get('computed', {})
                    improvement = multi_round_system.calculate_improvement(curr_metrics, prev_metrics)
                    logger.info(f"Improvement from previous round: {improvement:.2%}")
            
            # Step 7: Memory already updated in each round loop, no redundant update here
            
            # Build final result
            final_success = any(r['success'] for r in all_round_summaries)
            best_round = self._find_best_round_from_summaries(all_round_summaries)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"All {self.config.max_rounds} rounds completed")
            logger.info(f"Final success: {final_success}")
            logger.info(f"Best round: {best_round}")
            logger.info(f"{'='*60}")
            
            return TaskResult(
                success=final_success,
                task_type=self.task_type,
                execution_results=[],  # Already saved to file
                debate_result=debate_result,
                metrics={'round_summaries': all_round_summaries},
                generated_codes=all_generated_codes,
                round_results=previous_rounds,  # Keep previous_rounds for final return
                best_round=best_round,
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return TaskResult(
                success=False,
                task_type=self.task_type,
                error=str(e),
                generated_codes=all_generated_codes,
                execution_results=[],
                round_results=previous_rounds,
            )
        finally:
            # Close LLM session, release aiohttp resources
            close_tasks = []
            
            # Close code_generator session
            if hasattr(self, 'code_generator'):
                close_tasks.append(self.code_generator.close())
            
            # Close all expert agent sessions in debate_system
            if self.debate_system is not None:
                for expert in [
                    self.debate_system.architect,
                    self.debate_system.scientist,
                    self.debate_system.engineer,
                    self.debate_system.critic,
                ]:
                    if expert is not None and hasattr(expert, 'close'):
                        close_tasks.append(expert.close())
            
            # Close literature_retriever session (if any)
            if self.literature_retriever is not None and hasattr(self.literature_retriever, 'close'):
                close_tasks.append(self.literature_retriever.close())
            
            # Execute all close tasks
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

    @abstractmethod
    def _parse_data(self, **kwargs):
        """Parse data, implemented by subclasses"""
        pass
    
    def _retrieve_literature(self) -> str:
        """Retrieve literature"""
        if not self.literature_retriever or not self.data_info:
            return ""
        
        return self.literature_retriever.get_knowledge_context(
            task_description=f"{self.task_type} analysis",
            data_semantics=self.data_info,
            task_type=self.task_type,
        )
    
    async def _run_debate(self, literature_context: str) -> DebateResult:
        """Run multi-expert debate"""
        if not self.debate_system:
            raise ValueError("Debate system not initialized")
        
        # Build data info (including literature context)
        data_info = self.data_info.copy() if self.data_info else {}
        if literature_context:
            data_info['literature_context'] = literature_context
        
        # Run debate
        return await self.debate_system.debate(
            task_type=self.task_type,
            data_info=data_info,
        )
    
    async def _execute_with_llm_iterations(
        self,
        blueprint: Optional[Dict[str, Any]],
        max_iterations: int,
        previous_rounds: Optional[List[Any]] = None,
    ) -> List[ExecutionResult]:
        """
        Iteratively generate and execute code using LLM

        Workflow:
        1. LLM generates initial code
        2. Execute code
        3. If error occurs, LLM corrects code
        4. Repeat until success or max iterations reached
        """
        results = []
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            logger.info(f"Code iteration {iteration + 1}/{max_iterations}")
            
            # Get memory context
            memory_context = ""
            if self.memory_system:
                memory_context = self.memory_system.construct_memory_prompt(
                    task_type=self.task_type,
                    include_short_term=True,
                    include_episodic=True,
                    n_episodic=2,
                    data_name=self.data_name,
                )
                if previous_rounds:
                    for prev in reversed(previous_rounds):
                        if prev.execution_result.get('success'):
                            prev_metrics = prev.metrics.get('computed', {})
                            metric_str = ', '.join(f'{k}={v:.4f}' for k, v in prev_metrics.items() if isinstance(v, (int, float)))
                            bp = prev.blueprint
                            bp_str = ""
                            if bp:
                                parts = []
                                for k in ['architecture_type', 'latent_dim', 'loss_functions']:
                                    if bp.get(k):
                                        parts.append(f"{k}={bp[k]}")
                                bp_str = ", ".join(parts)
                            memory_context += f"\n\n## Previous Best Round ({prev.round_num})\nArchitecture: {bp_str}\nMetrics: {metric_str}"
                            break

                last_code_file = self._find_last_successful_code()
                if last_code_file:
                    try:
                        with open(last_code_file, 'r') as f:
                            code = f.read()
                        lines = code.split('\n')
                        key_lines = []
                        for line in lines:
                            s = line.strip()
                            if s.startswith('class ') or s.startswith('def ') or s.startswith('self.') or 'nn.Module' in s:
                                key_lines.append(line)
                            elif 'loss' in s.lower() and ('=' in s) and not s.startswith('#'):
                                key_lines.append(line)
                            elif 'optimizer' in s.lower() and ('=' in s) and not s.startswith('#'):
                                key_lines.append(line)
                        if key_lines:
                            snippet = '\n'.join(key_lines[:30])
                            memory_context += f"\n\n## Previous Successful Code Structure (key lines only):\n```python\n{snippet}\n```"
                    except Exception:
                        pass
            
            # Generate code (first time or needs correction)
            if iteration == 0 or not results[-1].success:
                try:
                    if iteration == 0:
                        logger.info("Generating initial code with LLM...")
                        code_result = await self.code_generator.generate_code(
                            task_type=self.task_type,
                            data_info=self.data_info or {},
                            blueprint=blueprint,
                            iteration=iteration,
                            memory_context=memory_context,
                        )
                    else:
                        logger.info("Fixing code with LLM based on error...")
                        error_msg = results[-1].error
                        code_result = await self.code_generator.fix_code(
                            original_code=self.current_code,
                            error_message=error_msg,
                            task_type=self.task_type,
                            data_info=self.data_info or {},
                            iteration=iteration,
                            blueprint=blueprint,
                            memory_context=memory_context,
                        )
                except Exception as gen_err:
                    logger.error(f"Code generation/fix failed: {gen_err}")
                    results.append(ExecutionResult(
                        success=False,
                        code="",
                        output="",
                        error=f"Code generation failed: {gen_err}",
                        metrics={},
                        execution_time=0.0,
                    ))
                    break
                
                self.current_code = code_result.code
                logger.info(f"Code generated/reasoning: {code_result.reasoning}, code length: {len(code_result.code)} chars, {len(code_result.code.splitlines())} lines")
            
            syntax_error_msg = None
            code_to_execute = self.current_code

            # Step 1: Check if current code has syntax errors
            try:
                compile(self.current_code, '<agent_generated>', 'exec')
            except SyntaxError as se:
                syntax_error_msg = f"SyntaxError: {se.msg} (line {se.lineno})"
                logger.warning(f"Syntax error detected before execution: {syntax_error_msg}")
                # Do NOT auto-repair. Let LLM fix it in the next iteration.

            execution_failed_due_to_syntax = syntax_error_msg is not None
            if execution_failed_due_to_syntax:
                logger.error(f"❌ SYNTAX ERROR in generated code: {syntax_error_msg}")
                result = ExecutionResult(
                    success=False,
                    code=self.current_code,
                    output="",
                    error=syntax_error_msg,
                    metrics={},
                    execution_time=0.0,
                )
            else:
                # Clean up temporary variables from previous execution to avoid memory accumulation
                self.executor.cleanup()
                result = self.executor.execute(code_to_execute)
            results.append(result)
            
            # Update short-term memory
            if self.memory_system:
                code_summary = self.current_code[:200].replace('\n', ' ')
                error_text = result.incomplete_reason or result.error
                self.memory_system.add_short_term(ShortTermState(
                    prompt=f"Iter {iteration}: {code_summary}",
                    code="",
                    error=error_text[:500] if error_text else "",
                    iteration=iteration,
                ))
            
            # If successful, end iteration
            if result.success:
                logger.info("✅ Code execution successful!")
                break

            # If failed and not the last iteration, continue (will auto-correct)
            if not result.success and iteration < max_iterations - 1:
                if result.incomplete_reason:
                    logger.warning(f"❌ Incomplete code rejected by validator: {result.incomplete_reason[:200]}")
                    logger.info("Will attempt to fix in next iteration (incomplete code)...")
                else:
                    logger.warning(f"❌ Execution failed: {result.error[:200]}")
                    logger.info("Will attempt to fix in next iteration...")
        
        return results
    
    async def _update_memory(
        self,
        execution_results: List[ExecutionResult],
        round_num: int = 1,
        round_results: List[Any] = None,
        blueprint: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Update memory system (supports multi-round iteration)

        Corresponds to formula: T = {B(t), θ(t), y(t)}
        - B(t): Blueprint
        - θ(t): Parameter configuration
        - y(t): Performance metrics
        """
        if not self.memory_system:
            return
        
        # Determine final result
        final_result = execution_results[-1]
        success = final_result.success
        metrics = self._compute_metrics(execution_results)
        computed = metrics.get('computed', {})
        
        code_snippet = final_result.code[:600] if final_result.code else ""
        error_snippet = ""
        if not success:
            errors = [r.error for r in execution_results if not r.success]
            error_snippet = errors[-1][:500] if errors else ""

        lesson = await self._summarize_experience(
            task_type=self.task_type,
            success=success,
            metrics=computed,
            blueprint=blueprint,
            code_snippet=code_snippet,
            error_snippet=error_snippet,
            round_num=round_num,
        )
        
        # Add to episodic memory - use new structure
        # T = {B(t), θ(t), y(t)}
        self.memory_system.add_episodic_memory(
            task_type=self.task_type,
            round_number=round_num,
            blueprint=blueprint or {},
            parameters=parameters or {},
            metrics=metrics,
            outcome="success" if success else "failure",
            lesson_learned=lesson,
            importance_score=0.8 if success else 0.6,
            data_name=self.data_name or "",
        )
        
        logger.info(f"[Memory] Added episodic memory: round={round_num}, outcome={success}, metrics keys={list(metrics.keys())}")
        
        # If last round, add summary memory
        if round_results and round_num == len(round_results):
            successful_rounds = [r for r in round_results if r.execution_result.get('success', False)]
            best_round = self._find_best_round(round_results)
            
            summary_lesson = f"[SUMMARY] {self.task_type}: {len(successful_rounds)}/{len(round_results)} rounds succeeded, best=round{best_round}"
            
            # Aggregate metrics from all rounds
            summary_metrics = {
                'total_rounds': len(round_results),
                'successful_rounds': len(successful_rounds),
                'best_round': best_round,
                'all_round_metrics': [r.metrics.get('computed', {}) for r in round_results],
            }
            
            self.memory_system.add_episodic_memory(
                task_type=self.task_type,
                round_number=0,
                blueprint=blueprint or {},
                parameters=parameters or {},
                metrics=summary_metrics,
                outcome="success" if successful_rounds else "failure",
                lesson_learned=summary_lesson,
                importance_score=0.9 if successful_rounds else 0.7,
                data_name=self.data_name or "",
            )

    async def _summarize_experience(
        self,
        task_type: str,
        success: bool,
        metrics: Dict[str, Any],
        blueprint: Optional[Dict[str, Any]],
        code_snippet: str,
        error_snippet: str,
        round_num: int,
    ) -> str:
        try:
            llm_client = self.code_generator._get_llm_client()
        except Exception:
            return self._fallback_lesson(task_type, success, metrics, blueprint, error_snippet, round_num)

        metric_str = ', '.join(f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float)))
        status = "SUCCESS" if success else "FAILURE"
        bp_str = ""
        if blueprint:
            parts = []
            for k in ['architecture_type', 'latent_dim', 'loss_functions']:
                if blueprint.get(k):
                    parts.append(f"{k}={blueprint[k]}")
            bp_str = ", ".join(parts)

        prompt = f"""Summarize this experiment round in 1-2 concise sentences. Focus on WHAT worked/failed and WHY.
Do NOT repeat raw numbers. Extract actionable insights only.

Task: {task_type} | Round {round_num} | Result: {status}
Architecture: {bp_str or 'N/A'}
Metrics: {metric_str or 'N/A'}"""

        if not success and error_snippet:
            prompt += f"\nError: {error_snippet[:300]}"
        if success and code_snippet:
            prompt += f"\nKey code pattern: {code_snippet[:300]}"

        prompt += """

Output ONLY the summary text, no JSON, no markdown, no labels."""

        try:
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            summary = response.strip() if isinstance(response, str) else (response.content.strip() if response and hasattr(response, 'content') and response.content else "")
            if summary:
                tag = "[SUCCESS]" if success else "[FAILURE]"
                return f"{tag} {summary}"
        except Exception as e:
            logger.warning(f"LLM experience summary failed: {e}")

        return self._fallback_lesson(task_type, success, metrics, blueprint, error_snippet, round_num)

    def _fallback_lesson(
        self,
        task_type: str,
        success: bool,
        metrics: Dict[str, Any],
        blueprint: Optional[Dict[str, Any]],
        error_snippet: str,
        round_num: int,
    ) -> str:
        metric_str = ', '.join(f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float)))
        if success:
            bp_str = ""
            if blueprint:
                parts = []
                for k in ['architecture_type', 'latent_dim', 'loss_functions']:
                    if blueprint.get(k):
                        parts.append(f"{k}={blueprint[k]}")
                bp_str = f" ({', '.join(parts)})" if parts else ""
            return f"[SUCCESS] {task_type}{bp_str}: {metric_str}"
        else:
            return f"[FAILURE] {task_type} round {round_num}: {error_snippet[:100]}"

    def _compute_metrics(self, execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """
        Compute task metrics

        Metric source priority:
        1. Extract metrics dict from successful execution return value (ExecutionResult.return_value)
        2. Search for metrics-related fields in execution result variables
        3. If none found, fall back to basic execution statistics

        Additional checks:
        - If code passes but is rejected by completeness validator (incomplete_reason not empty), mark as failure
        - If no metrics computed but code 'succeeded', also mark as failure (triggers next iteration correction)
        """
        metrics = {
            'n_iterations': len(execution_results),
            'n_success': sum(1 for r in execution_results if r.success),
            'total_execution_time': sum(r.execution_time for r in execution_results),
            'final_success': execution_results[-1].success if execution_results else False,
            'llm_generated': True,
            'incomplete_execution': False,
        }

        # Find the last successful execution result
        successful_results = [r for r in execution_results if r.success]
        if not successful_results:
            return metrics

        last_success = successful_results[-1]

        # Check if completeness validator rejected the code (syntax passed but logic incomplete)
        if last_success.incomplete_reason:
            metrics['incomplete_execution'] = True
            metrics['incomplete_reason'] = last_success.incomplete_reason
            metrics['final_success'] = False
            # Even if return_value / variables exist, mark as incomplete
            if 'incomplete_reason' not in metrics:
                metrics['incomplete_reason'] = last_success.incomplete_reason
            return metrics

        # Try to extract metrics from return_value
        has_computed_metrics = False
        if last_success.return_value is not None:
            rv = last_success.return_value
            if isinstance(rv, dict):
                if 'metrics' in rv and isinstance(rv['metrics'], dict):
                    metrics['computed'] = rv['metrics']
                    has_computed_metrics = True
                    for k, v in rv['metrics'].items():
                        metrics[f'computed_{k}'] = v
                for key in ['nmi', 'ari', 'asw', 'auroc', 'auprc', 'f1',
                            'r2', 'mse', 'pcc', 'pcc_logfc', 'r2_de', 'mse_de', 'pcc_de']:
                    if key in rv and rv[key] is not None:
                        metrics[f'computed_{key}'] = rv[key]
                        has_computed_metrics = True

        # Try to extract metrics from variables
        if not has_computed_metrics:
            vars_dict = last_success.variables
            if 'result' in vars_dict and isinstance(vars_dict['result'], dict):
                result_var = vars_dict['result']
                if 'metrics' in result_var and isinstance(result_var['metrics'], dict):
                    metrics['computed'] = result_var['metrics']
                    has_computed_metrics = True
                    for k, v in result_var['metrics'].items():
                        metrics[f'computed_{k}'] = v
                else:
                    for key in ['nmi', 'ari', 'asw', 'auroc', 'auprc', 'f1',
                                'r2', 'mse', 'pcc', 'pcc_logfc', 'r2_de', 'mse_de', 'pcc_de']:
                        if key in result_var and result_var[key] is not None:
                            metrics[f'computed_{key}'] = result_var[key]
                            has_computed_metrics = True

        # If code passes but no metrics were computed, the code is incomplete
        if not has_computed_metrics and last_success.return_value is None and 'result' not in last_success.variables:
            metrics['incomplete_execution'] = True
            metrics['incomplete_reason'] = (
                "Code executed successfully but returned no result dictionary and no metrics. "
                "The code likely only contains data preparation without actual analysis."
            )
            metrics['final_success'] = False

        # Extra info: store output summary
        if last_success.output:
            output_lines = last_success.output.strip().split('\n')
            metrics['execution_output_lines'] = len(output_lines)
            metrics['execution_summary'] = '\n'.join(output_lines[-5:]) if len(output_lines) > 5 else last_success.output.strip()

        return metrics
    
    def _save_successful_code(self, generated_codes: List[str], execution_results: List[ExecutionResult], round_num: int = 1):
        """Save successfully executed code to file"""
        try:
            from datetime import datetime
            import os
            
            # Create output directory
            output_dir = Path(self.config.output_dir) / "generated_code"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find the last successfully executed code
            successful_results = [r for r in execution_results if r.success]
            if not successful_results:
                return
            
            last_success = successful_results[-1]
            
            # Generate filename (includes round info and dataset name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name_part = f"{self.data_name}_" if self.data_name else ""
            filename = f"{data_name_part}{self.task_type}_round{round_num}_success_{timestamp}.py"
            filepath = output_dir / filename
            
            # Write code
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'Task: {self.task_type}\n')
                f.write(f'Round: {round_num}\n')
                f.write(f'Generated: {datetime.now().isoformat()}\n')
                f.write(f'Iterations: {len(generated_codes)}\n')
                f.write(f'Execution time: {last_success.execution_time:.2f}s\n')
                f.write(f'"""\n\n')
                
                # Write the final successful code
                if hasattr(last_success, 'code') and last_success.code:
                    f.write(last_success.code)
                elif generated_codes:
                    # If no code attribute, write the last generated code
                    f.write(generated_codes[-1])
            
            logger.info(f"Round {round_num} successful code saved to: {filepath}")
            
            # Also save history of all iteration code
            history_dir = output_dir / f"{data_name_part}{self.task_type}_round{round_num}_history_{timestamp}"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            for i, code in enumerate(generated_codes):
                history_file = history_dir / f"iteration_{i}.py"
                with open(history_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Round {round_num}, Iteration {i}"""\n\n')
                    f.write(code)
            
            logger.info(f"Round {round_num} code history saved to: {history_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save successful code: {e}")
    
    def _save_blueprint(self, blueprint: Dict[str, Any], round_num: int, metrics: Optional[Dict[str, Any]] = None):
        """Save blueprint to file"""
        try:
            import os
            import json
            from datetime import datetime
            
            # Create output directory
            output_dir = Path(self.config.output_dir) / "blueprints"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_type_safe = self.task_type.replace('_', '-')
            data_name_part = f"{self.data_name}_" if self.data_name else ""
            filename = f"{data_name_part}{task_type_safe}_round{round_num}_{timestamp}.json"
            filepath = output_dir / filename
            
            # Build data to save
            save_data = {
                'task_type': self.task_type,
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'blueprint': blueprint,
                'metrics': metrics or {},
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Blueprint for round {round_num} saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save blueprint: {e}")

    def _find_last_successful_code(self) -> Optional[str]:
        try:
            code_dir = Path(self.config.output_dir) / "generated_code"
            if not code_dir.exists():
                return None
            files = sorted(code_dir.glob(f"*{self.task_type}*success*.py"), reverse=True)
            return str(files[0]) if files else None
        except Exception:
            return None

    def _find_best_round(self, round_results: List[Any]) -> int:
        if not round_results:
            return 0

        best_round = 0
        best_score = -float('inf')

        for r in round_results:
            if hasattr(r, 'round_num'):
                rn = r.round_num
                metrics = r.metrics.get('computed', {}) if isinstance(r.metrics, dict) else {}
            elif isinstance(r, dict):
                rn = r.get('round_num', 0)
                metrics = r.get('metrics', {})
                if isinstance(metrics, dict):
                    metrics = metrics.get('computed', {})
            else:
                continue

            avg_score = _compute_normalized_avg(metrics)
            if avg_score > best_score:
                    best_score = avg_score
                    best_round = rn

        return best_round

    def _find_best_round_from_summaries(self, round_summaries: List[Dict]) -> int:
        return self._find_best_round(round_summaries)

    def _save_execution_results(self, execution_results: List[ExecutionResult], round_num: int):
        """Save execution results to file, free memory"""
        try:
            from datetime import datetime
            
            output_dir = Path(self.config.output_dir) / "execution_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.task_type}_round{round_num}_{timestamp}.json"
            filepath = output_dir / filename
            
            # Convert to serializable format
            results_data = []
            for i, result in enumerate(execution_results):
                results_data.append({
                    'iteration': i + 1,
                    'success': result.success,
                    'error': result.error,
                    'execution_time': result.execution_time,
                    'metrics': _make_serializable(result.metrics),
                })
            
            save_data = {
                'task_type': self.task_type,
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'results': results_data,
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved execution results to {filepath}")
            
            # Clear execution_results from memory, keep only necessary summary info
            execution_results.clear()
            
        except Exception as e:
            logger.error(f"Failed to save execution results: {e}")
    
    def _save_round_result(self, round_result):
        """Save round results to file, free memory"""
        try:
            from datetime import datetime
            
            output_dir = Path(self.config.output_dir) / "round_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.task_type}_round{round_result.round_num}_{timestamp}.json"
            filepath = output_dir / filename
            
            save_data = {
                'round_num': round_result.round_num,
                'timestamp': round_result.timestamp,
                'blueprint': _make_serializable(round_result.blueprint),
                'execution_result': _make_serializable(round_result.execution_result),
                'metrics': _make_serializable(round_result.metrics),
                'architect_feedback': _make_serializable(round_result.architect_feedback),
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved round result to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save round result: {e}")
    
    def _build_prompt_context(self) -> str:
        """Build prompt context"""
        context_parts = []
        
        # Data information
        if self.semantics:
            context_parts.append("## Data Information")
            context_parts.append(self.semantics.to_text_summary())
        
        # Memory context
        if self.memory_system:
            memory_context = self.memory_system.construct_memory_prompt(
                task_type=self.task_type,
                data_name=self.data_name,
            )
            if memory_context:
                context_parts.append("\n## Memory Context")
                context_parts.append(memory_context)
        
        return "\n\n".join(context_parts)
