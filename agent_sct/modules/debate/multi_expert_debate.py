"""
Multi-Expert Debate System

Refactored from the existing debate_system.py, implements a collaborative debate mechanism
with four-role experts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...agents.expert_agents import (
    ArchitectAgent, ScientistAgent, EngineerAgent, CriticAgent,
    create_expert_agent, AgentResponse
)

logger = logging.getLogger(__name__)

# Task-specific improvement guidance for analyze_and_improve prompt
_INTEGRATION_IMPROVE_GUIDE = """Focus improvements on:
- Increasing contrastive loss weight if knn_cross < 0.4
- Increasing classifier weight if asw_celltype < 0.5
- Adjusting KL weight if latent collapse detected"""

_PERTURBATION_IMPROVE_GUIDE = """Focus improvements on:
- R2/PCC good but MSE poor: model has systematic scaling bias. Add output rescaling layer, switch to Huber loss, or add an explicit MSE-minimizing loss term
- MSE good but R2/PCC poor: model is overfitting to mean predictions. Increase model capacity or add a correlation-aware auxiliary loss
- All metrics poor: simplify architecture, increase epochs, check data preprocessing
- DE metrics (r2_de, mse_de, pcc_de) worse than global: increase DE-gene loss weight"""

# Metric direction normalization (mirrors base_task.py logic)
_DEBATE_METRIC_DIRECTIONS = {
    'mse': 'lower', 'mse_de': 'lower', 'mae': 'lower', 'rmse': 'lower',
    'r2': 'higher', 'r2_de': 'higher',
    'pcc': 'higher', 'pcc_fc': 'higher', 'pcc_de': 'higher', 'pcc_logfc': 'higher',
    'nmi': 'higher', 'ari': 'higher', 'asw_celltype': 'higher',
    'asw_batch': 'higher', 'knn_cross': 'higher',
}


def _compute_normalized_avg_debate(metrics: Dict[str, Any]) -> float:
    vals = []
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        direction = _DEBATE_METRIC_DIRECTIONS.get(k, 'higher')
        vals.append(-v if direction == 'lower' else v)
    return sum(vals) / len(vals) if vals else -float('inf')


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON content from text, supporting markdown code blocks"""
    # Try to extract ```json ... ``` code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to extract ``` ... ``` generic code block
    match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No code block found, return original text
    return text.strip()


def _fix_incomplete_json(text: str) -> str:
    """Fix incomplete JSON (complete unclosed brackets, quotes, etc.)"""
    result = text.strip()

    # 1. Remove trailing commas (not allowed in JSON)
    result = re.sub(r',\s*}$', '}', result)
    result = re.sub(r',\s*\]$', ']', result)

    # 2. Remove incomplete key-value pairs at the end
    # e.g.: "key": "value without closing
    result = re.sub(r':\s*"[^"]*$', ': ""', result)
    result = re.sub(r',?\s*"[^"]*$', '', result)

    # 3. Complete missing closing brackets
    open_braces = result.count('{')
    close_braces = result.count('}')
    open_brackets = result.count('[')
    close_brackets = result.count(']')

    while close_braces < open_braces:
        result += '}'
        close_braces += 1
    while close_brackets < open_brackets:
        result += ']'
        close_brackets += 1

    # 4. Use raw_decode to try parsing the first complete JSON object
    # This is the most reliable method, automatically handles unterminated strings
    decoder = json.JSONDecoder()
    try:
        decoded, _ = decoder.raw_decode(result)
        return json.dumps(decoded)
    except json.JSONDecodeError:
        pass

    # 5. Alternative: truncate to the first complete JSON object
    first_json_end = result.rfind('}')
    if first_json_end != -1:
        first_json = result[:first_json_end + 1]
        try:
            json.loads(first_json)
            return first_json
        except json.JSONDecodeError:
            pass

    return result


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON, multiple strategies tried in order"""
    # Strategy 1: direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract markdown code block
    extracted = _extract_json_from_text(text)
    if extracted != text:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Strategy 3: fix incomplete JSON
    fixed = _fix_incomplete_json(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: fix then extract code block
    fixed_extracted = _extract_json_from_text(fixed)
    if fixed_extracted != fixed:
        try:
            return json.loads(fixed_extracted)
        except json.JSONDecodeError:
            pass

    return None


class DebatePosition(Enum):
    """Debate position"""
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"


@dataclass
class DebateRound:
    """Debate round record"""
    round_num: int
    blueprint: Dict[str, Any]
    evaluations: Dict[str, AgentResponse]
    consensus_score: float
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


@dataclass
class DebateResult:
    """Debate result"""
    final_blueprint: Dict[str, Any]
    consensus_reached: bool
    rounds: List[DebateRound]
    total_rounds: int
    final_consensus_score: float
    expert_positions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_blueprint': self.final_blueprint,
            'consensus_reached': self.consensus_reached,
            'total_rounds': self.total_rounds,
            'final_consensus_score': self.final_consensus_score,
            'expert_positions': self.expert_positions,
        }


class MultiExpertDebateSystem:
    """
    Multi-Expert Debate System

    Four-role expert collaborative debate, generating consensus blueprints through iterative refinement.

    Architect proposes initial blueprint
    -> Scientist, Engineer, Critic evaluate in parallel
    -> Compute consensus score
    -> If consensus not reached, Architect revises blueprint based on feedback
    -> Repeat until convergence or max rounds reached
    """
    
    def __init__(
        self,
        llm_config: Optional[Any] = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.75,
        min_rounds: int = 2,
        output_dir: str = "outputs",
        task_type: str = "unknown",
    ):
        self.llm_config = llm_config
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.min_rounds = min_rounds
        self.output_dir = output_dir
        self.task_type = task_type
        
        # Initialize four-role experts
        self.architect = ArchitectAgent(llm_config)
        self.scientist = ScientistAgent(llm_config)
        self.engineer = EngineerAgent(llm_config)
        self.critic = CriticAgent(llm_config)
        
        # Evaluator list (excluding architect)
        self.evaluators = {
            'scientist': self.scientist,
            'engineer': self.engineer,
            'critic': self.critic,
        }
        
        self.debate_history: List[DebateResult] = []
        
        logger.info(f"MultiExpertDebateSystem initialized")
        logger.info(f"  Max rounds: {max_rounds}")
        logger.info(f"  Consensus threshold: {consensus_threshold}")

    def _save_debate_outputs(self, debate_result: DebateResult, data_name: str = ""):
        """Save debate process detailed output to file, including each expert's evaluation and confidence"""
        try:
            output_dir = Path(self.output_dir) / "debate_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_type_safe = self.task_type.replace('_', '-') if self.task_type else "unknown"
            data_name_part = f"{data_name}_" if data_name else ""
            filename = f"{data_name_part}{task_type_safe}_debate_{timestamp}.json"
            filepath = output_dir / filename

            rounds_data = []
            for dr in debate_result.rounds:
                experts_data = []
                for name, resp in dr.evaluations.items():
                    try:
                        content_json = json.loads(resp.content) if isinstance(resp.content, str) else resp.content
                    except (json.JSONDecodeError, TypeError):
                        content_json = {"raw": str(resp.content)}
                    experts_data.append({
                        "expert": name,
                        "confidence": resp.confidence,
                        "reasoning": resp.reasoning,
                        "content": content_json,
                        "suggestions": resp.suggestions or [],
                        "concerns": resp.concerns or [],
                    })
                rounds_data.append({
                    "round_num": dr.round_num,
                    "consensus_score": dr.consensus_score,
                    "blueprint": dr.blueprint,
                    "experts": experts_data,
                    "timestamp": dr.timestamp,
                })

            save_data = {
                "task_type": self.task_type,
                "data_name": data_name,
                "total_rounds": debate_result.total_rounds,
                "consensus_reached": debate_result.consensus_reached,
                "final_consensus_score": debate_result.final_consensus_score,
                "expert_positions": debate_result.expert_positions,
                "rounds": rounds_data,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved debate outputs to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save debate outputs: {e}")
    
    # Task type descriptions
    TASK_DESCRIPTIONS = {
        "integration": """
Task: Multi-Omics Integration

Goal: Integrate multiple single-cell omics modalities (e.g., scRNA-seq, scATAC-seq, scProteomics) — whether paired or unpaired — into a shared latent space to enable joint analysis and cross-modality knowledge transfer.

Key Challenges:
- Different omics modalities have heterogeneous statistical properties (continuous expression, binary chromatin accessibility, protein counts, etc.)
- Unpaired data: cells may only be profiled in one modality, requiring cross-modality alignment without shared cell barcodes
- High dimensionality and sparsity across all modalities
- Need to preserve biological variation while removing technical batch effects across modalities
- Scalable integration when the number of modalities exceeds two

Expected Output:
- A shared latent representation where similar cells are close together regardless of modality
- Integrated embeddings suitable for clustering, visualization, and downstream analysis
- Cross-modality correspondence (e.g., RNA-ATAC links) for joint interpretation
""",
        "grn": """
Task: Multi-Omics Gene Regulatory Network Inference

Goal: Infer gene regulatory networks by jointly leveraging multiple single-cell omics modalities (e.g., scRNA-seq for expression, scATAC-seq for chromatin accessibility / TF binding) to recover transcription factor → target gene regulatory relationships.

Key Challenges:
- Single-cell multi-omics data is noisy and sparse across modalities
- Causality vs correlation: hard to distinguish direct regulation from indirect effects
- Linking distal regulatory elements (ATAC peaks) to target genes requires cross-modality inference
- Computational complexity for genome-wide networks with multi-omics evidence integration
- Need to incorporate prior biological knowledge (TF motifs, prior networks) alongside multi-omics signals

Expected Output:
- An adjacency matrix representing regulatory relationships with multi-omics evidence
- Edge list with confidence scores integrating expression correlation and chromatin accessibility
- Network topology suitable for downstream analysis (pathway enrichment, hub gene identification)
- Regulatory element-to-gene links connecting ATAC peaks to target genes
""",
        "perturbation": """
Task: Perturbation Response Prediction

Goal: Predict how cells will respond to genetic or chemical perturbations based on their control state.

Key Challenges:
- High-dimensional output space (thousands of genes)
- Complex non-linear relationships between perturbation and response
- Cell type-specific responses
- Limited training data for rare perturbations
- Need to disentangle perturbation effect from cell type baseline

Expected Output:
- Predicted gene expression profile after perturbation
- Differential expression compared to control
- Performance metrics: R², MSE, Pearson correlation on top differentially expressed genes
""",
    }

    async def debate(
        self,
        task_type: str,
        data_info: Dict[str, Any],
        initial_blueprint: Optional[Dict[str, Any]] = None,
    ) -> DebateResult:
        """
        Execute multi-expert debate

        Args:
            task_type: Task type (integration, grn, perturbation)
            data_info: Data information
            initial_blueprint: Initial blueprint (optional, otherwise generated by architect)

        Returns:
            DebateResult: Debate result
        """
        logger.info(f"Starting multi-expert debate for task: {task_type}")
        
        # Add task description to data_info
        task_description = self.TASK_DESCRIPTIONS.get(task_type, f"Task: {task_type}")
        data_info = data_info.copy()
        data_info['task_description'] = task_description
        
        rounds: List[DebateRound] = []
        
        # Step 1: Generate initial blueprint (if not provided)
        if initial_blueprint is None:
            logger.info("Generating initial blueprint by Architect...")
            initial_blueprint = await self._generate_initial_blueprint(task_type, data_info)
        
        current_blueprint = initial_blueprint
        
        # Step 2: Iterative debate
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Debate Round {round_num}/{self.max_rounds}")
            
            # Collect evaluations from three experts in parallel
            evaluations = await self._collect_evaluations(
                current_blueprint, data_info, task_type
            )
            
            # Calculate consensus score (pass round number for incremental bonus)
            consensus_score = self._calculate_consensus(evaluations, round_num)
            logger.info(f"  Consensus score: {consensus_score:.3f}")
            
            # Record this round
            debate_round = DebateRound(
                round_num=round_num,
                blueprint=current_blueprint.copy(),
                evaluations=evaluations,
                consensus_score=consensus_score,
            )
            rounds.append(debate_round)
            
            # Check if consensus is reached
            if consensus_score >= self.consensus_threshold and round_num >= self.min_rounds:
                logger.info(f"  ✓ Consensus reached at round {round_num}")
                break
            
            # If max rounds not reached, revise blueprint
            if round_num < self.max_rounds:
                logger.info("  Revising blueprint...")
                current_blueprint = await self._revise_blueprint(
                    current_blueprint, evaluations, data_info, task_type
                )
        
        # Select the blueprint with highest consensus
        best_round = max(rounds, key=lambda r: r.consensus_score)
        best_blueprint = best_round.blueprint
        best_consensus_score = best_round.consensus_score
        best_round_num = best_round.round_num
        
        # If the best blueprint is not from the last round, log a note
        if best_round_num != len(rounds):
            logger.info(f"  ℹ️  Round {best_round_num} has highest consensus ({best_consensus_score:.3f}), "
                       f"using it instead of final round ({consensus_score:.3f})")
        
        # Build result
        result = DebateResult(
            final_blueprint=best_blueprint,
            consensus_reached=best_consensus_score >= self.consensus_threshold,
            rounds=rounds,
            total_rounds=len(rounds),
            final_consensus_score=best_consensus_score,
            expert_positions=self._get_expert_positions(best_round.evaluations),
        )
        
        self.debate_history.append(result)
        
        logger.info(f"Debate completed. Best consensus: {best_consensus_score:.3f} (round {best_round_num})")
        
        # Print the final selected architecture
        logger.info("=" * 60)
        logger.info("FINAL SELECTED ARCHITECTURE")
        logger.info("=" * 60)
        blueprint = result.final_blueprint
        if isinstance(blueprint, dict):
            for key, value in blueprint.items():
                if key != 'reasoning':
                    logger.info(f"  {key}: {value}")
            if 'reasoning' in blueprint:
                logger.info(f"  reasoning: {blueprint['reasoning'][:200]}...")
        else:
            logger.info(f"  {blueprint}")
        logger.info("=" * 60)
        
        # Save debate detailed output
        data_name = data_info.get('data_name', '')
        self._save_debate_outputs(result, data_name)
        
        return result
    
    async def _generate_initial_blueprint(
        self,
        task_type: str,
        data_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate initial blueprint"""
        context = {
            'task_type': task_type,
            'data_info': data_info,
        }

        response = await self.architect.process(context)

        # Use fault-tolerant JSON parsing (multiple strategies tried in order)
        parsed = _try_parse_json(response.content)

        if parsed and isinstance(parsed, dict):
            # Prefer extraction from 'blueprint' key
            if 'blueprint' in parsed:
                logger.info("Blueprint extracted from 'blueprint' key")
                return parsed['blueprint']

            # Check for typical top-level blueprint fields
            if any(k in parsed for k in ('method', 'model', 'approach', 'architecture',
                                          'algorithm', 'model_architecture',
                                          'architecture_type', 'model_config')):
                logger.info("Blueprint extracted from top-level dict")
                return parsed

            # If there are other meaningful keys, also return as blueprint
            if len(parsed) > 0:
                logger.info(f"Blueprint extracted as raw dict with keys: {list(parsed.keys())}")
                return parsed

        # All strategies failed, return raw content
        logger.warning("Failed to parse architect response as JSON, using raw content")
        return {'raw_design': response.content, 'reasoning': response.reasoning}
    
    async def _collect_evaluations(
        self,
        blueprint: Dict[str, Any],
        data_info: Dict[str, Any],
        task_type: str = "unknown"
    ) -> Dict[str, AgentResponse]:
        """Collect evaluations from three experts in parallel"""
        context = {
            'blueprint': blueprint,
            'data_info': data_info,
            'task_type': task_type,
        }
        
        # Execute three evaluations in parallel
        tasks = [
            self.scientist.process(context),
            self.engineer.process(context),
            self.critic.process(context),
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        evaluations = {}
        for (name, _), response in zip(self.evaluators.items(), responses):
            if isinstance(response, Exception):
                logger.error(f"{name} evaluation failed: {response}")
                evaluations[name] = AgentResponse(
                    content="Error",
                    reasoning=str(response),
                    confidence=0.0,
                    concerns=["Evaluation failed"]
                )
            else:
                evaluations[name] = response
                # Record expert position
                try:
                    content = json.loads(response.content)
                    position = content.get('position', 'neutral')
                    logger.info(f"  {name.capitalize()}: {position} (confidence: {response.confidence:.2f})")
                except:
                    logger.info(f"  {name.capitalize()}: evaluated")
        
        return evaluations
    
    def _calculate_consensus(self, evaluations: Dict[str, AgentResponse], round_num: int = 1) -> float:
        """
        Calculate consensus score.

        Based on paper formula (4-11): C_t = (1/|E|) sum_{e in E} score_e(B_t) * (1 - alpha * sigma) + beta * t

        Where:
        - score_e(B_t): Expert e's score for blueprint B_t
        - sigma: Standard deviation of expert scores (measures disagreement)
        - alpha: Disagreement penalty coefficient
        - beta: Round incremental bonus coefficient
        - t: Current round number
        """
        if not evaluations:
            return 0.0

        scores = []
        for name, response in evaluations.items():
            # Parse position and severity
            position_score, severity = self._position_to_score(response.content)
            # Combine confidence and severity
            combined_score = position_score * response.confidence
            scores.append(combined_score)

        # Average score
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Calculate disagreement level (standard deviation)
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        # Disagreement penalty coefficient (reduced to minimize disagreement impact)
        alpha = 0.1

        # Consensus after disagreement penalty
        consensus_after_penalty = avg_score * (1 - alpha * std_dev)

        # Round incremental bonus (+0.05 per round)
        beta = 0.05
        round_bonus = beta * round_num

        # Final consensus = average score * (1 - disagreement penalty) + round bonus
        consensus = consensus_after_penalty + round_bonus

        return max(0.0, min(1.0, consensus))
    
    def _position_to_score(self, content: str) -> Tuple[float, float]:
        try:
            data = json.loads(content)
            position = data.get('position', 'neutral').lower()
            severity = data.get('severity', 0.5)

            if position == 'support':
                score = 0.7 + 0.3 * severity
                return score, severity
            elif position == 'neutral':
                score = 0.4 + 0.2 * severity
                return score, severity
            elif position == 'oppose':
                score = 0.4 - 0.3 * severity
                return max(0.1, score), severity
            else:
                return 0.5, 0.5
        except:
            return 0.5, 0.5
    
    async def _revise_blueprint(
        self,
        current_blueprint: Dict[str, Any],
        evaluations: Dict[str, AgentResponse],
        data_info: Dict[str, Any],
        task_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Revise blueprint based on evaluation feedback.

        Based on paper formula (4-13): B_{t+1} = LLM(B_t, F_t, P_revise)

        Feedback organized by role:
        - Scientist: ML/mathematical theory issues
        - Engineer: Engineering feasibility issues
        - Critic: Methodology comparison issues
        """
        # Organize feedback by role
        role_labels = {
            'scientist': 'ML/THEORY (Scientist)',
            'engineer': 'ENGINEERING (Engineer)',
            'critic': 'METHODOLOGY (Critic)',
        }
        
        feedback_parts = []
        for name, response in evaluations.items():
            label = role_labels.get(name, name.upper())
            feedback_parts.append(f"\n## {label}")
            feedback_parts.append(f"Position: {self._get_position_from_response(response)}")
            feedback_parts.append(f"Reasoning: {response.reasoning}")
            if response.concerns:
                feedback_parts.append(f"Concerns: {'; '.join(response.concerns[:3])}")
            if response.suggestions:
                feedback_parts.append(f"Suggestions: {'; '.join(response.suggestions[:3])}")
        
        feedback_text = '\n'.join(feedback_parts)
        
        # Build revision prompt
        try:
            blueprint_json = json.dumps(current_blueprint, indent=2)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize blueprint: {e}")
            blueprint_json = json.dumps({"raw_blueprint": str(current_blueprint)}, indent=2)

        # Get original task description (preserve so architect doesn't lose task context)
        task_description = data_info.get(
            'task_description',
            self.TASK_DESCRIPTIONS.get(task_type, f"Task: {task_type}")
        )

        prompt = f"""Revise the architecture blueprint for a {task_type} task based on expert feedback.
Address each expert's concerns separately since they evaluate from different perspectives.

Task: {task_description}

Current Blueprint:
{blueprint_json}

Expert Feedback (organized by evaluation perspective):
{feedback_text}

Output exactly this structure (all required fields, adapt field names freely to fit your architecture):
{{
  "architecture_type": "string",
  "model_config": {{"your own key-value pairs describing the architecture"}},
  "latent_dim": integer or null,
  "loss_functions": ["loss1", "loss2"] or [],
  "memory_requirements": {{"gpu_gb": number, "ram_gb": number}},
  "batch_size_recommendation": integer,
  "reasoning": "string explaining why this architecture addresses the feedback"
}}

Example:
{{"architecture_type": "VAE", "model_config": {{"encoder_layers": [512, 256], "decoder_layers": [256, 512]}}, "latent_dim": 64, "loss_functions": ["reconstruction", "kl"], "memory_requirements": {{"gpu_gb": 4, "ram_gb": 8}}, "batch_size_recommendation": 32, "reasoning": "..."}}

WRONG: ```json\n{{...}}\n``` (do NOT use code fences)
"""
        
        # Call architect for revision
        context = {
            'task_type': task_type,
            'data_info': {
                'task_description': task_description,
                'current_blueprint': current_blueprint,
                'feedback': feedback_parts,
                'original_data': data_info,
            }
        }
        
        response = await self.architect.process(context)

        # Use fault-tolerant JSON parsing
        parsed = _try_parse_json(response.content)
        if parsed:
            return parsed

        logger.warning("Failed to parse revised blueprint")
        return current_blueprint
    
    def _get_position_from_response(self, response: AgentResponse) -> str:
        """Extract position from response content"""
        try:
            data = json.loads(response.content)
            return data.get('position', 'unknown')
        except:
            return 'unknown'
    
    def _get_expert_positions(self, evaluations: Dict[str, AgentResponse]) -> Dict[str, str]:
        """Get expert final positions"""
        positions = {}
        for name, response in evaluations.items():
            try:
                data = json.loads(response.content)
                positions[name] = data.get('position', 'neutral')
            except:
                positions[name] = 'unknown'
        return positions
    
    def get_debate_statistics(self) -> Dict[str, Any]:
        """Get debate statistics"""
        if not self.debate_history:
            return {'total_debates': 0}
        
        total = len(self.debate_history)
        consensus_reached = sum(1 for d in self.debate_history if d.consensus_reached)
        avg_rounds = sum(d.total_rounds for d in self.debate_history) / total
        avg_consensus = sum(d.final_consensus_score for d in self.debate_history) / total
        
        return {
            'total_debates': total,
            'consensus_reached': consensus_reached,
            'consensus_rate': consensus_reached / total if total > 0 else 0,
            'avg_rounds': avg_rounds,
            'avg_consensus_score': avg_consensus,
        }


@dataclass
class IterationRound:
    """Experiment iteration round record"""
    round_num: int
    blueprint: Dict[str, Any]
    execution_result: Dict[str, Any]  # Code execution result
    metrics: Dict[str, Any]  # Evaluation metrics
    architect_feedback: str  # Architect's feedback on results
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


@dataclass
class IterationResult:
    """Multi-round iteration result"""
    final_blueprint: Dict[str, Any]
    rounds: List[IterationRound]
    total_rounds: int
    best_round: int
    best_metrics: Dict[str, Any]
    improvement_history: List[float]  # Improvement magnitude per round


class MultiRoundExperimentSystem:
    """
    Multi-round experiment iteration system

    Implements the experiment-feedback-improvement closed loop:
    1. Architect designs initial blueprint
    2. Code generator generates code
    3. Execute code and obtain results
    4. Architect analyzes results and improves blueprint
    5. Repeat until convergence or max rounds reached
    """
    
    def __init__(
        self,
        debate_system: MultiExpertDebateSystem,
        llm_config: Optional[Any] = None,
        max_rounds: int = 5,
        improvement_threshold: float = 0.05,  # Improvement threshold
    ):
        self.debate_system = debate_system
        self.llm_config = llm_config
        self.max_rounds = max_rounds
        self.improvement_threshold = improvement_threshold
        self.iteration_history: List[IterationResult] = []
        
        logger.info(f"MultiRoundExperimentSystem initialized")
        logger.info(f"  Max rounds: {max_rounds}")
        logger.info(f"  Improvement threshold: {improvement_threshold}")
    
    async def run_iteration(
        self,
        task_type: str,
        data_info: Dict[str, Any],
        initial_blueprint: Optional[Dict[str, Any]] = None,
        previous_rounds: Optional[List[IterationRound]] = None,
    ) -> IterationResult:
        """
        Run multi-round experiment iteration

        Args:
            task_type: Task type
            data_info: Data information
            initial_blueprint: Initial blueprint (optional)
            previous_rounds: Results from previous rounds (for incremental iteration)

        Returns:
            IterationResult: Iteration result
        """
        logger.info(f"="*60)
        logger.info(f"Starting Multi-Round Experiment Iteration")
        logger.info(f"Task: {task_type}")
        logger.info(f"Max rounds: {self.max_rounds}")
        logger.info(f"="*60)
        
        rounds: List[IterationRound] = previous_rounds.copy() if previous_rounds else []
        current_blueprint = initial_blueprint
        best_metrics = {}
        best_round = 0
        improvement_history = []
        
        # If no initial blueprint, generate one first
        if current_blueprint is None:
            logger.info("Generating initial blueprint...")
            debate_result = await self.debate_system.debate(task_type, data_info)
            current_blueprint = debate_result.final_blueprint
        
        for round_num in range(len(rounds) + 1, self.max_rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment Round {round_num}/{self.max_rounds}")
            logger.info(f"{'='*60}")
            
            # Return current blueprint, wait for external code execution
            # After external execution, call continue_iteration to continue
            return IterationResult(
                final_blueprint=current_blueprint,
                rounds=rounds,
                total_rounds=len(rounds),
                best_round=best_round,
                best_metrics=best_metrics,
                improvement_history=improvement_history,
            )
        
        # All rounds completed
        return IterationResult(
            final_blueprint=current_blueprint,
            rounds=rounds,
            total_rounds=len(rounds),
            best_round=best_round,
            best_metrics=best_metrics,
            improvement_history=improvement_history,
        )
    
    async def analyze_and_improve(
        self,
        task_type: str,
        data_info: Dict[str, Any],
        current_blueprint: Dict[str, Any],
        execution_result: Dict[str, Any],
        previous_rounds: List[IterationRound],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Analyze experiment results and improve blueprint.

        Args:
            task_type: Task type
            data_info: Data information
            current_blueprint: Current blueprint
            execution_result: Code execution result
            previous_rounds: Results from previous rounds

        Returns:
            (improved_blueprint, architect_feedback): Improved blueprint and architect feedback
        """
        logger.info("Analyzing execution results and improving blueprint...")

        # Build experiment history context
        history_context = self._build_history_context(previous_rounds)

        # Extract key metrics from current execution
        metrics_summary = self._extract_metrics_summary(execution_result)

        # ===== Find best round for few-shot context =====
        best_round_ctx = self._build_best_round_context(previous_rounds)

        # ===== Build locked architecture constraints =====
        locked_constraints = self._build_locked_constraints(current_blueprint, previous_rounds, task_type)

        # Build prompt for architect to analyze
        prompt = f"""Analyze the experiment results and improve the architecture blueprint.

Task: {task_type}

Current Blueprint:
{json.dumps(current_blueprint, indent=2)}

Execution Results:
{json.dumps(execution_result, indent=2, default=str)}

Metrics Summary:
{metrics_summary}

{history_context}

{best_round_ctx}

{locked_constraints}

Please provide:
1. Analysis of what worked well and what didn't
2. Specific improvements to the architecture
3. Revised blueprint in JSON format

IMPORTANT: Only tune hyperparameters (loss weights, learning rate, batch_size, latent_dim).
Do NOT change the fundamental architecture unless it clearly failed.
{_PERTURBATION_IMPROVE_GUIDE if task_type in ('perturbation', 'genetic_perturbation') else _INTEGRATION_IMPROVE_GUIDE}
- Increasing epochs or adjusting learning rate if underfitting

Response format:
{{
  "analysis": "Detailed analysis of results",
  "improvements": ["improvement1", "improvement2", ...],
  "blueprint": {{... revised blueprint ...}}
}}
"""
        
        # Call architect
        context = {
            'task_type': 'iteration_analysis',
            'data_info': {
                'current_blueprint': current_blueprint,
                'execution_result': execution_result,
                'history': [r.__dict__ for r in previous_rounds],
                'task_type': task_type,
            }
        }
        
        response = await self.debate_system.architect.process(context)
        
        # Parse response, try to extract blueprint
        parsed = _try_parse_json(response.content)
        
        def _extract_blueprint(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Extract blueprint from various possible JSON structures"""
            # Strategy 1: has 'blueprint' key directly
            if 'blueprint' in data and isinstance(data['blueprint'], dict):
                analysis = data.get('analysis', '') or data.get('reasoning', '') or ''
                return data['blueprint'], analysis, data.get('improvements', [])
            # Strategy 2: has revised_blueprint / improved_blueprint key directly
            for key in ('revised_blueprint', 'improved_blueprint', 'design', 'architecture'):
                if key in data and isinstance(data[key], dict):
                    analysis = data.get('analysis', '') or data.get('reasoning', '') or ''
                    return data[key], analysis, data.get('improvements', [])
            # Strategy 3: blueprint under 'data' field
            if 'data' in data and isinstance(data['data'], dict):
                inner = data['data']
                if 'blueprint' in inner:
                    analysis = data.get('analysis', '') or data.get('reasoning', '') or ''
                    return inner['blueprint'], analysis, data.get('improvements', [])
            # Strategy 4: top-level is blueprint content (no blueprint key but has method/model fields)
            blueprint_keys = {'method', 'model', 'approach', 'architecture', 'algorithm',
                             'data_preprocessing', 'model_architecture', 'training', 'loss',
                             'architecture_type', 'model_config', 'latent_dim', 'loss_functions',
                             'batch_size_recommendation'}
            if any(k in data for k in blueprint_keys):
                analysis = data.get('analysis', '') or data.get('reasoning', '') or ''
                improvements = data.get('improvements', [])
                return data, analysis, improvements
            return None
        
        if parsed:
            result = _extract_blueprint(parsed)
            if result:
                improved_blueprint, analysis, improvements = result
                logger.info(f"Blueprint extracted from structured JSON")
                logger.info(f"Analysis: {analysis[:200]}..." if analysis else "No analysis provided")
            else:
                # parsed exists but no usable blueprint structure, try text extraction
                logger.warning("JSON parsed but no blueprint key found, trying text extraction...")
                improved_blueprint, analysis, improvements = None, "No analysis", []
                # Fall through to text extraction below
        else:
            improved_blueprint, analysis, improvements = None, "No analysis", []
        
        # If structured JSON extraction fails, try extracting from text/code block
        if improved_blueprint is None:
            logger.warning("Failed to parse as JSON or no blueprint key, attempting text extraction...")
            
            # Try to extract JSON code block from text
            extracted = _extract_json_from_text(response.content)
            parsed_fallback = _try_parse_json(extracted)
            
            if parsed_fallback:
                result = _extract_blueprint(parsed_fallback)
                if result:
                    improved_blueprint, analysis, improvements = result
                    logger.info(f"Blueprint extracted from text code block")
                else:
                    # No recognizable blueprint structure, try building from reasoning/raw content
                    logger.warning("No blueprint key in text extraction either, building from content...")
                    # Try to extract reasoning and improvements fields
                    analysis = parsed_fallback.get('analysis', '') or parsed_fallback.get('reasoning', '') or response.reasoning or ''
                    improvements = parsed_fallback.get('improvements', [])
                    # Use the entire parsed_fallback as blueprint (it may contain useful fields)
                    improved_blueprint = parsed_fallback if isinstance(parsed_fallback, dict) else {}
            
            if improved_blueprint is None:
                # All parsing failed, build from raw content
                logger.warning(f"Could not extract structured blueprint, using raw content (len={len(response.content)})")
                improved_blueprint = {
                    'reasoning': response.reasoning or response.content[:1000],
                    'raw_content': response.content[:500],
                }
                analysis = "JSON parse failed, using raw content with reasoning"
                improvements = []
        
        # Build data info with execution results (so evaluators can see improvement intent)
        eval_data_info = data_info.copy() if data_info else {}
        eval_data_info['execution_result'] = execution_result
        eval_data_info['improvements'] = improvements
        eval_data_info['analysis'] = analysis
        
        # ===== Multi-expert evaluation (consistent with Round 1) =====
        # The improved blueprint needs multi-round iterative debate until consensus or max rounds
        debate_max_rounds = getattr(self.debate_system, 'max_rounds', 3)
        consensus_threshold = self.debate_system.consensus_threshold
        
        current_improved = improved_blueprint
        best_blueprint = improved_blueprint
        best_consensus = 0.0
        
        logger.info(f"Starting multi-round debate on improved blueprint (max {debate_max_rounds} rounds, threshold {consensus_threshold:.3f})...")
        
        for debate_round in range(1, debate_max_rounds + 1):
            logger.info(f"Debate iteration {debate_round}/{debate_max_rounds}...")
            
            # Collect evaluations from three experts in parallel
            evaluations = await self.debate_system._collect_evaluations(
                current_improved, eval_data_info, task_type
            )
            
            # Calculate consensus score
            consensus_score = self.debate_system._calculate_consensus(evaluations, round_num=debate_round)
            logger.info(f"  Consensus score: {consensus_score:.3f}")
            
            # Record expert positions
            expert_positions = self.debate_system._get_expert_positions(evaluations)
            for name, position in expert_positions.items():
                logger.info(f"    {name}: {position}")
            
            # Record best of this round
            if consensus_score > best_consensus:
                best_consensus = consensus_score
                best_blueprint = current_improved
            
            # Consensus reached, stop
            if consensus_score >= consensus_threshold:
                logger.info(f"  ✓ Consensus reached at iteration {debate_round} ({consensus_score:.3f})")
                break
            
            # No consensus and max rounds not reached, let architect revise
            if debate_round < debate_max_rounds:
                logger.info(f"  Revising blueprint based on expert feedback...")
                revised = await self.debate_system._revise_blueprint(
                    current_improved, evaluations, eval_data_info, task_type
                )
                
                # Parse revised blueprint
                parsed_revised = _try_parse_json(revised) if isinstance(revised, str) else revised
                if parsed_revised:
                    current_improved = parsed_revised
                    logger.info("Blueprint revised after expert feedback")
                else:
                    logger.warning("Failed to parse revised blueprint, keeping current")
        
        # Use the blueprint with highest consensus
        if best_consensus > consensus_score:
            logger.info(f"Using best blueprint (consensus {best_consensus:.3f}) instead of final ({consensus_score:.3f})")
            improved_blueprint = best_blueprint
        
        return improved_blueprint, analysis
    
    def _build_history_context(self, rounds: List[IterationRound]) -> str:
        """Build history experiment context"""
        if not rounds:
            return "No previous rounds."

        lines = ["\nPrevious Experiment Rounds:"]
        for r in rounds:
            lines.append(f"\nRound {r.round_num}:")
            lines.append(f"  Metrics: {json.dumps(r.metrics, indent=2)}")
            lines.append(f"  Key issues: {r.architect_feedback[:100]}...")

        return '\n'.join(lines)

    def _build_best_round_context(self, rounds: List[IterationRound]) -> str:
        """Build few-shot context from the best performing round."""
        if not rounds:
            return ""

        best_round = None
        best_avg = -float('inf')

        for r in rounds:
            computed = r.metrics.get('computed', {}) if isinstance(r.metrics, dict) else {}
            if not computed:
                continue
            avg = _compute_normalized_avg_debate(computed)
            if avg > best_avg:
                best_avg = avg
                best_round = r

        if best_round is None:
            return ""

        computed = best_round.metrics.get('computed', {}) if isinstance(best_round.metrics, dict) else {}
        metric_str = ', '.join(f'{k}={v:.4f}' for k, v in computed.items() if isinstance(v, (int, float)))

        bp = best_round.blueprint
        bp_summary = ""
        if bp:
            parts = []
            for k in ['architecture_type', 'latent_dim', 'loss_functions', 'encoder_config']:
                if bp.get(k):
                    parts.append(f"{k}={bp[k]}")
            bp_summary = ", ".join(parts)

        return f"""
## BEST PERFORMING ROUND (Round {best_round.round_num}) - USE AS REFERENCE
Architecture: {bp_summary or 'N/A'}
Metrics: {metric_str}
Average score: {best_avg:.4f}

The architecture from this round achieved the best results. When improving:
- Keep the same fundamental architecture type
- Only adjust hyperparameters (loss weights, dimensions, learning rate)
- Focus on the weakest metrics for improvement
"""

    def _build_locked_constraints(
        self,
        current_blueprint: Dict[str, Any],
        rounds: List[IterationRound],
        task_type: str = "integration"
    ) -> str:
        """Build constraints based on what has worked to reduce LLM freedom.

        Multi-omics integration metrics (nmi/ari/knn_cross/asw_celltype) are only
        meaningful for integration tasks. For perturbation tasks, skip this entirely.
        """
        if not rounds or not current_blueprint:
            return ""

        if task_type in ("perturbation", "genetic_perturbation"):
            return ""

        # Analyze which architectural choices have been consistently successful
        successful_architectures = []
        for r in rounds:
            computed = r.metrics.get('computed', {}) if isinstance(r.metrics, dict) else {}
            nmi = computed.get('nmi', 0) or 0
            ari = computed.get('ari', 0) or 0
            if nmi > 0.7 and ari > 0.5:
                bp = r.blueprint
                if bp:
                    arch_type = bp.get('architecture_type', bp.get('model', ''))
                    latent_dim = bp.get('latent_dim', '')
                    successful_architectures.append({
                        'arch_type': arch_type,
                        'latent_dim': latent_dim,
                        'nmi': nmi,
                        'ari': ari,
                    })

        if not successful_architectures:
            return ""

        # Determine locked parameters
        locked = []
        arch_types = set(a['arch_type'] for a in successful_architectures if a['arch_type'])
        if arch_types:
            locked.append(f"- Architecture type: keep as {list(arch_types)[0]}")

        latent_dims = set(a['latent_dim'] for a in successful_architectures if a['latent_dim'])
        if latent_dims:
            locked.append(f"- Latent dim: keep as {list(latent_dims)[0]} or increase, never decrease below 64")

        # Analyze metrics to suggest specific tuning
        all_metrics = []
        for r in rounds:
            computed = r.metrics.get('computed', {}) if isinstance(r.metrics, dict) else {}
            if computed:
                all_metrics.append(computed)

        tuning_hints = []
        if all_metrics:
            avg_knn = sum((m.get('knn_cross', 0) or 0) for m in all_metrics) / len(all_metrics)
            avg_asw = sum((m.get('asw_celltype', 0) or 0) for m in all_metrics) / len(all_metrics)
            if avg_knn < 0.4:
                tuning_hints.append("- KNN_cross is low: increase contrastive loss weight (try 2.5-5.0) and lower temperature (try 0.2-0.3)")
            if avg_asw < 0.5:
                tuning_hints.append("- ASW_celltype is low: increase cell type classifier weight (try 0.5-1.0)")

        locked_str = "\n".join(locked) if locked else ""
        tuning_str = "\n".join(tuning_hints) if tuning_hints else ""

        result = ""
        if locked_str:
            result += f"""
## LOCKED ARCHITECTURE CONSTRAINTS (do NOT change these unless all rounds failed):
{locked_str}
"""
        if tuning_str:
            result += f"""
## TUNING SUGGESTIONS (based on metric analysis):
{tuning_str}
"""
        return result
    
    def _extract_metrics_summary(self, execution_result: Dict[str, Any]) -> str:
        """Extract metrics summary"""
        metrics = execution_result.get('metrics', {})
        if not metrics:
            return "No metrics available"
        
        lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines) if lines else "No numeric metrics"
    
    def calculate_improvement(
        self,
        current_metrics: Dict[str, Any],
        previous_metrics: Dict[str, Any],
        higher_is_better: Optional[Dict[str, bool]] = None,
    ) -> float:
        if not previous_metrics:
            return 1.0

        default_directions = {
            'nmi': True, 'ari': True, 'asw_celltype': True,
            'nmi_rna': True, 'ari_rna': True, 'nmi_atac': True, 'ari_atac': True,
            'nmi_avg': True, 'ari_avg': True,
            'asw_batch': True, 'knn_cross': True,
            'auroc': True, 'auprc': True, 'f1': True,
            'r2': True, 'mse': False, 'pcc_logfc': True,
            # Perturbation-specific metrics
            'r2_de': True, 'mse_de': False, 'pcc_de': True, 'pcc_fc': True,
        }
        directions = {**default_directions, **({k: v for k, v in (higher_is_better or {}).items() if v is not None})}

        improvements = []
        for key in current_metrics:
            if key in previous_metrics:
                curr = current_metrics[key]
                prev = previous_metrics[key]
                if isinstance(curr, (int, float)) and isinstance(prev, (int, float)) and prev != 0:
                    change = (curr - prev) / abs(prev)
                    if not directions.get(key, True):
                        change = -change
                    improvements.append(change)

        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def check_metric_goals(
        self,
        current_metrics: Dict[str, Any],
        metric_goals: Optional[Dict[str, float]] = None,
        metric_directions: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether absolute goals are met.

        Args:
            current_metrics: Current metrics {"R2": 0.75, "MSE": 0.2, ...}
            metric_goals: Absolute goals {"R2": 0.8, "MSE": 0.1}
            metric_directions: Metric directions {"R2": "higher", "MSE": "lower"}

        Returns:
            (whether goals met, goal status info)
        """
        if not metric_goals:
            return False, ""
        
        # Default direction: assume higher is better
        default_direction = "higher"
        if metric_directions is None:
            metric_directions = {}
        
        achieved = []
        not_achieved = []
        
        for metric_name, goal_value in metric_goals.items():
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            direction = metric_directions.get(metric_name, default_direction)
            
            if not isinstance(current_value, (int, float)) or not isinstance(goal_value, (int, float)):
                continue
            
            if direction == "higher":
                is_met = current_value >= goal_value
            else:  # "lower"
                is_met = current_value <= goal_value
            
            status = "✓" if is_met else "✗"
            comparison = f"{'>=' if direction == 'higher' else '<='} {goal_value}"
            
            info = f"  {status} {metric_name}: {current_value:.4f} {comparison}"
            
            if is_met:
                achieved.append(info)
            else:
                not_achieved.append(info)
        
        if achieved or not_achieved:
            all_checks = achieved + not_achieved
            summary = f"Goals check: {len(achieved)}/{len(all_checks)} achieved\n" + "\n".join(all_checks)
        else:
            summary = ""
        
        return len(achieved) == len(metric_goals) and len(achieved) > 0, summary

async def run_debate(
    task_type: str,
    data_info: Dict[str, Any],
    llm_config: Optional[Any] = None,
    **kwargs
) -> DebateResult:
    """Convenience function: run debate"""
    debate_system = MultiExpertDebateSystem(llm_config=llm_config, **kwargs)
    return await debate_system.debate(task_type, data_info)
