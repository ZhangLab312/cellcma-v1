"""
Hierarchical Memory Module

Refactored from the existing memory_system.py, provides three-layer memory structure:
- Short-term: Short-term working memory (single execution cycle)
- Long-term: Long-term skill memory (cross-session experience)
- Episodic: Episodic memory (specific experiment scenarios)
"""

from __future__ import annotations

import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ShortTermState:
    """Short-term working memory state"""
    prompt: str = ""
    code: str = ""
    error: str = ""
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodicMemory:
    """Episodic memory entry - records complete task lifecycle.

    Structure corresponds to formula: T = {B(t), theta(t), y(t)}
    - B(t): Blueprint
    - theta(t): Parameter configuration
    - y(t): Performance metrics
    """
    memory_id: str
    task_type: str  # Task type
    round_number: int  # Round number t
    blueprint: Dict[str, Any]  # Blueprint B(t)
    parameters: Dict[str, Any]  # Parameter configuration theta(t)
    metrics: Dict[str, Any]  # Performance metrics y(t)
    outcome: str  # "success", "failure", "partial"
    lesson_learned: str  # Lessons learned
    importance_score: float  # Importance score 0-1
    data_name: str = ""  # Dataset name
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    # Legacy fields (deprecated but kept for migration)
    action: str = ""  # Deprecated
    result: Dict[str, Any] = field(default_factory=dict)  # Deprecated, use metrics instead
    context: Dict[str, Any] = field(default_factory=dict)  # Deprecated, use blueprint/parameters instead
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        if 'action' not in data:
            data['action'] = ""
        if 'result' not in data:
            data['result'] = {}
        if 'context' not in data:
            data['context'] = {}
        if 'data_name' not in data:
            data['data_name'] = ""
        return cls(**data)


@dataclass
class LongTermKnowledge:
    """Long-term knowledge entry - stores verified code implementations and parameter configurations.

    Structure corresponds to formula: M_long <- M_long union (task, c*, theta*)
    - task: Task description
    - verified_code: Verified code implementation (c*)
    - parameters: Parameter configuration (theta*)
    """
    knowledge_id: str
    task: str  # Task description
    verified_code: str  # Verified code implementation (c*)
    parameters: Dict[str, Any]  # Parameter configuration (theta*)
    category: str  # "successful_pattern", "common_error", "best_practice"
    description: str  # Experience description
    performance: Optional[float] = None  # Performance metrics y(t)
    usage_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalMemorySystem:
    """
    Hierarchical Memory System

    M = {M_short, M_long, M_episodic}

    - M_short: Short-term working memory, attached to model context window
    - M_long: Long-term skill memory, persisted across sessions
    - M_episodic: Episodic memory, records of specific experiment scenarios
    """
    
    def __init__(
        self,
        memory_dir: Optional[str] = None,
        max_short_term: int = 10,
        max_episodic: int = 1000,
        max_long_term: int = 500,
    ):
        self.memory_dir = Path(memory_dir) if memory_dir else Path.home() / ".agent_sct" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_short_term = max_short_term
        self.max_episodic = max_episodic
        self.max_long_term = max_long_term
        
        # Three-layer memory storage
        self._short_term: List[ShortTermState] = []
        self._episodic: List[EpisodicMemory] = []
        self._long_term: List[LongTermKnowledge] = []
        
        # Load persisted memory
        self._load_memory()
        
        logger.info(f"HierarchicalMemorySystem initialized")
        logger.info(f"  Memory dir: {self.memory_dir}")
        logger.info(f"  Episodic memories: {len(self._episodic)}")
        logger.info(f"  Long-term knowledge: {len(self._long_term)}")
    
    # ==================== Short-term Working Memory Operations ====================
    
    def add_short_term(self, state: ShortTermState):
        """Add short-term memory state"""
        self._short_term.append(state)
        
        # Maintain capacity limit
        if len(self._short_term) > self.max_short_term:
            self._short_term.pop(0)
    
    def get_short_term_context(self, n_recent: int = 3) -> str:
        """Get recent short-term memory as context"""
        if not self._short_term:
            return "No previous attempts in current session."
        
        lines = ["### Current Session History:\n"]
        
        for i, state in enumerate(self._short_term[-n_recent:], 1):
            status = "✅ Success" if not state.error else "❌ Failed"
            lines.append(f"{i}. {status} (Iteration {state.iteration})")
            if state.error:
                lines.append(f"   Error: {state.error[:100]}...")
        
        return "\n".join(lines)
    
    def clear_short_term(self):
        """Clear short-term memory"""
        self._short_term.clear()
    
    # ==================== Episodic Memory Operations ====================
    
    def add_episodic_memory(
        self,
        task_type: str,
        round_number: int,
        blueprint: Dict[str, Any],
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        outcome: str,
        lesson_learned: str,
        importance_score: float = 0.5,
        data_name: str = "",
    ) -> str:
        memory_id = self._generate_id(f"{task_type}_{round_number}")
        
        memory = EpisodicMemory(
            memory_id=memory_id,
            task_type=task_type,
            data_name=data_name,
            round_number=round_number,
            blueprint=blueprint,
            parameters=parameters,
            metrics=metrics,
            outcome=outcome,
            lesson_learned=lesson_learned,
            importance_score=importance_score,
        )
        
        self._episodic.append(memory)
        
        # Maintain capacity limit (evict by importance)
        if len(self._episodic) > self.max_episodic:
            self._episodic.sort(key=lambda m: m.importance_score)
            self._episodic.pop(0)
        
        # Sync to long-term memory (only extract successful and high-importance ones)
        if outcome == "success" and importance_score > 0.7:
            self._extract_to_long_term(memory)
        
        self._save_memory()
        
        logger.info(f"[Memory] Added episodic memory: id={memory_id}, task={task_type}, round={round_number}")
        
        return memory_id
    
    def retrieve_episodic_memories(
        self,
        task_type: Optional[str] = None,
        outcome: Optional[str] = None,
        context_keywords: Optional[List[str]] = None,
        top_k: int = 5,
        data_name: Optional[str] = None,
    ) -> List[EpisodicMemory]:
        matches = self._episodic.copy()

        if task_type:
            matches = [m for m in matches if m.task_type == task_type]

        if data_name:
            exact = [m for m in matches if m.data_name == data_name]
            if exact:
                matches = exact

        if outcome:
            matches = [m for m in matches if m.outcome == outcome]
        
        # Sort by keyword relevance
        if context_keywords:
            for memory in matches:
                score = self._calculate_keyword_match(
                    context_keywords,
                    f"{memory.lesson_learned}"  # Removed deprecated memory.action field
                )
                memory.access_count += 1
        
        # Sort by importance and time
        matches.sort(
            key=lambda m: (m.importance_score, m.timestamp),
            reverse=True
        )
        
        return matches[:top_k]
    
    def get_episodic_summary(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        memories = self._episodic
        if task_type:
            memories = [m for m in memories if m.task_type == task_type]
        
        outcomes = {"success": 0, "failure": 0, "partial": 0}
        for m in memories:
            outcomes[m.outcome] = outcomes.get(m.outcome, 0) + 1
        
        return {
            "total": len(memories),
            "outcomes": outcomes,
            "avg_importance": sum(m.importance_score for m in memories) / len(memories) if memories else 0,
        }
    
    # ==================== Long-term Knowledge Operations ====================
    
    def add_long_term_knowledge(
        self,
        task: str,
        verified_code: str,
        parameters: Dict[str, Any],
        category: str,
        description: str,
        performance: Optional[float] = None,
    ) -> str:
        """Add long-term knowledge entry.

        Corresponds to formula: M_long <- M_long union (task, c*, theta*)

        Args:
            task: Task description
            verified_code: Verified code implementation (c*)
            parameters: Parameter configuration (theta*)
            category: Category ("successful_pattern", "common_error", "best_practice")
            description: Experience description
            performance: Performance metrics (y)
        """
        knowledge_id = self._generate_id(f"{task}_{category}")
        
        knowledge = LongTermKnowledge(
            knowledge_id=knowledge_id,
            task=task,
            verified_code=verified_code,
            parameters=parameters,
            category=category,
            description=description,
            performance=performance,
        )
        
        self._long_term.append(knowledge)
        
        # Maintain capacity limit
        if len(self._long_term) > self.max_long_term:
            # Evict by usage frequency
            self._long_term.sort(key=lambda k: k.usage_count)
            self._long_term.pop(0)
        
        self._save_memory()
        
        return knowledge_id
    
    def retrieve_long_term_knowledge(
        self,
        task: Optional[str] = None,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[LongTermKnowledge]:
        """Retrieve long-term knowledge.

        Corresponds to formula: m* = argmax_{m in M_long} sim(q, m)

        Args:
            task: Task description filter
            category: Category filter ("successful_pattern", "common_error")
            keywords: Keyword matching
            top_k: Return count limit

        Returns:
            Long-term knowledge list sorted by relevance
        """
        matches = self._long_term.copy()
        
        # Filter by task type
        if task:
            matches = [k for k in matches if task.lower() in k.task.lower()]
        
        # Filter by category
        if category:
            matches = [k for k in matches if k.category == category]
        
        # Sort by keywords and update access statistics
        if keywords:
            for knowledge in matches:
                score = self._calculate_keyword_match(
                    keywords,
                    f"{knowledge.task} {knowledge.description}"
                )
                knowledge.usage_count += 1
                knowledge.last_accessed = datetime.now().isoformat()
        
        # Sort by success rate and usage frequency
        # Success rate = success_count / max(usage_count, 1)
        matches.sort(
            key=lambda k: (k.success_count / max(k.usage_count, 1), k.usage_count),
            reverse=True
        )
        
        return matches[:top_k]
    
    def retrieve_invalid_directions(
        self,
        task: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve historically failed directions, used to constrain search space.

        Corresponds to formula: theta in Theta_valid

        Args:
            task: Task description

        Returns:
            List of failed directions
        """
        errors = self.retrieve_long_term_knowledge(
            task=task,
            category="common_error",
            top_k=10,
        )
        # Return failed parameter configurations as constraints for subsequent search
        return [k.parameters for k in errors if k.parameters]
    
    # ==================== Helper Methods ====================
    
    def _extract_to_long_term(self, memory: EpisodicMemory):
        task = memory.task_type
        metrics = memory.metrics

        blueprint = memory.blueprint
        key_info = ""
        if blueprint:
            parts = []
            if blueprint.get('architecture_type'):
                parts.append(f"arch={blueprint['architecture_type']}")
            if blueprint.get('latent_dim'):
                parts.append(f"latent_dim={blueprint['latent_dim']}")
            if blueprint.get('loss_functions'):
                parts.append(f"loss={blueprint['loss_functions']}")
            key_info = " | ".join(parts) if parts else ""

        performance = None
        computed = metrics.get('computed', {}) if isinstance(metrics, dict) else {}
        for metric_key in ['computed_nmi', 'computed_r2', 'computed_auroc', 'computed_ari']:
            if metric_key in metrics:
                performance = metrics[metric_key]
                break

        if memory.outcome == "success":
            desc = memory.lesson_learned
            self.add_long_term_knowledge(
                task=task,
                verified_code=key_info,
                parameters=memory.parameters,
                category="successful_pattern",
                description=desc,
                performance=performance,
            )
            logger.info(f"[Memory] Extracted success pattern to long-term: task={task}, performance={performance}")

        elif memory.outcome == "failure":
            self.add_long_term_knowledge(
                task=task,
                verified_code="",
                parameters=memory.parameters,
                category="common_error",
                description=memory.lesson_learned,
                performance=performance,
            )
            logger.info(f"[Memory] Extracted failure pattern to long-term: task={task}")
    
    def _calculate_keyword_match(self, keywords: List[str], text: str) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # ==================== Persistence Operations ====================
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            memory_data = {
                "episodic": [m.to_dict() for m in self._episodic],
                "long_term": [k.to_dict() for k in self._long_term],
                "saved_at": datetime.now().isoformat(),
            }
            
            memory_file = self.memory_dir / "memory.pkl"
            with open(memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            
            # Also save JSON version for easy viewing
            json_file = self.memory_dir / "memory.json"
            with open(json_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")
    
    def _load_memory(self):
        """Load memory from disk"""
        try:
            memory_file = self.memory_dir / "memory.pkl"
            if memory_file.exists():
                with open(memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                
                # Load episodic memory
                for data in memory_data.get("episodic", []):
                    try:
                        self._episodic.append(EpisodicMemory.from_dict(data))
                    except Exception as e:
                        logger.warning(f"Failed to load episodic memory: {e}")
                
                # Load long-term knowledge
                for data in memory_data.get("long_term", []):
                    try:
                        self._long_term.append(LongTermKnowledge(**data))
                    except Exception as e:
                        logger.warning(f"Failed to load long-term knowledge: {e}")
                
                logger.info(f"Loaded memory from {memory_file}")
                
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
    
    # ==================== Unified Interface ====================
    
    def construct_memory_prompt(
        self,
        task_type: str,
        include_short_term: bool = True,
        include_episodic: bool = True,
        include_long_term: bool = True,
        n_episodic: int = 3,
        n_long_term: int = 3,
        data_name: Optional[str] = None,
    ) -> str:
        parts = []

        if include_episodic:
            episodes = self.retrieve_episodic_memories(
                task_type=task_type,
                top_k=n_episodic,
                data_name=data_name,
            )
            success_eps = [ep for ep in episodes if ep.outcome == "success"]
            if success_eps:
                parts.append("## Past Successful Experiences:")
                for ep in success_eps:
                    lesson = ep.lesson_learned
                    parts.append(f"- {lesson}")

        if include_long_term:
            patterns = self.retrieve_long_term_knowledge(
                task=task_type,
                category="successful_pattern",
                top_k=n_long_term
            )
            if patterns:
                parts.append("## Successful Patterns:")
                for p in patterns:
                    line = f"- {p.task}"
                    if p.verified_code:
                        line += f" | {p.verified_code}"
                    if p.performance is not None:
                        line += f" | perf={p.performance:.4f}"
                    parts.append(line)

            errors = self.retrieve_long_term_knowledge(
                task=task_type,
                category="common_error",
                top_k=2
            )
            if errors:
                parts.append("## Avoid:")
                for e in errors:
                    parts.append(f"- {e.description[:120]}")

        result = "\n\n".join(parts) if parts else ""
        logger.info(f"[Memory] construct_memory_prompt: short_term={len(self._short_term)}, episodic={len(self._episodic)}, long_term={len(self._long_term)}, prompt_len={len(result)}")
        return result
    
    def get_valid_parameter_constraints(self, task_type: str) -> Dict[str, Any]:
        """Get valid parameter constraints.

        Corresponds to formula: theta in Theta_valid
        Based on historical experience, returns parameter regions to avoid.
        """
        invalid_directions = self.retrieve_invalid_directions(task_type)
        
        constraints = {
            "description": f"Parameter constraints derived from {len(invalid_directions)} historical failures",
            "avoid_directions": invalid_directions,
        }
        
        return constraints
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "short_term": len(self._short_term),
            "episodic": {
                "total": len(self._episodic),
                "by_outcome": self.get_episodic_summary()["outcomes"],
            },
            "long_term": {
                "total": len(self._long_term),
                "by_category": {
                    cat: len([k for k in self._long_term if k.category == cat])
                    for cat in set(k.category for k in self._long_term)
                },
            },
        }


# Global memory system instance
_default_memory_system: Optional[HierarchicalMemorySystem] = None


def get_memory_system(
    memory_dir: Optional[str] = None,
    **kwargs
) -> HierarchicalMemorySystem:
    """Get global memory system instance"""
    global _default_memory_system
    
    if _default_memory_system is None:
        _default_memory_system = HierarchicalMemorySystem(
            memory_dir=memory_dir,
            **kwargs
        )
    
    return _default_memory_system
