"""
Expert Agents - Four-role Expert Agents

Refactored based on the existing debate_system.py, implementing four expert roles:
- Architect: Responsible for initial blueprint design
- Scientist: Responsible for theoretical validation
- Engineer: Responsible for engineering feasibility evaluation
- Critic: Responsible for critical review
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


def json_safe_serialize(obj: Any) -> Any:
    """Convert numpy/numpy-like types to Python native types"""
    if hasattr(obj, 'item'):  # numpy types
        return obj.item()
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return [json_safe_serialize(i) for i in obj]
        except (TypeError, AttributeError):
            pass
    return obj


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that supports numpy types"""
    def encode(self, o):
        return super().encode(json_safe_serialize(o))

    def default(self, o):
        if hasattr(o, 'item'):
            return o.item()
        if hasattr(o, '__iter__'):
            return list(o)
        return super().default(o)


class ArchitectAgent(BaseAgent):
    """
    Architect Agent

    Highly exploratory, responsible for:
    - Encoder/decoder selection
    - Latent space dimension design
    - Loss function combination
    - Generating initial task processing blueprint
    """
    
    def __init__(self, llm_config: Optional[Any] = None):
        system_prompt = """You are an Architect expert in deep learning for single-cell genomics.
Your role is to design innovative and effective neural network architectures.

Responsibilities:
1. Propose encoder/decoder architectures (VAE, GAN, Transformer, etc.)
2. Determine latent space dimensions based on data complexity
3. Design loss function combinations
4. Create the initial blueprint for the task
5. **CRITICAL: Consider hardware resource constraints when designing**

Guidelines:
- Be creative and explore different architectural choices
- Consider the specific characteristics of single-cell data (sparsity, high dimensionality)
- Propose multiple options when appropriate
- Provide clear reasoning for your choices
- **HARDWARE CONSTRAINTS (MANDATORY):**
  - Check available GPU memory and system RAM in data_info['hardware_info']
  - Design model architecture that fits within available memory
  - For limited GPU memory (< 8GB): use smaller batch sizes, fewer layers, smaller hidden dimensions
  - For limited system RAM: consider data streaming or chunked processing
  - Always specify memory requirements in your design
  - If data size exceeds available memory, propose sampling or batching strategies

**CRITICAL OUTPUT FORMAT:**
You MUST output ONLY a single valid JSON object. Do NOT use markdown code fences (no ```json or ```). Do NOT add any explanation or text before/after the JSON.

Output exactly this structure (adapt field names freely to fit your architecture, all required fields must be present):
{
  "architecture_type": "string",
  "model_config": {"your own key-value pairs describing the architecture"},
  "latent_dim": integer,
  "loss_functions": ["loss1", "loss2"],
  "memory_requirements": {"gpu_gb": number, "ram_gb": number},
  "batch_size_recommendation": integer,
  "reasoning": "string"
}

CORRECT example (output exactly this format, no code fences):
{"architecture_type": "VAE", "model_config": {"encoder_layers": [512, 256], "decoder_layers": [256, 512]}, "latent_dim": 64, "loss_functions": ["reconstruction", "kl"], "memory_requirements": {"gpu_gb": 4, "ram_gb": 8}, "batch_size_recommendation": 32, "reasoning": "VAE is suitable for single-cell data integration..."}

WRONG: ```json\n{"architecture_type": "..."}\n``` (do NOT use code fences)
WRONG: Here is my design:\n{"architecture_type": "..."} (do NOT add text before)
"""
        super().__init__("Architect", llm_config, system_prompt)
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Generate an initial architecture blueprint or a revised blueprint"""
        task_type = context.get('task_type', 'unknown')
        data_info = context.get('data_info', {})
        
        # Extract hardware information
        hardware_info = data_info.get('hardware_info', {})
        hw_text = ""
        if hardware_info:
            hw_parts = ["=== Hardware Resources ==="]
            if hardware_info.get('cpu_count'):
                hw_parts.append(f"CPU Cores: {hardware_info['cpu_count']}")
            if hardware_info.get('memory_gb'):
                hw_parts.append(f"System Memory: {hardware_info['memory_gb']} GB")
            if hardware_info.get('gpu_available'):
                hw_parts.append(f"GPU: {hardware_info.get('gpu_name', 'Unknown')}")
                if hardware_info.get('gpu_memory_gb'):
                    hw_parts.append(f"GPU Memory: {hardware_info['gpu_memory_gb']} GB")
            else:
                hw_parts.append("GPU: Not available")
            hw_text = "\n".join(hw_parts)
        
        # Check if this is a revision context
        current_blueprint = data_info.get('current_blueprint')
        feedback = data_info.get('feedback', [])
        task_description = data_info.get('task_description', '')
        
        if current_blueprint is not None and feedback:
            # ======= Revision mode =======
            # task_description was already populated from TASK_DESCRIPTIONS in _revise_blueprint
            if not task_description:
                task_description = f"Task: {task_type}"
            
            try:
                blueprint_json = json.dumps(current_blueprint, indent=2)
            except (TypeError, ValueError):
                blueprint_json = str(current_blueprint)
            
            feedback_text = '\n'.join(feedback) if isinstance(feedback, list) else str(feedback)
            
            prompt = f"""Revise the architecture blueprint for a {task_type} task based on expert feedback.
Address each expert's concerns — they evaluate from ML theory, engineering feasibility, and methodology comparison perspectives respectively.

Task: {task_description}

Current Blueprint:
{blueprint_json}

Expert Feedback:
{feedback_text}

**IMPORTANT: Design your architecture to fit within the available hardware resources.**
- If GPU memory is limited, use smaller batch sizes and fewer parameters
- If system RAM is limited, consider data sampling or streaming approaches
- Always specify estimated memory requirements in your design

**CRITICAL OUTPUT FORMAT:**
You MUST output ONLY a single valid JSON object. Do NOT use markdown code fences (no ```json or ```). Do NOT add any explanation or text before/after the JSON.

Output exactly this structure (adapt field names freely to fit your architecture, all required fields must be present):
{{
  "architecture_type": "string",
  "model_config": {{"your own key-value pairs describing the architecture"}},
  "latent_dim": integer,
  "loss_functions": ["loss1", "loss2"],
  "memory_requirements": {{"gpu_gb": number, "ram_gb": number}},
  "batch_size_recommendation": integer,
  "reasoning": "string explaining why this architecture addresses the feedback"
}}

Example:
{{"architecture_type": "VAE", "model_config": {{"encoder_layers": [512, 256], "decoder_layers": [256, 512]}}, "latent_dim": 64, "loss_functions": ["reconstruction", "kl"], "memory_requirements": {{"gpu_gb": 4, "ram_gb": 8}}, "batch_size_recommendation": 32, "reasoning": "..."}}

WRONG: ```json\n{{...}}\n``` (do NOT use code fences)
"""
        else:
            # ======= Initial generation mode =======
            if not task_description:
                task_description = data_info.get('task_description', f'Task: {task_type}')
            
            prompt = f"""Design a deep learning architecture for the following task:

{task_description}

Task Type: {task_type}

Data Information:
{json.dumps(json_safe_serialize(data_info), indent=2)}

{hw_text}

**IMPORTANT: Design your architecture to fit within the available hardware resources.**
- If GPU memory is limited, use smaller batch sizes and fewer parameters
- If system RAM is limited, consider data sampling or streaming approaches
- Always specify estimated memory requirements in your design

**CRITICAL OUTPUT FORMAT:**
You MUST output ONLY a single valid JSON object. Do NOT use markdown code fences (no ```json or ```). Do NOT add any explanation or text before/after the JSON.

Output exactly this structure (adapt field names freely to fit your architecture, all required fields must be present):
{{
  "architecture_type": "string",
  "model_config": {{"your own key-value pairs describing the architecture"}},
  "latent_dim": integer,
  "loss_functions": ["loss1", "loss2"],
  "memory_requirements": {{"gpu_gb": number, "ram_gb": number}},
  "batch_size_recommendation": integer,
  "reasoning": "string"
}}

Example:
{{"architecture_type": "VAE", "model_config": {{"encoder_layers": [512, 256], "decoder_layers": [256, 512]}}, "latent_dim": 64, "loss_functions": ["reconstruction", "kl"], "memory_requirements": {{"gpu_gb": 4, "ram_gb": 8}}, "batch_size_recommendation": 32, "reasoning": "..."}}

WRONG: ```json\n{{...}}\n``` (do NOT use code fences)
"""
        
        response_text = await self._call_llm(prompt, temperature=0.8)
        
        # Debug: print the first 500 characters of the raw response
        logger.debug(f"Architect raw response (first 500 chars): {response_text[:500]}")
        
        # Parse JSON response
        try:
            blueprint = self._extract_json(response_text)
            return AgentResponse(
                content=json.dumps(blueprint),
                reasoning=blueprint.get('reasoning', ''),
                confidence=0.7,
                suggestions=[],
                concerns=[]
            )
        except Exception as e:
            # Log raw response for debugging
            preview = response_text[:1000] if response_text else "(empty)"
            logger.warning(f"Failed to parse architect response: {e}")
            logger.warning(f"Raw response preview: {preview}")
            return AgentResponse(
                content=response_text,
                reasoning="Failed to parse structured output",
                confidence=0.3
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text, with fault-tolerant support for multiple formats"""
        import re

        # Strategy 0: Remove possible markdown code block wrapping
        stripped = text.strip()
        if stripped.startswith('```'):
            # Remove leading ```json or ``` markers
            match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```', stripped, re.DOTALL)
            if match:
                stripped = match.group(1).strip()
            else:
                # Only strip leading and trailing ```
                lines = stripped.split('\n')
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                stripped = '\n'.join(lines).strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove prefix text (e.g., "Here is the design:")
        # Find the position of the first {
        first_brace = stripped.find('{')
        if first_brace > 0:
            candidate = stripped[first_brace:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try raw_decode after removing prefix
        if first_brace > 0:
            candidate = stripped[first_brace:]
            decoder = json.JSONDecoder()
            try:
                result, _ = decoder.raw_decode(candidate)
                if isinstance(result, dict):
                    return result
            except:
                pass

        # Strategy 4: Try extracting from code blocks
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except:
                pass

        if "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except:
                pass

        # Strategy 5: Try extracting brace-delimited content
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 6: Use raw_decode to parse the first complete JSON object
        decoder = json.JSONDecoder()
        try:
            result, _ = decoder.raw_decode(text)
            if isinstance(result, dict):
                return result
        except:
            pass

        # Strategy 7: Find any {...} pattern
        matches = re.findall(r'\{[^{}]*\}', text)
        for match in matches:
            try:
                decoder2 = json.JSONDecoder()
                result, _ = decoder2.raw_decode(match)
                if isinstance(result, dict):
                    return result
            except:
                pass

        raise ValueError("No valid JSON found in response")


class ScientistAgent(BaseAgent):
    """
    Scientist Agent

    Responsible for theoretical validation:
    - Theoretical suitability of optimization algorithms
    - Convergence soundness of hyperparameters
    - Preventing logical flaws at the theoretical level
    """
    
    def __init__(self, llm_config: Optional[Any] = None):
        system_prompt = """You are a Scientist expert in machine learning theory and mathematical optimization.
Your role is to evaluate architectures from a **pure ML/theory perspective**.

Your unique evaluation angle — what other roles do NOT cover:
- Loss function: Does it have proper convergence guarantees? Vanishing gradients? Mode collapse?
- Optimization: Is the learning rate schedule theoretically sound? Will it converge?
- Capacity & expressiveness: Is latent dimension too small (underfitting) or too large (overfitting)?
- Regularization: Are the regularization terms mathematically justified?
- Gradient flow: Can gradients propagate through the entire network?

What you should NOT focus on (handled by other roles):
- Engineering feasibility → Engineer
- Comparison with other published methods → Critic
- Biological interpretation → Critic
- Code implementation details → Engineer

Output format: JSON with fields:
- position: "support" | "oppose" | "neutral"
- confidence: float (0-1), how certain are you about the theoretical analysis
- reasoning: str, your mathematical reasoning
- theoretical_concerns: list of str, specific theoretical flaws (be precise: name the concept, e.g., "VAE posterior collapse risk")
- suggestions: list of str, theoretically-grounded improvements (e.g., "use KL annealing to prevent posterior collapse")
"""
        super().__init__("Scientist", llm_config, system_prompt)
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Evaluate the theoretical soundness of the architecture"""
        blueprint = context.get('blueprint', {})
        data_info = context.get('data_info', {})
        task_description = data_info.get('task_description', 'Task evaluation')
        
        prompt = f"""Evaluate the theoretical soundness of the following architecture:

{task_description}

Architecture Blueprint:
{json.dumps(blueprint, indent=2)}

Data Information:
{json.dumps(json_safe_serialize(data_info), indent=2)}

Please provide your evaluation in JSON format.
"""
        
        response_text = await self._call_llm(prompt, temperature=0.6)
        
        try:
            evaluation = self._extract_json(response_text)
            return AgentResponse(
                content=json.dumps(evaluation),
                reasoning=evaluation.get('reasoning', ''),
                confidence=evaluation.get('confidence', 0.5),
                suggestions=evaluation.get('suggestions', []),
                concerns=evaluation.get('theoretical_concerns', [])
            )
        except Exception as e:
            logger.warning(f"Failed to parse scientist response: {e}")
            return AgentResponse(
                content=response_text,
                reasoning="Failed to parse structured output",
                confidence=0.3,
                concerns=["Parsing error"]
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text"""
        try:
            return json.loads(text)
        except:
            pass
        
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        raise ValueError("No JSON found in response")


class EngineerAgent(BaseAgent):
    """
    Engineer Agent

    Responsible for engineering feasibility evaluation:
    - Computational graph complexity
    - GPU memory usage
    - Code execution efficiency
    - Filtering out designs that cannot be implemented
    """
    
    def __init__(self, llm_config: Optional[Any] = None):
        system_prompt = """You are an Engineer expert in deep learning implementation and system optimization.
Your role is to evaluate the practical feasibility of proposed architectures.

Responsibilities:
1. Assess computational complexity
2. Estimate memory requirements
3. Evaluate code execution efficiency
4. Identify implementation challenges

Guidelines:
- Consider GPU memory constraints
- Check for efficient tensor operations
- Evaluate training/inference time
- Identify potential bottlenecks

Output format: JSON with fields:
- position: "support" | "oppose" | "neutral"
- confidence: float (0-1)
- reasoning: str
- concerns: list of str
- suggestions: list of str
- estimated_memory_gb: float
- estimated_training_time: str
"""
        super().__init__("Engineer", llm_config, system_prompt)
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Evaluate the engineering feasibility of the architecture"""
        blueprint = context.get('blueprint', {})
        data_info = context.get('data_info', {})
        task_description = data_info.get('task_description', 'Task evaluation')
        
        prompt = f"""Evaluate the implementation feasibility of the following architecture:

{task_description}

Architecture Blueprint:
{json.dumps(blueprint, indent=2)}

Data Information:
{json.dumps(json_safe_serialize(data_info), indent=2)}

Please provide your evaluation in JSON format.
"""
        
        response_text = await self._call_llm(prompt, temperature=0.6)
        
        try:
            evaluation = self._extract_json(response_text)
            return AgentResponse(
                content=json.dumps(evaluation),
                reasoning=evaluation.get('reasoning', ''),
                confidence=evaluation.get('confidence', 0.5),
                suggestions=evaluation.get('suggestions', []),
                concerns=evaluation.get('concerns', [])
            )
        except Exception as e:
            logger.warning(f"Failed to parse engineer response: {e}")
            return AgentResponse(
                content=response_text,
                reasoning="Failed to parse structured output",
                confidence=0.3,
                concerns=["Parsing error"]
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text"""
        try:
            return json.loads(text)
        except:
            pass
        
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        raise ValueError("No JSON found in response")


class CriticAgent(BaseAgent):
    """
    Critic Agent

    Responsible for critical review:
    - Whether batch effect correction is overloaded
    - Whether normalization is lost in the data pipeline
    - Whether parameters deviate significantly from the underlying data's sparse distribution
    - Ensuring the design does not violate biological priors
    """
    
    def __init__(self, llm_config: Optional[Any] = None):
        system_prompt = """You are a Critic expert in computational biology method comparison.
Your role is to evaluate architectures from a **methodology comparison perspective**.

Your unique evaluation angle — what other roles do NOT cover:
- Comparison: How does this design compare to established methods (e.g., SCENIC for GRN, scVI for integration, SciPlex for perturbation)?
- Limitations: What are the fundamental limitations of this approach? Known failure modes?
- Generalization: Will this approach work across different cell types, tissues, or perturbation types?
- Reproducibility: Are there hyperparameters that are hard to tune in practice?
- Benchmark gap: Does the design rely on assumptions not validated on real single-cell data?

What you should NOT focus on (handled by other roles):
- Mathematical convergence → Scientist
- GPU memory / compute cost → Engineer
- Loss function gradient flow → Scientist
- Batch normalization details → Engineer

Output format: JSON with fields:
- position: "support" | "oppose" | "neutral"
- confidence: float (0-1), how certain are you about the comparative analysis
- reasoning: str, your comparative reasoning (cite specific methods if relevant)
- limitations: list of str, fundamental limitations of the proposed design
- comparison_notes: list of str, how this compares to other known approaches
- suggestions: list of str, improvements based on known method strengths
"""
        super().__init__("Critic", llm_config, system_prompt)
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Review the architecture design from a methodology comparison perspective"""
        blueprint = context.get('blueprint', {})
        data_info = context.get('data_info', {})
        task_type = context.get('task_type', 'unknown')
        task_description = data_info.get('task_description', f'Task: {task_type}')
        
        prompt = f"""Compare the following architecture against established methods for {task_type} tasks.
Evaluate its strengths, weaknesses, and fundamental limitations.

{task_description}

Architecture Blueprint:
{json.dumps(blueprint, indent=2)}

Data Information:
{json.dumps(json_safe_serialize(data_info), indent=2)}

Please provide your comparative analysis in JSON format.
"""
        
        response_text = await self._call_llm(prompt, temperature=0.5)
        
        try:
            evaluation = self._extract_json(response_text)
            return AgentResponse(
                content=json.dumps(evaluation),
                reasoning=evaluation.get('reasoning', ''),
                confidence=evaluation.get('confidence', 0.5),
                suggestions=evaluation.get('suggestions', []),
                concerns=evaluation.get('limitations', []) + evaluation.get('comparison_notes', [])
            )
        except Exception as e:
            logger.warning(f"Failed to parse critic response: {e}")
            return AgentResponse(
                content=response_text,
                reasoning="Failed to parse structured output",
                confidence=0.3,
                concerns=["Parsing error"]
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text"""
        try:
            return json.loads(text)
        except:
            pass
        
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        raise ValueError("No JSON found in response")


# Agent factory function
def create_expert_agent(role: str, llm_config: Optional[Any] = None) -> BaseAgent:
    """Create an expert agent"""
    role_map = {
        'architect': ArchitectAgent,
        'scientist': ScientistAgent,
        'engineer': EngineerAgent,
        'critic': CriticAgent,
    }
    
    agent_class = role_map.get(role.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent role: {role}")
    
    return agent_class(llm_config)
