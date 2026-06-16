"""
Base Agent - Base Agent class

Base class for all expert agents, providing a unified LLM interface
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent response"""
    content: str
    reasoning: str = ""
    confidence: float = 0.5
    suggestions: List[str] = None
    concerns: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.concerns is None:
            self.concerns = []


class BaseAgent(ABC):
    """
    Base Agent class

    Base class for all expert agents, providing a unified LLM interaction interface
    """
    
    def __init__(
        self,
        role: str,
        llm_config: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.role = role
        self.llm_config = llm_config
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.llm_client = None
        
        logger.info(f"Agent initialized: {role}")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt"""
        return f"You are a {self.role} expert in computational biology and bioinformatics."
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Call the LLM"""
        # Should use the existing LLMClient
        # Simplified version; in practice, import Agent.llm.llm_client
        try:
            from Agent.llm.llm_client import LLMClient, LLMConfig
            
            if self.llm_client is None:
                if self.llm_config is None:
                    self.llm_config = LLMConfig.from_env()
                self.llm_client = LLMClient(self.llm_config)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"
    
    @abstractmethod
    async def process(
        self,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """
        Process input and return a response

        Args:
            context: Input context

        Returns:
            Agent response
        """
        pass
    
    async def close(self) -> None:
        """Close the LLM client's aiohttp session"""
        if self.llm_client is not None:
            try:
                await self.llm_client._reset_session()
            except Exception:
                pass
            self.llm_client = None
