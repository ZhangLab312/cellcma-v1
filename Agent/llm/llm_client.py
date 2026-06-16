"""
LLM Client - LLM Client

Unified interface supporting multiple providers:
- OpenAI
- Anthropic (Claude)
- Local models (vLLM, Ollama, LM Studio, etc.)
- Azure OpenAI
- Other services compatible with OpenAI API format
"""

import os
import json
import re
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, replace
from enum import Enum
import aiohttp
import asyncio

# Load .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root (using override to ensure it takes precedence over system env vars)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    env_local_path = Path(__file__).resolve().parents[2] / ".env.local"
    if env_local_path.exists():
        load_dotenv(env_local_path, override=True)
except ImportError:
    pass


EMBEDDED_SILICONFLOW_API_KEY = "sk-rqodtmrbigoeythbbcmdwxlzyfnlwcouceaegcdmdcfjojri"
EMBEDDED_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


DEEPSEEK_FALLBACK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or ""
DEEPSEEK_FALLBACK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
DEEPSEEK_FALLBACK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek-chat"


class LLMProvider(Enum):
    """LLM provider"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"  # vLLM, Ollama, LM Studio, etc.
    CUSTOM = "custom"


class LLMProviderLockedError(RuntimeError):
    """Raised when a strict-provider run hits a provider failure and fallback is forbidden."""

    pass


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider = LLMProvider.OPENAI
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 64000
    timeout: int = 300
    retry_times: int = 3
    retry_delay: float = 1.0
    allow_cross_provider_fallback: bool = True
    
    # Provider-specific configuration
    anthropic_version: str = "2023-06-01"  # Used only by Anthropic
    azure_deployment: str = ""  # Used only by Azure
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables"""
        provider_str = (
            os.getenv('PERTURB_LLM_PROVIDER')
            or os.getenv('LLM_PROVIDER')
            or (
                'openai'
                if (
                    os.getenv('OPENAI_API_KEY')
                    or os.getenv('OPENAI_BASE_URL')
                    or os.getenv('SILICONFLOW_API_KEY')
                    or os.getenv('SILICONFLOW_BASE_URL')
                )
                else 'openai'
            )
        ).lower()
        
        provider_map = {
            'openai': LLMProvider.OPENAI,
            'anthropic': LLMProvider.ANTHROPIC,
            'claude': LLMProvider.ANTHROPIC,
            'azure': LLMProvider.AZURE,
            'local': LLMProvider.LOCAL,
            'custom': LLMProvider.CUSTOM,
        }
        
        provider = provider_map.get(provider_str, LLMProvider.OPENAI)

        if provider == LLMProvider.OPENAI:
            base_url = (
                os.getenv('OPENAI_BASE_URL')
                or os.getenv('SILICONFLOW_BASE_URL')
                or os.getenv('LLM_BASE_URL')
                or EMBEDDED_SILICONFLOW_BASE_URL
            )
            api_key = (
                os.getenv('OPENAI_API_KEY')
                or os.getenv('SILICONFLOW_API_KEY')
                or os.getenv('LLM_API_KEY')
                or EMBEDDED_SILICONFLOW_API_KEY
            )
            # Prefer LLM_MODEL; fall back to other variables if not set
            model = os.getenv('LLM_MODEL')
            if not model:
                model = (
                    os.getenv('OPENAI_MODEL_NAME')
                    or os.getenv('SILICONFLOW_MODEL_NAME')
                    or 'gpt-4'
                )
        elif provider == LLMProvider.CUSTOM and provider_str == 'deepseek':
            base_url = (
                os.getenv('DEEPSEEK_BASE_URL')
                or os.getenv('LLM_BASE_URL')
                or 'https://api.deepseek.com/v1'
            )
            api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY') or ''
            model = os.getenv('DEEPSEEK_MODEL_NAME') or os.getenv('LLM_MODEL') or 'deepseek-chat'
        else:
            base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
            api_key = os.getenv('LLM_API_KEY', '')
            model = os.getenv('LLM_MODEL', 'gpt-4')

        return cls(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '4000')),
            timeout=int(os.getenv('LLM_TIMEOUT', '120')),
            allow_cross_provider_fallback=(
                os.getenv("ALLOW_CROSS_PROVIDER_FALLBACK", "true").lower() in ("true", "1", "yes")
            ),
        )
    
    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4") -> 'LLMConfig':
        """Create OpenAI configuration"""
        return cls(
            provider=LLMProvider.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model
        )
    
    @classmethod
    def anthropic(cls, api_key: str, model: str = "claude-3-sonnet-20240229") -> 'LLMConfig':
        """Create Anthropic configuration"""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key=api_key,
            model=model
        )
    
    @classmethod
    def local(cls, base_url: str = "http://localhost:8000/v1", model: str = "local-model") -> 'LLMConfig':
        """Create local model configuration"""
        return cls(
            provider=LLMProvider.LOCAL,
            base_url=base_url,
            api_key="not-needed",  # Local models typically don't need an API key
            model=model
        )
    
    def with_model(self, model: str) -> 'LLMConfig':
        """
        Switch model (keeping other configuration unchanged)

        Args:
            model: New model name

        Returns:
            New configuration object
        """
        import copy
        new_config = copy.copy(self)
        new_config.model = model
        return new_config
    
    def with_temperature(self, temperature: float) -> 'LLMConfig':
        """
        Switch temperature parameter (keeping other configuration unchanged)

        Args:
            temperature: New temperature value

        Returns:
            New configuration object
        """
        import copy
        new_config = copy.copy(self)
        new_config.temperature = temperature
        return new_config
    
    @classmethod
    def openai_gpt4(cls, api_key: str) -> 'LLMConfig':
        """Create OpenAI GPT-4 configuration"""
        return cls.openai(api_key=api_key, model="gpt-4")
    
    @classmethod
    def openai_gpt4_turbo(cls, api_key: str) -> 'LLMConfig':
        """Create OpenAI GPT-4 Turbo configuration"""
        return cls.openai(api_key=api_key, model="gpt-4-turbo-preview")
    
    @classmethod
    def openai_gpt35(cls, api_key: str) -> 'LLMConfig':
        """Create OpenAI GPT-3.5 configuration"""
        return cls.openai(api_key=api_key, model="gpt-3.5-turbo")
    
    @classmethod
    def anthropic_opus(cls, api_key: str) -> 'LLMConfig':
        """Create Anthropic Claude-3 Opus configuration"""
        return cls.anthropic(api_key=api_key, model="claude-3-opus-20240229")
    
    @classmethod
    def anthropic_sonnet(cls, api_key: str) -> 'LLMConfig':
        """Create Anthropic Claude-3 Sonnet configuration"""
        return cls.anthropic(api_key=api_key, model="claude-3-sonnet-20240229")
    
    @classmethod
    def anthropic_haiku(cls, api_key: str) -> 'LLMConfig':
        """Create Anthropic Claude-3 Haiku configuration"""
        return cls.anthropic(api_key=api_key, model="claude-3-haiku-20240307")


class LLMClient:
    """
    LLM Client

    Unified interface for accessing LLMs from different providers
    """
    
    _instances: "weakref.WeakSet[LLMClient]" = weakref.WeakSet()

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client

        Args:
            config: LLM configuration; if None, loads from environment variables
        """
        self.config = config or LLMConfig.from_env()
        self.session: Optional[aiohttp.ClientSession] = None
        self.__class__._instances.add(self)
        self._usage_tracker: Dict[str, Any] = {
            "run_started_at": time.time(),
            "request_attempts": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "models": {},
            "last_error": None,
        }
        
        # Validate configuration
        if not self.config.api_key and self.config.provider != LLMProvider.LOCAL:
            raise ValueError(f"API key is required for {self.config.provider.value}")

    def _normalize_usage(self, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _record_usage(
        self,
        *,
        success: bool,
        usage: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        tracker = self._usage_tracker
        tracker["request_attempts"] += 1
        if success:
            tracker["successful_responses"] += 1
        else:
            tracker["failed_responses"] += 1
            tracker["last_error"] = error

        normalized = self._normalize_usage(usage)
        tracker["prompt_tokens"] += normalized["prompt_tokens"]
        tracker["completion_tokens"] += normalized["completion_tokens"]
        tracker["total_tokens"] += normalized["total_tokens"]

        model_key = self.config.model or "unknown"
        per_model = tracker["models"].setdefault(
            model_key,
            {
                "provider": self.config.provider.value,
                "base_url": self.config.base_url,
                "request_attempts": 0,
                "successful_responses": 0,
                "failed_responses": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "last_request_id": None,
                "last_status_code": None,
                "last_error": None,
            },
        )
        per_model["request_attempts"] += 1
        if success:
            per_model["successful_responses"] += 1
        else:
            per_model["failed_responses"] += 1
            per_model["last_error"] = error
        per_model["prompt_tokens"] += normalized["prompt_tokens"]
        per_model["completion_tokens"] += normalized["completion_tokens"]
        per_model["total_tokens"] += normalized["total_tokens"]
        if request_id:
            per_model["last_request_id"] = request_id
        if status_code is not None:
            per_model["last_status_code"] = status_code

    def get_usage_summary(self) -> Dict[str, Any]:
        tracker = dict(self._usage_tracker)
        tracker["run_elapsed_seconds"] = max(0.0, time.time() - tracker["run_started_at"])
        tracker["provider"] = self.config.provider.value
        tracker["base_url"] = self.config.base_url
        tracker["model"] = self.config.model
        return tracker

    def _build_session(self) -> aiohttp.ClientSession:
        """Build a fresh aiohttp session for each clean attempt."""
        connector = aiohttp.TCPConnector(
            use_dns_cache=False,
            ttl_dns_cache=0,
            enable_cleanup_closed=True,
        )
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            trust_env=True,
            connector=connector,
        )

    async def _reset_session(self) -> None:
        """Close a stale session so the next retry re-resolves and reconnects."""
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass
            finally:
                self.session = None

    @classmethod
    async def shutdown_all_sessions(cls) -> None:
        """Best-effort shutdown for every live LLMClient session in this process."""
        clients = [client for client in list(cls._instances) if client is not None]
        for client in clients:
            try:
                await client._reset_session()
            except Exception:
                pass
        # Give aiohttp transports a short window to flush close callbacks.
        await asyncio.sleep(0.2)

    def _build_deepseek_fallback_config(self) -> Optional[LLMConfig]:
        """Return a DeepSeek-compatible fallback config when the primary provider is exhausted."""
        if not self.config.allow_cross_provider_fallback:
            return None
        current_base_url = (self.config.base_url or "").lower()
        if "api.deepseek.com" in current_base_url:
            return None
        if not DEEPSEEK_FALLBACK_API_KEY:
            return None
        return replace(
            self.config,
            provider=LLMProvider.CUSTOM,
            base_url=DEEPSEEK_FALLBACK_BASE_URL,
            api_key=DEEPSEEK_FALLBACK_API_KEY,
            model=DEEPSEEK_FALLBACK_MODEL,
            allow_cross_provider_fallback=False,
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = self._build_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Chat completion

        Args:
            messages: Message list in format [{"role": "user", "content": "..."}, ...]
            temperature: Temperature parameter (overrides configuration)
            max_tokens: Maximum token count (overrides configuration)
            stream: Whether to use streaming output

        Returns:
            Text generated by the LLM
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Select API call method based on provider
        if self.config.provider == LLMProvider.ANTHROPIC:
            return await self._chat_anthropic(messages, temp, max_tok, stream)
        else:
            # OpenAI format (including OpenAI, Azure, Local, Custom)
            return await self._chat_openai_format(messages, temp, max_tok, stream)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Synchronous compatibility wrapper for legacy agents.

        The perturbation agents currently call `llm.generate(...)` while this
        client is async-first. Keep that call shape working by routing it to
        `chat_completion`.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stale_session = self.session
        self.session = None

        async def _run() -> str:
            original_model = self.config.model
            if model:
                self.config.model = model
            try:
                if stale_session:
                    try:
                        await stale_session.close()
                    except Exception:
                        pass
                return await self.chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            finally:
                if self.session:
                    await self.session.close()
                    self.session = None
                self.config.model = original_model

        return asyncio.run(_run())

    def generate_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Alias for generate(). Provided so that wrappers expecting this exact name
        (e.g. ProgressiveCodeGenerator) work without modification.
        """
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    async def _chat_openai_format(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> str:
        """Call OpenAI-format API"""
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Azure requires special handling
        if self.config.provider == LLMProvider.AZURE:
            url = f"{self.config.base_url}/openai/deployments/{self.config.azure_deployment}/chat/completions?api-version=2023-05-15"
            headers = {
                "api-key": self.config.api_key,
                "Content-Type": "application/json"
            }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        def _provider_locked(message: str) -> LLMProviderLockedError:
            return LLMProviderLockedError(
                f"Provider-locked request failed for {self.config.base_url} / {self.config.model}: {message}"
            )
        
        def _parse_sse_stream(content: bytes) -> str:
            """
            Parse SSE streaming response, supporting multiple format variants.

            Supported formats:
            - data: {"choices": [...]}
            - data:{"choices":[...]}
            - data: [DONE]
            - [DONE]
            - (as well as empty lines, comment lines, etc.)

            Note: SSE line delimiter is CRLF (\r\n), and the data field value may contain
            newlines, so simple \n splitting is insufficient. Use line buffering to accumulate
            until an empty line is encountered.
            """
            full_content = []
            text = content.decode('utf-8', errors='replace')

            # Split by CRLF into "raw lines" (SSE spec: each event ends with an empty line)
            # But embedded newlines in content are not treated as line separators (JSON escaped as \n)
            raw_lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

            i = 0
            while i < len(raw_lines):
                raw_line = raw_lines[i]

                # Skip empty lines (event end markers)
                if not raw_line.strip():
                    i += 1
                    continue

                # Event lines must start with "data:"
                if not raw_line.startswith('data:'):
                    i += 1
                    continue

                data_part = raw_line[5:].strip()

                # [DONE] end marker
                if data_part == '[DONE]' or data_part == 'DONE':
                    break

                # If data_part starts with { but JSON is incomplete (may span multiple lines)
                # Try to merge subsequent lines until JSON is closed
                if data_part.startswith('{') and not _looks_complete(data_part):
                    j = i + 1
                    accumulated = data_part
                    while j < len(raw_lines):
                        next_line = raw_lines[j].rstrip()
                        if not next_line:  # Empty line: current event ends
                            break
                        accumulated += '\n' + next_line
                        if _looks_complete(accumulated):
                            data_part = accumulated
                            i = j  # Let outer loop continue from next event
                            break
                        j += 1

                # Try to parse
                try:
                    chunk_data = json.loads(data_part)
                    if 'choices' in chunk_data:
                        delta = chunk_data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            full_content.append(delta['content'])
                    elif 'message' in chunk_data and 'content' in chunk_data['message']:
                        # Non-streaming format but content-type is SSE
                        return chunk_data['message']['content']
                except json.JSONDecodeError:
                    # Try to find and extract JSON content directly
                    match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', data_part)
                    if match:
                        content_val = match.group(1)
                        content_val = content_val.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        full_content.append(content_val)
                    elif 'DONE' in data_part.upper():
                        break

                i += 1

            return ''.join(full_content)

        def _looks_complete(s: str) -> bool:
            """Check if a string looks like a complete JSON object/array"""
            s = s.strip()
            if not s:
                return False
            if s.startswith('{'):
                return s.count('{') == s.count('}')
            if s.startswith('['):
                return s.count('[') == s.count(']')
            return False

        # Retry mechanism
        for attempt in range(self.config.retry_times):
            try:
                if not self.session:
                    self.session = self._build_session()
                
                async with self.session.post(url, headers=headers, json=payload) as response:

                    if response.status == 200:
                        # Unified reading: use response.text() to get the full raw text
                        # This avoids the issue of body being consumed multiple times
                        raw_text = await response.text()

                        # Determine handling based on the stream request parameter (not Content-Type)
                        # Content-Type may be inaccurate, but the stream request parameter is explicit
                        if stream:
                            # Streaming SSE format
                            parsed = _parse_sse_stream(raw_text.encode('utf-8'))
                            if not parsed.strip():
                                print(f"[SSE] WARNING: parsed empty, raw_len={len(raw_text)}, preview={raw_text[:300]!r}")
                            self._record_usage(
                                success=True, usage=None,
                                request_id=response.headers.get("x-request-id"),
                                status_code=response.status,
                            )
                            return parsed

                        # Non-streaming JSON format
                        try:
                            data = json.loads(raw_text)
                        except json.JSONDecodeError as e:
                            print(f"[SSE] WARNING: failed to parse response as JSON: {e}, raw={raw_text[:300]!r}")
                            self._record_usage(success=False, error=f"json parse failed: {e}",
                                request_id=response.headers.get("x-request-id"),
                                status_code=response.status)
                            return ""

                        self._record_usage(
                            success=True,
                            usage=data.get("usage"),
                            request_id=response.headers.get("x-request-id"),
                            status_code=response.status,
                        )
                        raw_content = data['choices'][0]['message']['content']

                        # Defensive: if content itself is a JSON string (escaped double-layer JSON), parse it
                        stripped = raw_content.strip()
                        if stripped.startswith('{') or stripped.startswith('['):
                            try:
                                inner = json.loads(stripped)
                                return json.dumps(inner)
                            except json.JSONDecodeError:
                                pass

                        return raw_content
                    else:
                        error_text = await response.text()
                        self._record_usage(
                            success=False,
                            request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
                            status_code=response.status,
                            error=f"status={response.status}, body={error_text[:500]}",
                        )
                        if (
                            response.status == 403
                            and "insufficient" in error_text.lower()
                        ):
                            fallback_config = self._build_deepseek_fallback_config()
                            if fallback_config is not None:
                                original_config = self.config
                                await self._reset_session()
                                self.config = fallback_config
                                try:
                                    return await self._chat_openai_format(
                                        messages,
                                        temperature,
                                        max_tokens,
                                        stream,
                                    )
                                finally:
                                    self.config = original_config
                        if (
                            not self.config.allow_cross_provider_fallback
                            and response.status in {402, 403}
                        ):
                            raise _provider_locked(f"status={response.status}, body={error_text}")
                        raise Exception(f"API error: {response.status}, {error_text}")

            except asyncio.TimeoutError:
                await self._reset_session()
                err_msg = f"request timeout after {self.config.timeout}s"
                self._record_usage(success=False, error=err_msg)
                if not self.config.allow_cross_provider_fallback:
                    raise _provider_locked(err_msg)
                if attempt < self.config.retry_times - 1:
                    print(f"Attempt {attempt + 1} failed: {err_msg}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise Exception(
                        f"Failed after {self.config.retry_times} attempts: {err_msg}"
                    )
            except (
                aiohttp.ClientConnectorError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientSSLError,
                aiohttp.ServerDisconnectedError,
            ) as e:
                await self._reset_session()
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self._record_usage(success=False, error=err_msg)
                if not self.config.allow_cross_provider_fallback:
                    raise _provider_locked(err_msg)
                if attempt < self.config.retry_times - 1:
                    print(f"Attempt {attempt + 1} failed: {err_msg}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise Exception(
                        f"Failed after {self.config.retry_times} attempts: {err_msg}"
                    )
            except LLMProviderLockedError:
                await self._reset_session()
                raise
            except Exception as e:
                await self._reset_session()
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self._record_usage(success=False, error=err_msg)
                if attempt < self.config.retry_times - 1:
                    print(f"Attempt {attempt + 1} failed: {err_msg}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise Exception(
                        f"Failed after {self.config.retry_times} attempts: {err_msg}"
                    )
        
        return ""
    
    async def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> str:
        """Call Anthropic API"""
        url = f"{self.config.base_url}/messages"
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.config.anthropic_version,
            "Content-Type": "application/json"
        }
        
        # Convert message format
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages
        }
        
        if system_message:
            payload["system"] = system_message

        # Retry mechanism
        for attempt in range(self.config.retry_times):
            try:
                if not self.session:
                    self.session = self._build_session()

                async with self.session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._record_usage(
                            success=True,
                            usage=data.get("usage"),
                            request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
                            status_code=response.status,
                        )
                        return data['content'][0]['text']
                    else:
                        error_text = await response.text()
                        self._record_usage(
                            success=False,
                            request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
                            status_code=response.status,
                            error=f"status={response.status}, body={error_text[:500]}",
                        )
                        raise Exception(f"API error: {response.status}, {error_text}")

            except asyncio.TimeoutError:
                await self._reset_session()
                err_msg = f"request timeout after {self.config.timeout}s"
                self._record_usage(success=False, error=err_msg)
                if attempt < self.config.retry_times - 1:
                    print(f"Attempt {attempt + 1} failed: {err_msg}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise Exception(
                        f"Failed after {self.config.retry_times} attempts: {err_msg}"
                    )
            except Exception as e:
                await self._reset_session()
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self._record_usage(success=False, error=err_msg)
                if attempt < self.config.retry_times - 1:
                    print(f"Attempt {attempt + 1} failed: {err_msg}, retrying...")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise Exception(
                        f"Failed after {self.config.retry_times} attempts: {err_msg}"
                    )
        
        return ""
    
    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> str:
        """
        Generate code

        Args:
            prompt: Code generation prompt
            language: Programming language
            context: Context information

        Returns:
            Generated code
        """
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {language} programmer. Generate clean, well-documented code."
            },
            {
                "role": "user",
                "content": f"{context}\n\n{prompt}" if context else prompt
            }
        ]
        
        return await self.chat_completion(messages, temperature=0.3)
    
    async def analyze_architecture(
        self,
        task_description: str,
        data_info: Dict[str, Any],
        previous_attempts: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze and recommend architecture

        Args:
            task_description: Task description
            data_info: Data information
            previous_attempts: Previous attempts (for Reflexion)

        Returns:
            Architecture recommendation
        """
        context = f"""
Task: {task_description}

Data Information:
- Complexity: {data_info.get('complexity', 'unknown')}
- Number of cells: {data_info.get('n_cells', 'unknown')}
- Sparsity: {data_info.get('sparsity', 'unknown')}

Available architectures:
1. VAE (Variational Autoencoder) - Good for general purpose, handles sparsity well
2. Contrastive Learning - Good for alignment, works with sparse data
3. Adversarial Alignment - Good for batch correction, strong alignment
4. Optimal Transport - Good for distribution matching
5. Hybrid - Combines multiple approaches

Please recommend the best architecture and explain why.
"""
        
        if previous_attempts:
            context += "\n\nPrevious attempts:\n"
            for i, attempt in enumerate(previous_attempts):
                context += f"Attempt {i+1}: {attempt}\n"
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in multi-omics integration and deep learning architecture design. Provide detailed recommendations in JSON format."
            },
            {
                "role": "user",
                "content": context
            }
        ]
        
        response = await self.chat_completion(messages, temperature=0.5)
        
        # Try to parse JSON
        try:
            # Extract JSON part
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            # If parsing fails, return text
            return {
                "recommendation": response,
                "architecture": "hybrid",
                "confidence": 0.5
            }
    
    async def evaluate_result(
        self,
        result_description: str,
        metrics: Dict[str, float],
        criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate result

        Args:
            result_description: Result description
            metrics: Evaluation metrics
            criteria: Evaluation criteria

        Returns:
            Evaluation result
        """
        context = f"""
Please evaluate the following result:

{result_description}

Metrics:
{json.dumps(metrics, indent=2)}

Evaluation Criteria:
{chr(10).join(f"- {c}" for c in criteria)}

Provide your evaluation in JSON format with:
- overall_score (0-1)
- passed_criteria (list)
- failed_criteria (list)
- suggestions (list)
- detailed_feedback (object)
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator for bioinformatics and machine learning results. Be critical but fair."
            },
            {
                "role": "user",
                "content": context
            }
        ]
        
        response = await self.chat_completion(messages, temperature=0.3)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            return {
                "overall_score": 0.5,
                "passed_criteria": [],
                "failed_criteria": criteria,
                "suggestions": [response],
                "detailed_feedback": {"error": "Failed to parse JSON"}
            }
