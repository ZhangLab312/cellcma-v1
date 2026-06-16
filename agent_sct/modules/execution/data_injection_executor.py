"""
Data Injection Executor

Refactored from the existing Executor, implements a data-code separated execution model.
Pre-loads data into memory before code generation to avoid path/format issues.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import traceback
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Code execution result"""
    success: bool
    code: str
    output: str
    error: str
    execution_time: float
    variables: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    incomplete_reason: str = ""  # Reason for code completeness validation failure

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'execution_time': self.execution_time,
            'has_return_value': self.return_value is not None,
            'metrics': self.metrics,
            'incomplete_reason': self.incomplete_reason,
        }

    def extract_metrics(self) -> Dict[str, Any]:
        """Extract metrics dict from return_value or variables"""
        extracted = {}

        # First try extracting from return_value
        if self.return_value is not None:
            rv = self.return_value
            if isinstance(rv, dict):
                if 'metrics' in rv and isinstance(rv['metrics'], dict):
                    extracted.update(rv['metrics'])
                # Also support metrics at top level
                for key in ['nmi', 'ari', 'asw', 'auroc', 'auprc', 'f1',
                            'r2', 'mse', 'pcc_logfc', 'r2_de', 'mse_de', 'pcc_de']:
                    if key in rv and rv[key] is not None:
                        extracted[key] = rv[key]

        # Next try extracting from variables.result
        if not extracted and 'result' in self.variables:
            result_var = self.variables['result']
            if isinstance(result_var, dict):
                if 'metrics' in result_var and isinstance(result_var['metrics'], dict):
                    extracted.update(result_var['metrics'])
                else:
                    for key in ['nmi', 'ari', 'asw', 'auroc', 'auprc', 'f1',
                                'r2', 'mse', 'pcc_logfc', 'r2_de', 'mse_de', 'pcc_de']:
                        if key in result_var and result_var[key] is not None:
                            extracted[key] = result_var[key]

        self.metrics = extracted
        return extracted


class DataInjectionExecutor:
    """
    Data Injection Executor

    Core idea: Pre-load data into memory, code generation focuses on analysis logic.
    Execution environment formalized as: E = (X, Theta), where X is data tensor, Theta is program parameters.

    Refactored from existing Executor class, but changed to data injection mode.
    """
    
    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        timeout: int = 600,
        max_output_lines: int = 1000,
    ):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.timeout = timeout
        self.max_output_lines = max_output_lines
        
        # Execution environment state
        self._injected_data: Dict[str, Any] = {}
        self._execution_globals: Dict[str, Any] = {}
        self._execution_history: List[ExecutionResult] = []
        
        logger.info(f"DataInjectionExecutor initialized")
        logger.info(f"  Workspace: {self.workspace_dir}")
        logger.info(f"  Timeout: {timeout}s")
    
    def inject_data(
        self,
        data: Dict[str, Any],
        data_signature: Optional[str] = None,
    ) -> str:
        """
        Inject data into execution environment.

        Args:
            data: Data dictionary to inject, e.g., {'adata_rna': adata, 'adata_atac': adata}
            data_signature: Data signature

        Returns:
            injection_id: Injection ID
        """
        injection_id = data_signature or self._generate_id(str(data.keys()))
        
        # Store injected data
        self._injected_data[injection_id] = data
        
        # Prepare execution environment globals
        self._execution_globals = {
            # Standard libraries
            'np': np,
            'pd': pd,
            'torch': torch,
            'ad': ad,
            # Common PyTorch modules
            'nn': torch.nn,
            'F': torch.nn.functional,
            # Injected data
            **data,
        }
        
        logger.info(f"Data injected with ID: {injection_id}")
        logger.info(f"  Variables: {list(data.keys())}")
        
        return injection_id
    
    def inject_adata(
        self,
        adata: ad.AnnData,
        var_name: str = "adata",
        data_signature: Optional[str] = None,
    ) -> str:
        """
        Convenience method: inject a single AnnData object.

        Args:
            adata: AnnData object
            var_name: Variable name
            data_signature: Data signature

        Returns:
            injection_id: Injection ID
        """
        return self.inject_data(
            data={var_name: adata},
            data_signature=data_signature,
        )
    
    def inject_multi_omics(
        self,
        rna_adata: Optional[ad.AnnData] = None,
        atac_adata: Optional[ad.AnnData] = None,
        data_signature: Optional[str] = None,
    ) -> str:
        """
        Convenience method: inject multi-omics data.

        Args:
            rna_adata: RNA data
            atac_adata: ATAC data
            data_signature: Data signature

        Returns:
            injection_id: Injection ID
        """
        data = {}
        if rna_adata is not None:
            data['adata_rna'] = rna_adata
        if atac_adata is not None:
            data['adata_atac'] = atac_adata
        
        return self.inject_data(data, data_signature)
    
    def validate_code_completeness(
        self,
        code: str,
        exec_env: Dict[str, Any],
        return_value: Any,
    ) -> Tuple[bool, str]:
        """
        Verify whether the code contains complete analysis logic, not just data preparation code.

        Checks:
        1. Whether the code has actual analysis logic (not just class/function definitions)
        2. Whether a result dictionary is returned
        3. Whether there is actual computation/training/prediction code

        Returns:
            (is_complete, reason): (whether complete, reason for incompleteness)
        """
        code_lower = code.lower()

        # Check 1: whether a result dictionary is returned
        has_result = return_value is not None or 'result' in exec_env or '_return' in exec_env

        # Check 2: whether the code has actual analysis logic keywords
        # These keywords indicate the code contains actual model training/prediction/evaluation
        analysis_keywords = [
            'train', 'fit', 'model', 'predict', 'forward',
            'r2_score', 'mean_squared_error', 'pearsonr', 'accuracy_score',
            'loss', 'optimizer', 'adam', 'sgd',
            'nn.Module',
            'nn.Linear', 'nn.Sequential', 'nn.Embedding',
            'encoder', 'decoder', 'embed',
            'logfc', 'fold.change', 'de_gene',
            'compute_metrics', 'evaluate',
            'fit_transform', 'transform',
        ]

        found_keywords = []
        for kw in analysis_keywords:
            if kw.lower() in code_lower:
                found_keywords.append(kw)

        # Check 3: whether code only has Dimension Probe output without any actual computation
        has_analysis_logic = len(found_keywords) >= 2

        if has_result:
            return True, "result dictionary returned"

        if not has_analysis_logic:
            return False, (
                f"Code lacks analysis logic (no training/prediction/metrics computation). "
                f"No result dictionary returned. Found keywords: {found_keywords}"
            )

        return True, f"Code has analysis logic, found keywords: {found_keywords}"

    def execute(
        self,
        code: str,
        capture_output: bool = True,
        allow_shell: bool = False,
        allowed_modules: Optional[List[str]] = None,
        validate_completeness: bool = True,
    ) -> ExecutionResult:
        """
        Execute code.

        Args:
            code: Python code
            capture_output: Whether to capture output
            allow_shell: Whether to allow shell commands
            allowed_modules: List of allowed modules

        Returns:
            ExecutionResult: Execution result
        """
        import time
        start_time = time.time()

        # Prepare execution environment
        # Key fix: use the same dict as both globals and locals
        # This ensures class definitions and variables are accessible in the same namespace
        exec_env = self._execution_globals.copy()

        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        success = False
        error_msg = ""
        return_value = None

        # Log code information
        code_lines = code.split('\n')
        logger.info(f"="*60)
        logger.info(f"EXECUTING CODE")
        logger.info(f"="*60)
        logger.info(f"Code length: {len(code)} chars, {len(code_lines)} lines")
        logger.info(f"First 3 lines:\n" + '\n'.join(code_lines[:3]))
        logger.info(f"...")
        logger.info(f"Last 3 lines:\n" + '\n'.join(code_lines[-3:]))
        logger.info(f"="*60)


        # Auto-fix common missing imports
        code = self._auto_fix_imports(code)

        try:
            # Compile code
            logger.info("Compiling code...")
            compiled_code = compile(code, '<agent_generated>', 'exec')
            logger.info("Code compiled successfully")

            # Execute (using the same exec_env as both globals and locals)
            logger.info("Executing code...")
            with redirect_stdout(output_buffer) if capture_output else contextlib.nullcontext():
                with redirect_stderr(error_buffer) if capture_output else contextlib.nullcontext():
                    exec(compiled_code, exec_env, exec_env)

            # Extract return value (if any)
            if '_return' in exec_env:
                return_value = exec_env['_return']
                logger.info(f"Return value extracted: _return")
            elif 'result' in exec_env:
                return_value = exec_env['result']
                logger.info(f"Return value extracted: result")

            success = True
            execution_time = time.time() - start_time

            # Validate code completeness: ensure it's not just data preparation code
            if validate_completeness:
                is_complete, reason = self.validate_code_completeness(
                    code=code,
                    exec_env=exec_env,
                    return_value=return_value,
                )
                if not is_complete:
                    success = False
                    error_msg = (
                        f"Code is incomplete: {reason}\n\n"
                        f"The generated code lacks actual analysis logic (training/prediction/evaluation). "
                        f"It only contains data preparation code (e.g., Dataset class definition) "
                        f"without returning a `result` dictionary with metrics.\n\n"
                        f"Please generate complete code that includes:\n"
                        f"1. Model definition and training\n"
                        f"2. Prediction on held-out data\n"
                        f"3. Metric computation (r2, mse, pcc_logfc, etc.)\n"
                        f"4. Returning a `result = {{...}}` dictionary"
                    )
                    logger.warning(f"[VALIDATOR] Code completeness check FAILED: {reason}")
                    # Don't continue with output logging logic, build failure result directly
                else:
                    logger.info(f"[VALIDATOR] Code completeness check PASSED: {reason}")
            else:
                logger.info("[VALIDATOR] Completeness validation skipped")

            # Output execution result summary
            output_str = output_buffer.getvalue()
            error_str = error_buffer.getvalue()
            if success:  # Only log detailed output when truly successful
                logger.info(f"="*60)
                logger.info(f"EXECUTION SUCCESSFUL")
                logger.info(f"="*60)
                logger.info(f"Execution time: {execution_time:.2f}s")
                logger.info(f"Output length: {len(output_str)} chars")
                if output_str:
                    output_lines = output_str.strip().split('\n')
                    logger.info(f"Output lines: {len(output_lines)}")
                    logger.info(f"First 5 lines of output:\n" + '\n'.join(output_lines[:5]))
                    if len(output_lines) > 10:
                        logger.info(f"... ({len(output_lines)-10} lines omitted) ...")
                        logger.info(f"Last 5 lines of output:\n" + '\n'.join(output_lines[-5:]))
                if error_str:
                    logger.warning(f"Stderr output: {error_str[:500]}")
                logger.info(f"="*60)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"="*60)
            logger.error(f"EXECUTION FAILED")
            logger.error(f"="*60)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Execution time before failure: {execution_time:.2f}s")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"="*60)

            if 'CUDA' in str(e).upper() or 'cuda' in str(type(e).__name__).lower():
                logger.warning("CUDA error detected, attempting device state reset...")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass
                    torch.cuda.empty_cache()
                    logger.info("CUDA device state reset completed")

        # Build result
        result = ExecutionResult(
            success=success,
            code=code,
            output=output_buffer.getvalue()[:self.max_output_lines * 100],
            error=error_msg,
            execution_time=execution_time,
            variables={k: v for k, v in exec_env.items() if not k.startswith('_')},
            return_value=return_value,
            incomplete_reason=error_msg if (not success and "Code is incomplete" in error_msg) else "",
        )

        self._execution_history.append(result)

        return result
    
    def execute_with_retry(
        self,
        code: str,
        max_retries: int = 3,
        on_error_callback: Optional[callable] = None,
    ) -> ExecutionResult:
        """
        Execute with retry.

        Args:
            code: Python code
            max_retries: Maximum retry count
            on_error_callback: Error callback function for auto-fix

        Returns:
            ExecutionResult: Execution result
        """
        for attempt in range(max_retries):
            result = self.execute(code)
            
            if result.success:
                return result
            
            logger.warning(f"Execution attempt {attempt + 1}/{max_retries} failed")
            
            # If there's an error callback, try to fix
            if on_error_callback and attempt < max_retries - 1:
                try:
                    fixed_code = on_error_callback(code, result.error)
                    if fixed_code and fixed_code != code:
                        logger.info("Code fixed by callback, retrying...")
                        code = fixed_code
                        continue
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
        
        return result
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history"""
        return self._execution_history.copy()
    
    def clear_history(self):
        """Clear execution history"""
        self._execution_history.clear()
    
    def cleanup(self):
        """
        Clean up execution environment and free memory.

        Key fixes:
        1. Clean up temporary variables from the last execution in _execution_globals
        2. Only keep injected core data (np, pd, torch, etc.) and user-injected data
        3. Keep dimension probe variables (_GLOBAL_N_CELLS, etc.)
        4. Clear torch GPU cache
        5. Limit _execution_history length to prevent infinite accumulation
        """
        import gc as gc_module

        # Essential modules, injected data and dimension probe variables to keep
        essential_keys = {
            # Core modules
            'np', 'pd', 'torch', 'ad', 'nn', 'F',
            # User-injected data
            'adata', 'adata_rna', 'adata_atac',
            # Dimension probe variables (auto-set by _probe_data())
            '_GLOBAL_N_CELLS', '_GLOBAL_N_GENES',
            '_GLOBAL_CELL_TYPES', '_GLOBAL_N_CELL_TYPES',
            '_GLOBAL_BATCHES', '_GLOBAL_N_BATCHES',
            '_GLOBAL_PERTURBATIONS', '_GLOBAL_N_PERTURBATIONS',
            '_GLOBAL_RNA_N_CELLS', '_GLOBAL_RNA_N_GENES',
            '_GLOBAL_ATAC_N_CELLS', '_GLOBAL_ATAC_N_PEAKS',
        }

        # Filter keys to keep
        new_globals = {}
        for key in essential_keys:
            if key in self._execution_globals:
                new_globals[key] = self._execution_globals[key]

        # Re-add core modules (ensure correct types)
        new_globals.update({
            'np': np,
            'pd': pd,
            'torch': torch,
            'ad': ad,
            'nn': torch.nn,
            'F': torch.nn.functional,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                logger.warning("CUDA error detected during cleanup, resetting device state")
                torch.cuda.empty_cache()

        self._execution_globals = new_globals

        # Limit execution history length, keep only last 3 entries to avoid long-term memory accumulation
        if len(self._execution_history) > 3:
            self._execution_history = self._execution_history[-3:]
            logger.debug(f"Trimmed execution history to {len(self._execution_history)} entries")

        # Force garbage collection to release Python objects and C-level memory
        gc_module.collect()

        logger.debug(f"Execution environment cleaned. Remaining keys: {list(self._execution_globals.keys())}")

    def reset(self):
        """
        Fully reset the execution environment.
        Clear all injected data and execution history.
        """
        self._injected_data.clear()
        self._execution_globals.clear()
        self._execution_history.clear()
        logger.info("Executor fully reset")

    def _auto_fix_imports(self, code: str) -> str:
        """
        Auto-fix common classes/functions that are used but not imported in the code.
        When usage is detected without a corresponding import, automatically insert the import at the top of the file.
        """
        lines = code.split('\n')

        # Check if code already has these imports
        has_dataset_import = any(
            re.search(r'from\s+torch\.utils\.data\s+import\s+.*Dataset', l)
            for l in lines
        )
        has_dataloader_import = any(
            re.search(r'from\s+torch\.utils\.data\s+import\s+.*DataLoader', l)
            for l in lines
        )
        has_tensor_dataset_import = any(
            re.search(r'from\s+torch\.utils\.data\s+import\s+.*TensorDataset', l)
            for l in lines
        )

        # Check if code uses these classes without importing them
        uses_dataset = any(re.search(r'\bDataset\b', l) for l in lines)
        uses_dataloader = any(re.search(r'\bDataLoader\b', l) for l in lines)
        uses_tensor_dataset = any(re.search(r'\bTensorDataset\b', l) for l in lines)

        imports_to_add = []
        if uses_dataset and not has_dataset_import:
            imports_to_add.append('Dataset')
        if uses_dataloader and not has_dataloader_import:
            imports_to_add.append('DataLoader')
        if uses_tensor_dataset and not has_tensor_dataset_import:
            imports_to_add.append('TensorDataset')

        if not imports_to_add:
            return code

        import_str = f"from torch.utils.data import {', '.join(imports_to_add)}"
        logger.info(f"Auto-injecting missing import: {import_str}")

        # Find the first non-empty, non-comment line as the insertion point
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                insert_idx = i
                break
            insert_idx = i + 1

        lines.insert(insert_idx, import_str)
        return '\n'.join(lines)

    def get_injected_variables(self) -> List[str]:
        """Get the names of injected variables"""
        return list(self._execution_globals.keys())


def execute_code_with_data(
    code: str,
    data: Dict[str, Any],
    **kwargs
) -> ExecutionResult:
    """Convenience function: inject data and execute code"""
    executor = DataInjectionExecutor(**kwargs)
    executor.inject_data(data)
    return executor.execute(code)
