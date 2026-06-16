"""
Enhanced Logging System for Agent Monitoring

Features:
1. Structured log recording
2. Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
3. Dual output to file and console
4. JSON format logs for easy analysis
5. Agent decision process tracking
6. Detailed experiment result recording
7. [NEW] Auto-flush to prevent log loss
"""

import logging
import sys
import os
import json
import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
import traceback


class AgentLogger:
    """Logger dedicated to agent monitoring"""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Log file directory
            level: Log level
            enable_console: Whether to output to console
            enable_file: Whether to output to file
            enable_json: Whether to generate JSON format logs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers

        # Create formatters
        self.detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.simple_formatter = logging.Formatter(
            fmt='%(levelname)-8s | %(message)s'
        )

        # Console output
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self.simple_formatter)
            self.logger.addHandler(console_handler)

        # File output (text format)
        if enable_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(self.detailed_formatter)
            self.logger.addHandler(file_handler)
            self.log_file_path = log_file

        # JSON format logs (for easy analysis)
        if enable_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_log_file = self.log_dir / f"{name}_{timestamp}.jsonl"
            self.json_handler = JSONLogHandler(json_log_file)
            self.json_handler.setLevel(level)
            self.logger.addHandler(self.json_handler)
            self.json_log_file_path = json_log_file

    def debug(self, message: str, **kwargs):
        """DEBUG level log"""
        self.logger.debug(message, extra={'data': kwargs})
        self._flush_handlers()

    def info(self, message: str, **kwargs):
        """INFO level log"""
        self.logger.info(message, extra={'data': kwargs})
        self._flush_handlers()

    def warning(self, message: str, **kwargs):
        """WARNING level log"""
        self.logger.warning(message, extra={'data': kwargs})
        self._flush_handlers()

    def error(self, message: str, **kwargs):
        """ERROR level log"""
        self.logger.error(message, extra={'data': kwargs})
        self._flush_handlers()

    def critical(self, message: str, **kwargs):
        """CRITICAL level log"""
        self.logger.critical(message, extra={'data': kwargs})
        self._flush_handlers()

    def _flush_handlers(self):
        """Immediately flush all handlers to ensure logs are written."""
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except:
                pass  # Ignore flush errors

    def close(self):
        """Close the logger and release resources."""
        self._flush_handlers()
        for handler in self.logger.handlers:
            try:
                handler.close()
            except:
                pass
        self.logger.handlers.clear()

    # ================== Specialized Logging Methods ==================

    def log_agent_decision(
        self,
        agent_name: str,
        decision_type: str,
        decision: str,
        reasoning: str,
        context: Optional[Dict] = None
    ):
        """Log an agent decision."""
        self.info(
            f"[{agent_name}] Decision: {decision_type}",
            decision=decision,
            reasoning=reasoning,
            context=context or {},
            event_type="agent_decision"
        )

    def log_experiment_start(
        self,
        experiment_name: str,
        parameters: Dict,
        **kwargs
    ):
        """Log the start of an experiment."""
        self.info(
            f"🚀 Experiment Started: {experiment_name}",
            experiment=experiment_name,
            parameters=parameters,
            event_type="experiment_start",
            **kwargs
        )

    def log_experiment_end(
        self,
        experiment_name: str,
        success: bool,
        results: Dict,
        **kwargs
    ):
        """Log the end of an experiment."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.info(
            f"{status} | Experiment: {experiment_name}",
            experiment=experiment_name,
            success=success,
            results=results,
            event_type="experiment_end",
            **kwargs
        )

    def log_error_with_traceback(
        self,
        error_message: str,
        exception: Optional[Exception] = None
    ):
        """Log an error with traceback."""
        error_data = {
            "error_message": error_message,
            "traceback": traceback.format_exc() if exception else None
        }
        self.error(
            f"❌ ERROR: {error_message}",
            **error_data
        )

    def log_memory_operation(
        self,
        operation: str,
        memory_type: str,
        details: Dict
    ):
        """Log a memory system operation."""
        self.debug(
            f"[Memory] {operation}: {memory_type}",
            operation=operation,
            memory_type=memory_type,
            details=details,
            event_type="memory_operation"
        )

    def log_code_generation(
        self,
        agent_name: str,
        success: bool,
        code_length: int,
        error: Optional[str] = None
    ):
        """Log code generation."""
        status = "✅" if success else "❌"
        self.info(
            f"{status} [{agent_name}] Code Generation: {code_length} chars",
            agent=agent_name,
            success=success,
            code_length=code_length,
            error=error,
            event_type="code_generation"
        )

    def log_execution_attempt(
        self,
        attempt: int,
        max_attempts: int,
        success: bool,
        duration: Optional[float] = None,
        **kwargs
    ):
        """Log an execution attempt."""
        status = "✅" if success else "❌"
        duration_str = f" ({duration:.2f}s)" if duration else ""

        self.info(
            f"{status} Execution Attempt {attempt}/{max_attempts}{duration_str}",
            attempt=attempt,
            max_attempts=max_attempts,
            success=success,
            duration=duration,
            event_type="execution_attempt",
            **kwargs
        )

    @contextmanager
    def log_phase(self, phase_name: str, **kwargs):
        """Context manager: log the start and end of a phase."""
        self.info(f"▶️  Phase Started: {phase_name}", event_type="phase_start", **kwargs)
        start_time = datetime.now()

        try:
            yield
            duration = (datetime.now() - start_time).total_seconds()
            self.info(
                f"✅ Phase Completed: {phase_name} ({duration:.2f}s)",
                event_type="phase_end",
                duration=duration,
                **kwargs
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.error(
                f"❌ Phase Failed: {phase_name} ({duration:.2f}s)",
                event_type="phase_failed",
                duration=duration,
                error=str(e),
                **kwargs
            )
            raise

    def get_summary(self) -> Dict:
        """Get a summary of the logger."""
        return {
            "logger_name": self.name,
            "log_file": str(getattr(self, 'log_file_path', 'N/A')),
            "json_log_file": str(getattr(self, 'json_log_file_path', 'N/A')),
            "log_dir": str(self.log_dir)
        }


class JSONLogHandler(logging.Handler):
    """JSON format log handler."""

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename
        self.file = open(filename, 'a', encoding='utf-8')

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            # Add extra data
            if hasattr(record, 'data') and record.data:
                log_entry.update(record.data)

            # Write in JSONL format
            self.file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            self.file.flush()

        except Exception:
            self.handleError(record)

    def close(self):
        self.file.close()
        super().close()


# ================== Global Loggers ==================

_loggers = {}


def cleanup_all_loggers():
    """Clean up all loggers to ensure logs are written."""
    for name, logger_obj in _loggers.items():
        try:
            logger_obj._flush_handlers()
        except:
            pass


# Register cleanup on exit
atexit.register(cleanup_all_loggers)


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> AgentLogger:
    """
    Get or create a logger.

    Args:
        name: Logger name
        log_dir: Log directory
        level: Log level

    Returns:
        AgentLogger instance
    """
    if name not in _loggers:
        _loggers[name] = AgentLogger(
            name=name,
            log_dir=log_dir,
            level=level
        )
    return _loggers[name]


def setup_experiment_logging(
    experiment_name: str,
    log_dir: str = "logs"
) -> AgentLogger:
    """
    Set up a dedicated logger for an experiment.

    Args:
        experiment_name: Experiment name
        log_dir: Log directory

    Returns:
        AgentLogger instance
    """
    logger = get_logger(experiment_name, log_dir=log_dir)
    logger.info("="*60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)
    return logger


# ================== Log Analysis Tools ==================

class LogAnalyzer:
    """Log analysis tool."""

    def __init__(self, json_log_file: Path):
        self.json_log_file = json_log_file
        self.logs = self._load_logs()

    def _load_logs(self) -> list:
        """Load JSON logs."""
        logs = []
        try:
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            print(f"Failed to load log file: {e}")
        return logs

    def get_summary(self) -> Dict:
        """Get a summary of the logs."""
        total_logs = len(self.logs)
        level_counts = {}
        event_types = {}

        for log in self.logs:
            level = log.get('level', 'UNKNOWN')
            level_counts[level] = level_counts.get(level, 0) + 1

            event_type = log.get('event_type', 'none')
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_logs": total_logs,
            "level_counts": level_counts,
            "event_types": event_types
        }

    def get_errors(self) -> list:
        """Get all error logs."""
        return [log for log in self.logs if log.get('level') == 'ERROR']

    def get_agent_decisions(self) -> list:
        """Get all agent decisions."""
        return [log for log in self.logs if log.get('event_type') == 'agent_decision']

    def get_experiments(self) -> Dict:
        """Get all experiments."""
        experiments = {}
        for log in self.logs:
            if log.get('event_type') in ['experiment_start', 'experiment_end']:
                exp_name = log.get('experiment', 'unknown')
                if exp_name not in experiments:
                    experiments[exp_name] = {'start': None, 'end': None, 'success': None}

                if log.get('event_type') == 'experiment_start':
                    experiments[exp_name]['start'] = log.get('timestamp')
                elif log.get('event_type') == 'experiment_end':
                    experiments[exp_name]['end'] = log.get('timestamp')
                    experiments[exp_name]['success'] = log.get('success', False)

        return experiments

    def print_summary(self):
        """Print a summary of the logs."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("📊 Log Analysis Summary")
        print("="*60)
        print(f"Total Logs: {summary['total_logs']}")
        print("\nLog Levels:")
        for level, count in sorted(summary['level_counts'].items()):
            print(f"  - {level}: {count}")

        print("\nEvent Types:")
        for event_type, count in sorted(summary['event_types'].items()):
            print(f"  - {event_type}: {count}")

        errors = self.get_errors()
        if errors:
            print(f"\n⚠️  Errors Found: {len(errors)}")
            for error in errors[:5]:
                print(f"  - {error.get('message', 'Unknown')[:80]}")

        decisions = self.get_agent_decisions()
        if decisions:
            print(f"\n🧠 Agent Decisions: {len(decisions)}")

        experiments = self.get_experiments()
        if experiments:
            print(f"\n🧪 Experiments: {len(experiments)}")
            for exp_name, details in experiments.items():
                status = "✅" if details.get('success') else "❌"
                print(f"  {status} {exp_name}")

        print("="*60 + "\n")


# ================== Usage Example ==================

if __name__ == "__main__":
    # Create a logger
    logger = get_logger("test_logger", log_dir="logs")

    # Test various logging methods
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test agent decision logging
    logger.log_agent_decision(
        agent_name="hypothesis_agent",
        decision_type="algorithm_selection",
        decision="Use VAE with MMD loss",
        reasoning="MMD provides better distribution alignment for unpaired data",
        context={"round": 1, "nmi_score": 0.75}
    )

    # Test experiment logging
    logger.log_experiment_start(
        experiment_name="multi_omics_integration",
        parameters={"rounds": 5, "method": "VAE"}
    )

    logger.log_experiment_end(
        experiment_name="multi_omics_integration",
        success=True,
        results={"nmi": 0.85, "ari": 0.78}
    )

    # Test phase logging
    with logger.log_phase("data_preprocessing"):
        # Simulate data processing
        pass

    # Print summary
    summary = logger.get_summary()
    print(f"\nLogger Summary: {json.dumps(summary, indent=2)}")

    # Analyze logs
    if hasattr(logger, 'json_log_file_path'):
        analyzer = LogAnalyzer(logger.json_log_file_path)
        analyzer.print_summary()
