"""
Structured JSON logging utility for Aqwaya AI Orchestration Engine.

Provides consistent logging across all modules with structured JSON output.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import traceback


class StructuredLogger:
    """Structured JSON logger for consistent logging across all modules."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize the structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _json_formatter(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": getattr(record, 'module', None),
            "function": getattr(record, 'funcName', None),
            "line": getattr(record, 'lineno', None),
        }
        
        # Add any extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)
    
    def _log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log message with additional context."""
        extra = {"extra_fields": kwargs} if kwargs else {}
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context("CRITICAL", message, **kwargs)
    
    def log_api_call(self, 
                     service: str, 
                     method: str, 
                     duration_ms: int, 
                     tokens_used: Optional[int] = None,
                     status: str = "success",
                     **kwargs) -> None:
        """Log API call with standardized format."""
        context = {
            "service": service,
            "method": method,
            "duration_ms": duration_ms,
            "status": status
        }
        
        if tokens_used is not None:
            context["tokens_used"] = tokens_used
        
        context.update(kwargs)
        
        self.info(f"API call to {service}.{method}", **context)
    
    def log_generation_request(self,
                             module: str,
                             campaign_id: str,
                             input_size: int,
                             **kwargs) -> None:
        """Log generation request."""
        context = {
            "module": module,
            "campaign_id": campaign_id,
            "input_size": input_size
        }
        context.update(kwargs)
        
        self.info(f"Generation request for {module}", **context)
    
    def log_generation_response(self,
                              module: str,
                              campaign_id: str,
                              output_size: int,
                              duration_ms: int,
                              status: str = "success",
                              **kwargs) -> None:
        """Log generation response."""
        context = {
            "module": module,
            "campaign_id": campaign_id,
            "output_size": output_size,
            "duration_ms": duration_ms,
            "status": status
        }
        context.update(kwargs)
        
        self.info(f"Generation response for {module}", **context)


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Get or create a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)