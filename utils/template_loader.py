"""
Template loading and formatting utility for Aqwaya AI Orchestration Engine.

Provides utilities to load and format Jinja2/text prompt templates with campaign data.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateSyntaxError
from utils.logger import get_logger

logger = get_logger(__name__)


class TemplateLoader:
    """Template loading and formatting utility."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template loader.
        
        Args:
            template_dir: Directory containing template files. 
                         Defaults to 'prompt_templates' in project root.
        """
        if template_dir is None:
            # Get project root (assuming we're in utils/)
            project_root = Path(__file__).parent.parent
            template_dir = project_root / "prompt_templates"
        
        self.template_dir = Path(template_dir)
        
        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {self.template_dir}")
            self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False  # We'll handle escaping in sanitization
        )
        
        logger.info(f"Template loader initialized with directory: {self.template_dir}")
    
    def load_template(self, template_name: str) -> str:
        """Load a raw template file as a string.
        
        Args:
            template_name: Name of the template file (e.g., 'email_prompt.txt')
            
        Returns:
            Raw template content as string
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            IOError: If template file can't be read
        """
        template_path = self.template_dir / template_name
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"Loaded template: {template_name}", 
                        template_size=len(content))
            return content
            
        except IOError as e:
            logger.error(f"Failed to read template file: {template_path}", 
                        error=str(e))
            raise IOError(f"Failed to read template file: {template_path}") from e
    
    def format_template(self, 
                       template_name: str, 
                       context: Dict[str, Any],
                       use_jinja: bool = True) -> str:
        """Format a template with the provided context.
        
        Args:
            template_name: Name of the template file
            context: Dictionary containing template variables
            use_jinja: Whether to use Jinja2 for advanced templating (default: True)
                      If False, uses simple string formatting with {variable} syntax
            
        Returns:
            Formatted template content
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            TemplateSyntaxError: If template has syntax errors (Jinja2 only)
            KeyError: If required template variables are missing (string formatting only)
        """
        start_time = logger._get_current_time_ms() if hasattr(logger, '_get_current_time_ms') else 0
        
        if use_jinja:
            formatted = self._format_jinja_template(template_name, context)
        else:
            formatted = self._format_string_template(template_name, context)
        
        end_time = logger._get_current_time_ms() if hasattr(logger, '_get_current_time_ms') else 0
        
        logger.info(f"Template formatted: {template_name}",
                   template_size=len(formatted),
                   context_vars=list(context.keys()),
                   use_jinja=use_jinja,
                   duration_ms=end_time - start_time)
        
        return formatted
    
    def _format_jinja_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Format template using Jinja2."""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
            
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error in {template_name}",
                        error=str(e),
                        line_number=e.lineno if hasattr(e, 'lineno') else None)
            raise
        
        except Exception as e:
            logger.error(f"Failed to format Jinja template: {template_name}",
                        error=str(e))
            raise
    
    def _format_string_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Format template using simple string formatting."""
        try:
            template_content = self.load_template(template_name)
            return template_content.format(**context)
            
        except KeyError as e:
            logger.error(f"Missing template variable in {template_name}",
                        missing_var=str(e),
                        available_vars=list(context.keys()))
            raise
        
        except Exception as e:
            logger.error(f"Failed to format string template: {template_name}",
                        error=str(e))
            raise
    
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template file exists and is readable.
        
        Args:
            template_name: Name of the template file
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.load_template(template_name)
            return True
        except (FileNotFoundError, IOError):
            return False
    
    def list_templates(self) -> list[str]:
        """List all available template files.
        
        Returns:
            List of template filenames
        """
        if not self.template_dir.exists():
            return []
        
        templates = []
        for file_path in self.template_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.txt', '.jinja', '.j2']:
                templates.append(file_path.name)
        
        return sorted(templates)
    
    def get_template_variables(self, template_name: str) -> list[str]:
        """Extract variable names from a template.
        
        Args:
            template_name: Name of the template file
            
        Returns:
            List of variable names found in the template
        """
        try:
            template_content = self.load_template(template_name)
            
            # For Jinja2 templates, we'd need to parse AST
            # For now, simple regex for {variable} patterns
            import re
            variables = re.findall(r'\{([^}]+)\}', template_content)
            
            # Clean up variable names (remove whitespace, filters, etc.)
            cleaned_vars = []
            for var in variables:
                # Remove Jinja2 filters and whitespace
                clean_var = var.split('|')[0].strip()
                if clean_var and clean_var not in cleaned_vars:
                    cleaned_vars.append(clean_var)
            
            return cleaned_vars
            
        except Exception as e:
            logger.error(f"Failed to extract variables from template: {template_name}",
                        error=str(e))
            return []


# Global template loader instance
template_loader = TemplateLoader()


def load_template(template_name: str) -> str:
    """Load a raw template file as a string."""
    return template_loader.load_template(template_name)


def format_template(template_name: str, 
                   context: Dict[str, Any],
                   use_jinja: bool = True) -> str:
    """Format a template with the provided context."""
    return template_loader.format_template(template_name, context, use_jinja)


def validate_template(template_name: str) -> bool:
    """Validate that a template file exists and is readable."""
    return template_loader.validate_template(template_name)


def list_templates() -> list[str]:
    """List all available template files."""
    return template_loader.list_templates()


def get_template_variables(template_name: str) -> list[str]:
    """Extract variable names from a template."""
    return template_loader.get_template_variables(template_name)