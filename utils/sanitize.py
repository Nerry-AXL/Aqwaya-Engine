"""
HTML sanitization utility for Aqwaya AI Orchestration Engine.

Provides safe HTML sanitization to prevent XSS attacks and ensure clean output.
"""
import re
from typing import List, Optional
from html import escape
import bleach


class HTMLSanitizer:
    """HTML sanitization utility for cleaning AI-generated content."""
    
    # Allowed HTML tags for email content
    ALLOWED_EMAIL_TAGS = [
        'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'a', 'img',
        'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'table', 'tr', 'td', 'th', 'tbody', 'thead',
        'ul', 'ol', 'li', 'blockquote'
    ]
    
    # Allowed HTML tags for landing pages
    ALLOWED_LANDING_TAGS = ALLOWED_EMAIL_TAGS + [
        'section', 'article', 'header', 'footer', 'nav',
        'form', 'input', 'button', 'select', 'option', 'textarea',
        'label', 'fieldset', 'legend'
    ]
    
    # Allowed attributes for all tags
    ALLOWED_ATTRIBUTES = {
        '*': ['class', 'id', 'style'],
        'a': ['href', 'title', 'target', 'rel'],
        'img': ['src', 'alt', 'width', 'height', 'style'],
        'table': ['border', 'cellpadding', 'cellspacing', 'width'],
        'td': ['colspan', 'rowspan', 'align', 'valign'],
        'th': ['colspan', 'rowspan', 'align', 'valign'],
        'input': ['type', 'name', 'value', 'placeholder', 'required'],
        'button': ['type', 'name', 'value'],
        'select': ['name', 'required'],
        'option': ['value', 'selected'],
        'textarea': ['name', 'placeholder', 'required', 'rows', 'cols'],
        'label': ['for'],
        'form': ['action', 'method', 'enctype']
    }
    
    # Allowed CSS properties for inline styles
    ALLOWED_CSS_PROPERTIES = [
        'color', 'background-color', 'font-size', 'font-weight', 'font-family',
        'text-align', 'text-decoration', 'margin', 'margin-top', 'margin-bottom',
        'margin-left', 'margin-right', 'padding', 'padding-top', 'padding-bottom',
        'padding-left', 'padding-right', 'border', 'border-radius', 'width',
        'height', 'max-width', 'min-width', 'display', 'float', 'clear',
        'line-height', 'vertical-align'
    ]
    
    def __init__(self):
        """Initialize the HTML sanitizer."""
        pass
    
    def sanitize_email_html(self, html_content: str) -> str:
        """Sanitize HTML content for email use.
        
        Args:
            html_content: Raw HTML content to sanitize
            
        Returns:
            Sanitized HTML content safe for email
        """
        if not html_content:
            return ""
        
        # First pass: Remove dangerous elements
        cleaned = self._remove_dangerous_elements(html_content)
        
        # Second pass: Bleach sanitization
        cleaned = bleach.clean(
            cleaned,
            tags=self.ALLOWED_EMAIL_TAGS,
            attributes=self.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        # Third pass: Sanitize inline styles
        cleaned = self._sanitize_inline_styles(cleaned)
        
        # Fourth pass: Ensure email-safe formatting
        cleaned = self._ensure_email_safe_formatting(cleaned)
        
        return cleaned
    
    def sanitize_landing_html(self, html_content: str) -> str:
        """Sanitize HTML content for landing page use.
        
        Args:
            html_content: Raw HTML content to sanitize
            
        Returns:
            Sanitized HTML content safe for landing pages
        """
        if not html_content:
            return ""
        
        # First pass: Remove dangerous elements
        cleaned = self._remove_dangerous_elements(html_content)
        
        # Second pass: Bleach sanitization
        cleaned = bleach.clean(
            cleaned,
            tags=self.ALLOWED_LANDING_TAGS,
            attributes=self.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        # Third pass: Sanitize inline styles
        cleaned = self._sanitize_inline_styles(cleaned)
        
        return cleaned
    
    def sanitize_text(self, text_content: str) -> str:
        """Sanitize plain text content.
        
        Args:
            text_content: Raw text content to sanitize
            
        Returns:
            Sanitized text content
        """
        if not text_content:
            return ""
        
        # Escape HTML entities
        sanitized = escape(text_content)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Remove null bytes and other dangerous characters
        sanitized = sanitized.replace('\x00', '').replace('\r', '')
        
        return sanitized
    
    def _remove_dangerous_elements(self, html_content: str) -> str:
        """Remove dangerous HTML elements and attributes."""
        # Remove script tags and their content
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove link tags (external CSS)
        html_content = re.sub(r'<link[^>]*>', '', html_content, flags=re.IGNORECASE)
        
        # Remove meta tags
        html_content = re.sub(r'<meta[^>]*>', '', html_content, flags=re.IGNORECASE)
        
        # Remove iframe, object, embed tags
        dangerous_tags = ['iframe', 'object', 'embed', 'applet', 'base', 'form']
        for tag in dangerous_tags:
            html_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(f'<{tag}[^>]*>', '', html_content, flags=re.IGNORECASE)
        
        # Remove javascript: URLs
        html_content = re.sub(r'javascript:', '', html_content, flags=re.IGNORECASE)
        
        # Remove on* event handlers
        html_content = re.sub(r'\son\w+\s*=\s*["\'][^"\']*["\']', '', html_content, flags=re.IGNORECASE)
        
        return html_content
    
    def _sanitize_inline_styles(self, html_content: str) -> str:
        """Sanitize inline CSS styles."""
        def clean_style_attribute(match):
            style_content = match.group(1)
            cleaned_styles = []
            
            # Split by semicolon to get individual CSS declarations
            declarations = style_content.split(';')
            
            for declaration in declarations:
                if ':' not in declaration:
                    continue
                
                prop, value = declaration.split(':', 1)
                prop = prop.strip().lower()
                value = value.strip()
                
                # Only allow safe CSS properties
                if prop in self.ALLOWED_CSS_PROPERTIES:
                    # Remove potentially dangerous values
                    if not self._is_safe_css_value(value):
                        continue
                    
                    cleaned_styles.append(f"{prop}: {value}")
            
            return f'style="{"; ".join(cleaned_styles)}"' if cleaned_styles else ''
        
        # Clean style attributes
        html_content = re.sub(r'style\s*=\s*["\']([^"\']*)["\']', clean_style_attribute, html_content, flags=re.IGNORECASE)
        
        return html_content
    
    def _is_safe_css_value(self, value: str) -> bool:
        """Check if a CSS value is safe."""
        # Remove potentially dangerous CSS values
        dangerous_patterns = [
            r'expression\s*\(',  # CSS expressions
            r'javascript:',       # JavaScript URLs
            r'vbscript:',        # VBScript URLs
            r'data:',            # Data URLs (can be dangerous)
            r'@import',          # CSS imports
            r'url\s*\(',         # URL references (be cautious)
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _ensure_email_safe_formatting(self, html_content: str) -> str:
        """Ensure HTML is safe for email clients."""
        # Convert relative URLs to absolute (placeholder implementation)
        # In production, you'd want to handle this based on your domain
        html_content = re.sub(r'href\s*=\s*["\']/', 'href="https://yourdomain.com/', html_content)
        html_content = re.sub(r'src\s*=\s*["\']/', 'src="https://yourdomain.com/', html_content)
        
        # Ensure tables have proper attributes for email compatibility
        html_content = re.sub(r'<table(?!\s[^>]*border)', '<table border="0"', html_content)
        
        return html_content
    
    def validate_url(self, url: str) -> bool:
        """Validate that a URL is safe.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is safe, False otherwise
        """
        if not url:
            return False
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'vbscript:', 'data:', 'file:']
        url_lower = url.lower().strip()
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
        
        # Must be http/https or relative
        if not (url_lower.startswith('http://') or 
                url_lower.startswith('https://') or 
                url_lower.startswith('/') or
                url_lower.startswith('./')):
            return False
        
        return True


# Global sanitizer instance
sanitizer = HTMLSanitizer()

# Convenience functions
def sanitize_email_html(html_content: str) -> str:
    """Sanitize HTML content for email use."""
    return sanitizer.sanitize_email_html(html_content)

def sanitize_landing_html(html_content: str) -> str:
    """Sanitize HTML content for landing page use."""
    return sanitizer.sanitize_landing_html(html_content)

def sanitize_text(text_content: str) -> str:
    """Sanitize plain text content."""
    return sanitizer.sanitize_text(text_content)

def validate_url(url: str) -> bool:
    """Validate that a URL is safe."""
    return sanitizer.validate_url(url)