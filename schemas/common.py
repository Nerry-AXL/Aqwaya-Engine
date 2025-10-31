"""
Common Pydantic schemas for Aqwaya AI Orchestration Engine.

Defines shared data structures and validation models used across all modules.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, HttpUrl
from datetime import datetime
from enum import Enum
import uuid


class LanguageCode(str, Enum):
    """Supported language codes."""
    EN_US = "en-US"
    EN_NG = "en-NG"  
    EN_GB = "en-GB"
    FR_FR = "fr-FR"
    ES_ES = "es-ES"
    PT_BR = "pt-BR"
    AR_SA = "ar-SA"
    SW_KE = "sw-KE"


class GenerationStatus(str, Enum):
    """Status of generation requests."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    TIMEOUT = "timeout"


class BusinessProfile(BaseModel):
    """Business profile information."""
    name: str = Field(..., description="Business name")
    industry: str = Field(..., description="Business industry")
    website: Optional[HttpUrl] = Field(None, description="Business website URL")
    brand_tone: str = Field(..., description="Brand tone and voice")
    brand_keywords: List[str] = Field(..., description="Key brand keywords and phrases")
    brand_colors: Optional[List[str]] = Field(None, description="Brand colors (hex codes)")
    logo_url: Optional[HttpUrl] = Field(None, description="Brand logo URL")
    
    @field_validator('brand_keywords')
    def validate_brand_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one brand keyword is required")
        return [keyword.strip() for keyword in v if keyword.strip()]
    
    @field_validator('brand_tone')
    def validate_brand_tone(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Brand tone must be at least 3 characters")
        return v.strip()


class CallToAction(BaseModel):
    """Call-to-action configuration."""
    text: str = Field(..., description="CTA button/link text")
    url: Optional[HttpUrl] = Field(None, description="CTA destination URL")
    type: Optional[str] = Field("primary", description="CTA type (primary, secondary, etc.)")
    
    @field_validator('text')
    def validate_cta_text(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("CTA text must be at least 2 characters")
        return v.strip()


class CampaignInput(BaseModel):
    """Input schema for campaign generation requests."""
    campaign_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique campaign identifier")
    business: BusinessProfile = Field(..., description="Business profile information")
    prompt: str = Field(..., description="Campaign description and instructions")
    audience: str = Field(..., description="Target audience description")
    desired_cta: str = Field(..., description="Desired call-to-action text")
    language: LanguageCode = Field(default=LanguageCode.EN_US, description="Content language")
    channel_config: Dict[str, Any] = Field(default_factory=dict, description="Channel-specific configuration")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")
    
    # Common channel configuration defaults
    landing_url: Optional[HttpUrl] = Field(None, description="Landing page URL for CTAs")
    shortlink: Optional[str] = Field(None, description="Short link for messaging channels")
    email_max_length: Optional[int] = Field(2000, description="Maximum email content length")
    whatsapp_max_length: Optional[int] = Field(160, description="Maximum WhatsApp message length")
    landing_template: Optional[str] = Field("default", description="Landing page template type")
    
    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Campaign prompt must be at least 10 characters")
        return v.strip()
    
    @field_validator('audience')
    def validate_audience(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Audience description must be at least 5 characters")
        return v.strip()
    
    @field_validator('campaign_id')
    def validate_campaign_id(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v


class ModelMetadata(BaseModel):
    """Metadata about the AI model used for generation."""
    model: str = Field(..., description="Model identifier")
    tokens: Optional[int] = Field(None, description="Tokens consumed")
    ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    temperature: Optional[float] = Field(None, description="Model temperature setting")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens requested")
    cost_estimate: Optional[float] = Field(None, description="Estimated API cost")


class BaseGenerationResponse(BaseModel):
    """Base response schema for all generation modules."""
    module: str = Field(..., description="Module that generated the response")
    campaign_id: str = Field(..., description="Campaign identifier")
    model_meta: ModelMetadata = Field(..., description="Model execution metadata")
    status: GenerationStatus = Field(..., description="Generation status")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    
    @field_validator('error_message')
    def validate_error_message(cls, v, info):
        status = info.data.get('status') if hasattr(info, 'data') else None
        if status == GenerationStatus.ERROR and not v:
            raise ValueError("Error message is required when status is error")
        return v


class GenerationRequest(BaseModel):
    """Internal request wrapper for generation modules."""
    input: CampaignInput = Field(..., description="Campaign input data")
    module_name: str = Field(..., description="Name of the requesting module")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Request creation timestamp")


class ContentVariation(BaseModel):
    """Base class for content variations."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Variation identifier")
    content: str = Field(..., description="Generated content")
    length: int = Field(..., description="Content length in characters")
    
    @field_validator('length', mode='before')
    def set_length(cls, v, values):
        content = values.get('content', '')
        return len(content) if content else 0


class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


# Common validation utilities
def validate_hex_color(color: str) -> str:
    """Validate hex color code."""
    import re
    if not re.match(r'^#[0-9A-Fa-f]{6}$', color):
        raise ValueError(f"Invalid hex color code: {color}")
    return color


def validate_positive_integer(value: int) -> int:
    """Validate positive integer."""
    if value <= 0:
        raise ValueError("Value must be positive")
    return value


def validate_non_empty_string(value: str) -> str:
    """Validate non-empty string."""
    if not value or not value.strip():
        raise ValueError("Value cannot be empty")
    return value.strip()