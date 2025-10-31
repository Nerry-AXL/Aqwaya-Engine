"""
Email generator specific Pydantic schemas for Aqwaya AI Orchestration Engine.

Defines data structures and validation models for email generation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, HttpUrl
from schemas.common import BaseGenerationResponse, CallToAction, ContentVariation, GenerationStatus


class EmailContent(BaseModel):
    """Individual email content variation."""
    subject: str = Field(..., description="Email subject line")
    preheader: str = Field(..., description="Email preheader text")
    html_body: str = Field(..., description="HTML email body")
    plain_text: str = Field(..., description="Plain text email body")
    cta: CallToAction = Field(..., description="Call-to-action configuration")
    
    @field_validator('subject')
    def validate_subject(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Email subject must be at least 5 characters")
        if len(v) > 100:
            raise ValueError("Email subject should not exceed 100 characters")
        return v.strip()
    
    @field_validator('preheader')
    def validate_preheader(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Email preheader must be at least 10 characters")
        if len(v) > 150:
            raise ValueError("Email preheader should not exceed 150 characters")
        return v.strip()
    
    @field_validator('html_body')
    def validate_html_body(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("Email HTML body must be at least 50 characters")
        if len(v) > 10000:
            raise ValueError("Email HTML body should not exceed 10000 characters")
        return v.strip()
    
    @field_validator('plain_text')
    def validate_plain_text(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("Email plain text body must be at least 50 characters")
        if len(v) > 5000:
            raise ValueError("Email plain text body should not exceed 5000 characters")
        return v.strip()


class EmailVariation(ContentVariation):
    """Email content variation with additional metadata."""
    subject: str = Field(..., description="Email subject line")
    preheader: str = Field(..., description="Email preheader text") 
    html_body: str = Field(..., description="HTML email body")
    plain_text: str = Field(..., description="Plain text email body")
    cta_text: str = Field(..., description="Call-to-action text")
    cta_url: Optional[HttpUrl] = Field(None, description="Call-to-action URL")
    estimated_read_time: Optional[int] = Field(None, description="Estimated read time in seconds")
    
    @field_validator('length', mode='before')
    def set_length(cls, v, values):
        html_body = values.get('html_body', '')
        return len(html_body) if html_body else 0

    @field_validator('estimated_read_time', mode='before')
    def calculate_read_time(cls, v, values):
        # Estimate read time: ~200 words per minute
        plain_text = values.get('plain_text', '')
        if plain_text:
            word_count = len(plain_text.split())
            return max(10, int((word_count / 200) * 60))  # Minimum 10 seconds
        return 30  # Default


class EmailGenerationInput(BaseModel):
    """Email-specific input configuration."""
    max_variations: int = Field(default=2, description="Number of email variations to generate")
    email_type: str = Field(default="marketing", description="Type of email (marketing, transactional, newsletter)")
    include_images: bool = Field(default=True, description="Whether to include image suggestions")
    personalization_level: str = Field(default="medium", description="Level of personalization (low, medium, high)")
    urgency_level: str = Field(default="medium", description="Urgency level (low, medium, high)")
    
    @field_validator('max_variations')
    def validate_max_variations(cls, v):
        if v < 1 or v > 5:
            raise ValueError("max_variations must be between 1 and 5")
        return v
    
    @field_validator('email_type')
    def validate_email_type(cls, v):
        allowed_types = ["marketing", "transactional", "newsletter", "promotional", "announcement"]
        if v.lower() not in allowed_types:
            raise ValueError(f"email_type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator('personalization_level')
    def validate_personalization_level(cls, v):
        allowed_levels = ["low", "medium", "high"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"personalization_level must be one of: {', '.join(allowed_levels)}")
        return v.lower()
    
    @field_validator('urgency_level')
    def validate_urgency_level(cls, v):
        allowed_levels = ["low", "medium", "high"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"urgency_level must be one of: {', '.join(allowed_levels)}")
        return v.lower()


class EmailAnalytics(BaseModel):
    """Email analytics and quality metrics."""
    spam_score: Optional[float] = Field(None, description="Estimated spam score (0-100)")
    readability_score: Optional[float] = Field(None, description="Content readability score")
    sentiment_score: Optional[float] = Field(None, description="Content sentiment score (-1 to 1)")
    cta_prominence: Optional[str] = Field(None, description="CTA prominence assessment")
    mobile_friendly: Optional[bool] = Field(None, description="Mobile-friendliness assessment")
    
    @field_validator('spam_score')
    def validate_spam_score(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError("spam_score must be between 0 and 100")
        return v
    
    @field_validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        if v is not None and (v < -1 or v > 1):
            raise ValueError("sentiment_score must be between -1 and 1")
        return v


class EmailGenerationResponse(BaseGenerationResponse):
    """Email generation response schema."""
    emails: List[EmailVariation] = Field(..., description="Generated email variations")
    analytics: Optional[EmailAnalytics] = Field(None, description="Email analytics and metrics")
    recommendations: List[str] = Field(default_factory=list, description="Content improvement recommendations")
    
    def __init__(self, **data):
        # Set module name automatically
        data['module'] = 'email_generator'
        super().__init__(**data)
    
    @field_validator('emails')
    def validate_emails(cls, v, info):
        # Allow empty emails list if status is ERROR
        status = info.data.get('status') if hasattr(info, 'data') else None
        if (not v or len(v) == 0) and status != GenerationStatus.ERROR:
            raise ValueError("At least one email variation is required")
        if len(v) > 5:
            raise ValueError("Maximum 5 email variations allowed")
        return v


class EmailTemplate(BaseModel):
    """Email template configuration."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    html_template: str = Field(..., description="HTML template with placeholders")
    text_template: str = Field(..., description="Text template with placeholders")
    required_variables: List[str] = Field(..., description="Required template variables")
    optional_variables: List[str] = Field(default_factory=list, description="Optional template variables")
    category: str = Field(default="general", description="Template category")
    
    @field_validator('html_template')
    def validate_html_template(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("HTML template must be at least 20 characters")
        return v.strip()
    
    @field_validator('text_template')
    def validate_text_template(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("Text template must be at least 20 characters")
        return v.strip()


class EmailValidationResult(BaseModel):
    """Email content validation result."""
    is_valid: bool = Field(..., description="Whether the email content is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Content improvement suggestions")
    
    @field_validator('is_valid', mode='before')
    def set_validity(cls, v, values):
        errors = values.get('errors', [])
        return len(errors) == 0