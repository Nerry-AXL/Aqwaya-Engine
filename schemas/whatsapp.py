"""
WhatsApp/SMS generator specific Pydantic schemas for Aqwaya AI Orchestration Engine.

Defines data structures and validation models for WhatsApp/SMS generation.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, HttpUrl
from schemas.common import BaseGenerationResponse, ContentVariation
import re


class WhatsAppMessage(BaseModel):
    """Individual WhatsApp/SMS message."""
    id: str = Field(..., description="Message identifier")
    text: str = Field(..., description="Message text content")
    length: int = Field(..., description="Message length in characters")
    emoji_count: Optional[int] = Field(None, description="Number of emojis in message")
    url_count: Optional[int] = Field(None, description="Number of URLs in message")
    
    @field_validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Message ID is required")
        # Ensure valid identifier format
        if not re.match(r'^[a-zA-Z0-9_-]+$', v.strip()):
            raise ValueError("Message ID must contain only letters, numbers, underscore, or dash")
        return v.strip()
    
    @field_validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Message text must be at least 10 characters")
        if len(v) > 4096:  # WhatsApp limit
            raise ValueError("Message text should not exceed 4096 characters")
        return v.strip()
    
    @field_validator('length', mode='before')
    def set_length(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        return len(text) if text else 0

    @field_validator('emoji_count', mode='before')
    def count_emojis(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        if text:
            # Simple emoji detection (Unicode ranges)
            emoji_pattern = re.compile(
                r'[\U0001F600-\U0001F64F]|'  # emoticons
                r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
                r'[\U0001F680-\U0001F6FF]|'  # transport & map symbols
                r'[\U0001F1E0-\U0001F1FF]|'  # flags (iOS)
                r'[\U00002702-\U000027B0]|'  # dingbats
                r'[\U000024C2-\U0001F251]'   # enclosed characters
            )
            return len(emoji_pattern.findall(text))
        return 0
    
    @field_validator('url_count', mode='before')
    def count_urls(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        if text:
            # Simple URL detection
            url_pattern = re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            )
            return len(url_pattern.findall(text))
        return 0


class WhatsAppVariation(ContentVariation):
    """WhatsApp message variation with additional metadata."""
    message_id: str = Field(..., description="Message identifier")
    text: str = Field(..., description="Message text")
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    emoji_count: int = Field(default=0, description="Emoji count")
    urgency_score: Optional[float] = Field(None, description="Urgency score (0-1)")
    readability_score: Optional[float] = Field(None, description="Readability score (0-1)")
    
    @field_validator('length', mode='before')
    def set_length(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        return len(text) if text else 0

    @field_validator('character_count', mode='before')
    def set_character_count(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        return len(text) if text else 0

    @field_validator('word_count', mode='before')
    def set_word_count(cls, v, info):
        text = info.data.get('text', '') if hasattr(info, 'data') else ''
        return len(text.split()) if text else 0


class WhatsAppGenerationInput(BaseModel):
    """WhatsApp/SMS specific input configuration."""
    max_messages: int = Field(default=3, description="Number of message variations to generate")
    message_type: str = Field(default="promotional", description="Type of message (promotional, informational, reminder)")
    max_length: int = Field(default=160, description="Maximum message length")
    include_emojis: bool = Field(default=True, description="Whether to include emojis")
    urgency_level: str = Field(default="medium", description="Urgency level (low, medium, high)")
    personalization: bool = Field(default=False, description="Whether to include personalization placeholders")
    include_shortlink: bool = Field(default=True, description="Whether to include short links")
    
    @field_validator('max_messages')
    def validate_max_messages(cls, v):
        if v < 1 or v > 10:
            raise ValueError("max_messages must be between 1 and 10")
        return v
    
    @field_validator('message_type')
    def validate_message_type(cls, v):
        allowed_types = [
            "promotional", "informational", "reminder", "announcement", 
            "follow_up", "customer_service", "marketing"
        ]
        if v.lower() not in allowed_types:
            raise ValueError(f"message_type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator('max_length')
    def validate_max_length(cls, v):
        if v < 50 or v > 4096:  # WhatsApp limit is 4096
            raise ValueError("max_length must be between 50 and 4096")
        return v
    
    @field_validator('urgency_level')
    def validate_urgency_level(cls, v):
        allowed_levels = ["low", "medium", "high"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"urgency_level must be one of: {', '.join(allowed_levels)}")
        return v.lower()


class WhatsAppAnalytics(BaseModel):
    """WhatsApp/SMS analytics and quality metrics."""
    engagement_score: Optional[float] = Field(None, description="Estimated engagement score (0-100)")
    spam_likelihood: Optional[float] = Field(None, description="Spam likelihood score (0-1)")
    readability_score: Optional[float] = Field(None, description="Message readability score (0-100)")
    sentiment_score: Optional[float] = Field(None, description="Message sentiment (-1 to 1)")
    call_to_action_strength: Optional[float] = Field(None, description="CTA strength score (0-1)")
    
    @field_validator('engagement_score')
    def validate_engagement_score(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError("engagement_score must be between 0 and 100")
        return v
    
    @field_validator('spam_likelihood')
    def validate_spam_likelihood(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("spam_likelihood must be between 0 and 1")
        return v
    
    @field_validator('readability_score')
    def validate_readability_score(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError("readability_score must be between 0 and 100")
        return v
    
    @field_validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        if v is not None and (v < -1 or v > 1):
            raise ValueError("sentiment_score must be between -1 and 1")
        return v
    
    @field_validator('call_to_action_strength')
    def validate_cta_strength(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("call_to_action_strength must be between 0 and 1")
        return v


class WhatsAppGenerationResponse(BaseGenerationResponse):
    """WhatsApp/SMS generation response schema."""
    messages: List[WhatsAppMessage] = Field(..., description="Generated message variations")
    analytics: Optional[WhatsAppAnalytics] = Field(None, description="Message analytics and metrics")
    recommendations: List[str] = Field(default_factory=list, description="Content improvement recommendations")
    
    def __init__(self, **data):
        # Set module name automatically
        data['module'] = 'whatsapp_generator'
        super().__init__(**data)
    
    @field_validator('messages')
    def validate_messages(cls, v, info):
        # Allow empty messages list if status is ERROR
        status = info.data.get('status') if hasattr(info, 'data') else None
        from schemas.common import GenerationStatus
        if (not v or len(v) == 0) and status != GenerationStatus.ERROR:
            raise ValueError("At least one message variation is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 message variations allowed")
        return v


class WhatsAppTemplate(BaseModel):
    """WhatsApp message template configuration."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    category: str = Field(..., description="Template category")
    template_text: str = Field(..., description="Template text with placeholders")
    required_variables: List[str] = Field(..., description="Required template variables")
    optional_variables: List[str] = Field(default_factory=list, description="Optional template variables")
    max_length: int = Field(default=160, description="Maximum message length")
    language: str = Field(default="en", description="Template language")
    
    @field_validator('template_text')
    def validate_template_text(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Template text must be at least 10 characters")
        return v.strip()
    
    @field_validator('category')
    def validate_category(cls, v):
        allowed_categories = [
            "marketing", "utility", "authentication", "promotional", 
            "transactional", "customer_service"
        ]
        if v.lower() not in allowed_categories:
            raise ValueError(f"category must be one of: {', '.join(allowed_categories)}")
        return v.lower()


class MessageQualityCheck(BaseModel):
    """Message quality assessment."""
    passes_spam_filter: bool = Field(..., description="Whether message passes spam filters")
    character_efficiency: float = Field(..., description="Character efficiency score (0-1)")
    call_to_action_present: bool = Field(..., description="Whether CTA is present")
    brand_mention_count: int = Field(..., description="Number of brand mentions")
    urgency_indicators: List[str] = Field(default_factory=list, description="Urgency indicators found")
    potential_issues: List[str] = Field(default_factory=list, description="Potential issues identified")
    
    @field_validator('character_efficiency')
    def validate_efficiency(cls, v):
        if v < 0 or v > 1:
            raise ValueError("character_efficiency must be between 0 and 1")
        return v


class WhatsAppValidationResult(BaseModel):
    """WhatsApp message validation result."""
    is_valid: bool = Field(..., description="Whether the message is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Content improvement suggestions")
    quality_check: Optional[MessageQualityCheck] = Field(None, description="Quality assessment")
    
    @field_validator('is_valid', mode='before')
    def set_validity(cls, v, values):
        errors = values.get('errors', [])
        return len(errors) == 0


class SMSSpecificConfig(BaseModel):
    """SMS-specific configuration options."""
    character_limit: int = Field(default=160, description="SMS character limit")
    split_long_messages: bool = Field(default=True, description="Whether to split long messages")
    include_stop_instruction: bool = Field(default=True, description="Whether to include STOP instruction")
    sender_id: Optional[str] = Field(None, description="SMS sender ID")
    
    @field_validator('character_limit')
    def validate_character_limit(cls, v):
        if v < 70 or v > 918:  # SMS limits
            raise ValueError("character_limit must be between 70 and 918")
        return v
    
    @field_validator('sender_id')
    def validate_sender_id(cls, v):
        if v and (len(v) < 3 or len(v) > 11):
            raise ValueError("sender_id must be between 3 and 11 characters")
        return v


class WhatsAppBusinessConfig(BaseModel):
    """WhatsApp Business API specific configuration."""
    template_name: Optional[str] = Field(None, description="Approved template name")
    template_language: str = Field(default="en_US", description="Template language code")
    header_type: Optional[str] = Field(None, description="Header type (text, image, video, document)")
    footer_text: Optional[str] = Field(None, description="Footer text")
    buttons: Optional[List[Dict]] = Field(None, description="Interactive buttons")
    
    @field_validator('template_language')
    def validate_template_language(cls, v):
        # WhatsApp supported language codes
        allowed_languages = [
            "en_US", "es_ES", "pt_BR", "fr_FR", "ar", "hi", "id", "de", "it", 
            "ja", "ko", "ms", "nl", "pl", "ru", "th", "tr", "vi", "zh_CN", "zh_HK"
        ]
        if v not in allowed_languages:
            raise ValueError(f"template_language must be one of: {', '.join(allowed_languages)}")
        return v