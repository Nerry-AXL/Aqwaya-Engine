"""
Landing page generator specific Pydantic schemas for Aqwaya AI Orchestration Engine.

Defines data structures and validation models for landing page generation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, HttpUrl
from schemas.common import BaseGenerationResponse, CallToAction, ContentVariation


class HeroImageSuggestion(BaseModel):
    """Hero image suggestion."""
    desc: str = Field(..., description="Image description")
    alt_text: str = Field(..., description="Alt text for accessibility")
    style: Optional[str] = Field(None, description="Image style suggestion")
    dimensions: Optional[str] = Field(None, description="Recommended dimensions")
    
    @field_validator('desc')
    def validate_desc(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Image description must be at least 5 characters")
        return v.strip()
    
    @field_validator('alt_text')
    def validate_alt_text(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Alt text must be at least 3 characters")
        return v.strip()


class FormField(BaseModel):
    """Form field configuration."""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (text, email, tel, select, etc.)")
    label: str = Field(..., description="Field label")
    placeholder: Optional[str] = Field(None, description="Field placeholder text")
    required: bool = Field(default=True, description="Whether field is required")
    options: Optional[List[str]] = Field(None, description="Options for select fields")
    validation: Optional[Dict[str, Any]] = Field(None, description="Field validation rules")
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Field name is required")
        # Ensure valid HTML name attribute
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v.strip()):
            raise ValueError("Field name must start with letter and contain only letters, numbers, underscore, or dash")
        return v.strip()
    
    @field_validator('type')
    def validate_type(cls, v):
        allowed_types = [
            "text", "email", "tel", "number", "password", "url",
            "select", "checkbox", "radio", "textarea", "hidden"
        ]
        if v.lower() not in allowed_types:
            raise ValueError(f"Field type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator('label')
    def validate_label(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Field label must be at least 2 characters")
        return v.strip()


class LandingPageSection(BaseModel):
    """Landing page content section."""
    id: str = Field(..., description="Section identifier")
    type: str = Field(..., description="Section type")
    title: Optional[str] = Field(None, description="Section title")
    content: str = Field(..., description="Section content")
    order: int = Field(..., description="Section display order")
    style_class: Optional[str] = Field(None, description="CSS class for styling")
    
    @field_validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Section ID is required")
        # Ensure valid HTML ID
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v.strip()):
            raise ValueError("Section ID must start with letter and contain only letters, numbers, underscore, or dash")
        return v.strip()
    
    @field_validator('type')
    def validate_type(cls, v):
        allowed_types = [
            "hero", "features", "benefits", "testimonials", "faq", 
            "pricing", "gallery", "contact", "footer", "custom"
        ]
        if v.lower() not in allowed_types:
            raise ValueError(f"Section type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Section content must be at least 10 characters")
        return v.strip()
    
    @field_validator('order')
    def validate_order(cls, v):
        if v < 0:
            raise ValueError("Section order must be non-negative")
        return v


class LandingPageData(BaseModel):
    """Landing page content data."""
    slug: str = Field(..., description="URL slug for the landing page")
    title: str = Field(..., description="Page title")
    subtitle: str = Field(..., description="Page subtitle")
    hero_image_suggestions: List[HeroImageSuggestion] = Field(default_factory=list, description="Hero image suggestions")
    bullets: List[str] = Field(default_factory=list, description="Key benefit bullets")
    sections: List[LandingPageSection] = Field(default_factory=list, description="Page sections")
    form_fields: List[FormField] = Field(default_factory=list, description="Form fields")
    html_template: str = Field(..., description="Complete HTML template")
    cta: CallToAction = Field(..., description="Primary call-to-action")
    
    @field_validator('slug')
    def validate_slug(cls, v):
        if not v or not v.strip():
            raise ValueError("Page slug is required")
        # Ensure valid URL slug
        import re
        slug = v.strip().lower()
        if not re.match(r'^[a-z0-9-]+$', slug):
            raise ValueError("Slug must contain only lowercase letters, numbers, and hyphens")
        return slug
    
    @field_validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Page title must be at least 10 characters")
        if len(v) > 100:
            raise ValueError("Page title should not exceed 100 characters")
        return v.strip()
    
    @field_validator('subtitle')
    def validate_subtitle(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Page subtitle must be at least 10 characters")
        if len(v) > 200:
            raise ValueError("Page subtitle should not exceed 200 characters")
        return v.strip()
    
    @field_validator('html_template')
    def validate_html_template(cls, v):
        if not v or len(v.strip()) < 100:
            raise ValueError("HTML template must be at least 100 characters")
        return v.strip()
    
    @field_validator('bullets')
    def validate_bullets(cls, v):
        if len(v) < 3:
            raise ValueError("At least 3 benefit bullets are required")
        if len(v) > 10:
            raise ValueError("Maximum 10 bullets allowed")
        
        clean_bullets = []
        for bullet in v:
            if bullet and bullet.strip() and len(bullet.strip()) >= 5:
                clean_bullets.append(bullet.strip())
        
        if len(clean_bullets) < 3:
            raise ValueError("At least 3 valid bullets (5+ characters each) are required")
        
        return clean_bullets


class LandingPageVariation(ContentVariation):
    """Landing page variation with additional metadata."""
    slug: str = Field(..., description="URL slug")
    title: str = Field(..., description="Page title")
    subtitle: str = Field(..., description="Page subtitle")
    sections_count: int = Field(..., description="Number of sections")
    form_fields_count: int = Field(..., description="Number of form fields")
    estimated_conversion_rate: Optional[float] = Field(None, description="Estimated conversion rate")
    mobile_optimized: bool = Field(default=True, description="Whether page is mobile optimized")
    
    @field_validator('length', mode='before')
    def set_length(cls, v, values):
        # Calculate total content length
        title = values.get('title', '')
        subtitle = values.get('subtitle', '')
        content = values.get('content', '')
        return len(title) + len(subtitle) + len(content)
    
    @field_validator('estimated_conversion_rate')
    def validate_conversion_rate(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Conversion rate must be between 0 and 1")
        return v


class LandingPageGenerationInput(BaseModel):
    """Landing page specific input configuration."""
    template_style: str = Field(default="modern", description="Template style (modern, classic, minimal)")
    include_testimonials: bool = Field(default=True, description="Whether to include testimonials section")
    include_faq: bool = Field(default=True, description="Whether to include FAQ section")
    form_type: str = Field(default="lead", description="Type of form (lead, contact, newsletter)")
    max_sections: int = Field(default=5, description="Maximum number of sections")
    color_scheme: Optional[str] = Field(None, description="Color scheme preference")
    
    @field_validator('template_style')
    def validate_template_style(cls, v):
        allowed_styles = ["modern", "classic", "minimal", "bold", "elegant"]
        if v.lower() not in allowed_styles:
            raise ValueError(f"template_style must be one of: {', '.join(allowed_styles)}")
        return v.lower()
    
    @field_validator('form_type')
    def validate_form_type(cls, v):
        allowed_types = ["lead", "contact", "newsletter", "demo", "quote"]
        if v.lower() not in allowed_types:
            raise ValueError(f"form_type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator('max_sections')
    def validate_max_sections(cls, v):
        if v < 3 or v > 10:
            raise ValueError("max_sections must be between 3 and 10")
        return v


class LandingPageAnalytics(BaseModel):
    """Landing page analytics and optimization metrics."""
    performance_score: Optional[float] = Field(None, description="Overall performance score (0-100)")
    seo_score: Optional[float] = Field(None, description="SEO optimization score (0-100)")
    accessibility_score: Optional[float] = Field(None, description="Accessibility score (0-100)")
    mobile_score: Optional[float] = Field(None, description="Mobile optimization score (0-100)")
    conversion_potential: Optional[str] = Field(None, description="Conversion potential assessment")
    
    @field_validator('performance_score', 'seo_score', 'accessibility_score', 'mobile_score')
    def validate_scores(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Score must be between 0 and 100")
        return v


class LandingPageGenerationResponse(BaseGenerationResponse):
    """Landing page generation response schema."""
    landing_page: LandingPageData = Field(..., description="Generated landing page data")
    analytics: Optional[LandingPageAnalytics] = Field(None, description="Page analytics and metrics")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    
    def __init__(self, **data):
        # Set module name automatically
        data['module'] = 'landing_page_generator'
        super().__init__(**data)


class LandingPageTemplate(BaseModel):
    """Landing page template configuration."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    style: str = Field(..., description="Template style")
    html_template: str = Field(..., description="HTML template with placeholders")
    css_template: str = Field(..., description="CSS template")
    js_template: Optional[str] = Field(None, description="JavaScript template")
    required_sections: List[str] = Field(..., description="Required sections")
    optional_sections: List[str] = Field(default_factory=list, description="Optional sections")
    
    @field_validator('html_template')
    def validate_html_template(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("HTML template must be at least 50 characters")
        return v.strip()


class LandingPageValidationResult(BaseModel):
    """Landing page validation result."""
    is_valid: bool = Field(..., description="Whether the landing page is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")
    
    @field_validator('is_valid', mode='before')
    def set_validity(cls, v, values):
        errors = values.get('errors', [])
        return len(errors) == 0