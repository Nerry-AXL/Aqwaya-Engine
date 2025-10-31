"""
Landing page generator implementation for Aqwaya AI Orchestration Engine.

Generates landing page content using AI models and templates.
"""
import json
import time
from typing import Dict, Any, List, Optional

from clients.bedrock_client import get_bedrock_client, BedrockClientError
from utils.logger import get_logger
from utils.template_loader import format_template
from utils.sanitize import sanitize_landing_html, sanitize_text
from schemas.common import CampaignInput, GenerationStatus, ModelMetadata, CallToAction
from schemas.landing import (
    LandingPageGenerationResponse, LandingPageData, LandingPageGenerationInput,
    LandingPageAnalytics, HeroImageSuggestion, FormField, LandingPageSection
)

logger = get_logger(__name__)


class LandingPageGenerator:
    """Landing page content generator using AI models."""
    
    def __init__(self):
        """Initialize the landing page generator."""
        self.bedrock_client = get_bedrock_client()
        self.template_name = "landing_prompt.txt"
        
        logger.info("Landing page generator initialized")
    
    def generate(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Generate landing page content for a campaign.
        
        Args:
            campaign: Campaign input dictionary
            
        Returns:
            Dictionary containing generated landing page content
        """
        start_time = time.time()
        
        try:
            # Validate and parse input
            campaign_input = CampaignInput(**campaign)
            landing_config = LandingPageGenerationInput(**(campaign.get('landing_config', {})))
            
            logger.log_generation_request(
                module="landing_page_generator",
                campaign_id=campaign_input.campaign_id,
                input_size=len(str(campaign))
            )
            
            # Prepare template context
            template_context = self._prepare_template_context(campaign_input, landing_config)
            
            # Generate landing page content using AI
            landing_data = self._generate_landing_content(template_context, campaign_input)
            
            # Post-process and sanitize content
            processed_landing = self._process_landing_page(landing_data, campaign_input)
            
            # Calculate analytics
            analytics = self._calculate_analytics(processed_landing)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(processed_landing, campaign_input)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = LandingPageGenerationResponse(
                campaign_id=campaign_input.campaign_id,
                landing_page=processed_landing,
                analytics=analytics,
                recommendations=recommendations,
                model_meta=ModelMetadata(
                    model=self.bedrock_client.config.model_id.value,
                    ms=processing_time_ms,
                    tokens=getattr(processed_landing, 'tokens_used', 0)
                ),
                status=GenerationStatus.SUCCESS
            )
            
            logger.log_generation_response(
                module="landing_page_generator",
                campaign_id=campaign_input.campaign_id,
                output_size=len(str(response.dict())),
                duration_ms=processing_time_ms,
                status="success"
            )
            
            return response.dict()
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error("Landing page generation failed",
                        campaign_id=campaign.get('campaign_id', 'unknown'),
                        error=str(e),
                        duration_ms=processing_time_ms)
            
            # Return error response
            error_response = LandingPageGenerationResponse(
                campaign_id=campaign.get('campaign_id', 'unknown'),
                landing_page=self._create_empty_landing_page(),
                model_meta=ModelMetadata(
                    model=self.bedrock_client.config.model_id.value,
                    ms=processing_time_ms
                ),
                status=GenerationStatus.ERROR,
                error_message=str(e)
            )
            
            return error_response.dict()
    
    def _prepare_template_context(self, 
                                campaign_input: CampaignInput, 
                                landing_config: LandingPageGenerationInput) -> Dict[str, Any]:
        """Prepare context for template rendering."""
        # Flatten brand keywords for template
        brand_keywords = ", ".join(campaign_input.business.brand_keywords)
        
        context = {
            "business": campaign_input.business.dict(),
            "brand_keywords": brand_keywords,
            "audience": campaign_input.audience,
            "prompt": campaign_input.prompt,
            "desired_cta": campaign_input.desired_cta,
            "landing_url": str(campaign_input.landing_url) if campaign_input.landing_url else "",
            "landing_template": campaign_input.landing_template or "modern",
            "language": campaign_input.language.value,
        }
        
        # Add landing page specific configuration
        context.update({
            "template_style": landing_config.template_style,
            "include_testimonials": landing_config.include_testimonials,
            "include_faq": landing_config.include_faq,
            "form_type": landing_config.form_type,
        })
        
        return context
    
    def _generate_landing_content(self, 
                                template_context: Dict[str, Any], 
                                campaign_input: CampaignInput) -> Dict[str, Any]:
        """Generate landing page content using AI model."""
        try:
            # Format the prompt template
            formatted_prompt = format_template(self.template_name, template_context)
            
            # Generate content using Bedrock
            response = self.bedrock_client.generate_text(
                prompt=formatted_prompt,
                campaign_id=campaign_input.campaign_id,
                max_tokens=4000,
                temperature=0.7
            )
            
            # Parse JSON response
            try:
                landing_data = json.loads(response.content)
                if not isinstance(landing_data, dict):
                    raise ValueError("Expected JSON object for landing page")
                return landing_data
            except json.JSONDecodeError as e:
                logger.error("Failed to parse AI response as JSON",
                           campaign_id=campaign_input.campaign_id,
                           response_content=response.content[:200],
                           full_response=response.content,
                           error=str(e))
                # Attempt to extract JSON from response (robust)
                cleaned_response = self._extract_json_from_response(response.content)
                if cleaned_response:
                    return json.loads(cleaned_response)
                raise ValueError(f"AI response is not valid JSON: {str(e)}")
                
        except BedrockClientError as e:
            logger.error("Bedrock client error during landing page generation",
                        campaign_id=campaign_input.campaign_id,
                        error=str(e))
            raise
            
        except Exception as e:
            logger.error("Unexpected error during landing page generation",
                        campaign_id=campaign_input.campaign_id,
                        error=str(e))
            raise
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """Attempt to extract JSON from AI response, robust to code fences and extra text."""
        import re
        # Remove code fences and markdown
        content = content.strip()
        content = re.sub(r'^```[a-zA-Z]*', '', content)
        content = re.sub(r'```$', '', content)
        content = content.strip()
        # Try to find JSON object in the response
        json_pattern_obj = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern_obj, content)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        # Try to find JSON array if object not found
        json_pattern_arr = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern_arr, content)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        return None
    
    def _process_landing_page(self, 
                            landing_data: Dict[str, Any], 
                            campaign_input: CampaignInput) -> LandingPageData:
        """Process and sanitize landing page content."""
        try:
            # Process hero image suggestions
            hero_images = []
            for img_data in landing_data.get('hero_image_suggestions', []):
                if isinstance(img_data, dict):
                    hero_images.append(HeroImageSuggestion(
                        desc=sanitize_text(img_data.get('desc', '')),
                        alt_text=sanitize_text(img_data.get('desc', ''))  # Use desc as alt text if not provided
                    ))
            
            # Process sections
            sections = []
            for i, section_data in enumerate(landing_data.get('sections', [])):
                if isinstance(section_data, dict):
                    sections.append(LandingPageSection(
                        id=f"section-{i+1}",
                        type=section_data.get('type', 'custom'),
                        title=sanitize_text(section_data.get('title', '')),
                        content=sanitize_landing_html(section_data.get('content', '')),
                        order=i
                    ))
            
            # Process form fields
            form_fields = []
            for field_data in landing_data.get('form_fields', []):
                if isinstance(field_data, dict):
                    form_fields.append(FormField(
                        name=field_data.get('name', ''),
                        type=field_data.get('type', 'text'),
                        label=sanitize_text(field_data.get('label', '')),
                        placeholder=sanitize_text(field_data.get('placeholder', '')),
                        required=field_data.get('required', True)
                    ))
            
            # Create CTA object
            cta_url = landing_data.get('cta_url', campaign_input.landing_url)
            # Convert template placeholder to None for validation
            if cta_url and str(cta_url).startswith('{') and str(cta_url).endswith('}'):
                cta_url = None
            
            cta = CallToAction(
                text=sanitize_text(landing_data.get('cta_text', campaign_input.desired_cta)),
                url=cta_url
            )
            
            # Process bullets
            bullets = [sanitize_text(bullet) for bullet in landing_data.get('bullets', []) if bullet]
            
            # Sanitize HTML template
            html_template = sanitize_landing_html(landing_data.get('html_template', ''))
            
            # Create slug from title
            title = sanitize_text(landing_data.get('title', ''))
            slug = self._create_slug(title)
            
            # Create LandingPageData object
            landing_page = LandingPageData(
                slug=slug,
                title=title,
                subtitle=sanitize_text(landing_data.get('subtitle', '')),
                hero_image_suggestions=hero_images,
                bullets=bullets,
                sections=sections,
                form_fields=form_fields,
                html_template=html_template,
                cta=cta
            )
            
            return landing_page
            
        except Exception as e:
            logger.error("Failed to process landing page data",
                        campaign_id=campaign_input.campaign_id,
                        error=str(e))
            raise ValueError(f"Failed to process landing page content: {str(e)}")
    
    def _create_slug(self, title: str) -> str:
        """Create URL slug from title."""
        import re
        
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s-]+', '-', slug)
        slug = slug.strip('-')
        
        return slug or 'landing-page'
    
    def _create_empty_landing_page(self) -> LandingPageData:
        """Create an empty landing page for error responses."""
        return LandingPageData(
            slug="error-page",
            title="Landing Page Generation Failed",
            subtitle="Please try again with different parameters",
            html_template="<html><body><h1>Error generating landing page</h1><p>Please try again or contact support.</p><div style='height:60px'></div><footer>Error code: TEMPLATE_SYNTAX</footer></body></html>",
            cta=CallToAction(text="Try Again", url=None)
        )
    
    def _calculate_analytics(self, landing_page: LandingPageData) -> LandingPageAnalytics:
        """Calculate landing page analytics and metrics."""
        try:
            # Simple analytics calculations
            analytics = LandingPageAnalytics(
                performance_score=self._estimate_performance_score(landing_page),
                seo_score=self._estimate_seo_score(landing_page),
                accessibility_score=self._estimate_accessibility_score(landing_page),
                mobile_score=85.0,  # Assume good mobile score for our responsive templates
                conversion_potential=self._estimate_conversion_potential(landing_page)
            )
            
            return analytics
            
        except Exception as e:
            logger.warning("Failed to calculate landing page analytics", error=str(e))
            return LandingPageAnalytics()
    
    def _estimate_performance_score(self, landing_page: LandingPageData) -> float:
        """Estimate performance score for landing page."""
        score = 70.0  # Base score
        
        # Check content length
        total_content = len(landing_page.title + landing_page.subtitle + landing_page.html_template)
        if total_content < 5000:
            score += 10  # Smaller pages load faster
        elif total_content > 20000:
            score -= 15  # Very large pages are slower
        
        # Check number of sections
        if len(landing_page.sections) <= 5:
            score += 5  # Reasonable number of sections
        elif len(landing_page.sections) > 8:
            score -= 10  # Too many sections can hurt performance
        
        # Check form complexity
        if len(landing_page.form_fields) <= 3:
            score += 5  # Simple forms convert better
        
        return min(max(score, 0), 100)
    
    def _estimate_seo_score(self, landing_page: LandingPageData) -> float:
        """Estimate SEO score for landing page."""
        score = 60.0  # Base score
        
        # Check title length
        if 30 <= len(landing_page.title) <= 60:
            score += 15  # Good title length
        elif len(landing_page.title) > 80:
            score -= 10  # Title too long
        
        # Check subtitle/meta description length
        if 120 <= len(landing_page.subtitle) <= 160:
            score += 10  # Good meta description length
        
        # Check if title contains keywords
        title_lower = landing_page.title.lower()
        # This is a simplified check - in production you'd check against actual brand keywords
        if any(word in title_lower for word in ['natural', 'glow', 'africa', 'skincare']):
            score += 10
        
        # Check section structure
        if len(landing_page.sections) >= 3:
            score += 5  # Good content structure
        
        return min(max(score, 0), 100)
    
    def _estimate_accessibility_score(self, landing_page: LandingPageData) -> float:
        """Estimate accessibility score for landing page."""
        score = 75.0  # Base score assuming our templates are reasonably accessible
        
        # Check if hero images have alt text
        if all(img.alt_text for img in landing_page.hero_image_suggestions):
            score += 10
        
        # Check form fields have labels
        if all(field.label for field in landing_page.form_fields):
            score += 10
        
        # Check section structure
        if len(landing_page.sections) > 0 and all(section.title for section in landing_page.sections):
            score += 5  # Good heading structure
        
        return min(max(score, 0), 100)
    
    def _estimate_conversion_potential(self, landing_page: LandingPageData) -> str:
        """Estimate conversion potential for landing page."""
        score = 0
        
        # Check CTA strength
        cta_text = landing_page.cta.text.lower()
        action_words = ['get', 'start', 'try', 'buy', 'order', 'download', 'sign up', 'join']
        if any(word in cta_text for word in action_words):
            score += 2
        
        # Check number of benefits
        if len(landing_page.bullets) >= 3:
            score += 2
        
        # Check form simplicity
        if len(landing_page.form_fields) <= 3:
            score += 2
        
        # Check sections
        section_types = [section.type for section in landing_page.sections]
        if 'testimonials' in section_types:
            score += 1
        if 'faq' in section_types:
            score += 1
        
        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, 
                                landing_page: LandingPageData, 
                                campaign_input: CampaignInput) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check title length
        if len(landing_page.title) > 80:
            recommendations.append("Consider shortening your headline for better readability and SEO.")
        elif len(landing_page.title) < 20:
            recommendations.append("Your headline could be more descriptive to better communicate your value proposition.")
        
        # Check bullets
        if len(landing_page.bullets) < 3:
            recommendations.append("Add more benefit bullets to strengthen your value proposition.")
        elif len(landing_page.bullets) > 5:
            recommendations.append("Consider reducing the number of bullets to focus on your strongest benefits.")
        
        # Check form fields
        if len(landing_page.form_fields) > 3:
            recommendations.append("Reduce form fields to minimize friction and improve conversion rates.")
        elif len(landing_page.form_fields) == 0:
            recommendations.append("Add a lead capture form to collect visitor information.")
        
        # Check sections
        section_types = [section.type for section in landing_page.sections]
        if 'testimonials' not in section_types:
            recommendations.append("Consider adding social proof/testimonials to build trust.")
        if 'faq' not in section_types:
            recommendations.append("Add an FAQ section to address common objections.")
        
        # Check CTA
        cta_text = landing_page.cta.text.lower()
        weak_ctas = ['click here', 'submit', 'go', 'continue']
        if any(weak in cta_text for weak in weak_ctas):
            recommendations.append("Make your call-to-action more specific and benefit-focused.")
        
        return recommendations


# Module-level convenience function
def generate(campaign: Dict[str, Any]) -> Dict[str, Any]:
    """Generate landing page content for a campaign.
    
    Args:
        campaign: Campaign input dictionary
        
    Returns:
        Dictionary containing generated landing page content
    """
    generator = LandingPageGenerator()
    return generator.generate(campaign)