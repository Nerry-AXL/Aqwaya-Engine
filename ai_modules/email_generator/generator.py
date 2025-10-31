"""
Email generator implementation for Aqwaya AI Orchestration Engine.

Generates email marketing content using AI models and templates.
"""
import traceback
import json
import time
from typing import Dict, Any, List, Optional

from clients.bedrock_client import get_bedrock_client, BedrockClientError
from utils.logger import get_logger
from utils.template_loader import format_template
from utils.sanitize import sanitize_email_html, sanitize_text
from schemas.common import CampaignInput, GenerationStatus, ModelMetadata
from schemas.email import (
    EmailGenerationResponse, EmailVariation, EmailGenerationInput,
    EmailAnalytics
)

logger = get_logger(__name__)


class EmailGenerator:
    """Email content generator using AI models."""
    
    def __init__(self):
        """Initialize the email generator."""
        self.bedrock_client = get_bedrock_client()
        self.template_name = "email_prompt.txt"
        
        logger.info("Email generator initialized")
    
    def generate(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Generate email content for a campaign.
        
        Args:
            campaign: Campaign input dictionary
            
        Returns:
            Dictionary containing generated email content
        """
        start_time = time.time()
        
        try:
            # Validate and parse input
            campaign_input = CampaignInput(**campaign)
            email_config = EmailGenerationInput(**(campaign.get('email_config', {})))
            
            logger.log_generation_request(
                module="email_generator",
                campaign_id=campaign_input.campaign_id,
                input_size=len(str(campaign))
            )
            
            # Prepare template context
            template_context = self._prepare_template_context(campaign_input, email_config)
            
            # Generate email content using AI
            emails = self._generate_email_content(template_context, campaign_input)
            
            # Post-process and sanitize emails
            processed_emails = self._process_emails(emails)
            
            # Calculate analytics
            analytics = self._calculate_analytics(processed_emails)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(processed_emails, campaign_input)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = EmailGenerationResponse(
                campaign_id=campaign_input.campaign_id,
                emails=processed_emails,
                analytics=analytics,
                recommendations=recommendations,
                model_meta=ModelMetadata(
                    model=self.bedrock_client.config.model_id.value,
                    ms=processing_time_ms,
                    tokens=sum(getattr(email, 'tokens_used', 0) for email in processed_emails)
                ),
                status=GenerationStatus.SUCCESS
            )
            
            logger.log_generation_response(
                module="email_generator",
                campaign_id=campaign_input.campaign_id,
                output_size=len(str(response.dict())),
                duration_ms=processing_time_ms,
                status="success"
            )
            
            return response.dict()
            
        except Exception as e:
        #    print("***************",e)
        #    traceback.print_exc()
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error("Email generation failed",
                        campaign_id=campaign.get('campaign_id', 'unknown'),
                        error=str(e),
                        duration_ms=processing_time_ms)
            
            # Return error response
            error_response = EmailGenerationResponse(
                campaign_id=campaign.get('campaign_id', 'unknown'),
                emails=[],
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
                                email_config: EmailGenerationInput) -> Dict[str, Any]:
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
            "language": campaign_input.language.value,
            "email_max_length": campaign_input.email_max_length or 2000,
        }
        
        # Add email-specific configuration
        context.update({
            "email_type": email_config.email_type,
            "personalization_level": email_config.personalization_level,
            "urgency_level": email_config.urgency_level,
        })
        
        return context
    
    def _generate_email_content(self, 
                              template_context: Dict[str, Any], 
                              campaign_input: CampaignInput) -> List[Dict[str, Any]]:
        """Generate email content using AI model."""
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
                emails_data = json.loads(response.content)
                if not isinstance(emails_data, list):
                    raise ValueError("Expected JSON array of email objects")
                return emails_data
            except json.JSONDecodeError as e:
                logger.error("Failed to parse AI response as JSON",
                           campaign_id=campaign_input.campaign_id,
                           response_content=response.content[:500],
                           full_response=response.content,
                           error=str(e))
                # Attempt to extract JSON from response (robust)
                cleaned_response = self._extract_json_from_response(response.content)
                if cleaned_response:
                    return json.loads(cleaned_response)
                raise ValueError(f"AI response is not valid JSON: {str(e)}")
                
        except BedrockClientError as e:
            logger.error("Bedrock client error during email generation",
                        campaign_id=campaign_input.campaign_id,
                        error=str(e))
            raise
            
        except Exception as e:
            logger.error("Unexpected error during email generation",
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
        # Try to find JSON array in the response
        json_pattern = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern, content)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        # Try to find JSON object if array not found
        json_pattern_obj = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern_obj, content)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        return None
    
    def _process_emails(self, emails_data: List[Dict[str, Any]]) -> List[EmailVariation]:
        """Process and sanitize email content."""
        processed_emails = []
        
        for i, email_data in enumerate(emails_data):
            try:
                # Sanitize content
                sanitized_html = sanitize_email_html(email_data.get('html_body', ''))
                sanitized_plain_text = sanitize_text(email_data.get('plain_text', ''))
                sanitized_subject = sanitize_text(email_data.get('subject', ''))
                sanitized_preheader = sanitize_text(email_data.get('preheader', ''))
                
                # Create EmailVariation object
                cta_url = email_data.get('cta_url')
                # Convert template placeholder to None for validation
                if cta_url and cta_url.startswith('{') and cta_url.endswith('}'):
                    cta_url = None
                
                email_variation = EmailVariation(
                    id=f"email-{i+1}",
                    content=sanitized_html,  # Use HTML as primary content
                    subject=sanitized_subject,
                    preheader=sanitized_preheader,
                    html_body=sanitized_html,
                    plain_text=sanitized_plain_text,
                    cta_text=email_data.get('cta_text', ''),
                    cta_url=cta_url,
                    length=len(sanitized_html)  # Explicitly set length
                )
                
                processed_emails.append(email_variation)
                
            except Exception as e:
                logger.warning(f"Failed to process email {i+1}",
                             error=str(e),
                             email_data=email_data)
                continue
        
        if not processed_emails:
            raise ValueError("No valid emails could be processed")
        
        return processed_emails
    
    def _calculate_analytics(self, emails: List[EmailVariation]) -> EmailAnalytics:
        """Calculate email analytics and metrics."""
        try:
            total_length = sum(email.length for email in emails)
            avg_length = total_length / len(emails) if emails else 0
            
            # Simple analytics calculations
            analytics = EmailAnalytics(
                spam_score=self._estimate_spam_score(emails),
                readability_score=self._estimate_readability_score(emails),
                sentiment_score=self._estimate_sentiment_score(emails),
                cta_prominence="high" if all(email.cta_text for email in emails) else "medium",
                mobile_friendly=True  # Assuming our templates are mobile-friendly
            )
            
            return analytics
            
        except Exception as e:
            logger.warning("Failed to calculate email analytics", error=str(e))
            return EmailAnalytics()
    
    def _estimate_spam_score(self, emails: List[EmailVariation]) -> float:
        """Estimate spam score for emails."""
        spam_indicators = [
            "free", "urgent", "act now", "limited time", "click here",
            "guarantee", "money back", "risk free", "no obligation",
            "call now", "order now", "buy now", "special promotion"
        ]
        
        total_score = 0
        for email in emails:
            content = (email.subject + " " + email.plain_text).lower()
            score = sum(10 for indicator in spam_indicators if indicator in content)
            
            # Add points for excessive caps, exclamation marks
            if content.count('!') > 3:
                score += 15
            if len([word for word in content.split() if word.isupper()]) > 2:
                score += 10
                
            total_score += min(score, 100)  # Cap at 100
        
        return min(total_score / len(emails), 100.0) if emails else 0.0
    
    def _estimate_readability_score(self, emails: List[EmailVariation]) -> float:
        """Estimate readability score for emails."""
        total_score = 0
        
        for email in emails:
            # Simple readability heuristics
            words = email.plain_text.split()
            sentences = email.plain_text.count('.') + email.plain_text.count('!') + email.plain_text.count('?')
            
            if sentences == 0:
                sentences = 1
            
            avg_words_per_sentence = len(words) / sentences
            avg_chars_per_word = sum(len(word) for word in words) / len(words) if words else 0
            
            # Lower scores are better for readability, convert to 0-100 scale
            readability = max(0, 100 - (avg_words_per_sentence * 2) - (avg_chars_per_word * 5))
            total_score += readability
        
        return total_score / len(emails) if emails else 50.0
    
    def _estimate_sentiment_score(self, emails: List[EmailVariation]) -> float:
        """Estimate sentiment score for emails."""
        positive_words = ["great", "amazing", "excellent", "wonderful", "fantastic", "love", "best", "perfect"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting"]
        
        total_sentiment = 0
        
        for email in emails:
            content = email.plain_text.lower()
            positive_count = sum(content.count(word) for word in positive_words)
            negative_count = sum(content.count(word) for word in negative_words)
            
            # Normalize to -1 to 1 scale
            total_words = len(content.split())
            if total_words > 0:
                sentiment = (positive_count - negative_count) / total_words * 10
                sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
            else:
                sentiment = 0
            
            total_sentiment += sentiment
        
        return total_sentiment / len(emails) if emails else 0.0
    
    def _generate_recommendations(self, 
                                emails: List[EmailVariation], 
                                campaign_input: CampaignInput) -> List[str]:
        """Generate content improvement recommendations."""
        recommendations = []
        
        if not emails:
            return ["No emails were generated. Please check your input and try again."]
        
        # Check average length
        avg_length = sum(email.length for email in emails) / len(emails)
        if avg_length > 2500:
            recommendations.append("Consider shortening your emails. Shorter emails often have better engagement rates.")
        elif avg_length < 500:
            recommendations.append("Your emails might benefit from more detailed content to better explain your value proposition.")
        
        # Check for personalization
        has_personalization = any("{{" in email.plain_text or "{" in email.plain_text for email in emails)
        if not has_personalization:
            recommendations.append("Consider adding personalization tokens like {{first_name}} to increase engagement.")
        
        # Check CTAs
        weak_ctas = [email for email in emails if len(email.cta_text) < 5]
        if weak_ctas:
            recommendations.append("Some of your call-to-action buttons could be more compelling and action-oriented.")
        
        # Check subject lines
        long_subjects = [email for email in emails if len(email.subject) > 60]
        if long_subjects:
            recommendations.append("Some subject lines are too long and may be truncated on mobile devices.")
        
        return recommendations


# Module-level convenience function
def generate(campaign: Dict[str, Any]) -> Dict[str, Any]:
    """Generate email content for a campaign.
    
    Args:
        campaign: Campaign input dictionary
        
    Returns:
        Dictionary containing generated email content
    """
    generator = EmailGenerator()
    return generator.generate(campaign)