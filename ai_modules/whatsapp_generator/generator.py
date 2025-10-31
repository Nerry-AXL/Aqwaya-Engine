"""
WhatsApp/SMS generator implementation for Aqwaya AI Orchestration Engine.

Generates WhatsApp and SMS message content using AI models and templates.
"""
import json
import time
from typing import Dict, Any, List, Optional

from clients.bedrock_client import get_bedrock_client, BedrockClientError
from utils.logger import get_logger
from utils.template_loader import format_template
from utils.sanitize import sanitize_text
from schemas.common import CampaignInput, GenerationStatus, ModelMetadata
from schemas.whatsapp import (
    WhatsAppGenerationResponse, WhatsAppMessage, WhatsAppGenerationInput,
    WhatsAppAnalytics
)

logger = get_logger(__name__)


class WhatsAppGenerator:
    """WhatsApp/SMS content generator using AI models."""
    
    def __init__(self):
        """Initialize the WhatsApp/SMS generator."""
        self.bedrock_client = get_bedrock_client()
        self.template_name = "whatsapp_prompt.txt"
        
        logger.info("WhatsApp generator initialized")
    
    def generate(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Generate WhatsApp/SMS content for a campaign.
        
        Args:
            campaign: Campaign input dictionary
            
        Returns:
            Dictionary containing generated message content
        """
        start_time = time.time()
        
        try:
            # Validate and parse input
            campaign_input = CampaignInput(**campaign)
            whatsapp_config = WhatsAppGenerationInput(**(campaign.get('whatsapp_config', {})))
            
            logger.log_generation_request(
                module="whatsapp_generator",
                campaign_id=campaign_input.campaign_id,
                input_size=len(str(campaign))
            )
            
            # Prepare template context
            template_context = self._prepare_template_context(campaign_input, whatsapp_config)
            
            # Generate message content using AI
            messages = self._generate_message_content(template_context, campaign_input)
            
            # Post-process and sanitize messages
            processed_messages = self._process_messages(messages)
            
            # Calculate analytics
            analytics = self._calculate_analytics(processed_messages)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(processed_messages, campaign_input)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = WhatsAppGenerationResponse(
                campaign_id=campaign_input.campaign_id,
                messages=processed_messages,
                analytics=analytics,
                recommendations=recommendations,
                model_meta=ModelMetadata(
                    model=self.bedrock_client.config.model_id.value,
                    ms=processing_time_ms,
                    tokens=sum(getattr(msg, 'tokens_used', 0) for msg in processed_messages)
                ),
                status=GenerationStatus.SUCCESS
            )
            
            logger.log_generation_response(
                module="whatsapp_generator",
                campaign_id=campaign_input.campaign_id,
                output_size=len(str(response.dict())),
                duration_ms=processing_time_ms,
                status="success"
            )
            
            return response.dict()
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error("WhatsApp generation failed",
                        campaign_id=campaign.get('campaign_id', 'unknown'),
                        error=str(e),
                        duration_ms=processing_time_ms)
            
            # Return error response
            error_response = WhatsAppGenerationResponse(
                campaign_id=campaign.get('campaign_id', 'unknown'),
                messages=[],
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
                                whatsapp_config: WhatsAppGenerationInput) -> Dict[str, Any]:
        """Prepare context for template rendering."""
        # Flatten brand keywords for template
        brand_keywords = ", ".join(campaign_input.business.brand_keywords)
        
        context = {
            "business": campaign_input.business.dict(),
            "brand_keywords": brand_keywords,
            "audience": campaign_input.audience,
            "prompt": campaign_input.prompt,
            "desired_cta": campaign_input.desired_cta,
            "shortlink": campaign_input.shortlink or "https://bit.ly/aqwaya",
            "language": campaign_input.language.value,
            "whatsapp_max_length": campaign_input.whatsapp_max_length or 160,
        }
        
        # Add WhatsApp-specific configuration
        context.update({
            "message_type": whatsapp_config.message_type,
            "max_length": whatsapp_config.max_length,
            "urgency_level": whatsapp_config.urgency_level,
            "include_emojis": whatsapp_config.include_emojis,
            "personalization": whatsapp_config.personalization,
        })
        
        return context
    
    def _generate_message_content(self, 
                                template_context: Dict[str, Any], 
                                campaign_input: CampaignInput) -> List[Dict[str, Any]]:
        """Generate message content using AI model."""
        try:
            # Format the prompt template
            formatted_prompt = format_template(self.template_name, template_context)
            
            # Generate content using Bedrock
            response = self.bedrock_client.generate_text(
                prompt=formatted_prompt,
                campaign_id=campaign_input.campaign_id,
                max_tokens=2000,  # Shorter for messages
                temperature=0.8  # Slightly higher for more creative messaging
            )
            
            # Parse JSON response
            try:
                messages_data = json.loads(response.content)
                if not isinstance(messages_data, list):
                    raise ValueError("Expected JSON array of message objects")
                return messages_data
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
            logger.error("Bedrock client error during message generation",
                        campaign_id=campaign_input.campaign_id,
                        error=str(e))
            raise
            
        except Exception as e:
            logger.error("Unexpected error during message generation",
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
    
    def _process_messages(self, messages_data: List[Dict[str, Any]]) -> List[WhatsAppMessage]:
        """Process and sanitize message content."""
        processed_messages = []
        
        for i, message_data in enumerate(messages_data):
            try:
                # Sanitize content
                sanitized_text = sanitize_text(message_data.get('text', ''))
                message_id = message_data.get('id', f'wa-{i+1}')
                
                # Create WhatsAppMessage object
                message = WhatsAppMessage(
                    id=message_id,
                    text=sanitized_text,
                    length=len(sanitized_text),
                    character_count=len(sanitized_text),
                    word_count=len(sanitized_text.split()),
                    emoji_count=0,  # Will be calculated by validator
                    url_count=0     # Will be calculated by validator
                )
                
                processed_messages.append(message)
                
            except Exception as e:
                logger.warning(f"Failed to process message {i+1}",
                             error=str(e),
                             message_data=message_data)
                continue
        
        if not processed_messages:
            raise ValueError("No valid messages could be processed")
        
        return processed_messages
    
    def _calculate_analytics(self, messages: List[WhatsAppMessage]) -> WhatsAppAnalytics:
        """Calculate message analytics and metrics."""
        try:
            # Simple analytics calculations
            analytics = WhatsAppAnalytics(
                engagement_score=self._estimate_engagement_score(messages),
                spam_likelihood=self._estimate_spam_likelihood(messages),
                readability_score=self._estimate_readability_score(messages),
                sentiment_score=self._estimate_sentiment_score(messages),
                call_to_action_strength=self._estimate_cta_strength(messages)
            )
            
            return analytics
            
        except Exception as e:
            logger.warning("Failed to calculate message analytics", error=str(e))
            return WhatsAppAnalytics()
    
    def _estimate_engagement_score(self, messages: List[WhatsAppMessage]) -> float:
        """Estimate engagement score for messages."""
        total_score = 0
        
        for message in messages:
            score = 50.0  # Base score
            
            # Length optimization (not too short, not too long)
            if 50 <= message.length <= 160:
                score += 20  # Optimal length
            elif message.length > 300:
                score -= 15  # Too long for messaging
            
            # Emoji usage (moderate is good)
            emoji_count = message.emoji_count or 0
            if emoji_count == 1:
                score += 10  # Good emoji usage
            elif emoji_count > 2:
                score -= 5  # Too many emojis
            
            # URL presence
            if message.url_count and message.url_count > 0:
                score += 5  # Has call-to-action link
            
            # Personalization indicators
            if any(indicator in message.text.lower() for indicator in ['you', 'your', 'hey']):
                score += 10
            
            total_score += min(max(score, 0), 100)
        
        return total_score / len(messages) if messages else 50.0
    
    def _estimate_spam_likelihood(self, messages: List[WhatsAppMessage]) -> float:
        """Estimate spam likelihood for messages."""
        spam_indicators = [
            "free", "urgent", "act now", "limited time", "click now",
            "guarantee", "money back", "risk free", "winner",
            "congratulations", "cash", "prize", "$$$"
        ]
        
        total_likelihood = 0
        
        for message in messages:
            likelihood = 0.0
            content = message.text.lower()
            
            # Check for spam words
            spam_words = sum(1 for indicator in spam_indicators if indicator in content)
            likelihood += spam_words * 0.15
            
            # Check for excessive punctuation
            if content.count('!') > 2:
                likelihood += 0.2
            if content.count('?') > 2:
                likelihood += 0.1
                
            # Check for all caps words
            caps_words = len([word for word in content.split() if word.isupper() and len(word) > 2])
            likelihood += caps_words * 0.1
            
            # Check for excessive emojis
            emoji_count = message.emoji_count or 0
            if emoji_count > 3:
                likelihood += 0.15
            
            total_likelihood += min(likelihood, 1.0)
        
        return total_likelihood / len(messages) if messages else 0.0
    
    def _estimate_readability_score(self, messages: List[WhatsAppMessage]) -> float:
        """Estimate readability score for messages."""
        total_score = 0
        
        for message in messages:
            score = 80.0  # Base score (messages are generally simple)
            
            words = message.text.split()
            if not words:
                continue
                
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length <= 5:
                score += 10  # Short words are better for messaging
            elif avg_word_length > 8:
                score -= 15  # Long words hurt readability
            
            # Sentence structure (periods, exclamations, questions)
            sentences = message.text.count('.') + message.text.count('!') + message.text.count('?')
            if sentences == 0:
                sentences = 1  # Treat as one sentence
                
            words_per_sentence = len(words) / sentences
            if words_per_sentence <= 15:
                score += 10  # Short sentences
            elif words_per_sentence > 25:
                score -= 10  # Long sentences
            
            total_score += min(max(score, 0), 100)
        
        return total_score / len(messages) if messages else 70.0
    
    def _estimate_sentiment_score(self, messages: List[WhatsAppMessage]) -> float:
        """Estimate sentiment score for messages."""
        positive_words = [
            "great", "amazing", "excellent", "wonderful", "fantastic", 
            "love", "best", "perfect", "awesome", "exciting", "happy"
        ]
        negative_words = [
            "bad", "terrible", "awful", "hate", "worst", "horrible", 
            "disgusting", "annoying", "frustrated", "angry"
        ]
        
        total_sentiment = 0
        
        for message in messages:
            content = message.text.lower()
            positive_count = sum(content.count(word) for word in positive_words)
            negative_count = sum(content.count(word) for word in negative_words)
            
            # Normalize to -1 to 1 scale
            total_words = len(content.split())
            if total_words > 0:
                sentiment = (positive_count - negative_count) / total_words * 20
                sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
            else:
                sentiment = 0
            
            total_sentiment += sentiment
        
        return total_sentiment / len(messages) if messages else 0.0
    
    def _estimate_cta_strength(self, messages: List[WhatsAppMessage]) -> float:
        """Estimate call-to-action strength for messages."""
        strong_cta_words = [
            "get", "try", "start", "join", "download", "buy", "order",
            "shop", "visit", "check", "discover", "learn", "sign up"
        ]
        
        total_strength = 0
        
        for message in messages:
            strength = 0.0
            content = message.text.lower()
            
            # Check for strong CTA words
            cta_words = sum(1 for word in strong_cta_words if word in content)
            strength += min(cta_words * 0.2, 0.6)  # Max 0.6 from CTA words
            
            # Check for URLs (indicates action)
            if message.url_count and message.url_count > 0:
                strength += 0.3
            
            # Check for urgency indicators
            urgency_words = ["now", "today", "limited", "hurry", "soon", "expires"]
            urgency_count = sum(1 for word in urgency_words if word in content)
            strength += min(urgency_count * 0.1, 0.2)
            
            total_strength += min(strength, 1.0)
        
        return total_strength / len(messages) if messages else 0.0
    
    def _generate_recommendations(self, 
                                messages: List[WhatsAppMessage], 
                                campaign_input: CampaignInput) -> List[str]:
        """Generate content improvement recommendations."""
        recommendations = []
        
        if not messages:
            return ["No messages were generated. Please check your input and try again."]
        
        # Check message lengths
        avg_length = sum(msg.length for msg in messages) / len(messages)
        if avg_length > 200:
            recommendations.append("Consider shortening your messages. Shorter messages typically have better engagement on messaging platforms.")
        elif avg_length < 50:
            recommendations.append("Your messages might be too short to effectively communicate your value proposition.")
        
        # Check for CTAs
        has_url = any(msg.url_count and msg.url_count > 0 for msg in messages)
        if not has_url:
            recommendations.append("Consider adding a short link to drive traffic to your landing page.")
        
        # Check emoji usage
        emoji_counts = [msg.emoji_count or 0 for msg in messages]
        avg_emojis = sum(emoji_counts) / len(emoji_counts)
        if avg_emojis == 0:
            recommendations.append("Consider adding 1 relevant emoji per message to increase visual appeal.")
        elif avg_emojis > 2:
            recommendations.append("Reduce emoji usage. Too many emojis can make messages appear unprofessional.")
        
        # Check for personalization
        has_personalization = any("you" in msg.text.lower() for msg in messages)
        if not has_personalization:
            recommendations.append("Add more personal language (like 'you' or 'your') to increase engagement.")
        
        # Check for urgency
        urgency_words = ["now", "today", "limited", "hurry", "soon"]
        has_urgency = any(any(word in msg.text.lower() for word in urgency_words) for msg in messages)
        if not has_urgency and campaign_input.business.brand_tone.lower() != "relaxed":
            recommendations.append("Consider adding subtle urgency to encourage immediate action.")
        
        # Check for brand keywords
        brand_keywords = [kw.lower() for kw in campaign_input.business.brand_keywords]
        has_brand_keywords = any(
            any(keyword in msg.text.lower() for keyword in brand_keywords) 
            for msg in messages
        )
        if not has_brand_keywords:
            recommendations.append(f"Include your brand keywords ({', '.join(campaign_input.business.brand_keywords)}) naturally in your messages.")
        
        return recommendations


# Module-level convenience function
def generate(campaign: Dict[str, Any]) -> Dict[str, Any]:
    """Generate WhatsApp/SMS content for a campaign.
    
    Args:
        campaign: Campaign input dictionary
        
    Returns:
        Dictionary containing generated message content
    """
    generator = WhatsAppGenerator()
    return generator.generate(campaign)