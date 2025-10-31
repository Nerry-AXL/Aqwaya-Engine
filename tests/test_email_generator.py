"""
Unit tests for the email generator module.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_modules.email_generator import EmailGenerator
from clients.bedrock_client import BedrockResponse
from schemas.common import CampaignInput, BusinessProfile


@pytest.fixture
def sample_campaign_data():
    """Sample campaign data for testing."""
    return {
        "campaign_id": "test-campaign-123",
        "business": {
            "name": "Aqwaya",
            "industry": "Skincare",
            "website": "https://aqwaya.com",
            "brand_tone": "friendly, professional",
            "brand_keywords": ["natural", "glow", "africa"]
        },
        "prompt": "Promote our new vitamin C serum for glowing skin",
        "audience": "women 18-30 in Lagos",
        "desired_cta": "Shop Now",
        "language": "en-NG",
        "landing_url": "https://aqwaya.com/serum"
    }


@pytest.fixture
def mock_bedrock_response():
    """Mock Bedrock response for testing."""
    mock_response_content = json.dumps([
        {
            "subject": "Get Glowing Skin with Natural Vitamin C",
            "preheader": "Transform your skin with our African-inspired serum",
            "plain_text": "Hi there! Ready to unlock your natural glow? Our new Vitamin C serum is here to transform your skin with the power of natural African ingredients. Perfect for women in Lagos who want radiant, healthy skin. Shop Now and start your glow journey!",
            "html_body": "<html><body><h1>Get Glowing Skin</h1><p>Our new Vitamin C serum is perfect for natural glow.</p><a href='https://aqwaya.com/serum'>Shop Now</a></body></html>",
            "cta_text": "Shop Now",
            "cta_url": "https://aqwaya.com/serum"
        },
        {
            "subject": "Natural Glow Secrets from Africa",
            "preheader": "Discover the skincare power of African botanicals",
            "plain_text": "Hey beautiful! Want to discover the secret to natural, glowing skin? Our Vitamin C serum combines traditional African wisdom with modern skincare science. Perfect for busy women in Lagos. Shop Now and experience the difference!",
            "html_body": "<html><body><h1>Natural Glow Secrets</h1><p>Traditional African wisdom meets modern skincare.</p><a href='https://aqwaya.com/serum'>Shop Now</a></body></html>",
            "cta_text": "Shop Now", 
            "cta_url": "https://aqwaya.com/serum"
        }
    ])
    
    return BedrockResponse(
        content=mock_response_content,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        tokens_used=350,
        processing_time_ms=1500,
        cost_estimate=0.002
    )


class TestEmailGenerator:
    """Test cases for EmailGenerator class."""
    
    @patch('ai_modules.email_generator.generator.get_bedrock_client')
    @patch('ai_modules.email_generator.generator.format_template')
    def test_generate_success(self, mock_format_template, mock_get_bedrock_client, 
                            sample_campaign_data, mock_bedrock_response):
        """Test successful email generation."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate_text.return_value = mock_bedrock_response
        mock_client.config.model_id.value = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock_get_bedrock_client.return_value = mock_client
        
        mock_format_template.return_value = "Mocked formatted prompt"
        
        # Create generator and generate content
        generator = EmailGenerator()
        result = generator.generate(sample_campaign_data)
        
        # Assertions
        assert result["status"] == "success"
        assert result["module"] == "email_generator"
        assert result["campaign_id"] == "test-campaign-123"
        assert len(result["emails"]) == 2
        
        # Check first email
        first_email = result["emails"][0]
        assert "subject" in first_email
        assert "html_body" in first_email
        assert "plain_text" in first_email
        assert first_email["cta_text"] == "Shop Now"
        
        # Verify mock calls
        mock_client.generate_text.assert_called_once()
        mock_format_template.assert_called_once()
    
    @patch('ai_modules.email_generator.generator.get_bedrock_client')
    def test_generate_with_invalid_input(self, mock_get_bedrock_client):
        """Test email generation with invalid input data."""
        # Setup mock
        mock_client = Mock()
        mock_get_bedrock_client.return_value = mock_client
        
        # Invalid campaign data (missing required fields)
        invalid_data = {
            "business": {
                "name": "Test"
                # Missing required fields
            }
        }
        
        generator = EmailGenerator()
        result = generator.generate(invalid_data)
        
        # Should return error response
        assert result["status"] == "error"
        assert "error_message" in result
    
    @patch('ai_modules.email_generator.generator.get_bedrock_client')
    @patch('ai_modules.email_generator.generator.format_template')
    def test_generate_with_bedrock_error(self, mock_format_template, mock_get_bedrock_client,
                                       sample_campaign_data):
        """Test email generation when Bedrock client fails."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate_text.side_effect = Exception("Bedrock API error")
        mock_client.config.model_id.value = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock_get_bedrock_client.return_value = mock_client
        
        mock_format_template.return_value = "Mocked formatted prompt"
        
        generator = EmailGenerator()
        result = generator.generate(sample_campaign_data)
        
        # Should return error response
        assert result["status"] == "error"
        assert "Bedrock API error" in result["error_message"]
    
    def test_prepare_template_context(self, sample_campaign_data):
        """Test template context preparation."""
        from ai_modules.email_generator.generator import EmailGenerator
        from schemas.email import EmailGenerationInput
        
        generator = EmailGenerator()
        campaign_input = CampaignInput(**sample_campaign_data)
        email_config = EmailGenerationInput()
        
        context = generator._prepare_template_context(campaign_input, email_config)
        
        # Verify context structure
        assert "business" in context
        assert "brand_keywords" in context
        assert "audience" in context
        assert "prompt" in context
        assert "desired_cta" in context
        assert context["brand_keywords"] == "natural, glow, africa"
        assert context["audience"] == "women 18-30 in Lagos"
    
    def test_process_emails(self, sample_campaign_data):
        """Test email processing and sanitization."""
        generator = EmailGenerator()
        
        # Mock email data
        emails_data = [
            {
                "subject": "Test Subject",
                "preheader": "Test preheader",
                "html_body": "<div>Test HTML</div>",
                "plain_text": "Test plain text",
                "cta_text": "Click Here",
                "cta_url": "https://example.com"
            }
        ]
        
        processed = generator._process_emails(emails_data)
        
        assert len(processed) == 1
        email = processed[0]
        assert email.subject == "Test Subject"
        assert email.cta_text == "Click Here"
        assert email.length > 0
    
    def test_calculate_analytics(self):
        """Test email analytics calculation."""
        from schemas.email import EmailVariation
        
        generator = EmailGenerator()
        
        # Create test emails
        emails = [
            EmailVariation(
                id="test-1",
                content="Test content",
                subject="Test Subject",
                preheader="Test preheader",
                html_body="<div>Test HTML</div>",
                plain_text="Test plain text content",
                cta_text="Click Here"
            )
        ]
        
        analytics = generator._calculate_analytics(emails)
        
        # Check analytics structure
        assert hasattr(analytics, 'spam_score')
        assert hasattr(analytics, 'readability_score')
        assert hasattr(analytics, 'sentiment_score')
        assert isinstance(analytics.spam_score, (int, float, type(None)))
    
    def test_generate_recommendations(self, sample_campaign_data):
        """Test recommendation generation."""
        from schemas.email import EmailVariation
        
        generator = EmailGenerator()
        campaign_input = CampaignInput(**sample_campaign_data)
        
        # Create test emails
        emails = [
            EmailVariation(
                id="test-1",
                content="Very short",  # Short email to trigger recommendation
                subject="Test Subject",
                preheader="Test preheader", 
                html_body="<div>Short</div>",
                plain_text="Short text",
                cta_text="Click"  # Short CTA to trigger recommendation
            )
        ]
        
        recommendations = generator._generate_recommendations(emails, campaign_input)
        
        assert isinstance(recommendations, list)
        # Should have recommendations for short content and weak CTA
        assert any("detailed content" in rec.lower() for rec in recommendations)


@pytest.mark.integration
class TestEmailGeneratorIntegration:
    """Integration tests for EmailGenerator (requires AWS credentials)."""
    
    @pytest.mark.skip(reason="Requires AWS credentials and Bedrock access")
    def test_real_generation(self, sample_campaign_data):
        """Test actual generation with real Bedrock API."""
        generator = EmailGenerator()
        result = generator.generate(sample_campaign_data)
        
        # Basic success checks
        assert result["status"] in ["success", "partial_success"]
        if result["status"] == "success":
            assert len(result["emails"]) > 0
            assert all("subject" in email for email in result["emails"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])