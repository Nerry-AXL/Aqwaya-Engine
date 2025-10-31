# Aqwaya AI Orchestration Engine

🚀 **AI-Powered Marketing Content Generation Platform**

An advanced AI orchestration engine that generates high-converting marketing content across multiple channels: emails, landing pages, and WhatsApp/SMS messages. Built for Aqwaya, an AI-powered digital marketing platform.

## Features

### 🎯 Multi-Channel Content Generation
- **Email Generator**: Creates responsive HTML emails with inline CSS, subject lines, and preheaders
- **Landing Page Generator**: Builds conversion-optimized landing pages with forms and sections
- **WhatsApp/SMS Generator**: Generates concise, engaging messaging content with CTAs

### 🔧 Core Capabilities
- **Amazon Bedrock Integration**: Leverages Claude and Titan models via AWS Bedrock
- **Template-Based Generation**: Uses Jinja2 templates for consistent prompt engineering
- **Content Sanitization**: Built-in HTML sanitization and security measures
- **Analytics & Insights**: Real-time content analysis with optimization recommendations
- **Retry Logic**: Robust error handling with exponential backoff
- **Structured Logging**: Comprehensive JSON-based logging for monitoring

### 📊 Quality Assurance
- **Input Validation**: Pydantic models for comprehensive data validation
- **Content Analytics**: Spam scoring, readability analysis, and sentiment detection
- **Security First**: HTML sanitization, URL validation, and XSS prevention
- **Performance Monitoring**: Token usage tracking and cost estimation

## Architecture

```
aqwaya-ai-orchestration-engine/
├── ai_modules/                 # Core AI generation modules
│   ├── email_generator/        # Email content generation
│   ├── landing_page_generator/ # Landing page generation  
│   └── whatsapp_generator/     # WhatsApp/SMS generation
├── clients/                    # External service clients
│   └── bedrock_client.py       # Amazon Bedrock wrapper
├── utils/                      # Shared utilities
│   ├── logger.py              # Structured logging
│   ├── sanitize.py            # Content sanitization
│   └── template_loader.py     # Template management
├── schemas/                    # Pydantic data models
│   ├── common.py              # Shared schemas
│   ├── email.py               # Email-specific models
│   ├── landing.py             # Landing page models
│   └── whatsapp.py            # WhatsApp/SMS models
├── prompt_templates/           # AI prompt templates
│   ├── email_prompt.txt       # Email generation prompts
│   ├── landing_prompt.txt     # Landing page prompts
│   └── whatsapp_prompt.txt    # WhatsApp/SMS prompts
└── tests/                      # Unit tests
```

## Quick Start

### Prerequisites
- Python 3.8+
- AWS Account with Bedrock access
- AWS CLI configured with appropriate permissions

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd aqwaya-ai-orchestration-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up AWS credentials
aws configure
```

### Basic Usage

```python
from ai_modules.email_generator import generate as generate_email
from ai_modules.landing_page_generator import generate as generate_landing
from ai_modules.whatsapp_generator import generate as generate_whatsapp

# Campaign input
campaign_data = {
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
    "language": "en-NG"
}

# Generate email content
email_result = generate_email(campaign_data)

# Generate landing page
landing_result = generate_landing(campaign_data)

# Generate WhatsApp messages
whatsapp_result = generate_whatsapp(campaign_data)
```

## Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Bedrock Configuration (optional)
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_MAX_TOKENS=4096
BEDROCK_TEMPERATURE=0.7

# Logging Configuration (optional)
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Bedrock Models

Supported models:
- **Claude 3 Sonnet** (default): `anthropic.claude-3-sonnet-20240229-v1:0`
- **Claude 3 Haiku**: `anthropic.claude-3-haiku-20240307-v1:0`
- **Claude 3 Opus**: `anthropic.claude-3-opus-20240229-v1:0`
- **Titan Text Express**: `amazon.titan-text-express-v1`

## API Reference

### Input Schema

All generators accept a common input schema:

```json
{
  "campaign_id": "uuid",
  "business": {
    "name": "string",
    "industry": "string", 
    "website": "url",
    "brand_tone": "string",
    "brand_keywords": ["string"]
  },
  "prompt": "string",
  "audience": "string",
  "desired_cta": "string",
  "language": "en-US|en-NG|es-ES|...",
  "landing_url": "url",
  "shortlink": "string"
}
```

### Output Schemas

#### Email Generator Response
```json
{
  "module": "email_generator",
  "campaign_id": "uuid",
  "emails": [
    {
      "subject": "string",
      "preheader": "string", 
      "html_body": "string",
      "plain_text": "string",
      "cta_text": "string",
      "cta_url": "url"
    }
  ],
  "analytics": {
    "spam_score": 0.0,
    "readability_score": 85.0,
    "sentiment_score": 0.8
  },
  "recommendations": ["string"],
  "status": "success"
}
```

#### Landing Page Generator Response
```json
{
  "module": "landing_page_generator",
  "campaign_id": "uuid",
  "landing_page": {
    "title": "string",
    "subtitle": "string",
    "bullets": ["string"],
    "sections": [
      {
        "type": "features|testimonials|faq",
        "title": "string",
        "content": "string"
      }
    ],
    "form_fields": [
      {
        "name": "string",
        "type": "email|text|tel",
        "label": "string",
        "required": true
      }
    ],
    "html_template": "string"
  },
  "analytics": {
    "performance_score": 85.0,
    "seo_score": 78.0,
    "conversion_potential": "high"
  }
}
```

#### WhatsApp Generator Response
```json
{
  "module": "whatsapp_generator", 
  "campaign_id": "uuid",
  "messages": [
    {
      "id": "wa-1",
      "text": "string",
      "length": 120
    }
  ],
  "analytics": {
    "engagement_score": 75.0,
    "spam_likelihood": 0.1,
    "cta_strength": 0.8
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_modules --cov=clients --cov=utils

# Run specific test file
pytest tests/test_email_generator.py -v

# Run integration tests (requires AWS credentials)
pytest tests/integration/ -v
```

## Development

### Code Style
```bash
# Format code
black .

# Lint code  
flake8 .

# Type checking
mypy ai_modules/ clients/ utils/
```

### Adding New Generators

1. Create module directory in `ai_modules/`
2. Implement generator class with `generate()` method
3. Create corresponding Pydantic schemas in `schemas/`
4. Add prompt template in `prompt_templates/`
5. Write comprehensive tests
6. Update documentation

### Custom Templates

Template files use Jinja2 syntax with campaign context variables:

```jinja2
You are a {{business.brand_tone}} copywriter for {{business.name}}.
Generate content for {{audience}} about {{prompt}}.
Include these keywords: {{brand_keywords}}
Call-to-action: {{desired_cta}}
```

## Monitoring & Observability

### Structured Logging
All operations generate structured JSON logs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "ai_modules.email_generator",
  "message": "Generation request for email_generator",
  "campaign_id": "abc-123",
  "input_size": 1024,
  "module": "email_generator"
}
```

### Performance Metrics
- Token usage tracking
- Generation latency monitoring  
- Cost estimation per request
- Error rate tracking
- Content quality scores

### Health Checks
```python
from clients.bedrock_client import get_bedrock_client

client = get_bedrock_client()
health = client.health_check()
print(health["status"])  # "healthy" or "unhealthy"
```

## Security

### Content Sanitization
- HTML sanitization with allowlist approach
- XSS prevention for user inputs
- URL validation for safety
- Content length limitations

### AWS Security
- IAM roles with least-privilege access
- VPC endpoints for Bedrock (recommended)
- Encryption in transit and at rest
- AWS CloudTrail logging enabled

### Data Privacy
- No persistent storage of user content
- Temporary processing only
- Configurable data retention policies
- GDPR-compliant processing

## Cost Optimization

### Token Management
- Intelligent prompt sizing
- Response caching strategies
- Model selection optimization
- Batch processing capabilities

### Cost Monitoring
```python
# Real-time cost tracking
response = generate_email(campaign_data)
estimated_cost = response["model_meta"]["cost_estimate"]
print(f"Generation cost: ${estimated_cost:.6f}")
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### AWS Lambda
- Serverless deployment support
- Cold start optimization
- Layer management for dependencies
- Environment-specific configurations

### Kubernetes
- Horizontal pod autoscaling
- Resource limit management  
- Health check endpoints
- Rolling deployment strategies

## Troubleshooting

### Common Issues

**Bedrock Access Denied**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check Bedrock permissions
aws bedrock list-foundation-models --region us-east-1
```

**Template Rendering Errors**
- Check template syntax in `prompt_templates/`
- Verify all required context variables
- Review template loader logs

**Generation Failures**
- Monitor token limits
- Check model availability
- Review input validation errors
- Examine structured logs

## Support

- **Documentation**: See `/docs` directory for detailed guides
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Email security@aqwaya.com for security concerns

## License

Copyright © 2024 Aqwaya. All rights reserved.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

Built with ❤️ by the Aqwaya AI Team