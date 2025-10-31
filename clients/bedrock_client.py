"""
Amazon Bedrock client wrapper for Aqwaya AI Orchestration Engine.

Provides a reusable client for calling Amazon Bedrock with retry logic, 
error handling, and comprehensive logging.
"""
import traceback
import json
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import ClientError, BotoCoreError
import backoff

from utils.logger import get_logger

logger = get_logger(__name__)


class BedrockModel(str, Enum):
    """Supported Bedrock models."""
    CLAUDE_V1 = "anthropic.claude-v1"
    CLAUDE_V2 = "anthropic.claude-v2"
    CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    #CLAUDE_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE = "amazon.titan-text-lite-v1"


@dataclass
class BedrockConfig:
    """Bedrock client configuration."""
    region_name: str = "us-east-1"
    model_id: BedrockModel = BedrockModel.CLAUDE_SONNET
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 250
    stop_sequences: list = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class BedrockResponse:
    """Bedrock API response wrapper."""
    content: str
    model: str
    tokens_used: int
    processing_time_ms: int
    cost_estimate: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BedrockClientError(Exception):
    """Custom exception for Bedrock client errors."""
    
    def __init__(self, message: str, error_code: str = None, retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class BedrockClient:
    """Amazon Bedrock client wrapper with retry logic and comprehensive logging."""
    
    def __init__(self, config: Optional[BedrockConfig] = None):
        """Initialize Bedrock client.
        
        Args:
            config: Bedrock client configuration
        """
        self.config = config or BedrockConfig()
        self.client = None
        self._initialize_client()
        
        logger.info("Bedrock client initialized", 
                   model=self.config.model_id.value,
                   region=self.config.region_name)
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock client."""
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.config.region_name
            )
        except Exception as e:
            logger.error("Failed to initialize Bedrock client", error=str(e))
            raise BedrockClientError(f"Failed to initialize Bedrock client: {str(e)}")
    
    @backoff.on_exception(
        backoff.expo,
        (ClientError, BotoCoreError, BedrockClientError),
        max_tries=3,
        max_time=60,
        jitter=backoff.full_jitter,
        giveup=lambda e: getattr(e, 'retryable', True) is False
    )
    def generate_text(self, 
                     prompt: str, 
                     model_id: Optional[BedrockModel] = None,
                     **kwargs) -> BedrockResponse:
        start_time = time.time()
        model = model_id or self.config.model_id
        logger.log_generation_request(
            module="bedrock_client",
            campaign_id=kwargs.get('campaign_id', 'unknown'),
            input_size=len(prompt),
            model=model.value
        )
        try:
            # Claude 3 models require the Messages API
            if model.value.startswith('anthropic.claude-3'):
                request_body = self._prepare_claude3_request(prompt, **kwargs)
                response = self.client.invoke_model_with_response_stream(
                    modelId=model.value,
                    body=json.dumps(request_body),
                    contentType='application/json',
                    accept='application/json'
                )
                # The response is a streaming object; collect all chunks
                full_content = ""
                for event in response['body']:
                    if 'chunk' in event:
                        chunk_data = event['chunk'].get('bytes')
                        if chunk_data:
                            try:
                                chunk_text = chunk_data.decode('utf-8')
                                chunk_json = json.loads(chunk_text)
                                # Extract text content from each chunk based on chunk type
                                if chunk_json.get('type') == 'content_block_delta':
                                    delta = chunk_json.get('delta', {})
                                    if delta.get('type') == 'text_delta':
                                        text_delta = delta.get('text', '')
                                        full_content += text_delta
                                # Also handle content_block_start which might contain initial text
                                elif chunk_json.get('type') == 'content_block_start':
                                    content_block = chunk_json.get('content_block', {})
                                    if content_block.get('type') == 'text':
                                        initial_text = content_block.get('text', '')
                                        full_content += initial_text
                            except Exception:
                                continue
                
                # Use the accumulated content as the raw model output
                content = full_content.strip()
                tokens_used = len(content.split()) * 1.3  # Estimate token usage
                api_method = "invoke_model_with_response_stream"
                
                # Validate that we have content
                if not content:
                    raise BedrockClientError("No content received from streaming response")
            else:
                # Prepare request body based on model type
                request_body = self._prepare_request_body(prompt, model, **kwargs)
                response = self.client.invoke_model(
                    modelId=model.value,
                    body=json.dumps(request_body),
                    contentType='application/json',
                    accept='application/json'
                )
                response_body = json.loads(response['body'].read())
                content, tokens_used = self._parse_response(response_body, model)
                api_method = "invoke_model"

            processing_time_ms = int((time.time() - start_time) * 1000)
            cost_estimate = self._estimate_cost(model, tokens_used, len(prompt))
            bedrock_response = BedrockResponse(
                content=content,
                model=model.value,
                tokens_used=tokens_used,
                processing_time_ms=processing_time_ms,
                cost_estimate=cost_estimate,
                metadata={
                    'request_id': response.get('ResponseMetadata', {}).get('RequestId'),
                    'http_status': response.get('ResponseMetadata', {}).get('HTTPStatusCode'),
                    **kwargs
                }
            )
            logger.log_api_call(
                service="bedrock",
                method=api_method,
                duration_ms=processing_time_ms,
                tokens_used=tokens_used,
                status="success",
                model=model.value,
                cost_estimate=cost_estimate
            )
            return bedrock_response
        except ClientError as e:
            print("***************", e)
            traceback.print_exc()
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            retryable_errors = {
                'ThrottlingException', 'ServiceUnavailable', 'InternalServerError',
                'RequestTimeout', 'TooManyRequestsException'
            }
            retryable = error_code in retryable_errors
            logger.error("Bedrock API error",
                        error_code=error_code,
                        error_message=error_message,
                        retryable=retryable,
                        model=model.value)
            raise BedrockClientError(
                f"Bedrock API error: {error_message}",
                error_code=error_code,
                retryable=retryable
            )
        except Exception as e:
            import traceback
            logger.error("Unexpected error in Bedrock client", error=str(e))
            print("*************** Unexpected error in Bedrock client:", e)
            traceback.print_exc()
            raise BedrockClientError(f"Unexpected error: {str(e)}", retryable=True)
    def _prepare_claude3_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request body for Claude 3 models (Messages API)."""
        messages = kwargs.get('messages')
        if not messages:
            # If no messages provided, wrap prompt as a single user message
            messages = [
                {"role": "user", "content": prompt}
            ]
        return {
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "anthropic_version": "bedrock-2023-05-31"
        }

    def _parse_claude3_response(self, response_body: Dict[str, Any]) -> Tuple[str, int]:
        """Parse Claude 3 Messages API response."""
        # Claude 3 returns a 'content' field in the last message
        messages = response_body.get('content', [])
        if isinstance(messages, list) and messages:
            content = messages[-1].get('text', '').strip()
        else:
            content = response_body.get('content', '').strip()
        # Token usage is not always provided; estimate if missing
        tokens_used = response_body.get('usage', {}).get('output_tokens')
        if tokens_used is None:
            tokens_used = len(content.split()) * 1.3
        return content, int(tokens_used)
    
    def _prepare_request_body(self, 
                            prompt: str, 
                            model: BedrockModel, 
                            **kwargs) -> Dict[str, Any]:
        """Prepare request body based on model type."""
        if model.value.startswith('anthropic.claude'):
            return self._prepare_claude_request(prompt, **kwargs)
        elif model.value.startswith('amazon.titan'):
            return self._prepare_titan_request(prompt, **kwargs)
        else:
            raise BedrockClientError(f"Unsupported model: {model.value}")
    
    def _prepare_claude_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request body for Claude models."""
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
        }
        
        if self.config.stop_sequences:
            body["stop_sequences"] = self.config.stop_sequences
            
        return body
    
    def _prepare_titan_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request body for Titan models."""
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "topP": kwargs.get('top_p', self.config.top_p),
                "stopSequences": self.config.stop_sequences
            }
        }
    
    def _parse_response(self, response_body: Dict[str, Any], model: BedrockModel) -> Tuple[str, int]:
        """Parse response body based on model type."""
        if model.value.startswith('anthropic.claude'):
            return self._parse_claude_response(response_body)
        elif model.value.startswith('amazon.titan'):
            return self._parse_titan_response(response_body)
        else:
            raise BedrockClientError(f"Unknown model response format: {model.value}")
    
    def _parse_claude_response(self, response_body: Dict[str, Any]) -> Tuple[str, int]:
        """Parse Claude model response."""
        content = response_body.get('completion', '').strip()
        
        # Estimate token usage (Claude doesn't always return exact counts)
        tokens_used = response_body.get('stop_reason') == 'max_tokens' and self.config.max_tokens or len(content.split()) * 1.3
        tokens_used = int(tokens_used)
        
        return content, tokens_used
    
    def _parse_titan_response(self, response_body: Dict[str, Any]) -> Tuple[str, int]:
        """Parse Titan model response."""
        results = response_body.get('results', [])
        if not results:
            raise BedrockClientError("No results in Titan response")
        
        content = results[0].get('outputText', '').strip()
        
        # Get token count from response
        token_count = response_body.get('inputTextTokenCount', 0) + \
                     results[0].get('tokenCount', len(content.split()))
        
        return content, int(token_count)
    
    def _estimate_cost(self, model: BedrockModel, tokens_used: int, input_length: int) -> float:
        """Estimate API call cost based on model and usage."""
        # Pricing estimates (per 1000 tokens) - these should be updated regularly
        pricing = {
            BedrockModel.CLAUDE_HAIKU: {"input": 0.00025, "output": 0.00125},
            BedrockModel.CLAUDE_SONNET: {"input": 0.003, "output": 0.015},
            BedrockModel.CLAUDE_OPUS: {"input": 0.015, "output": 0.075},
            BedrockModel.CLAUDE_V1: {"input": 0.008, "output": 0.024},
            BedrockModel.CLAUDE_V2: {"input": 0.008, "output": 0.024},
            BedrockModel.CLAUDE_INSTANT_V1: {"input": 0.0008, "output": 0.0024},
            BedrockModel.TITAN_TEXT_EXPRESS: {"input": 0.0013, "output": 0.0017},
            BedrockModel.TITAN_TEXT_LITE: {"input": 0.0003, "output": 0.0004},
        }
        
        if model not in pricing:
            return None
        
        # Estimate input tokens (rough approximation)
        input_tokens = len(input_length.split()) * 1.3 if isinstance(input_length, str) else input_length / 4
        output_tokens = max(0, tokens_used - input_tokens)
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        
        return round(input_cost + output_cost, 6)
    
    async def generate_text_async(self, 
                                prompt: str, 
                                model_id: Optional[BedrockModel] = None,
                                **kwargs) -> BedrockResponse:
        """Async version of generate_text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_text, prompt, model_id, **kwargs)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the Bedrock client."""
        try:
            # Simple test call
            test_response = self.generate_text(
                "Test connection", 
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "model": self.config.model_id.value,
                "region": self.config.region_name,
                "response_time_ms": test_response.processing_time_ms
            }
            
        except Exception as e:
            logger.error("Bedrock health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model_id.value,
                "region": self.config.region_name
            }


# Global client instance (lazy initialization)
_default_client = None


def get_bedrock_client(config: Optional[BedrockConfig] = None) -> BedrockClient:
    """Get or create the default Bedrock client instance.
    
    Args:
        config: Optional configuration for new client
        
    Returns:
        BedrockClient instance
    """
    global _default_client
    
    if _default_client is None or config is not None:
        _default_client = BedrockClient(config)
    
    return _default_client


# Convenience functions
def generate_text(prompt: str, **kwargs) -> BedrockResponse:
    """Generate text using the default Bedrock client."""
    client = get_bedrock_client()
    return client.generate_text(prompt, **kwargs)


async def generate_text_async(prompt: str, **kwargs) -> BedrockResponse:
    """Generate text asynchronously using the default Bedrock client."""
    client = get_bedrock_client()
    return await client.generate_text_async(prompt, **kwargs)