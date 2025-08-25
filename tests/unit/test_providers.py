"""
Comprehensive unit tests for agent.providers module.
Tests provider configuration, model initialization, and validation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai

from agent.providers import (
    get_llm_model,
    get_embedding_client,
    get_embedding_model,
    get_ingestion_model,
    get_llm_provider,
    get_embedding_provider,
    validate_configuration,
    get_model_info
)


class TestLLMModelConfiguration:
    """Test LLM model configuration functionality."""
    
    def test_get_llm_model_default_config(self):
        """Test getting LLM model with default configuration."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-key'
        }):
            model = get_llm_model()
            
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'gpt-4-turbo-preview'
            assert isinstance(model._provider, OpenAIProvider)
    
    def test_get_llm_model_with_override(self):
        """Test getting LLM model with model choice override."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-key'
        }):
            model = get_llm_model(model_choice='gpt-3.5-turbo')
            
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'gpt-3.5-turbo'
    
    def test_get_llm_model_fallback_values(self):
        """Test LLM model with fallback environment values."""
        with patch.dict(os.environ, {}, clear=True):
            model = get_llm_model()
            
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'gpt-4-turbo-preview'  # Default fallback
            assert model._provider.base_url.rstrip('/') == 'https://api.openai.com/v1'
            # Note: api_key is not publicly accessible on OpenAIProvider
    
    def test_get_llm_model_custom_base_url(self):
        """Test LLM model with custom base URL."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'llama2',
            'LLM_BASE_URL': 'http://localhost:11434/v1',
            'LLM_API_KEY': 'ollama'
        }):
            model = get_llm_model()
            
            assert model.model_name == 'llama2'
            assert model._provider.base_url.rstrip('/') == 'http://localhost:11434/v1'
            # Note: api_key is not publicly accessible on OpenAIProvider


class TestEmbeddingConfiguration:
    """Test embedding client and model configuration."""
    
    def test_get_embedding_client_default(self):
        """Test getting embedding client with default configuration."""
        with patch.dict(os.environ, {
            'EMBEDDING_BASE_URL': 'https://api.openai.com/v1',
            'EMBEDDING_API_KEY': 'test-key'
        }):
            client = get_embedding_client()
            
            assert isinstance(client, openai.AsyncOpenAI)
            assert str(client.base_url).rstrip('/') == 'https://api.openai.com/v1'
            assert client.api_key == 'test-key'
    
    def test_get_embedding_client_fallback_values(self):
        """Test embedding client with fallback values."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_embedding_client()
            
            assert isinstance(client, openai.AsyncOpenAI)
            assert str(client.base_url).rstrip('/') == 'https://api.openai.com/v1'
            assert client.api_key == 'ollama'
    
    def test_get_embedding_client_custom_config(self):
        """Test embedding client with custom configuration."""
        with patch.dict(os.environ, {
            'EMBEDDING_BASE_URL': 'http://localhost:11434/v1',
            'EMBEDDING_API_KEY': 'custom-key'
        }):
            client = get_embedding_client()
            
            assert str(client.base_url).rstrip('/') == 'http://localhost:11434/v1'
            assert client.api_key == 'custom-key'
    
    def test_get_embedding_model_default(self):
        """Test getting embedding model name with default."""
        with patch.dict(os.environ, {}, clear=True):
            model = get_embedding_model()
            assert model == 'text-embedding-3-small'
    
    def test_get_embedding_model_custom(self):
        """Test getting embedding model name with custom value."""
        with patch.dict(os.environ, {
            'EMBEDDING_MODEL': 'text-embedding-ada-002'
        }):
            model = get_embedding_model()
            assert model == 'text-embedding-ada-002'


class TestIngestionModel:
    """Test ingestion-specific model configuration."""
    
    def test_get_ingestion_model_no_specific_config(self):
        """Test ingestion model falls back to main model when no specific config."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-key'
        }, clear=True):
            model = get_ingestion_model()
            
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'gpt-4-turbo-preview'
    
    def test_get_ingestion_model_with_specific_config(self):
        """Test ingestion model with specific configuration."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'INGESTION_LLM_CHOICE': 'gpt-3.5-turbo',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-key'
        }):
            model = get_ingestion_model()
            
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'gpt-3.5-turbo'
    
    def test_get_ingestion_model_empty_specific_config(self):
        """Test ingestion model with empty specific config."""
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'INGESTION_LLM_CHOICE': '',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-key'
        }):
            model = get_ingestion_model()
            
            # Should fall back to main model when INGESTION_LLM_CHOICE is empty
            assert model.model_name == 'gpt-4-turbo-preview'


class TestProviderInfo:
    """Test provider information functions."""
    
    def test_get_llm_provider_default(self):
        """Test getting LLM provider with default value."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_llm_provider()
            assert provider == 'openai'
    
    def test_get_llm_provider_custom(self):
        """Test getting LLM provider with custom value."""
        with patch.dict(os.environ, {'LLM_PROVIDER': 'anthropic'}):
            provider = get_llm_provider()
            assert provider == 'anthropic'
    
    def test_get_embedding_provider_default(self):
        """Test getting embedding provider with default value."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_embedding_provider()
            assert provider == 'openai'
    
    def test_get_embedding_provider_custom(self):
        """Test getting embedding provider with custom value."""
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'cohere'}):
            provider = get_embedding_provider()
            assert provider == 'cohere'


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_configuration_all_present(self):
        """Test validation when all required variables are present."""
        with patch.dict(os.environ, {
            'LLM_API_KEY': 'test-llm-key',
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'EMBEDDING_API_KEY': 'test-embedding-key',
            'EMBEDDING_MODEL': 'text-embedding-3-small'
        }):
            assert validate_configuration() is True
    
    def test_validate_configuration_missing_single_var(self):
        """Test validation when single required variable is missing."""
        with patch.dict(os.environ, {
            'LLM_API_KEY': 'test-llm-key',
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'EMBEDDING_API_KEY': 'test-embedding-key'
            # Missing EMBEDDING_MODEL
        }, clear=True):
            with patch('builtins.print') as mock_print:
                result = validate_configuration()
                assert result is False
                mock_print.assert_called_once()
                assert 'EMBEDDING_MODEL' in mock_print.call_args[0][0]
    
    def test_validate_configuration_missing_multiple_vars(self):
        """Test validation when multiple required variables are missing."""
        with patch.dict(os.environ, {
            'LLM_API_KEY': 'test-llm-key'
            # Missing LLM_CHOICE, EMBEDDING_API_KEY, EMBEDDING_MODEL
        }, clear=True):
            with patch('builtins.print') as mock_print:
                result = validate_configuration()
                assert result is False
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                assert 'LLM_CHOICE' in error_message
                assert 'EMBEDDING_API_KEY' in error_message
                assert 'EMBEDDING_MODEL' in error_message
    
    def test_validate_configuration_all_missing(self):
        """Test validation when all required variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('builtins.print') as mock_print:
                result = validate_configuration()
                assert result is False
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                required_vars = ['LLM_API_KEY', 'LLM_CHOICE', 'EMBEDDING_API_KEY', 'EMBEDDING_MODEL']
                for var in required_vars:
                    assert var in error_message
    
    def test_validate_configuration_empty_values(self):
        """Test validation treats empty strings as missing."""
        with patch.dict(os.environ, {
            'LLM_API_KEY': '',
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'EMBEDDING_API_KEY': 'test-embedding-key',
            'EMBEDDING_MODEL': ''
        }):
            with patch('builtins.print') as mock_print:
                result = validate_configuration()
                assert result is False
                error_message = mock_print.call_args[0][0]
                assert 'LLM_API_KEY' in error_message
                assert 'EMBEDDING_MODEL' in error_message


class TestModelInfo:
    """Test model information functionality."""
    
    def test_get_model_info_complete(self):
        """Test getting model info with complete configuration."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'EMBEDDING_PROVIDER': 'openai',
            'EMBEDDING_MODEL': 'text-embedding-3-small',
            'EMBEDDING_BASE_URL': 'https://api.openai.com/v1',
            'INGESTION_LLM_CHOICE': 'gpt-3.5-turbo'
        }):
            info = get_model_info()
            
            expected_info = {
                "llm_provider": "openai",
                "llm_model": "gpt-4-turbo-preview",
                "llm_base_url": "https://api.openai.com/v1",
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_base_url": "https://api.openai.com/v1",
                "ingestion_model": "gpt-3.5-turbo",
            }
            
            assert info == expected_info
    
    def test_get_model_info_minimal(self):
        """Test getting model info with minimal configuration."""
        with patch.dict(os.environ, {}, clear=True):
            info = get_model_info()
            
            expected_info = {
                "llm_provider": "openai",
                "llm_model": None,
                "llm_base_url": None,
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_base_url": None,
                "ingestion_model": "same as main",
            }
            
            assert info == expected_info
    
    def test_get_model_info_no_ingestion_model(self):
        """Test getting model info when no specific ingestion model is set."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic',
            'LLM_CHOICE': 'claude-3-opus',
            'EMBEDDING_PROVIDER': 'cohere',
            'EMBEDDING_MODEL': 'embed-english-v3.0'
        }, clear=True):
            info = get_model_info()
            
            assert info["llm_provider"] == "anthropic"
            assert info["llm_model"] == "claude-3-opus"
            assert info["embedding_provider"] == "cohere"
            assert info["embedding_model"] == "embed-english-v3.0"
            assert info["ingestion_model"] == "same as main"


class TestErrorHandling:
    """Test error handling in provider functions."""
    
    def test_get_llm_model_with_invalid_provider_config(self):
        """Test behavior when provider configuration might be invalid."""
        # This mainly tests that the function doesn't crash with unusual inputs
        with patch.dict(os.environ, {
            'LLM_CHOICE': 'invalid-model-name',
            'LLM_BASE_URL': 'not-a-valid-url',
            'LLM_API_KEY': 'potentially-invalid-key'
        }):
            # Should still create a model object (validation happens at usage time)
            model = get_llm_model()
            assert isinstance(model, OpenAIModel)
            assert model.model_name == 'invalid-model-name'
    
    def test_get_embedding_client_with_none_values(self):
        """Test embedding client creation with None-like values."""
        with patch.dict(os.environ, {
            'EMBEDDING_BASE_URL': '',
            'EMBEDDING_API_KEY': ''
        }):
            # When set to empty string, it gets empty string (not fallback)
            client = get_embedding_client()
            assert isinstance(client, openai.AsyncOpenAI)
            assert str(client.base_url) == ''
            assert client.api_key == ''


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_mixed_provider_configuration(self):
        """Test configuration with mixed providers."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic',
            'LLM_CHOICE': 'claude-3-opus',
            'LLM_BASE_URL': 'https://api.anthropic.com/v1',
            'LLM_API_KEY': 'anthropic-key',
            'EMBEDDING_PROVIDER': 'openai',
            'EMBEDDING_MODEL': 'text-embedding-3-small',
            'EMBEDDING_BASE_URL': 'https://api.openai.com/v1',
            'EMBEDDING_API_KEY': 'openai-key'
        }):
            # This should work fine - different providers for LLM and embeddings
            llm_model = get_llm_model()
            embedding_client = get_embedding_client()
            
            assert isinstance(llm_model, OpenAIModel)  # Uses OpenAI-compatible interface
            assert llm_model.model_name == 'claude-3-opus'
            assert llm_model._provider.base_url.rstrip('/') == 'https://api.anthropic.com/v1'
            
            assert isinstance(embedding_client, openai.AsyncOpenAI)
            assert str(embedding_client.base_url).rstrip('/') == 'https://api.openai.com/v1'
            assert embedding_client.api_key == 'openai-key'
    
    def test_local_development_configuration(self):
        """Test typical local development configuration."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'ollama',
            'LLM_CHOICE': 'llama2',
            'LLM_BASE_URL': 'http://localhost:11434/v1',
            'LLM_API_KEY': 'ollama',
            'EMBEDDING_PROVIDER': 'ollama',
            'EMBEDDING_MODEL': 'nomic-embed-text',
            'EMBEDDING_BASE_URL': 'http://localhost:11434/v1',
            'EMBEDDING_API_KEY': 'ollama'
        }):
            llm_model = get_llm_model()
            embedding_client = get_embedding_client()
            
            assert llm_model.model_name == 'llama2'
            assert llm_model._provider.base_url.rstrip('/') == 'http://localhost:11434/v1'
            # Note: api_key is not publicly accessible on OpenAIProvider
            
            assert str(embedding_client.base_url).rstrip('/') == 'http://localhost:11434/v1'
            assert embedding_client.api_key == 'ollama'
            
            assert get_embedding_model() == 'nomic-embed-text'
            assert validate_configuration() is True
    
    def test_production_configuration(self):
        """Test typical production configuration."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'LLM_CHOICE': 'gpt-4-turbo-preview',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'sk-...',
            'EMBEDDING_PROVIDER': 'openai',
            'EMBEDDING_MODEL': 'text-embedding-3-small',
            'EMBEDDING_BASE_URL': 'https://api.openai.com/v1',
            'EMBEDDING_API_KEY': 'sk-...',
            'INGESTION_LLM_CHOICE': 'gpt-3.5-turbo'
        }):
            assert validate_configuration() is True
            
            llm_model = get_llm_model()
            ingestion_model = get_ingestion_model()
            
            assert llm_model.model_name == 'gpt-4-turbo-preview'
            assert ingestion_model.model_name == 'gpt-3.5-turbo'
            
            info = get_model_info()
            assert info["llm_model"] == 'gpt-4-turbo-preview'
            assert info["ingestion_model"] == 'gpt-3.5-turbo'