import pytest
import asyncio
import sys
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.novel_crew import NovelGenerationCrew
from agents.generator_agent import GeneratorAgent
from agents.tools.qa_tools import QATools
from qa_system.character_qa import CharacterQAAgent
from qa_system.structural_qa import StructuralQAAgent
from qa_system.style_qa import StyleQAAgent
from agents.technical_qa import TechnicalQAAgent


class TestAgentIntegration:
    """Test suite for agent integration functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.sample_project_id = "test_project_123"
        self.sample_prompt = "Write a dramatic scene where the protagonist faces a difficult choice"
        self.sample_settings = {
            "length_words": 800,
            "temperature": 0.7,
            "tone": "dramatic",
            "style": "literary",
            "genre": "fantasy",
            "max_revision_rounds": 1
        }
        
        self.sample_context = {
            "combined_text": "The protagonist is a young wizard named Aria who must choose between saving her mentor or protecting the village.",
            "semantic": {"chunks": ["chunk1", "chunk2"]},
            "characters": {
                "characters": [
                    {
                        "name": "Aria",
                        "description": "Young wizard apprentice",
                        "traits": ["brave", "impulsive", "loyal"],
                        "role": "protagonist"
                    }
                ],
                "relationships": [
                    {
                        "character": "Aria",
                        "relationship": "student of",
                        "related_to": "Master Eldrin",
                        "details": "Mentor-apprentice relationship"
                    }
                ]
            }
        }
        
        self.sample_generated_content = """# Chapter 12: The Choice

Aria stood at the crossroads, her staff trembling in her grip. The village burned behind her, while ahead, Master Eldrin lay trapped beneath the fallen stones of the ancient tower. She could only save one - her beloved mentor or the innocent villagers who depended on her magic to quell the flames.

"Choose wisely, young one," whispered the wind, carrying the screams of both her mentor and her people."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock database clients"""
        qdrant_client = MagicMock()
        neo4j_client = MagicMock()
        mongodb_client = MagicMock()
        
        return qdrant_client, neo4j_client, mongodb_client

    @pytest.fixture
    def mock_openai_api_key(self):
        """Mock OpenAI API key"""
        return "test-api-key-12345"

    @pytest.mark.asyncio
    async def test_novel_crew_initialization_success(self, mock_clients, mock_openai_api_key):
        """Test successful NovelGenerationCrew initialization"""
        qdrant_client, neo4j_client, mongodb_client = mock_clients
        
        # Test initialization
        crew = NovelGenerationCrew(
            qdrant_client=qdrant_client,
            neo4j_client=neo4j_client,
            mongodb_client=mongodb_client,
            openai_api_key=mock_openai_api_key
        )
        
        # Assertions
        assert crew.qdrant_client == qdrant_client
        assert crew.neo4j_client == neo4j_client
        assert crew.mongodb_client == mongodb_client
        assert crew.openai_api_key == mock_openai_api_key
        assert crew.context_retriever is not None
        assert crew.novel_tools is not None
        assert crew.qa_tools is not None
        assert crew.generator_agent is not None
        assert crew.structural_qa is not None
        assert crew.character_qa is not None
        assert crew.style_qa is not None
        assert crew.technical_qa is not None

    @pytest.mark.asyncio
    async def test_novel_crew_initialization_missing_api_key(self, mock_clients):
        """Test NovelGenerationCrew initialization with missing API key"""
        qdrant_client, neo4j_client, mongodb_client = mock_clients
        
        # Test initialization without API key should raise error
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            crew = NovelGenerationCrew(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                mongodb_client=mongodb_client,
                openai_api_key=None
            )

    @pytest.mark.asyncio
    async def test_generator_agent_content_generation(self, mock_openai_api_key):
        """Test GeneratorAgent content generation"""
        with patch('agents.generator_agent.AsyncOpenAI') as mock_openai:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = self.sample_generated_content
            mock_response.usage.total_tokens = 500
            mock_response.usage.prompt_tokens = 300
            mock_response.usage.completion_tokens = 200
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Test generation
            generator = GeneratorAgent(api_key=mock_openai_api_key)
            result = await generator.generate(
                prompt=self.sample_prompt,
                context=self.sample_context["combined_text"],
                length_words=800,
                temperature=0.7
            )
            
            # Assertions
            assert result == self.sample_generated_content
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0.7
            assert call_args[1]['max_tokens'] == 2000

    @pytest.mark.asyncio
    async def test_generator_agent_revision_with_feedback(self, mock_openai_api_key):
        """Test GeneratorAgent revision functionality"""
        feedback = "The dialogue needs more emotional depth and the pacing is too fast."
        revised_content = self.sample_generated_content.replace("trembling", "shaking with barely contained emotion")
        
        with patch('agents.generator_agent.AsyncOpenAI') as mock_openai:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = revised_content
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Test revision
            generator = GeneratorAgent(api_key=mock_openai_api_key)
            result = await generator.revise_with_feedback(
                original_content=self.sample_generated_content,
                feedback=feedback,
                context=self.sample_context["combined_text"],
                temperature=0.6
            )
            
            # Assertions
            assert result == revised_content
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0.6

    @pytest.mark.asyncio
    async def test_character_qa_agent_review(self, mock_openai_api_key):
        """Test CharacterQAAgent review functionality"""
        sample_qa_result = {
            "score": 85,
            "issues": [
                {
                    "loc": 150,
                    "type": "dialogue",
                    "issue": "Dialogue could be more distinctive to character voice",
                    "suggestion": "Add more personality-specific speech patterns",
                    "severity": "medium",
                    "character_name": "Aria"
                }
            ],
            "patches": [
                {
                    "loc": 150,
                    "original": '"Choose wisely, young one," whispered the wind',
                    "replacement": '"Choose wisely, little star," whispered the wind using her mentor\'s pet name'
                }
            ]
        }
        
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_openai:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(sample_qa_result)
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Test character QA
            char_qa = CharacterQAAgent(api_key=mock_openai_api_key)
            result = await char_qa.review(
                text=self.sample_generated_content,
                character_context=self.sample_context["characters"]
            )
            
            # Assertions
            assert result["score"] == 85
            assert len(result["issues"]) == 1
            assert result["issues"][0]["character_name"] == "Aria"
            assert len(result["patches"]) == 1
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_qa_tools_validation(self):
        """Test QATools validation functionality"""
        qa_tools = QATools()
        
        # Test with valid QA result
        valid_qa_result = {
            "score": 90,
            "issues": [
                {
                    "loc": 100,
                    "type": "style",
                    "issue": "Word repetition",
                    "suggestion": "Use synonyms",
                    "severity": "low"
                }
            ],
            "patches": []
        }
        
        validated = qa_tools.validate_qa_result(valid_qa_result, "style")
        
        # Assertions
        assert validated["score"] == 90
        assert len(validated["issues"]) == 1
        assert validated["agent_type"] == "style"
        assert "validated_at" in validated

    @pytest.mark.asyncio
    async def test_qa_tools_aggregation(self):
        """Test QATools result aggregation"""
        qa_tools = QATools()
        
        # Mock multiple QA results
        qa_results = [
            {
                "score": 85,
                "agent_type": "technical",
                "issues": [{"type": "grammar", "severity": "low"}],
                "patches": []
            },
            {
                "score": 80,
                "agent_type": "character",
                "issues": [{"type": "dialogue", "severity": "medium"}],
                "patches": []
            },
            {
                "score": 75,
                "agent_type": "structural",
                "issues": [{"type": "pacing", "severity": "high"}],
                "patches": []
            }
        ]
        
        aggregated = qa_tools.aggregate_qa_results(qa_results)
        
        # Assertions
        assert aggregated["total_issues"] == 3
        assert aggregated["overall_score"] > 0
        assert "technical" in aggregated["agent_scores"]
        assert "character" in aggregated["agent_scores"]
        assert "structural" in aggregated["agent_scores"]
        assert aggregated["requires_revision"] is True  # Due to high severity issue
        assert len(aggregated["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_parallel_qa_execution(self, mock_openai_api_key):
        """Test parallel QA agent execution"""
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_char_openai, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct_openai, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style_openai, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech_openai:
            
            # Setup mock responses for each agent
            mock_responses = [
                {"score": 85, "issues": [], "patches": []},
                {"score": 80, "issues": [{"type": "pacing", "severity": "low"}], "patches": []},
                {"score": 90, "issues": [], "patches": []},
                {"score": 88, "issues": [], "patches": []}
            ]
            
            for mock_openai, mock_response in zip(
                [mock_char_openai, mock_struct_openai, mock_style_openai, mock_tech_openai],
                mock_responses
            ):
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = json.dumps(mock_response)
                mock_client.chat.completions.create.return_value = mock_resp
                mock_openai.return_value = mock_client
            
            # Create QA agents
            qa_tools = QATools()
            qa_agents = [
                CharacterQAAgent(api_key=mock_openai_api_key),
                StructuralQAAgent(api_key=mock_openai_api_key),
                StyleQAAgent(api_key=mock_openai_api_key),
                TechnicalQAAgent(api_key=mock_openai_api_key)
            ]
            
            # Test parallel execution
            results = await qa_tools.run_parallel_qa(
                text=self.sample_generated_content,
                qa_agents=qa_agents,
                context=self.sample_context,
                timeout=30
            )
            
            # Assertions
            assert len(results) == 4
            for result in results:
                assert "score" in result
                assert isinstance(result["score"], (int, float))

    @pytest.mark.asyncio
    async def test_full_workflow_execution_success(self, mock_clients, mock_openai_api_key):
        """Test complete novel generation workflow execution"""
        qdrant_client, neo4j_client, mongodb_client = mock_clients
        
        with patch('agents.tools.context_tools.ContextRetriever') as mock_context_retriever, \
             patch('agents.generator_agent.AsyncOpenAI') as mock_gen_openai, \
             patch('qa_system.character_qa.AsyncOpenAI') as mock_char_openai, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct_openai, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style_openai, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech_openai:
            
            # Setup context retriever mock
            mock_context_instance = AsyncMock()
            mock_context_instance.retrieve_comprehensive_context.return_value = self.sample_context
            mock_context_instance.get_plot_context.return_value = {"main_plot": "Hero's journey"}
            mock_context_retriever.return_value = mock_context_instance
            
            # Setup generator mock
            mock_gen_client = AsyncMock()
            mock_gen_response = MagicMock()
            mock_gen_response.choices = [MagicMock()]
            mock_gen_response.choices[0].message.content = self.sample_generated_content
            mock_gen_response.usage.total_tokens = 500
            mock_gen_response.usage.prompt_tokens = 300
            mock_gen_response.usage.completion_tokens = 200
            mock_gen_client.chat.completions.create.return_value = mock_gen_response
            mock_gen_openai.return_value = mock_gen_client
            
            # Setup QA mocks
            qa_mock_responses = [
                {"score": 85, "issues": [], "patches": []},  # character
                {"score": 80, "issues": [], "patches": []},  # structural  
                {"score": 90, "issues": [], "patches": []},  # style
                {"score": 88, "issues": [], "patches": []}   # technical
            ]
            
            for mock_openai, mock_response in zip(
                [mock_char_openai, mock_struct_openai, mock_style_openai, mock_tech_openai],
                qa_mock_responses
            ):
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = json.dumps(mock_response)
                mock_client.chat.completions.create.return_value = mock_resp
                mock_openai.return_value = mock_client
            
            # Test workflow execution
            crew = NovelGenerationCrew(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                mongodb_client=mongodb_client,
                openai_api_key=mock_openai_api_key
            )
            
            result = await crew.execute_generation_workflow(
                project_id=self.sample_project_id,
                prompt=self.sample_prompt,
                settings=self.sample_settings
            )
            
            # Assertions
            assert "content" in result
            assert result["content"] == self.sample_generated_content
            assert result["word_count"] > 0
            assert result["character_count"] > 0
            assert "qa_analysis" in result
            assert "context_used" in result
            assert "generation_metadata" in result
            assert result["generation_metadata"]["project_id"] == self.sample_project_id
            assert result["revision_count"] >= 0
            
            # Verify context retrieval was called
            mock_context_instance.retrieve_comprehensive_context.assert_called_once()
            
            # Verify generation was called
            mock_gen_client.chat.completions.create.assert_called()

    @pytest.mark.asyncio
    async def test_workflow_execution_with_revision(self, mock_clients, mock_openai_api_key):
        """Test workflow execution that triggers revision"""
        qdrant_client, neo4j_client, mongodb_client = mock_clients
        
        with patch('agents.tools.context_tools.ContextRetriever') as mock_context_retriever, \
             patch('agents.generator_agent.AsyncOpenAI') as mock_gen_openai, \
             patch('qa_system.character_qa.AsyncOpenAI') as mock_char_openai, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct_openai, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style_openai, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech_openai:
            
            # Setup context retriever mock
            mock_context_instance = AsyncMock()
            mock_context_instance.retrieve_comprehensive_context.return_value = self.sample_context
            mock_context_instance.get_plot_context.return_value = {"main_plot": "Hero's journey"}
            mock_context_retriever.return_value = mock_context_instance
            
            # Setup generator mock to return both original and revised content
            mock_gen_client = AsyncMock()
            original_response = MagicMock()
            original_response.choices = [MagicMock()]
            original_response.choices[0].message.content = self.sample_generated_content
            original_response.usage.total_tokens = 500
            original_response.usage.prompt_tokens = 300
            original_response.usage.completion_tokens = 200
            
            revised_content = self.sample_generated_content.replace("trembling", "shaking with emotion")
            revised_response = MagicMock()
            revised_response.choices = [MagicMock()]
            revised_response.choices[0].message.content = revised_content
            
            mock_gen_client.chat.completions.create.side_effect = [original_response, revised_response]
            mock_gen_openai.return_value = mock_gen_client
            
            # Setup QA mocks with issues that trigger revision
            qa_mock_responses = [
                {
                    "score": 70, 
                    "issues": [
                        {
                            "type": "dialogue",
                            "issue": "Character voice inconsistent",
                            "suggestion": "Use more authentic dialogue",
                            "severity": "high",
                            "loc": 100
                        }
                    ], 
                    "patches": []
                },  # character - low score triggers revision
                {"score": 75, "issues": [], "patches": []},  # structural  
                {"score": 80, "issues": [], "patches": []},  # style
                {"score": 85, "issues": [], "patches": []}   # technical
            ]
            
            for mock_openai, mock_response in zip(
                [mock_char_openai, mock_struct_openai, mock_style_openai, mock_tech_openai],
                qa_mock_responses
            ):
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = json.dumps(mock_response)
                mock_client.chat.completions.create.return_value = mock_resp
                mock_openai.return_value = mock_client
            
            # Test workflow execution
            crew = NovelGenerationCrew(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                mongodb_client=mongodb_client,
                openai_api_key=mock_openai_api_key
            )
            
            result = await crew.execute_generation_workflow(
                project_id=self.sample_project_id,
                prompt=self.sample_prompt,
                settings=self.sample_settings
            )
            
            # Assertions
            assert result["content"] == revised_content  # Should use revised content
            assert result["revision_count"] == 1
            assert result["qa_analysis"]["requires_revision"] is True
            
            # Verify both generation calls were made (original + revision)
            assert mock_gen_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_workflow_execution_error_handling(self, mock_clients, mock_openai_api_key):
        """Test workflow error handling"""
        qdrant_client, neo4j_client, mongodb_client = mock_clients
        
        with patch('agents.tools.context_tools.ContextRetriever') as mock_context_retriever:
            # Setup context retriever to raise exception
            mock_context_instance = AsyncMock()
            mock_context_instance.retrieve_comprehensive_context.side_effect = Exception("Context retrieval failed")
            mock_context_retriever.return_value = mock_context_instance
            
            # Test workflow execution
            crew = NovelGenerationCrew(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client,
                mongodb_client=mongodb_client,
                openai_api_key=mock_openai_api_key
            )
            
            result = await crew.execute_generation_workflow(
                project_id=self.sample_project_id,
                prompt=self.sample_prompt,
                settings=self.sample_settings
            )
            
            # Assertions for error result
            assert result["content"] == ""
            assert "error" in result
            assert "Context retrieval failed" in result["error"]
            assert result["word_count"] == 0
            assert result["character_count"] == 0
            assert result["revision_count"] == 0
            assert result["qa_analysis"]["overall_score"] == 0


if __name__ == "__main__":
    pytest.main([__file__])