"""
Comprehensive Integration Tests for Agent and Module Correlation
Tests the interaction between different system components and their correlation.
"""

import pytest
import asyncio
import sys
import os
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.novel_crew import NovelGenerationCrew
from agents.generator_agent import GeneratorAgent
from agents.tools.qa_tools import QATools
from agents.tools.context_tools import ContextRetriever
from agents.tools.novel_tools import NovelTools
from qa_system.character_qa import CharacterQAAgent
from qa_system.structural_qa import StructuralQAAgent
from qa_system.style_qa import StyleQAAgent
from agents.technical_qa import TechnicalQAAgent
from embeddings.embedder import TextEmbedder
from embeddings.chunker import TextChunker


class TestIntegrationComprehensive:
    """Comprehensive test suite for system integration and module correlation"""

    def setup_method(self):
        """Setup comprehensive test fixtures"""
        self.test_project_id = "integration_test_project"
        self.test_prompt = "Write a compelling dialogue between two characters discussing their past"
        self.test_settings = {
            "length_words": 1000,
            "temperature": 0.8,
            "tone": "conversational",
            "style": "literary",
            "genre": "drama",
            "max_revision_rounds": 2
        }
        
        # Mock context data that flows through different modules
        self.mock_context = {
            "combined_text": "Sarah and Marcus had been best friends since childhood, but their recent argument created a rift between them.",
            "semantic": {
                "chunks": [
                    "Sarah and Marcus childhood friendship",
                    "Recent argument created rift",
                    "Long history together"
                ]
            },
            "characters": {
                "characters": [
                    {
                        "name": "Sarah",
                        "description": "Thoughtful writer, tends to overthink",
                        "traits": ["analytical", "sensitive", "loyal"],
                        "role": "protagonist"
                    },
                    {
                        "name": "Marcus",
                        "description": "Pragmatic engineer, direct communicator", 
                        "traits": ["logical", "honest", "stubborn"],
                        "role": "deuteragonist"
                    }
                ],
                "relationships": [
                    {
                        "character": "Sarah",
                        "relationship": "childhood friend of",
                        "related_to": "Marcus",
                        "details": "Best friends since age 8, recent tension"
                    }
                ]
            }
        }
        
        self.mock_generated_content = '''# Chapter 5: The Conversation

"We need to talk about what happened," Sarah said, settling into the chair across from Marcus at their usual coffee shop.

Marcus looked up from his laptop, his jaw tightening. "I thought we'd said everything that needed saying."

"Did we? Because I keep replaying our argument, and I don't think either of us was really listening." Sarah wrapped her hands around her mug, seeking warmth and comfort.

"I was listening. You said I was being stubborn and unrealistic about the job offer." His voice carried a defensive edge.

"And you said I was being overly cautious and holding you back." Sarah met his eyes. "But what were we really fighting about, Marcus? Because it wasn't about the job."

Marcus closed his laptop slowly, finally giving her his full attention. "Then what was it about?"

"Us. Our friendship. The fact that we're both scared of change but too proud to admit it."'''

    @pytest.fixture
    def mock_database_clients(self):
        """Create comprehensive mock database clients with realistic methods"""
        # Mock Qdrant client
        qdrant_client = MagicMock()
        qdrant_client.search = AsyncMock(return_value=[
            MagicMock(payload={"text": "relevant context 1"}, score=0.9),
            MagicMock(payload={"text": "relevant context 2"}, score=0.8)
        ])
        qdrant_client.upsert = AsyncMock()
        
        # Mock Neo4j client  
        neo4j_client = MagicMock()
        neo4j_client.execute_query = AsyncMock(return_value=[
            {"character": "Sarah", "relationship": "friend", "target": "Marcus"}
        ])
        neo4j_client.close = AsyncMock()
        
        # Mock MongoDB client
        mongodb_client = MagicMock()
        mongodb_client.find_one = AsyncMock(return_value={
            "_id": "test_doc",
            "content": "sample content",
            "metadata": {"type": "character_note"}
        })
        mongodb_client.insert_one = AsyncMock()
        
        return qdrant_client, neo4j_client, mongodb_client

    @pytest.fixture
    def mock_openai_api_key(self):
        """Mock OpenAI API key"""
        return "test-integration-api-key-12345"

    @pytest.mark.asyncio
    async def test_end_to_end_novel_generation_workflow(self, mock_database_clients, mock_openai_api_key):
        """Test complete end-to-end workflow from prompt to final content"""
        qdrant_client, neo4j_client, mongodb_client = mock_database_clients
        
        # Mock all OpenAI calls across different agents
        with patch('agents.generator_agent.AsyncOpenAI') as mock_gen_openai, \
             patch('qa_system.character_qa.AsyncOpenAI') as mock_char_openai, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct_openai, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style_openai, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech_openai:
            
            # Setup generator response
            gen_response = MagicMock()
            gen_response.choices = [MagicMock()]
            gen_response.choices[0].message.content = self.mock_generated_content
            gen_response.usage.total_tokens = 800
            
            gen_client = AsyncMock()
            gen_client.chat.completions.create.return_value = gen_response
            mock_gen_openai.return_value = gen_client
            
            # Setup QA responses
            qa_responses = [
                {"score": 88, "issues": [{"type": "dialogue", "severity": "low"}], "patches": []},
                {"score": 85, "issues": [{"type": "pacing", "severity": "medium"}], "patches": []},
                {"score": 90, "issues": [], "patches": []},
                {"score": 87, "issues": [{"type": "grammar", "severity": "low"}], "patches": []}
            ]
            
            for mock_openai, qa_response in zip(
                [mock_char_openai, mock_struct_openai, mock_style_openai, mock_tech_openai],
                qa_responses
            ):
                qa_client = AsyncMock()
                qa_resp = MagicMock()
                qa_resp.choices = [MagicMock()]
                qa_resp.choices[0].message.content = json.dumps(qa_response)
                qa_client.chat.completions.create.return_value = qa_resp
                mock_openai.return_value = qa_client
            
            # Initialize crew and test complete workflow
            crew = NovelGenerationCrew(
                qdrant_client=qdrant_client,
                neo4j_client=neo4j_client, 
                mongodb_client=mongodb_client,
                openai_api_key=mock_openai_api_key
            )
            
            # Execute full generation workflow
            result = await crew.execute_generation_workflow(
                project_id=self.test_project_id,
                prompt=self.test_prompt,
                settings=self.test_settings
            )
            
            # Verify end-to-end data flow
            assert result is not None
            assert "content" in result
            assert "qa_analysis" in result
            assert "generation_metadata" in result
            assert result["qa_analysis"]["overall_score"] > 0
            assert "word_count" in result
            assert result["word_count"] > 0
            
            # Verify all agents were called in proper sequence
            gen_client.chat.completions.create.assert_called()
            assert all(client.chat.completions.create.called for client in [
                qa_client for _, qa_client in [(mock_char_openai, qa_client) 
                for mock_openai in [mock_char_openai, mock_struct_openai, mock_style_openai, mock_tech_openai]
                for qa_client in [mock_openai.return_value]]
            ])

    @pytest.mark.asyncio 
    async def test_agent_coordination_and_task_handoff(self, mock_database_clients, mock_openai_api_key):
        """Test how agents coordinate and hand off tasks between each other"""
        qdrant_client, neo4j_client, mongodb_client = mock_database_clients
        
        with patch('agents.generator_agent.AsyncOpenAI') as mock_gen_openai, \
             patch('qa_system.character_qa.AsyncOpenAI') as mock_char_openai:
            
            # Track call order and data handoff
            call_order = []
            
            def track_generator_call(*args, **kwargs):
                call_order.append(("generator", kwargs.get("messages", [{}])[-1].get("content", "")))
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = self.mock_generated_content
                return response
            
            def track_qa_call(*args, **kwargs):
                call_order.append(("character_qa", kwargs.get("messages", [{}])[-1].get("content", "")))
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = json.dumps({
                    "score": 85, 
                    "issues": [], 
                    "patches": [],
                    "character_consistency": True
                })
                return response
            
            gen_client = AsyncMock()
            gen_client.chat.completions.create.side_effect = track_generator_call
            mock_gen_openai.return_value = gen_client
            
            qa_client = AsyncMock()  
            qa_client.chat.completions.create.side_effect = track_qa_call
            mock_char_openai.return_value = qa_client
            
            # Test agent coordination
            generator = GeneratorAgent(api_key=mock_openai_api_key)
            char_qa = CharacterQAAgent(api_key=mock_openai_api_key)
            
            # Step 1: Generate content
            generated_content = await generator.generate(
                prompt=self.test_prompt,
                context=self.mock_context["combined_text"],
                length_words=1000
            )
            
            # Step 2: QA reviews generated content
            qa_result = await char_qa.review(
                text=generated_content,
                character_context=self.mock_context["characters"]
            )
            
            # Verify proper handoff sequence
            assert len(call_order) == 2
            assert call_order[0][0] == "generator"
            assert call_order[1][0] == "character_qa"
            assert self.test_prompt.lower() in call_order[0][1].lower()
            assert "sarah" in call_order[1][1].lower() or "marcus" in call_order[1][1].lower()
            assert qa_result["character_consistency"] is True

    @pytest.mark.asyncio
    async def test_cross_module_data_flow_validation(self, mock_database_clients, mock_openai_api_key):
        """Test data flow and transformation across different system modules"""
        qdrant_client, neo4j_client, mongodb_client = mock_database_clients
        
        # Track data transformations across modules
        data_flow = {}
        
        # Mock context retriever with data tracking
        with patch.object(ContextRetriever, 'get_context') as mock_get_context:
            mock_get_context.return_value = {
                "semantic_results": ["context1", "context2"],
                "character_data": self.mock_context["characters"],
                "retrieved_at": time.time()
            }
            data_flow["context_retrieval"] = mock_get_context.return_value
            
            context_retriever = ContextRetriever(qdrant_client, neo4j_client)
            retrieved_context = await context_retriever.get_context(
                query=self.test_prompt,
                project_id=self.test_project_id
            )
            
            # Verify context structure and content
            assert "semantic_results" in retrieved_context
            assert "character_data" in retrieved_context
            assert retrieved_context["character_data"]["characters"][0]["name"] == "Sarah"
            
        # Test embedding module data flow
        with patch('embeddings.embedder.AsyncOpenAI') as mock_embed_openai:
            mock_embed_client = AsyncMock()
            mock_embed_response = MagicMock()
            mock_embed_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_embed_client.embeddings.create.return_value = mock_embed_response
            mock_embed_openai.return_value = mock_embed_client
            
            embedder = TextEmbedder(api_key=mock_openai_api_key)
            embeddings = await embedder.embed_texts([self.mock_context["combined_text"]])
            
            data_flow["embeddings"] = embeddings
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 3  # [0.1, 0.2, 0.3]
            
        # Test chunker module
        chunker = TextChunker()
        chunks = chunker.chunk_text(self.mock_generated_content, chunk_size=200)
        data_flow["chunks"] = chunks
        
        # Verify data consistency across modules
        assert len(data_flow) == 3
        assert all(key in data_flow for key in ["context_retrieval", "embeddings", "chunks"])
        assert len(data_flow["chunks"]) > 0

    @pytest.mark.asyncio
    async def test_module_dependency_injection_failures(self, mock_openai_api_key):
        """Test system behavior when module dependencies fail to inject properly"""
        
        # Test missing database clients
        with pytest.raises(Exception):
            crew = NovelGenerationCrew(
                qdrant_client=None,  # Missing required client
                neo4j_client=None,   # Missing required client
                mongodb_client=None, # Missing required client
                openai_api_key=mock_openai_api_key
            )
            # Attempt to use context retriever without clients
            await crew.context_retriever.get_context("test query", "test_project")
        
        # Test invalid API key scenarios - NovelGenerationCrew allows None API key
        # but GeneratorAgent should fail
        crew = NovelGenerationCrew(
            qdrant_client=MagicMock(),
            neo4j_client=MagicMock(), 
            mongodb_client=MagicMock(),
            openai_api_key=None  # Missing API key
        )
        
        # The actual error should occur when trying to use the generator
        with pytest.raises(AttributeError):  # or whatever error the actual implementation throws
            await crew.execute_generation_workflow(
                project_id="test",
                prompt="test prompt",
                settings={"length_words": 100}
            )
        
        # Test partial dependency failures
        mock_clients = (MagicMock(), MagicMock(), MagicMock())
        crew = NovelGenerationCrew(*mock_clients, openai_api_key=mock_openai_api_key)
        
        # Simulate database connection failure during runtime
        crew.qdrant_client.search.side_effect = ConnectionError("Database connection failed")
        
        with pytest.raises(ConnectionError):
            await crew.context_retriever.retrieve_comprehensive_context(
                project_id="test_project",
                query="test query"
            )

    @pytest.mark.asyncio
    async def test_qa_agents_correlation_analysis(self, mock_openai_api_key):
        """Test correlation analysis between different QA agents"""
        
        # Mock responses with correlated issues
        correlated_responses = {
            "character": {
                "score": 70,
                "issues": [
                    {"type": "dialogue", "severity": "high", "character": "Sarah", "line": 5},
                    {"type": "consistency", "severity": "medium", "character": "Marcus", "line": 12}
                ]
            },
            "structural": {
                "score": 65, 
                "issues": [
                    {"type": "pacing", "severity": "high", "location": "lines 1-10"},
                    {"type": "flow", "severity": "medium", "location": "line 12"}
                ]
            },
            "style": {
                "score": 75,
                "issues": [
                    {"type": "repetition", "severity": "low", "phrase": "looked up", "line": 5},
                    {"type": "voice", "severity": "medium", "inconsistency": "formal/informal", "line": 12}
                ]
            },
            "technical": {
                "score": 80,
                "issues": [
                    {"type": "grammar", "severity": "low", "error": "comma splice", "line": 5}
                ]
            }
        }
        
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_char, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech:
            
            # Setup correlated responses
            for mock_client, agent_type in [
                (mock_char, "character"), (mock_struct, "structural"),
                (mock_style, "style"), (mock_tech, "technical")
            ]:
                client = AsyncMock()
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = json.dumps(correlated_responses[agent_type])
                client.chat.completions.create.return_value = response
                mock_client.return_value = client
            
            # Initialize agents and run parallel QA
            agents = [
                CharacterQAAgent(api_key=mock_openai_api_key),
                StructuralQAAgent(api_key=mock_openai_api_key), 
                StyleQAAgent(api_key=mock_openai_api_key),
                TechnicalQAAgent(api_key=mock_openai_api_key)
            ]
            
            # Run QA agents in parallel
            qa_tasks = [
                agents[0].review(self.mock_generated_content, self.mock_context["characters"]),
                agents[1].review(self.mock_generated_content, {}),
                agents[2].review(self.mock_generated_content, {}),
                agents[3].review(self.mock_generated_content, {})
            ]
            
            results = await asyncio.gather(*qa_tasks, return_exceptions=True)
            
            # Analyze correlations
            qa_tools = QATools()
            aggregated = qa_tools.aggregate_qa_results([
                {**result, "agent_type": agent_type} 
                for result, agent_type in zip(results, ["character", "structural", "style", "technical"])
            ])
            
            # Verify correlation detection
            assert "correlated_issues" in aggregated
            line_5_issues = [issue for result in results for issue in result["issues"] 
                           if "line" in issue and (issue.get("line") == 5 or "5" in str(issue.get("location", "")))]
            assert len(line_5_issues) >= 2  # Multiple agents flagged line 5
            
            line_12_issues = [issue for result in results for issue in result["issues"]
                            if "line" in issue and (issue.get("line") == 12 or "12" in str(issue.get("location", "")))]
            assert len(line_12_issues) >= 2  # Multiple agents flagged line 12

    @pytest.mark.asyncio
    async def test_database_client_connection_errors(self, mock_openai_api_key):
        """Test system resilience when database connections fail"""
        
        # Create clients that simulate connection failures
        failing_qdrant = MagicMock()
        failing_qdrant.search = AsyncMock(side_effect=ConnectionError("Qdrant connection lost"))
        
        failing_neo4j = MagicMock() 
        failing_neo4j.execute_query = AsyncMock(side_effect=TimeoutError("Neo4j query timeout"))
        
        failing_mongodb = MagicMock()
        failing_mongodb.find_one = AsyncMock(side_effect=Exception("MongoDB connection refused"))
        
        # Test graceful degradation
        crew = NovelGenerationCrew(
            qdrant_client=failing_qdrant,
            neo4j_client=failing_neo4j,
            mongodb_client=failing_mongodb,
            openai_api_key=mock_openai_api_key
        )
        
        # Test context retrieval with failing databases
        with pytest.raises((ConnectionError, TimeoutError)):
            await crew.context_retriever.retrieve_comprehensive_context(
                project_id="project_id",
                query="test query"
            )
            
        # Test novel tools with failing MongoDB (novel tools likely aren't async)
        with pytest.raises(Exception):
            crew.novel_tools.get_character_context(["test"], "project_id")
        
        # Verify system doesn't crash completely
        assert crew.generator_agent is not None
        assert crew.qa_tools is not None

    @pytest.mark.asyncio
    async def test_agent_parallel_execution_timing(self, mock_database_clients, mock_openai_api_key):
        """Test timing and coordination of parallel agent execution"""
        qdrant_client, neo4j_client, mongodb_client = mock_database_clients
        
        execution_times = {}
        
        def create_timed_mock(agent_name, base_delay=0.1):
            async def timed_completion(*args, **kwargs):
                start_time = time.time()
                await asyncio.sleep(base_delay)  # Simulate processing time
                end_time = time.time()
                execution_times[agent_name] = (start_time, end_time, end_time - start_time)
                
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = json.dumps({
                    "score": 85, "issues": [], "patches": [],
                    "processing_time": end_time - start_time
                })
                return response
            return timed_completion
        
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_char, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style, \
             patch('agents.technical_qa.AsyncOpenAI') as mock_tech:
            
            # Setup timed mocks with different processing delays
            mock_clients = [
                (mock_char, "character", 0.1),
                (mock_struct, "structural", 0.15), 
                (mock_style, "style", 0.12),
                (mock_tech, "technical", 0.08)
            ]
            
            for mock_openai, agent_name, delay in mock_clients:
                client = AsyncMock()
                client.chat.completions.create = create_timed_mock(agent_name, delay)
                mock_openai.return_value = client
            
            # Execute parallel QA
            agents = [
                CharacterQAAgent(api_key=mock_openai_api_key),
                StructuralQAAgent(api_key=mock_openai_api_key),
                StyleQAAgent(api_key=mock_openai_api_key), 
                TechnicalQAAgent(api_key=mock_openai_api_key)
            ]
            
            start_parallel = time.time()
            
            # Run agents in parallel
            tasks = [
                agents[0].review(self.mock_generated_content, self.mock_context["characters"]),
                agents[1].review(self.mock_generated_content, {}),
                agents[2].review(self.mock_generated_content, {}),
                agents[3].review(self.mock_generated_content, {})
            ]
            
            results = await asyncio.gather(*tasks)
            end_parallel = time.time()
            total_parallel_time = end_parallel - start_parallel
            
            # Verify parallel execution efficiency
            individual_times = [times[2] for times in execution_times.values()]
            total_sequential_time = sum(individual_times)
            
            # Parallel should be significantly faster than sequential
            assert total_parallel_time < total_sequential_time * 0.7
            assert len(execution_times) == 4
            
            # Verify overlap in execution times (agents ran concurrently)
            start_times = [times[0] for times in execution_times.values()]
            end_times = [times[1] for times in execution_times.values()]
            
            # Check that some agents started before others finished (overlap)
            max_start = max(start_times)
            min_end = min(end_times)
            assert max_start < min_end  # Some overlap occurred

    @pytest.mark.asyncio
    async def test_invalid_context_data_handling(self, mock_database_clients, mock_openai_api_key):
        """Test system handling of invalid or corrupted context data"""
        qdrant_client, neo4j_client, mongodb_client = mock_database_clients
        
        crew = NovelGenerationCrew(
            qdrant_client=qdrant_client,
            neo4j_client=neo4j_client,
            mongodb_client=mongodb_client, 
            openai_api_key=mock_openai_api_key
        )
        
        # Test with completely invalid context
        invalid_contexts = [
            None,  # Null context
            {},    # Empty context
            {"invalid": "structure"},  # Wrong structure
            {"characters": {"invalid": "format"}},  # Invalid characters format
            {"semantic": "not_a_dict"},  # Wrong semantic format
            {"combined_text": 123},  # Wrong type for text
        ]
        
        for invalid_context in invalid_contexts:
            # Test that system handles invalid context gracefully
            try:
                with patch('agents.generator_agent.AsyncOpenAI') as mock_gen:
                    gen_client = AsyncMock()
                    gen_response = MagicMock()
                    gen_response.choices = [MagicMock()]
                    gen_response.choices[0].message.content = "Generated content with invalid context"
                    gen_client.chat.completions.create.return_value = gen_response
                    mock_gen.return_value = gen_client
                    
                    result = await crew.generator_agent.generate(
                        prompt=self.test_prompt,
                        context=invalid_context.get("combined_text", "") if invalid_context else "",
                        length_words=500
                    )
                    
                    # Should still generate something, even with invalid context
                    assert isinstance(result, str)
                    assert len(result) > 0
                    
            except (TypeError, KeyError, AttributeError) as e:
                # Expected for some invalid context types
                assert invalid_context in [None, {"semantic": "not_a_dict"}, {"combined_text": 123}]
        
        # Test QA agents with invalid character context
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_char:
            char_client = AsyncMock() 
            char_response = MagicMock()
            char_response.choices = [MagicMock()]
            char_response.choices[0].message.content = json.dumps({
                "score": 50,  # Lower score due to invalid context
                "issues": [{"type": "context_error", "severity": "high"}],
                "patches": []
            })
            char_client.chat.completions.create.return_value = char_response
            mock_char.return_value = char_client
            
            char_qa = CharacterQAAgent(api_key=mock_openai_api_key)
            result = await char_qa.review(
                text=self.mock_generated_content,
                character_context={"invalid": "character_format"}
            )
            
            # Should handle gracefully and indicate context issues
            assert result["score"] <= 60  # Lower score due to context issues
            assert any("context" in issue.get("type", "") for issue in result["issues"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])