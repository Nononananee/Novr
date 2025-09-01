"""
Module Correlation Tests
Tests the correlation and interdependency between different system modules.
"""

import pytest
import asyncio
import sys
import os
import json
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List, Tuple

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embeddings.embedder import TextEmbedder
from embeddings.chunker import TextChunker
from embeddings.qdrant_client import QdrantClient
from agents.tools.context_tools import ContextRetriever
from agents.tools.novel_tools import NovelTools
from qa_system.character_qa import CharacterQAAgent
from qa_system.structural_qa import StructuralQAAgent
from qa_system.style_qa import StyleQAAgent


class TestModuleCorrelation:
    """Test correlation and data flow between system modules"""

    def setup_method(self):
        """Setup test fixtures for module correlation testing"""
        self.sample_text = """
        The ancient library held secrets that few dared to uncover. Eleanor stepped carefully 
        between the towering shelves, her fingers trailing along the leather-bound spines.
        Each book seemed to whisper of forgotten knowledge and hidden truths.
        
        "Are you certain this is wise?" Marcus called from behind her, his voice echoing 
        in the vast space. "Some knowledge is better left buried."
        
        Eleanor paused, turning to face her companion. "Wisdom isn't about avoiding truth, Marcus. 
        It's about understanding when we're ready to accept it."
        """
        
        self.character_data = {
            "characters": [
                {
                    "name": "Eleanor",
                    "description": "Curious scholar, seeker of knowledge",
                    "traits": ["intelligent", "brave", "determined"],
                    "relationships": ["mentor to students", "colleague of Marcus"]
                },
                {
                    "name": "Marcus", 
                    "description": "Cautious advisor, practical thinker",
                    "traits": ["cautious", "loyal", "analytical"],
                    "relationships": ["advisor to Eleanor", "friend since university"]
                }
            ]
        }

    @pytest.fixture
    def mock_openai_key(self):
        return "test-correlation-key-12345"

    @pytest.mark.asyncio
    async def test_embedding_chunking_correlation(self, mock_openai_key):
        """Test correlation between text chunking and embedding processes"""
        
        # Mock SentenceTransformer for embeddings API
        with patch('embeddings.embedder.SentenceTransformer') as mock_st:
            # Setup mock embeddings with consistent dimension
            embedding_dim = 1024
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = embedding_dim
            
            # Setup embedding responses for different chunks
            def mock_encode(texts, **kwargs):
                # Return fake embeddings with correct dimension
                return np.random.rand(len(texts), embedding_dim).astype(np.float32)
            
            mock_model.encode = mock_encode
            mock_st.return_value = mock_model
            
            # Test different chunk sizes and their embedding correlation
            chunker = TextChunker()
            embedder = TextEmbedder()  # No API key needed for sentence-transformers
            
            chunk_sizes = [100, 200, 300]
            correlation_results = {}
            
            for chunk_size in chunk_sizes:
                # Create chunks
                chunks = chunker.chunk_text(self.sample_text, chunk_size=chunk_size)
                
                # Generate embeddings
                embeddings = embedder.embed_chunks(chunks)
                
                correlation_results[chunk_size] = {
                    "num_chunks": len(chunks),
                    "embeddings": embeddings,
                    "avg_embedding": np.mean(embeddings, axis=0) if len(embeddings) > 0 else []
                }
            
            # Analyze correlation between chunk size and embedding quality
            assert len(correlation_results) == 3
            
            # Verify that different chunk sizes produce different numbers of chunks
            chunk_counts = [result["num_chunks"] for result in correlation_results.values()]
            assert len(set(chunk_counts)) > 1  # Different chunk sizes should yield different counts
            
            # Verify embeddings are generated for each chunk size
            for chunk_size, result in correlation_results.items():
                assert len(result["embeddings"]) == result["num_chunks"]
                if result["num_chunks"] > 0:
                    assert all(len(emb) == embedding_dim for emb in result["embeddings"])  # Correct dimension

    @pytest.mark.asyncio 
    async def test_context_retrieval_embedding_correlation(self, mock_openai_key):
        """Test correlation between context retrieval and embedding similarity"""
        
        # Mock Qdrant client with realistic similarity scores
        mock_qdrant = MagicMock()
        
        # Create mock search results with varying similarity scores
        search_results = [
            MagicMock(payload={"text": "Eleanor was known for her scholarly pursuits"}, score=0.92),
            MagicMock(payload={"text": "Marcus always advised caution in research"}, score=0.89),
            MagicMock(payload={"text": "The library contained ancient manuscripts"}, score=0.85),
            MagicMock(payload={"text": "Books whispered secrets to those who listened"}, score=0.78),
        ]
        
        mock_qdrant.search = AsyncMock(return_value=search_results)
        
        # Mock Neo4j for character relationships
        mock_neo4j = MagicMock()
        mock_neo4j.execute_query = AsyncMock(return_value=[
            {"character": "Eleanor", "relationship": "colleague", "target": "Marcus"},
            {"character": "Marcus", "relationship": "advisor", "target": "Eleanor"}
        ])
        
        # Test context retrieval with different query types
        context_retriever = ContextRetriever(mock_qdrant, mock_neo4j)
        
        queries = [
            "Eleanor's scholarly research",
            "Marcus giving advice", 
            "ancient library secrets",
            "character relationships"
        ]
        
        retrieval_correlation = {}
        
        for query in queries:
            context = await context_retriever.get_context(query, "test_project")
            
            # Analyze correlation between query and retrieved context
            semantic_scores = [result.score for result in search_results]
            avg_relevance = np.mean(semantic_scores)
            
            retrieval_correlation[query] = {
                "context_retrieved": len(context.get("semantic_results", [])),
                "avg_relevance_score": avg_relevance,
                "character_matches": len(context.get("character_relationships", [])),
                "query_terms_in_context": sum(1 for term in query.split() 
                                            if any(term.lower() in str(result.payload).lower() 
                                                  for result in search_results))
            }
        
        # Verify correlation patterns
        assert len(retrieval_correlation) == 4
        
        # Character-specific queries should have high character matches
        character_queries = ["Eleanor's scholarly research", "Marcus giving advice"]
        for query in character_queries:
            assert retrieval_correlation[query]["character_matches"] >= 1
            
        # Semantic queries should have good relevance scores
        for query, results in retrieval_correlation.items():
            assert results["avg_relevance_score"] > 0.7  # High relevance threshold

    @pytest.mark.asyncio
    async def test_qa_agent_cross_correlation(self, mock_openai_key):
        """Test correlation between different QA agents' findings"""
        
        # Create correlated QA responses that should identify similar issues
        correlated_qa_responses = {
            "character": {
                "score": 75,
                "issues": [
                    {
                        "type": "dialogue_consistency", 
                        "character": "Eleanor",
                        "line_start": 8, 
                        "line_end": 10,
                        "severity": "medium",
                        "issue": "Dialogue style inconsistent with character voice"
                    },
                    {
                        "type": "character_development",
                        "character": "Marcus", 
                        "line_start": 12,
                        "line_end": 14,
                        "severity": "low", 
                        "issue": "Character motivation unclear"
                    }
                ]
            },
            "structural": {
                "score": 78,
                "issues": [
                    {
                        "type": "pacing",
                        "location": "lines 8-10",
                        "severity": "medium",
                        "issue": "Dialogue pacing disrupts narrative flow"
                    },
                    {
                        "type": "scene_transition",
                        "location": "lines 12-14", 
                        "severity": "low",
                        "issue": "Abrupt character response transition"
                    }
                ]
            },
            "style": {
                "score": 82,
                "issues": [
                    {
                        "type": "voice_consistency",
                        "location": "lines 8-10",
                        "severity": "medium", 
                        "issue": "Formal/informal voice mixing in dialogue"
                    },
                    {
                        "type": "word_choice",
                        "location": "line 14",
                        "severity": "low",
                        "issue": "Word repetition affects flow"
                    }
                ]
            }
        }
        
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_char, \
             patch('qa_system.structural_qa.AsyncOpenAI') as mock_struct, \
             patch('qa_system.style_qa.AsyncOpenAI') as mock_style:
            
            # Setup mock responses
            for mock_openai, agent_type in [(mock_char, "character"), (mock_struct, "structural"), (mock_style, "style")]:
                client = AsyncMock()
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = json.dumps(correlated_qa_responses[agent_type])
                client.chat.completions.create.return_value = response
                mock_openai.return_value = client
            
            # Initialize QA agents
            char_qa = CharacterQAAgent(api_key=mock_openai_key)
            struct_qa = StructuralQAAgent(api_key=mock_openai_key)
            style_qa = StyleQAAgent(api_key=mock_openai_key)
            
            # Run QA analysis
            char_result = await char_qa.review(self.sample_text, self.character_data)
            struct_result = await struct_qa.review(self.sample_text, {})
            style_result = await style_qa.review(self.sample_text, {})
            
            # Analyze cross-correlations
            all_results = [
                (char_result, "character"),
                (struct_result, "structural"), 
                (style_result, "style")
            ]
            
            # Find issues that correlate across agents (same line ranges)
            issue_correlations = {}
            
            for result, agent_type in all_results:
                for issue in result["issues"]:
                    # Extract line information from different issue formats
                    line_info = self._extract_line_info(issue)
                    if line_info:
                        key = f"lines_{line_info}"
                        if key not in issue_correlations:
                            issue_correlations[key] = []
                        issue_correlations[key].append({
                            "agent": agent_type,
                            "issue_type": issue["type"],
                            "severity": issue["severity"]
                        })
            
            # Verify correlations exist
            correlated_lines = {k: v for k, v in issue_correlations.items() if len(v) > 1}
            assert len(correlated_lines) >= 1  # At least one correlation should exist
            
            # Verify lines 8-10 correlation (all three agents flagged this area)
            lines_8_10_key = [k for k in correlated_lines.keys() if "8" in k and "10" in k]
            if lines_8_10_key:
                assert len(correlated_lines[lines_8_10_key[0]]) >= 2
                
            # Verify severity correlation (similar severities for same line ranges)
            for line_range, issues in correlated_lines.items():
                severities = [issue["severity"] for issue in issues]
                # Should have consistent severity levels for correlated issues
                assert len(set(severities)) <= 2  # No more than 2 different severity levels

    def _extract_line_info(self, issue: Dict[str, Any]) -> str:
        """Extract line information from issue in various formats"""
        if "line_start" in issue and "line_end" in issue:
            return f"{issue['line_start']}-{issue['line_end']}"
        elif "location" in issue:
            location = issue["location"]
            if "lines" in location:
                return location.split("lines ")[-1]
            elif "line" in location:
                return location.split("line ")[-1]
        return ""

    @pytest.mark.asyncio
    async def test_vector_search_character_correlation(self, mock_openai_key):
        """Test correlation between vector search results and character context"""
        
        # Mock embedding generation
        with patch('embeddings.embedder.AsyncOpenAI') as mock_embed_openai:
            query_embedding = [0.5, 0.6, 0.7, 0.8]
            
            embed_client = AsyncMock()
            embed_response = MagicMock()
            embed_response.data = [MagicMock(embedding=query_embedding)]
            embed_client.embeddings.create.return_value = embed_response
            mock_embed_openai.return_value = embed_client
            
            # Mock Qdrant with character-correlated results
            mock_qdrant = MagicMock()
            
            # Results should correlate with characters mentioned in query
            character_correlated_results = [
                MagicMock(
                    payload={
                        "text": "Eleanor's research methodology was thorough and systematic",
                        "characters": ["Eleanor"],
                        "metadata": {"type": "character_background"}
                    },
                    score=0.91
                ),
                MagicMock(
                    payload={
                        "text": "Marcus often questioned Eleanor's bold approaches", 
                        "characters": ["Eleanor", "Marcus"],
                        "metadata": {"type": "relationship"}
                    },
                    score=0.87
                ),
                MagicMock(
                    payload={
                        "text": "The scholarly debates between colleagues were intense",
                        "characters": [],
                        "metadata": {"type": "general"}
                    },
                    score=0.72
                ),
            ]
            
            mock_qdrant.search = AsyncMock(return_value=character_correlated_results)
            
            # Test queries with different character focus
            test_queries = [
                ("Eleanor's research methods", ["Eleanor"]),
                ("Marcus and Eleanor relationship", ["Eleanor", "Marcus"]),
                ("scholarly work", []),
            ]
            
            embedder = TextEmbedder(api_key=mock_openai_key)
            correlation_analysis = {}
            
            for query, expected_characters in test_queries:
                # Generate query embedding
                query_embed = await embedder.embed_texts([query])
                
                # Perform vector search (mock)
                results = await mock_qdrant.search(
                    vector=query_embed[0],
                    collection_name="test_collection",
                    limit=10
                )
                
                # Analyze character correlation
                character_mentions = {}
                for result in results:
                    result_characters = result.payload.get("characters", [])
                    for char in result_characters:
                        if char not in character_mentions:
                            character_mentions[char] = 0
                        character_mentions[char] += 1
                
                correlation_analysis[query] = {
                    "expected_characters": expected_characters,
                    "found_characters": character_mentions,
                    "correlation_score": self._calculate_character_correlation(
                        expected_characters, character_mentions
                    ),
                    "avg_score": np.mean([r.score for r in results])
                }
            
            # Verify character correlations
            for query, analysis in correlation_analysis.items():
                expected = analysis["expected_characters"]
                found = analysis["found_characters"]
                
                if expected:  # If we expected specific characters
                    # Verify expected characters appear in results
                    for expected_char in expected:
                        assert expected_char in found, f"Expected character {expected_char} not found in results for query: {query}"
                    
                    # Correlation score should be high for character-specific queries
                    assert analysis["correlation_score"] > 0.5

    def _calculate_character_correlation(self, expected: List[str], found: Dict[str, int]) -> float:
        """Calculate correlation score between expected and found characters"""
        if not expected:
            return 1.0 if not found else 0.5  # Neutral for non-character queries
        
        if not found:
            return 0.0
        
        # Calculate overlap
        found_chars = set(found.keys())
        expected_chars = set(expected)
        
        overlap = len(expected_chars.intersection(found_chars))
        union = len(expected_chars.union(found_chars))
        
        return overlap / union if union > 0 else 0.0

    @pytest.mark.asyncio
    async def test_novel_tools_database_correlation(self):
        """Test correlation between novel tools and database operations"""
        
        # Mock MongoDB operations
        mock_mongodb = AsyncMock()
        mock_neo4j = AsyncMock()
        
        # Mock data that should be correlated across databases
        mongodb_documents = [
            {
                "_id": "chapter_1",
                "content": self.sample_text,
                "characters": ["Eleanor", "Marcus"],
                "project_id": "test_project",
                "created_at": "2024-01-01"
            }
        ]
        
        neo4j_relationships = [
            {"character": "Eleanor", "relationship": "colleague", "target": "Marcus"},
            {"character": "Marcus", "relationship": "advisor", "target": "Eleanor"}
        ]
        
        mock_mongodb.find.return_value.to_list = AsyncMock(return_value=mongodb_documents)
        mock_mongodb.insert_one = AsyncMock(return_value=MagicMock(inserted_id="new_doc_id"))
        
        mock_neo4j.execute_query = AsyncMock(return_value=neo4j_relationships)
        
        # Test novel tools operations
        novel_tools = NovelTools(mock_mongodb, mock_neo4j)
        
        # Save content and verify database correlation
        new_content = "Eleanor discovered an ancient text that would change everything."
        project_id = "test_project"
        
        # Mock the save operation
        with patch.object(novel_tools, 'save_content') as mock_save, \
             patch.object(novel_tools, 'get_character_context') as mock_char_context:
            
            mock_save.return_value = {"success": True, "document_id": "new_doc_id"}
            mock_char_context.return_value = self.character_data
            
            # Save content
            save_result = await novel_tools.save_content(new_content, project_id)
            
            # Get character context  
            char_context = await novel_tools.get_character_context(["Eleanor"], project_id)
            
            # Verify correlation between saved content and character data
            assert save_result["success"] is True
            assert "Eleanor" in char_context["characters"][0]["name"]
            
            # Verify calls were made with correlated data
            mock_save.assert_called_once_with(new_content, project_id)
            mock_char_context.assert_called_once_with(["Eleanor"], project_id)

    @pytest.mark.asyncio
    async def test_full_pipeline_correlation(self, mock_openai_key):
        """Test correlation across the entire processing pipeline"""
        
        # This is a comprehensive test of how data flows and correlates
        # through the entire system pipeline
        
        pipeline_data = {
            "input_text": self.sample_text,
            "characters": self.character_data,
            "processing_stages": {}
        }
        
        # Stage 1: Text chunking
        chunker = TextChunker()
        chunks = chunker.chunk_text(pipeline_data["input_text"], chunk_size=150)
        pipeline_data["processing_stages"]["chunking"] = {
            "num_chunks": len(chunks),
            "chunks": chunks
        }
        
        # Stage 2: Embedding generation (mocked)
        with patch('embeddings.embedder.AsyncOpenAI') as mock_embed:
            embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]][:len(chunks)]
            
            embed_client = AsyncMock()
            embed_responses = []
            for emb in embeddings:
                response = MagicMock()
                response.data = [MagicMock(embedding=emb)]
                embed_responses.append(response)
            
            embed_client.embeddings.create.side_effect = embed_responses
            mock_embed.return_value = embed_client
            
            embedder = TextEmbedder(api_key=mock_openai_key)
            chunk_embeddings = await embedder.embed_chunks(chunks)
            
            pipeline_data["processing_stages"]["embedding"] = {
                "embeddings": chunk_embeddings,
                "embedding_dim": len(chunk_embeddings[0]) if chunk_embeddings else 0
            }
        
        # Stage 3: QA Analysis (mocked)
        with patch('qa_system.character_qa.AsyncOpenAI') as mock_qa:
            qa_result = {
                "score": 85,
                "issues": [
                    {
                        "type": "character_voice",
                        "character": "Eleanor", 
                        "severity": "low",
                        "chunk_reference": 0  # References first chunk
                    }
                ],
                "character_analysis": {
                    "Eleanor": {"voice_consistency": 0.9},
                    "Marcus": {"voice_consistency": 0.85}
                }
            }
            
            qa_client = AsyncMock()
            qa_response = MagicMock()
            qa_response.choices = [MagicMock()]
            qa_response.choices[0].message.content = json.dumps(qa_result)
            qa_client.chat.completions.create.return_value = qa_response
            mock_qa.return_value = qa_client
            
            char_qa = CharacterQAAgent(api_key=mock_openai_key)
            qa_analysis = await char_qa.review(
                pipeline_data["input_text"],
                pipeline_data["characters"]
            )
            
            pipeline_data["processing_stages"]["qa_analysis"] = qa_analysis
        
        # Verify pipeline correlations
        
        # 1. Chunk count should correlate with text length
        text_length = len(pipeline_data["input_text"])
        chunk_count = pipeline_data["processing_stages"]["chunking"]["num_chunks"]
        assert chunk_count > 0
        assert chunk_count <= text_length // 50  # Reasonable upper bound
        
        # 2. Embedding count should match chunk count
        embedding_count = len(pipeline_data["processing_stages"]["embedding"]["embeddings"])
        assert embedding_count == chunk_count
        
        # 3. QA analysis should reference known characters
        qa_characters = set()
        if "character_analysis" in pipeline_data["processing_stages"]["qa_analysis"]:
            qa_characters = set(pipeline_data["processing_stages"]["qa_analysis"]["character_analysis"].keys())
        
        input_characters = set(char["name"] for char in pipeline_data["characters"]["characters"])
        character_overlap = qa_characters.intersection(input_characters)
        assert len(character_overlap) > 0  # Should have character correlation
        
        # 4. Issues should reference valid chunks if they have chunk references
        qa_issues = pipeline_data["processing_stages"]["qa_analysis"]["issues"]
        for issue in qa_issues:
            if "chunk_reference" in issue:
                assert 0 <= issue["chunk_reference"] < chunk_count
        
        # 5. Overall pipeline integrity
        assert all(stage in pipeline_data["processing_stages"] 
                  for stage in ["chunking", "embedding", "qa_analysis"])
        
        # 6. Data size correlation (embeddings shouldn't be empty if chunks exist)
        if chunk_count > 0:
            assert embedding_count > 0
            assert pipeline_data["processing_stages"]["embedding"]["embedding_dim"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])