"""
Simple backend validation tests for the approval workflow system.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

# Test imports
from agent.models import (
    ApprovalRequest, ProposalItem, ValidationResult, 
    ProposalResponse, ApprovalDecision, Neo4jPushResult
)
from agent.consistency_validators import (
    run_all_validators, fact_check_validator, 
    behavior_consistency_validator, trope_detector_validator
)


class TestModels:
    """Test Pydantic models for approval workflow."""
    
    def test_proposal_item_creation(self):
        """Test ProposalItem model creation."""
        item = ProposalItem(
            type="character",
            name="Test Character",
            attributes={"personality": "brave"},
            confidence=0.8,
            excerpt="A brave warrior stood tall."
        )
        
        assert item.type == "character"
        assert item.name == "Test Character"
        assert item.confidence == 0.8
        assert "personality" in item.attributes
    
    def test_approval_request_creation(self):
        """Test ApprovalRequest model creation."""
        item = ProposalItem(
            type="character",
            name="Hero",
            attributes={"role": "protagonist"}
        )
        
        request = ApprovalRequest(
            kind="character",
            items=[item],
            source_doc="test.md",
            confidence=0.9
        )
        
        assert request.kind == "character"
        assert len(request.items) == 1
        assert request.confidence == 0.9
    
    def test_validation_result_creation(self):
        """Test ValidationResult model creation."""
        result = ValidationResult(
            validator_name="fact_checker",
            score=0.85,
            violations=[{"type": "minor", "message": "test"}],
            suggestions=["Check this fact"]
        )
        
        assert result.validator_name == "fact_checker"
        assert result.score == 0.85
        assert len(result.violations) == 1
        assert len(result.suggestions) == 1


class TestConsistencyValidators:
    """Test consistency validation functions."""
    
    @pytest.mark.asyncio
    async def test_fact_check_validator(self):
        """Test fact checking validator."""
        content = "John was born in 1990 and died in 2025."
        entity_data = {"name": "John", "type": "character"}
        established_facts = set()
        
        result = await fact_check_validator(content, entity_data, established_facts)
        
        assert "score" in result
        assert "violations" in result
        assert "suggestions" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_behavior_consistency_validator(self):
        """Test behavior consistency validator."""
        content = "The peaceful character attacked violently."
        entity_data = {
            "name": "PeacefulPerson",
            "attributes": {"personality": "peaceful calm"}
        }
        
        result = await behavior_consistency_validator(content, entity_data)
        
        assert "score" in result
        assert "violations" in result
        assert result["validator_type"] == "behavior_consistency"
        # Should detect contradiction between peaceful and violent behavior
        assert result["score"] < 1.0
    
    @pytest.mark.asyncio
    async def test_trope_detector_validator(self):
        """Test trope detection validator."""
        content = "The chosen one with a dark secret must fulfill the prophecy."
        entity_data = {"name": "Hero", "type": "character"}
        
        result = await trope_detector_validator(content, entity_data)
        
        assert "score" in result
        assert "violations" in result
        assert result["validator_type"] == "trope_detector"
        # Should detect multiple tropes
        assert len(result["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_run_all_validators(self):
        """Test running all validators together."""
        content = "The amazing chosen one was born in 3000 and speaks perfectly."
        entity_data = {
            "name": "FutureHero",
            "type": "character",
            "attributes": {"speech": "formal"}
        }
        established_facts = set()
        
        results = await run_all_validators(content, entity_data, established_facts)
        
        # Should have results from all validators
        expected_validators = [
            "fact_checker", "behavior_consistency", 
            "dialogue_style", "trope_detector", "timeline_consistency"
        ]
        
        for validator in expected_validators:
            assert validator in results
            assert "score" in results[validator]
            assert "violations" in results[validator]
            assert "suggestions" in results[validator]


class TestDatabaseOperations:
    """Test database operations with mocked connections."""
    
    @pytest.mark.asyncio
    async def test_create_proposal_mock(self):
        """Test proposal creation with mocked database."""
        from agent.db_utils import create_proposal
        
        # Mock the database pool
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = "test-uuid-123"
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            proposal_id = await create_proposal(
                kind="character",
                payload={"test": "data"},
                source_doc="test.md",
                confidence=0.8
            )
            
            assert proposal_id == "test-uuid-123"
            mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_proposal_mock(self):
        """Test proposal retrieval with mocked database."""
        from agent.db_utils import get_proposal
        
        # Mock the database pool
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_row = {
                "id": "test-uuid-123",
                "kind": "character",
                "status": "pending",
                "confidence": 0.8,
                "payload": '{"test": "data"}',
                "created_at": datetime.now(),
                "validation_count": 2,
                "avg_validation_score": 0.75,
                "risk_level": "low_risk"
            }
            mock_conn.fetchrow.return_value = mock_row
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            result = await get_proposal("test-uuid-123")
            
            assert result is not None
            assert result["id"] == "test-uuid-123"
            assert result["kind"] == "character"
            assert result["payload"]["test"] == "data"  # JSON parsed


class TestApprovalAPI:
    """Test approval API endpoints with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_approval_request_processing(self):
        """Test processing of approval requests."""
        # Mock all dependencies
        with patch('agent.approval_api.create_proposal') as mock_create, \
             patch('agent.approval_api.run_all_validators') as mock_validators, \
             patch('agent.approval_api.store_validation_result') as mock_store, \
             patch('agent.approval_api.get_proposal') as mock_get:
            
            # Setup mocks
            mock_create.return_value = "test-proposal-id"
            mock_validators.return_value = {
                "fact_checker": {
                    "score": 0.9,
                    "violations": [],
                    "suggestions": []
                }
            }
            mock_store.return_value = "validation-id"
            mock_get.return_value = {
                "id": "test-proposal-id",
                "status": "pending",
                "kind": "character",
                "confidence": 0.8,
                "created_at": datetime.now(),
                "risk_level": "low_risk"
            }
            
            # Import and test the endpoint function logic
            from agent.approval_api import create_new_proposal
            from agent.models import ApprovalRequest, ProposalItem
            
            # Create test request
            item = ProposalItem(
                type="character",
                name="Test Hero",
                excerpt="A brave hero emerged."
            )
            request = ApprovalRequest(
                kind="character",
                items=[item],
                confidence=0.8
            )
            
            # This would normally be called by FastAPI
            # We're testing the core logic
            result = await create_new_proposal(request)
            
            assert isinstance(result, ProposalResponse)
            assert result.proposal_id == "test-proposal-id"
            assert result.status == "pending"


def test_neo4j_push_result_model():
    """Test Neo4j push result model."""
    result = Neo4jPushResult(
        transaction_id="tx-123",
        nodes_created=2,
        relationships_created=1,
        errors=[]
    )
    
    assert result.transaction_id == "tx-123"
    assert result.nodes_created == 2
    assert result.relationships_created == 1
    assert len(result.errors) == 0


def test_approval_decision_model():
    """Test approval decision model."""
    decision = ApprovalDecision(
        action="approve",
        processed_by="user123",
        selected_items=[0, 1]
    )
    
    assert decision.action == "approve"
    assert decision.processed_by == "user123"
    assert decision.selected_items == [0, 1]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
