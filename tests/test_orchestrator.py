import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.orchestrator import JobOrchestrator
from backend.app.schemas.requests import GenerateRequest

class TestJobOrchestrator:
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_request = {
            "user_id": "test_user",
            "project_id": "test_project",
            "chapter_id": "test_chapter",
            "prompt": "Write a test chapter",
            "settings": {
                "length_words": 1000,
                "max_revision_rounds": 2
            }
        }
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing"""
        with patch('backend.app.services.orchestrator.redis') as mock_redis, \
             patch('backend.app.services.orchestrator.Queue') as mock_queue, \
             patch('backend.app.services.orchestrator.get_mongodb') as mock_mongodb:
            
            # Setup mocks
            mock_redis_client = MagicMock()
            mock_redis.from_url.return_value = mock_redis_client
            
            mock_queue_instance = MagicMock()
            mock_queue.return_value = mock_queue_instance
            
            mock_db = AsyncMock()
            mock_mongodb.return_value = mock_db
            
            orchestrator = JobOrchestrator()
            orchestrator.redis_client = mock_redis_client
            orchestrator.queue = mock_queue_instance
            
            return orchestrator, mock_db, mock_queue_instance
    
    @pytest.mark.asyncio
    async def test_enqueue_job_success(self, mock_orchestrator):
        """Test successful job enqueueing"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mocks
        mock_db.create_job.return_value = "mock_doc_id"
        mock_job = MagicMock()
        mock_job.id = "mock_rq_job_id"
        mock_queue.enqueue.return_value = mock_job
        
        # Test enqueueing
        job_id = await orchestrator.enqueue_job(self.sample_request)
        
        # Assertions
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        
        # Verify MongoDB create_job was called
        mock_db.create_job.assert_called_once()
        call_args = mock_db.create_job.call_args[0][0]
        assert call_args["user_id"] == "test_user"
        assert call_args["project_id"] == "test_project"
        assert call_args["state"] == "queued"
        assert call_args["progress"] == 0.0
        
        # Verify Redis queue enqueue was called
        mock_queue.enqueue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enqueue_job_redis_failure(self, mock_orchestrator):
        """Test job enqueueing when Redis fails"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mocks
        mock_db.create_job.return_value = "mock_doc_id"
        mock_db.update_job.return_value = True
        mock_queue.enqueue.side_effect = Exception("Redis connection failed")
        
        # Test enqueueing should raise exception
        with pytest.raises(Exception):
            await orchestrator.enqueue_job(self.sample_request)
        
        # Verify that job status was updated to failed
        mock_db.update_job.assert_called_once()
        update_args = mock_db.update_job.call_args[0][1]
        assert update_args["state"] == "failed"
        assert "Redis connection failed" in update_args["error"]
    
    @pytest.mark.asyncio
    async def test_get_job_status_success(self, mock_orchestrator):
        """Test successful job status retrieval"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mock job data
        mock_job_data = {
            "job_id": "test_job_123",
            "state": "running",
            "progress": 0.5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:30:00Z",
            "result": None,
            "error": None
        }
        mock_db.get_job.return_value = mock_job_data
        
        # Test getting job status
        status = await orchestrator.get_job_status("test_job_123")
        
        # Assertions
        assert status["job_id"] == "test_job_123"
        assert status["status"] == "running"
        assert status["progress"] == 0.5
        assert status["result"] is None
        assert status["error"] is None
        
        mock_db.get_job.assert_called_once_with("test_job_123")
    
    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, mock_orchestrator):
        """Test job status retrieval when job doesn't exist"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mock to return None (job not found)
        mock_db.get_job.return_value = None
        
        # Test getting job status should raise ValueError
        with pytest.raises(ValueError, match="Job test_job_123 not found"):
            await orchestrator.get_job_status("test_job_123")
    
    @pytest.mark.asyncio
    async def test_update_job_status(self, mock_orchestrator):
        """Test job status update"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mock
        mock_db.update_job.return_value = True
        
        # Test updating job status
        updates = {
            "state": "success",
            "progress": 1.0,
            "result": {"version_id": "v123"}
        }
        result = await orchestrator.update_job_status("test_job_123", updates)
        
        # Assertions
        assert result is True
        mock_db.update_job.assert_called_once_with("test_job_123", updates)
    
    @pytest.mark.asyncio
    async def test_get_queue_info_success(self, mock_orchestrator):
        """Test successful queue info retrieval"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mocks
        mock_queue.__len__ = MagicMock(return_value=5)
        mock_failed_registry = MagicMock()
        mock_failed_registry.__len__ = MagicMock(return_value=2)
        mock_queue.failed_job_registry = mock_failed_registry
        mock_queue.workers = [MagicMock(), MagicMock()]  # 2 workers
        mock_queue.name = "novel_generation"
        
        # Test getting queue info
        info = await orchestrator.get_queue_info()
        
        # Assertions
        assert info["queued_jobs"] == 5
        assert info["failed_jobs"] == 2
        assert info["workers"] == 2
        assert info["queue_name"] == "novel_generation"
    
    @pytest.mark.asyncio
    async def test_get_queue_info_failure(self, mock_orchestrator):
        """Test queue info retrieval failure"""
        orchestrator, mock_db, mock_queue = mock_orchestrator
        
        # Setup mock to raise exception
        mock_queue.__len__ = MagicMock(side_effect=Exception("Redis error"))
        
        # Test getting queue info
        info = await orchestrator.get_queue_info()
        
        # Should return error info
        assert "error" in info
        assert "Redis error" in info["error"]

class TestJobOrchestrationIntegration:
    """Integration tests for job orchestration"""
    
    @pytest.mark.asyncio
    async def test_full_job_lifecycle_mock(self):
        """Test complete job lifecycle with mocks"""
        with patch('backend.app.services.orchestrator.redis') as mock_redis, \
             patch('backend.app.services.orchestrator.Queue') as mock_queue, \
             patch('backend.app.services.orchestrator.get_mongodb') as mock_mongodb:
            
            # Setup mocks
            mock_redis_client = MagicMock()
            mock_redis.from_url.return_value = mock_redis_client
            
            mock_queue_instance = MagicMock()
            mock_queue.return_value = mock_queue_instance
            
            mock_db = AsyncMock()
            mock_mongodb.return_value = mock_db
            
            orchestrator = JobOrchestrator()
            
            # Mock successful job creation and enqueueing
            mock_db.create_job.return_value = "doc_id"
            mock_job = MagicMock()
            mock_job.id = "rq_job_id"
            mock_queue_instance.enqueue.return_value = mock_job
            
            # Step 1: Enqueue job
            request_data = {
                "user_id": "user1",
                "project_id": "proj1", 
                "prompt": "Test prompt",
                "settings": {}
            }
            job_id = await orchestrator.enqueue_job(request_data)
            assert job_id is not None
            
            # Step 2: Check initial status
            mock_db.get_job.return_value = {
                "job_id": job_id,
                "state": "queued",
                "progress": 0.0,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "result": None,
                "error": None
            }
            
            status = await orchestrator.get_job_status(job_id)
            assert status["status"] == "queued"
            assert status["progress"] == 0.0
            
            # Step 3: Update to running
            mock_db.update_job.return_value = True
            success = await orchestrator.update_job_status(job_id, {
                "state": "running",
                "progress": 0.5
            })
            assert success is True
            
            # Step 4: Complete job
            success = await orchestrator.update_job_status(job_id, {
                "state": "success",
                "progress": 1.0,
                "result": {"version_id": "v123", "content": "Generated content"}
            })
            assert success is True

if __name__ == "__main__":
    pytest.main([__file__])