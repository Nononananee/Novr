from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"

class GenerateResponse(BaseModel):
    """Response model for generation request"""
    status: str = Field(..., description="Request status")
    code: int = Field(..., description="HTTP status code")
    data: Dict[str, Any] = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "queued",
                "code": 202,
                "data": {
                    "job_id": "job_123456789",
                    "estimated_completion": "2024-01-01T12:30:00Z"
                },
                "message": "Chapter generation job queued successfully"
            }
        }

class JobResponse(BaseModel):
    """Response model for job status"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, description="Progress percentage (0-1)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_123456789",
                "status": "success",
                "progress": 1.0,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:30:00Z",
                "result": {
                    "version_id": "version_abc123",
                    "chapter_content": "The forest was alive with magic...",
                    "qa_score": 85,
                    "revision_count": 1
                }
            }
        }

class QAResult(BaseModel):
    """QA analysis result"""
    score: int = Field(..., description="Quality score (0-100)")
    issues: List[Dict[str, Any]] = Field(default=[], description="Identified issues")
    patches: List[Dict[str, Any]] = Field(default=[], description="Suggested patches")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 85,
                "issues": [
                    {
                        "loc": 45,
                        "issue": "Grammar: subject-verb disagreement",
                        "suggestion": "Change 'was' to 'were'"
                    }
                ],
                "patches": [
                    {
                        "loc": 45,
                        "replacement": "The trees were swaying"
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    environment: str = Field(..., description="Environment name")
    services: Dict[str, str] = Field(..., description="Service endpoints")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))