from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class GenerateRequest(BaseModel):
    """Request model for chapter generation"""
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(..., description="Project identifier")
    chapter_id: Optional[str] = Field(None, description="Chapter identifier (optional)")
    prompt: str = Field(..., description="Generation prompt/instruction")
    settings: Dict[str, Any] = Field(
        default={
            "length_words": 1200,
            "max_revision_rounds": 2,
            "temperature": 0.7,
            "tone": "engaging",
            "style": "narrative"
        },
        description="Generation settings"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "project_id": "fantasy_novel_1",
                "chapter_id": "chapter_1",
                "prompt": "Write the opening chapter where the protagonist discovers their magical abilities in a mysterious forest",
                "settings": {
                    "length_words": 1200,
                    "max_revision_rounds": 2,
                    "temperature": 0.7,
                    "tone": "mysterious",
                    "style": "descriptive"
                }
            }
        }

class QAReviewRequest(BaseModel):
    """Request model for QA review"""
    text: str = Field(..., description="Text to review")
    review_type: str = Field(default="technical", description="Type of QA review")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The protagonist walked through the forest, feeling a strange energy...",
                "review_type": "technical"
            }
        }