"""
Enhanced Generation Pipeline for Optimized Real-world Content Success
Refactored from large monolithic file into modular components.
"""

import logging
from typing import Dict, List, Optional, Any

# Import all components from modular structure
from .components.optimization import (
    OptimizationLevel,
    QualityCheckpoint,
    ContentOptimizer
)

from .components.quality import QualityValidator

from .components.metrics import (
    EnhancedGenerationMetrics,
    MetricsCalculator,
    analyze_generation_performance
)

from .components.pipeline import EnhancedGenerationPipeline

logger = logging.getLogger(__name__)

# Import base generation types with fallback
try:
    from .generation_pipeline import (
        GenerationType, GenerationMode, 
        GenerationRequest, GenerationResult
    )
    GENERATION_PIPELINE_AVAILABLE = True
except ImportError:
    GENERATION_PIPELINE_AVAILABLE = False
    logger.warning("Generation pipeline not available, using fallback implementations")
    
    # Create robust mock classes for testing
    class GenerationType:
        NARRATIVE_CONTINUATION = "narrative_continuation"
        CHARACTER_DIALOGUE = "character_dialogue"
        SCENE_DESCRIPTION = "scene_description"
    
    class GenerationMode:
        AUTOMATIC = "automatic"
        SEMI_AUTOMATIC = "semi_automatic"
    
    class GenerationRequest:
        def __init__(self, content="", generation_type=None, max_tokens=500):
            self.content = content
            self.generation_type = generation_type or GenerationType.NARRATIVE_CONTINUATION
            self.max_tokens = max_tokens
        
        def copy(self):
            return GenerationRequest(self.content, self.generation_type, self.max_tokens)
    
    class GenerationResult:
        def __init__(self, generated_content="", metadata=None):
            self.generated_content = generated_content
            self.metadata = metadata or {}
            self.fallback_used = True


# Global enhanced pipeline instance
enhanced_pipeline = EnhancedGenerationPipeline(OptimizationLevel.ADAPTIVE)


# Convenience functions for backward compatibility
async def generate_optimized_content(request: GenerationRequest, 
                                   optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE) -> GenerationResult:
    """
    Generate content with enhanced optimization.
    
    Args:
        request: Generation request
        optimization_level: Level of optimization to apply
        
    Returns:
        Enhanced generation result
    """
    pipeline = EnhancedGenerationPipeline(optimization_level)
    return await pipeline.generate_content(request)


# Export all important classes and functions for backward compatibility
__all__ = [
    # Core classes
    "EnhancedGenerationPipeline",
    "ContentOptimizer",
    "QualityValidator", 
    "MetricsCalculator",
    
    # Enums
    "OptimizationLevel",
    
    # Data classes
    "EnhancedGenerationMetrics",
    "QualityCheckpoint",
    
    # Generation types (from base or fallback)
    "GenerationType",
    "GenerationMode",
    "GenerationRequest", 
    "GenerationResult",
    
    # Global instance
    "enhanced_pipeline",
    
    # Convenience functions
    "generate_optimized_content",
    "analyze_generation_performance"
]