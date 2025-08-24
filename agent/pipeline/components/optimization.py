"""Content optimization components for the generation pipeline."""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for generation."""
    FAST = "fast"           # Basic optimization, fastest generation
    BALANCED = "balanced"   # Good balance of quality and speed
    QUALITY = "quality"     # Maximum quality, slower generation
    ADAPTIVE = "adaptive"   # Adapt based on content complexity


@dataclass
class QualityCheckpoint:
    """Quality checkpoint during generation."""
    checkpoint_name: str
    quality_score: float
    timestamp: float
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ContentOptimizer:
    """Handles content optimization strategies."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.quality_thresholds = {
            OptimizationLevel.FAST: 0.7,
            OptimizationLevel.BALANCED: 0.85,
            OptimizationLevel.QUALITY: 0.95,
            OptimizationLevel.ADAPTIVE: 0.9
        }
    
    async def optimize_generation_request(self, request) -> Any:
        """Optimize generation request based on content analysis."""
        
        # Analyze request content for optimization opportunities
        content_analysis = await self._analyze_request_content(request)
        
        # Create optimized request
        optimized_request = request.copy() if hasattr(request, 'copy') else request
        
        # Adjust parameters based on content complexity
        if content_analysis.get("complexity_score", 0.5) > 0.7:
            # High complexity content needs more careful handling
            if hasattr(optimized_request, 'max_tokens'):
                optimized_request.max_tokens = min(optimized_request.max_tokens * 1.2, 2000)
            
            # Increase quality requirements
            if hasattr(optimized_request, 'quality_threshold'):
                optimized_request.quality_threshold = max(0.9, getattr(optimized_request, 'quality_threshold', 0.8))
        
        # Optimize for content type
        content_type = content_analysis.get("primary_type", "narrative")
        if content_type == "dialogue":
            optimized_request.chunking_strategy = "dialogue_preserving"
        elif content_type == "action":
            optimized_request.chunking_strategy = "action_oriented"
        elif content_analysis.get("character_count", 0) > 3:
            optimized_request.chunking_strategy = "character_focused"
        
        return optimized_request
    
    async def prepare_enhanced_context(self, request, base_pipeline) -> Dict[str, Any]:
        """Prepare enhanced context using optimization strategies."""
        
        # Step 1: Get base context from existing pipeline
        base_context = await base_pipeline._prepare_generation_context(request)
        
        # Step 2: Apply enhanced chunking if content exists
        if base_context.get("content_chunks"):
            # Convert to enhanced chunks
            raw_content = " ".join([chunk.get("content", "") for chunk in base_context["content_chunks"]])
            
            if raw_content.strip():
                try:
                    from memory.enhanced_chunking_strategies import chunk_novel_content, analyze_chunking_performance
                    
                    enhanced_chunks = await chunk_novel_content(
                        raw_content,
                        max_chunk_size=800,
                        strategy=getattr(request, 'chunking_strategy', None)
                    )
                    
                    base_context["enhanced_chunks"] = enhanced_chunks
                    base_context["chunking_performance"] = await analyze_chunking_performance(enhanced_chunks)
                except ImportError:
                    logger.warning("Enhanced chunking not available, using basic chunking")
                    base_context["chunking_performance"] = {"overall_performance": 0.7}
        
        # Step 3: Apply context quality optimization
        if base_context.get("context_string"):
            try:
                from ..optimization.enhanced_context_optimizer import optimize_context_with_quality_assurance
                
                context_chunks = [{"content": base_context["context_string"], "id": "main_context"}]
                
                optimization_result = await optimize_context_with_quality_assurance(
                    chunks=context_chunks,
                    max_tokens=getattr(request, 'max_tokens', 1500),
                    query_context=getattr(request, 'content', ''),
                    target_quality=self.quality_thresholds[self.optimization_level]
                )
                
                base_context["optimized_context"] = optimization_result.optimized_context
                base_context["context_quality_score"] = optimization_result.quality_score
                base_context["optimization_ratio"] = optimization_result.optimization_ratio
            except ImportError:
                logger.warning("Enhanced context optimizer not available")
                base_context["context_quality_score"] = 0.8
                base_context["optimized_context"] = base_context.get("context_string", "")
        
        # Step 4: Add enhanced metadata
        base_context["enhancement_metadata"] = {
            "optimization_level": self.optimization_level.value,
            "timestamp": time.time(),
            "strategies_applied": ["chunking", "context_optimization"]
        }
        
        return base_context
    
    async def validate_context_quality(self, context: Dict[str, Any]) -> QualityCheckpoint:
        """Validate context quality against thresholds."""
        
        checkpoint_name = "context_quality_validation"
        start_time = time.time()
        
        # Get context for quality analysis
        context_text = context.get("optimized_context") or context.get("context_string", "")
        
        if not context_text.strip():
            return QualityCheckpoint(
                checkpoint_name=checkpoint_name,
                quality_score=0.0,
                timestamp=start_time,
                passed=False,
                issues=["Empty context provided"],
                suggestions=["Ensure content is available for context generation"]
            )
        
        # Analyze context quality
        try:
            from ..optimization.enhanced_context_optimizer import analyze_context_quality
            quality_assessment = await analyze_context_quality(context_text)
            quality_score = quality_assessment.overall_score
            issues = quality_assessment.quality_issues
            suggestions = quality_assessment.improvement_suggestions
        except ImportError:
            # Fallback quality assessment
            quality_score = min(0.9, len(context_text.split()) / 100)  # Simple word count based
            issues = []
            suggestions = []
            if quality_score < 0.5:
                issues.append("Context appears too short")
                suggestions.append("Provide more detailed context")
        
        # Check against threshold
        threshold = self.quality_thresholds[self.optimization_level]
        passed = quality_score >= threshold
        
        return QualityCheckpoint(
            checkpoint_name=checkpoint_name,
            quality_score=quality_score,
            timestamp=start_time,
            passed=passed,
            issues=issues if not passed else [],
            suggestions=suggestions if not passed else []
        )
    
    async def _analyze_request_content(self, request) -> Dict[str, Any]:
        """Analyze request content characteristics."""
        import re
        
        content = getattr(request, 'content', '')
        
        if not content:
            return {"complexity_score": 0.5, "primary_type": "narrative"}
        
        # Simple content analysis
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        dialogue_count = content.count('"')
        character_mentions = len(re.findall(r'\\b[A-Z][a-z]+\\b', content))
        
        # Calculate complexity
        complexity_factors = [
            min(1.0, word_count / 100),
            min(1.0, sentence_count / 10),
            min(1.0, dialogue_count / 5),
            min(1.0, character_mentions / 5)
        ]
        complexity_score = sum(complexity_factors) / len(complexity_factors)
        
        # Determine primary type
        if dialogue_count > 2:
            primary_type = "dialogue"
        elif any(word in content.lower() for word in ['ran', 'grabbed', 'jumped', 'hit']):
            primary_type = "action"
        else:
            primary_type = "narrative"
        
        return {
            "complexity_score": complexity_score,
            "primary_type": primary_type,
            "word_count": word_count,
            "character_count": character_mentions
        }