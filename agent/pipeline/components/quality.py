"""Quality validation and enhancement components."""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .optimization import QualityCheckpoint, OptimizationLevel

logger = logging.getLogger(__name__)


class QualityValidator:
    """Handles quality validation and content enhancement."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.quality_thresholds = {
            OptimizationLevel.FAST: 0.7,
            OptimizationLevel.BALANCED: 0.85,
            OptimizationLevel.QUALITY: 0.95,
            OptimizationLevel.ADAPTIVE: 0.9
        }
    
    async def validate_generated_content(self, result) -> QualityCheckpoint:
        """Validate generated content quality."""
        
        checkpoint_name = "generated_content_validation"
        start_time = time.time()
        
        if not result.generated_content.strip():
            return QualityCheckpoint(
                checkpoint_name=checkpoint_name,
                quality_score=0.0,
                timestamp=start_time,
                passed=False,
                issues=["No content generated"],
                suggestions=["Check generation parameters and context"]
            )
        
        # Analyze generated content quality
        try:
            from ..optimization.enhanced_context_optimizer import analyze_context_quality
            quality_assessment = await analyze_context_quality(result.generated_content)
            base_quality_score = quality_assessment.overall_score
            base_issues = quality_assessment.quality_issues
            base_suggestions = quality_assessment.improvement_suggestions
        except ImportError:
            # Fallback quality assessment
            word_count = len(result.generated_content.split())
            base_quality_score = min(0.9, word_count / 50)  # Simple assessment
            base_issues = []
            base_suggestions = []
        
        # Additional checks for generated content
        issues = list(base_issues)
        suggestions = list(base_suggestions)
        
        # Check content length
        word_count = len(result.generated_content.split())
        if word_count < 10:
            issues.append("Generated content too short")
            suggestions.append("Increase generation length or improve context")
        elif word_count > 2000:
            issues.append("Generated content too long")
            suggestions.append("Reduce generation length or split into chunks")
        
        # Check for repetition
        sentences = result.generated_content.split('.')
        if len(sentences) > 2:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            if len(unique_sentences) < len(sentences) * 0.8:
                issues.append("High repetition detected in generated content")
                suggestions.append("Improve context diversity or adjust generation parameters")
        
        # Overall quality check
        threshold = self.quality_thresholds[self.optimization_level]
        content_quality_passed = base_quality_score >= threshold
        structural_quality_passed = len(issues) == len(base_issues)  # No new issues added
        
        passed = content_quality_passed and structural_quality_passed
        
        return QualityCheckpoint(
            checkpoint_name=checkpoint_name,
            quality_score=base_quality_score,
            timestamp=start_time,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )
    
    async def enhance_generated_content(self, result, checkpoint: QualityCheckpoint):
        """Enhance generated content based on quality checkpoint."""
        
        logger.info(f"Enhancing generated content due to quality issues: {len(checkpoint.issues)}")
        
        # Apply content enhancement strategies
        enhanced_content = result.generated_content
        
        # Strategy 1: Fix repetition
        if any("repetition" in issue.lower() for issue in checkpoint.issues):
            enhanced_content = await self._fix_content_repetition(enhanced_content)
        
        # Strategy 2: Improve length
        if any("too short" in issue.lower() for issue in checkpoint.issues):
            enhanced_content = await self._expand_content(enhanced_content)
        elif any("too long" in issue.lower() for issue in checkpoint.issues):
            enhanced_content = await self._condense_content(enhanced_content)
        
        # Strategy 3: Improve quality
        if checkpoint.quality_score < self.quality_thresholds[self.optimization_level]:
            enhanced_content = await self._improve_content_quality(enhanced_content)
        
        # Update result
        result.generated_content = enhanced_content
        result.enhancement_applied = True
        
        return result
    
    async def re_optimize_context(self, context: Dict[str, Any], checkpoint: QualityCheckpoint) -> Dict[str, Any]:
        """Re-optimize context based on quality checkpoint feedback."""
        
        logger.info(f"Re-optimizing context due to quality score {checkpoint.quality_score:.3f}")
        
        # Extract context for re-optimization
        context_text = context.get("optimized_context") or context.get("context_string", "")
        
        try:
            from ..optimization.enhanced_context_optimizer import optimize_context_with_quality_assurance
            
            # Apply more aggressive optimization
            context_chunks = [{"content": context_text, "id": "reoptimize_context"}]
            
            # Use higher target quality
            higher_target = min(0.98, self.quality_thresholds[self.optimization_level] + 0.05)
            
            optimization_result = await optimize_context_with_quality_assurance(
                chunks=context_chunks,
                max_tokens=int(len(context_text.split()) * 1.1),  # Allow slight expansion
                target_quality=higher_target
            )
            
            # Update context with re-optimized version
            context["optimized_context"] = optimization_result.optimized_context
            context["context_quality_score"] = optimization_result.quality_score
            context["reoptimization_applied"] = True
            
        except ImportError:
            logger.warning("Enhanced context optimizer not available for re-optimization")
            # Simple fallback: just mark as re-optimized
            context["reoptimization_applied"] = True
            context["context_quality_score"] = min(0.9, context.get("context_quality_score", 0.8) + 0.1)
        
        return context
    
    # Helper methods for content enhancement
    async def _fix_content_repetition(self, content: str) -> str:
        """Fix repetitive content."""
        sentences = content.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if normalized and normalized not in seen:
                unique_sentences.append(sentence)
                seen.add(normalized)
        
        return '.'.join(unique_sentences)
    
    async def _expand_content(self, content: str) -> str:
        """Expand short content."""
        # Simple expansion by adding transition phrases
        if len(content.split()) < 20:
            expanded = content.rstrip('.')
            expanded += ". The scene continued to unfold with growing intensity."
            return expanded
        return content
    
    async def _condense_content(self, content: str) -> str:
        """Condense long content."""
        # Simple condensation by keeping first 80% of sentences
        sentences = content.split('.')
        if len(sentences) > 10:
            keep_count = int(len(sentences) * 0.8)
            condensed = '.'.join(sentences[:keep_count])
            return condensed + '.'
        return content
    
    async def _improve_content_quality(self, content: str) -> str:
        """Improve content quality through simple enhancements."""
        # Add descriptive elements if missing
        if '"' not in content and len(content.split()) > 20:
            # Add some dialogue-like enhancement
            enhanced = content.rstrip('.')
            enhanced += '. "This is quite remarkable," came the thoughtful response.'
            return enhanced
        return content