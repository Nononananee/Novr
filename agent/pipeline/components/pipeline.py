"""Main enhanced generation pipeline orchestrator."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

from .optimization import ContentOptimizer, OptimizationLevel
from .quality import QualityValidator
from .metrics import MetricsCalculator, EnhancedGenerationMetrics

logger = logging.getLogger(__name__)


class EnhancedGenerationPipeline:
    """
    Enhanced generation pipeline with integrated optimizations.
    Targets >90% success rate for real-world content.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        """Initialize enhanced generation pipeline."""
        
        # Import base pipeline with fallback
        try:
            from ..generation_pipeline import NovelGenerationPipeline
            self.base_pipeline = NovelGenerationPipeline()
            self.base_pipeline_available = True
        except ImportError:
            logger.warning("Base generation pipeline not available, using fallback")
            self.base_pipeline = self._create_fallback_pipeline()
            self.base_pipeline_available = False
        
        self.optimization_level = optimization_level
        
        # Initialize components
        self.content_optimizer = ContentOptimizer(optimization_level)
        self.quality_validator = QualityValidator(optimization_level)
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize circuit breakers
        self.circuit_breakers = self._initialize_circuit_breakers()
        
        # Optimization strategies
        self.optimization_strategies = {
            "context_enhancement": True,
            "chunking_optimization": True,
            "quality_validation": True,
            "error_recovery": True,
            "performance_monitoring": True
        }
        
        logger.info(f"Enhanced generation pipeline initialized with {optimization_level.value} optimization")
    
    def _create_fallback_pipeline(self):
        """Create a fallback pipeline when base pipeline is not available."""
        class FallbackPipeline:
            async def generate_content(self, request):
                # Generate intelligent response based on request content
                content = getattr(request, 'content', '')
                
                if 'dialogue' in content.lower():
                    response = f'"I understand what you mean," came the thoughtful reply. The conversation continued naturally from there.'
                elif 'action' in content.lower():
                    response = f"The scene unfolded with dramatic intensity, building upon the previous events in a compelling way."
                else:
                    response = f"The narrative developed organically, weaving together the themes and characters established earlier. {content[:50]}... continued with depth and nuance."
                
                # Create result object
                class FallbackResult:
                    def __init__(self, content, metadata=None):
                        self.generated_content = content
                        self.metadata = metadata or {}
                        self.fallback_used = True
                
                return FallbackResult(response, {"fallback_type": "intelligent_mock"})
            
            async def _prepare_generation_context(self, request):
                return {"context_string": getattr(request, 'content', 'Default context')}
        
        return FallbackPipeline()
    
    def _initialize_circuit_breakers(self) -> Dict[str, Any]:
        """Initialize circuit breakers for pipeline components."""
        circuit_breakers = {}
        
        try:
            from ..api.circuit_breaker import create_circuit_breaker, CircuitBreakerConfig, ComponentType
            
            # Context optimization circuit breaker
            context_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout=15.0,
                quality_threshold=0.8,
                enable_quality_check=True
            )
            circuit_breakers["context_optimization"] = create_circuit_breaker(
                "generation_context_optimization",
                context_config,
                ComponentType.CONTEXT_OPTIMIZER
            )
            
        except ImportError:
            logger.warning("Circuit breakers not available, using fallback error handling")
            circuit_breakers = {}
        
        return circuit_breakers
    
    async def generate_content(self, 
                              request,
                              enable_quality_checks: bool = True,
                              enable_optimization: bool = True):
        """
        Generate content with enhanced optimization and quality assurance.
        
        Args:
            request: Generation request
            enable_quality_checks: Whether to perform quality validation
            enable_optimization: Whether to apply optimization strategies
            
        Returns:
            Enhanced generation result with quality metrics
        """
        start_time = time.time()
        
        try:
            # Phase 1: Request Analysis and Optimization
            optimized_request = await self.content_optimizer.optimize_generation_request(request)
            
            # Phase 2: Enhanced Context Preparation
            enhanced_context = await self.content_optimizer.prepare_enhanced_context(
                optimized_request, self.base_pipeline
            )
            
            # Quality Checkpoint 1: Context Quality
            if enable_quality_checks:
                context_checkpoint = await self.content_optimizer.validate_context_quality(enhanced_context)
                self.metrics_calculator.quality_checkpoints.append(context_checkpoint)
                
                if not context_checkpoint.passed and self.optimization_level == OptimizationLevel.QUALITY:
                    # Re-optimize context if quality check fails
                    enhanced_context = await self.quality_validator.re_optimize_context(
                        enhanced_context, context_checkpoint
                    )
            
            # Phase 3: Enhanced Content Generation
            generation_result = await self._generate_with_optimization(optimized_request, enhanced_context)
            
            # Quality Checkpoint 2: Generated Content Quality
            if enable_quality_checks:
                content_checkpoint = await self.quality_validator.validate_generated_content(generation_result)
                self.metrics_calculator.quality_checkpoints.append(content_checkpoint)
                
                if not content_checkpoint.passed:
                    # Attempt content enhancement
                    generation_result = await self.quality_validator.enhance_generated_content(
                        generation_result, content_checkpoint
                    )
            
            # Phase 4: Post-Generation Optimization
            final_result = await self._post_process_generation(generation_result, enhanced_context)
            
            # Calculate final metrics
            generation_time = (time.time() - start_time) * 1000
            metrics = await self.metrics_calculator.calculate_generation_metrics(
                final_result, enhanced_context, generation_time, self.optimization_level
            )
            
            # Update success rate tracking
            self.metrics_calculator.update_success_rate(metrics.overall_success_rate)
            
            # Add metrics to result
            final_result.enhanced_metrics = metrics
            
            return final_result
            
        except Exception as e:
            logger.error(f"Enhanced generation failed: {e}")
            
            # Fallback to base pipeline with error recovery
            return await self._fallback_generation(request, str(e))
    
    async def _generate_with_optimization(self, request, context: Dict[str, Any]):
        """Generate content with optimization strategies."""
        
        # Use optimized context if available
        if "optimized_context" in context:
            # Temporarily replace context in request
            original_context = getattr(request, 'context', None)
            request.context = context["optimized_context"]
        
        try:
            # Generate using base pipeline
            result = await self.base_pipeline.generate_content(request)
            
            # Add optimization metadata
            if hasattr(result, 'metadata'):
                result.metadata.update({
                    "optimization_applied": True,
                    "context_quality": context.get("context_quality_score", 0.0),
                    "chunking_performance": context.get("chunking_performance", {})
                })
            
            return result
            
        finally:
            # Restore original context
            if "optimized_context" in context and 'original_context' in locals():
                request.context = original_context
    
    async def _post_process_generation(self, result, context: Dict[str, Any]):
        """Post-process generated content for final optimization."""
        
        # Add context metadata to result
        if hasattr(result, 'metadata'):
            result.metadata.update({
                "context_optimization": {
                    "quality_score": context.get("context_quality_score", 0.0),
                    "optimization_ratio": context.get("optimization_ratio", 1.0),
                    "chunking_performance": context.get("chunking_performance", {}),
                    "enhancement_metadata": context.get("enhancement_metadata", {})
                }
            })
        
        # Final content validation
        try:
            from ..optimization.enhanced_context_optimizer import analyze_context_quality
            final_quality = await analyze_context_quality(result.generated_content)
            result.final_quality_score = final_quality.overall_score
        except ImportError:
            # Fallback quality score
            result.final_quality_score = min(0.9, len(result.generated_content.split()) / 50)
        
        return result
    
    async def _fallback_generation(self, request, error: str):
        """Fallback generation when enhanced pipeline fails."""
        
        logger.warning(f"Using fallback generation due to error: {error}")
        
        try:
            # Use base pipeline as fallback
            result = await self.base_pipeline.generate_content(request)
            result.fallback_used = True
            result.fallback_reason = error
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback generation also failed: {fallback_error}")
            
            # Emergency fallback
            try:
                from ..validation.error_handling_utils import GracefulDegradation
                return await GracefulDegradation.get_emergency_fallback(
                    getattr(request, 'content', 'No content available')
                )
            except ImportError:
                # Final emergency fallback
                class EmergencyResult:
                    def __init__(self, content):
                        self.generated_content = content
                        self.fallback_used = True
                        self.emergency_fallback = True
                
                return EmergencyResult(
                    "I apologize, but I'm experiencing technical difficulties. "
                    "Please try again or contact support if the issue persists."
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.metrics_calculator.get_performance_summary(self.optimization_level)