"""Metrics and performance tracking for the generation pipeline."""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .optimization import OptimizationLevel

logger = logging.getLogger(__name__)


@dataclass
class EnhancedGenerationMetrics:
    """Enhanced metrics for generation performance."""
    generation_time_ms: float
    context_quality_score: float
    chunking_performance: float
    content_coherence: float
    character_consistency: float
    narrative_flow: float
    overall_success_rate: float
    optimization_level: OptimizationLevel
    error_count: int = 0
    retry_count: int = 0


class MetricsCalculator:
    """Handles calculation and tracking of generation metrics."""
    
    def __init__(self):
        self.success_rate_history = []
        self.quality_checkpoints = []
    
    async def calculate_generation_metrics(self, 
                                         result,
                                         context: Dict[str, Any],
                                         generation_time: float,
                                         optimization_level: OptimizationLevel) -> EnhancedGenerationMetrics:
        """Calculate comprehensive generation metrics."""
        
        # Extract quality scores
        context_quality = context.get("context_quality_score", 0.0)
        content_quality = getattr(result, 'final_quality_score', 0.0)
        
        # Calculate chunking performance
        chunking_perf = context.get("chunking_performance", {})
        chunking_score = chunking_perf.get("overall_performance", 0.0)
        
        # Calculate coherence (simplified)
        coherence_score = (context_quality + content_quality) / 2
        
        # Calculate character consistency (placeholder)
        character_consistency = 0.85  # Would be calculated from actual character analysis
        
        # Calculate narrative flow (placeholder)
        narrative_flow = 0.80  # Would be calculated from narrative analysis
        
        # Calculate overall success rate
        success_factors = [
            context_quality,
            content_quality,
            chunking_score,
            coherence_score,
            character_consistency,
            narrative_flow
        ]
        
        overall_success = sum(success_factors) / len(success_factors)
        
        return EnhancedGenerationMetrics(
            generation_time_ms=generation_time,
            context_quality_score=context_quality,
            chunking_performance=chunking_score,
            content_coherence=coherence_score,
            character_consistency=character_consistency,
            narrative_flow=narrative_flow,
            overall_success_rate=overall_success,
            optimization_level=optimization_level,
            error_count=len([cp for cp in self.quality_checkpoints if not cp.passed]),
            retry_count=0  # Would track actual retries
        )
    
    def update_success_rate(self, current_success: float):
        """Update success rate tracking."""
        self.success_rate_history.append(current_success)
        
        # Keep only recent history
        if len(self.success_rate_history) > 100:
            self.success_rate_history = self.success_rate_history[-100:]
    
    def get_performance_summary(self, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.success_rate_history:
            return {"error": "No generation history available"}
        
        import statistics
        
        recent_success_rate = statistics.mean(self.success_rate_history[-10:]) if len(self.success_rate_history) >= 10 else statistics.mean(self.success_rate_history)
        overall_success_rate = statistics.mean(self.success_rate_history)
        
        return {
            "total_generations": len(self.success_rate_history),
            "overall_success_rate": overall_success_rate,
            "recent_success_rate": recent_success_rate,
            "optimization_level": optimization_level.value,
            "quality_checkpoints_performed": len(self.quality_checkpoints),
            "successful_checkpoints": len([cp for cp in self.quality_checkpoints if cp.passed]),
            "average_quality_score": statistics.mean([cp.quality_score for cp in self.quality_checkpoints]) if self.quality_checkpoints else 0.0
        }


async def analyze_generation_performance(results: List[Any]) -> Dict[str, Any]:
    """
    Analyze performance of multiple generation results.
    
    Args:
        results: List of generation results to analyze
        
    Returns:
        Performance analysis
    """
    if not results:
        return {"error": "No results provided"}
    
    # Extract metrics from results
    success_rates = []
    quality_scores = []
    generation_times = []
    
    for result in results:
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            success_rates.append(metrics.overall_success_rate)
            quality_scores.append(metrics.context_quality_score)
            generation_times.append(metrics.generation_time_ms)
    
    if not success_rates:
        return {"error": "No enhanced metrics found in results"}
    
    import statistics
    
    return {
        "total_results": len(results),
        "average_success_rate": statistics.mean(success_rates),
        "average_quality_score": statistics.mean(quality_scores),
        "average_generation_time_ms": statistics.mean(generation_times),
        "success_rate_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
        "results_above_90_percent": sum(1 for rate in success_rates if rate >= 0.9),
        "performance_grade": "A" if statistics.mean(success_rates) >= 0.9 else "B" if statistics.mean(success_rates) >= 0.8 else "C"
    }