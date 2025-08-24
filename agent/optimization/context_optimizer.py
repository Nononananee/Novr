"""
Context Optimizer for LLM Token Limits
Addresses context truncation and token limit issues with intelligent optimization.
"""

import logging
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Context element priority levels."""
    CRITICAL = "critical"      # Must include (main characters, plot points)
    HIGH = "high"             # Important (character development, conflicts)
    MEDIUM = "medium"         # Supporting (descriptions, minor characters)
    LOW = "low"              # Background (atmosphere, minor details)


@dataclass
class ContextElement:
    """Individual context element with metadata."""
    content: str
    priority: ContextPriority
    token_count: int
    element_type: str  # "character", "plot", "setting", "dialogue", "narrative"
    importance_score: float
    recency_score: float
    relevance_score: float
    source_chunk_id: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of context optimization."""
    optimized_context: str
    total_tokens: int
    elements_included: int
    elements_excluded: int
    optimization_ratio: float
    quality_score: float


class ContextOptimizer:
    """Optimize context for LLM token limits while preserving quality."""
    
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 8000):
        """
        Initialize context optimizer.
        
        Args:
            model_name: LLM model name for token counting
            max_tokens: Maximum tokens allowed in context
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reserve_tokens = int(max_tokens * 0.2)  # Reserve 20% for generation
        self.available_tokens = max_tokens - self.reserve_tokens
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using default encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback estimation
            return len(text) // 4
    
    def optimize_context(
        self,
        context_elements: List[ContextElement],
        generation_type: str = "continuation",
        target_characters: List[str] = None,
        preserve_narrative_flow: bool = True
    ) -> OptimizationResult:
        """
        Optimize context to fit within token limits.
        
        Args:
            context_elements: List of context elements to optimize
            generation_type: Type of generation (affects prioritization)
            target_characters: Characters to prioritize
            preserve_narrative_flow: Whether to preserve narrative sequence
        
        Returns:
            Optimized context result
        """
        logger.info(f"Optimizing context: {len(context_elements)} elements, "
                   f"target: {self.available_tokens} tokens")
        
        # Calculate initial token count
        total_initial_tokens = sum(elem.token_count for elem in context_elements)
        
        if total_initial_tokens <= self.available_tokens:
            # No optimization needed
            optimized_context = self._build_context_string(context_elements)
            return OptimizationResult(
                optimized_context=optimized_context,
                total_tokens=total_initial_tokens,
                elements_included=len(context_elements),
                elements_excluded=0,
                optimization_ratio=1.0,
                quality_score=1.0
            )
        
        # Optimization needed
        logger.info(f"Context optimization required: {total_initial_tokens} -> {self.available_tokens} tokens")
        
        # Step 1: Prioritize elements
        prioritized_elements = self._prioritize_elements(
            context_elements,
            generation_type,
            target_characters
        )
        
        # Step 2: Select elements within token limit
        selected_elements = self._select_elements_within_limit(
            prioritized_elements,
            preserve_narrative_flow
        )
        
        # Step 3: Optimize individual elements if needed
        if sum(elem.token_count for elem in selected_elements) > self.available_tokens:
            selected_elements = self._compress_elements(selected_elements)
        
        # Step 4: Build optimized context
        optimized_context = self._build_context_string(selected_elements)
        final_token_count = self.count_tokens(optimized_context)
        
        # Calculate metrics
        optimization_ratio = len(selected_elements) / len(context_elements)
        quality_score = self._calculate_quality_score(selected_elements, context_elements)
        
        result = OptimizationResult(
            optimized_context=optimized_context,
            total_tokens=final_token_count,
            elements_included=len(selected_elements),
            elements_excluded=len(context_elements) - len(selected_elements),
            optimization_ratio=optimization_ratio,
            quality_score=quality_score
        )
        
        logger.info(f"Context optimized: {len(selected_elements)}/{len(context_elements)} elements, "
                   f"{final_token_count} tokens, quality: {quality_score:.3f}")
        
        return result
    
    def _prioritize_elements(
        self,
        elements: List[ContextElement],
        generation_type: str,
        target_characters: List[str] = None
    ) -> List[ContextElement]:
        """Prioritize context elements based on generation needs."""
        
        # Boost scores for relevant elements
        for element in elements:
            # Character relevance boost
            if target_characters:
                for character in target_characters:
                    if character.lower() in element.content.lower():
                        element.relevance_score += 0.3
            
            # Generation type boost
            if generation_type == "dialogue" and element.element_type == "dialogue":
                element.relevance_score += 0.2
            elif generation_type == "action" and element.element_type == "narrative":
                element.relevance_score += 0.2
            elif generation_type == "character_development" and element.element_type == "character":
                element.relevance_score += 0.3
        
        # Calculate combined priority score
        def priority_score(elem: ContextElement) -> float:
            priority_weights = {
                ContextPriority.CRITICAL: 1.0,
                ContextPriority.HIGH: 0.8,
                ContextPriority.MEDIUM: 0.6,
                ContextPriority.LOW: 0.4
            }
            
            base_score = priority_weights[elem.priority]
            combined_score = (
                base_score * 0.4 +
                elem.importance_score * 0.3 +
                elem.relevance_score * 0.2 +
                elem.recency_score * 0.1
            )
            
            return combined_score
        
        # Sort by priority score (highest first)
        return sorted(elements, key=priority_score, reverse=True)
    
    def _select_elements_within_limit(
        self,
        prioritized_elements: List[ContextElement],
        preserve_narrative_flow: bool
    ) -> List[ContextElement]:
        """Select elements that fit within token limit."""
        
        selected = []
        current_tokens = 0
        
        # Always include critical elements first
        critical_elements = [e for e in prioritized_elements if e.priority == ContextPriority.CRITICAL]
        
        for element in critical_elements:
            if current_tokens + element.token_count <= self.available_tokens:
                selected.append(element)
                current_tokens += element.token_count
            else:
                # Try to compress critical element
                compressed = self._compress_single_element(element)
                if current_tokens + compressed.token_count <= self.available_tokens:
                    selected.append(compressed)
                    current_tokens += compressed.token_count
        
        # Add other elements in priority order
        remaining_elements = [e for e in prioritized_elements if e.priority != ContextPriority.CRITICAL]
        
        for element in remaining_elements:
            if current_tokens + element.token_count <= self.available_tokens:
                selected.append(element)
                current_tokens += element.token_count
            else:
                # Check if we can fit a compressed version
                compressed = self._compress_single_element(element)
                if current_tokens + compressed.token_count <= self.available_tokens:
                    selected.append(compressed)
                    current_tokens += compressed.token_count
        
        # Sort by narrative order if requested
        if preserve_narrative_flow and selected:
            # Sort by source chunk ID or recency
            selected.sort(key=lambda x: (x.source_chunk_id or "", x.recency_score), reverse=True)
        
        return selected
    
    def _compress_elements(self, elements: List[ContextElement]) -> List[ContextElement]:
        """Compress elements to fit within token limit."""
        
        compressed = []
        
        for element in elements:
            compressed_element = self._compress_single_element(element)
            compressed.append(compressed_element)
        
        return compressed
    
    def _compress_single_element(self, element: ContextElement) -> ContextElement:
        """Compress a single context element."""
        
        content = element.content
        target_reduction = 0.3  # Reduce by 30%
        
        # Compression strategies
        if element.element_type == "dialogue":
            # Preserve dialogue, compress surrounding narrative
            compressed_content = self._compress_dialogue_context(content)
        elif element.element_type == "narrative":
            # Compress descriptive passages
            compressed_content = self._compress_narrative(content)
        elif element.element_type == "setting":
            # Compress setting descriptions
            compressed_content = self._compress_setting_description(content)
        else:
            # Generic compression
            compressed_content = self._generic_compression(content, target_reduction)
        
        # Create compressed element
        compressed_element = ContextElement(
            content=compressed_content,
            priority=element.priority,
            token_count=self.count_tokens(compressed_content),
            element_type=element.element_type,
            importance_score=element.importance_score * 0.9,  # Slight quality reduction
            recency_score=element.recency_score,
            relevance_score=element.relevance_score,
            source_chunk_id=element.source_chunk_id
        )
        
        return compressed_element
    
    def _compress_dialogue_context(self, content: str) -> str:
        """Compress dialogue while preserving speech."""
        
        # Split into dialogue and narrative parts
        lines = content.split('\n')
        compressed_lines = []
        
        for line in lines:
            if '"' in line or "'" in line:
                # Preserve dialogue lines
                compressed_lines.append(line)
            else:
                # Compress narrative lines
                if len(line.split()) > 10:
                    # Keep first and last few words
                    words = line.split()
                    compressed = ' '.join(words[:5] + ['...'] + words[-3:])
                    compressed_lines.append(compressed)
                else:
                    compressed_lines.append(line)
        
        return '\n'.join(compressed_lines)
    
    def _compress_narrative(self, content: str) -> str:
        """Compress narrative content."""
        
        sentences = re.split(r'[.!?]+', content)
        
        # Keep important sentences (with key narrative elements)
        important_keywords = [
            'suddenly', 'however', 'but', 'because', 'therefore',
            'realized', 'discovered', 'decided', 'remembered'
        ]
        
        compressed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Keep sentences with important keywords
            if any(keyword in sentence.lower() for keyword in important_keywords):
                compressed_sentences.append(sentence)
            elif len(compressed_sentences) == 0:  # Always keep first sentence
                compressed_sentences.append(sentence)
            elif len(sentence.split()) <= 8:  # Keep short sentences
                compressed_sentences.append(sentence)
        
        return '. '.join(compressed_sentences) + '.'
    
    def _compress_setting_description(self, content: str) -> str:
        """Compress setting descriptions."""
        
        # Remove excessive adjectives and keep core description
        words = content.split()
        
        # Remove redundant adjectives
        adjectives = ['beautiful', 'magnificent', 'gorgeous', 'stunning', 'amazing', 'incredible']
        filtered_words = [word for word in words if word.lower() not in adjectives]
        
        # If too much was removed, keep some adjectives
        if len(filtered_words) < len(words) * 0.7:
            # Keep first occurrence of each adjective type
            seen_adjectives = set()
            final_words = []
            
            for word in words:
                if word.lower() in adjectives:
                    if word.lower() not in seen_adjectives:
                        final_words.append(word)
                        seen_adjectives.add(word.lower())
                else:
                    final_words.append(word)
            
            return ' '.join(final_words)
        
        return ' '.join(filtered_words)
    
    def _generic_compression(self, content: str, target_reduction: float) -> str:
        """Generic content compression."""
        
        sentences = re.split(r'[.!?]+', content)
        target_count = int(len(sentences) * (1 - target_reduction))
        
        if target_count >= len(sentences):
            return content
        
        # Keep first, last, and most important sentences
        if target_count <= 2:
            return sentences[0] + '. ' + sentences[-1] + '.'
        
        # Keep first and last, distribute rest
        keep_sentences = [sentences[0]]
        
        # Select middle sentences based on length and keywords
        middle_sentences = sentences[1:-1]
        if middle_sentences and target_count > 2:
            # Sort by importance (sentences with key words)
            key_words = ['but', 'however', 'because', 'suddenly', 'realized', 'decided']
            
            def sentence_importance(sent: str) -> float:
                score = 0.0
                sent_lower = sent.lower()
                
                # Keyword bonus
                for keyword in key_words:
                    if keyword in sent_lower:
                        score += 0.5
                
                # Length penalty (prefer concise sentences)
                word_count = len(sent.split())
                if word_count < 15:
                    score += 0.2
                elif word_count > 25:
                    score -= 0.2
                
                return score
            
            sorted_middle = sorted(middle_sentences, key=sentence_importance, reverse=True)
            keep_sentences.extend(sorted_middle[:target_count-2])
        
        keep_sentences.append(sentences[-1])
        
        return '. '.join(s.strip() for s in keep_sentences if s.strip()) + '.'
    
    def _build_context_string(self, elements: List[ContextElement]) -> str:
        """Build context string from selected elements."""
        
        # Group elements by type for better organization
        grouped_elements = {
            "character": [],
            "plot": [],
            "setting": [],
            "dialogue": [],
            "narrative": []
        }
        
        for element in elements:
            element_type = element.element_type
            if element_type in grouped_elements:
                grouped_elements[element_type].append(element)
            else:
                grouped_elements["narrative"].append(element)
        
        # Build context sections
        context_parts = []
        
        # Character context
        if grouped_elements["character"]:
            context_parts.append("=== CHARACTER CONTEXT ===")
            for elem in grouped_elements["character"]:
                context_parts.append(elem.content)
            context_parts.append("")
        
        # Plot context
        if grouped_elements["plot"]:
            context_parts.append("=== PLOT CONTEXT ===")
            for elem in grouped_elements["plot"]:
                context_parts.append(elem.content)
            context_parts.append("")
        
        # Setting context
        if grouped_elements["setting"]:
            context_parts.append("=== SETTING CONTEXT ===")
            for elem in grouped_elements["setting"]:
                context_parts.append(elem.content)
            context_parts.append("")
        
        # Recent narrative and dialogue
        recent_elements = grouped_elements["narrative"] + grouped_elements["dialogue"]
        if recent_elements:
            # Sort by recency
            recent_elements.sort(key=lambda x: x.recency_score, reverse=True)
            
            context_parts.append("=== RECENT CONTEXT ===")
            for elem in recent_elements:
                context_parts.append(elem.content)
        
        return '\n'.join(context_parts)
    
    def _calculate_quality_score(
        self,
        selected_elements: List[ContextElement],
        original_elements: List[ContextElement]
    ) -> float:
        """Calculate quality score for optimized context."""
        
        if not original_elements:
            return 1.0
        
        # Factor 1: Element retention ratio
        retention_ratio = len(selected_elements) / len(original_elements)
        
        # Factor 2: Critical element preservation
        original_critical = sum(1 for e in original_elements if e.priority == ContextPriority.CRITICAL)
        selected_critical = sum(1 for e in selected_elements if e.priority == ContextPriority.CRITICAL)
        critical_preservation = selected_critical / original_critical if original_critical > 0 else 1.0
        
        # Factor 3: Importance score preservation
        original_importance = sum(e.importance_score for e in original_elements) / len(original_elements)
        selected_importance = sum(e.importance_score for e in selected_elements) / len(selected_elements) if selected_elements else 0
        importance_preservation = selected_importance / original_importance if original_importance > 0 else 1.0
        
        # Combined quality score
        quality_score = (
            retention_ratio * 0.3 +
            critical_preservation * 0.4 +
            importance_preservation * 0.3
        )
        
        return min(quality_score, 1.0)
    
    def create_context_elements_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        generation_context: Dict[str, Any] = None
    ) -> List[ContextElement]:
        """Create context elements from raw chunks."""
        
        elements = []
        generation_context = generation_context or {}
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            # Determine element type
            element_type = self._classify_element_type(content, metadata)
            
            # Calculate scores
            importance_score = metadata.get("importance_score", 0.5)
            recency_score = 1.0 - (i / len(chunks))  # More recent = higher score
            relevance_score = self._calculate_relevance_score(content, generation_context)
            
            # Determine priority
            priority = self._determine_priority(element_type, importance_score, metadata)
            
            element = ContextElement(
                content=content,
                priority=priority,
                token_count=self.count_tokens(content),
                element_type=element_type,
                importance_score=importance_score,
                recency_score=recency_score,
                relevance_score=relevance_score,
                source_chunk_id=chunk.get("chunk_id")
            )
            
            elements.append(element)
        
        return elements
    
    def _classify_element_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Classify context element type."""
        
        # Check metadata first
        if "element_type" in metadata:
            return metadata["element_type"]
        
        # Analyze content
        content_lower = content.lower()
        
        if '"' in content or "'" in content:
            return "dialogue"
        elif any(word in content_lower for word in ['character', 'personality', 'trait']):
            return "character"
        elif any(word in content_lower for word in ['plot', 'story', 'event', 'happen']):
            return "plot"
        elif any(word in content_lower for word in ['room', 'building', 'place', 'location']):
            return "setting"
        else:
            return "narrative"
    
    def _calculate_relevance_score(self, content: str, generation_context: Dict[str, Any]) -> float:
        """Calculate relevance score based on generation context."""
        
        score = 0.5  # Base score
        content_lower = content.lower()
        
        # Character relevance
        target_characters = generation_context.get("target_characters", [])
        for character in target_characters:
            if character.lower() in content_lower:
                score += 0.2
        
        # Plot thread relevance
        plot_threads = generation_context.get("plot_threads", [])
        for thread in plot_threads:
            if thread.lower() in content_lower:
                score += 0.1
        
        # Generation type relevance
        generation_type = generation_context.get("generation_type", "")
        if generation_type == "dialogue" and '"' in content:
            score += 0.2
        elif generation_type == "action" and any(word in content_lower for word in ['fight', 'run', 'battle']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_priority(self, element_type: str, importance_score: float, metadata: Dict[str, Any]) -> ContextPriority:
        """Determine priority level for context element."""
        
        # Check explicit priority in metadata
        if "priority" in metadata:
            priority_map = {
                "critical": ContextPriority.CRITICAL,
                "high": ContextPriority.HIGH,
                "medium": ContextPriority.MEDIUM,
                "low": ContextPriority.LOW
            }
            return priority_map.get(metadata["priority"], ContextPriority.MEDIUM)
        
        # Determine based on type and importance
        if element_type == "character" and importance_score >= 0.8:
            return ContextPriority.CRITICAL
        elif element_type == "plot" and importance_score >= 0.7:
            return ContextPriority.HIGH
        elif element_type == "dialogue" and importance_score >= 0.6:
            return ContextPriority.HIGH
        elif importance_score >= 0.8:
            return ContextPriority.HIGH
        elif importance_score >= 0.6:
            return ContextPriority.MEDIUM
        else:
            return ContextPriority.LOW


# Factory function
def create_context_optimizer(model_name: str = "gpt-4", max_tokens: int = 8000) -> ContextOptimizer:
    """Create context optimizer with specified configuration."""
    return ContextOptimizer(model_name, max_tokens)


# Example usage
async def main():
    """Example usage of context optimizer."""
    
    optimizer = create_context_optimizer(max_tokens=1000)  # Small limit for testing
    
    # Create sample context elements
    elements = [
        ContextElement(
            content="Emma is the main protagonist of our story. She is brave and determined.",
            priority=ContextPriority.CRITICAL,
            token_count=15,
            element_type="character",
            importance_score=0.9,
            recency_score=0.8,
            relevance_score=0.9
        ),
        ContextElement(
            content="The Victorian mansion loomed against the stormy sky with its Gothic architecture.",
            priority=ContextPriority.MEDIUM,
            token_count=12,
            element_type="setting",
            importance_score=0.6,
            recency_score=0.5,
            relevance_score=0.7
        ),
        ContextElement(
            content='"I must find the truth," Emma said with determination in her voice.',
            priority=ContextPriority.HIGH,
            token_count=13,
            element_type="dialogue",
            importance_score=0.8,
            recency_score=0.9,
            relevance_score=0.8
        )
    ]
    
    # Optimize context
    result = optimizer.optimize_context(
        context_elements=elements,
        generation_type="character_development",
        target_characters=["Emma"]
    )
    
    print(f"Optimization result:")
    print(f"  Original elements: {len(elements)}")
    print(f"  Optimized elements: {result.elements_included}")
    print(f"  Token count: {result.total_tokens}")
    print(f"  Quality score: {result.quality_score:.3f}")
    print(f"\nOptimized context:\n{result.optimized_context}")


if __name__ == "__main__":
    asyncio.run(main())