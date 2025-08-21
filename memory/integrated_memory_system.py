# src/rag/integrated_memory_system.py
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import json
from datetime import datetime, timedelta

from .memory_manager import NovelMemoryManager, MemoryChunk, MemoryType, MemoryPriority
from ..data.chunker import NovelChunker, ChunkingStrategy
from ..novel.coherence.consistency_enforcer import LongTermConsistencyManager
from ..novel.character_management.repository import CharacterRepository

@dataclass
class GenerationContext:
    """Complete context for novel generation including memory and consistency"""
    current_chapter: int
    current_word_count: int
    target_characters: List[str]
    active_plot_threads: List[str]
    generation_intent: str  # "continue", "dialogue", "scene", "character_intro"
    tone_requirements: Dict[str, float]
    constraints: Optional[Any] = None
    pov_character: Optional[str] = None
    scene_location: Optional[str] = None

@dataclass
class GenerationResult:
    """Result of generation with memory and consistency metadata"""
    generated_content: str
    memory_chunks_created: List[str]  # Chunk IDs
    consistency_issues: List[Any]
    memory_tokens_used: int
    generation_metadata: Dict[str, Any]
    quality_score: float = 0.0
    originality_score: float = 0.0

class IntegratedNovelMemorySystem:
    """Integrated system combining memory management, chunking, and consistency"""
    
    def __init__(self, 
                 vectorstore_client=None,
                 llm_client=None,
                 character_repo=None,
                 max_memory_tokens: int = 32000,
                 consistency_level: str = "high"):
        
        # Core components
        self.memory_manager = NovelMemoryManager(
            max_context_tokens=max_memory_tokens,
            vectorstore_client=vectorstore_client
        )
        self.chunker = NovelChunker()
        self.consistency_manager = LongTermConsistencyManager(
            memory_manager=self.memory_manager,
            character_repo=character_repo
        )
        
        # Clients
        self.vectorstore = vectorstore_client
        self.llm_client = llm_client
        self.character_repo = character_repo
        
        # Configuration
        self.consistency_level = consistency_level
        self.max_memory_tokens = max_memory_tokens
        
        # Statistics
        self.stats = {
            'total_generations': 0,
            'consistency_issues_found': 0,
            'memory_chunks_created': 0,
            'average_context_size': 0
        }

    async def generate_with_full_context(self, 
                                       generation_context: GenerationContext,
                                       prompt: str = None) -> GenerationResult:
        """Generate content with full memory and consistency context"""
        
        try:
            # Step 1: Build optimal memory context
            memory_context = await self._build_memory_context(generation_context)
            
            # Step 2: Check consistency before generation
            consistency_ok, pre_issues = await self.consistency_manager.check_consistency_before_generation(
                context=memory_context,
                generation_intent=generation_context.__dict__
            )
            
            if not consistency_ok and self.consistency_level == "strict":
                return GenerationResult(
                    generated_content="",
                    memory_chunks_created=[],
                    consistency_issues=pre_issues,
                    memory_tokens_used=len(memory_context.split()),
                    generation_metadata={"error": "Critical consistency issues found"},
                    quality_score=0.0
                )
            
            # Step 3: Generate content with LLM
            generated_content = await self._generate_content_with_llm(
                context=memory_context,
                generation_context=generation_context,
                prompt=prompt
            )
            
            # Step 4: Validate generated content for consistency
            content_valid, post_issues = await self.consistency_manager.validate_generated_content(
                new_content=generated_content,
                context=memory_context
            )
            
            # Step 5: Fix consistency issues if possible
            if not content_valid:
                generated_content, remaining_issues = await self.consistency_manager.fix_consistency_issues(
                    content=generated_content,
                    issues=post_issues
                )
            else:
                remaining_issues = post_issues
            
            # Step 6: Process new content into memory chunks
            new_chunks = await self._process_new_content_into_memory(
                content=generated_content,
                generation_context=generation_context
            )
            
            # Step 7: Update memory system
            for chunk in new_chunks:
                await self.memory_manager.store_memory_chunk(chunk)
            
            # Step 8: Calculate quality metrics
            quality_score = await self._calculate_content_quality(generated_content, memory_context)
            originality_score = await self._calculate_originality_score(generated_content)
            
            # Update statistics
            self._update_statistics(memory_context, new_chunks, remaining_issues)
            
            return GenerationResult(
                generated_content=generated_content,
                memory_chunks_created=[chunk.id for chunk in new_chunks],
                consistency_issues=remaining_issues,
                memory_tokens_used=len(memory_context.split()),
                generation_metadata={
                    "context_chunks_used": len(memory_context.split("===")),
                    "consistency_checks_passed": content_valid,
                    "auto_fixes_applied": len(post_issues) - len(remaining_issues)
                },
                quality_score=quality_score,
                originality_score=originality_score
            )
            
        except Exception as e:
            return GenerationResult(
                generated_content="",
                memory_chunks_created=[],
                consistency_issues=[],
                memory_tokens_used=0,
                generation_metadata={"error": str(e)},
                quality_score=0.0,
                originality_score=0.0
            )

    async def _build_memory_context(self, generation_context: GenerationContext) -> str:
        """Build comprehensive memory context for generation"""
        
        # Calculate current position in novel
        current_position = generation_context.current_word_count
        
        # Build context using memory manager
        memory_context = await self.memory_manager.build_context_for_generation(
            current_position=current_position,
            generation_type=generation_context.generation_intent,
            target_characters=generation_context.target_characters,
            plot_threads=generation_context.active_plot_threads
        )
        
        # Add character-specific context if POV character specified
        if generation_context.pov_character:
            character_context = await self._build_character_specific_context(
                generation_context.pov_character
            )
            memory_context = f"{memory_context}\n\n=== POV CHARACTER CONTEXT ===\n{character_context}"
        
        # Add scene-specific context if location specified
        if generation_context.scene_location:
            location_context = await self._build_location_context(
                generation_context.scene_location
            )
            memory_context = f"{memory_context}\n\n=== SCENE LOCATION ===\n{location_context}"
        
        return memory_context

    async def _generate_content_with_llm(self, 
                                       context: str, 
                                       generation_context: GenerationContext,
                                       prompt: str = None) -> str:
        """Generate content using LLM with full context"""
        
        if not self.llm_client:
            return "LLM client not configured"
        
        # Build generation prompt
        system_prompt = self._build_system_prompt(generation_context)
        
        if prompt:
            user_prompt = f"{context}\n\n=== GENERATION REQUEST ===\n{prompt}"
        else:
            user_prompt = f"{context}\n\n=== GENERATION REQUEST ===\nContinue the novel from this point, maintaining consistency with all established elements."
        
        # Add constraints if specified
        if generation_context.constraints:
            constraint_text = self._format_constraints(generation_context.constraints)
            user_prompt = f"{user_prompt}\n\n=== CREATIVE CONSTRAINTS ===\n{constraint_text}"
        
        try:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,  # Configurable
                temperature=0.8,  # Configurable based on creativity needs
            )
            return response.strip()
        except Exception as e:
            return f"Generation failed: {str(e)}"

    async def _process_new_content_into_memory(self, 
                                             content: str, 
                                             generation_context: GenerationContext) -> List[MemoryChunk]:
        """Process newly generated content into memory chunks"""
        
        # Use adaptive chunking based on content characteristics
        novel_chunks = self.chunker.adaptive_chunking(
            text=content,
            context={
                'characters': generation_context.target_characters,
                'chapter': generation_context.current_chapter,
                'generation_type': generation_context.generation_intent
            }
        )
        
        # Convert to memory chunks with enhanced metadata
        memory_chunks = []
        for i, chunk in enumerate(novel_chunks):
            
            # Determine memory type based on chunk analysis
            memory_type = self._determine_memory_type(chunk)
            
            # Calculate priority based on content and context
            priority = self._calculate_chunk_priority(chunk, generation_context)
            
            # Create memory chunk
            memory_chunk = MemoryChunk(
                id=f"gen_{generation_context.current_chapter}_{i}_{datetime.now().timestamp()}",
                content=chunk.content,
                memory_type=memory_type,
                priority=priority,
                chapter_range=(generation_context.current_chapter, generation_context.current_chapter),
                word_range=(generation_context.current_word_count, 
                          generation_context.current_word_count + len(chunk.content.split())),
                characters_involved=chunk.narrative_elements.get('primary_characters', []),
                plot_threads=generation_context.active_plot_threads,
                tags=self._generate_chunk_tags(chunk, generation_context),
                embedding=None  # Will be generated when stored
            )
            
            memory_chunks.append(memory_chunk)
        
        return memory_chunks

    def _determine_memory_type(self, chunk) -> MemoryType:
        """Determine the memory type for a chunk"""
        
        chunk_type = chunk.chunk_type.lower()
        
        if 'dialogue' in chunk_type:
            return MemoryType.CHARACTER_STATE
        elif 'character' in chunk_type:
            return MemoryType.CHARACTER_STATE
        elif 'temporal' in chunk_type:
            return MemoryType.TIMELINE_EVENT
        elif chunk.importance_score > 0.8:
            return MemoryType.PLOT_POINT
        else:
            return MemoryType.WORLD_ELEMENT

    def _calculate_chunk_priority(self, chunk, generation_context: GenerationContext) -> MemoryPriority:
        """Calculate priority for memory chunk"""
        
        # Base priority on importance score
        importance = chunk.importance_score
        
        # Boost priority for key elements
        if any(char in generation_context.target_characters for char in 
               chunk.narrative_elements.get('primary_characters', [])):
            importance += 0.2
        
        if generation_context.generation_intent in ['climax', 'resolution', 'major_plot_point']:
            importance += 0.3
        
        # Convert to priority enum
        if importance >= 0.9:
            return MemoryPriority.CRITICAL
        elif importance >= 0.7:
            return MemoryPriority.HIGH
        elif importance >= 0.5:
            return MemoryPriority.MEDIUM
        else:
            return MemoryPriority.LOW

    def _generate_chunk_tags(self, chunk, generation_context: GenerationContext) -> List[str]:
        """Generate tags for memory chunk"""
        
        tags = []
        
        # Add chunk-based tags
        if hasattr(chunk, 'tags'):
            tags.extend(chunk.tags)
        
        # Add context-based tags
        tags.append(f"chapter_{generation_context.current_chapter}")
        tags.append(f"intent_{generation_context.generation_intent}")
        
        if generation_context.pov_character:
            tags.append(f"pov_{generation_context.pov_character}")
        
        if generation_context.scene_location:
            tags.append(f"location_{generation_context.scene_location}")
        
        # Add content-based tags
        content_lower = chunk.content.lower()
        if '"' in chunk.content or "'" in chunk.content:
            tags.append("dialogue")
        
        if any(word in content_lower for word in ['fight', 'battle', 'combat']):
            tags.append("action")
        
        if any(word in content_lower for word in ['love', 'kiss', 'romance']):
            tags.append("romance")
        
        if any(word in content_lower for word in ['magic', 'spell', 'enchant']):
            tags.append("magic")
        
        return list(set(tags))  # Remove duplicates

    async def _build_character_specific_context(self, character_id: str) -> str:
        """Build context specific to a POV character"""
        
        if not self.character_repo:
            return f"POV Character: {character_id}"
        
        try:
            character = await self.character_repo.get_character(character_id)
            if not character:
                return f"POV Character: {character_id} (not found in repository)"
            
            context_parts = [
                f"POV CHARACTER: {character.name}",
                f"Current emotional state: {character.get_current_emotional_state()}",
                f"Key personality traits: {', '.join(character.personality.get('traits', []))}",
                f"Current motivations: {', '.join(character.motivations)}",
                f"Speech patterns: {', '.join(character.speech_patterns)}"
            ]
            
            # Add recent character developments
            recent_developments = await self._get_recent_character_developments(character_id)
            if recent_developments:
                context_parts.append(f"Recent developments: {recent_developments}")
            
            return '\n'.join(context_parts)
            
        except Exception as e:
            return f"POV Character: {character_id} (error loading context: {e})"

    async def _build_location_context(self, location: str) -> str:
        """Build context for scene location"""
        
        # Get location-specific memories
        location_memories = await self._get_location_memories(location)
        
        context_parts = [f"LOCATION: {location}"]
        
        if location_memories:
            context_parts.append("Previous scenes at this location:")
            for memory in location_memories[:3]:  # Last 3 scenes at this location
                context_parts.append(f"- {memory['summary']}")
        
        return '\n'.join(context_parts)

    async def _calculate_content_quality(self, content: str, context: str) -> float:
        """Calculate quality score for generated content"""
        
        quality_factors = {
            'length_appropriate': len(content.split()) > 50,  # Not too short
            'character_consistency': True,  # Would implement actual check
            'narrative_flow': True,  # Would implement actual check
            'dialogue_quality': '"' in content,  # Has dialogue (if expected)
            'descriptive_balance': True  # Would implement actual check
        }
        
        # Calculate weighted score
        weights = {'length_appropriate': 0.2, 'character_consistency': 0.3, 
                  'narrative_flow': 0.3, 'dialogue_quality': 0.1, 'descriptive_balance': 0.1}
        
        score = sum(weights[factor] for factor, passed in quality_factors.items() if passed)
        return min(score, 1.0)

    async def _calculate_originality_score(self, content: str) -> float:
        """Calculate originality score for content"""
        
        # Simple originality metrics (would be more sophisticated)
        originality_factors = {
            'unique_phrases': len(set(content.lower().split())) / len(content.split()) if content else 0,
            'avoiding_cliches': not any(cliche in content.lower() 
                                      for cliche in ['suddenly', 'it was a dark', 'little did they know']),
            'creative_language': any(word in content.lower() 
                                   for word in ['shimmer', 'cascade', 'whisper', 'dance'])
        }
        
        # Weight factors
        weights = {'unique_phrases': 0.4, 'avoiding_cliches': 0.4, 'creative_language': 0.2}
        
        score = sum(weights[factor] * (1.0 if passed else 0.0) if isinstance(passed, bool) 
                   else weights[factor] * passed
                   for factor, passed in originality_factors.items())
        
        return min(score, 1.0)

    def _build_system_prompt(self, generation_context: GenerationContext) -> str:
        """Build system prompt for LLM generation"""
        
        prompt_parts = [
            "You are an expert novelist generating content for a long-form novel.",
            "Your task is to continue the story while maintaining perfect consistency with all established elements.",
            "",
            f"Current context: Chapter {generation_context.current_chapter}, ~{generation_context.current_word_count} words written",
            f"Generation intent: {generation_context.generation_intent}",
        ]
        
        if generation_context.target_characters:
            prompt_parts.append(f"Focus characters: {', '.join(generation_context.target_characters)}")
        
        if generation_context.pov_character:
            prompt_parts.append(f"Point of view: {generation_context.pov_character}")
        
        if generation_context.tone_requirements:
            tone_desc = ', '.join(f"{tone}: {score:.1f}" 
                                for tone, score in generation_context.tone_requirements.items())
            prompt_parts.append(f"Tone requirements: {tone_desc}")
        
        prompt_parts.extend([
            "",
            "Guidelines:",
            "- Maintain absolute consistency with established character traits, relationships, and world rules",
            "- Follow the narrative tone and style established in previous content",
            "- Avoid clichés and predictable plot developments",
            "- Show don't tell, use vivid sensory details",
            "- Ensure dialogue reflects each character's unique voice",
            "- Keep plot progression logical and motivated"
        ])
        
        return '\n'.join(prompt_parts)

    def _format_constraints(self, constraints) -> str:
        """Format creative constraints for LLM prompt"""
        
        if not constraints:
            return "No specific constraints"
        
        constraint_parts = []
        
        if hasattr(constraints, 'forbidden_tropes'):
            constraint_parts.append(f"Avoid these tropes: {', '.join(constraints.forbidden_tropes)}")
        
        if hasattr(constraints, 'style_requirements'):
            constraint_parts.append(f"Style requirements: {constraints.style_requirements}")
        
        if hasattr(constraints, 'originality_requirements'):
            constraint_parts.append(f"Originality level required: {constraints.originality_requirements}")
        
        return '\n'.join(constraint_parts) if constraint_parts else "Standard creative constraints apply"

    async def _get_recent_character_developments(self, character_id: str) -> str:
        """Get recent developments for a character"""
        
        # Get character-specific memory chunks from recent content
        if character_id in self.memory_manager.character_memories:
            recent_chunk_ids = self.memory_manager.character_memories[character_id][-3:]  # Last 3 chunks
            
            developments = []
            for chunk_id in recent_chunk_ids:
                if chunk_id in self.memory_manager.memory_chunks:
                    chunk = self.memory_manager.memory_chunks[chunk_id]
                    # Extract key developments (simplified)
                    developments.append(chunk.content[:100] + "...")
            
            return ' | '.join(developments)
        
        return ""

    async def _get_location_memories(self, location: str) -> List[Dict]:
        """Get memories associated with a location"""
        
        # Search for chunks with location tags
        location_memories = []
        
        for chunk in self.memory_manager.memory_chunks.values():
            if f"location_{location}" in chunk.tags:
                location_memories.append({
                    'summary': chunk.content[:150] + "...",
                    'chapter': chunk.chapter_range[0]
                })
        
        # Sort by recency
        location_memories.sort(key=lambda x: x['chapter'], reverse=True)
        
        return location_memories

    def _update_statistics(self, memory_context: str, new_chunks: List[MemoryChunk], issues: List[Any]):
        """Update system statistics"""
        
        self.stats['total_generations'] += 1
        self.stats['consistency_issues_found'] += len(issues)
        self.stats['memory_chunks_created'] += len(new_chunks)
        
        # Update average context size
        context_size = len(memory_context.split())
        total_gens = self.stats['total_generations']
        current_avg = self.stats['average_context_size']
        self.stats['average_context_size'] = ((current_avg * (total_gens - 1)) + context_size) / total_gens

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        memory_stats = self.memory_manager.get_memory_statistics()
        consistency_stats = self.consistency_manager.get_consistency_report()
        
        return {
            'memory_system': memory_stats,
            'consistency_system': consistency_stats,
            'generation_system': self.stats,
            'overall_health': {
                'memory_utilization': memory_stats['memory_usage']['active_context_chunks'] / 
                                    (memory_stats['total_chunks'] or 1),
                'consistency_compliance': 1.0 - (consistency_stats['critical_issues'] / 
                                                (consistency_stats['active_issues'] or 1)),
                'average_generation_quality': 0.85  # Would track actual quality scores
            }
        }

    async def cleanup_old_memories(self, days_old: int = 30, keep_critical: bool = True):
        """Clean up old, less relevant memories to manage storage"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        chunks_to_remove = []
        for chunk_id, chunk in self.memory_manager.memory_chunks.items():
            
            # Keep critical memories regardless of age
            if keep_critical and chunk.priority == MemoryPriority.CRITICAL:
                continue
            
            # Remove old, rarely accessed chunks
            if (chunk.last_accessed < cutoff_date and 
                chunk.access_count < 3 and 
                chunk.priority in [MemoryPriority.LOW, MemoryPriority.CONTEXTUAL]):
                
                chunks_to_remove.append(chunk_id)
        
        # Remove from memory and update indices
        for chunk_id in chunks_to_remove:
            if chunk_id in self.memory_manager.memory_chunks:
                del self.memory_manager.memory_chunks[chunk_id]
        
        # Update character and plot thread indices
        await self._rebuild_memory_indices()
        
        return len(chunks_to_remove)

    async def _rebuild_memory_indices(self):
        """Rebuild memory indices after cleanup"""
        
        # Clear existing indices
        self.memory_manager.character_memories.clear()
        self.memory_manager.plot_thread_memories.clear()
        self.memory_manager.chapter_index.clear()
        
        # Rebuild from remaining chunks
        for chunk_id, chunk in self.memory_manager.memory_chunks.items():
            
            # Update character index
            for character in chunk.characters_involved:
                if character not in self.memory_manager.character_memories:
                    self.memory_manager.character_memories[character] = []
                self.memory_manager.character_memories[character].append(chunk_id)
            
            # Update plot thread index
            for thread in chunk.plot_threads:
                if thread not in self.memory_manager.plot_thread_memories:
                    self.memory_manager.plot_thread_memories[thread] = []
                self.memory_manager.plot_thread_memories[thread].append(chunk_id)
            
            # Update chapter index
            for chapter in range(chunk.chapter_range[0], chunk.chapter_range[1] + 1):
                if chapter not in self.memory_manager.chapter_index:
                    self.memory_manager.chapter_index[chapter] = []
                self.memory_manager.chapter_index[chapter].append(chunk_id)