# src/data/chunker.py - Advanced Novel Chunking
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from nltk.tokenize import sent_tokenize
import spacy

class ChunkingStrategy(Enum):
    NARRATIVE_FLOW = "narrative_flow"      # Follow story beats
    CHARACTER_FOCUS = "character_focus"    # Group by character scenes  
    DIALOGUE_SCENES = "dialogue_scenes"    # Separate dialogue from narrative
    TEMPORAL_BREAKS = "temporal_breaks"    # Split on time jumps
    EMOTIONAL_BEATS = "emotional_beats"    # Split on emotional shifts
    POV_CHANGES = "pov_changes"           # Split on perspective changes

@dataclass
class NovelChunk:
    id: str
    content: str
    strategy_used: ChunkingStrategy
    narrative_elements: Dict[str, any]  # Characters, emotions, timeline, etc.
    chunk_type: str  # "scene", "dialogue", "description", "transition"
    importance_score: float
    dependency_chunks: List[str]  # IDs of related chunks
    
class NovelChunker:
    def __init__(self):
        # Load NLP model for analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def chunk_by_narrative_flow(self, text: str, max_chunk_size: int = 1000) -> List[NovelChunk]:
        """Chunk based on narrative flow and story beats"""
        
        # Detect narrative beats
        beats = self._detect_story_beats(text)
        chunks = []
        
        for i, beat in enumerate(beats):
            # If beat is too long, sub-chunk it
            if len(beat['content'].split()) > max_chunk_size:
                sub_chunks = self._sub_chunk_narrative_beat(beat, max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunk = NovelChunk(
                    id=f"narrative_{i}",
                    content=beat['content'],
                    strategy_used=ChunkingStrategy.NARRATIVE_FLOW,
                    narrative_elements=beat['elements'],
                    chunk_type=beat['type'],
                    importance_score=beat['importance'],
                    dependency_chunks=beat.get('dependencies', [])
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_character_focus(self, text: str, characters: List[str]) -> List[NovelChunk]:
        """Chunk based on character presence and focus"""
        
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_characters = set()
        chunks = []
        chunk_id = 0
        
        for paragraph in paragraphs:
            # Analyze character presence in paragraph
            para_characters = self._extract_characters_from_text(paragraph, characters)
            
            # If significant character change, start new chunk
            if para_characters and not para_characters.intersection(current_characters) and current_chunk:
                # Finalize current chunk
                chunk = self._create_character_chunk(current_chunk, current_characters, chunk_id)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [paragraph]
                current_characters = para_characters
                chunk_id += 1
            else:
                current_chunk.append(paragraph)
                current_characters.update(para_characters)
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_character_chunk(current_chunk, current_characters, chunk_id)
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_dialogue_scenes(self, text: str) -> List[NovelChunk]:
        """Separate dialogue-heavy sections from narrative descriptions"""
        
        chunks = []
        current_section = []
        section_type = None  # "dialogue" or "narrative"
        chunk_id = 0
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            para_type = self._classify_paragraph_type(paragraph)
            
            if section_type and para_type != section_type and current_section:
                # Create chunk from current section
                chunk = NovelChunk(
                    id=f"dialogue_scene_{chunk_id}",
                    content='\n\n'.join(current_section),
                    strategy_used=ChunkingStrategy.DIALOGUE_SCENES,
                    narrative_elements={'type': section_type, 'dialogue_ratio': self._calculate_dialogue_ratio('\n\n'.join(current_section))},
                    chunk_type=section_type,
                    importance_score=self._calculate_dialogue_importance('\n\n'.join(current_section)),
                    dependency_chunks=[]
                )
                chunks.append(chunk)
                
                # Start new section
                current_section = [paragraph]
                section_type = para_type
                chunk_id += 1
            else:
                current_section.append(paragraph)
                if not section_type:
                    section_type = para_type
        
        # Add final chunk
        if current_section:
            chunk = NovelChunk(
                id=f"dialogue_scene_{chunk_id}",
                content='\n\n'.join(current_section),
                strategy_used=ChunkingStrategy.DIALOGUE_SCENES,
                narrative_elements={'type': section_type},
                chunk_type=section_type,
                importance_score=self._calculate_dialogue_importance('\n\n'.join(current_section)),
                dependency_chunks=[]
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_temporal_breaks(self, text: str) -> List[NovelChunk]:
        """Chunk based on temporal shifts and time jumps"""
        
        # Time indicators
        time_patterns = [
            r'\b(meanwhile|later|earlier|the next day|hours later|weeks passed|months later|years ago)\b',
            r'\b(suddenly|then|after|before|during|while)\b',
            r'\b\d+\s+(minutes?|hours?|days?|weeks?|months?|years?)\s+(later|ago|before|after)\b'
        ]
        
        chunks = []
        current_chunk = []
        timeline_position = 0
        chunk_id = 0
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Check for temporal indicators
            has_time_shift = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in time_patterns)
            
            if has_time_shift and current_chunk:
                # Finalize current temporal chunk
                chunk = NovelChunk(
                    id=f"temporal_{chunk_id}",
                    content=' '.join(current_chunk),
                    strategy_used=ChunkingStrategy.TEMPORAL_BREAKS,
                    narrative_elements={
                        'timeline_position': timeline_position,
                        'temporal_markers': self._extract_temporal_markers(' '.join(current_chunk))
                    },
                    chunk_type="temporal_sequence",
                    importance_score=self._calculate_temporal_importance(' '.join(current_chunk)),
                    dependency_chunks=[]
                )
                chunks.append(chunk)
                
                # Start new temporal sequence
                current_chunk = [sentence]
                timeline_position += 1
                chunk_id += 1
            else:
                current_chunk.append(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk = NovelChunk(
                id=f"temporal_{chunk_id}",
                content=' '.join(current_chunk),
                strategy_used=ChunkingStrategy.TEMPORAL_BREAKS,
                narrative_elements={
                    'timeline_position': timeline_position,
                    'temporal_markers': self._extract_temporal_markers(' '.join(current_chunk))
                },
                chunk_type="temporal_sequence",
                importance_score=self._calculate_temporal_importance(' '.join(current_chunk)),
                dependency_chunks=[]
            )
            chunks.append(chunk)
        
        return chunks
    
    def adaptive_chunking(self, text: str, context: Dict) -> List[NovelChunk]:
        """Adaptive chunking based on novel context and requirements"""
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_characteristics(text)
        
        # Choose optimal chunking strategy
        if text_analysis['dialogue_heavy']:
            chunks = self.chunk_by_dialogue_scenes(text)
        elif text_analysis['character_focused']:
            chunks = self.chunk_by_character_focus(text, context.get('characters', []))
        elif text_analysis['temporal_complex']:
            chunks = self.chunk_by_temporal_breaks(text)
        else:
            chunks = self.chunk_by_narrative_flow(text)
        
        # Post-process chunks for optimization
        optimized_chunks = self._optimize_chunks(chunks, context)
        
        return optimized_chunks
    
    def _detect_story_beats(self, text: str) -> List[Dict]:
        """Detect major story beats and narrative structure"""
        # This would use more sophisticated analysis in real implementation
        
        beats = []
        paragraphs = text.split('\n\n')
        
        # Simple beat detection based on paragraph structure and keywords
        beat_indicators = {
            'setup': ['introduce', 'began', 'started', 'once upon'],
            'conflict': ['suddenly', 'but', 'however', 'conflict', 'problem'],
            'climax': ['finally', 'at last', 'climax', 'peak', 'ultimate'],
            'resolution': ['ended', 'concluded', 'resolved', 'finally', 'peace']
        }
        
        current_beat = {'content': '', 'type': 'narrative', 'importance': 0.5, 'elements': {}}
        
        for paragraph in paragraphs:
            # Determine beat type
            beat_type = 'narrative'  # default
            importance = 0.5
            
            for beat_name, keywords in beat_indicators.items():
                if any(keyword in paragraph.lower() for keyword in keywords):
                    beat_type = beat_name
                    importance = 0.8 if beat_name in ['conflict', 'climax'] else 0.6
                    break
            
            current_beat['content'] += paragraph + '\n\n'
            current_beat['type'] = beat_type
            current_beat['importance'] = max(current_beat['importance'], importance)
            
            # If paragraph is long or beat type changes significantly, start new beat
            if len(current_beat['content'].split()) > 800:
                beats.append(current_beat.copy())
                current_beat = {'content': '', 'type': 'narrative', 'importance': 0.5, 'elements': {}}
        
        if current_beat['content']:
            beats.append(current_beat)
        
        return beats
    
    def _extract_characters_from_text(self, text: str, known_characters: List[str]) -> set:
        """Extract character references from text"""
        found_characters = set()
        
        for character in known_characters:
            # Check for full name or parts of name
            character_parts = character.split()
            for part in character_parts:
                if len(part) > 2 and part.lower() in text.lower():
                    found_characters.add(character)
                    break
        
        return found_characters
    
    def _classify_paragraph_type(self, paragraph: str) -> str:
        """Classify paragraph as dialogue-heavy or narrative"""
        # Count dialogue markers
        dialogue_markers = paragraph.count('"') + paragraph.count("'")
        paragraph_length = len(paragraph)
        
        if paragraph_length > 0:
            dialogue_ratio = dialogue_markers / paragraph_length
            return "dialogue" if dialogue_ratio > 0.1 else "narrative"
        
        return "narrative"
    
    def _analyze_text_characteristics(self, text: str) -> Dict:
        """Analyze text to determine optimal chunking strategy"""
        
        dialogue_ratio = (text.count('"') + text.count("'")) / len(text) if text else 0
        
        # Count character names (simplified)
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', text)
        character_density = len(set(potential_names)) / len(text.split()) if text else 0
        
        # Count temporal markers
        temporal_markers = len(re.findall(r'\b(then|later|before|after|meanwhile)\b', text, re.IGNORECASE))
        temporal_density = temporal_markers / len(text.split()) if text else 0
        
        return {
            'dialogue_heavy': dialogue_ratio > 0.15,
            'character_focused': character_density > 0.05,
            'temporal_complex': temporal_density > 0.02
        }
    
    # Additional helper methods would be implemented here...
    def _calculate_dialogue_ratio(self, text: str) -> float:
        """Calculate ratio of dialogue to narrative in text"""
        if not text:
            return 0.0
        return (text.count('"') + text.count("'")) / len(text)
    
    def _calculate_dialogue_importance(self, text: str) -> float:
        """Calculate importance score for dialogue sections"""
        # Higher importance for character development dialogue
        development_keywords = ['feel', 'think', 'believe', 'remember', 'love', 'hate']
        score = 0.5
        
        for keyword in development_keywords:
            if keyword in text.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_temporal_markers(self, text: str) -> List[str]:
        """Extract temporal markers from text"""
        patterns = [
            r'\b(meanwhile|later|earlier|the next day|hours later|weeks passed|months later|years ago)\b',
            r'\b\d+\s+(minutes?|hours?|days?|weeks?|months?|years?)\s+(later|ago|before|after)\b'
        ]
        
        markers = []
        for pattern in patterns:
            markers.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return markers
    
    def _calculate_temporal_importance(self, text: str) -> float:
        """Calculate importance of temporal chunks"""
        # Higher importance for major time jumps
        major_time_words = ['years', 'months', 'decades']
        score = 0.5
        
        for word in major_time_words:
            if word in text.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    def _create_character_chunk(self, content_lines: List[str], characters: set, chunk_id: int) -> NovelChunk:
        """Create a character-focused chunk"""
        return NovelChunk(
            id=f"character_{chunk_id}",
            content='\n\n'.join(content_lines),
            strategy_used=ChunkingStrategy.CHARACTER_FOCUS,
            narrative_elements={
                'primary_characters': list(characters),
                'character_count': len(characters),
                'interaction_type': self._classify_character_interaction('\n\n'.join(content_lines))
            },
            chunk_type="character_scene",
            importance_score=0.7 if len(characters) > 1 else 0.5,
            dependency_chunks=[]
        )
    
    def _classify_character_interaction(self, text: str) -> str:
        """Classify the type of character interaction in text"""
        if '"' in text or "'" in text:
            return "dialogue"
        elif any(word in text.lower() for word in ['fight', 'battle', 'conflict']):
            return "conflict"
        elif any(word in text.lower() for word in ['love', 'kiss', 'romance']):
            return "romance"
        else:
            return "interaction"
    
    def _sub_chunk_narrative_beat(self, beat: Dict, max_size: int) -> List[NovelChunk]:
        """Sub-chunk a narrative beat that's too large"""
        content = beat['content']
        sentences = sent_tokenize(content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        sub_chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > max_size and current_chunk:
                # Create sub-chunk
                chunk = NovelChunk(
                    id=f"{beat.get('id', 'beat')}_{sub_chunk_id}",
                    content=' '.join(current_chunk),
                    strategy_used=ChunkingStrategy.NARRATIVE_FLOW,
                    narrative_elements=beat['elements'],
                    chunk_type=f"{beat['type']}_part",
                    importance_score=beat['importance'],
                    dependency_chunks=[]
                )
                chunks.append(chunk)
                
                current_chunk = [sentence]
                current_size = sentence_size
                sub_chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final sub-chunk
        if current_chunk:
            chunk = NovelChunk(
                id=f"{beat.get('id', 'beat')}_{sub_chunk_id}",
                content=' '.join(current_chunk),
                strategy_used=ChunkingStrategy.NARRATIVE_FLOW,
                narrative_elements=beat['elements'],
                chunk_type=f"{beat['type']}_part",
                importance_score=beat['importance'],
                dependency_chunks=[]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _optimize_chunks(self, chunks: List[NovelChunk], context: Dict) -> List[NovelChunk]:
        """Post-process chunks for optimization"""
        optimized = []
        
        for chunk in chunks:
            # Merge very small chunks with adjacent ones
            if len(chunk.content.split()) < 100 and optimized:
                # Merge with previous chunk if compatible
                prev_chunk = optimized[-1]
                if self._chunks_compatible_for_merge(chunk, prev_chunk):
                    merged_chunk = self._merge_chunks(prev_chunk, chunk)
                    optimized[-1] = merged_chunk
                    continue
            
            # Split overly large chunks
            if len(chunk.content.split()) > 1500:
                split_chunks = self._split_large_chunk(chunk)
                optimized.extend(split_chunks)
            else:
                optimized.append(chunk)
        
        return optimized
    
    def _chunks_compatible_for_merge(self, chunk1: NovelChunk, chunk2: NovelChunk) -> bool:
        """Check if two chunks can be merged"""
        # Same chunking strategy and similar chunk type
        return (chunk1.strategy_used == chunk2.strategy_used and 
                chunk1.chunk_type == chunk2.chunk_type)
    
    def _merge_chunks(self, chunk1: NovelChunk, chunk2: NovelChunk) -> NovelChunk:
        """Merge two compatible chunks"""
        return NovelChunk(
            id=f"{chunk1.id}_merged_{chunk2.id}",
            content=f"{chunk1.content}\n\n{chunk2.content}",
            strategy_used=chunk1.strategy_used,
            narrative_elements={**chunk1.narrative_elements, **chunk2.narrative_elements},
            chunk_type=chunk1.chunk_type,
            importance_score=max(chunk1.importance_score, chunk2.importance_score),
            dependency_chunks=list(set(chunk1.dependency_chunks + chunk2.dependency_chunks))
        )
    
    def _split_large_chunk(self, chunk: NovelChunk) -> List[NovelChunk]:
        """Split a chunk that's too large"""
        sentences = sent_tokenize(chunk.content)
        target_size = 800  # Target words per chunk
        
        split_chunks = []
        current_content = []
        current_size = 0
        split_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > target_size and current_content:
                # Create split chunk
                split_chunk = NovelChunk(
                    id=f"{chunk.id}_split_{split_id}",
                    content=' '.join(current_content),
                    strategy_used=chunk.strategy_used,
                    narrative_elements=chunk.narrative_elements,
                    chunk_type=f"{chunk.chunk_type}_split",
                    importance_score=chunk.importance_score,
                    dependency_chunks=chunk.dependency_chunks
                )
                split_chunks.append(split_chunk)
                
                current_content = [sentence]
                current_size = sentence_size
                split_id += 1
            else:
                current_content.append(sentence)
                current_size += sentence_size
        
        # Add final split chunk
        if current_content:
            split_chunk = NovelChunk(
                id=f"{chunk.id}_split_{split_id}",
                content=' '.join(current_content),
                strategy_used=chunk.strategy_used,
                narrative_elements=chunk.narrative_elements,
                chunk_type=f"{chunk.chunk_type}_split",
                importance_score=chunk.importance_score,
                dependency_chunks=chunk.dependency_chunks
            )
            split_chunks.append(split_chunk)
        
        return split_chunks