"""
Emotional Memory System Implementation
Implements character emotional state tracking and development.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Emotion categories based on Plutchik's wheel."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class EmotionIntensity(Enum):
    """Emotion intensity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class EmotionalState:
    """Character emotional state at a specific point."""
    character_name: str
    dominant_emotion: EmotionCategory
    intensity: EmotionIntensity
    emotion_vector: Dict[str, float]  # All emotion scores
    trigger_event: Optional[str] = None
    context: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5
    source_chunk_id: Optional[str] = None


@dataclass
class EmotionalArc:
    """Emotional arc for a character across story."""
    character_name: str
    arc_name: str
    start_emotion: EmotionalState
    current_emotion: EmotionalState
    peak_emotion: Optional[EmotionalState] = None
    emotional_journey: List[EmotionalState] = field(default_factory=list)
    plot_thread: Optional[str] = None
    is_active: bool = True


class EmotionalMemorySystem:
    """System for tracking and managing character emotional states."""
    
    def __init__(self, db_utils=None):
        """Initialize emotional memory system."""
        self.db_utils = db_utils
        
        # Emotion analysis configuration
        self.emotion_keywords = {
            EmotionCategory.JOY: ['happy', 'joyful', 'excited', 'delighted', 'cheerful', 'elated'],
            EmotionCategory.TRUST: ['trust', 'faith', 'believe', 'confidence', 'reliance'],
            EmotionCategory.FEAR: ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'panic'],
            EmotionCategory.SURPRISE: ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            EmotionCategory.SADNESS: ['sad', 'melancholy', 'depressed', 'sorrowful', 'grief', 'mourning'],
            EmotionCategory.DISGUST: ['disgusted', 'revolted', 'repulsed', 'sickened', 'distaste'],
            EmotionCategory.ANGER: ['angry', 'furious', 'rage', 'irritated', 'annoyed', 'livid'],
            EmotionCategory.ANTICIPATION: ['anticipate', 'eager', 'expectant', 'impatient', 'waiting']
        }
        
        # Character emotional states
        self.character_states: Dict[str, EmotionalState] = {}
        self.emotional_arcs: Dict[str, List[EmotionalArc]] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, EmotionalState] = {}
    
    async def analyze_emotional_content(
        self,
        content: str,
        characters: List[str],
        chunk_id: Optional[str] = None,
        method: str = "keyword_analysis"
    ) -> List[EmotionalState]:
        """
        Analyze emotional content for characters.
        
        Args:
            content: Text content to analyze
            characters: List of character names to analyze
            chunk_id: Source chunk ID
            method: Analysis method to use
        
        Returns:
            List of emotional states detected
            
        Raises:
            ValueError: If content or characters are invalid
            RuntimeError: If analysis fails
        """
        if not content or not characters:
            raise ValueError("Content and characters must not be empty")
        
        if not content.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        
        if not isinstance(characters, list) or not all(isinstance(c, str) for c in characters):
            raise ValueError("Characters must be a list of strings")
        
        emotional_states = []
        
        for character in characters:
            try:
                # Check if character is mentioned in content
                if not self._character_mentioned_in_content(character, content):
                    continue
                
                # Extract emotional state
                emotional_state = await self._extract_character_emotion(
                    character, content, chunk_id, method
                )
                
                if emotional_state:
                    emotional_states.append(emotional_state)
                    
                    # Update character's current state
                    self.character_states[character] = emotional_state
                    
                    # Update emotional arcs
                    await self._update_emotional_arcs(character, emotional_state)
            
            except Exception as e:
                logger.error(f"Error analyzing emotion for {character}: {e}")
                raise RuntimeError(f"Failed to analyze emotions for {character}: {e}") from e
        
        return emotional_states
    
    async def _extract_character_emotion(
        self,
        character: str,
        content: str,
        chunk_id: Optional[str],
        method: str
    ) -> Optional[EmotionalState]:
        """Extract emotional state for a specific character."""
        
        # Create cache key
        cache_key = f"{character}_{hash(content)}_{method}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        if method == "keyword_analysis":
            emotional_state = await self._keyword_based_emotion_analysis(
                character, content, chunk_id
            )
        elif method == "context_analysis":
            emotional_state = await self._context_based_emotion_analysis(
                character, content, chunk_id
            )
        else:
            logger.warning(f"Unknown emotion analysis method: {method}")
            return None
        
        # Cache result
        if emotional_state:
            self.analysis_cache[cache_key] = emotional_state
        
        return emotional_state
    
    async def _keyword_based_emotion_analysis(
        self,
        character: str,
        content: str,
        chunk_id: Optional[str]
    ) -> Optional[EmotionalState]:
        """Analyze emotions using keyword matching."""
        
        # Find character context in content
        character_context = self._extract_character_context(character, content)
        if not character_context:
            return None
        
        # Calculate emotion scores
        emotion_scores = {}
        total_matches = 0
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                # Count keyword occurrences in character context
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', 
                                       character_context.lower()))
                score += matches
                total_matches += matches
            
            emotion_scores[emotion.value] = score
        
        if total_matches == 0:
            return None
        
        # Normalize scores
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total_matches
        
        # Find dominant emotion
        dominant_emotion_str = max(emotion_scores, key=emotion_scores.get)
        dominant_emotion = EmotionCategory(dominant_emotion_str)
        
        # Calculate intensity
        max_score = emotion_scores[dominant_emotion_str]
        if max_score >= 0.6:
            intensity = EmotionIntensity.HIGH
        elif max_score >= 0.4:
            intensity = EmotionIntensity.MEDIUM
        else:
            intensity = EmotionIntensity.LOW
        
        # Extract trigger event
        trigger_event = self._extract_trigger_event(character_context)
        
        return EmotionalState(
            character_name=character,
            dominant_emotion=dominant_emotion,
            intensity=intensity,
            emotion_vector=emotion_scores,
            trigger_event=trigger_event,
            context=character_context[:200] + "..." if len(character_context) > 200 else character_context,
            confidence_score=max_score,
            source_chunk_id=chunk_id
        )
    
    async def _context_based_emotion_analysis(
        self,
        character: str,
        content: str,
        chunk_id: Optional[str]
    ) -> Optional[EmotionalState]:
        """Analyze emotions using contextual analysis."""
        
        # This would use more sophisticated NLP or LLM-based analysis
        # For now, fall back to keyword analysis
        return await self._keyword_based_emotion_analysis(character, content, chunk_id)
    
    def _character_mentioned_in_content(self, character: str, content: str) -> bool:
        """Check if character is mentioned in content."""
        
        # Check for full name
        if character.lower() in content.lower():
            return True
        
        # Check for first name only
        first_name = character.split()[0] if ' ' in character else character
        if first_name.lower() in content.lower():
            return True
        
        # Check for pronouns near character mentions
        # This is a simplified approach
        return False
    
    def _extract_character_context(self, character: str, content: str) -> str:
        """Extract context around character mentions."""
        
        sentences = re.split(r'[.!?]+', content)
        character_sentences = []
        
        for sentence in sentences:
            if character.lower() in sentence.lower():
                character_sentences.append(sentence.strip())
        
        return '. '.join(character_sentences)
    
    def _extract_trigger_event(self, context: str) -> Optional[str]:
        """Extract the event that triggered the emotion."""
        
        # Look for action verbs and events
        trigger_patterns = [
            r'(saw|heard|felt|realized|discovered|found|learned|remembered)',
            r'(said|told|asked|replied|whispered|shouted)',
            r'(happened|occurred|took place|began|started|ended)'
        ]
        
        for pattern in trigger_patterns:
            matches = re.findall(pattern, context.lower())
            if matches:
                # Find the sentence containing the trigger
                sentences = context.split('.')
                for sentence in sentences:
                    if any(match in sentence.lower() for match in matches):
                        return sentence.strip()[:100] + "..." if len(sentence) > 100 else sentence.strip()
        
        return None
    
    async def _update_emotional_arcs(self, character: str, new_state: EmotionalState):
        """Update emotional arcs for character."""
        
        if character not in self.emotional_arcs:
            self.emotional_arcs[character] = []
        
        # Find active arc or create new one
        active_arc = None
        for arc in self.emotional_arcs[character]:
            if arc.is_active:
                active_arc = arc
                break
        
        if not active_arc:
            # Create new emotional arc
            active_arc = EmotionalArc(
                character_name=character,
                arc_name=f"{character}_arc_{len(self.emotional_arcs[character]) + 1}",
                start_emotion=new_state,
                current_emotion=new_state,
                plot_thread=new_state.context
            )
            self.emotional_arcs[character].append(active_arc)
        else:
            # Update existing arc
            active_arc.current_emotion = new_state
            active_arc.emotional_journey.append(new_state)
            
            # Check for emotional peak
            if (not active_arc.peak_emotion or 
                new_state.intensity.value > active_arc.peak_emotion.intensity.value):
                active_arc.peak_emotion = new_state
    
    async def validate_emotional_consistency(self, character_name: str, new_emotion: EmotionCategory) -> bool:
        """Validate emotional consistency for a character's new emotion."""
        # Retrieve recent emotional states
        recent_states = await self.get_character_emotional_history(character_name, limit=5)
        if not recent_states:
            return True  # No history, so consistent

        # Check for conflicting emotions in recent states
        for state in recent_states:
            if self._emotions_conflict(state.dominant_emotion.value, new_emotion.value):
                return False
        return True

    async def get_character_emotional_history(self, character_name: str, limit: int = 10) -> List[EmotionalState]:
        """Retrieve recent emotional states for a character from the database."""
        if self.db_utils:
            try:
                rows = await self.db_utils.fetch_character_emotions(character_name, limit)
                return [
                    EmotionalState(
                        character_name=row["character_name"],
                        dominant_emotion=EmotionCategory(row["dominant_emotion"]),
                        intensity=EmotionIntensity(row["intensity"]),
                        emotion_vector=row["emotion_vector"],
                        trigger_event=row["trigger_event"],
                        confidence_score=row["confidence_score"],
                        source_chunk_id=row["chunk_id"]
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Error fetching emotional history: {e}")
                return []
        else:
            # Fallback to in-memory or empty
            return []

    async def get_emotional_context_for_generation(
        self,
        characters: List[str],
        generation_type: str = "continuation"
    ) -> Dict[str, Any]:
        """Get emotional context for content generation."""
        
        emotional_context = {
            "character_emotions": {},
            "emotional_tension": 0.0,
            "dominant_mood": "neutral",
            "emotional_conflicts": []
        }
        
        character_emotions = {}
        
        for character in characters:
            current_state = self.character_states.get(character)
            if current_state:
                character_emotions[character] = {
                    "current_emotion": current_state.dominant_emotion.value,
                    "intensity": current_state.intensity.value,
                    "recent_trigger": current_state.trigger_event
                }
        
        emotional_context["character_emotions"] = character_emotions
        
        # Calculate emotional tension
        if len(character_emotions) > 1:
            emotional_context["emotional_tension"] = self._calculate_emotional_tension(
                list(character_emotions.values())
            )
        
        # Determine dominant mood
        if character_emotions:
            emotions = [state["current_emotion"] for state in character_emotions.values()]
            emotional_context["dominant_mood"] = max(set(emotions), key=emotions.count)
        
        # Detect emotional conflicts
        emotional_context["emotional_conflicts"] = self._detect_emotional_conflicts(
            character_emotions
        )
        
        return emotional_context
    
    def _calculate_emotional_tension(self, character_emotions: List[Dict[str, Any]]) -> float:
        """Calculate emotional tension between characters."""
        
        if len(character_emotions) < 2:
            return 0.0
        
        # Simplified tension calculation
        tension_score = 0.0
        
        for i, emotion1 in enumerate(character_emotions):
            for emotion2 in character_emotions[i+1:]:
                # Check for conflicting emotions
                if self._emotions_conflict(emotion1["current_emotion"], emotion2["current_emotion"]):
                    tension_score += 0.3
                
                # High intensity emotions increase tension
                if emotion1["intensity"] == "high" or emotion2["intensity"] == "high":
                    tension_score += 0.2
        
        return min(tension_score, 1.0)
    
    def _emotions_conflict(self, emotion1: str, emotion2: str) -> bool:
        """Check if two emotions are conflicting."""
        
        conflicting_pairs = [
            ("joy", "sadness"),
            ("trust", "disgust"),
            ("fear", "anger"),
            ("surprise", "anticipation")
        ]
        
        for pair in conflicting_pairs:
            if (emotion1 in pair and emotion2 in pair) and emotion1 != emotion2:
                return True
        
        return False
    
    def _detect_emotional_conflicts(self, character_emotions: Dict[str, Dict[str, Any]]) -> List[str]:
        """Detect emotional conflicts between characters."""
        
        conflicts = []
        characters = list(character_emotions.keys())
        
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                emotion1 = character_emotions[char1]["current_emotion"]
                emotion2 = character_emotions[char2]["current_emotion"]
                
                if self._emotions_conflict(emotion1, emotion2):
                    conflicts.append(f"{char1} ({emotion1}) vs {char2} ({emotion2})")
        
        return conflicts
    
    async def test_store_emotional_analysis(self):
        from datetime import datetime
        from unittest.mock import AsyncMock

        # Setup mock db_utils
        self.db_utils = AsyncMock()
        self.db_utils.save_character_emotions = AsyncMock(return_value=None)

        # Create dummy emotional states
        emotional_states = [
            EmotionalState(
                character_name="Alice",
                dominant_emotion=EmotionCategory.JOY,
                intensity=EmotionIntensity.HIGH,
                emotion_vector={"joy": 0.9, "sadness": 0.1},
                trigger_event="Alice found a treasure",
                confidence_score=0.95,
                source_chunk_id="chunk1"
            ),
            EmotionalState(
                character_name="Bob",
                dominant_emotion=EmotionCategory.SADNESS,
                intensity=EmotionIntensity.MEDIUM,
                emotion_vector={"joy": 0.2, "sadness": 0.8},
                trigger_event="Bob lost his way",
                confidence_score=0.85,
                source_chunk_id="chunk2"
            )
        ]

        run_id = "test_run_123"
        scene_id = "scene_abc"

        # Call store_emotional_analysis
        result = await self.store_emotional_analysis(emotional_states, run_id, scene_id)

        # Assert db_utils.save_character_emotions called once
        self.db_utils.save_character_emotions.assert_called_once()

        # Assert result is True
        assert result is True

    async def store_emotional_analysis(
        self,
        emotional_states: List[EmotionalState],
        run_id: str,
        scene_id: Optional[str] = None
    ) -> bool:
        """Store emotional analysis results in database."""
        
        if not self.db_utils:
            logger.warning("No database utils available for storing emotional analysis")
            return False
        
        try:
            emotions_data = []
            
            for state in emotional_states:
                emotion_data = {
                    'scene_id': scene_id,
                    'chunk_id': state.source_chunk_id,
                    'character_name': state.character_name,
                    'emotion_vector': state.emotion_vector,
                    'dominant_emotion': state.dominant_emotion.value,
                    'intensity': state.intensity.value,
                    'emotion_category': self._get_emotion_category(state.dominant_emotion),
                    'trigger_event': state.trigger_event,
                    'related_character': None,  # Could be enhanced
                    'source_type': 'narrative',  # Could be enhanced to detect dialogue
                    'confidence_score': state.confidence_score,
                    'method': 'keyword_analysis',
                    'model_name': 'keyword_analyzer',
                    'model_version': 'v1.0',
                    'prompt_hash': None,
                    'span_start': 0,  # Could be enhanced
                    'span_end': 0,  # Could be enhanced
                    'sentence_index': 0,  # Could be enhanced
                    'intra_chunk_order': 0,  # Could be enhanced
                    'intensity_calibrated': self._calibrate_intensity(state.intensity),
                    'top_emotions': self._get_top_emotions(state.emotion_vector, top_k=3)
                }
                emotions_data.append(emotion_data)
            
            # Store using the database utility function
            await self.db_utils.save_character_emotions(run_id, emotions_data)
            
            logger.info(f"Stored {len(emotions_data)} emotional states for run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing emotional analysis: {e}")
            return False

    
    def _get_emotion_category(self, emotion: EmotionCategory) -> str:
        """Get emotion category (positive/negative/neutral)."""
        
        positive_emotions = [EmotionCategory.JOY, EmotionCategory.TRUST, EmotionCategory.ANTICIPATION]
        negative_emotions = [EmotionCategory.FEAR, EmotionCategory.SADNESS, EmotionCategory.DISGUST, EmotionCategory.ANGER]
        
        if emotion in positive_emotions:
            return "positive"
        elif emotion in negative_emotions:
            return "negative"
        else:
            return "neutral"
    
    def _calibrate_intensity(self, intensity: EmotionIntensity) -> float:
        """Convert intensity enum to calibrated float value."""
        
        intensity_map = {
            EmotionIntensity.LOW: 0.25,
            EmotionIntensity.MEDIUM: 0.5,
            EmotionIntensity.HIGH: 0.75,
            EmotionIntensity.EXTREME: 1.0
        }
        
        return intensity_map.get(intensity, 0.5)
    
    def _get_top_emotions(self, emotion_vector: Dict[str, float], top_k: int = 3) -> List[str]:
        """Get top K emotions from emotion vector."""
        
        sorted_emotions = sorted(emotion_vector.items(), key=lambda x: x[1], reverse=True)
        return [emotion for emotion, score in sorted_emotions[:top_k] if score > 0]
    
    async def create_analysis_run(
        self,
        method: str = "keyword_analysis",
        model_name: str = "keyword_analyzer",
        model_version: str = "v1.0",
        params: Dict[str, Any] = None
    ) -> str:
        """Create new emotion analysis run."""
        
        if self.db_utils:
            try:
                run_id = await self.db_utils.create_emotion_analysis_run(
                    method=method,
                    model_name=model_name,
                    model_version=model_version,
                    params=params or {}
                )
                return run_id
            except Exception as e:
                logger.error(f"Error creating analysis run: {e}")
        
        # Fallback to timestamp-based ID
        return f"run_{int(datetime.now().timestamp())}"
    
    async def complete_analysis_run(self, run_id: str, status: str = "success", error: str = None):
        """Mark analysis run as complete."""
        
        if self.db_utils:
            try:
                await self.db_utils.update_emotion_analysis_run_status(
                    run_id=run_id,
                    status=status,
                    error=error
                )
            except Exception as e:
                logger.error(f"Error completing analysis run: {e}")
    
    def get_emotional_memory_stats(self) -> Dict[str, Any]:
        """Get emotional memory system statistics."""
        
        return {
            "tracked_characters": len(self.character_states),
            "active_emotional_arcs": sum(
                len([arc for arc in arcs if arc.is_active])
                for arcs in self.emotional_arcs.values()
            ),
            "total_emotional_arcs": sum(len(arcs) for arcs in self.emotional_arcs.values()),
            "cache_size": len(self.analysis_cache),
            "emotion_categories_tracked": len(self.emotion_keywords),
            "current_character_emotions": {
                char: {
                    "emotion": state.dominant_emotion.value,
                    "intensity": state.intensity.value,
                    "confidence": state.confidence_score
                }
                for char, state in self.character_states.items()
            }
        }


# Factory function
def create_emotional_memory_system(db_utils=None) -> EmotionalMemorySystem:
    """Create emotional memory system."""
    return EmotionalMemorySystem(db_utils)


# Example usage
async def main():
    """Example usage of emotional memory system."""
    
    # Create emotional memory system
    ems = create_emotional_memory_system()
    
    # Test content
    test_content = '''
    Emma felt a surge of fear as she heard the footsteps approaching. 
    Her heart raced with anxiety, but she tried to remain calm.
    
    "Don't worry," John said with confidence, though he was secretly worried.
    "We'll figure this out together."
    
    Emma looked at him with gratitude and trust, feeling slightly better.
    '''
    
    # Analyze emotions
    characters = ["Emma", "John"]
    run_id = await ems.create_analysis_run()
    
    emotional_states = await ems.analyze_emotional_content(
        content=test_content,
        characters=characters,
        chunk_id="test_chunk_1"
    )
    
    # Display results
    print(f"Emotional analysis completed for {len(emotional_states)} characters:")
    for state in emotional_states:
        print(f"  {state.character_name}: {state.dominant_emotion.value} "
              f"({state.intensity.value}) - {state.trigger_event}")
    
    # Get emotional context
    context = await ems.get_emotional_context_for_generation(characters)
    print(f"\nEmotional context: {context}")
    
    # Get stats
    stats = ems.get_emotional_memory_stats()
    print(f"\nSystem stats: {stats}")
    
    await ems.complete_analysis_run(run_id)


if __name__ == "__main__":
    asyncio.run(main())