"""
Consistency Validators for Creative Novel Generation

This module provides validators to ensure consistency in character development,
plot progression, world building, and emotional tone across generated content.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .models import EmotionalTone
from .enhanced_context_builder import ContextBuildResult

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None
    suggestion: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    score: float
    issues: List[str]
    suggestions: List[str]
    validator_name: str
    details: Dict[str, Any]
    
    def __init__(self, validator_name: str, is_valid: bool = True, score: float = 1.0):
        self.validator_name = validator_name
        self.is_valid = is_valid
        self.score = score
        self.issues = []
        self.suggestions = []
        self.details = {}
    
    def add_issue(self, message: str, suggestion: str = None):
        """Add an issue to the validation result."""
        self.issues.append(message)
        if suggestion:
            self.suggestions.append(suggestion)
        self.is_valid = False
        self.score = max(0.0, self.score - 0.2)  # Reduce score for each issue


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def validate(self, content: str, *args, **kwargs) -> ValidationResult:
        """Validate content and return result."""
        pass
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self, content: str) -> List[str]:
        """Extract words from content."""
        return re.findall(r'\b\w+\b', content.lower())
    
    def _calculate_base_score(self, issues_count: int, max_issues: int = 5) -> float:
        """Calculate base score based on number of issues."""
        if issues_count == 0:
            return 1.0
        return max(0.0, 1.0 - (issues_count / max_issues))


class CharacterConsistencyValidator(BaseValidator):
    """Validates character consistency in generated content."""
    
    def __init__(self):
        super().__init__("character_consistency")
        # Common character traits and personality indicators
        self.personality_indicators = {
            "confident": ["bold", "assertive", "sure", "certain", "decisive"],
            "shy": ["quiet", "hesitant", "timid", "reserved", "withdrawn"],
            "aggressive": ["angry", "hostile", "forceful", "violent", "harsh"],
            "kind": ["gentle", "caring", "compassionate", "warm", "helpful"],
            "intelligent": ["clever", "smart", "wise", "analytical", "thoughtful"],
            "impulsive": ["sudden", "rash", "hasty", "spontaneous", "reckless"]
        }
    
    async def validate(self, content: str, target_characters: List[str], 
                      context_result: Optional[ContextBuildResult] = None) -> ValidationResult:
        """Validate character consistency."""
        result = ValidationResult(self.name)
        
        try:
            if not target_characters:
                result.details["note"] = "No target characters specified"
                return result
            
            # Extract character profiles from context
            character_profiles = {}
            if context_result and context_result.character_profiles:
                character_profiles = context_result.character_profiles
            
            # Analyze each character in the content
            for character in target_characters:
                char_issues = await self._validate_character(content, character, character_profiles.get(character))
                
                for issue in char_issues:
                    result.add_issue(
                        f"Character '{character}': {issue['message']}", 
                        issue.get('suggestion')
                    )
            
            # Check for character voice consistency
            voice_issues = self._check_dialogue_consistency(content, target_characters)
            for issue in voice_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            result.details["characters_analyzed"] = target_characters
            result.details["character_mentions"] = self._count_character_mentions(content, target_characters)
            
            self.logger.info(f"Character validation completed: {result.score:.2f} score, {len(result.issues)} issues")
            return result
            
        except Exception as e:
            self.logger.error(f"Character validation failed: {e}")
            result.add_issue(f"Validation error: {str(e)}")
            return result
    
    async def _validate_character(self, content: str, character: str, 
                                profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a specific character."""
        issues = []
        
        # Check if character is mentioned
        if character.lower() not in content.lower():
            return issues  # Character not in content, no validation needed
        
        # If we have a profile, check consistency
        if profile:
            # Check personality consistency
            personality_issues = self._check_personality_consistency(content, character, profile)
            issues.extend(personality_issues)
            
            # Check relationship consistency
            relationship_issues = self._check_relationship_consistency(content, character, profile)
            issues.extend(relationship_issues)
        
        return issues
    
    def _check_personality_consistency(self, content: str, character: str, 
                                     profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check personality consistency for a character."""
        issues = []
        
        # Extract traits from profile
        traits = profile.get('traits', [])
        if not traits:
            return issues
        
        # Look for contradictory personality indicators
        content_words = self._extract_words(content)
        
        for trait in traits:
            if trait in self.personality_indicators:
                positive_indicators = self.personality_indicators[trait]
                
                # Find contradictory traits
                contradictory_traits = self._find_contradictory_traits(trait)
                
                for contra_trait in contradictory_traits:
                    if contra_trait in self.personality_indicators:
                        contra_indicators = self.personality_indicators[contra_trait]
                        
                        # Check if contradictory indicators are present
                        found_contradictions = [word for word in content_words if word in contra_indicators]
                        
                        if found_contradictions:
                            issues.append({
                                'message': f"Personality inconsistency - {character} shows both {trait} and {contra_trait} traits",
                                'suggestion': f"Ensure {character}'s actions align with established {trait} personality"
                            })
        
        return issues
    
    def _find_contradictory_traits(self, trait: str) -> List[str]:
        """Find traits that contradict the given trait."""
        contradictions = {
            "confident": ["shy"],
            "shy": ["confident", "aggressive"],
            "aggressive": ["kind", "shy"],
            "kind": ["aggressive"],
            "impulsive": ["thoughtful"],
            "intelligent": []  # Intelligence doesn't directly contradict other traits
        }
        return contradictions.get(trait, [])
    
    def _check_relationship_consistency(self, content: str, character: str, 
                                      profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check relationship consistency."""
        issues = []
        
        relationships = profile.get('relationships', [])
        if not relationships:
            return issues
        
        # This is a simplified check - in a real implementation,
        # you'd have more sophisticated relationship analysis
        for relationship in relationships:
            if isinstance(relationship, dict):
                other_char = relationship.get('character')
                relation_type = relationship.get('type')
                
                if other_char and other_char.lower() in content.lower():
                    # Check if the interaction matches the relationship type
                    if relation_type == 'enemy' and 'friend' in content.lower():
                        issues.append({
                            'message': f"Relationship inconsistency - {character} and {other_char} are enemies but content suggests friendship",
                            'suggestion': f"Ensure interactions between {character} and {other_char} reflect their antagonistic relationship"
                        })
        
        return issues
    
    def _check_dialogue_consistency(self, content: str, characters: List[str]) -> List[Dict[str, Any]]:
        """Check dialogue consistency across characters."""
        issues = []
        
        # Extract dialogue patterns
        dialogue_pattern = r'"([^"]*)"'
        dialogues = re.findall(dialogue_pattern, content)
        
        if len(dialogues) < 2:
            return issues  # Not enough dialogue to check consistency
        
        # Simple check for dialogue variety
        if len(set(dialogues)) < len(dialogues) * 0.7:  # If 70% or more are identical
            issues.append({
                'message': "Dialogue lacks variety - characters sound too similar",
                'suggestion': "Ensure each character has a distinct voice and speaking pattern"
            })
        
        return issues
    
    def _count_character_mentions(self, content: str, characters: List[str]) -> Dict[str, int]:
        """Count mentions of each character."""
        mentions = {}
        content_lower = content.lower()
        
        for character in characters:
            count = content_lower.count(character.lower())
            mentions[character] = count
        
        return mentions


class PlotConsistencyValidator(BaseValidator):
    """Validates plot consistency and narrative flow."""
    
    def __init__(self):
        super().__init__("plot_consistency")
        self.plot_elements = {
            "conflict_indicators": ["conflict", "problem", "challenge", "obstacle", "tension"],
            "resolution_indicators": ["solved", "resolved", "concluded", "ended", "finished"],
            "progression_indicators": ["then", "next", "after", "later", "meanwhile", "suddenly"]
        }
    
    async def validate(self, content: str, narrative_constraints: Dict[str, Any], 
                      context_result: Optional[ContextBuildResult] = None) -> ValidationResult:
        """Validate plot consistency."""
        result = ValidationResult(self.name)
        
        try:
            # Check narrative flow
            flow_issues = self._check_narrative_flow(content)
            for issue in flow_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            # Check plot progression
            progression_issues = self._check_plot_progression(content, narrative_constraints)
            for issue in progression_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            # Check for plot holes
            plot_hole_issues = self._check_plot_holes(content, context_result)
            for issue in plot_hole_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            result.details["narrative_elements"] = self._analyze_narrative_elements(content)
            
            self.logger.info(f"Plot validation completed: {result.score:.2f} score, {len(result.issues)} issues")
            return result
            
        except Exception as e:
            self.logger.error(f"Plot validation failed: {e}")
            result.add_issue(f"Validation error: {str(e)}")
            return result
    
    def _check_narrative_flow(self, content: str) -> List[Dict[str, Any]]:
        """Check narrative flow and pacing."""
        issues = []
        
        sentences = self._extract_sentences(content)
        if len(sentences) < 3:
            return issues
        
        # Check for abrupt transitions
        transition_words = ["however", "but", "although", "meanwhile", "then", "next", "after"]
        transitions_found = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transitions_found += 1
        
        # If very few transitions in longer content, flag it
        if len(sentences) > 5 and transitions_found == 0:
            issues.append({
                'message': "Narrative flow could be improved with better transitions",
                'suggestion': "Add transitional phrases to connect ideas and improve flow"
            })
        
        return issues
    
    def _check_plot_progression(self, content: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check plot progression against constraints."""
        issues = []
        
        if not constraints:
            return issues
        
        # Check if required plot elements are present
        required_elements = constraints.get('required_elements', [])
        content_lower = content.lower()
        
        for element in required_elements:
            if element.lower() not in content_lower:
                issues.append({
                    'message': f"Required plot element '{element}' is missing",
                    'suggestion': f"Include '{element}' to meet narrative requirements"
                })
        
        # Check forbidden elements
        forbidden_elements = constraints.get('forbidden_elements', [])
        for element in forbidden_elements:
            if element.lower() in content_lower:
                issues.append({
                    'message': f"Forbidden plot element '{element}' is present",
                    'suggestion': f"Remove or modify content containing '{element}'"
                })
        
        return issues
    
    def _check_plot_holes(self, content: str, context_result: Optional[ContextBuildResult]) -> List[Dict[str, Any]]:
        """Check for potential plot holes."""
        issues = []
        
        # This is a simplified implementation
        # In practice, you'd have more sophisticated plot hole detection
        
        if context_result and context_result.plot_threads:
            # Check if content contradicts established plot threads
            for thread in context_result.plot_threads:
                thread_content = thread.get('content', '')
                # Simple contradiction check (this would be more sophisticated in practice)
                if 'not' in content.lower() and any(word in thread_content.lower() for word in content.lower().split()):
                    issues.append({
                        'message': "Potential plot contradiction detected",
                        'suggestion': "Ensure new content aligns with established plot threads"
                    })
        
        return issues
    
    def _analyze_narrative_elements(self, content: str) -> Dict[str, Any]:
        """Analyze narrative elements in the content."""
        content_lower = content.lower()
        
        analysis = {
            'has_conflict': any(word in content_lower for word in self.plot_elements['conflict_indicators']),
            'has_resolution': any(word in content_lower for word in self.plot_elements['resolution_indicators']),
            'has_progression': any(word in content_lower for word in self.plot_elements['progression_indicators']),
            'sentence_count': len(self._extract_sentences(content)),
            'word_count': len(self._extract_words(content))
        }
        
        return analysis


class WorldBuildingValidator(BaseValidator):
    """Validates world building consistency."""
    
    def __init__(self):
        super().__init__("world_building")
    
    async def validate(self, content: str, scene_location: str, 
                      context_result: Optional[ContextBuildResult] = None) -> ValidationResult:
        """Validate world building consistency."""
        result = ValidationResult(self.name)
        
        try:
            # Check location consistency
            location_issues = self._check_location_consistency(content, scene_location, context_result)
            for issue in location_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            # Check world rules consistency
            rules_issues = self._check_world_rules(content, context_result)
            for issue in rules_issues:
                result.add_issue(issue['message'], issue.get('suggestion'))
            
            result.details["locations_mentioned"] = self._extract_locations(content)
            result.details["scene_location"] = scene_location
            
            self.logger.info(f"World building validation completed: {result.score:.2f} score, {len(result.issues)} issues")
            return result
            
        except Exception as e:
            self.logger.error(f"World building validation failed: {e}")
            result.add_issue(f"Validation error: {str(e)}")
            return result
    
    def _check_location_consistency(self, content: str, scene_location: str, 
                                  context_result: Optional[ContextBuildResult]) -> List[Dict[str, Any]]:
        """Check location consistency."""
        issues = []
        
        if not scene_location:
            return issues
        
        # Check if the specified location is mentioned or implied
        content_lower = content.lower()
        location_lower = scene_location.lower()
        
        if location_lower not in content_lower:
            # This might be okay if the location is implied
            issues.append({
                'message': f"Scene location '{scene_location}' is not explicitly mentioned",
                'suggestion': f"Consider adding references to '{scene_location}' to establish setting"
            })
        
        # Check against established world elements
        if context_result and context_result.world_elements:
            for location, descriptions in context_result.world_elements.items():
                if location.lower() in content_lower:
                    # Check if description is consistent
                    # This is simplified - real implementation would be more sophisticated
                    pass
        
        return issues
    
    def _check_world_rules(self, content: str, context_result: Optional[ContextBuildResult]) -> List[Dict[str, Any]]:
        """Check consistency with established world rules."""
        issues = []
        
        # This is a placeholder for more sophisticated world rule checking
        # In practice, you'd have a knowledge base of world rules to check against
        
        return issues
    
    def _extract_locations(self, content: str) -> List[str]:
        """Extract location references from content."""
        # Simple location extraction - in practice, you'd use NER or more sophisticated methods
        location_indicators = ["at", "in", "on", "near", "by", "inside", "outside"]
        locations = []
        
        sentences = self._extract_sentences(content)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if word.lower() in location_indicators and i + 1 < len(words):
                    potential_location = words[i + 1]
                    if potential_location.istitle():  # Likely a proper noun
                        locations.append(potential_location)
        
        return list(set(locations))


class EmotionalToneValidator(BaseValidator):
    """Validates emotional tone consistency."""
    
    def __init__(self):
        super().__init__("emotional_tone")
        self.tone_indicators = {
            EmotionalTone.JOY: ["happy", "joyful", "delighted", "cheerful", "elated", "pleased"],
            EmotionalTone.SADNESS: ["sad", "sorrowful", "melancholy", "grief", "despair", "mournful"],
            EmotionalTone.ANGER: ["angry", "furious", "rage", "irritated", "hostile", "livid"],
            EmotionalTone.FEAR: ["afraid", "scared", "terrified", "anxious", "worried", "frightened"],
            EmotionalTone.SURPRISE: ["surprised", "shocked", "amazed", "astonished", "startled"],
            EmotionalTone.DISGUST: ["disgusted", "revolted", "repulsed", "sickened"],
            EmotionalTone.ANTICIPATION: ["excited", "eager", "hopeful", "expectant"],
            EmotionalTone.TRUST: ["trusting", "confident", "secure", "faithful"],
            EmotionalTone.NEUTRAL: ["calm", "peaceful", "serene", "balanced"]
        }
    
    async def validate(self, content: str, expected_tone: EmotionalTone, 
                      context_result: Optional[ContextBuildResult] = None) -> ValidationResult:
        """Validate emotional tone consistency."""
        result = ValidationResult(self.name)
        
        try:
            # Analyze the emotional tone of the content
            detected_tones = self._analyze_emotional_content(content)
            
            # Check if expected tone is present
            if expected_tone not in detected_tones or detected_tones[expected_tone] < 0.3:
                result.add_issue(
                    f"Expected emotional tone '{expected_tone}' is not sufficiently present",
                    f"Add more words and phrases that convey {expected_tone}"
                )
            
            # Check for conflicting tones
            conflicting_tones = self._find_conflicting_tones(expected_tone, detected_tones)
            for tone, strength in conflicting_tones:
                if strength > 0.4:  # Strong conflicting tone
                    result.add_issue(
                        f"Conflicting emotional tone '{tone}' detected (strength: {strength:.2f})",
                        f"Reduce {tone} indicators to maintain {expected_tone} consistency"
                    )
            
            result.details["expected_tone"] = expected_tone
            result.details["detected_tones"] = detected_tones
            result.details["dominant_tone"] = max(detected_tones, key=detected_tones.get) if detected_tones else None
            
            self.logger.info(f"Emotional tone validation completed: {result.score:.2f} score, {len(result.issues)} issues")
            return result
            
        except Exception as e:
            self.logger.error(f"Emotional tone validation failed: {e}")
            result.add_issue(f"Validation error: {str(e)}")
            return result
    
    def _analyze_emotional_content(self, content: str) -> Dict[EmotionalTone, float]:
        """Analyze emotional content and return tone strengths."""
        content_words = self._extract_words(content)
        tone_scores = {}
        
        for tone, indicators in self.tone_indicators.items():
            matches = sum(1 for word in content_words if word in indicators)
            # Normalize by content length
            score = matches / max(len(content_words), 1)
            tone_scores[tone] = score
        
        return tone_scores
    
    def _find_conflicting_tones(self, expected_tone: EmotionalTone, 
                              detected_tones: Dict[EmotionalTone, float]) -> List[Tuple[EmotionalTone, float]]:
        """Find tones that conflict with the expected tone."""
        conflicts = {
            EmotionalTone.JOY: [EmotionalTone.SADNESS, EmotionalTone.ANGER, EmotionalTone.FEAR],
            EmotionalTone.SADNESS: [EmotionalTone.JOY, EmotionalTone.ANTICIPATION],
            EmotionalTone.ANGER: [EmotionalTone.JOY, EmotionalTone.TRUST],
            EmotionalTone.FEAR: [EmotionalTone.JOY, EmotionalTone.TRUST],
            EmotionalTone.SURPRISE: [],  # Surprise can coexist with other emotions
            EmotionalTone.DISGUST: [EmotionalTone.JOY],
            EmotionalTone.ANTICIPATION: [EmotionalTone.SADNESS, EmotionalTone.FEAR],
            EmotionalTone.TRUST: [EmotionalTone.FEAR, EmotionalTone.ANGER],
            EmotionalTone.NEUTRAL: []  # Neutral doesn't conflict
        }
        
        conflicting_tones = []
        potential_conflicts = conflicts.get(expected_tone, [])
        
        for tone in potential_conflicts:
            if tone in detected_tones:
                conflicting_tones.append((tone, detected_tones[tone]))
        
        return conflicting_tones


# Factory functions for easy instantiation
def create_character_validator() -> CharacterConsistencyValidator:
    """Create a character consistency validator."""
    return CharacterConsistencyValidator()


def create_plot_validator() -> PlotConsistencyValidator:
    """Create a plot consistency validator."""
    return PlotConsistencyValidator()


def create_world_validator() -> WorldBuildingValidator:
    """Create a world building validator."""
    return WorldBuildingValidator()


def create_tone_validator() -> EmotionalToneValidator:
    """Create an emotional tone validator."""
    return EmotionalToneValidator()


def create_all_validators() -> Dict[str, BaseValidator]:
    """Create all validators."""
    return {
        "character": create_character_validator(),
        "plot": create_plot_validator(),
        "world": create_world_validator(),
        "tone": create_tone_validator()
    }