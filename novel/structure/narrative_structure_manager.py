"""
Narrative Structure Management System for maintaining coherent story structures.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class NarrativeStructure(Enum):
    """Types of narrative structures."""
    THREE_ACT = "three_act"
    HEROS_JOURNEY = "heros_journey"
    FIVE_ACT = "five_act"
    FREYTAGS_PYRAMID = "freytags_pyramid"
    SEVEN_POINT_STORY = "seven_point_story"
    CUSTOM = "custom"


class StructureStage(Enum):
    """Stages within narrative structures."""
    # Three-Act Structure
    SETUP = "setup"
    CONFRONTATION = "confrontation"
    RESOLUTION = "resolution"
    
    # Hero's Journey
    ORDINARY_WORLD = "ordinary_world"
    CALL_TO_ADVENTURE = "call_to_adventure"
    REFUSAL_OF_CALL = "refusal_of_call"
    MEETING_MENTOR = "meeting_mentor"
    CROSSING_THRESHOLD = "crossing_threshold"
    TESTS_ALLIES_ENEMIES = "tests_allies_enemies"
    APPROACH_INMOST_CAVE = "approach_inmost_cave"
    ORDEAL = "ordeal"
    REWARD = "reward"
    ROAD_BACK = "road_back"
    RESURRECTION = "resurrection"
    RETURN_ELIXIR = "return_elixir"
    
    # Freytag's Pyramid
    EXPOSITION = "exposition"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    DENOUEMENT = "denouement"


@dataclass
class StructurePoint:
    """A specific point in the narrative structure."""
    stage: StructureStage
    expected_position: float  # 0.0 to 1.0 (percentage through story)
    description: str
    key_elements: List[str] = field(default_factory=list)
    character_arcs: Dict[str, str] = field(default_factory=dict)
    plot_threads: List[str] = field(default_factory=list)
    emotional_tone: Optional[str] = None
    pacing: str = "medium"  # slow, medium, fast
    
    
@dataclass
class StructureTemplate:
    """Template for a narrative structure."""
    name: str
    structure_type: NarrativeStructure
    description: str
    structure_points: List[StructurePoint]
    total_acts: int
    recommended_length: Dict[str, int]  # word counts for different sections
    
    
@dataclass
class StructureValidationResult:
    """Result of structure validation."""
    is_valid: bool
    current_stage: StructureStage
    expected_stage: StructureStage
    position_accuracy: float  # 0.0 to 1.0
    missing_elements: List[str]
    structural_issues: List[str]
    recommendations: List[str]
    confidence_score: float


class NarrativeStructureManager:
    """Manages narrative structure templates and validation."""
    
    def __init__(self, db_utils=None):
        self.db_utils = db_utils
        self.structure_templates = self._initialize_structure_templates()
        self.active_structure: Optional[StructureTemplate] = None
        self.current_position: float = 0.0
        self.structure_history: List[Dict[str, Any]] = []
        
    def _initialize_structure_templates(self) -> Dict[NarrativeStructure, StructureTemplate]:
        """Initialize predefined narrative structure templates."""
        
        templates = {}
        
        # Three-Act Structure
        three_act_points = [
            StructurePoint(
                stage=StructureStage.SETUP,
                expected_position=0.25,
                description="Introduce characters, world, and initial conflict",
                key_elements=["character_introduction", "world_building", "inciting_incident"],
                emotional_tone="establishing",
                pacing="medium"
            ),
            StructurePoint(
                stage=StructureStage.CONFRONTATION,
                expected_position=0.75,
                description="Main conflict development and complications",
                key_elements=["rising_tension", "character_development", "obstacles"],
                emotional_tone="escalating",
                pacing="fast"
            ),
            StructurePoint(
                stage=StructureStage.RESOLUTION,
                expected_position=1.0,
                description="Conflict resolution and conclusion",
                key_elements=["climax", "resolution", "character_growth"],
                emotional_tone="resolving",
                pacing="medium"
            )
        ]
        
        templates[NarrativeStructure.THREE_ACT] = StructureTemplate(
            name="Three-Act Structure",
            structure_type=NarrativeStructure.THREE_ACT,
            description="Classic three-act dramatic structure",
            structure_points=three_act_points,
            total_acts=3,
            recommended_length={"act1": 25000, "act2": 50000, "act3": 25000}
        )
        
        # Hero's Journey (simplified)
        heros_journey_points = [
            StructurePoint(
                stage=StructureStage.ORDINARY_WORLD,
                expected_position=0.1,
                description="Hero's normal life before transformation",
                key_elements=["character_establishment", "normal_world"],
                emotional_tone="stable"
            ),
            StructurePoint(
                stage=StructureStage.CALL_TO_ADVENTURE,
                expected_position=0.15,
                description="Hero faces a problem or challenge",
                key_elements=["inciting_incident", "call_to_action"],
                emotional_tone="disruption"
            ),
            StructurePoint(
                stage=StructureStage.CROSSING_THRESHOLD,
                expected_position=0.25,
                description="Hero commits to the adventure",
                key_elements=["commitment", "point_of_no_return"],
                emotional_tone="determination"
            ),
            StructurePoint(
                stage=StructureStage.TESTS_ALLIES_ENEMIES,
                expected_position=0.5,
                description="Hero faces challenges and makes allies",
                key_elements=["character_development", "relationship_building", "obstacles"],
                emotional_tone="growth"
            ),
            StructurePoint(
                stage=StructureStage.ORDEAL,
                expected_position=0.75,
                description="Hero faces greatest fear or challenge",
                key_elements=["climax", "transformation", "crisis"],
                emotional_tone="intense"
            ),
            StructurePoint(
                stage=StructureStage.RETURN_ELIXIR,
                expected_position=1.0,
                description="Hero returns transformed with wisdom",
                key_elements=["resolution", "wisdom_gained", "new_equilibrium"],
                emotional_tone="triumphant"
            )
        ]
        
        templates[NarrativeStructure.HEROS_JOURNEY] = StructureTemplate(
            name="Hero's Journey",
            structure_type=NarrativeStructure.HEROS_JOURNEY,
            description="Joseph Campbell's monomyth structure",
            structure_points=heros_journey_points,
            total_acts=3,
            recommended_length={"setup": 20000, "journey": 60000, "return": 20000}
        )
        
        return templates
    
    def set_active_structure(self, structure_type: NarrativeStructure) -> bool:
        """Set the active narrative structure for the story."""
        
        try:
            if structure_type in self.structure_templates:
                self.active_structure = self.structure_templates[structure_type]
                logger.info(f"Set active structure to: {self.active_structure.name}")
                return True
            else:
                logger.error(f"Structure type {structure_type} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting active structure: {e}")
            return False
    
    def update_story_position(self, current_word_count: int, total_expected_words: int):
        """Update the current position in the story."""
        
        try:
            self.current_position = min(1.0, current_word_count / total_expected_words)
            logger.debug(f"Updated story position to: {self.current_position:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating story position: {e}")
    
    def get_current_stage(self) -> Optional[StructureStage]:
        """Get the current narrative stage based on story position."""
        
        if not self.active_structure:
            return None
        
        try:
            # Find the current stage based on position
            for point in self.active_structure.structure_points:
                if self.current_position <= point.expected_position:
                    return point.stage
            
            # If we're past all points, return the last stage
            return self.active_structure.structure_points[-1].stage
            
        except Exception as e:
            logger.error(f"Error getting current stage: {e}")
            return None
    
    def validate_structure_adherence(self, 
                                   current_chapter: int,
                                   current_word_count: int,
                                   total_expected_words: int,
                                   plot_threads: List[str],
                                   character_arcs: Dict[str, str]) -> StructureValidationResult:
        """Validate how well the story adheres to the chosen structure."""
        
        if not self.active_structure:
            return StructureValidationResult(
                is_valid=False,
                current_stage=StructureStage.SETUP,
                expected_stage=StructureStage.SETUP,
                position_accuracy=0.0,
                missing_elements=[],
                structural_issues=["No active structure set"],
                recommendations=["Set an active narrative structure"],
                confidence_score=0.0
            )
        
        try:
            # Update position
            self.update_story_position(current_word_count, total_expected_words)
            
            # Get current and expected stages
            current_stage = self.get_current_stage()
            expected_stage = self._get_expected_stage_for_position(self.current_position)
            
            # Calculate position accuracy
            position_accuracy = self._calculate_position_accuracy(current_stage, expected_stage)
            
            # Check for missing elements
            missing_elements = self._check_missing_elements(current_stage, plot_threads, character_arcs)
            
            # Identify structural issues
            structural_issues = self._identify_structural_issues(
                current_stage, expected_stage, self.current_position
            )
            
            # Generate recommendations
            recommendations = self._generate_structure_recommendations(
                current_stage, expected_stage, missing_elements, structural_issues
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_structure_confidence(
                position_accuracy, missing_elements, structural_issues
            )
            
            is_valid = len(structural_issues) == 0 and position_accuracy > 0.7
            
            return StructureValidationResult(
                is_valid=is_valid,
                current_stage=current_stage or StructureStage.SETUP,
                expected_stage=expected_stage or StructureStage.SETUP,
                position_accuracy=position_accuracy,
                missing_elements=missing_elements,
                structural_issues=structural_issues,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error validating structure adherence: {e}")
            return StructureValidationResult(
                is_valid=False,
                current_stage=StructureStage.SETUP,
                expected_stage=StructureStage.SETUP,
                position_accuracy=0.0,
                missing_elements=[],
                structural_issues=[f"Validation error: {e}"],
                recommendations=["Fix validation errors"],
                confidence_score=0.0
            )
    
    def get_structure_guidance(self, target_stage: Optional[StructureStage] = None) -> Dict[str, Any]:
        """Get guidance for the current or target narrative stage."""
        
        if not self.active_structure:
            return {"error": "No active structure set"}
        
        try:
            stage = target_stage or self.get_current_stage()
            if not stage:
                return {"error": "Could not determine current stage"}
            
            # Find the structure point for this stage
            structure_point = None
            for point in self.active_structure.structure_points:
                if point.stage == stage:
                    structure_point = point
                    break
            
            if not structure_point:
                return {"error": f"No guidance available for stage: {stage}"}
            
            return {
                "stage": stage.value,
                "description": structure_point.description,
                "key_elements": structure_point.key_elements,
                "emotional_tone": structure_point.emotional_tone,
                "pacing": structure_point.pacing,
                "character_arcs": structure_point.character_arcs,
                "plot_threads": structure_point.plot_threads,
                "expected_position": structure_point.expected_position,
                "current_position": self.current_position
            }
            
        except Exception as e:
            logger.error(f"Error getting structure guidance: {e}")
            return {"error": str(e)}
    
    def _get_expected_stage_for_position(self, position: float) -> Optional[StructureStage]:
        """Get the expected stage for a given position in the story."""
        
        if not self.active_structure:
            return None
        
        for point in self.active_structure.structure_points:
            if position <= point.expected_position:
                return point.stage
        
        return self.active_structure.structure_points[-1].stage
    
    def _calculate_position_accuracy(self, current_stage: Optional[StructureStage], 
                                   expected_stage: Optional[StructureStage]) -> float:
        """Calculate how accurately positioned the story is structurally."""
        
        if not current_stage or not expected_stage:
            return 0.0
        
        if current_stage == expected_stage:
            return 1.0
        
        # Calculate distance between stages
        if not self.active_structure:
            return 0.0
        
        current_index = -1
        expected_index = -1
        
        for i, point in enumerate(self.active_structure.structure_points):
            if point.stage == current_stage:
                current_index = i
            if point.stage == expected_stage:
                expected_index = i
        
        if current_index == -1 or expected_index == -1:
            return 0.0
        
        # Calculate accuracy based on distance
        distance = abs(current_index - expected_index)
        max_distance = len(self.active_structure.structure_points) - 1
        
        return max(0.0, 1.0 - (distance / max_distance))
    
    def _check_missing_elements(self, current_stage: Optional[StructureStage],
                              plot_threads: List[str],
                              character_arcs: Dict[str, str]) -> List[str]:
        """Check for missing structural elements."""
        
        missing = []
        
        if not current_stage or not self.active_structure:
            return missing
        
        # Find the structure point for current stage
        structure_point = None
        for point in self.active_structure.structure_points:
            if point.stage == current_stage:
                structure_point = point
                break
        
        if not structure_point:
            return missing
        
        # Check for missing key elements
        for element in structure_point.key_elements:
            if element not in plot_threads:
                missing.append(f"Missing key element: {element}")
        
        # Check for missing character arcs
        for character, expected_arc in structure_point.character_arcs.items():
            if character not in character_arcs or character_arcs[character] != expected_arc:
                missing.append(f"Missing character arc: {character} -> {expected_arc}")
        
        return missing
    
    def _identify_structural_issues(self, current_stage: Optional[StructureStage],
                                  expected_stage: Optional[StructureStage],
                                  position: float) -> List[str]:
        """Identify structural issues in the story."""
        
        issues = []
        
        if not current_stage or not expected_stage:
            issues.append("Cannot determine story stage")
            return issues
        
        # Check if story is ahead or behind expected structure
        if current_stage != expected_stage:
            if self._is_stage_ahead(current_stage, expected_stage):
                issues.append(f"Story is ahead of expected structure (at {current_stage.value}, expected {expected_stage.value})")
            else:
                issues.append(f"Story is behind expected structure (at {current_stage.value}, expected {expected_stage.value})")
        
        # Check for pacing issues
        if position < 0.1 and current_stage != StructureStage.SETUP:
            issues.append("Story may be moving too fast in early stages")
        elif position > 0.9 and current_stage not in [StructureStage.RESOLUTION, StructureStage.DENOUEMENT]:
            issues.append("Story may be running long without proper resolution")
        
        return issues
    
    def _is_stage_ahead(self, current_stage: StructureStage, expected_stage: StructureStage) -> bool:
        """Check if current stage is ahead of expected stage."""
        
        if not self.active_structure:
            return False
        
        current_index = -1
        expected_index = -1
        
        for i, point in enumerate(self.active_structure.structure_points):
            if point.stage == current_stage:
                current_index = i
            if point.stage == expected_stage:
                expected_index = i
        
        return current_index > expected_index
    
    def _generate_structure_recommendations(self, current_stage: Optional[StructureStage],
                                          expected_stage: Optional[StructureStage],
                                          missing_elements: List[str],
                                          structural_issues: List[str]) -> List[str]:
        """Generate recommendations for improving structure adherence."""
        
        recommendations = []
        
        if missing_elements:
            recommendations.append("Address missing structural elements in upcoming content")
            for element in missing_elements[:3]:  # Top 3 missing elements
                recommendations.append(f"Consider adding: {element}")
        
        if structural_issues:
            if "ahead of expected structure" in str(structural_issues):
                recommendations.append("Consider slowing down the pace and developing current elements more deeply")
            elif "behind expected structure" in str(structural_issues):
                recommendations.append("Consider advancing the plot or introducing the next structural element")
        
        if current_stage and expected_stage and current_stage != expected_stage:
            recommendations.append(f"Focus on transitioning from {current_stage.value} to {expected_stage.value}")
        
        return recommendations
    
    def _calculate_structure_confidence(self, position_accuracy: float,
                                      missing_elements: List[str],
                                      structural_issues: List[str]) -> float:
        """Calculate confidence score for structure adherence."""
        
        base_score = position_accuracy
        
        # Penalize for missing elements
        element_penalty = len(missing_elements) * 0.1
        
        # Penalize for structural issues
        issue_penalty = len(structural_issues) * 0.15
        
        confidence = max(0.0, base_score - element_penalty - issue_penalty)
        
        return min(1.0, confidence)
    
    async def store_structure_checkpoint(self, chapter: int, word_count: int, 
                                       validation_result: StructureValidationResult):
        """Store a structure validation checkpoint."""
        
        if not self.db_utils:
            return
        
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "chapter": chapter,
                "word_count": word_count,
                "position": self.current_position,
                "structure_type": self.active_structure.structure_type.value if self.active_structure else None,
                "current_stage": validation_result.current_stage.value,
                "expected_stage": validation_result.expected_stage.value,
                "position_accuracy": validation_result.position_accuracy,
                "confidence_score": validation_result.confidence_score,
                "is_valid": validation_result.is_valid,
                "issues_count": len(validation_result.structural_issues),
                "missing_elements_count": len(validation_result.missing_elements)
            }
            
            await self.db_utils.execute_query(
                """
                INSERT INTO structure_checkpoints 
                (checkpoint_data, created_at) 
                VALUES ($1, $2)
                """,
                json.dumps(checkpoint_data),
                datetime.now()
            )
            
            logger.info(f"Stored structure checkpoint for chapter {chapter}")
            
        except Exception as e:
            logger.error(f"Error storing structure checkpoint: {e}")
    
    def get_available_structures(self) -> List[Dict[str, Any]]:
        """Get list of available narrative structures."""
        
        structures = []
        
        for structure_type, template in self.structure_templates.items():
            structures.append({
                "type": structure_type.value,
                "name": template.name,
                "description": template.description,
                "total_acts": template.total_acts,
                "stages": [point.stage.value for point in template.structure_points],
                "recommended_length": template.recommended_length
            })
        
        return structures