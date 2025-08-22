"""
Character repository for managing character information and consistency.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """Represents a character in the novel."""
    name: str
    description: str
    personality_traits: List[str] = field(default_factory=list)
    background: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)
    character_arc: str = ""
    physical_description: str = ""
    dialogue_style: str = ""
    motivations: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CharacterInteraction:
    """Represents an interaction between characters."""
    character1: str
    character2: str
    interaction_type: str  # "dialogue", "conflict", "cooperation", etc.
    context: str
    emotional_tone: str
    chapter: int
    timestamp: datetime = field(default_factory=datetime.now)


class CharacterRepository:
    """Repository for managing character data and consistency."""
    
    def __init__(self, db_utils=None):
        """
        Initialize character repository.
        
        Args:
            db_utils: Database utilities for persistence (optional)
        """
        self.db_utils = db_utils
        self._characters: Dict[str, Character] = {}
        self._interactions: List[CharacterInteraction] = []
        
        # Statistics
        self.stats = {
            'total_characters': 0,
            'total_interactions': 0,
            'consistency_checks_performed': 0,
            'consistency_issues_found': 0
        }
    
    async def add_character(self, character: Character) -> bool:
        """
        Add a new character to the repository.
        
        Args:
            character: Character to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._characters[character.name] = character
            self.stats['total_characters'] += 1
            
            if self.db_utils:
                # Persist to database if available
                await self._persist_character(character)
            
            logger.info(f"Added character: {character.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add character {character.name}: {e}")
            return False
    
    async def get_character(self, name: str) -> Optional[Character]:
        """
        Get a character by name.
        
        Args:
            name: Character name
            
        Returns:
            Character if found, None otherwise
        """
        try:
            if name in self._characters:
                return self._characters[name]
            
            if self.db_utils:
                # Try to load from database
                character = await self._load_character(name)
                if character:
                    self._characters[name] = character
                    return character
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get character {name}: {e}")
            return None
    
    async def update_character(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update character information.
        
        Args:
            name: Character name
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            character = await self.get_character(name)
            if not character:
                logger.warning(f"Character {name} not found for update")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(character, key):
                    setattr(character, key, value)
            
            character.updated_at = datetime.now()
            
            if self.db_utils:
                await self._persist_character(character)
            
            logger.info(f"Updated character: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update character {name}: {e}")
            return False
    
    async def record_interaction(self, interaction: CharacterInteraction) -> bool:
        """
        Record an interaction between characters.
        
        Args:
            interaction: Character interaction to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._interactions.append(interaction)
            self.stats['total_interactions'] += 1
            
            if self.db_utils:
                await self._persist_interaction(interaction)
            
            logger.info(f"Recorded interaction between {interaction.character1} and {interaction.character2}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return False
    
    async def get_character_interactions(self, character_name: str) -> List[CharacterInteraction]:
        """
        Get all interactions involving a specific character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            List of interactions involving the character
        """
        try:
            interactions = [
                interaction for interaction in self._interactions
                if interaction.character1 == character_name or interaction.character2 == character_name
            ]
            
            if self.db_utils:
                # Load additional interactions from database
                db_interactions = await self._load_character_interactions(character_name)
                interactions.extend(db_interactions)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to get interactions for {character_name}: {e}")
            return []
    
    async def check_character_consistency(self, character_name: str, new_content: str) -> Dict[str, Any]:
        """
        Check if new content is consistent with character profile.
        
        Args:
            character_name: Name of the character
            new_content: New content to check
            
        Returns:
            Dictionary with consistency analysis results
        """
        try:
            self.stats['consistency_checks_performed'] += 1
            
            character = await self.get_character(character_name)
            if not character:
                return {
                    'is_consistent': False,
                    'issues': [f"Character {character_name} not found in repository"],
                    'confidence': 0.0
                }
            
            issues = []
            
            # Basic consistency checks
            # Check for personality trait consistency
            if character.personality_traits:
                # This is a simplified check - in a real implementation,
                # you'd use NLP to analyze the content
                content_lower = new_content.lower()
                for trait in character.personality_traits:
                    if trait.lower() in ['aggressive', 'violent'] and 'gently' in content_lower:
                        issues.append(f"Content suggests gentle behavior but {character_name} is characterized as {trait}")
            
            # Check dialogue style consistency
            if character.dialogue_style and '"' in new_content:
                # Simplified dialogue style check
                if character.dialogue_style == 'formal' and 'ain\'t' in new_content.lower():
                    issues.append(f"Informal language used but {character_name} has formal dialogue style")
            
            is_consistent = len(issues) == 0
            if not is_consistent:
                self.stats['consistency_issues_found'] += len(issues)
            
            return {
                'is_consistent': is_consistent,
                'issues': issues,
                'confidence': 0.8 if is_consistent else 0.3,
                'character_traits_checked': len(character.personality_traits),
                'dialogue_style_checked': bool(character.dialogue_style)
            }
            
        except Exception as e:
            logger.error(f"Failed to check character consistency for {character_name}: {e}")
            return {
                'is_consistent': False,
                'issues': [f"Error checking consistency: {str(e)}"],
                'confidence': 0.0
            }
    
    async def get_all_characters(self) -> List[Character]:
        """
        Get all characters in the repository.
        
        Returns:
            List of all characters
        """
        try:
            characters = list(self._characters.values())
            
            if self.db_utils:
                # Load additional characters from database
                db_characters = await self._load_all_characters()
                # Merge with in-memory characters (avoid duplicates)
                for db_char in db_characters:
                    if db_char.name not in self._characters:
                        characters.append(db_char)
                        self._characters[db_char.name] = db_char
            
            return characters
            
        except Exception as e:
            logger.error(f"Failed to get all characters: {e}")
            return []
    
    async def _persist_character(self, character: Character):
        """Persist character to database (if available)."""
        if not self.db_utils:
            return
        
        try:
            # This would implement actual database persistence
            # For now, it's a placeholder
            logger.debug(f"Persisting character {character.name} to database")
        except Exception as e:
            logger.error(f"Failed to persist character {character.name}: {e}")
    
    async def _load_character(self, name: str) -> Optional[Character]:
        """Load character from database (if available)."""
        if not self.db_utils:
            return None
        
        try:
            # This would implement actual database loading
            # For now, it's a placeholder
            logger.debug(f"Loading character {name} from database")
            return None
        except Exception as e:
            logger.error(f"Failed to load character {name}: {e}")
            return None
    
    async def _persist_interaction(self, interaction: CharacterInteraction):
        """Persist interaction to database (if available)."""
        if not self.db_utils:
            return
        
        try:
            # This would implement actual database persistence
            logger.debug(f"Persisting interaction to database")
        except Exception as e:
            logger.error(f"Failed to persist interaction: {e}")
    
    async def _load_character_interactions(self, character_name: str) -> List[CharacterInteraction]:
        """Load character interactions from database (if available)."""
        if not self.db_utils:
            return []
        
        try:
            # This would implement actual database loading
            logger.debug(f"Loading interactions for {character_name} from database")
            return []
        except Exception as e:
            logger.error(f"Failed to load interactions for {character_name}: {e}")
            return []
    
    async def _load_all_characters(self) -> List[Character]:
        """Load all characters from database (if available)."""
        if not self.db_utils:
            return []
        
        try:
            # This would implement actual database loading
            logger.debug("Loading all characters from database")
            return []
        except Exception as e:
            logger.error(f"Failed to load all characters: {e}")
            return []