"""
Factory for creating properly configured memory systems with all components.
"""

import logging
from typing import Optional

from .integrated_memory_system import IntegratedNovelMemorySystem
from .emotional_memory_system import EmotionalMemorySystem
from novel.character_management.repository import CharacterRepository
from novel.structure.narrative_structure_manager import NarrativeStructureManager
from novel.style.style_consistency_manager import StyleConsistencyManager

logger = logging.getLogger(__name__)


def create_integrated_memory_system(
    vectorstore_client=None,
    llm_client=None,
    character_repo: Optional[CharacterRepository] = None,
    db_utils=None,
    max_memory_tokens: int = 32000,
    consistency_level: str = "high"
) -> IntegratedNovelMemorySystem:
    """
    Factory function to create a fully configured IntegratedNovelMemorySystem
    with all components including emotional memory.
    
    Args:
        vectorstore_client: Vector store client for embeddings
        llm_client: LLM client for generation
        character_repo: Character repository for character management
        db_utils: Database utilities for emotional memory system
        max_memory_tokens: Maximum tokens for memory context
        consistency_level: Consistency checking level
        
    Returns:
        Fully configured IntegratedNovelMemorySystem
    """
    
    try:
        # Create the integrated memory system with all components
        memory_system = IntegratedNovelMemorySystem(
            vectorstore_client=vectorstore_client,
            llm_client=llm_client,
            character_repo=character_repo,
            db_utils=db_utils,
            max_memory_tokens=max_memory_tokens,
            consistency_level=consistency_level
        )
        
        logger.info("Successfully created IntegratedNovelMemorySystem with emotional memory")
        return memory_system
        
    except Exception as e:
        logger.error(f"Failed to create IntegratedNovelMemorySystem: {e}")
        # Create a fallback system without emotional memory
        logger.warning("Creating fallback memory system without emotional memory")
        
        return IntegratedNovelMemorySystem(
            vectorstore_client=vectorstore_client,
            llm_client=llm_client,
            character_repo=character_repo,
            db_utils=None,  # Disable emotional memory
            max_memory_tokens=max_memory_tokens,
            consistency_level=consistency_level
        )


def create_emotional_memory_system(db_utils=None) -> Optional[EmotionalMemorySystem]:
    """
    Factory function to create an EmotionalMemorySystem.
    
    Args:
        db_utils: Database utilities for the emotional memory system
        
    Returns:
        EmotionalMemorySystem instance or None if creation fails
    """
    
    try:
        if db_utils is None:
            logger.warning("No database utilities provided for emotional memory system")
            return None
            
        emotional_memory = EmotionalMemorySystem(db_utils=db_utils)
        logger.info("Successfully created EmotionalMemorySystem")
        return emotional_memory
        
    except Exception as e:
        logger.error(f"Failed to create EmotionalMemorySystem: {e}")
        return None


def validate_memory_system_components(memory_system: IntegratedNovelMemorySystem) -> dict:
    """
    Validate that all components of the memory system are properly initialized.
    
    Args:
        memory_system: The memory system to validate
        
    Returns:
        Dictionary with validation results
    """
    
    validation_results = {
        'memory_manager': False,
        'chunker': False,
        'consistency_manager': False,
        'emotional_memory': False,
        'structure_manager': False,
        'style_manager': False,
        'overall_status': 'failed'
    }
    
    try:
        # Check memory manager
        if hasattr(memory_system, 'memory_manager') and memory_system.memory_manager is not None:
            validation_results['memory_manager'] = True
            
        # Check chunker
        if hasattr(memory_system, 'chunker') and memory_system.chunker is not None:
            validation_results['chunker'] = True
            
        # Check consistency manager
        if hasattr(memory_system, 'consistency_manager') and memory_system.consistency_manager is not None:
            validation_results['consistency_manager'] = True
            
        # Check emotional memory
        if hasattr(memory_system, 'emotional_memory') and memory_system.emotional_memory is not None:
            validation_results['emotional_memory'] = True
            
        # Check structure manager
        if hasattr(memory_system, 'structure_manager') and memory_system.structure_manager is not None:
            validation_results['structure_manager'] = True
            
        # Check style manager
        if hasattr(memory_system, 'style_manager') and memory_system.style_manager is not None:
            validation_results['style_manager'] = True
            
        # Overall status
        core_components = ['memory_manager', 'chunker', 'consistency_manager']
        advanced_components = ['emotional_memory', 'structure_manager', 'style_manager']
        
        if all(validation_results[comp] for comp in core_components):
            advanced_count = sum(validation_results[comp] for comp in advanced_components)
            if advanced_count == len(advanced_components):
                validation_results['overall_status'] = 'complete'
            elif advanced_count > 0:
                validation_results['overall_status'] = 'partial'
            else:
                validation_results['overall_status'] = 'basic'
        else:
            validation_results['overall_status'] = 'failed'
            
    except Exception as e:
        logger.error(f"Error validating memory system components: {e}")
        validation_results['error'] = str(e)
    
    return validation_results