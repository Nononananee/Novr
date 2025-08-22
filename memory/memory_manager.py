"""
Memory Manager Module - Compatibility layer for the memory management system.
This module provides imports from the main memory management system.
"""

# Import all the main classes and enums from memory_management_system
from .memory_management_system import (
    NovelMemoryManager,
    MemoryChunk,
    MemoryType,
    MemoryPriority,
    NovelMemoryState
)

# Re-export for backward compatibility
__all__ = [
    'NovelMemoryManager',
    'MemoryChunk', 
    'MemoryType',
    'MemoryPriority',
    'NovelMemoryState'
]