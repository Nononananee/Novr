# Helper functions for memory management system
from typing import List, Dict, Any
from datetime import datetime
import re
import hashlib

def extract_characters(content: str) -> List[str]:
    """Extract character names from content"""
    # Look for capitalized names (potential characters)
    potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    
    # Filter out common non-character words
    non_characters = {'The', 'This', 'That', 'Chapter', 'Book', 'Story', 'Time', 'Day', 'Night', 'Morning', 'Evening'}
    characters = [name for name in potential_names if name not in non_characters]
    
    return list(set(characters))  # Remove duplicates

def extract_plot_threads(content: str) -> List[str]:
    """Extract plot threads from content"""
    plot_keywords = {
        'mystery': ['mystery', 'secret', 'hidden', 'unknown', 'puzzle'],
        'romance': ['love', 'romance', 'relationship', 'heart', 'kiss'],
        'conflict': ['fight', 'battle', 'war', 'conflict', 'struggle'],
        'quest': ['journey', 'quest', 'mission', 'search', 'find'],
        'betrayal': ['betray', 'deceive', 'lie', 'trick', 'backstab']
    }
    
    threads = []
    content_lower = content.lower()
    
    for thread_type, keywords in plot_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            threads.append(thread_type)
    
    return threads

def extract_tags(content: str) -> List[str]:
    """Extract tags from content"""
    tags = []
    content_lower = content.lower()
    
    # Dialogue tags
    if '"' in content or "'" in content:
        tags.append('dialogue')
    
    # Action tags
    action_words = ['fight', 'run', 'jump', 'climb', 'battle', 'attack']
    if any(word in content_lower for word in action_words):
        tags.append('action')
    
    # Emotional tags
    emotion_words = ['sad', 'happy', 'angry', 'fear', 'joy', 'love', 'hate']
    if any(word in content_lower for word in emotion_words):
        tags.append('emotional')
    
    # Setting tags
    if any(word in content_lower for word in ['forest', 'castle', 'city', 'village', 'mountain']):
        tags.append('setting')
    
    return tags

def chunk_by_size(content: str, chunk_size: int = 1000, overlap_size: int = 200) -> List[str]:
    """Chunk content by size when it's too large"""
    words = content.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + 1 > chunk_size and current_chunk:
            # Add overlap
            overlap_words = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = overlap_words + [word]
            current_size = len(current_chunk)
        else:
            current_chunk.append(word)
            current_size += 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def generate_embedding(content: str) -> List[float]:
    """Generate embedding for content"""
    # Simple hash-based pseudo-embedding for demonstration
    content_hash = hashlib.md5(content.encode()).hexdigest()
    
    # Convert hash to list of floats (384 dimensions as example)
    embedding = []
    for i in range(0, len(content_hash), 2):
        hex_pair = content_hash[i:i+2]
        embedding.append(int(hex_pair, 16) / 255.0)  # Normalize to 0-1
    
    # Pad to 384 dimensions
    while len(embedding) < 384:
        embedding.append(0.0)
    
    return embedding[:384]

def validate_consistency_rules(content: str, established_facts: set) -> List[Dict[str, Any]]:
    """Validate content against established consistency rules"""
    issues = []
    
    # Check for contradictions with established facts
    for fact in established_facts:
        if 'never' in fact and any(word in content.lower() for word in fact.split()[1:]):
            # Check if content contradicts a "never" fact
            contradiction_words = fact.split()[1:]  # Remove "never"
            if any(word in content.lower() for word in contradiction_words):
                issues.append({
                    'type': 'fact_contradiction',
                    'description': f"Content contradicts established fact: {fact}",
                    'severity': 'high'
                })
    
    return issues

def extract_temporal_markers(content: str) -> List[str]:
    """Extract temporal markers from content"""
    patterns = [
        r'\b(meanwhile|later|earlier|the next day|hours later|weeks passed|months later|years ago)\b',
        r'\b\d+\s+(minutes?|hours?|days?|weeks?|months?|years?)\s+(later|ago|before|after)\b'
    ]
    
    markers = []
    for pattern in patterns:
        markers.extend(re.findall(pattern, content, re.IGNORECASE))
    
    return markers

def calculate_content_importance(content: str) -> float:
    """Calculate importance score for content"""
    importance_keywords = {
        'critical': ['death', 'betrayal', 'revelation', 'climax', 'murder', 'war'],
        'high': ['conflict', 'romance', 'discovery', 'battle', 'love', 'secret'],
        'medium': ['dialogue', 'conversation', 'meeting', 'journey', 'travel'],
        'low': ['description', 'atmosphere', 'setting', 'weather']
    }
    
    content_lower = content.lower()
    score = 0.3  # Base score
    
    for level, keywords in importance_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            if level == 'critical':
                score += 0.6
            elif level == 'high':
                score += 0.4
            elif level == 'medium':
                score += 0.2
            elif level == 'low':
                score += 0.1
    
    return min(score, 1.0)

def detect_narrative_elements(content: str) -> Dict[str, Any]:
    """Detect narrative elements in content"""
    elements = {
        'has_dialogue': '"' in content or "'" in content,
        'has_action': any(word in content.lower() for word in ['fight', 'run', 'battle', 'attack']),
        'has_emotion': any(word in content.lower() for word in ['sad', 'happy', 'angry', 'love', 'fear']),
        'has_description': any(word in content.lower() for word in ['beautiful', 'dark', 'bright', 'cold', 'warm']),
        'character_count': len(extract_characters(content)),
        'word_count': len(content.split()),
        'sentence_count': len([s for s in content.split('.') if s.strip()])
    }
    
    return elements
