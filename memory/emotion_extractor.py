import re
from typing import List, Dict, Any

# Based on Plutchik's wheel of emotions, simplified
EMOTION_KEYWORDS = {
    'joy': ['happy', 'joyful', 'excited', 'delighted', 'cheerful', 'elated', 'glee'],
    'trust': ['trust', 'faith', 'believe', 'confidence', 'reliance'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'panic', 'fearful'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
    'sadness': ['sad', 'melancholy', 'depressed', 'sorrowful', 'grief', 'mourning', 'unhappy'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'distaste'],
    'anger': ['angry', 'furious', 'rage', 'irritated', 'annoyed', 'livid', 'outraged'],
    'anticipation': ['anticipate', 'eager', 'expectant', 'impatient', 'waiting'],
}

EMOTION_CATEGORIES = {
    'positive': ['joy', 'trust', 'anticipation'],
    'negative': ['fear', 'sadness', 'disgust', 'anger'],
    'neutral': ['surprise']
}

def get_emotion_category(emotion: str) -> str:
    """Gets the high-level category for a given emotion."""
    for category, emotions in EMOTION_CATEGORIES.items():
        if emotion in emotions:
            return category
    return 'neutral'

def extract_emotions_from_chunk(chunk_content: str, characters: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts emotional states for characters from a text chunk.
    This is a basic keyword-based implementation and serves as a starting point.
    
    Args:
        chunk_content: The text content of the document chunk.
        characters: A list of character names present in the chunk.
        
    Returns:
        A list of dictionaries, each representing an emotional snapshot to be stored.
    """
    extractions = []
    sentences = re.split(r'(?<=[.!?]) +|

', chunk_content) # Split by sentence or paragraph
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        for character in characters:
            # Simple check if character is mentioned in the sentence or nearby context.
            # A more advanced system would use coreference resolution.
            if character.lower() in sentence.lower():
                
                emotion_scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS.keys()}
                total_hits = 0
                
                # Simple keyword matching within the sentence
                for emotion, keywords in EMOTION_KEYWORDS.items():
                    for keyword in keywords:
                        if f' {keyword}' in sentence.lower():
                            emotion_scores[emotion] += 1.0
                            total_hits += 1
                
                if total_hits > 0:
                    # Normalize scores
                    for emotion in emotion_scores:
                        emotion_scores[emotion] /= total_hits
                        
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    intensity = emotion_scores[dominant_emotion]
                    
                    # Create a result dictionary that matches the new database schema
                    extraction_data = {
                        "character_name": character,
                        "emotion_vector": emotion_scores,
                        "dominant_emotion": dominant_emotion,
                        "intensity": intensity,
                        "emotion_category": get_emotion_category(dominant_emotion),
                        "trigger_event": f"Presence in sentence: '{sentence[:75]}...'",
                        "source_type": 'narrative', # Default, could be improved with dialogue detection
                        "confidence_score": 0.5, # Placeholder for keyword method
                        "method": "keyword_v1",
                        "span_start": chunk_content.find(sentence),
                        "span_end": chunk_content.find(sentence) + len(sentence),
                        "sentence_index": i,
                        "intra_chunk_order": len(extractions)
                    }
                    extractions.append(extraction_data)

    return extractions
