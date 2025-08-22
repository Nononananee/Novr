import asyncio
import pytest
from unittest.mock import AsyncMock
from memory.emotional_memory_system import EmotionalMemorySystem, EmotionalState, EmotionCategory, EmotionIntensity

@pytest.mark.asyncio
async def test_store_emotional_analysis():
    ems = EmotionalMemorySystem()
    ems.db_utils = AsyncMock()
    ems.db_utils.save_character_emotions = AsyncMock(return_value=None)

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

    result = await ems.store_emotional_analysis(emotional_states, run_id, scene_id)

    ems.db_utils.save_character_emotions.assert_called_once()
    assert result is True

@pytest.mark.asyncio
async def test_validate_emotional_consistency():
    ems = EmotionalMemorySystem()
    ems.db_utils = AsyncMock()

    # Mock get_character_emotional_history to return conflicting emotion
    ems.get_character_emotional_history = AsyncMock(return_value=[
        EmotionalState(
            character_name="Alice",
            dominant_emotion=EmotionCategory.JOY,
            intensity=EmotionIntensity.HIGH,
            emotion_vector={"joy": 0.9},
            trigger_event="",
            confidence_score=0.9,
            source_chunk_id="chunk1"
        )
    ])

    # New emotion conflicting with JOY is SADNESS
    result_conflict = await ems.validate_emotional_consistency("Alice", EmotionCategory.SADNESS)
    assert result_conflict is False

    # New emotion not conflicting
    result_no_conflict = await ems.validate_emotional_consistency("Alice", EmotionCategory.TRUST)
    assert result_no_conflict is True

@pytest.mark.asyncio
async def test_get_character_emotional_history_error_handling():
    ems = EmotionalMemorySystem()
    ems.db_utils = AsyncMock()

    # Simulate exception in db_utils.fetch_character_emotions
    async def raise_exception(*args, **kwargs):
        raise Exception("DB error")

    ems.db_utils.fetch_character_emotions = raise_exception

    result = await ems.get_character_emotional_history("Alice", limit=5)
    assert result == []

if __name__ == "__main__":
    asyncio.run(test_store_emotional_analysis())