"""
Consistency validators for proposal validation (Fixed version).
"""

import logging
import re
from typing import Dict, List, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)


async def run_all_validators(
    content: str,
    entity_data: Dict[str, Any],
    established_facts: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Run all available validators on the content and entity data.
    
    Args:
        content: Text content to validate
        entity_data: Entity information to validate
        established_facts: Set of established facts from knowledge graph
    
    Returns:
        Dictionary of validator results
    """
    validators = {
        "fact_checker": await fact_check_validator(content, entity_data, established_facts),
        "behavior_consistency": await behavior_consistency_validator(content, entity_data),
        "dialogue_style": await dialogue_style_validator(content, entity_data),
        "trope_detector": await trope_detector_validator(content, entity_data),
        "timeline_consistency": await timeline_consistency_validator(content, entity_data)
    }
    
    return validators


async def fact_check_validator(
    content: str,
    entity_data: Dict[str, Any],
    established_facts: Set[str]
) -> Dict[str, Any]:
    """
    Validate facts against established knowledge.
    
    Checks for contradictions with known facts like birthplaces, relationships, etc.
    """
    try:
        violations = []
        suggestions = []
        score = 1.0
        
        # Extract factual claims from content
        factual_patterns = [
            r"born in (\w+)",
            r"lives in (\w+)",
            r"works at (\w+)",
            r"married to (\w+)",
            r"died in (\d{4})"
        ]
        
        for pattern in factual_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Fix f-string issue by doing string operations outside f-string
                entity_name = entity_data.get('name', 'unknown')
                pattern_text = pattern.replace('(\\w+)', match).replace('(\\d{4})', match)
                fact_claim = f"{entity_name} {pattern_text}"
                
                # Check against established facts (simplified)
                if any(fact_claim.lower() in established_fact.lower() for established_fact in established_facts):
                    # Potential contradiction found
                    violations.append({
                        "type": "fact_contradiction",
                        "claim": fact_claim,
                        "severity": "high",
                        "location": content.find(match)
                    })
                    score -= 0.2
                    suggestions.append(f"Verify claim: {fact_claim}")
        
        # Check for impossible dates
        current_year = datetime.now().year
        year_matches = re.findall(r'\b(19|20)\d{2}\b', content)
        for year_str in year_matches:
            year = int(year_str)
            if year > current_year:
                violations.append({
                    "type": "impossible_date",
                    "claim": f"Future date: {year}",
                    "severity": "medium",
                    "location": content.find(year_str)
                })
                score -= 0.1
                suggestions.append(f"Check date: {year} is in the future")
        
        return {
            "score": max(0.0, score),
            "violations": violations,
            "suggestions": suggestions,
            "validator_type": "fact_check"
        }
        
    except Exception as e:
        logger.error(f"Fact check validator failed: {e}")
        return {
            "score": 0.5,
            "violations": [{"type": "validator_error", "message": str(e)}],
            "suggestions": ["Manual fact checking required"],
            "validator_type": "fact_check"
        }


async def behavior_consistency_validator(
    content: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate character behavior consistency.
    
    Checks if character actions align with established personality traits.
    """
    try:
        violations = []
        suggestions = []
        score = 0.8  # Default score
        
        entity_name = entity_data.get("name", "")
        attributes = entity_data.get("attributes", {})
        
        # Define personality-behavior mappings
        personality_behaviors = {
            "aggressive": ["attack", "fight", "argue", "shout"],
            "peaceful": ["mediate", "calm", "negotiate", "help"],
            "intelligent": ["analyze", "study", "research", "calculate"],
            "emotional": ["cry", "laugh", "rage", "love"],
            "logical": ["reason", "deduce", "conclude", "prove"]
        }
        
        # Check for personality traits in attributes
        personality_traits = []
        for key, value in attributes.items():
            if key.lower() in ["personality", "trait", "character"]:
                if isinstance(value, str):
                    personality_traits.extend(value.lower().split())
                elif isinstance(value, list):
                    personality_traits.extend([str(v).lower() for v in value])
        
        # Analyze content for behavioral indicators
        content_lower = content.lower()
        
        # Check for consistency
        for trait in personality_traits:
            if trait in personality_behaviors:
                # Check for contradictory behaviors
                contradictory_traits = {
                    "aggressive": ["peaceful"],
                    "peaceful": ["aggressive"],
                    "logical": ["emotional"],
                    "emotional": ["logical"]
                }
                
                if trait in contradictory_traits:
                    for contrary_trait in contradictory_traits[trait]:
                        if contrary_trait in personality_behaviors:
                            contrary_behaviors = personality_behaviors[contrary_trait]
                            found_contrary = any(behavior in content_lower for behavior in contrary_behaviors)
                            
                            if found_contrary:
                                violations.append({
                                    "type": "behavior_contradiction",
                                    "trait": trait,
                                    "contrary_behavior": contrary_trait,
                                    "severity": "medium"
                                })
                                score -= 0.15
                                suggestions.append(f"Character shows {contrary_trait} behavior but is described as {trait}")
        
        return {
            "score": max(0.0, score),
            "violations": violations,
            "suggestions": suggestions,
            "validator_type": "behavior_consistency"
        }
        
    except Exception as e:
        logger.error(f"Behavior consistency validator failed: {e}")
        return {
            "score": 0.5,
            "violations": [{"type": "validator_error", "message": str(e)}],
            "suggestions": ["Manual behavior consistency check required"],
            "validator_type": "behavior_consistency"
        }


async def dialogue_style_validator(
    content: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate dialogue style consistency.
    
    Checks if character speech patterns are consistent.
    """
    try:
        violations = []
        suggestions = []
        score = 0.8
        
        # Extract dialogue from content
        dialogue_patterns = [
            r'"([^"]+)"',  # Standard quotes
            r"'([^']+)'",  # Single quotes
            r"said\s+[^:]+:\s*(.+)",  # "said X: content"
        ]
        
        dialogues = []
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dialogues.extend(matches)
        
        if not dialogues:
            return {
                "score": 1.0,
                "violations": [],
                "suggestions": [],
                "validator_type": "dialogue_style"
            }
        
        # Analyze dialogue characteristics
        total_words = 0
        formal_words = 0
        informal_words = 0
        
        formal_indicators = ["indeed", "certainly", "furthermore", "however", "therefore"]
        informal_indicators = ["yeah", "nah", "gonna", "wanna", "ain't"]
        
        for dialogue in dialogues:
            words = dialogue.lower().split()
            total_words += len(words)
            
            for word in words:
                if word in formal_indicators:
                    formal_words += 1
                elif word in informal_indicators:
                    informal_words += 1
        
        # Check for style consistency
        if total_words > 0:
            formal_ratio = formal_words / total_words
            informal_ratio = informal_words / total_words
            
            # Flag if character switches between very formal and very informal
            if formal_ratio > 0.1 and informal_ratio > 0.1:
                violations.append({
                    "type": "inconsistent_speech_style",
                    "formal_ratio": formal_ratio,
                    "informal_ratio": informal_ratio,
                    "severity": "low"
                })
                score -= 0.1
                suggestions.append("Character speech style varies between formal and informal")
        
        return {
            "score": max(0.0, score),
            "violations": violations,
            "suggestions": suggestions,
            "validator_type": "dialogue_style"
        }
        
    except Exception as e:
        logger.error(f"Dialogue style validator failed: {e}")
        return {
            "score": 0.5,
            "violations": [{"type": "validator_error", "message": str(e)}],
            "suggestions": ["Manual dialogue style check required"],
            "validator_type": "dialogue_style"
        }


async def trope_detector_validator(
    content: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect common tropes and clichés.
    
    Flags overused narrative elements.
    """
    try:
        violations = []
        suggestions = []
        score = 1.0
        
        # Common tropes to detect
        tropes = {
            "chosen_one": ["chosen one", "the one", "prophecy", "destiny"],
            "love_triangle": ["torn between", "choose between", "love triangle"],
            "mentor_death": ["mentor died", "teacher killed", "wise old"],
            "dark_past": ["dark secret", "hidden past", "mysterious background"],
            "amnesia": ["lost memory", "can't remember", "forgotten past"],
            "evil_twin": ["evil twin", "dark side", "doppelganger"]
        }
        
        content_lower = content.lower()
        
        for trope_name, keywords in tropes.items():
            for keyword in keywords:
                if keyword in content_lower:
                    violations.append({
                        "type": "common_trope",
                        "trope": trope_name,
                        "keyword": keyword,
                        "severity": "low"
                    })
                    score -= 0.05
                    suggestions.append(f"Consider avoiding the '{trope_name}' trope")
        
        # Check for overused adjectives
        overused_adjectives = ["amazing", "incredible", "unbelievable", "perfect", "flawless"]
        for adj in overused_adjectives:
            count = content_lower.count(adj)
            if count > 2:
                violations.append({
                    "type": "overused_word",
                    "word": adj,
                    "count": count,
                    "severity": "low"
                })
                score -= 0.02 * count
                suggestions.append(f"Word '{adj}' used {count} times - consider variety")
        
        return {
            "score": max(0.0, score),
            "violations": violations,
            "suggestions": suggestions,
            "validator_type": "trope_detector"
        }
        
    except Exception as e:
        logger.error(f"Trope detector validator failed: {e}")
        return {
            "score": 0.5,
            "violations": [{"type": "validator_error", "message": str(e)}],
            "suggestions": ["Manual trope detection required"],
            "validator_type": "trope_detector"
        }


async def timeline_consistency_validator(
    content: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate timeline consistency.
    
    Checks for temporal contradictions and impossible sequences.
    """
    try:
        violations = []
        suggestions = []
        score = 1.0
        
        # Extract temporal references
        temporal_patterns = [
            r"(\d+)\s+years?\s+ago",
            r"in\s+(\d{4})",
            r"(\d+)\s+years?\s+old",
            r"after\s+(\d+)\s+years?",
            r"before\s+(\d+)\s+years?"
        ]
        
        temporal_refs = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    temporal_refs.append(int(match))
                except ValueError:
                    continue
        
        # Check for impossible ages
        for age in temporal_refs:
            if age > 150:  # Unrealistic human age
                violations.append({
                    "type": "impossible_age",
                    "value": age,
                    "severity": "medium"
                })
                score -= 0.1
                suggestions.append(f"Age {age} seems unrealistic for human character")
            elif age < 0:
                violations.append({
                    "type": "negative_time",
                    "value": age,
                    "severity": "high"
                })
                score -= 0.2
                suggestions.append(f"Negative time value: {age}")
        
        # Check for sequence consistency
        sequence_words = ["first", "then", "next", "finally", "before", "after"]
        sequence_count = sum(1 for word in sequence_words if word in content.lower())
        
        if sequence_count > 5:
            # Many sequence words might indicate complex timeline
            suggestions.append("Complex timeline detected - verify sequence consistency")
        
        return {
            "score": max(0.0, score),
            "violations": violations,
            "suggestions": suggestions,
            "validator_type": "timeline_consistency"
        }
        
    except Exception as e:
        logger.error(f"Timeline consistency validator failed: {e}")
        return {
            "score": 0.5,
            "violations": [{"type": "validator_error", "message": str(e)}],
            "suggestions": ["Manual timeline consistency check required"],
            "validator_type": "timeline_consistency"
        }
