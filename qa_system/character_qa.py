import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import re

logger = logging.getLogger(__name__)

class CharacterQAAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize Character QA Agent for character consistency and dialogue authenticity
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized Character QA Agent with model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for character QA"""
        return """You are a professional character development editor specializing in character consistency, dialogue authenticity, and character voice analysis. Your role is to ensure characters remain true to their established personalities and relationships.

Your task is to analyze text and return a JSON response with the following structure:
{
    "score": integer (0-100, where 100 is perfect),
    "issues": [
        {
            "loc": integer (approximate character position),
            "type": "consistency|dialogue|voice|motivation|relationship",
            "issue": "description of the character issue",
            "suggestion": "suggested improvement",
            "severity": "low|medium|high",
            "character_name": "name of character involved"
        }
    ],
    "patches": [
        {
            "loc": integer (character position),
            "original": "text to replace",
            "replacement": "improved text"
        }
    ]
}

Focus on:
- Character consistency with established traits and personality
- Dialogue authenticity and unique character voice
- Character motivation alignment with actions
- Relationship dynamics consistency
- Character growth and development appropriateness
- Emotional responses fitting character psychology
- Speech patterns and vocabulary consistency

Scoring Guidelines:
- 90-100: Excellent character consistency and authentic dialogue
- 80-89: Good characterization, minor voice issues
- 70-79: Acceptable characters, some inconsistencies
- 60-69: Character issues affecting believability
- Below 60: Major character problems requiring revision

IMPORTANT: Return ONLY valid JSON. Do not include any other text or explanations."""
    
    async def review(self, text: str, character_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Review text for character consistency and dialogue issues
        
        Args:
            text: Text to review
            character_context: Character profiles and relationships
            
        Returns:
            Dictionary with score, issues, and patches
        """
        try:
            logger.info(f"Reviewing {len(text)} characters for character consistency")
            
            system_prompt = self._build_system_prompt()
            
            # Build character context information
            context_info = ""
            if character_context:
                if "characters" in character_context:
                    context_info += "\nCHARACTER PROFILES:\n"
                    for char in character_context["characters"]:
                        context_info += f"- {char.get('name', 'Unknown')}: {char.get('description', '')}\n"
                        if char.get('traits'):
                            context_info += f"  Traits: {', '.join(char['traits'])}\n"
                        if char.get('role'):
                            context_info += f"  Role: {char['role']}\n"
                
                if "relationship_network" in character_context:
                    context_info += "\nCHARACTER RELATIONSHIPS:\n"
                    for rel in character_context["relationship_network"]:
                        context_info += f"- {rel.get('character', '')} {rel.get('relationship', '')} {rel.get('related_to', '')}\n"
                        if rel.get('details'):
                            context_info += f"  Details: {rel['details']}\n"
            
            user_prompt = f"""Analyze the following text for character consistency, dialogue authenticity, and character voice:

{context_info}

TEXT TO REVIEW:
{text}

Focus on:
1. Do characters act consistently with their established personalities?
2. Is dialogue authentic to each character's voice and background?
3. Are character motivations clear and consistent?
4. Do character relationships reflect established dynamics?
5. Are emotional responses appropriate for each character?
6. Do speech patterns remain consistent for each character?

Return your analysis as valid JSON following the specified format."""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1500,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                qa_result = json.loads(response_text)
                
                # Validate required fields
                if not all(key in qa_result for key in ["score", "issues", "patches"]):
                    raise ValueError("Missing required fields in Character QA response")
                
                # Ensure score is within valid range
                qa_result["score"] = max(0, min(100, int(qa_result["score"])))
                
                # Validate and enrich issues
                for issue in qa_result["issues"]:
                    if not all(key in issue for key in ["loc", "type", "issue", "suggestion"]):
                        logger.warning(f"Invalid character issue structure: {issue}")
                    
                    # Ensure severity is set
                    if "severity" not in issue:
                        issue["severity"] = "medium"
                    
                    # Ensure character_name is set
                    if "character_name" not in issue:
                        issue["character_name"] = "Unknown"
                
                logger.info(f"Character QA completed - Score: {qa_result['score']}, "
                           f"Issues: {len(qa_result['issues'])}, Patches: {len(qa_result['patches'])}")
                
                return qa_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Character QA JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                return self._create_fallback_result(text, "JSON parsing error")
                
        except Exception as e:
            logger.error(f"Failed to perform Character QA review: {e}")
            return self._create_fallback_result(text, str(e))
    
    def _create_fallback_result(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback QA result when analysis fails"""
        basic_issues = []
        score = 75  # Default moderate score for character analysis
        
        # Basic character consistency heuristics
        
        # Check for dialogue presence and formatting
        dialogue_count = text.count('"')
        if dialogue_count > 0:
            # Check for proper dialogue formatting
            if not re.search(r'"\s*[A-Z]', text):
                basic_issues.append({
                    "loc": text.find('"'),
                    "type": "dialogue",
                    "issue": "Dialogue formatting may be inconsistent",
                    "suggestion": "Ensure dialogue starts with capital letters",
                    "severity": "low",
                    "character_name": "Unknown"
                })
                score -= 3
        
        # Check for character name consistency
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', text)
        name_counts = {}
        for name in potential_names:
            if len(name) > 2:  # Filter out short words
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Look for potential character names mentioned multiple times
        main_characters = [name for name, count in name_counts.items() if count > 1]
        
        if not main_characters and len(text.split()) > 200:
            basic_issues.append({
                "loc": 0,
                "type": "consistency",
                "issue": "No clear character names identified in substantial text",
                "suggestion": "Ensure character names are used consistently",
                "severity": "medium",
                "character_name": "Unknown"
            })
            score -= 10
        
        # Check for emotional consistency indicators
        emotion_words = ['angry', 'sad', 'happy', 'excited', 'worried', 'calm', 'frustrated']
        emotions_found = [word for word in emotion_words if word in text.lower()]
        
        if len(emotions_found) > 3:
            # Multiple emotions might indicate inconsistency
            basic_issues.append({
                "loc": text.lower().find(emotions_found[0]),
                "type": "consistency",
                "issue": "Multiple emotional states detected - check for consistency",
                "suggestion": "Ensure emotional transitions are logical and well-motivated",
                "severity": "low",
                "character_name": "Unknown"
            })
            score -= 5
        
        return {
            "score": max(0, score),
            "issues": basic_issues,
            "patches": [],
            "error": f"Character QA analysis failed: {error_msg}",
            "fallback": True
        }
    
    def get_quality_summary(self, qa_result: Dict[str, Any]) -> str:
        """Generate human-readable character quality summary"""
        score = qa_result.get("score", 0)
        issues = qa_result.get("issues", [])
        
        if score >= 90:
            quality_level = "Excellent Character Consistency"
        elif score >= 80:
            quality_level = "Good Character Development"
        elif score >= 70:
            quality_level = "Acceptable Character Work"
        elif score >= 60:
            quality_level = "Character Issues Present"
        else:
            quality_level = "Major Character Problems"
        
        issue_types = {}
        character_issues = {}
        
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            character_name = issue.get("character_name", "Unknown")
            
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            character_issues[character_name] = character_issues.get(character_name, 0) + 1
        
        summary = f"Character Quality: {quality_level} (Score: {score}/100)\n"
        
        if issues:
            summary += f"Character issues found: {len(issues)}\n"
            summary += "By type:\n"
            for issue_type, count in issue_types.items():
                summary += f"  - {issue_type}: {count}\n"
            
            if len(character_issues) > 1:
                summary += "By character:\n"
                for char_name, count in character_issues.items():
                    summary += f"  - {char_name}: {count}\n"
        else:
            summary += "No character issues found.\n"
        
        return summary