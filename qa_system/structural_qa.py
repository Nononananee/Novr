import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import re

logger = logging.getLogger(__name__)

class StructuralQAAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize Structural QA Agent for plot consistency and story structure analysis
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized Structural QA Agent with model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for structural QA"""
        return """You are a professional story editor specializing in plot structure, pacing, and narrative consistency. Your role is to analyze story content for structural issues and provide improvement suggestions.

Your task is to analyze text and return a JSON response with the following structure:
{
    "score": integer (0-100, where 100 is perfect),
    "issues": [
        {
            "loc": integer (approximate character position),
            "type": "plot|pacing|structure|continuity|transition",
            "issue": "description of the structural issue",
            "suggestion": "suggested improvement",
            "severity": "low|medium|high"
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
- Plot consistency and logical flow
- Story pacing (too fast/slow, uneven rhythm)
- Chapter/scene structure and organization
- Narrative continuity and timeline consistency
- Smooth transitions between scenes/paragraphs
- Character motivation consistency
- Conflict development and resolution
- Story arc progression

Scoring Guidelines:
- 90-100: Excellent structure, compelling flow
- 80-89: Good structure, minor pacing issues
- 70-79: Acceptable structure, some inconsistencies
- 60-69: Structural problems affecting readability
- Below 60: Major structural issues requiring revision

IMPORTANT: Return ONLY valid JSON. Do not include any other text or explanations."""
    
    async def review(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Review text for structural issues
        
        Args:
            text: Text to review
            context: Additional context (plot outline, character info, etc.)
            
        Returns:
            Dictionary with score, issues, and patches
        """
        try:
            logger.info(f"Reviewing {len(text)} characters for structural issues")
            
            system_prompt = self._build_system_prompt()
            
            # Build context information
            context_info = ""
            if context:
                if "plot_outline" in context:
                    context_info += f"\nPLOT OUTLINE: {context['plot_outline']}"
                if "chapter_summary" in context:
                    context_info += f"\nCHAPTER SUMMARY: {context['chapter_summary']}"
                if "previous_events" in context:
                    context_info += f"\nPREVIOUS EVENTS: {context['previous_events']}"
                if "story_arc_progress" in context:
                    context_info += f"\nSTORY ARC PROGRESS: {context['story_arc_progress']}"
            
            user_prompt = f"""Analyze the following text for plot consistency, pacing, and structural issues:

{context_info}

TEXT TO REVIEW:
{text}

Focus on:
1. Does the plot flow logically from previous events?
2. Is the pacing appropriate for this part of the story?
3. Are scene transitions smooth and natural?
4. Does the chapter structure serve the narrative effectively?
5. Are character actions consistent with their motivations?
6. Does the conflict develop appropriately?

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
                    raise ValueError("Missing required fields in Structural QA response")
                
                # Ensure score is within valid range
                qa_result["score"] = max(0, min(100, int(qa_result["score"])))
                
                # Validate and enrich issues
                for issue in qa_result["issues"]:
                    if not all(key in issue for key in ["loc", "type", "issue", "suggestion"]):
                        logger.warning(f"Invalid structural issue structure: {issue}")
                    
                    # Ensure severity is set
                    if "severity" not in issue:
                        issue["severity"] = "medium"
                
                logger.info(f"Structural QA completed - Score: {qa_result['score']}, "
                           f"Issues: {len(qa_result['issues'])}, Patches: {len(qa_result['patches'])}")
                
                return qa_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Structural QA JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                return self._create_fallback_result(text, "JSON parsing error")
                
        except Exception as e:
            logger.error(f"Failed to perform Structural QA review: {e}")
            return self._create_fallback_result(text, str(e))
    
    def _create_fallback_result(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback QA result when analysis fails"""
        word_count = len(text.split())
        
        basic_issues = []
        score = 70  # Default moderate score for structural analysis
        
        # Basic structural heuristics
        paragraphs = text.split('\n\n')
        
        # Check for very short or very long paragraphs
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50 and paragraph.strip():
                basic_issues.append({
                    "loc": text.find(paragraph),
                    "type": "structure",
                    "issue": "Very short paragraph may disrupt pacing",
                    "suggestion": "Consider expanding or combining with adjacent paragraphs",
                    "severity": "low"
                })
                score -= 3
            elif len(paragraph.strip()) > 1000:
                basic_issues.append({
                    "loc": text.find(paragraph),
                    "type": "pacing",
                    "issue": "Very long paragraph may slow pacing",
                    "suggestion": "Consider breaking into smaller paragraphs",
                    "severity": "medium"
                })
                score -= 5
        
        # Check for dialogue balance
        dialogue_markers = text.count('"')
        if dialogue_markers < 4 and word_count > 500:
            basic_issues.append({
                "loc": 0,
                "type": "structure",
                "issue": "Limited dialogue may affect pacing and character development",
                "suggestion": "Consider adding dialogue to break up narrative",
                "severity": "low"
            })
            score -= 5
        
        # Check for scene transitions
        if not re.search(r'\n\s*\n', text) and word_count > 300:
            basic_issues.append({
                "loc": len(text) // 2,
                "type": "transition",
                "issue": "No clear scene breaks detected",
                "suggestion": "Consider adding paragraph breaks for better pacing",
                "severity": "medium"
            })
            score -= 8
        
        return {
            "score": max(0, score),
            "issues": basic_issues,
            "patches": [],
            "error": f"Structural QA analysis failed: {error_msg}",
            "fallback": True
        }
    
    def get_quality_summary(self, qa_result: Dict[str, Any]) -> str:
        """Generate human-readable structural quality summary"""
        score = qa_result.get("score", 0)
        issues = qa_result.get("issues", [])
        
        if score >= 90:
            quality_level = "Excellent Structure"
        elif score >= 80:
            quality_level = "Good Structure"
        elif score >= 70:
            quality_level = "Acceptable Structure"
        elif score >= 60:
            quality_level = "Needs Structural Improvement"
        else:
            quality_level = "Major Structural Issues"
        
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        summary = f"Structural Quality: {quality_level} (Score: {score}/100)\n"
        
        if issues:
            summary += f"Structural issues found: {len(issues)}\n"
            for issue_type, count in issue_types.items():
                summary += f"  - {issue_type}: {count}\n"
        else:
            summary += "No structural issues found.\n"
        
        return summary