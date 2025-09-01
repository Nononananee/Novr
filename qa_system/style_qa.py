import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import re

logger = logging.getLogger(__name__)

class StyleQAAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize Style QA Agent for writing style consistency and prose quality
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized Style QA Agent with model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for style QA"""
        return """You are a professional prose editor specializing in writing style, tone consistency, and literary quality. Your role is to ensure the writing maintains consistent style, appropriate tone, and high prose quality throughout.

Your task is to analyze text and return a JSON response with the following structure:
{
    "score": integer (0-100, where 100 is perfect),
    "issues": [
        {
            "loc": integer (approximate character position),
            "type": "tone|style|prose|repetition|flow|voice",
            "issue": "description of the style issue",
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
- Writing style consistency throughout the text
- Tone appropriateness and consistency
- Prose quality and literary merit
- Sentence variety and rhythm
- Word choice and vocabulary level
- Repetitive phrases or overused words
- Narrative voice consistency
- Show vs. tell balance
- Descriptive language effectiveness

Scoring Guidelines:
- 90-100: Excellent prose with consistent, engaging style
- 80-89: Good writing style, minor inconsistencies
- 70-79: Acceptable style, some areas need improvement
- 60-69: Style issues affecting reading experience
- Below 60: Major style problems requiring significant revision

IMPORTANT: Return ONLY valid JSON. Do not include any other text or explanations."""
    
    async def review(self, text: str, style_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Review text for style consistency and prose quality
        
        Args:
            text: Text to review
            style_context: Style guidelines and preferences
            
        Returns:
            Dictionary with score, issues, and patches
        """
        try:
            logger.info(f"Reviewing {len(text)} characters for style and prose quality")
            
            system_prompt = self._build_system_prompt()
            
            # Build style context information
            context_info = ""
            if style_context:
                if "target_tone" in style_context:
                    context_info += f"\nTARGET TONE: {style_context['target_tone']}"
                if "writing_style" in style_context:
                    context_info += f"\nWRITING STYLE: {style_context['writing_style']}"
                if "genre" in style_context:
                    context_info += f"\nGENRE: {style_context['genre']}"
                if "target_audience" in style_context:
                    context_info += f"\nTARGET AUDIENCE: {style_context['target_audience']}"
                if "style_notes" in style_context:
                    context_info += f"\nSTYLE NOTES: {style_context['style_notes']}"
            
            user_prompt = f"""Analyze the following text for writing style consistency, tone appropriateness, and prose quality:

{context_info}

TEXT TO REVIEW:
{text}

Focus on:
1. Is the writing style consistent throughout?
2. Is the tone appropriate and maintained?
3. Is the prose quality high and engaging?
4. Are there repetitive phrases or overused words?
5. Is there good sentence variety and rhythm?
6. Is the narrative voice consistent?
7. Is there effective balance of show vs. tell?
8. Are descriptions vivid and appropriate?

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
                    raise ValueError("Missing required fields in Style QA response")
                
                # Ensure score is within valid range
                qa_result["score"] = max(0, min(100, int(qa_result["score"])))
                
                # Validate and enrich issues
                for issue in qa_result["issues"]:
                    if not all(key in issue for key in ["loc", "type", "issue", "suggestion"]):
                        logger.warning(f"Invalid style issue structure: {issue}")
                    
                    # Ensure severity is set
                    if "severity" not in issue:
                        issue["severity"] = "medium"
                
                logger.info(f"Style QA completed - Score: {qa_result['score']}, "
                           f"Issues: {len(qa_result['issues'])}, Patches: {len(qa_result['patches'])}")
                
                return qa_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Style QA JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                return self._create_fallback_result(text, "JSON parsing error")
                
        except Exception as e:
            logger.error(f"Failed to perform Style QA review: {e}")
            return self._create_fallback_result(text, str(e))
    
    def _create_fallback_result(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback QA result when analysis fails"""
        basic_issues = []
        score = 75  # Default moderate score for style analysis
        
        # Basic style analysis heuristics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Check for repetitive words
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?";')
            if len(word_lower) > 4:  # Only check longer words
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        repetitive_words = [(word, count) for word, count in word_freq.items() 
                           if count > 3 and word not in ['that', 'with', 'have', 'this', 'they', 'were', 'been']]
        
        for word, count in repetitive_words[:3]:  # Report top 3 repetitive words
            basic_issues.append({
                "loc": text.lower().find(word),
                "type": "repetition",
                "issue": f"Word '{word}' appears {count} times - may be overused",
                "suggestion": f"Consider using synonyms or rephrasing to reduce repetition of '{word}'",
                "severity": "low" if count < 6 else "medium"
            })
            score -= 3
        
        # Check sentence length variety
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            # Check for lack of sentence variety
            short_sentences = sum(1 for length in sentence_lengths if length < 8)
            long_sentences = sum(1 for length in sentence_lengths if length > 20)
            
            if short_sentences / len(sentence_lengths) > 0.8:
                basic_issues.append({
                    "loc": 0,
                    "type": "style",
                    "issue": "Predominantly short sentences may create choppy rhythm",
                    "suggestion": "Consider combining some sentences for better flow",
                    "severity": "medium"
                })
                score -= 8
            
            elif long_sentences / len(sentence_lengths) > 0.6:
                basic_issues.append({
                    "loc": 0,
                    "type": "style",
                    "issue": "Many long sentences may affect readability",
                    "suggestion": "Consider breaking some long sentences for better pacing",
                    "severity": "medium"
                })
                score -= 6
        
        # Check for adverb overuse (basic check)
        adverbs = re.findall(r'\b\w+ly\b', text)
        if len(adverbs) > len(words) * 0.05:  # More than 5% adverbs
            basic_issues.append({
                "loc": text.find(adverbs[0]) if adverbs else 0,
                "type": "prose",
                "issue": f"High adverb usage detected ({len(adverbs)} adverbs)",
                "suggestion": "Consider replacing some adverbs with stronger verbs or more specific descriptions",
                "severity": "low"
            })
            score -= 5
        
        # Check for passive voice indicators
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(text.lower().count(indicator) for indicator in passive_indicators)
        
        if passive_count > len(words) * 0.08:  # More than 8% passive indicators
            basic_issues.append({
                "loc": 0,
                "type": "voice",
                "issue": "Possible overuse of passive voice",
                "suggestion": "Consider using more active voice constructions",
                "severity": "low"
            })
            score -= 4
        
        return {
            "score": max(0, score),
            "issues": basic_issues,
            "patches": [],
            "error": f"Style QA analysis failed: {error_msg}",
            "fallback": True
        }
    
    def get_quality_summary(self, qa_result: Dict[str, Any]) -> str:
        """Generate human-readable style quality summary"""
        score = qa_result.get("score", 0)
        issues = qa_result.get("issues", [])
        
        if score >= 90:
            quality_level = "Excellent Prose Style"
        elif score >= 80:
            quality_level = "Good Writing Style"
        elif score >= 70:
            quality_level = "Acceptable Style"
        elif score >= 60:
            quality_level = "Style Needs Improvement"
        else:
            quality_level = "Major Style Issues"
        
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        summary = f"Style Quality: {quality_level} (Score: {score}/100)\n"
        
        if issues:
            summary += f"Style issues found: {len(issues)}\n"
            for issue_type, count in issue_types.items():
                summary += f"  - {issue_type}: {count}\n"
        else:
            summary += "No style issues found.\n"
        
        return summary