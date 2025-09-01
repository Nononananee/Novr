import os
import json
import logging
from typing import Dict, Any, List, Optional
import openai
from openai import AsyncOpenAI
import re

logger = logging.getLogger(__name__)

class TechnicalQAAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize Technical QA Agent for grammar and formatting review - supports OpenRouter and OpenAI
        
        Args:
            api_key: API key (OpenRouter or OpenAI)
            model: Model name to use
        """
        # Try OpenRouter first, then fallback to OpenAI
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        
        # Determine which service to use
        if self.openrouter_api_key:
            self.api_key = self.openrouter_api_key
            self.base_url = "https://openrouter.ai/api/v1"
            self.service = "OpenRouter"
            logger.info(f"Using OpenRouter API with model: {model}")
        elif self.openai_api_key:
            self.api_key = self.openai_api_key
            self.base_url = None  # Use default OpenAI base URL
            self.service = "OpenAI"
            logger.info(f"Using OpenAI API with model: {model}")
        else:
            raise ValueError("Either OPENROUTER_API_KEY or OPENAI_API_KEY is required")
        
        # Initialize client with appropriate configuration
        if self.base_url:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized Technical QA Agent with {self.service} - model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for technical QA"""
        return """You are a professional technical editor specializing in grammar, spelling, punctuation, and formatting for creative writing. Your role is to identify and provide fixes for technical writing issues.

Your task is to analyze text and return a JSON response with the following structure:
{
    "score": integer (0-100, where 100 is perfect),
    "issues": [
        {
            "loc": integer (approximate character position),
            "type": "grammar|spelling|punctuation|formatting|style",
            "issue": "description of the issue",
            "suggestion": "suggested fix",
            "severity": "low|medium|high"
        }
    ],
    "patches": [
        {
            "loc": integer (character position),
            "original": "text to replace",
            "replacement": "corrected text"
        }
    ]
}

Focus on:
- Grammar errors (subject-verb agreement, tense consistency, etc.)
- Spelling mistakes
- Punctuation errors (missing/incorrect commas, periods, quotes, etc.)
- Formatting issues (inconsistent paragraph breaks, dialogue formatting)
- Basic style issues (repetitive words, awkward phrasing)

Scoring Guidelines:
- 90-100: Excellent, minimal issues
- 80-89: Good, minor issues
- 70-79: Acceptable, some issues to address
- 60-69: Needs improvement, multiple issues
- Below 60: Significant issues requiring revision

IMPORTANT: Return ONLY valid JSON. Do not include any other text or explanations."""
    
    async def review(self, text: str, review_type: str = "technical") -> Dict[str, Any]:
        """
        Review text for technical issues
        
        Args:
            text: Text to review
            review_type: Type of review (currently only 'technical')
            
        Returns:
            Dictionary with score, issues, and patches
        """
        try:
            logger.info(f"Reviewing {len(text)} characters for technical issues")
            
            system_prompt = self._build_system_prompt()
            user_prompt = f"""Analyze the following text for grammar, spelling, punctuation, and formatting issues:

TEXT TO REVIEW:
{text}

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
                    raise ValueError("Missing required fields in QA response")
                
                # Ensure score is within valid range
                qa_result["score"] = max(0, min(100, int(qa_result["score"])))
                
                # Validate issues structure
                for issue in qa_result["issues"]:
                    if not all(key in issue for key in ["loc", "type", "issue", "suggestion"]):
                        logger.warning(f"Invalid issue structure: {issue}")
                
                # Validate patches structure
                for patch in qa_result["patches"]:
                    if not all(key in patch for key in ["loc", "replacement"]):
                        logger.warning(f"Invalid patch structure: {patch}")
                
                logger.info(f"QA review completed - Score: {qa_result['score']}, "
                           f"Issues: {len(qa_result['issues'])}, Patches: {len(qa_result['patches'])}")
                
                return qa_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse QA JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                
                # Return fallback result
                return self._create_fallback_result(text, "JSON parsing error")
                
        except Exception as e:
            logger.error(f"Failed to perform QA review: {e}")
            return self._create_fallback_result(text, str(e))
    
    def _create_fallback_result(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback QA result when analysis fails"""
        # Basic heuristic scoring
        word_count = len(text.split())
        char_count = len(text)
        
        # Simple checks
        basic_issues = []
        score = 75  # Default moderate score
        
        # Check for common issues
        if re.search(r'\s{2,}', text):
            basic_issues.append({
                "loc": 0,
                "type": "formatting",
                "issue": "Multiple consecutive spaces found",
                "suggestion": "Use single spaces between words",
                "severity": "low"
            })
            score -= 5
        
        if not re.search(r'[.!?]$', text.strip()):
            basic_issues.append({
                "loc": len(text) - 1,
                "type": "punctuation", 
                "issue": "Text doesn't end with proper punctuation",
                "suggestion": "Add appropriate ending punctuation",
                "severity": "medium"
            })
            score -= 10
        
        # Check for very short or very long sentences
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 5 and sentence.strip():
                basic_issues.append({
                    "loc": text.find(sentence),
                    "type": "style",
                    "issue": "Very short sentence may need expansion",
                    "suggestion": "Consider expanding or combining with adjacent sentences",
                    "severity": "low"
                })
                score -= 2
        
        return {
            "score": max(0, score),
            "issues": basic_issues,
            "patches": [],
            "error": f"QA analysis failed: {error_msg}",
            "fallback": True
        }
    
    def apply_patches(self, text: str, patches: List[Dict[str, Any]]) -> str:
        """
        Apply patches to text
        
        Args:
            text: Original text
            patches: List of patch dictionaries
            
        Returns:
            Text with patches applied
        """
        if not patches:
            return text
        
        logger.info(f"Applying {len(patches)} patches to text")
        
        # Sort patches by location in reverse order to maintain positions
        sorted_patches = sorted(patches, key=lambda x: x.get("loc", 0), reverse=True)
        
        modified_text = text
        applied_count = 0
        
        for patch in sorted_patches:
            try:
                loc = patch.get("loc", 0)
                original = patch.get("original", "")
                replacement = patch.get("replacement", "")
                
                if original and original in modified_text:
                    # Find and replace the original text
                    modified_text = modified_text.replace(original, replacement, 1)
                    applied_count += 1
                elif loc < len(modified_text):
                    # Apply replacement at specific location
                    # This is a simplified approach - in practice, you'd want more sophisticated text manipulation
                    before = modified_text[:loc]
                    after = modified_text[loc:]
                    
                    # Try to find a reasonable boundary for replacement
                    word_boundary = after.find(' ')
                    if word_boundary > 0:
                        after = replacement + after[word_boundary:]
                        modified_text = before + after
                        applied_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to apply patch at location {patch.get('loc', 0)}: {e}")
        
        logger.info(f"Successfully applied {applied_count}/{len(patches)} patches")
        return modified_text
    
    def get_quality_summary(self, qa_result: Dict[str, Any]) -> str:
        """Generate a human-readable quality summary"""
        score = qa_result.get("score", 0)
        issues = qa_result.get("issues", [])
        
        if score >= 90:
            quality_level = "Excellent"
        elif score >= 80:
            quality_level = "Good"
        elif score >= 70:
            quality_level = "Acceptable"
        elif score >= 60:
            quality_level = "Needs Improvement"
        else:
            quality_level = "Poor"
        
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        summary = f"Quality: {quality_level} (Score: {score}/100)\n"
        
        if issues:
            summary += f"Issues found: {len(issues)}\n"
            for issue_type, count in issue_types.items():
                summary += f"  - {issue_type}: {count}\n"
        else:
            summary += "No issues found.\n"
        
        return summary