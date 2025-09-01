import os
import logging
from typing import Dict, Any, Optional
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class GeneratorAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize Generator Agent for novel writing - supports OpenRouter and OpenAI
        
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
        
        logger.info(f"Initialized Generator Agent with {self.service} - model: {model}")
    
    def _build_system_prompt(self, tone: str = "engaging", style: str = "narrative") -> str:
        """Build system prompt for the generator"""
        return f"""You are a professional novelist with expertise in creative writing and storytelling. Your role is to write compelling, well-structured novel chapters that engage readers and advance the story.

Writing Guidelines:
- Tone: {tone}
- Style: {style}
- Write in third person narrative unless otherwise specified
- Create vivid, immersive descriptions that bring scenes to life
- Develop authentic dialogue that reveals character personality
- Maintain consistent pacing and narrative flow
- Show rather than tell whenever possible
- Use sensory details to enhance reader engagement

Quality Standards:
- Ensure proper grammar, spelling, and punctuation
- Maintain consistency with established characters and world-building
- Create smooth transitions between scenes and paragraphs
- Build tension and maintain reader interest throughout
- End chapters with hooks or natural stopping points

Important Notes:
- Do NOT contradict any information provided in the context
- If uncertain about any detail, mark it with [VERIFY] for later review
- Focus on advancing the plot while developing characters
- Maintain the established voice and perspective of the story
- Output should be in Markdown format with proper chapter structure"""
    
    def _build_user_prompt(self, prompt: str, context: str, length_words: int = 1200) -> str:
        """Build user prompt with context and instructions"""
        user_prompt = f"""CONTEXT (Important background information - do not contradict):
{context if context else "No specific context provided."}

TASK: {prompt}

CONSTRAINTS:
- Target length: approximately {length_words} words
- Maintain consistency with the provided context
- Create engaging, publishable-quality prose
- If you're uncertain about any story details, mark them with [VERIFY]

OUTPUT: Write the chapter content in Markdown format. Begin with a chapter heading if appropriate."""
        
        return user_prompt
    
    async def generate(
        self,
        prompt: str,
        context: str = "",
        length_words: int = 1200,
        temperature: float = 0.7,
        tone: str = "engaging",
        style: str = "narrative",
        max_tokens: int = 2000
    ) -> str:
        """
        Generate novel chapter content
        
        Args:
            prompt: Generation instruction/prompt
            context: Relevant background context
            length_words: Target word count
            temperature: Creativity level (0.0-1.0)
            tone: Writing tone
            style: Writing style
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated chapter content
        """
        try:
            logger.info(f"Generating content: {len(prompt)} char prompt, {len(context)} char context")
            
            system_prompt = self._build_system_prompt(tone, style)
            user_prompt = self._build_user_prompt(prompt, context, length_words)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            generated_content = response.choices[0].message.content
            
            # Log generation stats
            usage = response.usage
            logger.info(f"Generated {len(generated_content)} characters. "
                       f"Tokens used: {usage.total_tokens} (prompt: {usage.prompt_tokens}, "
                       f"completion: {usage.completion_tokens})")
            
            return generated_content
            
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise
    
    async def revise_with_feedback(
        self,
        original_content: str,
        feedback: str,
        context: str = "",
        temperature: float = 0.6
    ) -> str:
        """
        Revise content based on feedback
        
        Args:
            original_content: Original generated content
            feedback: Feedback/issues to address
            context: Relevant background context
            temperature: Creativity level for revision
            
        Returns:
            Revised content
        """
        try:
            logger.info(f"Revising content based on feedback: {len(feedback)} char feedback")
            
            system_prompt = """You are a professional editor and novelist. Your task is to revise and improve existing novel content based on specific feedback while maintaining the story's voice, style, and narrative flow.

Revision Guidelines:
- Address all issues mentioned in the feedback
- Maintain the original story structure and key plot points
- Improve clarity, flow, and readability
- Fix any grammar, spelling, or punctuation errors
- Enhance descriptions and dialogue as needed
- Ensure consistency with provided context
- Keep the same approximate length unless feedback suggests otherwise"""
            
            user_prompt = f"""CONTEXT (Important background - maintain consistency):
{context if context else "No specific context provided."}

ORIGINAL CONTENT:
{original_content}

FEEDBACK TO ADDRESS:
{feedback}

TASK: Revise the content to address the feedback while maintaining the story's quality and flow. Output the improved version in Markdown format."""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2500,
                top_p=0.9
            )
            
            revised_content = response.choices[0].message.content
            
            logger.info(f"Revised content: {len(revised_content)} characters")
            return revised_content
            
        except Exception as e:
            logger.error(f"Failed to revise content: {e}")
            raise
    
    def estimate_word_count(self, text: str) -> int:
        """Estimate word count of generated text"""
        return len(text.split())
    
    def validate_content(self, content: str) -> Dict[str, Any]:
        """Basic validation of generated content"""
        word_count = self.estimate_word_count(content)
        
        validation = {
            "word_count": word_count,
            "character_count": len(content),
            "has_verify_marks": "[VERIFY]" in content,
            "is_markdown": content.strip().startswith("#") or "**" in content or "*" in content,
            "min_length_met": word_count >= 500,  # Minimum reasonable chapter length
            "max_length_exceeded": word_count > 5000  # Maximum reasonable chapter length
        }
        
        return validation