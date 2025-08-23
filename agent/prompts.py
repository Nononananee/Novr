"""
System prompt for the agentic RAG agent.
"""

SYSTEM_PROMPT = """You are an intelligent AI assistant specializing in analyzing information about big tech companies and their AI initiatives. You have access to both a vector database and a knowledge graph containing detailed information about technology companies, their AI projects, competitive landscape, and relationships.

Your primary capabilities include:
1. **Vector Search**: Finding relevant information using semantic similarity search across documents
2. **Knowledge Graph Search**: Exploring relationships, entities, and temporal facts in the knowledge graph
3. **Hybrid Search**: Combining both vector and graph searches for comprehensive results
4. **Document Retrieval**: Accessing complete documents when detailed context is needed

When answering questions:
- Always search for relevant information before responding
- Combine insights from both vector search and knowledge graph when applicable
- Cite your sources by mentioning document titles and specific facts
- Consider temporal aspects - some information may be time-sensitive
- Look for relationships and connections between companies and technologies
- Be specific about which companies are involved in which AI initiatives

Your responses should be:
- Accurate and based on the available data
- Well-structured and easy to understand
- Comprehensive while remaining concise
- Transparent about the sources of information

Use the knowledge graph tool only when the user asks about two companies in the same question. Otherwise, use just the vector store tool.

Remember to:
- Use vector search for finding similar content and detailed explanations
- Use knowledge graph for understanding relationships between companies or initiatives
- Combine both approaches when asked only"""


# Novel-specific system prompt
NOVEL_SYSTEM_PROMPT = """You are an AI assistant specialized in novel writing and creative storytelling. You have access to a comprehensive knowledge base of literary elements, character development techniques, and plot structures.

Your primary capabilities include:
1. **Character Development**: Creating compelling characters with depth and growth arcs
2. **Plot Construction**: Building engaging narratives with proper pacing and structure
3. **World Building**: Developing immersive settings and environments
4. **Emotional Depth**: Infusing stories with appropriate emotional resonance
5. **Genre Adaptation**: Tailoring content to specific genres and styles
6. **Consistency Tracking**: Maintaining character, plot, and world consistency

When assisting with novel writing:
- Focus on character consistency and development
- Maintain plot coherence across chapters
- Ensure setting descriptions enhance the narrative
- Create emotional resonance appropriate to scenes
- Adapt your style to the specified genre and tone

Your responses should be:
- Creative and engaging while maintaining consistency
- Emotionally appropriate for the context
- Genre-appropriate in style and content
- Focused on advancing the narrative purpose
- Mindful of character voices and perspectives

Special considerations:
- Balance show vs. tell appropriately
- Maintain consistent point of view
- Use sensory details to enhance immersion
- Create dialogue that reflects character personalities
- Preserve narrative flow and pacing"""


# Character-focused prompts
CHARACTER_ANALYSIS_PROMPT = """Analyze the character development and consistency for {character_name}.

Focus on:
1. **Personality Consistency**: Does the character behave according to established traits?
2. **Development Arc**: How has the character grown or changed?
3. **Dialogue Voice**: Is the character's speech pattern consistent?
4. **Relationships**: How do their interactions reflect their personality?
5. **Motivations**: Are their actions driven by clear motivations?

Provide specific examples and suggestions for improvement."""


CHARACTER_DEVELOPMENT_PROMPT = """Help develop the character {character_name} for the scene.

Consider:
1. **Current Emotional State**: What is the character feeling?
2. **Motivations**: What does the character want in this scene?
3. **Obstacles**: What's preventing them from getting what they want?
4. **Growth Opportunity**: How can this scene advance their arc?
5. **Relationships**: How do other characters affect them?

Generate character-appropriate dialogue and actions."""


# Emotional analysis prompts
EMOTIONAL_ANALYSIS_PROMPT = """Analyze the emotional content and consistency in the text.

Focus on:
1. **Dominant Emotions**: What emotions are most present?
2. **Emotional Arc**: How do emotions change throughout?
3. **Character Emotions**: How does each character feel?
4. **Emotional Triggers**: What causes emotional changes?
5. **Consistency**: Do emotions match the situation?

Provide suggestions for enhancing emotional impact."""


EMOTIONAL_SCENE_PROMPT = """Generate content with a {emotional_tone} emotional tone at {intensity} intensity.

Guidelines:
1. **Emotional Authenticity**: Make emotions feel genuine and earned
2. **Character Consistency**: Ensure emotions fit the characters
3. **Situational Appropriateness**: Match emotions to the context
4. **Sensory Details**: Use physical sensations to convey emotion
5. **Dialogue**: Let emotion come through in speech patterns

Create a scene that effectively conveys the target emotion."""


# Plot-focused prompts
PLOT_ANALYSIS_PROMPT = """Analyze the plot structure and consistency.

Examine:
1. **Plot Threads**: Are all plot lines properly developed?
2. **Pacing**: Is the story moving at appropriate pace?
3. **Causality**: Do events follow logically?
4. **Tension**: Is there appropriate conflict?
5. **Stakes**: Are consequences clear and meaningful?

Identify plot holes and suggest improvements."""


PLOT_DEVELOPMENT_PROMPT = """Help develop the plot for the current scene/chapter.

Consider:
1. **Current Situation**: What's happening right now in the story?
2. **Conflict**: What obstacles or challenges are present?
3. **Goals**: What are the characters trying to achieve?
4. **Consequences**: What happens if they succeed or fail?
5. **Next Steps**: How does this scene advance the overall plot?

Suggest plot developments that maintain tension and advance the story."""


# Style and consistency prompts
STYLE_CONSISTENCY_PROMPT = """Check the writing style consistency in the content.

Examine:
1. **Point of View**: Is the POV consistent throughout?
2. **Tense**: Is the tense usage consistent?
3. **Voice**: Does the narrative voice remain consistent?
4. **Tone**: Is the overall tone appropriate and consistent?
5. **Genre Elements**: Are genre conventions properly followed?

Identify inconsistencies and suggest corrections."""


DIALOGUE_CONSISTENCY_PROMPT = """Analyze dialogue consistency for the characters.

Check:
1. **Character Voice**: Does each character have a distinct voice?
2. **Speech Patterns**: Are individual speech patterns maintained?
3. **Vocabulary**: Is word choice appropriate for each character?
4. **Formality Level**: Is the level of formality consistent per character?
5. **Emotional Expression**: Do characters express emotions authentically?

Suggest improvements for character voice distinctiveness."""


# Genre-specific prompts
FANTASY_WRITING_PROMPT = """Generate fantasy content with appropriate world-building elements.

Include:
1. **Magic System**: Consistent rules for how magic works
2. **World Details**: Unique aspects of the fantasy world
3. **Mythology**: Background lore that enriches the story
4. **Creatures**: Fantastical beings that serve the narrative
5. **Culture**: Social structures and customs of the world

Ensure all fantasy elements serve the story and characters."""


MYSTERY_WRITING_PROMPT = """Generate mystery content that maintains suspense and intrigue.

Include:
1. **Clues**: Subtle hints that advance the mystery
2. **Red Herrings**: Misdirection that doesn't feel unfair
3. **Investigation**: Logical steps in solving the mystery
4. **Suspects**: Multiple viable suspects with motives
5. **Revelation**: Satisfying resolution that makes sense

Build tension while playing fair with the reader."""


# Utility functions
def get_prompt_for_task(task_type: str, **kwargs) -> str:
    """
    Get the appropriate prompt for a specific task.
    
    Args:
        task_type: Type of task (character_analysis, plot_development, etc.)
        **kwargs: Additional parameters for prompt formatting
    
    Returns:
        Formatted prompt string
    """
    prompt_map = {
        'character_analysis': CHARACTER_ANALYSIS_PROMPT,
        'character_development': CHARACTER_DEVELOPMENT_PROMPT,
        'emotional_analysis': EMOTIONAL_ANALYSIS_PROMPT,
        'emotional_scene': EMOTIONAL_SCENE_PROMPT,
        'plot_analysis': PLOT_ANALYSIS_PROMPT,
        'plot_development': PLOT_DEVELOPMENT_PROMPT,
        'style_consistency': STYLE_CONSISTENCY_PROMPT,
        'dialogue_consistency': DIALOGUE_CONSISTENCY_PROMPT,
        'fantasy_writing': FANTASY_WRITING_PROMPT,
        'mystery_writing': MYSTERY_WRITING_PROMPT
    }
    
    prompt_template = prompt_map.get(task_type, NOVEL_SYSTEM_PROMPT)
    
    try:
        return prompt_template.format(**kwargs)
    except KeyError:
        return prompt_template


def generate_context_aware_prompt(
    base_task: str,
    character_context: dict = None,
    plot_context: dict = None,
    emotional_context: dict = None,
    style_context: dict = None
) -> str:
    """
    Generate a context-aware prompt that incorporates multiple aspects.
    
    Args:
        base_task: Base task type
        character_context: Character information
        plot_context: Plot information
        emotional_context: Emotional context
        style_context: Style requirements
    
    Returns:
        Enhanced prompt with context
    """
    base_prompt = get_prompt_for_task(base_task)
    
    context_additions = []
    
    if character_context:
        context_additions.append(f"Character Context: {character_context}")
    
    if plot_context:
        context_additions.append(f"Plot Context: {plot_context}")
    
    if emotional_context:
        context_additions.append(f"Emotional Context: {emotional_context}")
    
    if style_context:
        context_additions.append(f"Style Requirements: {style_context}")
    
    if context_additions:
        enhanced_prompt = base_prompt + "\n\nAdditional Context:\n" + "\n".join(context_additions)
        return enhanced_prompt
    
    return base_prompt