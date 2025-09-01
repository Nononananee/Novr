"""
CrewAI Agent Configurations for Novel Generation System
Python-based configuration for agents with optional YAML override support
"""

import os
from typing import Dict, Any, List
import yaml
from pathlib import Path

# Default agent configurations
DEFAULT_AGENT_CONFIGS = {
    "writer": {
        "model": "gpt-4o-mini",
        "role": "Professional Novelist",
        "goal": "Create compelling, well-structured novel chapters that engage readers and advance the story",
        "backstory": """You are a professional novelist with years of experience in creative writing. 
        You excel at creating vivid scenes, authentic dialogue, and compelling character development. 
        You understand story structure, pacing, and the importance of maintaining consistency with 
        established world-building and character personalities.""",
        "temperature": 0.7,
        "max_tokens": 2000,
        "tools": ["context_retrieval", "character_lookup", "plot_lookup"],
        "allow_delegation": False,
        "verbose": True
    },
    
    "structural_qa": {
        "model": "gpt-4o-mini", 
        "role": "Story Structure Analyst",
        "goal": "Ensure plot consistency, proper pacing, and strong narrative structure",
        "backstory": """You are an expert story editor who specializes in narrative structure and plot development. 
        You have a keen eye for pacing issues, plot holes, and structural problems that can weaken a story. 
        You understand the importance of proper story arcs, character motivation, and scene transitions.""",
        "temperature": 0.1,
        "max_tokens": 1500,
        "tools": ["structural_analysis"],
        "threshold": 70,
        "allow_delegation": False,
        "verbose": True
    },
    
    "character_qa": {
        "model": "gpt-4o-mini",
        "role": "Character Development Specialist", 
        "goal": "Maintain character consistency and authentic dialogue throughout the story",
        "backstory": """You are a character development expert who ensures that characters remain true to 
        their established personalities, relationships, and growth arcs. You have extensive experience 
        in dialogue writing and character voice consistency.""",
        "temperature": 0.1,
        "max_tokens": 1500,
        "tools": ["character_analysis", "relationship_lookup"],
        "threshold": 75,
        "allow_delegation": False,
        "verbose": True
    },
    
    "style_qa": {
        "model": "gpt-4o-mini",
        "role": "Prose Style Editor",
        "goal": "Maintain consistent writing style, tone, and high prose quality",
        "backstory": """You are a professional prose editor with expertise in literary style and voice. 
        You ensure that the writing maintains consistent tone, appropriate style for the genre, 
        and high-quality prose throughout the work.""",
        "temperature": 0.1,
        "max_tokens": 1500,
        "tools": ["style_analysis"],
        "threshold": 70,
        "allow_delegation": False,
        "verbose": True
    },
    
    "technical_qa": {
        "model": "gpt-4o-mini",
        "role": "Technical Editor",
        "goal": "Ensure perfect grammar, spelling, punctuation, and formatting",
        "backstory": """You are a meticulous technical editor who catches every grammar mistake, 
        spelling error, and formatting inconsistency. You have an eye for detail and ensure 
        that the technical quality of the writing is flawless.""",
        "temperature": 0.1,
        "max_tokens": 1500,
        "tools": ["technical_analysis"],
        "threshold": 80,
        "allow_delegation": False,
        "verbose": True
    }
}

# Default task configurations
DEFAULT_TASK_CONFIGS = {
    "generation": {
        "priority": 1,
        "timeout": 300,  # 5 minutes
        "retry_count": 2,
        "description_template": """
        Generate a novel chapter based on the following requirements:
        
        Project: {project_id}
        Prompt: {prompt}
        Settings: {settings}
        
        Requirements:
        1. Retrieve relevant context from the project's worldbook and character information
        2. Generate engaging, well-structured content that fits the story
        3. Maintain consistency with established characters and world-building
        4. Target approximately {length_words} words
        5. Use appropriate tone: {tone}
        6. Follow writing style: {style}
        
        Output: Complete chapter content in markdown format
        """,
        "expected_output": "A complete chapter in markdown format with proper structure and engaging content"
    },
    
    "structural_qa": {
        "priority": 2,
        "timeout": 120,  # 2 minutes
        "retry_count": 1,
        "parallel": True,
        "description_template": """
        Analyze the generated chapter for structural issues:
        
        1. Plot consistency and logical flow
        2. Pacing and rhythm
        3. Scene structure and organization
        4. Character motivation consistency
        5. Narrative continuity
        6. Conflict development
        
        Context: {context}
        
        Provide detailed feedback with specific suggestions for improvement.
        Output: JSON format with score, issues, and patches
        """,
        "expected_output": "JSON analysis with structural assessment and improvement suggestions"
    },
    
    "character_qa": {
        "priority": 2,
        "timeout": 120,
        "retry_count": 1,
        "parallel": True,
        "description_template": """
        Analyze the chapter for character consistency:
        
        1. Character personality consistency
        2. Dialogue authenticity and voice
        3. Character relationship dynamics
        4. Emotional responses appropriateness
        5. Character development progression
        6. Speech patterns consistency
        
        Character Context: {character_context}
        
        Provide specific feedback for each character involved.
        Output: JSON format with score, issues, and patches
        """,
        "expected_output": "JSON analysis with character consistency assessment"
    },
    
    "style_qa": {
        "priority": 2,
        "timeout": 120,
        "retry_count": 1,
        "parallel": True,
        "description_template": """
        Analyze the chapter for style and prose quality:
        
        1. Writing style consistency
        2. Tone appropriateness
        3. Prose quality and flow
        4. Sentence variety and rhythm
        5. Word choice and vocabulary
        6. Repetitive phrases identification
        7. Show vs. tell balance
        
        Style Guidelines: {style_context}
        
        Provide suggestions for style improvements.
        Output: JSON format with score, issues, and patches
        """,
        "expected_output": "JSON analysis with style quality assessment"
    },
    
    "technical_qa": {
        "priority": 2,
        "timeout": 120,
        "retry_count": 1,
        "parallel": True,
        "description_template": """
        Perform technical editing on the chapter:
        
        1. Grammar and syntax errors
        2. Spelling mistakes
        3. Punctuation errors
        4. Formatting consistency
        5. Capitalization issues
        6. Basic style problems
        
        Provide specific corrections and patches.
        Output: JSON format with score, issues, and patches
        """,
        "expected_output": "JSON analysis with technical quality assessment and corrections"
    },
    
    "revision": {
        "priority": 3,
        "timeout": 300,
        "retry_count": 1,
        "description_template": """
        Revise the chapter based on QA feedback:
        
        Original Content: {original_content}
        
        QA Feedback: {qa_feedback}
        
        Requirements:
        1. Address all high-priority issues identified by QA agents
        2. Maintain the original story intent and structure
        3. Improve overall quality while preserving the author's voice
        4. Ensure all revisions are consistent with the established world and characters
        
        Output: Revised chapter content in markdown format
        """,
        "expected_output": "Improved chapter content addressing QA feedback"
    }
}

class CrewConfigManager:
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        self.agent_configs = DEFAULT_AGENT_CONFIGS.copy()
        self.task_configs = DEFAULT_TASK_CONFIGS.copy()
        
        # Load YAML overrides if they exist
        self._load_yaml_configs()
    
    def _load_yaml_configs(self):
        """Load YAML configuration overrides"""
        try:
            # Load agent configurations
            agents_yaml = self.config_dir / "agents.yaml"
            if agents_yaml.exists():
                with open(agents_yaml, 'r') as f:
                    yaml_agents = yaml.safe_load(f)
                    if yaml_agents:
                        self._merge_configs(self.agent_configs, yaml_agents)
                        print(f"Loaded agent configurations from {agents_yaml}")
            
            # Load task configurations
            tasks_yaml = self.config_dir / "tasks.yaml"
            if tasks_yaml.exists():
                with open(tasks_yaml, 'r') as f:
                    yaml_tasks = yaml.safe_load(f)
                    if yaml_tasks:
                        self._merge_configs(self.task_configs, yaml_tasks)
                        print(f"Loaded task configurations from {tasks_yaml}")
                        
        except Exception as e:
            print(f"Warning: Failed to load YAML configs: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """Merge override configuration into base configuration"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        config = self.agent_configs.get(agent_name, {}).copy()
        
        # Add environment variable overrides
        env_prefix = f"AGENT_{agent_name.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        config[config_key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[config_key] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        config[config_key] = float(value)
                    else:
                        config[config_key] = value
                except:
                    config[config_key] = value
        
        return config
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific task"""
        return self.task_configs.get(task_name, {}).copy()
    
    def get_all_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent configurations"""
        return {name: self.get_agent_config(name) for name in self.agent_configs.keys()}
    
    def get_all_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all task configurations"""
        return {name: self.get_task_config(name) for name in self.task_configs.keys()}
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]):
        """Update agent configuration at runtime"""
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].update(updates)
        else:
            self.agent_configs[agent_name] = updates
    
    def update_task_config(self, task_name: str, updates: Dict[str, Any]):
        """Update task configuration at runtime"""
        if task_name in self.task_configs:
            self.task_configs[task_name].update(updates)
        else:
            self.task_configs[task_name] = updates
    
    def get_qa_thresholds(self) -> Dict[str, int]:
        """Get QA score thresholds for all agents"""
        thresholds = {}
        for agent_name, config in self.agent_configs.items():
            if 'threshold' in config:
                thresholds[agent_name] = config['threshold']
        return thresholds
    
    def export_configs(self, output_dir: str = "exported_configs"):
        """Export current configurations to YAML files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export agent configs
        with open(output_path / "agents.yaml", 'w') as f:
            yaml.dump(self.agent_configs, f, default_flow_style=False, indent=2)
        
        # Export task configs
        with open(output_path / "tasks.yaml", 'w') as f:
            yaml.dump(self.task_configs, f, default_flow_style=False, indent=2)
        
        print(f"Configurations exported to {output_path}")

# Global configuration manager instance
config_manager = CrewConfigManager()

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Convenience function to get agent configuration"""
    return config_manager.get_agent_config(agent_name)

def get_task_config(task_name: str) -> Dict[str, Any]:
    """Convenience function to get task configuration"""
    return config_manager.get_task_config(task_name)

def get_qa_thresholds() -> Dict[str, int]:
    """Convenience function to get QA thresholds"""
    return config_manager.get_qa_thresholds()