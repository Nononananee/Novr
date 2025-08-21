# src/novel/coherence/consistency_enforcer.py
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import json

class ConsistencyLevel(Enum):
    CRITICAL = "critical"      # Must never conflict (character names, deaths, etc.)
    HIGH = "high"             # Important traits (personality, relationships)
    MEDIUM = "medium"         # Secondary details (appearance, preferences) 
    LOW = "low"              # Minor details (clothing, mood)

class InconsistencyType(Enum):
    CHARACTER_CONTRADICTION = "character_contradiction"
    TIMELINE_CONFLICT = "timeline_conflict"
    WORLD_RULE_VIOLATION = "world_rule_violation"
    RELATIONSHIP_INCONSISTENCY = "relationship_inconsistency"
    PLOT_HOLE = "plot_hole"
    TONE_MISMATCH = "tone_mismatch"

@dataclass
class ConsistencyRule:
    id: str
    name: str
    description: str
    level: ConsistencyLevel
    rule_type: str  # "character", "world", "plot", "timeline"
    validation_function: str  # Function name to call
    parameters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class InconsistencyIssue:
    id: str
    type: InconsistencyType
    level: ConsistencyLevel
    description: str
    conflicting_chunks: List[str]  # Chunk IDs that conflict
    suggested_resolution: Optional[str] = None
    auto_fixable: bool = False
    fixed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConsistencyState:
    """Current state of all consistency-relevant elements"""
    characters: Dict[str, Dict[str, Any]]  # character_id -> properties
    world_rules: Dict[str, Any]
    timeline_events: List[Dict[str, Any]]
    relationships: Dict[Tuple[str, str], Dict[str, Any]]
    plot_threads: Dict[str, Dict[str, Any]]
    established_facts: Set[str]
    last_updated: datetime = field(default_factory=datetime.now)

class LongTermConsistencyManager:
    def __init__(self, memory_manager=None, character_repo=None):
        self.memory_manager = memory_manager
        self.character_repo = character_repo
        
        # Consistency tracking
        self.consistency_rules: Dict[str, ConsistencyRule] = {}
        self.active_issues: List[InconsistencyIssue] = []
        self.consistency_state = ConsistencyState(
            characters={},
            world_rules={},
            timeline_events=[],
            relationships={},
            plot_threads={},
            established_facts=set()
        )
        
        # Rule definitions
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default consistency rules"""
        
        # Character consistency rules
        self.add_consistency_rule(ConsistencyRule(
            id="character_death_permanence",
            name="Character Death Permanence",
            description="Dead characters cannot appear alive in later scenes",
            level=ConsistencyLevel.CRITICAL,
            rule_type="character",
            validation_function="validate_character_death_consistency"
        ))
        
        self.add_consistency_rule(ConsistencyRule(
            id="character_physical_traits",
            name="Physical Trait Consistency",
            description="Character physical descriptions must remain consistent",
            level=ConsistencyLevel.HIGH,
            rule_type="character",
            validation_function="validate_character_physical_consistency"
        ))
        
        # Timeline consistency rules
        self.add_consistency_rule(ConsistencyRule(
            id="temporal_causality",
            name="Temporal Causality",
            description="Events must follow logical temporal sequence",
            level=ConsistencyLevel.CRITICAL,
            rule_type="timeline",
            validation_function="validate_temporal_causality"
        ))
        
        # World building rules
        self.add_consistency_rule(ConsistencyRule(
            id="magic_system_rules",
            name="Magic System Consistency",
            description="Magic system rules must be followed consistently",
            level=ConsistencyLevel.HIGH,
            rule_type="world",
            validation_function="validate_magic_system_consistency"
        ))
        
        # Relationship rules
        self.add_consistency_rule(ConsistencyRule(
            id="relationship_evolution",
            name="Relationship Evolution Logic",
            description="Relationship changes must be logically motivated",
            level=ConsistencyLevel.MEDIUM,
            rule_type="relationship",
            validation_function="validate_relationship_evolution"
        ))

    async def check_consistency_before_generation(self, 
                                                context: str, 
                                                generation_intent: Dict) -> Tuple[bool, List[InconsistencyIssue]]:
        """Check consistency before generating new content"""
        
        issues = []
        
        # Extract entities and facts from context
        entities = await self._extract_entities_from_context(context)
        
        # Check against all active rules
        for rule in self.consistency_rules.values():
            if not rule.active:
                continue
            
            rule_issues = await self._apply_consistency_rule(rule, entities, context, generation_intent)
            issues.extend(rule_issues)
        
        # Determine if generation should proceed
        critical_issues = [issue for issue in issues if issue.level == ConsistencyLevel.CRITICAL]
        can_proceed = len(critical_issues) == 0
        
        return can_proceed, issues

    async def validate_generated_content(self, 
                                       new_content: str, 
                                       context: str) -> Tuple[bool, List[InconsistencyIssue]]:
        """Validate newly generated content for consistency"""
        
        issues = []
        
        # Extract new entities and facts
        new_entities = await self._extract_entities_from_content(new_content)
        
        # Check consistency with existing state
        for rule in self.consistency_rules.values():
            if not rule.active:
                continue
                
            rule_issues = await self._validate_new_content_against_rule(
                rule, new_content, new_entities, context
            )
            issues.extend(rule_issues)
        
        # Update consistency state if no critical issues
        critical_issues = [issue for issue in issues if issue.level == ConsistencyLevel.CRITICAL]
        
        if len(critical_issues) == 0:
            await self._update_consistency_state(new_content, new_entities)
        
        return len(critical_issues) == 0, issues

    async def fix_consistency_issues(self, 
                                   content: str, 
                                   issues: List[InconsistencyIssue]) -> Tuple[str, List[InconsistencyIssue]]:
        """Attempt to fix consistency issues in content"""
        
        fixed_content = content
        remaining_issues = []
        
        for issue in issues:
            if issue.auto_fixable:
                try:
                    fixed_content = await self._auto_fix_issue(fixed_content, issue)
                    issue.fixed = True
                except Exception as e:
                    print(f"Failed to auto-fix issue {issue.id}: {e}")
                    remaining_issues.append(issue)
            else:
                # Generate suggested fix
                issue.suggested_resolution = await self._generate_fix_suggestion(issue)
                remaining_issues.append(issue)
        
        return fixed_content, remaining_issues

    async def _extract_entities_from_context(self, context: str) -> Dict[str, List[Any]]:
        """Extract characters, places, objects, etc. from context"""
        entities = {
            'characters': [],
            'locations': [],
            'objects': [],
            'events': [],
            'relationships': [],
            'timeline_markers': []
        }
        
        # Use character repository if available
        if self.character_repo:
            known_characters = await self.character_repo.get_all_characters()
            for char in known_characters:
                if char.name.lower() in context.lower():
                    entities['characters'].append({
                        'name': char.name,
                        'id': char.id,
                        'mentions': context.lower().count(char.name.lower())
                    })
        
        # Extract locations (simplified - would use NER in real implementation)
        location_patterns = [r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 
                           r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)']
        
        # Extract timeline markers
        temporal_patterns = [r'\b(yesterday|today|tomorrow|last week|next month)\b',
                           r'\b\d+\s+(days?|weeks?|months?|years?)\s+(ago|later)\b']
        
        # Would implement full NER extraction here
        
        return entities

    async def _apply_consistency_rule(self, 
                                    rule: ConsistencyRule, 
                                    entities: Dict, 
                                    context: str, 
                                    generation_intent: Dict) -> List[InconsistencyIssue]:
        """Apply a specific consistency rule"""
        
        issues = []
        
        try:
            # Call the appropriate validation function
            if hasattr(self, rule.validation_function):
                validation_func = getattr(self, rule.validation_function)
                rule_issues = await validation_func(entities, context, generation_intent, rule)
                issues.extend(rule_issues)
        except Exception as e:
            print(f"Error applying rule {rule.id}: {e}")
        
        return issues

    async def validate_character_death_consistency(self, 
                                                 entities: Dict, 
                                                 context: str, 
                                                 generation_intent: Dict, 
                                                 rule: ConsistencyRule) -> List[InconsistencyIssue]:
        """Validate that dead characters don't appear alive"""
        issues = []
        
        # Get list of dead characters from consistency state
        dead_characters = {char_id: char_data for char_id, char_data in 
                          self.consistency_state.characters.items() 
                          if char_data.get('status') == 'dead'}
        
        # Check if any dead characters are mentioned as alive in context
        for char_id, char_data in dead_characters.items():
            char_name = char_data.get('name', '')
            
            # Look for alive indicators
            alive_patterns = [
                f"{char_name} said",
                f"{char_name} walked",
                f"{char_name} thought",
                f"{char_name} smiled"
            ]
            
            for pattern in alive_patterns:
                if pattern.lower() in context.lower():
                    issue = InconsistencyIssue(
                        id=f"dead_char_alive_{char_id}_{datetime.now().timestamp()}",
                        type=InconsistencyType.CHARACTER_CONTRADICTION,
                        level=ConsistencyLevel.CRITICAL,
                        description=f"Dead character {char_name} appears to be alive in content",
                        conflicting_chunks=[],  # Would populate with specific chunk IDs
                        suggested_resolution=f"Remove references to {char_name} being alive, or clarify this is a flashback/memory",
                        auto_fixable=False
                    )
                    issues.append(issue)
        
        return issues

    async def validate_character_physical_consistency(self, 
                                                    entities: Dict, 
                                                    context: str, 
                                                    generation_intent: Dict, 
                                                    rule: ConsistencyRule) -> List[InconsistencyIssue]:
        """Validate character physical trait consistency"""
        issues = []
        
        for character in entities.get('characters', []):
            char_id = character.get('id')
            char_name = character.get('name')
            
            if char_id and char_id in self.consistency_state.characters:
                stored_traits = self.consistency_state.characters[char_id].get('physical_traits', {})
                
                # Check for conflicting physical descriptions in context
                # This would be more sophisticated in real implementation
                trait_patterns = {
                    'hair_color': [r'(\w+)\s+hair', r'hair\s+(?:was|is)\s+(\w+)'],
                    'eye_color': [r'(\w+)\s+eyes', r'eyes\s+(?:were|are)\s+(\w+)'],
                    'height': [r'(tall|short|average\s+height)']
                }
                
                for trait_type, patterns in trait_patterns.items():
                    if trait_type in stored_traits:
                        stored_value = stored_traits[trait_type]
                        
                        # Look for conflicting descriptions
                        for pattern in patterns:
                            import re
                            matches = re.findall(pattern, context, re.IGNORECASE)
                            for match in matches:
                                if isinstance(match, tuple):
                                    match = match[0] if match else ''
                                
                                if match and match.lower() != stored_value.lower():
                                    issue = InconsistencyIssue(
                                        id=f"physical_inconsistency_{char_id}_{trait_type}_{datetime.now().timestamp()}",
                                        type=InconsistencyType.CHARACTER_CONTRADICTION,
                                        level=ConsistencyLevel.HIGH,
                                        description=f"Character {char_name} {trait_type} described as '{match}' but previously established as '{stored_value}'",
                                        conflicting_chunks=[],
                                        suggested_resolution=f"Change description to match established {trait_type}: '{stored_value}'",
                                        auto_fixable=True
                                    )
                                    issues.append(issue)
        
        return issues

    async def validate_temporal_causality(self, 
                                        entities: Dict, 
                                        context: str, 
                                        generation_intent: Dict, 
                                        rule: ConsistencyRule) -> List[InconsistencyIssue]:
        """Validate temporal causality in events"""
        issues = []
        
        # Extract timeline events from context
        timeline_markers = entities.get('timeline_markers', [])
        
        # Check against established timeline
        for event in self.consistency_state.timeline_events:
            # Look for temporal conflicts
            event_time = event.get('timestamp', 0)
            event_description = event.get('description', '')
            
            # Check for contradictory temporal references
            for marker in timeline_markers:
                if isinstance(marker, str):
                    # Simple temporal conflict detection
                    if 'before' in marker.lower() and event_time > 0:
                        # Check if event is referenced as happening before something that already occurred
                        if any(keyword in context.lower() for keyword in ['happened before', 'occurred before']):
                            issue = InconsistencyIssue(
                                id=f"temporal_conflict_{event.get('id', 'unknown')}_{datetime.now().timestamp()}",
                                type=InconsistencyType.TIMELINE_CONFLICT,
                                level=ConsistencyLevel.HIGH,
                                description=f"Temporal conflict detected: Event '{event_description}' timeline inconsistency",
                                conflicting_chunks=[],
                                suggested_resolution="Review event sequence and ensure temporal consistency",
                                auto_fixable=False
                            )
                            issues.append(issue)
        
        return issues

    async def _update_consistency_state(self, new_content: str, entities: Dict):
        """Update consistency state with new information"""
        
        # Update character states
        for character in entities.get('characters', []):
            char_id = character.get('id')
            char_name = character.get('name', '')
            if char_id:
                if char_id not in self.consistency_state.characters:
                    self.consistency_state.characters[char_id] = {
                        'name': char_name,
                        'physical_traits': {},
                        'personality_traits': {},
                        'status': 'alive',
                        'relationships': {},
                        'last_seen_chapter': 0
                    }
                
                # Extract and update character traits from content
                char_data = self.consistency_state.characters[char_id]
                
                # Update physical traits if mentioned
                if 'hair' in new_content.lower():
                    import re
                    hair_matches = re.findall(r'(\w+)\s+hair|hair\s+(?:was|is)\s+(\w+)', new_content.lower())
                    for match in hair_matches:
                        hair_color = match[0] or match[1]
                        if hair_color and char_name.lower() in new_content.lower():
                            char_data['physical_traits']['hair_color'] = hair_color
                
                # Check for character death
                death_indicators = ['died', 'killed', 'dead', 'perished', 'murdered']
                if any(indicator in new_content.lower() for indicator in death_indicators):
                    if char_name.lower() in new_content.lower():
                        char_data['status'] = 'dead'
        
        # Update timeline - extract and add new events
        timeline_events = entities.get('timeline_markers', [])
        for event in timeline_events:
            if isinstance(event, str):
                new_event = {
                    'id': f"event_{len(self.consistency_state.timeline_events)}",
                    'description': event,
                    'timestamp': len(self.consistency_state.timeline_events),
                    'content_reference': new_content[:100] + "..."
                }
                self.consistency_state.timeline_events.append(new_event)
        
        # Update world rules - look for new magical/world rule establishments
        magic_keywords = ['magic', 'spell', 'enchant', 'curse', 'potion', 'wizard', 'witch']
        if any(keyword in new_content.lower() for keyword in magic_keywords):
            # Extract potential magic rules
            import re
            rule_patterns = [
                r'magic\s+(?:can|cannot|must|will)\s+([^.]+)',
                r'spells?\s+(?:require|need|use)\s+([^.]+)',
                r'(?:wizards?|witches?)\s+(?:can|cannot)\s+([^.]+)'
            ]
            
            for pattern in rule_patterns:
                matches = re.findall(pattern, new_content.lower())
                for match in matches:
                    rule_key = f"magic_rule_{len(self.consistency_state.world_rules)}"
                    self.consistency_state.world_rules[rule_key] = {
                        'type': 'magic_system',
                        'description': match.strip(),
                        'established_in': new_content[:50] + "..."
                    }
        
        # Add established facts
        fact_indicators = ['always', 'never', 'must', 'cannot', 'will always', 'will never']
        for indicator in fact_indicators:
            if indicator in new_content.lower():
                # Extract the fact
                import re
                fact_pattern = f'{indicator}\\s+([^.]+)'
                matches = re.findall(fact_pattern, new_content.lower())
                for match in matches:
                    self.consistency_state.established_facts.add(f"{indicator} {match.strip()}")
        
        self.consistency_state.last_updated = datetime.now()

    async def _auto_fix_issue(self, content: str, issue: InconsistencyIssue) -> str:
        """Automatically fix simple consistency issues"""
        
        fixed_content = content
        
        if issue.type == InconsistencyType.CHARACTER_CONTRADICTION:
            # Replace incorrect physical descriptions
            import re
            
            # Extract character name from issue description
            char_match = re.search(r'Character (\w+)', issue.description)
            if char_match:
                char_name = char_match.group(1)
                
                # Get correct traits from consistency state
                char_data = None
                for char_id, data in self.consistency_state.characters.items():
                    if data.get('name', '').lower() == char_name.lower():
                        char_data = data
                        break
                
                if char_data:
                    # Fix hair color inconsistencies
                    if 'hair_color' in char_data.get('physical_traits', {}):
                        correct_hair = char_data['physical_traits']['hair_color']
                        
                        # Find and replace incorrect hair descriptions
                        hair_patterns = [
                            r'(\w+)\s+hair',
                            r'hair\s+(?:was|is)\s+(\w+)'
                        ]
                        
                        for pattern in hair_patterns:
                            matches = re.finditer(pattern, fixed_content, re.IGNORECASE)
                            for match in matches:
                                if char_name.lower() in fixed_content[max(0, match.start()-50):match.end()+50].lower():
                                    # Replace with correct hair color
                                    if match.group(1):
                                        fixed_content = fixed_content.replace(
                                            match.group(0), 
                                            f"{correct_hair} hair"
                                        )
                                    elif len(match.groups()) > 1 and match.group(2):
                                        fixed_content = fixed_content.replace(
                                            match.group(0), 
                                            f"hair was {correct_hair}"
                                        )
        
        elif issue.type == InconsistencyType.TIMELINE_CONFLICT:
            # Fix simple temporal inconsistencies
            temporal_fixes = {
                'yesterday': 'the day before',
                'tomorrow': 'the next day',
                'last week': 'the previous week',
                'next month': 'the following month'
            }
            
            for incorrect, correct in temporal_fixes.items():
                if incorrect in fixed_content.lower():
                    fixed_content = re.sub(
                        re.escape(incorrect), 
                        correct, 
                        fixed_content, 
                        flags=re.IGNORECASE
                    )
        
        return fixed_content

    async def _generate_fix_suggestion(self, issue: InconsistencyIssue) -> str:
        """Generate human-readable fix suggestion for complex issues"""
        
        suggestions = {
            InconsistencyType.CHARACTER_CONTRADICTION: "Review character descriptions and align with established traits",
            InconsistencyType.TIMELINE_CONFLICT: "Check event sequence and resolve temporal conflicts",
            InconsistencyType.WORLD_RULE_VIOLATION: "Ensure world rules are consistently applied",
            InconsistencyType.PLOT_HOLE: "Add connecting events or explanation for plot progression",
        }
        
        return suggestions.get(issue.type, "Manual review required")

    def add_consistency_rule(self, rule: ConsistencyRule):
        """Add a new consistency rule"""
        self.consistency_rules[rule.id] = rule

    def get_consistency_report(self) -> Dict:
        """Generate comprehensive consistency report"""
        return {
            'total_rules': len(self.consistency_rules),
            'active_rules': len([r for r in self.consistency_rules.values() if r.active]),
            'active_issues': len(self.active_issues),
            'critical_issues': len([i for i in self.active_issues if i.level == ConsistencyLevel.CRITICAL]),
            'consistency_state_summary': {
                'tracked_characters': len(self.consistency_state.characters),
                'timeline_events': len(self.consistency_state.timeline_events),
                'world_rules': len(self.consistency_state.world_rules),
                'established_facts': len(self.consistency_state.established_facts)
            },
            'last_updated': self.consistency_state.last_updated.isoformat()
        }