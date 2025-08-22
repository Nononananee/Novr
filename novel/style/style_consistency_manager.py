"""
Style Consistency Management System for maintaining consistent writing style.
"""

import logging
import re
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import json

logger = logging.getLogger(__name__)


class WritingStyle(Enum):
    """Types of writing styles."""
    LITERARY = "literary"
    COMMERCIAL = "commercial"
    GENRE_FANTASY = "genre_fantasy"
    GENRE_MYSTERY = "genre_mystery"
    GENRE_ROMANCE = "genre_romance"
    GENRE_SCIFI = "genre_scifi"
    YOUNG_ADULT = "young_adult"
    MIDDLE_GRADE = "middle_grade"
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    CUSTOM = "custom"


class StyleMetric(Enum):
    """Style metrics to analyze."""
    SENTENCE_LENGTH = "sentence_length"
    PARAGRAPH_LENGTH = "paragraph_length"
    VOCABULARY_COMPLEXITY = "vocabulary_complexity"
    DIALOGUE_RATIO = "dialogue_ratio"
    DESCRIPTION_RATIO = "description_ratio"
    ACTION_RATIO = "action_ratio"
    PUNCTUATION_PATTERNS = "punctuation_patterns"
    TENSE_CONSISTENCY = "tense_consistency"
    POV_CONSISTENCY = "pov_consistency"
    TONE_CONSISTENCY = "tone_consistency"


@dataclass
class StyleFingerprint:
    """Fingerprint of a writing style."""
    avg_sentence_length: float
    avg_paragraph_length: float
    vocabulary_diversity: float  # Type-token ratio
    common_words: Dict[str, int]
    punctuation_frequency: Dict[str, float]
    dialogue_percentage: float
    description_percentage: float
    action_percentage: float
    tense_distribution: Dict[str, float]
    pov_pattern: str  # first, second, third
    tone_indicators: Dict[str, float]
    complexity_score: float
    readability_score: float
    
    
@dataclass
class StyleAnalysisResult:
    """Result of style analysis."""
    fingerprint: StyleFingerprint
    consistency_score: float  # 0.0 to 1.0
    style_deviations: List[str]
    recommendations: List[str]
    confidence: float
    analyzed_word_count: int
    

@dataclass
class StyleGuide:
    """Style guide for a specific writing style."""
    name: str
    style_type: WritingStyle
    target_fingerprint: StyleFingerprint
    tolerance_ranges: Dict[str, Tuple[float, float]]  # metric -> (min, max)
    style_rules: List[str]
    forbidden_patterns: List[str]
    encouraged_patterns: List[str]
    

class StyleConsistencyManager:
    """Manages style consistency analysis and enforcement."""
    
    def __init__(self, db_utils=None):
        self.db_utils = db_utils
        self.style_guides = self._initialize_style_guides()
        self.active_style_guide: Optional[StyleGuide] = None
        self.baseline_fingerprint: Optional[StyleFingerprint] = None
        self.style_history: List[Dict[str, Any]] = []
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text analysis."""
        
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.dialogue_pattern = re.compile(r'"[^"]*"')
        self.action_pattern = re.compile(r'\b(ran|walked|jumped|grabbed|threw|hit|moved|turned|looked|saw|heard)\b', re.IGNORECASE)
        self.description_pattern = re.compile(r'\b(beautiful|dark|bright|tall|small|cold|warm|soft|hard|smooth|rough)\b', re.IGNORECASE)
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        
        # Tense patterns
        self.past_tense_pattern = re.compile(r'\b\w+ed\b|\bwas\b|\bwere\b|\bhad\b', re.IGNORECASE)
        self.present_tense_pattern = re.compile(r'\bis\b|\bare\b|\bam\b|\bhas\b|\bhave\b', re.IGNORECASE)
        self.future_tense_pattern = re.compile(r'\bwill\b|\bshall\b|\bgoing to\b', re.IGNORECASE)
        
        # POV patterns
        self.first_person_pattern = re.compile(r'\b(I|me|my|mine|we|us|our|ours)\b', re.IGNORECASE)
        self.second_person_pattern = re.compile(r'\b(you|your|yours)\b', re.IGNORECASE)
        self.third_person_pattern = re.compile(r'\b(he|she|it|they|him|her|them|his|hers|its|their|theirs)\b', re.IGNORECASE)
    
    def _initialize_style_guides(self) -> Dict[WritingStyle, StyleGuide]:
        """Initialize predefined style guides."""
        
        guides = {}
        
        # Literary Fiction Style
        literary_fingerprint = StyleFingerprint(
            avg_sentence_length=18.5,
            avg_paragraph_length=85.0,
            vocabulary_diversity=0.65,
            common_words={"the": 100, "and": 80, "of": 70, "to": 65, "a": 60},
            punctuation_frequency={".": 0.12, ",": 0.08, ";": 0.02, ":": 0.01},
            dialogue_percentage=0.25,
            description_percentage=0.40,
            action_percentage=0.35,
            tense_distribution={"past": 0.70, "present": 0.25, "future": 0.05},
            pov_pattern="third",
            tone_indicators={"contemplative": 0.3, "descriptive": 0.4, "introspective": 0.3},
            complexity_score=0.75,
            readability_score=0.65
        )
        
        guides[WritingStyle.LITERARY] = StyleGuide(
            name="Literary Fiction",
            style_type=WritingStyle.LITERARY,
            target_fingerprint=literary_fingerprint,
            tolerance_ranges={
                "avg_sentence_length": (15.0, 22.0),
                "vocabulary_diversity": (0.60, 0.70),
                "dialogue_percentage": (0.20, 0.35),
                "complexity_score": (0.70, 0.80)
            },
            style_rules=[
                "Use varied sentence structures",
                "Employ rich, descriptive language",
                "Focus on character interiority",
                "Avoid clichés and overused phrases"
            ],
            forbidden_patterns=[
                r'\bvery\s+\w+',  # "very" + adjective
                r'\bsaid\s+\w+ly',  # adverbs with "said"
                r'\bit was\s+\w+\s+that'  # expletive constructions
            ],
            encouraged_patterns=[
                r'\b\w+ing\b.*\b\w+ed\b',  # mixing participles and past tense
                r'[,;]\s*\w+',  # complex punctuation usage
            ]
        )
        
        # Commercial Fiction Style
        commercial_fingerprint = StyleFingerprint(
            avg_sentence_length=14.2,
            avg_paragraph_length=65.0,
            vocabulary_diversity=0.55,
            common_words={"the": 120, "and": 95, "to": 80, "a": 75, "of": 70},
            punctuation_frequency={".": 0.15, ",": 0.10, "!": 0.02, "?": 0.01},
            dialogue_percentage=0.40,
            description_percentage=0.25,
            action_percentage=0.35,
            tense_distribution={"past": 0.75, "present": 0.20, "future": 0.05},
            pov_pattern="third",
            tone_indicators={"engaging": 0.4, "accessible": 0.4, "dynamic": 0.2},
            complexity_score=0.60,
            readability_score=0.80
        )
        
        guides[WritingStyle.COMMERCIAL] = StyleGuide(
            name="Commercial Fiction",
            style_type=WritingStyle.COMMERCIAL,
            target_fingerprint=commercial_fingerprint,
            tolerance_ranges={
                "avg_sentence_length": (12.0, 16.0),
                "vocabulary_diversity": (0.50, 0.60),
                "dialogue_percentage": (0.35, 0.45),
                "readability_score": (0.75, 0.85)
            },
            style_rules=[
                "Keep sentences clear and accessible",
                "Use active voice predominantly",
                "Balance dialogue and action",
                "Maintain engaging pace"
            ],
            forbidden_patterns=[
                r'\b\w{15,}\b',  # Very long words
                r'[;:]{2,}',  # Multiple semicolons/colons
            ],
            encouraged_patterns=[
                r'"[^"]*[!?]"',  # Exclamatory/questioning dialogue
                r'\b\w+ed\s+\w+ly\b',  # Past tense + adverb combinations
            ]
        )
        
        # Young Adult Style
        ya_fingerprint = StyleFingerprint(
            avg_sentence_length=12.8,
            avg_paragraph_length=55.0,
            vocabulary_diversity=0.50,
            common_words={"I": 150, "the": 110, "and": 90, "to": 85, "was": 80},
            punctuation_frequency={".": 0.14, ",": 0.09, "!": 0.03, "?": 0.02},
            dialogue_percentage=0.50,
            description_percentage=0.20,
            action_percentage=0.30,
            tense_distribution={"past": 0.60, "present": 0.35, "future": 0.05},
            pov_pattern="first",
            tone_indicators={"conversational": 0.5, "emotional": 0.3, "immediate": 0.2},
            complexity_score=0.45,
            readability_score=0.85
        )
        
        guides[WritingStyle.YOUNG_ADULT] = StyleGuide(
            name="Young Adult",
            style_type=WritingStyle.YOUNG_ADULT,
            target_fingerprint=ya_fingerprint,
            tolerance_ranges={
                "avg_sentence_length": (10.0, 15.0),
                "dialogue_percentage": (0.45, 0.55),
                "complexity_score": (0.40, 0.50),
                "readability_score": (0.80, 0.90)
            },
            style_rules=[
                "Use contemporary, accessible language",
                "Emphasize dialogue and internal voice",
                "Keep descriptions concise",
                "Maintain emotional immediacy"
            ],
            forbidden_patterns=[
                r'\b\w{12,}\b',  # Very long words
                r'\bwhom\b',  # Formal pronouns
            ],
            encouraged_patterns=[
                r'"[^"]*\?"',  # Questions in dialogue
                r'\bI\s+\w+',  # First person constructions
            ]
        )
        
        return guides
    
    def set_active_style_guide(self, style_type: WritingStyle) -> bool:
        """Set the active style guide."""
        
        try:
            if style_type in self.style_guides:
                self.active_style_guide = self.style_guides[style_type]
                logger.info(f"Set active style guide to: {self.active_style_guide.name}")
                return True
            else:
                logger.error(f"Style guide {style_type} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting active style guide: {e}")
            return False
    
    def analyze_text_style(self, text: str) -> StyleFingerprint:
        """Analyze the style of a given text."""
        
        try:
            # Basic text metrics
            sentences = self.sentence_pattern.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            paragraphs = self.paragraph_pattern.split(text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            words = self.word_pattern.findall(text.lower())
            
            # Calculate metrics
            avg_sentence_length = statistics.mean([len(s.split()) for s in sentences]) if sentences else 0
            avg_paragraph_length = statistics.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
            
            # Vocabulary diversity (Type-Token Ratio)
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / len(words) if words else 0
            
            # Common words
            word_counts = Counter(words)
            common_words = dict(word_counts.most_common(20))
            
            # Punctuation analysis
            punctuation_chars = self.punctuation_pattern.findall(text)
            punct_counts = Counter(punctuation_chars)
            total_chars = len(text)
            punctuation_frequency = {char: count/total_chars for char, count in punct_counts.items()}
            
            # Content type analysis
            dialogue_matches = len(self.dialogue_pattern.findall(text))
            action_matches = len(self.action_pattern.findall(text))
            description_matches = len(self.description_pattern.findall(text))
            
            total_content_markers = dialogue_matches + action_matches + description_matches
            if total_content_markers > 0:
                dialogue_percentage = dialogue_matches / total_content_markers
                action_percentage = action_matches / total_content_markers
                description_percentage = description_matches / total_content_markers
            else:
                dialogue_percentage = action_percentage = description_percentage = 0.33
            
            # Tense analysis
            past_matches = len(self.past_tense_pattern.findall(text))
            present_matches = len(self.present_tense_pattern.findall(text))
            future_matches = len(self.future_tense_pattern.findall(text))
            
            total_tense_markers = past_matches + present_matches + future_matches
            if total_tense_markers > 0:
                tense_distribution = {
                    "past": past_matches / total_tense_markers,
                    "present": present_matches / total_tense_markers,
                    "future": future_matches / total_tense_markers
                }
            else:
                tense_distribution = {"past": 0.7, "present": 0.25, "future": 0.05}
            
            # POV analysis
            first_person_matches = len(self.first_person_pattern.findall(text))
            second_person_matches = len(self.second_person_pattern.findall(text))
            third_person_matches = len(self.third_person_pattern.findall(text))
            
            pov_counts = {
                "first": first_person_matches,
                "second": second_person_matches,
                "third": third_person_matches
            }
            pov_pattern = max(pov_counts, key=pov_counts.get)
            
            # Tone indicators (simplified)
            tone_indicators = self._analyze_tone_indicators(text)
            
            # Complexity and readability scores (simplified)
            complexity_score = min(1.0, (avg_sentence_length / 20.0) + (vocabulary_diversity * 0.5))
            readability_score = max(0.0, 1.0 - (complexity_score * 0.8))
            
            return StyleFingerprint(
                avg_sentence_length=avg_sentence_length,
                avg_paragraph_length=avg_paragraph_length,
                vocabulary_diversity=vocabulary_diversity,
                common_words=common_words,
                punctuation_frequency=punctuation_frequency,
                dialogue_percentage=dialogue_percentage,
                description_percentage=description_percentage,
                action_percentage=action_percentage,
                tense_distribution=tense_distribution,
                pov_pattern=pov_pattern,
                tone_indicators=tone_indicators,
                complexity_score=complexity_score,
                readability_score=readability_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text style: {e}")
            # Return default fingerprint
            return StyleFingerprint(
                avg_sentence_length=15.0,
                avg_paragraph_length=70.0,
                vocabulary_diversity=0.5,
                common_words={},
                punctuation_frequency={},
                dialogue_percentage=0.33,
                description_percentage=0.33,
                action_percentage=0.33,
                tense_distribution={"past": 0.7, "present": 0.25, "future": 0.05},
                pov_pattern="third",
                tone_indicators={},
                complexity_score=0.5,
                readability_score=0.5
            )
    
    def _analyze_tone_indicators(self, text: str) -> Dict[str, float]:
        """Analyze tone indicators in the text."""
        
        tone_patterns = {
            "formal": re.compile(r'\b(therefore|furthermore|consequently|nevertheless|moreover)\b', re.IGNORECASE),
            "conversational": re.compile(r'\b(well|okay|yeah|sure|really|actually)\b', re.IGNORECASE),
            "emotional": re.compile(r'\b(love|hate|fear|joy|anger|sadness|excited|worried)\b', re.IGNORECASE),
            "descriptive": re.compile(r'\b(beautiful|gorgeous|stunning|magnificent|terrible|awful)\b', re.IGNORECASE),
            "action": re.compile(r'\b(suddenly|quickly|immediately|instantly|rapidly)\b', re.IGNORECASE)
        }
        
        tone_scores = {}
        total_words = len(self.word_pattern.findall(text))
        
        for tone, pattern in tone_patterns.items():
            matches = len(pattern.findall(text))
            tone_scores[tone] = matches / total_words if total_words > 0 else 0
        
        return tone_scores
    
    def validate_style_consistency(self, text: str, 
                                 baseline_fingerprint: Optional[StyleFingerprint] = None) -> StyleAnalysisResult:
        """Validate style consistency against baseline or active style guide."""
        
        try:
            # Analyze current text
            current_fingerprint = self.analyze_text_style(text)
            
            # Determine comparison baseline
            if baseline_fingerprint:
                target_fingerprint = baseline_fingerprint
                tolerance_ranges = self._generate_default_tolerances(target_fingerprint)
            elif self.active_style_guide:
                target_fingerprint = self.active_style_guide.target_fingerprint
                tolerance_ranges = self.active_style_guide.tolerance_ranges
            elif self.baseline_fingerprint:
                target_fingerprint = self.baseline_fingerprint
                tolerance_ranges = self._generate_default_tolerances(target_fingerprint)
            else:
                # No baseline available
                return StyleAnalysisResult(
                    fingerprint=current_fingerprint,
                    consistency_score=0.5,
                    style_deviations=["No baseline for comparison"],
                    recommendations=["Establish a baseline style or set an active style guide"],
                    confidence=0.0,
                    analyzed_word_count=len(self.word_pattern.findall(text))
                )
            
            # Calculate consistency score and deviations
            consistency_score, deviations = self._calculate_consistency_score(
                current_fingerprint, target_fingerprint, tolerance_ranges
            )
            
            # Generate recommendations
            recommendations = self._generate_style_recommendations(
                current_fingerprint, target_fingerprint, deviations
            )
            
            # Calculate confidence based on text length and analysis quality
            word_count = len(self.word_pattern.findall(text))
            confidence = min(1.0, word_count / 1000.0)  # Full confidence at 1000+ words
            
            return StyleAnalysisResult(
                fingerprint=current_fingerprint,
                consistency_score=consistency_score,
                style_deviations=deviations,
                recommendations=recommendations,
                confidence=confidence,
                analyzed_word_count=word_count
            )
            
        except Exception as e:
            logger.error(f"Error validating style consistency: {e}")
            return StyleAnalysisResult(
                fingerprint=self.analyze_text_style(text),
                consistency_score=0.0,
                style_deviations=[f"Analysis error: {e}"],
                recommendations=["Fix analysis errors"],
                confidence=0.0,
                analyzed_word_count=0
            )
    
    def _calculate_consistency_score(self, current: StyleFingerprint, 
                                   target: StyleFingerprint,
                                   tolerance_ranges: Dict[str, Tuple[float, float]]) -> Tuple[float, List[str]]:
        """Calculate consistency score and identify deviations."""
        
        scores = []
        deviations = []
        
        # Check numeric metrics
        numeric_metrics = [
            ("avg_sentence_length", current.avg_sentence_length, target.avg_sentence_length),
            ("avg_paragraph_length", current.avg_paragraph_length, target.avg_paragraph_length),
            ("vocabulary_diversity", current.vocabulary_diversity, target.vocabulary_diversity),
            ("dialogue_percentage", current.dialogue_percentage, target.dialogue_percentage),
            ("description_percentage", current.description_percentage, target.description_percentage),
            ("action_percentage", current.action_percentage, target.action_percentage),
            ("complexity_score", current.complexity_score, target.complexity_score),
            ("readability_score", current.readability_score, target.readability_score)
        ]
        
        for metric_name, current_value, target_value in numeric_metrics:
            if metric_name in tolerance_ranges:
                min_val, max_val = tolerance_ranges[metric_name]
                if min_val <= current_value <= max_val:
                    scores.append(1.0)
                else:
                    # Calculate how far outside the range
                    if current_value < min_val:
                        deviation = (min_val - current_value) / min_val
                        deviations.append(f"{metric_name} too low: {current_value:.2f} (expected: {min_val:.2f}-{max_val:.2f})")
                    else:
                        deviation = (current_value - max_val) / max_val
                        deviations.append(f"{metric_name} too high: {current_value:.2f} (expected: {min_val:.2f}-{max_val:.2f})")
                    
                    score = max(0.0, 1.0 - deviation)
                    scores.append(score)
            else:
                # Use percentage difference if no tolerance range
                if target_value > 0:
                    diff = abs(current_value - target_value) / target_value
                    score = max(0.0, 1.0 - diff)
                    scores.append(score)
                    
                    if diff > 0.2:  # 20% difference threshold
                        deviations.append(f"{metric_name} differs significantly: {current_value:.2f} vs {target_value:.2f}")
        
        # Check POV consistency
        if current.pov_pattern != target.pov_pattern:
            deviations.append(f"POV inconsistency: using {current.pov_pattern} person, expected {target.pov_pattern} person")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Check tense consistency
        tense_score = self._compare_tense_distributions(current.tense_distribution, target.tense_distribution)
        scores.append(tense_score)
        
        if tense_score < 0.8:
            deviations.append("Tense distribution differs from target style")
        
        # Calculate overall consistency score
        overall_score = statistics.mean(scores) if scores else 0.0
        
        return overall_score, deviations
    
    def _compare_tense_distributions(self, current: Dict[str, float], target: Dict[str, float]) -> float:
        """Compare tense distributions."""
        
        score = 0.0
        total_weight = 0.0
        
        for tense in ["past", "present", "future"]:
            current_val = current.get(tense, 0.0)
            target_val = target.get(tense, 0.0)
            
            if target_val > 0:
                diff = abs(current_val - target_val) / target_val
                tense_score = max(0.0, 1.0 - diff)
                score += tense_score * target_val
                total_weight += target_val
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def _generate_default_tolerances(self, fingerprint: StyleFingerprint) -> Dict[str, Tuple[float, float]]:
        """Generate default tolerance ranges for a fingerprint."""
        
        return {
            "avg_sentence_length": (fingerprint.avg_sentence_length * 0.8, fingerprint.avg_sentence_length * 1.2),
            "vocabulary_diversity": (fingerprint.vocabulary_diversity * 0.9, fingerprint.vocabulary_diversity * 1.1),
            "dialogue_percentage": (fingerprint.dialogue_percentage * 0.8, fingerprint.dialogue_percentage * 1.2),
            "complexity_score": (fingerprint.complexity_score * 0.9, fingerprint.complexity_score * 1.1),
            "readability_score": (fingerprint.readability_score * 0.9, fingerprint.readability_score * 1.1)
        }
    
    def _generate_style_recommendations(self, current: StyleFingerprint, 
                                      target: StyleFingerprint,
                                      deviations: List[str]) -> List[str]:
        """Generate recommendations for improving style consistency."""
        
        recommendations = []
        
        # Sentence length recommendations
        if abs(current.avg_sentence_length - target.avg_sentence_length) > 3:
            if current.avg_sentence_length > target.avg_sentence_length:
                recommendations.append("Consider using shorter, more concise sentences")
            else:
                recommendations.append("Consider varying sentence length with some longer, more complex sentences")
        
        # Dialogue recommendations
        if abs(current.dialogue_percentage - target.dialogue_percentage) > 0.1:
            if current.dialogue_percentage < target.dialogue_percentage:
                recommendations.append("Consider adding more dialogue to match target style")
            else:
                recommendations.append("Consider balancing dialogue with more narrative description")
        
        # Vocabulary recommendations
        if abs(current.vocabulary_diversity - target.vocabulary_diversity) > 0.1:
            if current.vocabulary_diversity < target.vocabulary_diversity:
                recommendations.append("Consider using more varied vocabulary")
            else:
                recommendations.append("Consider using simpler, more accessible language")
        
        # POV recommendations
        if current.pov_pattern != target.pov_pattern:
            recommendations.append(f"Maintain consistent {target.pov_pattern} person point of view")
        
        # General recommendations based on deviations
        if len(deviations) > 3:
            recommendations.append("Multiple style inconsistencies detected - consider reviewing target style guide")
        
        return recommendations
    
    def set_baseline_from_text(self, text: str):
        """Set baseline style fingerprint from a sample text."""
        
        try:
            self.baseline_fingerprint = self.analyze_text_style(text)
            logger.info("Baseline style fingerprint established from provided text")
            
        except Exception as e:
            logger.error(f"Error setting baseline from text: {e}")
    
    def get_available_style_guides(self) -> List[Dict[str, Any]]:
        """Get list of available style guides."""
        
        guides = []
        
        for style_type, guide in self.style_guides.items():
            guides.append({
                "type": style_type.value,
                "name": guide.name,
                "description": f"Style guide for {guide.name.lower()} writing",
                "target_sentence_length": guide.target_fingerprint.avg_sentence_length,
                "target_dialogue_percentage": guide.target_fingerprint.dialogue_percentage,
                "complexity_level": guide.target_fingerprint.complexity_score,
                "readability_level": guide.target_fingerprint.readability_score
            })
        
        return guides
    
    async def store_style_analysis(self, text: str, analysis_result: StyleAnalysisResult):
        """Store style analysis results."""
        
        if not self.db_utils:
            return
        
        try:
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "word_count": analysis_result.analyzed_word_count,
                "consistency_score": analysis_result.consistency_score,
                "confidence": analysis_result.confidence,
                "deviations_count": len(analysis_result.style_deviations),
                "fingerprint": {
                    "avg_sentence_length": analysis_result.fingerprint.avg_sentence_length,
                    "vocabulary_diversity": analysis_result.fingerprint.vocabulary_diversity,
                    "dialogue_percentage": analysis_result.fingerprint.dialogue_percentage,
                    "complexity_score": analysis_result.fingerprint.complexity_score,
                    "readability_score": analysis_result.fingerprint.readability_score,
                    "pov_pattern": analysis_result.fingerprint.pov_pattern
                }
            }
            
            await self.db_utils.execute_query(
                """
                INSERT INTO style_analyses 
                (analysis_data, created_at) 
                VALUES ($1, $2)
                """,
                json.dumps(analysis_data),
                datetime.now()
            )
            
            logger.info(f"Stored style analysis for {analysis_result.analyzed_word_count} words")
            
        except Exception as e:
            logger.error(f"Error storing style analysis: {e}")