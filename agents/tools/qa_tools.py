import logging
import json
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class QATools:
    def __init__(self):
        """Initialize QA tools for parsing and validation"""
        logger.info("Initialized QATools")
    
    def validate_qa_result(self, qa_result: Dict[str, Any], agent_type: str = "unknown") -> Dict[str, Any]:
        """
        Validate QA result structure and content
        
        Args:
            qa_result: QA result to validate
            agent_type: Type of QA agent (for logging)
            
        Returns:
            Validated and cleaned QA result
        """
        try:
            # Ensure required fields exist
            validated_result = {
                "score": 0,
                "issues": [],
                "patches": [],
                "agent_type": agent_type,
                "validated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Validate score
            if "score" in qa_result:
                score = qa_result["score"]
                if isinstance(score, (int, float)):
                    validated_result["score"] = max(0, min(100, int(score)))
                else:
                    logger.warning(f"Invalid score type in {agent_type} QA: {type(score)}")
                    validated_result["score"] = 50  # Default moderate score
            
            # Validate issues
            if "issues" in qa_result and isinstance(qa_result["issues"], list):
                validated_issues = []
                for issue in qa_result["issues"]:
                    if isinstance(issue, dict):
                        validated_issue = self._validate_issue(issue, agent_type)
                        if validated_issue:
                            validated_issues.append(validated_issue)
                validated_result["issues"] = validated_issues
            
            # Validate patches
            if "patches" in qa_result and isinstance(qa_result["patches"], list):
                validated_patches = []
                for patch in qa_result["patches"]:
                    if isinstance(patch, dict):
                        validated_patch = self._validate_patch(patch, agent_type)
                        if validated_patch:
                            validated_patches.append(validated_patch)
                validated_result["patches"] = validated_patches
            
            # Preserve additional fields
            for key, value in qa_result.items():
                if key not in ["score", "issues", "patches"] and key not in validated_result:
                    validated_result[key] = value
            
            logger.info(f"Validated {agent_type} QA result: score={validated_result['score']}, "
                       f"issues={len(validated_result['issues'])}, patches={len(validated_result['patches'])}")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Failed to validate {agent_type} QA result: {e}")
            return {
                "score": 50,
                "issues": [],
                "patches": [],
                "agent_type": agent_type,
                "validation_error": str(e),
                "validated_at": datetime.now(timezone.utc).isoformat()
            }
    
    def _validate_issue(self, issue: Dict[str, Any], agent_type: str) -> Optional[Dict[str, Any]]:
        """Validate individual issue structure"""
        try:
            validated_issue = {}
            
            # Required fields
            required_fields = ["loc", "type", "issue", "suggestion"]
            for field in required_fields:
                if field not in issue:
                    logger.warning(f"Missing required field '{field}' in {agent_type} issue")
                    return None
                validated_issue[field] = str(issue[field])
            
            # Validate location
            try:
                validated_issue["loc"] = max(0, int(issue["loc"]))
            except (ValueError, TypeError):
                validated_issue["loc"] = 0
            
            # Validate severity
            valid_severities = ["low", "medium", "high"]
            severity = issue.get("severity", "medium").lower()
            validated_issue["severity"] = severity if severity in valid_severities else "medium"
            
            # Agent-specific fields
            if agent_type == "character" and "character_name" in issue:
                validated_issue["character_name"] = str(issue["character_name"])
            
            return validated_issue
            
        except Exception as e:
            logger.error(f"Failed to validate issue: {e}")
            return None
    
    def _validate_patch(self, patch: Dict[str, Any], agent_type: str) -> Optional[Dict[str, Any]]:
        """Validate individual patch structure"""
        try:
            validated_patch = {}
            
            # Required fields
            if "loc" not in patch or "replacement" not in patch:
                logger.warning(f"Missing required fields in {agent_type} patch")
                return None
            
            # Validate location
            try:
                validated_patch["loc"] = max(0, int(patch["loc"]))
            except (ValueError, TypeError):
                validated_patch["loc"] = 0
            
            # Validate replacement text
            validated_patch["replacement"] = str(patch["replacement"])
            
            # Optional original text
            if "original" in patch:
                validated_patch["original"] = str(patch["original"])
            
            return validated_patch
            
        except Exception as e:
            logger.error(f"Failed to validate patch: {e}")
            return None
    
    def aggregate_qa_results(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple QA results into a comprehensive assessment
        
        Args:
            qa_results: List of QA results from different agents
            
        Returns:
            Aggregated QA assessment
        """
        try:
            if not qa_results:
                return {
                    "overall_score": 0,
                    "total_issues": 0,
                    "total_patches": 0,
                    "agent_scores": {},
                    "issue_breakdown": {},
                    "recommendations": []
                }
            
            # Calculate overall score (weighted average)
            agent_weights = {
                "technical": 0.3,
                "structural": 0.25,
                "character": 0.25,
                "style": 0.2
            }
            
            weighted_score = 0
            total_weight = 0
            agent_scores = {}
            all_issues = []
            all_patches = []
            
            for result in qa_results:
                agent_type = result.get("agent_type", "unknown")
                score = result.get("score", 0)
                weight = agent_weights.get(agent_type, 0.2)
                
                weighted_score += score * weight
                total_weight += weight
                agent_scores[agent_type] = score
                
                all_issues.extend(result.get("issues", []))
                all_patches.extend(result.get("patches", []))
            
            overall_score = int(weighted_score / total_weight) if total_weight > 0 else 0
            
            # Analyze issue breakdown
            issue_breakdown = {}
            severity_breakdown = {"low": 0, "medium": 0, "high": 0}
            
            for issue in all_issues:
                issue_type = issue.get("type", "unknown")
                severity = issue.get("severity", "medium")
                
                issue_breakdown[issue_type] = issue_breakdown.get(issue_type, 0) + 1
                severity_breakdown[severity] += 1
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_score, agent_scores, issue_breakdown, severity_breakdown
            )
            
            aggregated_result = {
                "overall_score": overall_score,
                "total_issues": len(all_issues),
                "total_patches": len(all_patches),
                "agent_scores": agent_scores,
                "issue_breakdown": issue_breakdown,
                "severity_breakdown": severity_breakdown,
                "recommendations": recommendations,
                "requires_revision": overall_score < 75 or severity_breakdown["high"] > 0,
                "aggregated_at": datetime.now(timezone.utc).isoformat()
            }

            # Detect correlated issues across agents by line/location
            correlation_map = {}
            for issue in all_issues:
                # Normalize location key
                key = None
                if isinstance(issue, dict):
                    if "line_start" in issue and "line_end" in issue:
                        key = f"{issue['line_start']}-{issue['line_end']}"
                    elif "location" in issue:
                        key = str(issue["location"]) 
                    elif "line" in issue:
                        key = str(issue["line"]) 
                    elif "loc" in issue:
                        key = str(issue["loc"]) 

                if not key:
                    key = "unknown"

                correlation_map.setdefault(key, []).append(issue)

            correlated = {k: v for k, v in correlation_map.items() if len(v) > 1 and k != "unknown"}
            aggregated_result["correlated_issues"] = correlated
            
            logger.info(f"Aggregated QA results: overall_score={overall_score}, "
                       f"total_issues={len(all_issues)}, requires_revision={aggregated_result['requires_revision']}")
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Failed to aggregate QA results: {e}")
            return {
                "overall_score": 0,
                "total_issues": 0,
                "total_patches": 0,
                "agent_scores": {},
                "issue_breakdown": {},
                "recommendations": ["QA aggregation failed - manual review required"],
                "error": str(e)
            }
    
    def _generate_recommendations(self, 
                                overall_score: int,
                                agent_scores: Dict[str, int],
                                issue_breakdown: Dict[str, int],
                                severity_breakdown: Dict[str, int]) -> List[str]:
        """Generate recommendations based on QA analysis"""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 60:
            recommendations.append("Major revision required - consider rewriting significant portions")
        elif overall_score < 75:
            recommendations.append("Moderate revision needed - address key issues before proceeding")
        elif overall_score < 85:
            recommendations.append("Minor improvements needed - polish and refine")
        else:
            recommendations.append("Excellent quality - ready for publication")
        
        # Agent-specific recommendations
        for agent_type, score in agent_scores.items():
            if score < 70:
                if agent_type == "technical":
                    recommendations.append("Focus on grammar, spelling, and formatting improvements")
                elif agent_type == "structural":
                    recommendations.append("Review plot flow, pacing, and story structure")
                elif agent_type == "character":
                    recommendations.append("Strengthen character consistency and dialogue authenticity")
                elif agent_type == "style":
                    recommendations.append("Improve writing style, tone consistency, and prose quality")
        
        # Issue-specific recommendations
        if issue_breakdown.get("dialogue", 0) > 3:
            recommendations.append("Multiple dialogue issues detected - review character voices")
        
        if issue_breakdown.get("pacing", 0) > 2:
            recommendations.append("Pacing issues found - consider scene structure and transitions")
        
        if issue_breakdown.get("consistency", 0) > 2:
            recommendations.append("Consistency problems detected - review character and plot continuity")
        
        # Severity-based recommendations
        if severity_breakdown["high"] > 0:
            recommendations.append(f"{severity_breakdown['high']} high-severity issues require immediate attention")
        
        if severity_breakdown["medium"] > 5:
            recommendations.append("Multiple medium-severity issues - prioritize fixes before proceeding")
        
        return recommendations
    
    async def run_parallel_qa(self, 
                            text: str,
                            qa_agents: List[Any],
                            context: Dict[str, Any] = None,
                            timeout: int = 30) -> List[Dict[str, Any]]:
        """
        Run multiple QA agents in parallel
        
        Args:
            text: Text to analyze
            qa_agents: List of QA agent instances
            context: Context information for agents
            timeout: Timeout in seconds
            
        Returns:
            List of QA results
        """
        try:
            logger.info(f"Running parallel QA with {len(qa_agents)} agents")
            
            # Create tasks for each QA agent
            tasks = []
            for agent in qa_agents:
                if hasattr(agent, 'review'):
                    if hasattr(agent, '__class__') and 'Character' in agent.__class__.__name__:
                        # Character QA needs character context
                        task = agent.review(text, context.get("characters", {}))
                    elif hasattr(agent, '__class__') and 'Structural' in agent.__class__.__name__:
                        # Structural QA needs plot context
                        task = agent.review(text, context.get("plot", {}))
                    elif hasattr(agent, '__class__') and 'Style' in agent.__class__.__name__:
                        # Style QA needs style context
                        task = agent.review(text, context.get("style", {}))
                    else:
                        # Technical QA and others
                        task = agent.review(text)
                    
                    tasks.append(task)
            
            # Run tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results and handle exceptions
            qa_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"QA agent {i} failed: {result}")
                    # Create fallback result
                    qa_results.append({
                        "score": 50,
                        "issues": [],
                        "patches": [],
                        "agent_type": f"agent_{i}",
                        "error": str(result)
                    })
                else:
                    qa_results.append(result)
            
            logger.info(f"Parallel QA completed: {len(qa_results)} results")
            return qa_results
            
        except asyncio.TimeoutError:
            logger.error(f"Parallel QA timed out after {timeout} seconds")
            return []
        except Exception as e:
            logger.error(f"Failed to run parallel QA: {e}")
            return []
    
    def format_qa_summary(self, aggregated_result: Dict[str, Any]) -> str:
        """Format QA results into human-readable summary"""
        try:
            summary_parts = []
            
            # Overall assessment
            overall_score = aggregated_result.get("overall_score", 0)
            summary_parts.append(f"=== QUALITY ASSESSMENT SUMMARY ===")
            summary_parts.append(f"Overall Score: {overall_score}/100")
            
            # Quality level
            if overall_score >= 90:
                quality_level = "Excellent"
            elif overall_score >= 80:
                quality_level = "Good"
            elif overall_score >= 70:
                quality_level = "Acceptable"
            elif overall_score >= 60:
                quality_level = "Needs Improvement"
            else:
                quality_level = "Poor"
            
            summary_parts.append(f"Quality Level: {quality_level}")
            
            # Agent scores
            agent_scores = aggregated_result.get("agent_scores", {})
            if agent_scores:
                summary_parts.append("\n=== AGENT SCORES ===")
                for agent_type, score in agent_scores.items():
                    summary_parts.append(f"{agent_type.title()}: {score}/100")
            
            # Issue summary
            total_issues = aggregated_result.get("total_issues", 0)
            severity_breakdown = aggregated_result.get("severity_breakdown", {})
            
            if total_issues > 0:
                summary_parts.append(f"\n=== ISSUES FOUND ===")
                summary_parts.append(f"Total Issues: {total_issues}")
                summary_parts.append(f"High Severity: {severity_breakdown.get('high', 0)}")
                summary_parts.append(f"Medium Severity: {severity_breakdown.get('medium', 0)}")
                summary_parts.append(f"Low Severity: {severity_breakdown.get('low', 0)}")
            
            # Recommendations
            recommendations = aggregated_result.get("recommendations", [])
            if recommendations:
                summary_parts.append("\n=== RECOMMENDATIONS ===")
                for i, rec in enumerate(recommendations, 1):
                    summary_parts.append(f"{i}. {rec}")
            
            # Revision requirement
            requires_revision = aggregated_result.get("requires_revision", False)
            summary_parts.append(f"\n=== ACTION REQUIRED ===")
            summary_parts.append("Revision Required: " + ("Yes" if requires_revision else "No"))
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to format QA summary: {e}")
            return f"QA Summary formatting failed: {str(e)}"