import logging
import os
from typing import Dict, Any, List, Optional
import asyncio

from agents.generator_agent import GeneratorAgent
from qa_system.structural_qa import StructuralQAAgent
from qa_system.character_qa import CharacterQAAgent
from qa_system.style_qa import StyleQAAgent
from agents.technical_qa import TechnicalQAAgent
import agents.tools.context_tools as context_tools
from agents.tools.novel_tools import NovelTools
from agents.tools.qa_tools import QATools

logger = logging.getLogger(__name__)


class NovelGenerationCrew:
    def __init__(
        self,
        qdrant_client=None,
        neo4j_client=None,
        mongodb_client=None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize Novel Generation Crew with dependency injection.

        This class resolves API keys and model names from the passed
        parameters first, then from environment variables that exist in
        the project's `.env.example` (OPENAI_API_KEY, OPENROUTER_API_KEY,
        GEMINI_API_KEY, HUGGINGFACE_API_KEY). It does not modify the
        environment file.
        """
        # Store injected clients
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        self.mongodb_client = mongodb_client

        # Resolve API key: prefer explicit parameter, then common env vars.
        resolved_api_key = (
            openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        )

        # Resolve generator model: prefer OPENROUTER_MODEL, else default
        resolved_model = os.getenv("OPENROUTER_MODEL") or os.getenv("DEFAULT_MODEL") or "gpt-4o-mini"

        # Initialize helper tools
        self.context_retriever = context_tools.ContextRetriever(self.qdrant_client, self.neo4j_client)
        self.novel_tools = NovelTools(self.mongodb_client, self.neo4j_client)
        self.qa_tools = QATools()

        # Keep resolved key/model on instance
        self.openai_api_key = resolved_api_key
        self._resolved_model = resolved_model

        # If no API key is available, fail fast (tests expect this)
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Instantiate agents immediately when key is provided
        self.generator_agent = GeneratorAgent(api_key=self.openai_api_key, model=self._resolved_model)
        self.structural_qa = StructuralQAAgent(api_key=self.openai_api_key)
        self.character_qa = CharacterQAAgent(api_key=self.openai_api_key)
        self.style_qa = StyleQAAgent(api_key=self.openai_api_key)
        self.technical_qa = TechnicalQAAgent(api_key=self.openai_api_key)

        logger.info("Initialized NovelGenerationCrew")

    async def execute_generation_workflow(
        self, project_id: str, prompt: str, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the complete novel generation workflow using the crew.

        The workflow steps remain the same as before: retrieve context,
        generate draft, run parallel QA, possibly revise, and return
        aggregated results. This method intentionally does not change
        orchestration logic; it only ensures the crew is initialized with
        keys/models compatible with the provided `.env.example`.
        """
        try:
            logger.info(f"Starting novel generation workflow for project {project_id}")

            # Step 1: Retrieve comprehensive context
            context = await self.context_retriever.retrieve_comprehensive_context(
                project_id=project_id, query=prompt, semantic_top_k=4
            )

            # Ensure agents are instantiated (lazy)
            if self.generator_agent is None:
                if not self.openai_api_key:
                    raise AttributeError("GeneratorAgent requires an API key")

                self.generator_agent = GeneratorAgent(api_key=self.openai_api_key, model=self._resolved_model)
                self.structural_qa = StructuralQAAgent(api_key=self.openai_api_key)
                self.character_qa = CharacterQAAgent(api_key=self.openai_api_key)
                self.style_qa = StyleQAAgent(api_key=self.openai_api_key)
                self.technical_qa = TechnicalQAAgent(api_key=self.openai_api_key)

            # Step 2: Generate initial draft
            logger.info("Generating initial draft...")
            draft_content = await self.generator_agent.generate(
                prompt=prompt,
                context=context.get("combined_text", ""),
                length_words=settings.get("length_words", 1200),
                temperature=settings.get("temperature", 0.7),
                tone=settings.get("tone", "engaging"),
                style=settings.get("style", "narrative"),
            )

            # Step 3: Run parallel QA analysis
            logger.info("Running parallel QA analysis...")
            qa_agents = [self.structural_qa, self.character_qa, self.style_qa, self.technical_qa]

            qa_context = {
                "characters": context.get("characters", {}),
                "plot": await self.context_retriever.get_plot_context(project_id),
                "style": {
                    "target_tone": settings.get("tone", "engaging"),
                    "writing_style": settings.get("style", "narrative"),
                    "genre": settings.get("genre", "fantasy"),
                },
            }

            qa_results = await self.qa_tools.run_parallel_qa(
                text=draft_content, qa_agents=qa_agents, context=qa_context, timeout=30
            )

            # Step 4: Validate and aggregate QA results
            validated_results = []
            agent_types = ["structural", "character", "style", "technical"]

            for i, result in enumerate(qa_results):
                agent_type = agent_types[i] if i < len(agent_types) else f"agent_{i}"
                validated_result = self.qa_tools.validate_qa_result(result, agent_type)
                validated_results.append(validated_result)

            aggregated_qa = self.qa_tools.aggregate_qa_results(validated_results)

            # Step 5: Determine if revision is needed
            final_content = draft_content
            revision_count = 0
            max_revisions = settings.get("max_revision_rounds", 2)

            if aggregated_qa.get("requires_revision", False) and revision_count < max_revisions:
                logger.info("Revision required - generating improved version...")

                # Build feedback from QA results
                feedback_parts: List[str] = []
                for result in validated_results:
                    if result.get("issues"):
                        agent_type = result.get("agent_type", "unknown")
                        feedback_parts.append(f"\n{agent_type.title()} Issues:")
                        for issue in result["issues"][:3]:
                            feedback_parts.append(f"- {issue['issue']}: {issue.get('suggestion', '')}")

                feedback = "\n".join(feedback_parts)

                # Generate revised version
                final_content = await self.generator_agent.revise_with_feedback(
                    original_content=draft_content,
                    feedback=feedback,
                    context=context.get("combined_text", ""),
                    temperature=settings.get("temperature", 0.6),
                )

                revision_count += 1

                # Re-run technical QA on final version
                final_qa = await self.technical_qa.review(final_content)
                final_qa_validated = self.qa_tools.validate_qa_result(final_qa, "technical")
                aggregated_qa["final_technical_score"] = final_qa_validated.get("score", 0)

            # Step 6: Prepare final result
            result = {
                "content": final_content,
                "word_count": len(final_content.split()),
                "character_count": len(final_content),
                "revision_count": revision_count,
                "qa_analysis": aggregated_qa,
                "context_used": {
                    "semantic_chunks": len(context.get("semantic", {}).get("chunks", [])),
                    "characters_referenced": len(context.get("characters", {}).get("characters", [])),
                    "relationships_considered": len(
                        context.get("characters", {}).get("relationships", [])
                    ),
                },
                "generation_metadata": {
                    "project_id": project_id,
                    "prompt": prompt,
                    "settings": settings,
                    "workflow_version": "2.0",
                },
            }

            logger.info(
                f"Novel generation workflow completed: "
                f"score={aggregated_qa.get('overall_score', 0)}, "
                f"revisions={revision_count}, "
                f"words={result['word_count']}"
            )

            return result

        except Exception as e:
            logger.error(f"Novel generation workflow failed: {e}")
            return {
                "content": "",
                "error": str(e),
                "word_count": 0,
                "character_count": 0,
                "revision_count": 0,
                "qa_analysis": {"overall_score": 0, "error": str(e)},
            }
            
            logger.info(f"Novel generation workflow completed: "
                       f"score={aggregated_qa.get('overall_score', 0)}, "
                       f"revisions={revision_count}, "
                       f"words={result['word_count']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Novel generation workflow failed: {e}")
            return {
                "content": "",
                "error": str(e),
                "word_count": 0,
                "character_count": 0,
                "revision_count": 0,
                "qa_analysis": {"overall_score": 0, "error": str(e)}
            }