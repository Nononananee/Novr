"""
Utility functions untuk API operations yang dapat di-test secara terpisah.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..models import ToolCall
from .api_exceptions import ValidationError, DatabaseError
from .api_retry import retry_decorator

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manager untuk conversation operations."""
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def get_or_create_session(request_data: Dict[str, Any]) -> str:
        """Get existing session atau create new one."""
        from .db_utils import get_session, create_session
        
        session_id = request_data.get("session_id")
        
        if session_id:
            try:
                session = await get_session(session_id)
                if session:
                    return session_id
            except Exception as e:
                logger.warning(f"Failed to get session {session_id}: {e}")
        
        # Create new session
        try:
            return await create_session(
                user_id=request_data.get("user_id"),
                metadata=request_data.get("metadata", {})
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create session: {e}", "create_session")
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def get_conversation_context(
        session_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        from .db_utils import get_session_messages
        
        try:
            messages = await get_session_messages(session_id, limit=max_messages)
            
            return [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []  # Return empty context pada error
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def save_conversation_turn(
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save conversation turn ke database."""
        from .db_utils import add_message
        
        try:
            # Save user message
            await add_message(
                session_id=session_id,
                role="user",
                content=user_message,
                metadata=metadata or {}
            )
            
            # Save assistant message
            await add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_message,
                metadata=metadata or {}
            )
        except Exception as e:
            raise DatabaseError(f"Failed to save conversation: {e}", "save_conversation")


class ToolCallExtractor:
    """Extractor untuk tool calls dari AI result."""
    
    @staticmethod
    def extract_tool_calls(result) -> List[ToolCall]:
        """Extract tool calls dari Pydantic AI result."""
        tools_used = []
        
        try:
            messages = result.all_messages()
            
            for message in messages:
                if hasattr(message, 'parts'):
                    for part in message.parts:
                        if part.__class__.__name__ == 'ToolCallPart':
                            try:
                                tool_call = ToolCallExtractor._parse_tool_call_part(part)
                                if tool_call:
                                    tools_used.append(tool_call)
                            except Exception as e:
                                logger.debug(f"Failed to parse tool call part: {e}")
                                continue
        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
        
        return tools_used
    
    @staticmethod
    def _parse_tool_call_part(part) -> Optional[ToolCall]:
        """Parse single tool call part."""
        try:
            # Extract tool information
            tool_name = str(part.tool_name) if hasattr(part, 'tool_name') else 'unknown'
            
            # Get args
            tool_args = {}
            if hasattr(part, 'args') and part.args is not None:
                if isinstance(part.args, str):
                    try:
                        tool_args = json.loads(part.args)
                    except json.JSONDecodeError:
                        tool_args = {}
                elif isinstance(part.args, dict):
                    tool_args = part.args
            
            # Alternative: use args_as_dict method if available
            if hasattr(part, 'args_as_dict'):
                try:
                    tool_args = part.args_as_dict()
                except:
                    pass
            
            # Get tool call ID
            tool_call_id = None
            if hasattr(part, 'tool_call_id'):
                tool_call_id = str(part.tool_call_id) if part.tool_call_id else None
            
            return ToolCall(
                tool_name=tool_name,
                args=tool_args,
                tool_call_id=tool_call_id
            )
        except Exception as e:
            logger.debug(f"Failed to parse tool call: {e}")
            return None


class AgentExecutor:
    """Executor untuk agent operations."""
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def execute_agent(
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        save_conversation: bool = True
    ) -> Tuple[str, List[ToolCall]]:
        """Execute agent dengan message."""
        from .agent import rag_agent, AgentDependencies
        
        try:
            # Create dependencies
            deps = AgentDependencies(
                session_id=session_id,
                user_id=user_id
            )
            
            # Get conversation context
            context = await ConversationManager.get_conversation_context(session_id)
            
            # Build prompt dengan context
            full_prompt = AgentExecutor._build_prompt_with_context(message, context)
            
            # Run agent
            result = await rag_agent.run(full_prompt, deps=deps)
            
            response = result.data
            tools_used = ToolCallExtractor.extract_tool_calls(result)
            
            # Save conversation jika diminta
            if save_conversation:
                await ConversationManager.save_conversation_turn(
                    session_id=session_id,
                    user_message=message,
                    assistant_message=response,
                    metadata={
                        "user_id": user_id,
                        "tool_calls": len(tools_used)
                    }
                )
            
            return response, tools_used
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            
            if save_conversation:
                try:
                    await ConversationManager.save_conversation_turn(
                        session_id=session_id,
                        user_message=message,
                        assistant_message=error_response,
                        metadata={"error": str(e)}
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save error conversation: {save_error}")
            
            return error_response, []
    
    @staticmethod
    def _build_prompt_with_context(message: str, context: List[Dict[str, str]]) -> str:
        """Build prompt dengan conversation context."""
        if not context:
            return message
        
        context_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context[-6:]  # Last 3 turns
        ])
        
        return f"Previous conversation:\n{context_str}\n\nCurrent question: {message}"


class SearchOperations:
    """Operations untuk berbagai jenis search."""
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def execute_vector_search(query: str, limit: int = 10) -> Tuple[List[Any], float]:
        """Execute vector search dan return results dengan timing."""
        from .tools import vector_search_tool, VectorSearchInput
        
        start_time = datetime.now()
        
        try:
            input_data = VectorSearchInput(query=query, limit=limit)
            results = await vector_search_tool(input_data)
            
            end_time = datetime.now()
            query_time = (end_time - start_time).total_seconds() * 1000
            
            return results, query_time
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def execute_graph_search(query: str) -> Tuple[List[Any], float]:
        """Execute graph search dan return results dengan timing."""
        from .tools import graph_search_tool, GraphSearchInput
        
        start_time = datetime.now()
        
        try:
            input_data = GraphSearchInput(query=query)
            results = await graph_search_tool(input_data)
            
            end_time = datetime.now()
            query_time = (end_time - start_time).total_seconds() * 1000
            
            return results, query_time
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise
    
    @staticmethod
    @retry_decorator(max_retries=2)
    async def execute_hybrid_search(query: str, limit: int = 10) -> Tuple[List[Any], float]:
        """Execute hybrid search dan return results dengan timing."""
        from .tools import hybrid_search_tool, HybridSearchInput
        
        start_time = datetime.now()
        
        try:
            input_data = HybridSearchInput(query=query, limit=limit)
            results = await hybrid_search_tool(input_data)
            
            end_time = datetime.now()
            query_time = (end_time - start_time).total_seconds() * 1000
            
            return results, query_time
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise


class HealthChecker:
    """Health check operations."""
    
    @staticmethod
    async def check_database_health() -> bool:
        """Check database health."""
        try:
            from .db_utils import test_connection
            return await test_connection()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @staticmethod
    async def check_graph_health() -> bool:
        """Check graph database health."""
        try:
            from .graph_utils import test_graph_connection
            return await test_graph_connection()
        except Exception as e:
            logger.error(f"Graph health check failed: {e}")
            return False
    
    @staticmethod
    async def get_comprehensive_health() -> Dict[str, Any]:
        """Get comprehensive health status."""
        db_status = await HealthChecker.check_database_health()
        graph_status = await HealthChecker.check_graph_health()
        
        # Determine overall status
        if db_status and graph_status:
            status = "healthy"
        elif db_status or graph_status:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "database": db_status,
            "graph_database": graph_status,
            "llm_connection": True,  # Assume OK if we can respond
            "timestamp": datetime.now()
        }


class RequestValidator:
    """Validator untuk API requests."""
    
    @staticmethod
    def validate_chat_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chat request data."""
        from .input_validation import validate_text_input
        
        message = request_data.get("message", "")
        
        validation_result = validate_text_input(
            message,
            min_length=1,
            max_length=50000,
            sanitize=True,
            check_security=True
        )
        
        if not validation_result["valid"]:
            raise ValidationError(
                f"Invalid input: {', '.join(validation_result['errors'])}",
                field="message"
            )
        
        # Return sanitized data
        return {
            **request_data,
            "message": validation_result["sanitized_data"]
        }
    
    @staticmethod
    def validate_search_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search request data."""
        from .input_validation import validate_text_input, validate_numeric_input
        
        query = request_data.get("query", "")
        limit = request_data.get("limit", 10)
        
        # Validate query
        query_validation = validate_text_input(
            query,
            min_length=1,
            max_length=1000,
            sanitize=True
        )
        
        if not query_validation["valid"]:
            raise ValidationError(
                f"Invalid query: {', '.join(query_validation['errors'])}",
                field="query"
            )
        
        # Validate limit
        limit_validation = validate_numeric_input(
            limit,
            min_value=1,
            max_value=100
        )
        
        if not limit_validation["valid"]:
            raise ValidationError(
                f"Invalid limit: {', '.join(limit_validation['errors'])}",
                field="limit"
            )
        
        return {
            **request_data,
            "query": query_validation["sanitized_data"],
            "limit": limit_validation["validated_data"]
        }
