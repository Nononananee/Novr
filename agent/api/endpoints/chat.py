"""Chat endpoints for the API."""

import json
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..api_exceptions import APIBaseException, ServiceUnavailableError
from ..api_cache import cached_operation, cache_key_for_chat
from ..api_config import APIConfig
from ..api_utils import (
    ConversationManager,
    AgentExecutor,
    RequestValidator,
    ToolCallExtractor
)
from ...models import ChatRequest, ChatResponse, ToolCall
from ...monitoring.advanced_system_monitor import monitor_operation, ComponentType
from ...core.db_utils import add_message
from ...agent import rag_agent, AgentDependencies

# Import pydantic_ai untuk streaming functionality
try:
    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
@monitor_operation("chat_endpoint", ComponentType.API_LAYER)
async def chat(request: ChatRequest):
    """Enhanced non-streaming chat endpoint dengan caching dan retry."""
    try:
        # Validate dan sanitize request
        validated_data = RequestValidator.validate_chat_request(request.dict())
        
        # Get atau create session
        session_id = await ConversationManager.get_or_create_session(validated_data)
        
        # Check cache untuk response
        cache_key = cache_key_for_chat(session_id, validated_data["message"])
        
        async def execute_chat_operation():
            # Execute agent dengan utilities
            response, tools_used = await AgentExecutor.execute_agent(
                message=validated_data["message"],
                session_id=session_id,
                user_id=validated_data.get("user_id")
            )
            
            return {
                "message": response,
                "tools_used": [tool.dict() for tool in tools_used],
                "session_id": session_id,
                "metadata": {"search_type": str(request.search_type)}
            }
        
        # Execute dengan caching
        result = await cached_operation(
            cache_key=cache_key,
            operation_func=execute_chat_operation,
            ttl=APIConfig.CACHE_TTL
        )
        
        # Convert tools_used kembali ke ToolCall objects
        tools_used = [ToolCall(**tool) for tool in result["tools_used"]]
        
        return ChatResponse(
            message=result["message"],
            session_id=result["session_id"],
            tools_used=tools_used,
            metadata=result["metadata"]
        )
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise ServiceUnavailableError("chat", f"Chat service error: {e}")


@router.post("/stream")
@monitor_operation("chat_stream", ComponentType.API_LAYER)
async def chat_stream(request: ChatRequest):
    """Enhanced streaming chat endpoint dengan error handling."""
    try:
        # Validate request
        validated_data = RequestValidator.validate_chat_request(request.dict())
        
        # Get or create session
        session_id = await ConversationManager.get_or_create_session(validated_data)
        
        async def generate_stream():
            """Generate streaming response dengan comprehensive error handling."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\\n\\n"
                
                # Create dependencies
                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=validated_data.get("user_id")
                )
                
                # Get conversation context
                context = await ConversationManager.get_conversation_context(session_id)
                
                # Build input dengan context
                full_prompt = validated_data["message"]
                if context:
                    context_str = "\\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in context[-6:]
                    ])
                    full_prompt = f"Previous conversation:\\n{context_str}\\n\\nCurrent question: {validated_data['message']}"
                
                # Save user message immediately
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=validated_data["message"],
                    metadata={"user_id": validated_data.get("user_id")}
                )
                
                full_response = ""
                
                # Stream using agent.iter() pattern dengan error handling
                try:
                    async with rag_agent.iter(full_prompt, deps=deps) as run:
                        async for node in run:
                            if rag_agent.is_model_request_node(node):
                                # Stream tokens dari model
                                async with node.stream(run.ctx) as request_stream:
                                    async for event in request_stream:
                                        try:
                                            if not PYDANTIC_AI_AVAILABLE:
                                                # Fallback jika pydantic_ai tidak tersedia
                                                logger.warning("pydantic_ai not available, using fallback streaming")
                                                delta_content = str(event)
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\\n\\n"
                                                full_response += delta_content
                                                continue
                                            
                                            if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                                delta_content = event.part.content
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\\n\\n"
                                                full_response += delta_content
                                                
                                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                                delta_content = event.delta.content_delta
                                                yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\\n\\n"
                                                full_response += delta_content
                                        except Exception as stream_error:
                                            logger.warning(f"Stream event error: {stream_error}")
                                            continue
                    
                    # Extract tools used dari final result
                    result = run.result
                    tools_used = ToolCallExtractor.extract_tool_calls(result)
                    
                    # Send tools used information
                    if tools_used:
                        tools_data = [
                            {
                                "tool_name": tool.tool_name,
                                "args": tool.args,
                                "tool_call_id": tool.tool_call_id
                            }
                            for tool in tools_used
                        ]
                        yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\\n\\n"
                    
                    # Save assistant response
                    await add_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "streamed": True,
                            "tool_calls": len(tools_used)
                        }
                    )
                    
                    yield f"data: {json.dumps({'type': 'end'})}\\n\\n"
                    
                except Exception as agent_error:
                    logger.error(f"Agent streaming error: {agent_error}")
                    # Fallback response
                    fallback_response = "I apologize, but I encountered an error while processing your request."
                    yield f"data: {json.dumps({'type': 'text', 'content': fallback_response})}\\n\\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': str(agent_error)})}\\n\\n"
                    yield f"data: {json.dumps({'type': 'end'})}\\n\\n"
                
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                error_chunk = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\\n\\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except APIBaseException:
        raise
    except Exception as e:
        logger.error(f"Streaming chat setup failed: {e}")
        raise ServiceUnavailableError("chat_stream", f"Streaming chat error: {e}")