"""
Approval workflow API endpoints for human-in-the-loop validation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .models import (
    ApprovalRequest, ProposalResponse, ApprovalDecision, 
    ValidationResult, Neo4jPushResult, ErrorResponse
)
from .db_utils import (
    create_proposal, get_proposal, list_proposals, 
    update_proposal_status, store_validation_result, get_validation_results
)
from .graph_utils import GraphitiClient
from .consistency_validators_fixed import run_all_validators

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/approval", tags=["approval"])

# Templates for UI
templates = Jinja2Templates(directory="templates")


@router.post("/propose", response_model=ProposalResponse)
async def create_new_proposal(request: ApprovalRequest):
    """
    Create a new proposal for human review.
    
    This endpoint accepts proposed entities/relationships and stores them
    for human validation before pushing to Neo4j.
    """
    try:
        # Create proposal in database
        proposal_id = await create_proposal(
            kind=request.kind,
            payload={
                "items": [item.dict() for item in request.items],
                "metadata": {
                    "total_items": len(request.items),
                    "item_types": list(set(item.type for item in request.items))
                }
            },
            source_doc=request.source_doc,
            suggested_by=request.suggested_by,
            confidence=request.confidence
        )
        
        # Run validation on each item
        validation_results = []
        for i, item in enumerate(request.items):
            try:
                validators = await run_all_validators(
                    content=item.excerpt or item.name,
                    entity_data=item.dict(),
                    established_facts=set()  # TODO: Get from graph
                )
                
                for validator_name, result in validators.items():
                    validation_result = ValidationResult(
                        validator_name=validator_name,
                        score=result.get("score", 0.5),
                        violations=result.get("violations", []),
                        suggestions=result.get("suggestions", [])
                    )
                    validation_results.append(validation_result)
                    
                    # Store in database
                    await store_validation_result(
                        proposal_id=proposal_id,
                        validator_name=validator_name,
                        score=validation_result.score,
                        violations=validation_result.violations,
                        suggestions=validation_result.suggestions
                    )
                    
            except Exception as e:
                logger.error(f"Validation failed for item {i}: {e}")
                # Continue with other items
        
        # Get the created proposal with validation data
        proposal_data = await get_proposal(proposal_id)
        if not proposal_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve created proposal")
        
        return ProposalResponse(
            proposal_id=proposal_id,
            status=proposal_data["status"],
            kind=proposal_data["kind"],
            confidence=proposal_data["confidence"],
            created_at=proposal_data["created_at"],
            validation_results=validation_results,
            risk_level=proposal_data.get("risk_level")
        )
        
    except Exception as e:
        logger.error(f"Failed to create proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{proposal_id}", response_model=ProposalResponse)
async def get_proposal_details(proposal_id: str):
    """
    Get detailed information about a proposal including validation results.
    """
    try:
        proposal_data = await get_proposal(proposal_id)
        if not proposal_data:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        # Get validation results
        validation_data = await get_validation_results(proposal_id)
        validation_results = [
            ValidationResult(
                validator_name=v["validator_name"],
                score=v["score"],
                violations=v["violations"],
                suggestions=v["suggestions"]
            )
            for v in validation_data
        ]
        
        return ProposalResponse(
            proposal_id=proposal_id,
            status=proposal_data["status"],
            kind=proposal_data["kind"],
            confidence=proposal_data["confidence"],
            created_at=proposal_data["created_at"],
            processed_at=proposal_data.get("processed_at"),
            processed_by=proposal_data.get("processed_by"),
            neo4j_tx=proposal_data.get("neo4j_tx"),
            validation_results=validation_results,
            risk_level=proposal_data.get("risk_level")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get proposal {proposal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{proposal_id}/approve")
async def approve_proposal(proposal_id: str, decision: ApprovalDecision):
    """
    Approve a proposal and push approved items to Neo4j.
    """
    try:
        proposal_data = await get_proposal(proposal_id)
        if not proposal_data:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        if proposal_data["status"] != "pending":
            raise HTTPException(
                status_code=400, 
                detail=f"Proposal is already {proposal_data['status']}"
            )
        
        if decision.action == "reject":
            # Simply mark as rejected
            await update_proposal_status(
                proposal_id=proposal_id,
                status="rejected",
                processed_by=decision.processed_by,
                rejection_reason=decision.rejection_reason
            )
            
            return {"status": "rejected", "message": "Proposal rejected"}
        
        # Approve and push to Neo4j
        try:
            payload = proposal_data["payload"]
            items = payload.get("items", [])
            
            # Filter items if partial approval
            if decision.selected_items:
                items = [items[i] for i in decision.selected_items if i < len(items)]
            
            # Push to Neo4j
            neo4j_result = await push_items_to_neo4j(items, decision.processed_by)
            
            # Update proposal status
            await update_proposal_status(
                proposal_id=proposal_id,
                status="approved",
                processed_by=decision.processed_by,
                neo4j_tx=neo4j_result.dict()
            )
            
            return {
                "status": "approved",
                "neo4j_result": neo4j_result.dict(),
                "message": f"Successfully pushed {neo4j_result.nodes_created} nodes and {neo4j_result.relationships_created} relationships"
            }
            
        except Exception as e:
            # Mark as failed
            await update_proposal_status(
                proposal_id=proposal_id,
                status="failed",
                processed_by=decision.processed_by,
                errors=[str(e)]
            )
            raise HTTPException(status_code=500, detail=f"Failed to push to Neo4j: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve proposal {proposal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ProposalResponse])
async def list_all_proposals(
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """
    List proposals with optional filtering.
    """
    try:
        proposals_data = await list_proposals(
            status=status,
            kind=kind,
            limit=limit,
            offset=offset
        )
        
        results = []
        for proposal in proposals_data:
            results.append(ProposalResponse(
                proposal_id=proposal["id"],
                status=proposal["status"],
                kind=proposal["kind"],
                confidence=proposal["confidence"],
                created_at=proposal["created_at"],
                processed_at=proposal.get("processed_at"),
                processed_by=proposal.get("processed_by"),
                risk_level=proposal.get("risk_level")
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list proposals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ui/review", response_class=HTMLResponse)
async def approval_ui(request: Request):
    """
    Serve the approval UI for human review.
    """
    try:
        # Get pending proposals
        pending_proposals = await list_proposals(status="pending", limit=10)
        
        return templates.TemplateResponse(
            "approval_flow.html",
            {
                "request": request,
                "proposals": pending_proposals,
                "title": "Proposal Review Dashboard"
            }
        )
    except Exception as e:
        logger.error(f"Failed to render approval UI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Internal function for Neo4j operations
async def push_items_to_neo4j(items: List[Dict[str, Any]], user_id: str) -> Neo4jPushResult:
    """
    Push approved items to Neo4j using MERGE queries for idempotency.
    """
    graph_client = GraphitiClient()
    await graph_client.initialize()
    
    try:
        nodes_created = 0
        relationships_created = 0
        nodes_updated = 0
        relationships_updated = 0
        errors = []
        
        # Process each item
        for item in items:
            try:
                item_type = item.get("type")
                
                if item_type == "character":
                    # Create character node
                    canonical_id = item.get("canonical_id") or str(uuid4())
                    
                    # Use Graphiti to add character information
                    episode_content = f"Character: {item['name']}\n"
                    if item.get("attributes"):
                        for key, value in item["attributes"].items():
                            episode_content += f"{key}: {value}\n"
                    
                    await graph_client.add_episode(
                        episode_id=f"character_{canonical_id}",
                        content=episode_content,
                        source=f"approval_{user_id}",
                        timestamp=datetime.now(),
                        metadata={
                            "type": "character",
                            "canonical_id": canonical_id,
                            "name": item["name"],
                            "attributes": item.get("attributes", {}),
                            "aliases": item.get("aliases", [])
                        }
                    )
                    nodes_created += 1
                    
                elif item_type == "relationship":
                    # Handle relationships through Graphiti episodes
                    for rel in item.get("relationships", []):
                        episode_content = f"Relationship: {item['name']} {rel.get('type', 'RELATED_TO')} {rel.get('to_name', 'unknown')}"
                        
                        await graph_client.add_episode(
                            episode_id=f"rel_{uuid4()}",
                            content=episode_content,
                            source=f"approval_{user_id}",
                            timestamp=datetime.now(),
                            metadata={
                                "type": "relationship",
                                "from_entity": item["name"],
                                "to_entity": rel.get("to_name"),
                                "relationship_type": rel.get("type", "RELATED_TO")
                            }
                        )
                        relationships_created += 1
                
            except Exception as e:
                error_msg = f"Failed to process item {item.get('name', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return Neo4jPushResult(
            transaction_id=str(uuid4()),  # Generate a transaction ID for tracking
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            nodes_updated=nodes_updated,
            relationships_updated=relationships_updated,
            errors=errors
        )
        
    finally:
        await graph_client.close()
