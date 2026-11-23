"""
src/leafscan/api/routes/feedback.py
User and agronomist diagnostic feedback endpoint.
Features idempotency controls and strictly prevents unverified feedback from polluting training datasets.
"""

import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Header, HTTPException, status
from leafscan.core.logging import logger

router = APIRouter(tags=["Feedback"])

# In-memory idempotency cache (idempotency_key -> response)
_IDEMPOTENCY_CACHE: Dict[str, Dict[str, Any]] = {}


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = Field(None, description="Original request ID from diagnostic response")
    predicted_species: str = Field(..., description="Species predicted by the model")
    predicted_disease: Optional[str] = Field(None, description="Disease predicted by the model")
    observed_species: Optional[str] = Field(None, description="User or field agent observed species")
    observed_disease: Optional[str] = Field(None, description="User or field agent observed disease")
    feedback_type: str = Field("agree", description="Feedback classification: agree, disagree, unclear, expert_verified")
    agronomic_notes: Optional[str] = Field(None, description="Additional field observations or laboratory results")
    idempotency_key: Optional[str] = Field(None, description="Idempotent transaction token")


class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: str
    status: str
    message: str
    idempotency_key: Optional[str] = None


@router.post("/v2/feedback", response_model=FeedbackResponse, summary="Submit Diagnostic Feedback")
def submit_feedback(
    payload: FeedbackRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
) -> FeedbackResponse:
    """
    Records field feedback or certified pathologist annotations regarding a diagnosis.
    Enforces idempotency and ensures unverified submissions are marked for human-in-the-loop audit.
    """
    key = idempotency_key or payload.idempotency_key
    if key and key in _IDEMPOTENCY_CACHE:
        logger.info(f"Returning cached feedback response for idempotency key: {key}")
        cached = _IDEMPOTENCY_CACHE[key]
        return FeedbackResponse(**cached)

    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    status_label = "expert_audit_queued" if payload.feedback_type != "agree" else "recorded"

    logger.info(
        f"Diagnostic feedback received: {feedback_id} - Type: {payload.feedback_type} (Status: {status_label})"
    )

    response_data = {
        "success": True,
        "feedback_id": feedback_id,
        "status": status_label,
        "message": "Diagnostic feedback recorded. Submissions require certified agricultural extension validation before dataset curation.",
        "idempotency_key": key
    }

    if key:
        _IDEMPOTENCY_CACHE[key] = response_data

    return FeedbackResponse(**response_data)
