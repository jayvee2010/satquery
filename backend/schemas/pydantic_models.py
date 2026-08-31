from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ImageMetadata(BaseModel):
    filename: str
    is_geotiff: bool
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    transform: List[float]
    bounds: Dict[str, float]
    latlon_bounds: Optional[Dict[str, Any]] = None
    pixel_size_x: float
    pixel_size_y: float

class SessionState(BaseModel):
    session_id: str
    optical_pre: Optional[ImageMetadata] = None
    optical_post: Optional[ImageMetadata] = None
    sar_post: Optional[ImageMetadata] = None
    active_optical_pre_path: Optional[str] = None
    active_optical_post_path: Optional[str] = None
    active_sar_post_path: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: str

class ExecutionStep(BaseModel):
    step_number: int
    name: str
    status: str # e.g. "SUCCESS", "WARNING", "FAILED"
    details: str

class EvidenceFusion(BaseModel):
    modalities_used: List[str]
    agreement_status: str # "CONFIRMED", "PARTIAL", "CONFLICTING", "INSUFFICIENT"
    optical_findings: Optional[str] = None
    sar_findings: Optional[str] = None
    confidence_assessment: str
    limitations: List[str]

class QueryResponse(BaseModel):
    answer: str
    confidence: str # e.g. "High", "Moderate", "Low", "Insufficient evidence"
    task_classified: str
    evidence_fusion: EvidenceFusion
    spatial_statistics: Optional[Dict[str, Any]] = None
    execution_trace: List[ExecutionStep]
    # Paths or visual indicators (like base64 grids) to display on map
    visual_evidence: Dict[str, Any] = {}
