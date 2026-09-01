import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, Optional

from backend.schemas.pydantic_models import QueryRequest, QueryResponse, SessionState
from backend.models.registry import ModelRegistry
from backend.agents.orchestrator import OrchestratorAgent
from backend.tools.geospatial import validate_tiff, classify_input_configuration

from fastapi.staticfiles import StaticFiles

# Directories setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEMO_DIR = os.path.join(BASE_DIR, "demo")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Remote Sensing Vision-Language Assistant (ISRO PS 26167)",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve demo imagery static files
app.mount("/api/demo", StaticFiles(directory=DEMO_DIR), name="demo")

# Shared memory session storage
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Initialize Registry and Orchestrator
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
weights_path = os.path.join(MODEL_DIR, "adapted_rs_model.pth")
registry = ModelRegistry(adapted_model_path=weights_path)
orchestrator = OrchestratorAgent(model_registry=registry)

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "gemini_enabled": orchestrator.use_gemini}

@app.get("/api/models")
def list_models():
    """Returns active models and tools in the registry."""
    return registry.list_models()

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Retrieves session state, active images, and input configuration."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id": session_id,
            "image_1": None,
            "image_2": None,
            "active_image_1_path": None,
            "active_image_2_path": None,
            "configuration": "EMPTY",
            "detected_modalities": [],
            "spatial_alignment": False,
            "analyses": []
        }
    
    state = SESSIONS[session_id]
    config_info = classify_input_configuration(state.get("image_1"), state.get("image_2"))
    
    return {
        "session_id": state["session_id"],
        "image_1": state.get("image_1"),
        "image_2": state.get("image_2"),
        "configuration": config_info["configuration"],
        "detected_modalities": config_info.get("modalities", []),
        "spatial_alignment": config_info.get("spatially_aligned", False),
        "details": config_info.get("details", "")
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    slot: Optional[str] = Form(None), # "image_1", "image_2"
    modality: Optional[str] = Form(None), # legacy compatibility
    session_id: str = Form(...)
):
    """
    Uploads a satellite image (GeoTIFF), validates its geospatial metadata,
    infers its sensor modality, and updates the workspace configuration.
    """
    # Resolve target slot
    target_slot = slot or modality or "image_1"
    if target_slot in ["optical_post", "optical", "sar"]:
        target_slot = "image_1"
    elif target_slot in ["optical_pre", "sar_post"]:
        target_slot = "image_2"
        
    if target_slot not in ["image_1", "image_2"]:
        target_slot = "image_1"

    if session_id not in SESSIONS:
        get_session(session_id)
        
    session_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".tif"
        
    file_path = os.path.join(session_dir, f"{target_slot}_{uuid.uuid4().hex[:6]}{ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Validate TIFF metadata & auto-detect modality
        metadata = validate_tiff(file_path)
        
        # Save to session
        SESSIONS[session_id][target_slot] = metadata
        SESSIONS[session_id][f"active_{target_slot}_path"] = file_path
        
        # Also maintain legacy keys for backward compatibility
        if metadata.get("modality") == "SAR":
            SESSIONS[session_id]["sar_post"] = metadata
            SESSIONS[session_id]["active_sar_post_path"] = file_path
        elif target_slot == "image_2" or "pre" in file.filename.lower():
            SESSIONS[session_id]["optical_pre"] = metadata
            SESSIONS[session_id]["active_optical_pre_path"] = file_path
        else:
            SESSIONS[session_id]["optical_post"] = metadata
            SESSIONS[session_id]["active_optical_post_path"] = file_path
            
        # Update input configuration
        config_info = classify_input_configuration(
            SESSIONS[session_id].get("image_1"), 
            SESSIONS[session_id].get("image_2")
        )
        SESSIONS[session_id]["configuration"] = config_info["configuration"]
        SESSIONS[session_id]["detected_modalities"] = config_info["modalities"]
        SESSIONS[session_id]["spatial_alignment"] = config_info["spatially_aligned"]
        
        return JSONResponse(status_code=200, content={
            "message": f"Successfully uploaded and validated {target_slot}.",
            "slot": target_slot,
            "metadata": metadata,
            "configuration": config_info
        })
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Image upload validation failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_satquery(request: QueryRequest):
    """
    Submits a natural language query to the Orchestrator Agent for investigation.
    """
    session_id = request.session_id
    if session_id not in SESSIONS:
        get_session(session_id)
        
    session = SESSIONS[session_id]
    
    try:
        result = orchestrator.execute_workflow(request.query, session)
        
        session["analyses"].append({
            "query": request.query,
            "answer": result["answer"],
            "task": result["task_classified"],
            "confidence": result["confidence"],
            "input_configuration": result.get("input_configuration"),
            "stats": result.get("spatial_statistics"),
            "limitations": result["evidence_fusion"]["limitations"]
        })
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis orchestration error: {str(e)}")

@app.get("/api/report/{session_id}")
def download_report(session_id: str):
    """
    Generates and downloads a markdown investigation report for the active session.
    """
    if session_id not in SESSIONS or not SESSIONS[session_id]["analyses"]:
        raise HTTPException(status_code=404, detail="No analyses found for this session. Please perform an investigation query first.")
        
    session = SESSIONS[session_id]
    last_analysis = session["analyses"][-1]
    
    report_lines = [
        "# SATQUERY AI - MULTIMODAL REMOTE SENSING INVESTIGATION REPORT",
        f"**Session ID:** {session_id}",
        f"**ISRO Problem Statement:** 26167",
        f"**Input Configuration:** {last_analysis.get('input_configuration', 'SINGLE_IMAGE')}",
        "---",
        "## 1. Imagery Inputs Used",
    ]
    
    for slot_name in ["image_1", "image_2"]:
        meta = session.get(slot_name)
        if meta:
            report_lines.append(f"### Input: {slot_name.upper()} ({meta.get('modality', 'UNKNOWN')})")
            report_lines.append(f"- **Filename:** {meta['filename']}")
            report_lines.append(f"- **Dimensions:** {meta['width']} x {meta['height']} ({meta['bands']} bands)")
            report_lines.append(f"- **CRS:** {meta['crs']}")
            if meta.get("latlon_bounds") and meta["latlon_bounds"].get("center"):
                c = meta["latlon_bounds"]["center"]
                report_lines.append(f"- **Center (Lat/Lon):** {c[0]:.5f}°N, {c[1]:.5f}°E")
        else:
            report_lines.append(f"### Input: {slot_name.upper()}\n- *Not provided (Optional for single-image analysis)*")
            
    report_lines.extend([
        "---",
        "## 2. Investigation Request",
        f"**User Query:** \"{last_analysis['query']}\"",
        f"**Assessed Task:** {last_analysis['task']}",
        f"**System Confidence:** {last_analysis['confidence']}",
        "---",
        "## 3. Analytical Findings",
        last_analysis["answer"],
        "---",
        "## 4. Geospatial Data Statistics",
    ])
    
    stats = last_analysis.get("stats")
    if stats:
        if "optical_water_percentage" in stats:
            report_lines.append(f"- **Optical Water Extent:** {stats['optical_water_percentage']*100:.1f}% area")
        if "sar_water_percentage" in stats:
            report_lines.append(f"- **SAR Water Extent:** {stats['sar_water_percentage']*100:.1f}% area")
        if "flooded_area_km2" in stats:
            report_lines.append(f"- **Flooded Area (Expansion):** **{stats['flooded_area_km2']:.3f} km²**")
            report_lines.append(f"- **Stable Water Area:** {stats['stable_area_km2']:.3f} km²")
            report_lines.append(f"- **Receded Area:** {stats['receded_area_km2']:.3f} km²")
            
        if "landcover_intersection" in stats:
            report_lines.append("\n### Affected Land Cover Intersection:")
            for cat, data in stats["landcover_intersection"].items():
                report_lines.append(f"  - **{cat}:** {data['area_affected_km2']:.3f} km² ({data['pixels_affected']} pixels)")
    else:
        report_lines.append("*No numerical geospatial statistics were calculated for this query.*")
        
    report_lines.extend([
        "---",
        "## 5. Limitations & Caveats",
    ])
    
    if last_analysis.get("limitations"):
        for lim in last_analysis["limitations"]:
            report_lines.append(f"- *{lim}*")
    else:
        report_lines.append("- *No special analysis limitations flagged.*")
        
    report_lines.extend([
        "---",
        "## 6. System Execution Provenance",
        "This analysis was processed in an auditable workflow by these components:",
    ])
    
    for model in registry.list_models():
        report_lines.append(f"- **{model['name']}** (v{model['version']}) - {model['description']}")
        
    report_content = "\n".join(report_lines)
    
    session_dir = os.path.join(DATA_DIR, session_id)
    report_path = os.path.join(session_dir, "investigation_report.md")
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    return FileResponse(
        path=report_path, 
        filename=f"satquery_report_{session_id}.md", 
        media_type="text/markdown"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
