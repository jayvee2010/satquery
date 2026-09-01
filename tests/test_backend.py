import os
import numpy as np
import pytest
from backend.tools.geospatial import validate_tiff, are_compatible, classify_input_configuration
from backend.tools.flood_detector import detect_water_optical, detect_water_sar
from backend.tools.change_analyzer import analyze_change
from backend.tools.landcover_intersect import intersect_flood_landcover
from backend.models.registry import ModelRegistry
from backend.agents.orchestrator import OrchestratorAgent

# Setup directories for testing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

# -------------------------------------------------------------
# GEOSPATIAL & TOOL LEVEL TESTS
# -------------------------------------------------------------

def test_geospatial_validation_and_modality():
    """Verify metadata extraction, modality detection, and spatial bounds."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    assert os.path.exists(opt_path) and os.path.exists(sar_path), "Demo TIFFs must exist. Run scripts/generate_demo_data.py."
    
    meta_opt = validate_tiff(opt_path)
    assert meta_opt["is_geotiff"] is True
    assert meta_opt["modality"] == "MULTISPECTRAL" # 4 bands
    assert meta_opt["crs"] == "EPSG:32645"
    assert meta_opt["latlon_bounds"] is not None
    
    meta_sar = validate_tiff(sar_path)
    assert meta_sar["is_geotiff"] is True
    assert meta_sar["modality"] == "SAR"
    assert meta_sar["bands"] in [1, 2] # Dual-polarization VH/VV has 2 bands

def test_input_configuration_classification():
    """Verify workspace input configuration states."""
    opt = validate_tiff(os.path.join(DEMO_DIR, "optical_post.tif"))
    sar = validate_tiff(os.path.join(DEMO_DIR, "sar_post.tif"))
    opt_pre = validate_tiff(os.path.join(DEMO_DIR, "optical_pre.tif"))
    
    # 1. Single Optical
    c1 = classify_input_configuration(opt, None)
    assert c1["configuration"] == "SINGLE_IMAGE"
    assert c1["modalities"] == ["MULTISPECTRAL"]
    
    # 2. Single SAR
    c2 = classify_input_configuration(sar, None)
    assert c2["configuration"] == "SINGLE_IMAGE"
    assert c2["modalities"] == ["SAR"]
    
    # 3. Cross-Modal Optical + SAR Pair
    c3 = classify_input_configuration(opt, sar)
    assert c3["configuration"] == "OPTICAL_SAR_PAIR"
    assert c3["spatially_aligned"] is True
    
    # 4. Bi-Temporal Pair (Optical Pre + Post)
    c4 = classify_input_configuration(opt_pre, opt)
    assert c4["configuration"] == "BI_TEMPORAL_PAIR"
    assert c4["spatially_aligned"] is True
    
    # 5. Incompatible Pair
    bad_opt = opt.copy()
    bad_opt["crs"] = "EPSG:4326"
    c5 = classify_input_configuration(opt, bad_opt)
    assert c5["configuration"] == "INCOMPATIBLE_INPUT"
    assert c5["spatially_aligned"] is False

# -------------------------------------------------------------
# 10 WORKFLOW COMPLIANCE TESTS (SECTION 29)
# -------------------------------------------------------------

@pytest.fixture
def agent():
    registry = ModelRegistry()
    return OrchestratorAgent(model_registry=registry)

def test_1_single_optical_vqa(agent):
    """Test 1: Single Optical Image VQA query 'Is there water in this image?'"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    meta_opt = validate_tiff(opt_path)
    
    session = {
        "image_1": meta_opt,
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Is there water in this image?", session)
    assert res["task_classified"] == "VQA"
    assert res["input_configuration"] == "SINGLE_IMAGE"
    assert "optical_water_percentage" in res["spatial_statistics"]
    assert res["spatial_statistics"]["optical_water_percentage"] > 0.0
    assert "Water" in res["answer"] or "water" in res["answer"]

def test_2_single_sar_vqa(agent):
    """Test 2: Single SAR Image VQA query 'Is there water or flooding visible?'"""
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    meta_sar = validate_tiff(sar_path)
    
    session = {
        "image_1": meta_sar,
        "image_2": None,
        "active_image_1_path": sar_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Is there water or flooding visible?", session)
    assert res["task_classified"] == "VQA"
    assert res["input_configuration"] == "SINGLE_IMAGE"
    assert "sar_water_percentage" in res["spatial_statistics"]
    assert res["spatial_statistics"]["sar_water_percentage"] > 0.0
    assert "SAR" in res["answer"] or "backscatter" in res["answer"]

def test_3_single_grounding(agent):
    """Test 3: Single Image Grounding query 'Highlight the water body'"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    meta_opt = validate_tiff(opt_path)
    
    session = {
        "image_1": meta_opt,
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Highlight the water body", session)
    assert res["task_classified"] == "GROUNDING"
    assert "optical_water_mask" in res["visual_evidence"]
    assert len(res["visual_evidence"]["optical_water_mask"]) == 256
    assert np.any(np.array(res["visual_evidence"]["optical_water_mask"]) == 1)

def test_4_single_captioning_and_landcover(agent):
    """Test 4: Single Image Captioning / Scene Description query"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    meta_opt = validate_tiff(opt_path)
    
    session = {
        "image_1": meta_opt,
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Describe the land cover and major objects", session)
    assert res["task_classified"] == "LAND_COVER_IMPACT"
    assert "lulc_grid" in res["visual_evidence"]
    assert len(res["visual_evidence"]["lulc_grid"]) == 256
    assert "landcover_intersection" in res["spatial_statistics"]

def test_5_optical_sar_cross_modal_fusion(agent):
    """Test 5: Cross-Modal Optical + SAR pair analysis query"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": validate_tiff(sar_path),
        "active_image_1_path": opt_path,
        "active_image_2_path": sar_path
    }
    
    res = agent.execute_workflow("Use both images together to identify flood-affected areas", session)
    assert res["task_classified"] == "OPTICAL_SAR"
    assert res["input_configuration"] == "OPTICAL_SAR_PAIR"
    assert "optical_water_mask" in res["visual_evidence"]
    assert "sar_water_mask" in res["visual_evidence"]
    assert "Optical (NDWI)" in res["evidence_fusion"]["modalities_used"]
    assert "SAR Backscatter" in res["evidence_fusion"]["modalities_used"]

def test_6_temporal_change_analysis(agent):
    """Test 6: Bi-Temporal pair query 'What changed between these two observations?'"""
    opt_pre = os.path.join(DEMO_DIR, "optical_pre.tif")
    opt_post = os.path.join(DEMO_DIR, "optical_post.tif")
    
    session = {
        "image_1": validate_tiff(opt_pre),
        "image_2": validate_tiff(opt_post),
        "active_image_1_path": opt_pre,
        "active_image_2_path": opt_post
    }
    
    res = agent.execute_workflow("What changed between these two observations?", session)
    assert res["task_classified"] == "CHANGE_ANALYSIS"
    assert res["input_configuration"] == "BI_TEMPORAL_PAIR"
    assert "flooded_mask" in res["visual_evidence"]
    assert res["spatial_statistics"]["flooded_area_km2"] > 0.0

def test_7_temporal_flood_increase(agent):
    """Test 7: Bi-Temporal query 'Where did flooding increase?'"""
    opt_pre = os.path.join(DEMO_DIR, "optical_pre.tif")
    opt_post = os.path.join(DEMO_DIR, "optical_post.tif")
    
    session = {
        "image_1": validate_tiff(opt_pre),
        "image_2": validate_tiff(opt_post),
        "active_image_1_path": opt_pre,
        "active_image_2_path": opt_post
    }
    
    res = agent.execute_workflow("Where did flooding increase after the event?", session)
    assert res["task_classified"] == "CHANGE_ANALYSIS"
    assert res["spatial_statistics"]["flooded_area_km2"] > 0.0
    assert "Newly Inundated Area" in res["answer"]

def test_8_missing_temporal_input_rejection(agent):
    """Test 8: Temporal query with only single image is gracefully rejected requesting 2nd image."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("What changed?", session)
    assert res["confidence"] == "Insufficient evidence"
    assert "Change analysis requires two spatially corresponding observations" in res["answer"]
    assert res["task_classified"] == "CHANGE_ANALYSIS"

def test_9_missing_sar_input_rejection(agent):
    """Test 9: Cross-modal query with only optical image is gracefully rejected requesting SAR."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Use optical and SAR together to verify", session)
    assert res["confidence"] == "Insufficient evidence"
    assert "requires both an Optical/Multispectral image and a corresponding SAR" in res["answer"]
    assert res["task_classified"] == "OPTICAL_SAR"

def test_10_incompatible_pair_rejection(agent):
    """Test 10: Incompatible images are blocked with clear explanation."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    meta_good = validate_tiff(opt_path)
    meta_bad = meta_good.copy()
    meta_bad["crs"] = "EPSG:4326" # CRS mismatch
    
    session = {
        "image_1": meta_good,
        "image_2": meta_bad,
        "active_image_1_path": opt_path,
        "active_image_2_path": opt_path
    }
    
    res = agent.execute_workflow("What changed?", session)
    assert res["confidence"] == "Insufficient evidence"
    assert "spatial incompatibility" in res["answer"]
