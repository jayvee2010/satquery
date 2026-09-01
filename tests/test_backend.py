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
    assert meta_sar["bands"] in [1, 2]

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
# MANDATORY 8 TEST CASES (SECTION 19)
# -------------------------------------------------------------

@pytest.fixture
def agent():
    registry = ModelRegistry()
    return OrchestratorAgent(model_registry=registry)

def test_mandatory_1_scene_description(agent):
    """TEST 1 — SCENE DESCRIPTION: 'describe the image' -> SCENE_DESCRIPTION"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("describe the image", session)
    assert res["task_classified"] == "SCENE_DESCRIPTION"
    assert "**Answer**" in res["answer"]
    assert "mixture of agricultural land" in res["answer"] or "scene contains" in res["answer"]
    assert "Task: Single-Image Scene Description" in res["answer"]
    assert "EuroSAT ResNet18 Scene Classifier" in res["answer"]
    # Verify it does NOT claim flood impact
    assert "Task: Flood Impact" not in res["answer"]
    assert "Newly inundated area:" not in res["answer"]
    assert "scene_landcover_distribution" in res["spatial_statistics"]

def test_mandatory_2_water_vqa(agent):
    """TEST 2 — WATER VQA: 'Is there water in this image?' -> VQA"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Is there water in this image?", session)
    assert res["task_classified"] == "VQA"
    assert "**Answer**" in res["answer"]
    assert "Water is detected in the analyzed scene" in res["answer"]
    assert "Task: Single-Image VQA" in res["answer"]
    assert "Optical Water Detector" in res["answer"]
    assert "optical_water_percentage" in res["spatial_statistics"]

def test_mandatory_3_grounding(agent):
    """TEST 3 — GROUNDING: 'Highlight the water body.' -> GROUNDING"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Highlight the water body.", session)
    assert res["task_classified"] == "GROUNDING"
    assert "**Answer**" in res["answer"]
    assert "highlighted on the map" in res["answer"]
    assert "Task: Text-Guided Region Grounding" in res["answer"]
    assert "optical_water_mask" in res["visual_evidence"]

def test_mandatory_4_map_explanation(agent):
    """TEST 4 — MAP EXPLANATION: 'What are the different coloured squares on the map?' -> MAP_INTERPRETATION"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("What are the different coloured squares on the map?", session)
    assert res["task_classified"] == "MAP_INTERPRETATION"
    assert "**Answer**" in res["answer"]
    assert "coloured squares on the map represent" in res["answer"]
    assert "Task: Map Layer Interpretation" in res["answer"]
    assert "Leaflet Geo-referenced Overlay Renderer" in res["answer"]
    assert "Blue / Cyan" in res["answer"]
    assert "Yellow / Green" in res["answer"]

def test_mandatory_5_change_analysis(agent):
    """TEST 5 — CHANGE ANALYSIS: 'What changed between these two dates?' -> CHANGE_ANALYSIS"""
    opt_pre = os.path.join(DEMO_DIR, "optical_pre.tif")
    opt_post = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_pre),
        "image_2": validate_tiff(opt_post),
        "active_image_1_path": opt_pre,
        "active_image_2_path": opt_post
    }
    
    res = agent.execute_workflow("What changed between these two dates?", session)
    assert res["task_classified"] == "CHANGE_ANALYSIS"
    assert "**Answer**" in res["answer"]
    assert "The later observation shows an increase in inundated area" in res["answer"]
    assert "Newly inundated area:" in res["answer"]
    assert "Task: Bi-temporal Change Analysis" in res["answer"]
    assert "flooded_mask" in res["visual_evidence"]

def test_mandatory_6_optical_sar(agent):
    """TEST 6 — OPTICAL + SAR: 'Use the optical and SAR images together to identify water.' -> OPTICAL_SAR"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": validate_tiff(sar_path),
        "active_image_1_path": opt_path,
        "active_image_2_path": sar_path
    }
    
    res = agent.execute_workflow("Use the optical and SAR images together to identify water.", session)
    assert res["task_classified"] == "OPTICAL_SAR"
    assert "**Answer**" in res["answer"]
    assert "Both optical and SAR observations provide evidence" in res["answer"]
    assert "Task: Cross-Modal Optical–SAR Analysis" in res["answer"]
    assert "Optical (NDWI)" in res["evidence_fusion"]["modalities_used"]
    assert "SAR Backscatter" in res["evidence_fusion"]["modalities_used"]

def test_mandatory_7_lulc_impact(agent):
    """TEST 7 — LULC IMPACT: 'How much agricultural land is affected?' -> LAND_COVER_IMPACT"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("How much agricultural land is affected?", session)
    assert res["task_classified"] == "LAND_COVER_IMPACT"
    assert "**Answer**" in res["answer"]
    assert "Task: Flood Impact / Land-Cover Analysis" in res["answer"]
    assert "EuroSAT ResNet18 Classifier" in res["answer"]
    assert "landcover_intersection" in res["spatial_statistics"]
    assert "Agriculture intersecting the flood mask:" in res["answer"]

def test_mandatory_8_missing_temporal_fallback(agent):
    """TEST 8 — MISSING TEMPORAL INPUT: 1 image + 'What changed between these two dates?' -> Fallback"""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("What changed between these two dates?", session)
    assert res["confidence"] == "Insufficient evidence"
    assert "Change analysis requires two spatially corresponding observations" in res["answer"]
    assert res["task_classified"] == "CHANGE_ANALYSIS"

def test_single_sar_vqa(agent):
    """Single SAR Image VQA query 'Is there water or flooding visible?'"""
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    session = {
        "image_1": validate_tiff(sar_path),
        "image_2": None,
        "active_image_1_path": sar_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("Is there water or flooding visible?", session)
    assert res["task_classified"] == "VQA"
    assert "**Answer**" in res["answer"]
    assert "SAR Water Detector" in res["answer"]
    assert "sar_water_percentage" in res["spatial_statistics"]

def test_incompatible_pair_fallback(agent):
    """Incompatible images are blocked with clear explanation."""
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
