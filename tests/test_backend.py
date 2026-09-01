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
# DIRECT QUERY ROUTER & SEMANTIC INTENT TESTS (ENGLISH & HINDI)
# -------------------------------------------------------------

@pytest.fixture
def agent():
    registry = ModelRegistry()
    return OrchestratorAgent(model_registry=registry)

def test_english_query_routing_variations(agent):
    """Verify all English description queries route strictly to SCENE_DESCRIPTION."""
    config = {"configuration": "SINGLE_IMAGE", "modalities": ["OPTICAL"]}
    
    queries = [
        "describe the picture",
        "describe the image",
        "describe this image",
        "describe this picture",
        "describe this scene",
        "what does this image show?",
        "what is visible here?",
        "what can you see?",
        "give me a description of the image",
        "summarize this image",
        "give me a scene description",
        "what is shown in this picture?"
    ]
    for q in queries:
        routed = agent.route_query_fallback(q, config)
        assert routed["task"] == "SCENE_DESCRIPTION", f"Failed for query: '{q}' -> got {routed['task']}"

def test_hindi_query_routing_variations(agent):
    """Verify all Hindi queries route accurately to their respective semantic tasks."""
    config = {"configuration": "SINGLE_IMAGE", "modalities": ["OPTICAL"]}
    
    assert agent.route_query_fallback("इस तस्वीर का वर्णन करो", config)["task"] == "SCENE_DESCRIPTION"
    assert agent.route_query_fallback("इस चित्र का विवरण दें", config)["task"] == "SCENE_DESCRIPTION"
    assert agent.route_query_fallback("इस चित्र में क्या दिखाई दे रहा है?", config)["task"] == "SCENE_DESCRIPTION"
    assert agent.route_query_fallback("इस चित्र में पानी है क्या?", config)["task"] == "VQA"
    assert agent.route_query_fallback("पानी को हाइलाइट करो", config)["task"] == "GROUNDING"
    assert agent.route_query_fallback("इन दोनों तस्वीरों में क्या बदलाव हुआ?", config)["task"] == "CHANGE_ANALYSIS"
    assert agent.route_query_fallback("ऑप्टिकल और SAR तस्वीरों का एक साथ विश्लेषण करो", config)["task"] == "OPTICAL_SAR"
    assert agent.route_query_fallback("कितनी कृषि भूमि प्रभावित हुई?", config)["task"] == "LAND_COVER_IMPACT"
    assert agent.route_query_fallback("नक्शे पर अलग-अलग रंग के डिब्बे क्या हैं?", config)["task"] == "MAP_INTERPRETATION"

# -------------------------------------------------------------
# WORKFLOW EXECUTION & MULTILINGUAL EQUIVALENCE TESTS
# -------------------------------------------------------------

def test_english_scene_description_workflow(agent):
    """Verify English 'describe the picture' outputs structured English description."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("describe the picture", session, language="en")
    assert res["task_classified"] == "SCENE_DESCRIPTION"
    assert "**Answer**" in res["answer"]
    assert "**Evidence**" in res["answer"]
    assert "**Confidence**" in res["answer"]
    assert "**Execution Summary**" in res["answer"]
    assert "Task: Single-Image Scene Description" in res["answer"]
    assert "EuroSAT ResNet18 Scene Classifier" in res["answer"]

def test_hindi_scene_description_workflow(agent):
    """Verify Hindi 'इस तस्वीर का वर्णन करो' outputs structured Hindi description with identical numerical evidence."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res_en = agent.execute_workflow("describe the image", session, language="en")
    res_hi = agent.execute_workflow("इस तस्वीर का वर्णन करो", session, language="hi")
    
    assert res_hi["task_classified"] == "SCENE_DESCRIPTION"
    assert "**उत्तर**" in res_hi["answer"]
    assert "**साक्ष्य**" in res_hi["answer"]
    assert "**विश्वसनीयता**" in res_hi["answer"]
    assert "**निष्पादन सारांश**" in res_hi["answer"]
    assert "कार्य: Single-Image Scene Description" in res_hi["answer"]
    assert "EuroSAT ResNet18 Scene Classifier" in res_hi["answer"]
    
    # Numerical evidence equivalence
    assert res_en["spatial_statistics"]["optical_water_percentage"] == res_hi["spatial_statistics"]["optical_water_percentage"]
    assert res_en["spatial_statistics"]["scene_landcover_distribution"] == res_hi["spatial_statistics"]["scene_landcover_distribution"]

def test_hindi_water_vqa_workflow(agent):
    """Verify Hindi 'इस चित्र में पानी है क्या?' outputs structured Hindi VQA response."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("इस चित्र में पानी है क्या?", session, language="hi")
    assert res["task_classified"] == "VQA"
    assert "**उत्तर**" in res["answer"]
    assert "जल क्षेत्र का पता चला है" in res["answer"]
    assert "**साक्ष्य**" in res["answer"]
    assert "**विश्वसनीयता**" in res["answer"]
    assert "**निष्पादन सारांश**" in res["answer"]
    assert "कार्य: Single-Image VQA" in res["answer"]
    assert "Optical Water Detector" in res["answer"]
    assert "optical_water_percentage" in res["spatial_statistics"]

def test_hindi_grounding_workflow(agent):
    """Verify Hindi 'पानी को हाइलाइट करो' outputs structured Hindi Grounding response."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("पानी को हाइलाइट करो", session, language="hi")
    assert res["task_classified"] == "GROUNDING"
    assert "**उत्तर**" in res["answer"]
    assert "हाइलाइट" in res["answer"]
    assert "कार्य: Text-Guided Region Grounding" in res["answer"]
    assert "optical_water_mask" in res["visual_evidence"]

def test_hindi_change_analysis_workflow(agent):
    """Verify Hindi 'इन दोनों तस्वीरों में क्या बदलाव हुआ?' outputs structured Hindi change metrics."""
    opt_pre = os.path.join(DEMO_DIR, "optical_pre.tif")
    opt_post = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_pre),
        "image_2": validate_tiff(opt_post),
        "active_image_1_path": opt_pre,
        "active_image_2_path": opt_post
    }
    
    res_en = agent.execute_workflow("What changed between these two dates?", session, language="en")
    res_hi = agent.execute_workflow("इन दोनों तस्वीरों में क्या बदलाव हुआ?", session, language="hi")
    
    assert res_hi["task_classified"] == "CHANGE_ANALYSIS"
    assert "**उत्तर**" in res_hi["answer"]
    assert "नया जलमग्न क्षेत्र:" in res_hi["answer"]
    assert "कार्य: Bi-temporal Change Analysis" in res_hi["answer"]
    assert "Change Analyzer" in res_hi["answer"]
    
    # Exact numerical equivalence
    assert res_en["spatial_statistics"]["flooded_area_km2"] == res_hi["spatial_statistics"]["flooded_area_km2"]

def test_hindi_optical_sar_workflow(agent):
    """Verify Hindi 'ऑप्टिकल और SAR तस्वीरों का एक साथ विश्लेषण करो' outputs structured Hindi fusion report."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": validate_tiff(sar_path),
        "active_image_1_path": opt_path,
        "active_image_2_path": sar_path
    }
    
    res = agent.execute_workflow("ऑप्टिकल और SAR तस्वीरों का एक साथ विश्लेषण करो", session, language="hi")
    assert res["task_classified"] == "OPTICAL_SAR"
    assert "**उत्तर**" in res["answer"]
    assert "ऑप्टिकल और SAR दोनों अवलोकन" in res["answer"]
    assert "कार्य: Cross-Modal Optical–SAR Analysis" in res["answer"]
    assert "SAR Backscatter" in res["evidence_fusion"]["modalities_used"]

def test_hindi_missing_temporal_fallback(agent):
    """Verify Hindi temporal query with single image returns polite Hindi guidance."""
    opt_path = os.path.join(DEMO_DIR, "optical_post.tif")
    session = {
        "image_1": validate_tiff(opt_path),
        "image_2": None,
        "active_image_1_path": opt_path,
        "active_image_2_path": None
    }
    
    res = agent.execute_workflow("इन दोनों तस्वीरों में क्या बदलाव हुआ?", session, language="hi")
    assert res["confidence"] == "अपर्याप्त साक्ष्य"
    assert "परिवर्तन विश्लेषण के लिए" in res["answer"]
    assert "छवि 2 में दूसरा चित्र अपलोड करें" in res["answer"]
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

def test_rgb_water_detection_precision():
    """Verify 3-band RGB optical water detector correctly rejects green agricultural fields."""
    user_img = os.path.join(BASE_DIR, "data", "session_43ca4eec-3e23-4ce7-993c-c5bf6f16a689", "image_1_d33c0f.tif")
    if os.path.exists(user_img):
        res = detect_water_optical(user_img)
        assert res["method"].startswith("Visible Water Index")
        # Ensure green farmland is not falsely classified as >50% water (previously 89.9%)
        assert res["water_percentage"] < 0.20, f"Expected <20% water on farmland scene, got {res['water_percentage']:.1%}"

