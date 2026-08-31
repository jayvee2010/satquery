import os
import numpy as np
import pytest
from backend.tools.geospatial import validate_tiff, are_compatible
from backend.tools.flood_detector import detect_water_optical, detect_water_sar
from backend.tools.change_analyzer import analyze_change
from backend.tools.landcover_intersect import intersect_flood_landcover
from backend.models.registry import ModelRegistry
from backend.agents.orchestrator import OrchestratorAgent

# Setup directories for testing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

def test_geospatial_validation():
    """Verify metadata extraction and spatial bounds of generated demo images."""
    opt_pre_path = os.path.join(DEMO_DIR, "optical_pre.tif")
    assert os.path.exists(opt_pre_path), "Demo optical_pre.tif must exist to run tests. Run scripts/generate_demo_data.py first."
    
    meta = validate_tiff(opt_pre_path)
    assert meta["is_geotiff"] is True
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["bands"] == 4
    assert meta["crs"] == "EPSG:32645"
    assert meta["pixel_size_x"] == 10.0
    assert meta["latlon_bounds"] is not None
    assert "center" in meta["latlon_bounds"]

def test_are_compatible():
    """Test co-registration validation."""
    meta1 = {
        "crs": "EPSG:32645",
        "width": 256,
        "height": 256,
        "bounds": {"left": 500000.0, "bottom": 2897440.0, "right": 502560.0, "top": 2900000.0}
    }
    # Exactly matching
    assert are_compatible(meta1, meta1)["compatible"] is True
    
    # Different CRS
    meta2 = meta1.copy()
    meta2["crs"] = "EPSG:4326"
    assert are_compatible(meta1, meta2)["compatible"] is False
    
    # No overlap
    meta3 = meta1.copy()
    meta3["bounds"] = {"left": 600000.0, "bottom": 2897440.0, "right": 602560.0, "top": 2900000.0}
    assert are_compatible(meta1, meta3)["compatible"] is False

def test_flood_detection_optical():
    """Verify NDWI calculation and thresholding returns correct dimensions and values."""
    opt_pre_path = os.path.join(DEMO_DIR, "optical_pre.tif")
    res = detect_water_optical(opt_pre_path, threshold=0.05)
    
    assert res["method"] == "NDWI (Normalized Difference Water Index)"
    assert len(res["water_mask"]) == 256
    assert len(res["water_mask"][0]) == 256
    assert 0.0 < res["water_percentage"] < 1.0
    
    # Check that pixels are binary (0 or 1)
    mask = np.array(res["water_mask"])
    assert np.all(np.isin(mask, [0, 1]))

def test_flood_detection_sar():
    """Verify SAR low-backscatter water extraction."""
    sar_path = os.path.join(DEMO_DIR, "sar_post.tif")
    res = detect_water_sar(sar_path, threshold_db=-18.0, sar_band=1)
    
    assert res["method"] == "SAR Low-Backscatter Thresholding"
    assert len(res["water_mask"]) == 256
    assert 0.0 < res["water_percentage"] < 1.0
    
    # Check threshold logic: low backscatter (< -18dB) is flagged as water (1)
    db = np.array(res["db_grid"])
    mask = np.array(res["water_mask"])
    water_indices = db < -18.0
    assert np.all(mask[water_indices] == 1)

def test_change_analyzer():
    """Verify bi-temporal change area and logical mask subtraction."""
    meta = {
        "width": 256,
        "height": 256,
        "pixel_size_x": 10.0,
        "pixel_size_y": 10.0,
        "crs": "EPSG:32645"
    }
    
    # Pre-mask: 2x2 grid where bottom row is water (1)
    pre = [[0, 0], [1, 1]]
    # Post-mask: bottom row is water + top right is now water (flooded)
    post = [[0, 1], [1, 1]]
    
    res = analyze_change(pre, post, meta, meta)
    
    # Expected flooded: top right pixel [[0, 1], [0, 0]]
    expected_flooded = [[0, 1], [0, 0]]
    assert res["flooded_mask"] == expected_flooded
    assert res["flooded_pixels"] == 1
    assert res["stable_pixels"] == 2
    assert res["receded_pixels"] == 0
    
    # Pixel area: 10m * 10m = 100 m2 = 0.0001 km2
    assert res["pixel_area_km2"] == 0.0001
    assert res["flooded_area_km2"] == 0.0001

def test_landcover_intersection():
    """Verify LULC grid intersection area coverage counts."""
    # 2x2 mask: top right flooded
    flood_mask = [[0, 1], [0, 0]]
    # 2x2 LULC: top right is AnnualCrop (index 0)
    lulc = [[1, 0], [7, 7]]
    
    # Pixel area: 0.0001 km2
    res = intersect_flood_landcover(flood_mask, lulc, 0.0001)
    
    assert res["total_pixels_affected"] == 1
    assert res["total_area_affected_km2"] == 0.0001
    # Check that category "Agriculture" is affected by 1 pixel (0.0001 km2)
    assert res["consolidated_statistics"]["Agriculture"]["pixels_affected"] == 1
    assert res["consolidated_statistics"]["Agriculture"]["area_affected_km2"] == 0.0001
    # Check that Forest/Veg is 0 pixels affected
    assert res["consolidated_statistics"]["Forest/Veg"]["pixels_affected"] == 0

def test_orchestrator_agent_routing():
    """Verify query classification and input requirements mapping."""
    registry = ModelRegistry()
    agent = OrchestratorAgent(model_registry=registry)
    
    # Test fallback routing
    session = {}
    r1 = agent.route_query("Is there water in the post image?", session)
    assert r1["task"] == "VQA"
    
    r2 = agent.route_query("Highlight the flooded region in the map", session)
    assert r2["task"] == "GROUNDING"
    
    r3 = agent.route_query("What changed between these two dates?", session)
    assert r3["task"] == "CHANGE_ANALYSIS"
    
    r4 = agent.route_query("How much agriculture was affected by the flood?", session)
    assert r4["task"] == "LAND_COVER_IMPACT"
    
    r5 = agent.route_query("Use optical and SAR together to verify", session)
    assert r5["task"] == "OPTICAL_SAR"
