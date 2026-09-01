import os
import rasterio
from rasterio.warp import transform_bounds, transform as warp_transform
import numpy as np
import re

def detect_modality(src, filepath: str) -> str:
    """
    Infers the sensor modality of a raster from band counts, descriptions, tags, and filename.
    Returns: 'OPTICAL', 'SAR', 'MULTISPECTRAL', or 'UNKNOWN'.
    """
    fn = os.path.basename(filepath).lower()
    descriptions = [str(d).lower() if d else "" for d in (src.descriptions or [])]
    tags = {str(k).lower(): str(v).lower() for k, v in src.tags().items()}
    tag_blob = " ".join(tags.values()) + " " + " ".join(descriptions) + " " + fn

    # 1. Check for SAR keywords
    sar_keywords = ["sar", "sentinel-1", "s1", "risat", "radar", "backscatter", "sigma0", "gamma0", "vv", "vh", "hh", "hv"]
    if any(k in tag_blob for k in sar_keywords):
        return "SAR"
        
    # 2. Check band count
    if src.count == 1:
        # Check if dtype is float or single-channel radar intensity
        return "SAR" if any(k in fn for k in ["sar", "radar", "s1", "vh", "vv"]) else "OPTICAL"
    elif src.count == 3:
        return "OPTICAL"
    elif src.count >= 4:
        return "MULTISPECTRAL"
        
    return "UNKNOWN"

def extract_acquisition_date(src, filepath: str) -> str:
    """
    Extracts acquisition timestamp or date from raster metadata or filename if available.
    """
    fn = os.path.basename(filepath).lower()
    
    # Check tags
    tags = {str(k).lower(): str(v) for k, v in src.tags().items()}
    for key in ["datetime", "date", "acquisition_date", "time_coverage_start"]:
        if key in tags:
            return tags[key]
            
    # Check filename for YYYYMMDD pattern
    match = re.search(r'(20\d{2}[01]\d[0-3]\d)', fn)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        
    if "pre" in fn or "before" in fn:
        return "Pre-Event"
    elif "post" in fn or "after" in fn:
        return "Post-Event"
        
    return "Unknown Date"

def validate_tiff(filepath: str) -> dict:
    """
    Opens and validates a TIFF/GeoTIFF file, extracting key spatial metadata,
    sensor modality, and coordinate transformations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with rasterio.open(filepath) as src:
            meta = src.meta.copy()
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            
            # Calculate bounds manually to bypass python 3.12 affine cached_property bug
            left = transform.c
            right = transform.c + width * transform.a
            top = transform.f
            bottom = transform.f + height * transform.e
            
            min_x = min(left, right)
            max_x = max(left, right)
            min_y = min(bottom, top)
            max_y = max(bottom, top)
            
            # Determine bounds in EPSG:4326 (WGS84) for map visualization
            latlon_bounds = None
            if crs is not None:
                try:
                    min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, 'EPSG:4326', min_x, min_y, max_x, max_y)
                    latlon_bounds = {
                        "min_lat": min_lat,
                        "min_lon": min_lon,
                        "max_lat": max_lat,
                        "max_lon": max_lon,
                        "center": [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                    }
                except Exception as e:
                    latlon_bounds = {
                        "min_lat": None, "min_lon": None, "max_lat": None, "max_lon": None,
                        "center": None,
                        "warning": f"Could not transform CRS {crs} to EPSG:4326: {str(e)}"
                    }
            
            # Detect modality & band semantics
            modality = detect_modality(src, filepath)
            acquisition_date = extract_acquisition_date(src, filepath)
            band_descriptions = [str(d) if d else f"Band {i+1}" for i, d in enumerate(src.descriptions or [])]
            
            crs_str = str(crs) if crs else "Unknown/None"
            is_geotiff = crs is not None and bool(crs.to_string())
            
            return {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "is_geotiff": is_geotiff,
                "width": meta["width"],
                "height": meta["height"],
                "bands": meta["count"],
                "dtype": str(meta["dtype"]),
                "crs": crs_str,
                "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                "bounds": {
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top
                },
                "latlon_bounds": latlon_bounds,
                "pixel_size_x": transform.a,
                "pixel_size_y": abs(transform.e),
                "modality": modality,
                "acquisition_date": acquisition_date,
                "band_descriptions": band_descriptions
            }
            
    except Exception as e:
        raise ValueError(f"Invalid TIFF file or error reading metadata: {str(e)}")

def are_compatible(meta1: dict, meta2: dict, threshold_pct: float = 0.05) -> dict:
    """
    Validates whether two images are spatially compatible (same CRS, overlapping extent).
    Returns a dict with compatibility flag and details/limitations.
    """
    reasons = []
    compatible = True
    
    if not meta1 or not meta2:
        return {"compatible": False, "reasons": ["Missing metadata for one or both images."], "limitations": ["Missing metadata"]}
    
    # 1. Check CRS
    if meta1.get("crs") != meta2.get("crs"):
        compatible = False
        reasons.append(f"CRS mismatch: {meta1.get('crs')} vs {meta2.get('crs')}")
        
    # 2. Check Dimensions
    if meta1.get("width") != meta2.get("width") or meta1.get("height") != meta2.get("height"):
        reasons.append(f"Dimension mismatch: {meta1.get('width')}x{meta1.get('height')} vs {meta2.get('width')}x{meta2.get('height')}")
        
    # 3. Check Overlap
    b1 = meta1.get("bounds", {})
    b2 = meta2.get("bounds", {})
    
    x_left = max(b1.get("left", 0), b2.get("left", 0))
    y_bottom = max(b1.get("bottom", 0), b2.get("bottom", 0))
    x_right = min(b1.get("right", 0), b2.get("right", 0))
    y_top = min(b1.get("top", 0), b2.get("top", 0))
    
    if x_right <= x_left or y_top <= y_bottom:
        compatible = False
        reasons.append("No geographic overlap between images.")
    else:
        area1 = (b1["right"] - b1["left"]) * (b1["top"] - b1["bottom"])
        overlap_area = (x_right - x_left) * (y_top - y_bottom)
        overlap_pct = overlap_area / (area1 if area1 != 0 else 1.0)
        if overlap_pct < 0.9:
            reasons.append(f"Geographic overlap is only {overlap_pct:.1%}. Recommend >90% for reliable comparison.")
            if overlap_pct < 0.5:
                compatible = False

    return {
        "compatible": compatible,
        "overlap_verified": compatible and len(reasons) == 0,
        "reasons": reasons,
        "limitations": reasons if not compatible or len(reasons) > 0 else []
    }

def classify_input_configuration(meta1: dict, meta2: dict = None) -> dict:
    """
    Classifies the active workspace into:
    - 'SINGLE_IMAGE'
    - 'OPTICAL_SAR_PAIR'
    - 'BI_TEMPORAL_PAIR'
    - 'INCOMPATIBLE_INPUT'
    - 'EMPTY'
    """
    if not meta1 and not meta2:
        return {
            "configuration": "EMPTY",
            "modalities": [],
            "spatially_aligned": False,
            "details": "No satellite images loaded."
        }
        
    if meta1 and not meta2:
        mod = meta1.get("modality", "UNKNOWN")
        return {
            "configuration": "SINGLE_IMAGE",
            "modalities": [mod],
            "spatially_aligned": True,
            "details": f"Single {mod} image ready for analysis."
        }
        
    if meta2 and not meta1:
        mod = meta2.get("modality", "UNKNOWN")
        return {
            "configuration": "SINGLE_IMAGE",
            "modalities": [mod],
            "spatially_aligned": True,
            "details": f"Single {mod} image ready for analysis."
        }
        
    # Both meta1 and meta2 are present
    comp = are_compatible(meta1, meta2)
    if not comp["compatible"]:
        return {
            "configuration": "INCOMPATIBLE_INPUT",
            "modalities": [meta1.get("modality", "UNKNOWN"), meta2.get("modality", "UNKNOWN")],
            "spatially_aligned": False,
            "details": "Spatial incompatibility: " + "; ".join(comp["reasons"])
        }
        
    mod1 = meta1.get("modality", "UNKNOWN")
    mod2 = meta2.get("modality", "UNKNOWN")
    
    # Check for Optical + SAR
    is_opt1 = mod1 in ["OPTICAL", "MULTISPECTRAL"]
    is_sar1 = mod1 == "SAR"
    is_opt2 = mod2 in ["OPTICAL", "MULTISPECTRAL"]
    is_sar2 = mod2 == "SAR"
    
    if (is_opt1 and is_sar2) or (is_sar1 and is_opt2):
        return {
            "configuration": "OPTICAL_SAR_PAIR",
            "modalities": [mod1, mod2],
            "spatially_aligned": True,
            "details": "Spatially co-registered Optical + SAR cross-modal pair."
        }
        
    # Both Optical or Both SAR -> Bi-Temporal
    return {
        "configuration": "BI_TEMPORAL_PAIR",
        "modalities": [mod1, mod2],
        "spatially_aligned": True,
        "details": f"Spatially aligned multi-temporal {mod1} pair."
    }

def get_pixel_coords(meta: dict, lon: float, lat: float) -> tuple:
    """
    Converts WGS84 GPS coordinate (lon, lat) to pixel coordinates (x, y).
    """
    try:
        from pyproj import Transformer
        crs_str = meta.get("crs")
        if crs_str == "Unknown/None":
            return None
        
        transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x_proj, y_proj = transformer.transform(lon, lat)
        
        a, b, c, d, e, f = meta.get("transform")
        if b == 0 and d == 0:
            px = int(round((x_proj - c) / a))
            py = int(round((y_proj - f) / e))
        else:
            det = a * e - b * d
            px = int(round((e * (x_proj - c) - b * (y_proj - f)) / det))
            py = int(round((-d * (x_proj - c) + a * (y_proj - f)) / det))
            
        if 0 <= px < meta["width"] and 0 <= py < meta["height"]:
            return px, py
        return None
    except Exception:
        return None

def project_point(from_crs: str, to_crs: str, x: float, y: float) -> tuple:
    """
    Projects a single coordinate point (x, y) from one CRS to another.
    Returns (x_proj, y_proj).
    """
    try:
        xs, ys = warp_transform(from_crs, to_crs, [x], [y])
        return xs[0], ys[0]
    except Exception as e:
        print(f"Error projecting coordinates: {str(e)}")
        return x, y
