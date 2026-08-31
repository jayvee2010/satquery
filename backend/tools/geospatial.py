import os
import rasterio
from rasterio.warp import transform_bounds
import numpy as np

def validate_tiff(filepath: str) -> dict:
    """
    Opens and validates a TIFF/GeoTIFF file, extracting key spatial metadata.
    Converts spatial bounds to WGS84 (latitude/longitude) for mapping compatibility.
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
                    # Transform bounds to EPSG:4326
                    min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, 'EPSG:4326', min_x, min_y, max_x, max_y)
                    latlon_bounds = {
                        "min_lat": min_lat,
                        "min_lon": min_lon,
                        "max_lat": max_lat,
                        "max_lon": max_lon,
                        "center": [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                    }
                except Exception as e:
                    # Fallback if transformation fails
                    latlon_bounds = {
                        "min_lat": None, "min_lon": None, "max_lat": None, "max_lon": None,
                        "center": None,
                        "warning": f"Could not transform CRS {crs} to EPSG:4326: {str(e)}"
                    }
            
            # Formulate structured metadata
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
                "pixel_size_y": abs(transform.e) # typical resolution
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
    
    # 1. Check CRS
    if meta1.get("crs") != meta2.get("crs"):
        compatible = False
        reasons.append(f"CRS mismatch: {meta1.get('crs')} vs {meta2.get('crs')}")
        
    # 2. Check Dimensions
    if meta1.get("width") != meta2.get("width") or meta1.get("height") != meta2.get("height"):
        # We allow slight mismatch if they cover the same area (needs resampling)
        reasons.append(f"Dimension mismatch: {meta1.get('width')}x{meta1.get('height')} vs {meta2.get('width')}x{meta2.get('height')}")
        
    # 3. Check Overlap
    b1 = meta1.get("bounds", {})
    b2 = meta2.get("bounds", {})
    
    # Calculate intersection of bounds
    x_left = max(b1.get("left", 0), b2.get("left", 0))
    y_bottom = max(b1.get("bottom", 0), b2.get("bottom", 0))
    x_right = min(b1.get("right", 0), b2.get("right", 0))
    y_top = min(b1.get("top", 0), b2.get("top", 0))
    
    if x_right <= x_left or y_top <= y_bottom:
        compatible = False
        reasons.append("No geographic overlap between images.")
    else:
        # Check how much they overlap relative to the first image area
        area1 = (b1["right"] - b1["left"]) * (b1["top"] - b1["bottom"])
        overlap_area = (x_right - x_left) * (y_top - y_bottom)
        overlap_pct = overlap_area / area1
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

def get_pixel_coords(meta: dict, lon: float, lat: float) -> tuple:
    """
    Converts WGS84 GPS coordinate (lon, lat) to pixel coordinates (x, y).
    """
    try:
        from pyproj import Transformer
        crs_str = meta.get("crs")
        if crs_str == "Unknown/None":
            return None
        
        # Setup projection transformer from WGS84 (4326) to image CRS
        transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x_proj, y_proj = transformer.transform(lon, lat)
        
        # Apply inverse affine transform to get pixel coordinates
        # transform parameters: a, b, c, d, e, f -> Affine(a, b, c, d, e, f)
        a, b, c, d, e, f = meta.get("transform")
        
        # Solve system:
        # x_proj = a*px + b*py + c
        # y_proj = d*px + e*py + f
        # Assuming b = d = 0 for standard non-rotated imagery
        if b == 0 and d == 0:
            px = int(round((x_proj - c) / a))
            py = int(round((y_proj - f) / e))
        else:
            # Full linear system solver
            # Affine formula: X = M * P
            # [x_proj - c]   [a  b]   [px]
            # [y_proj - f] = [d  e] * [py]
            det = a * e - b * d
            px = int(round((e * (x_proj - c) - b * (y_proj - f)) / det))
            py = int(round((-d * (x_proj - c) + a * (y_proj - f)) / det))
            
        # Check if coordinates fall inside image dimensions
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
        from rasterio.warp import transform as warp_transform
        xs, ys = warp_transform(from_crs, to_crs, [x], [y])
        return xs[0], ys[0]
    except Exception as e:
        print(f"Error projecting coordinates: {str(e)}")
        return x, y
