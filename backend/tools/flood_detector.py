import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from scipy import ndimage
import os

def clean_water_mask(raw_mask: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
    """
    Applies spatial morphological post-processing to eliminate salt-and-pepper noise,
    isolated shadow pixels, and radar speckle while preserving genuine contiguous water bodies.
    1. 3x3 Median filtering to remove isolated 1-2 pixel spikes.
    2. Connected component labeling to discard tiny isolated noise clusters (< min_cluster_size).
    3. Binary hole filling for continuous water bodies.
    """
    if np.sum(raw_mask) == 0:
        return raw_mask
        
    # 1. 3x3 Median filter for noise suppression
    filtered = ndimage.median_filter(raw_mask, size=3)
    
    # 2. Connected component labeling (Minimum Mapping Unit)
    labeled_array, num_features = ndimage.label(filtered)
    if num_features > 0:
        sizes = ndimage.sum(filtered, labeled_array, range(1, num_features + 1))
        # Keep clusters >= min_cluster_size
        keep_labels = np.where(sizes >= min_cluster_size)[0] + 1
        cleaned = np.isin(labeled_array, keep_labels).astype(np.uint8)
    else:
        cleaned = filtered
        
    # 3. Fill small internal holes in large water bodies
    if np.sum(cleaned) > 50:
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        
    return cleaned

def detect_water_optical(filepath: str, threshold: float = 0.05, green_band: int = 2, nir_band: int = 4) -> dict:
    """
    Detects water in optical imagery:
    1. For Multispectral imagery (4+ bands with NIR):
       Calculates the standard McFeeters Normalized Difference Water Index (NDWI):
       NDWI = (Green - NIR) / (Green + NIR)
       Water has NDWI > threshold (water absorbs NIR while reflecting green).
       
    2. For 3-band RGB imagery (without NIR):
       Applies physical Visible Water Indexing (VWI = (Blue - Red) / (Blue + Red)) with
       scene-level brightness cutoff (water is darker than surrounding land) and chlorophyll
       excess-green rejection to prevent agricultural false alarms.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        with rasterio.open(filepath) as src:
            meta = src.meta.copy()
            width, height = src.width, src.height
            
            # Auto-detect Green and NIR band indices
            g_idx = green_band
            nir_idx = nir_band
            
            descriptions = [d.lower() if d else "" for d in src.descriptions] if src.descriptions else []
            found_by_name = False
            for idx, desc in enumerate(descriptions, start=1):
                if any(k in desc for k in ["nir", "near", "infra", "b08", "b8"]):
                    nir_idx = idx
                    found_by_name = True
                elif any(k in desc for k in ["green", "b03", "b3"]):
                    g_idx = idx
                    found_by_name = True
                    
            if not found_by_name:
                if src.count in [12, 13]: # Sentinel-2 standard 12/13 bands
                    g_idx = 3
                    nir_idx = 8
                elif src.count >= 4:
                    g_idx = 2
                    nir_idx = 4
                    
            if src.count >= 4:
                # Ensure band indices are within bounds
                g_idx = min(g_idx, src.count)
                nir_idx = min(nir_idx, src.count)
                
                green = src.read(g_idx).astype(np.float32)
                nir = src.read(nir_idx).astype(np.float32)
                
                # Avoid division by zero
                denominator = green + nir
                denominator[denominator == 0] = 1e-5
                ndwi = (green - nir) / denominator
                
                # Thresholding and shadow guardrail
                raw_mask = (ndwi > threshold).astype(np.uint8)
                water_mask = clean_water_mask(raw_mask, min_cluster_size=8)
                water_pct = float(np.sum(water_mask) / (width * height))
                
                limitations = []
                method = "NDWI (Normalized Difference Water Index with NIR)"
            else:
                # 3-Band RGB optical imagery (without NIR)
                r_idx, g_idx, b_idx = 1, 2, 3
                if src.colorinterp:
                    for idx, ci in enumerate(src.colorinterp, start=1):
                        if ci == ColorInterp.red: r_idx = idx
                        elif ci == ColorInterp.green: g_idx = idx
                        elif ci == ColorInterp.blue: b_idx = idx
                elif src.descriptions:
                    for idx, desc in enumerate(src.descriptions, start=1):
                        d = desc.lower() if desc else ""
                        if "red" in d or "b04" in d or "b4" in d: r_idx = idx
                        elif "green" in d or "b03" in d or "b3" in d: g_idx = idx
                        elif "blue" in d or "b02" in d or "b2" in d: b_idx = idx
                        
                red = src.read(r_idx).astype(np.float32)
                green = src.read(g_idx).astype(np.float32)
                blue = src.read(b_idx).astype(np.float32)
                
                v = (red + green + blue) / 3.0
                
                # 1. Vegetation Discrimination (Chlorophyll Green Peak):
                # Crops and grass reflect high green: G > B * 1.05 and G > R * 1.15
                is_vegetation = (green > blue * 1.05) & (green > red * 1.15)
                
                # 2. Soil / Urban Discrimination: Red is dominant or high brightness
                is_soil_urban = (red >= green * 0.95) | (red >= blue * 0.95) | (v > 110.0)
                
                # 3. Water Absorption & Contrast Criteria:
                # - Water absorbs visible light: V is low
                # - Blue is high relative to Red: (B - R) / (B + R) > 0.15 and B > R * 1.15
                # - Green does not dominate over Blue: G < B * 1.1
                # - Strictly not vegetation and not soil/urban
                # - Non-zero signal (V >= 10)
                p15_v = np.percentile(v, 15)
                water_cutoff_v = min(56.0, max(45.0, p15_v))
                
                denom_br = blue + red
                denom_br[denom_br == 0] = 1e-5
                vwi = (blue - red) / denom_br
                
                raw_mask = (
                    (v <= water_cutoff_v) & 
                    (vwi > 0.15) & 
                    (blue > red * 1.15) & 
                    (green < blue * 1.1) & 
                    (~is_vegetation) & 
                    (~is_soil_urban) & 
                    (v >= 10.0)
                ).astype(np.uint8)
                
                # Morphological noise cleaning (3x3 median filter + MMU area filter)
                water_mask = clean_water_mask(raw_mask, min_cluster_size=10)
                water_pct = float(np.sum(water_mask) / (width * height))
                limitations = ["Optical image lacks NIR band. Applied Visible Water Index (VWI) with chlorophyll rejection and spatial morphological cleaning."]
                method = "Visible Water Index (VWI RGB visible band analysis)"
                ndwi = vwi
                
            return {
                "method": method,
                "water_mask": water_mask.tolist(), # nested list for API JSON response
                "water_percentage": water_pct,
                "ndwi_grid": ndwi.tolist(),
                "threshold": threshold,
                "limitations": limitations
            }
            
    except Exception as e:
        raise ValueError(f"Error calculating optical water mask: {str(e)}")

def detect_water_sar(filepath: str, threshold_db: float = -18.0, sar_band: int = 1) -> dict:
    """
    Detects water in SAR (Synthetic Aperture Radar) imagery.
    Water has low backscatter (smooth surface, specular reflection).
    Uses a threshold on polarization backscatter (typically VH or VV in dB).
    Applies morphological filtering to eliminate radar speckle noise.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        with rasterio.open(filepath) as src:
            width, height = src.width, src.height
            band_data = src.read(min(sar_band, src.count)).astype(np.float32)
            
            is_db = np.any(band_data < 0)
            db_data = band_data
            limitations = []
            if not is_db:
                safe_val = np.clip(band_data, 1e-5, None)
                if np.max(band_data) > 1.0:
                    safe_val = safe_val / np.max(band_data)
                db_data = 20 * np.log10(safe_val)
                limitations.append("SAR backscatter was converted from raw amplitude to relative decibels (dB) using log scale normalization.")
            
            # Thresholding + Morphological speckle filtering
            raw_mask = (db_data < threshold_db).astype(np.uint8)
            water_mask = clean_water_mask(raw_mask, min_cluster_size=8)
            water_pct = float(np.sum(water_mask) / (width * height))
            
            return {
                "method": "SAR Low-Backscatter Thresholding",
                "water_mask": water_mask.tolist(),
                "water_percentage": water_pct,
                "db_grid": db_data.tolist(),
                "threshold_db": threshold_db,
                "limitations": limitations
            }
            
    except Exception as e:
        raise ValueError(f"Error calculating SAR water mask: {str(e)}")
