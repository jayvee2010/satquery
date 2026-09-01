import numpy as np
import rasterio
from rasterio.enums import ColorInterp
import os

def detect_water_optical(filepath: str, threshold: float = 0.05, green_band: int = 2, nir_band: int = 4) -> dict:
    """
    Detects water in optical imagery:
    1. For Multispectral imagery (4+ bands with NIR):
       Calculates the standard McFeeters Normalized Difference Water Index (NDWI):
       NDWI = (Green - NIR) / (Green + NIR)
       Water has NDWI > threshold (water absorbs NIR while reflecting green).
       
    2. For 3-band RGB imagery (without NIR):
       Applies Dark Object Subtraction (DOS1) to remove atmospheric Rayleigh scattering (blue haze pedestal),
       followed by the Normalized Visible Water Index (VWI / NDWI_RGB) with Excess Green (ExG) vegetation suppression
       and soil/urban rejection.
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
                
                # Thresholding to get binary mask (1 = water, 0 = non-water)
                water_mask = (ndwi > threshold).astype(np.uint8)
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
                        
                r_raw = src.read(r_idx).astype(np.float32)
                g_raw = src.read(g_idx).astype(np.float32)
                b_raw = src.read(b_idx).astype(np.float32)
                
                # Apply Dark Object Subtraction (DOS1) to remove atmospheric path radiance / Rayleigh blue haze
                p1_r = np.percentile(r_raw, 1)
                p1_g = np.percentile(g_raw, 1)
                p1_b = np.percentile(b_raw, 1)
                
                r = np.maximum(0.0, r_raw - p1_r)
                g = np.maximum(0.0, g_raw - p1_g)
                b = np.maximum(0.0, b_raw - p1_b)
                
                v_surf = (r + g + b) / 3.0
                
                # Excess Green (Vegetation discrimination on surface reflectance)
                exg = (2.0 * g) - r - b
                is_vegetation = (g > b * 1.05) & (exg > 1.0) & (g > r * 1.05)
                
                # Soil / Built-up discrimination (Red dominance)
                is_soil_urban = (r >= g * 1.05) | (r >= b * 1.05)
                
                # Visible Water Index (Blue - Red surface contrast)
                denom_br = b + r
                denom_br[denom_br == 0] = 1e-5
                vwi = (b - r) / denom_br
                
                # Water Body Identification:
                # - High Blue-to-Red contrast (VWI > 0.15 and Blue > Red * 1.15)
                # - Blue is dominant or co-dominant with Green (Blue >= Green * 0.95)
                # - Strictly NOT vegetation (excludes crops, grass, forests)
                # - Strictly NOT soil/urban/roads
                # - Surface brightness is within valid water range (1.0 <= v_surf <= 120.0)
                water_mask = (
                    (vwi > 0.15) & 
                    (b >= g * 0.95) & 
                    (b > r * 1.15) & 
                    (~is_vegetation) & 
                    (~is_soil_urban) & 
                    (v_surf >= 1.0) & 
                    (v_surf <= 120.0)
                ).astype(np.uint8)
                
                water_pct = float(np.sum(water_mask) / (width * height))
                limitations = ["Optical image lacks NIR band. Applied Visible Water Index (VWI) with DOS1 atmospheric haze correction to distinguish water from agricultural vegetation."]
                method = "Visible Water Index (with DOS1 Atmospheric Haze Correction)"
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
    
    If the image values are amplitudes (positive integers/floats > 0), they are converted to dB:
    dB = 20 * log10(amplitude) - calibration_constant
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        with rasterio.open(filepath) as src:
            width, height = src.width, src.height
            band_data = src.read(min(sar_band, src.count)).astype(np.float32)
            
            # Check if values are in dB or raw amplitudes
            # Decibel values are typically negative (e.g. -30 to -5)
            # Amplitudes are positive values (e.g. 0 to 1000+)
            is_db = np.any(band_data < 0)
            
            db_data = band_data
            limitations = []
            if not is_db:
                # Convert raw amplitude to scale resembling dB
                # Standard formula: 20 * log10(val)
                # Avoid log(0)
                safe_val = np.clip(band_data, 1e-5, None)
                # Normalize values first if they look like 16-bit integers
                if np.max(band_data) > 1.0:
                    safe_val = safe_val / np.max(band_data)
                db_data = 20 * np.log10(safe_val)
                limitations.append("SAR backscatter was converted from raw amplitude to relative decibels (dB) using log scale normalization.")
            
            # Thresholding (pixels with backscatter less than threshold are water)
            water_mask = (db_data < threshold_db).astype(np.uint8)
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
