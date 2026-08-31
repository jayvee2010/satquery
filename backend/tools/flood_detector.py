import numpy as np
import rasterio
import os

def detect_water_optical(filepath: str, threshold: float = 0.05, green_band: int = 2, nir_band: int = 4) -> dict:
    """
    Detects water in optical imagery using the Normalized Difference Water Index (NDWI).
    NDWI = (Green - NIR) / (Green + NIR)
    
    If the image does not have 4 bands, it falls back to a color-based green-red difference index.
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
                if src.count in [12, 13]: # Sentinel-2 standard
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
                method = "NDWI (Normalized Difference Water Index)"
            else:
                # Fallback for RGB (3 bands)
                # NDWI is not possible without NIR. We use a modified Green-Red ratio.
                # Normalized Green-Red Difference Index (NGRDI) = (Green - Red) / (Green + Red)
                red = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32)
                
                denominator = green + red
                denominator[denominator == 0] = 1e-5
                ngrdi = (green - red) / denominator
                
                # Water in RGB usually has slightly higher green and lower red than land,
                # but it's less reliable.
                # Let's also check absolute brightness (water is dark)
                blue = src.read(3).astype(np.float32)
                mean_brightness = (red + green + blue) / 3.0
                
                # Custom RGB water mask: green > red and mean_brightness is low
                water_mask = ((ngrdi > 0.05) & (mean_brightness < 100)).astype(np.uint8)
                water_pct = float(np.sum(water_mask) / (width * height))
                
                limitations = ["Optical image lacks NIR band. Falling back to NGRDI (color difference index) which has lower reliability and is prone to vegetation/shadow confusion."]
                method = "NGRDI (Green-Red Difference Index with brightness threshold)"
                ndwi = ngrdi # for visualization compatibility
                
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
