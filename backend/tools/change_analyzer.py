import numpy as np
import os

def analyze_change(pre_mask_list: list, post_mask_list: list, meta_pre: dict, meta_post: dict) -> dict:
    """
    Compares pre-event and post-event binary water masks to detect:
    - Flooded areas (new water: post & ~pre)
    - Receded areas (dry-up: pre & ~post)
    - Stable water (unchanged: pre & post)
    
    Calculates the exact area in square kilometers (km2).
    """
    pre_mask = np.array(pre_mask_list, dtype=np.uint8)
    post_mask = np.array(post_mask_list, dtype=np.uint8)
    
    if pre_mask.shape != post_mask.shape:
        raise ValueError("Pre-event and post-event masks must have the same dimensions for pixel-wise comparison.")
        
    # Calculate difference masks
    # 1 = water, 0 = land
    flooded = (post_mask == 1) & (pre_mask == 0)
    receded = (post_mask == 0) & (pre_mask == 1)
    stable = (post_mask == 1) & (pre_mask == 1)
    
    # Calculate areas
    width, height = meta_post["width"], meta_post["height"]
    pixel_size_x = meta_post.get("pixel_size_x", 10.0)
    pixel_size_y = meta_post.get("pixel_size_y", 10.0)
    crs_str = meta_post.get("crs", "Unknown")
    
    # Calculate pixel area in square kilometers
    is_degrees = "4326" in crs_str or "WGS 84" in crs_str or (abs(pixel_size_x) < 0.1)
    
    if is_degrees:
        # Approximate degree to meters conversion at average latitude
        # 1 degree lat approx = 111.12 km
        # 1 degree lon approx = 111.12 km * cos(lat)
        latlon_bounds = meta_post.get("latlon_bounds", {})
        center_lat = 0.0
        if latlon_bounds and latlon_bounds.get("center"):
            center_lat = latlon_bounds["center"][0]
        
        lat_km = 111.12
        lon_km = 111.12 * np.cos(np.radians(center_lat))
        
        pixel_width_km = abs(pixel_size_x) * lon_km
        pixel_height_km = abs(pixel_size_y) * lat_km
        pixel_area_km2 = pixel_width_km * pixel_height_km
    else:
        # Assuming metric projections (e.g. UTM meters)
        # pixel_size is in meters, convert area to km2
        pixel_area_km2 = (abs(pixel_size_x) * abs(pixel_size_y)) / 1_000_000.0
        
    flooded_count = int(np.sum(flooded))
    receded_count = int(np.sum(receded))
    stable_count = int(np.sum(stable))
    
    flooded_area = flooded_count * pixel_area_km2
    receded_area = receded_count * pixel_area_km2
    stable_area = stable_count * pixel_area_km2
    
    return {
        "flooded_pixels": flooded_count,
        "receded_pixels": receded_count,
        "stable_pixels": stable_count,
        "flooded_area_km2": float(round(flooded_area, 4)),
        "receded_area_km2": float(round(receded_area, 4)),
        "stable_area_km2": float(round(stable_area, 4)),
        "flooded_mask": flooded.astype(np.uint8).tolist(),
        "receded_mask": receded.astype(np.uint8).tolist(),
        "stable_mask": stable.astype(np.uint8).tolist(),
        "pixel_area_km2": pixel_area_km2,
        "crs_used": crs_str
    }
