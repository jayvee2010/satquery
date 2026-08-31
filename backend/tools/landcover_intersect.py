import numpy as np
import os
from backend.tools.geospatial import project_point

# EuroSAT Class Indices:
# 0: AnnualCrop
# 1: Forest
# 2: HerbaceousVegetation
# 3: Highway
# 4: Industrial
# 5: Pasture
# 6: PermanentCrop
# 7: Residential
# 8: River
# 9: SeaLake

EUROSAT_CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", 
    "Industrial", "Pasture", "Permanent Crop", "Residential", 
    "River", "Sea & Lake"
]

# Simplify EuroSAT classes into consolidated report categories
CATEGORY_MAPPING = {
    0: "Agriculture",      # AnnualCrop
    1: "Forest/Veg",        # Forest
    2: "Forest/Veg",        # HerbaceousVegetation
    3: "Built-up/Infra",    # Highway
    4: "Built-up/Infra",    # Industrial
    5: "Agriculture",      # Pasture
    6: "Agriculture",      # PermanentCrop
    7: "Built-up/Infra",    # Residential
    8: "Water Bodies",      # River
    9: "Water Bodies"       # SeaLake
}

def intersect_flood_landcover(flood_mask_list: list, lulc_grid_list: list, pixel_area_km2: float, meta: dict = None) -> dict:
    """
    Intersects the binary flood mask with the Land Use / Land Cover (LULC) grid.
    Calculates the affected area (in km2) for each EuroSAT class and consolidated category.
    Computes the spatial centroid (Lat/Lon) of the flooded pixels for each category.
    """
    flood_mask = np.array(flood_mask_list, dtype=np.uint8)
    lulc_grid = np.array(lulc_grid_list, dtype=np.int32)
    
    if flood_mask.shape != lulc_grid.shape:
        raise ValueError(f"Flood mask shape {flood_mask.shape} and LULC grid shape {lulc_grid.shape} must match.")
        
    # Get coordinates where flooding occurred (flood_mask == 1)
    flood_indices = flood_mask == 1
    flooded_lulc = lulc_grid[flood_indices]
    
    # Calculate counts and areas per raw EuroSAT class
    class_stats = {}
    for class_idx, class_name in enumerate(EUROSAT_CLASSES):
        count = int(np.sum(flooded_lulc == class_idx))
        area_km2 = count * pixel_area_km2
        class_stats[class_name] = {
            "class_index": class_idx,
            "pixels_affected": count,
            "area_affected_km2": float(round(area_km2, 4))
        }
        
    # Calculate stats and centroids per consolidated category
    categories = ["Agriculture", "Forest/Veg", "Built-up/Infra", "Water Bodies", "Other"]
    category_stats = {cat: {"pixels_affected": 0, "area_affected_km2": 0.0, "centroid": None} for cat in categories}
    
    for val in flooded_lulc:
        category = CATEGORY_MAPPING.get(val, "Other")
        category_stats[category]["pixels_affected"] += 1
        
    for cat in category_stats:
        category_stats[cat]["area_affected_km2"] = float(round(category_stats[cat]["pixels_affected"] * pixel_area_km2, 4))

    # Calculate centroids if metadata is provided
    if meta is not None and "transform" in meta and "crs" in meta:
        transform = meta["transform"]
        a, b, c, d, e, f = transform
        crs = meta["crs"]
        
        for cat in categories:
            matching_classes = [class_idx for class_idx, category in CATEGORY_MAPPING.items() if category == cat]
            matches = np.isin(lulc_grid, matching_classes) & (flood_mask == 1)
            rows, cols = np.where(matches)
            
            if len(rows) > 0:
                row_avg = float(np.mean(rows))
                col_avg = float(np.mean(cols))
                
                # Apply affine transform: X = a*col + b*row + c, Y = d*col + e*row + f
                x_proj = a * col_avg + b * row_avg + c
                y_proj = d * col_avg + e * row_avg + f
                
                # Project back to GPS coordinates (WGS84)
                lon, lat = project_point(crs, "EPSG:4326", x_proj, y_proj)
                
                category_stats[cat]["centroid"] = {
                    "latitude": float(round(lat, 5)),
                    "longitude": float(round(lon, 5)),
                    "easting": float(round(x_proj, 1)),
                    "northing": float(round(y_proj, 1))
                }
        
    # Total affected area
    total_pixels = int(np.sum(flood_mask))
    total_area_km2 = total_pixels * pixel_area_km2
    
    return {
        "total_pixels_affected": total_pixels,
        "total_area_affected_km2": float(round(total_area_km2, 4)),
        "consolidated_statistics": category_stats,
        "detailed_statistics": class_stats
    }
