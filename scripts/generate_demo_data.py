import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_demo():
    print("Generating scientifically valid simulated GeoTIFFs for SatQuery AI demo...")
    
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(base_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    
    # Image properties
    width, height = 256, 256
    pixel_size = 10.0 # 10 meters (Sentinel-2 resolution)
    
    # UTM Zone 45N (EPSG:32645) - Covers Assam / Brahmaputra River, India
    crs = "EPSG:32645"
    
    # Construct affine transform directly: Affine(a, b, c, d, e, f)
    # a=pixel width, b=rotation, c=upper-left x, d=rotation, e=pixel height (negative), f=upper-left y
    transform = rasterio.Affine(pixel_size, 0.0, 500000.0, 0.0, -pixel_size, 2900000.0)
    
    # Define features on the grid
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # 1. Base River Path (diagonal: Y = X + noise)
    # Define a river path
    river_center = X - 30
    river_width_pre = 15
    river_mask_pre = np.abs(Y - river_center) < river_width_pre
    
    # 2. Flooded River Path (expanded river width + agricultural flooding)
    river_width_post = 40
    river_mask_post = np.abs(Y - river_center) < river_width_post
    # Add a flood blowout on the right side in a specific region (simulating dike breach)
    breach = (X > 100) & (X < 180) & (Y > river_center) & (Y < river_center + 60)
    river_mask_post = river_mask_post | breach
    
    # 3. Land Cover Categories
    # We will generate a base land cover classification:
    # 0 = Agriculture (crop fields in grid patterns)
    # 1 = Forest (clumped regions)
    # 7 = Residential/Built-up (block clusters)
    # 8 = Water
    lulc = np.zeros((height, width), dtype=np.uint8)
    
    # Agriculture as default
    lulc[:] = 0 # AnnualCrop
    
    # Forest in top-left and bottom-left patches
    forest_mask = (X < 80) & (Y < 120)
    lulc[forest_mask] = 1 # Forest
    
    # Built-up residential areas in small grids
    residential_mask1 = (X > 180) & (Y > 180) # bottom-right town
    residential_mask2 = (X > 200) & (Y < 60)   # top-right village
    lulc[residential_mask1] = 7 # Residential
    lulc[residential_mask2] = 7 # Residential
    
    # Original river
    lulc[river_mask_pre] = 8 # River / Water
    
    # Define spectral profile mapping for pre-event Optical (B, G, R, NIR)
    # Background:
    # Water: B=110, G=120, R=80,  NIR=50
    # Agri:  B=70,  G=150, R=60,  NIR=220
    # Forest:B=50,  G=110, R=40,  NIR=250
    # Resi:  B=150, G=140, R=180, NIR=120
    
    def create_optical_image(water_mask, cloud_mask=None):
        opt = np.zeros((4, height, width), dtype=np.float32)
        
        # Apply agricultural defaults
        opt[0] = 70.0  # Blue
        opt[1] = 150.0 # Green
        opt[2] = 60.0  # Red
        opt[3] = 220.0 # NIR
        
        # Forest overlay
        opt[0][forest_mask] = 50.0
        opt[1][forest_mask] = 110.0
        opt[2][forest_mask] = 40.0
        opt[3][forest_mask] = 250.0
        
        # Residential overlay
        opt[0][residential_mask1] = 150.0
        opt[1][residential_mask1] = 140.0
        opt[2][residential_mask1] = 180.0
        opt[3][residential_mask1] = 120.0
        
        opt[0][residential_mask2] = 150.0
        opt[1][residential_mask2] = 140.0
        opt[2][residential_mask2] = 180.0
        opt[3][residential_mask2] = 120.0
        
        # Water overlay
        opt[0][water_mask] = 110.0
        opt[1][water_mask] = 120.0
        opt[2][water_mask] = 80.0
        opt[3][water_mask] = 50.0
        
        # Add noise
        for b in range(4):
            opt[b] += np.random.normal(0, 5, (height, width))
            
        # Add Cloud (very bright in all bands)
        if cloud_mask is not None:
            for b in range(4):
                opt[b][cloud_mask] = 250.0 + np.random.normal(0, 3, np.sum(cloud_mask))
                
        # Clip to valid 8-bit range
        opt = np.clip(opt, 0, 255).astype(np.uint8)
        return opt

    # Create cloud mask for post-event optical (top right corner circle)
    cloud_center_x, cloud_center_y, cloud_radius = 210, 80, 45
    cloud_mask = ((X - cloud_center_x)**2 + (Y - cloud_center_y)**2) < cloud_radius**2
    
    # 4. Generate Optical Pre and Post images
    opt_pre = create_optical_image(river_mask_pre)
    opt_post = create_optical_image(river_mask_post, cloud_mask)
    
    # 5. Generate SAR Post image (VV and VH bands)
    # VV: Water ~ -22 dB, Agri ~ -12 dB, Forest ~ -8 dB, Residential ~ -6 dB
    # VH: Water ~ -25 dB, Agri ~ -16 dB, Forest ~ -11 dB, Residential ~ -10 dB
    # SAR penetrates clouds, so cloud_mask is ignored!
    sar_post = np.zeros((2, height, width), dtype=np.float32)
    
    # Agriculture default
    sar_post[0] = -12.0 # VV
    sar_post[1] = -16.0 # VH
    
    # Forest
    sar_post[0][forest_mask] = -8.0
    sar_post[1][forest_mask] = -11.0
    
    # Residential
    sar_post[0][residential_mask1] = -6.0
    sar_post[1][residential_mask1] = -10.0
    sar_post[0][residential_mask2] = -6.0
    sar_post[1][residential_mask2] = -10.0
    
    # Flooded water mask (includes river post)
    sar_post[0][river_mask_post] = -22.0
    sar_post[1][river_mask_post] = -25.0
    
    # Add speckle noise (standard in SAR)
    sar_post[0] += np.random.normal(0, 1.5, (height, width))
    sar_post[1] += np.random.normal(0, 2.0, (height, width))
    
    # Write files using rasterio
    def write_geotiff(path, data, crs, transform):
        count = data.shape[0]
        dtype = data.dtype
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(data)
        print(f"Saved: {path} ({count} bands, CRS={crs}, {width}x{height})")
        
    write_geotiff(os.path.join(demo_dir, "optical_pre.tif"), opt_pre, crs, transform)
    write_geotiff(os.path.join(demo_dir, "optical_post.tif"), opt_post, crs, transform)
    write_geotiff(os.path.join(demo_dir, "sar_post.tif"), sar_post, crs, transform)
    
    # Save the LULC grid as a numpy file for the backend model mock registry to load
    # representing our "adapted LULC output" for the pre-event
    np.save(os.path.join(demo_dir, "demo_lulc.npy"), lulc)
    print("Demo data generation complete!")

if __name__ == "__main__":
    generate_demo()
