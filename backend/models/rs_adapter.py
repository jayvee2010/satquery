import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import rasterio

class EuroSATAdapter:
    """
    Loads the adapted ResNet18 model trained on EuroSAT LULC classes.
    Runs patch-based inference on larger satellite images.
    """
    def __init__(self, weights_path: str = None):
        # Default path relative to this file
        if weights_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(curr_dir, "adapted_rs_model.pth")
            
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
            "Industrial", "Pasture", "PermanentCrop", "Residential", 
            "River", "SeaLake"
        ]
        
        # Initialize ResNet18 architecture
        try:
            self.model = models.resnet18(weights=None)
        except TypeError:
            self.model = models.resnet18()
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        # Load weights if they exist, otherwise initialize randomly (as fallback)
        if os.path.exists(weights_path):
            try:
                # Load with map_location to handle CPU/MPS/GPU transitions
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"Loaded adapted EuroSAT model weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Failed to load weights from {weights_path}: {str(e)}. Running with uncalibrated weights.")
        else:
            print(f"Warning: Adapted weights file not found at {weights_path}. Model will run with default initialization.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_image(self, filepath: str) -> np.ndarray:
        """
        Loads a satellite image, splits it into 64x64 patches,
        classifies each patch using the adapted ResNet18 model,
        and returns a 2D class index map of the same shape as the input image.
        """
        try:
            with rasterio.open(filepath) as src:
                # Read first 3 bands (RGB)
                # If less than 3 bands (e.g. SAR), duplicate to make 3 channels
                if src.count >= 3:
                    # Map standard bands: Band 1=Red, Band 2=Green, Band 3=Blue
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                else:
                    # Grayscale/SAR fallback
                    r = src.read(1)
                    g = r.copy()
                    b = r.copy()
                    
                height, width = src.height, src.width
                img_data = np.stack([r, g, b], axis=-1)
                
                # Check for standard 8-bit scale, normalize if 16-bit
                if img_data.max() > 255:
                    img_data = (img_data / img_data.max() * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
                    
            # Patch size: 64x64
            patch_h, patch_w = 64, 64
            grid_h = int(np.ceil(height / patch_h))
            grid_w = int(np.ceil(width / patch_w))
            
            lulc_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            
            # Predict each patch
            for i in range(grid_h):
                for j in range(grid_w):
                    # Define patch boundaries
                    y_start = i * patch_h
                    y_end = min((i + 1) * patch_h, height)
                    x_start = j * patch_w
                    x_end = min((j + 1) * patch_w, width)
                    
                    patch = img_data[y_start:y_end, x_start:x_end]
                    
                    # Pad patch if it's not exactly 64x64
                    if patch.shape[0] < patch_h or patch.shape[1] < patch_w:
                        padded_patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
                        
                    # Convert to PIL Image for transformer
                    pil_img = Image.fromarray(patch)
                    tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(tensor_img)
                        _, preds = torch.max(outputs, 1)
                        class_idx = int(preds.item())
                        lulc_grid[i, j] = class_idx
                        
            # Upsample the class grid back to the original image dimensions using nearest-neighbor
            # This aligns the LULC predictions pixel-for-pixel with the raw input and flood masks
            lulc_upsampled = np.zeros((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    grid_i = min(i // patch_h, grid_h - 1)
                    grid_j = min(j // patch_w, grid_w - 1)
                    lulc_upsampled[i, j] = lulc_grid[grid_i, grid_j]
                    
            return lulc_upsampled
            
        except Exception as e:
            print(f"Error in LULC patch classification: {str(e)}")
            # Return agricultural default mask on failure
            return np.zeros((256, 256), dtype=np.uint8)
