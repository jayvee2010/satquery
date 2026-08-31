import os
from backend.models.rs_adapter import EuroSATAdapter

class ModelRegistry:
    """
    Registry for models active in the SatQuery system.
    Handles lazy-loading and metadata queries.
    """
    def __init__(self, adapted_model_path: str = None):
        self.adapted_model_path = adapted_model_path
        self._lulc_model = None
        
        # Static registry metadata
        self.registry = {
            "adapted_lulc_classifier": {
                "name": "SatQuery-LULC-ResNet18-Adapter",
                "version": "1.0.0",
                "checkpoint": "adapted_rs_model.pth",
                "framework": "PyTorch 2.3",
                "adapted_dataset": "EuroSAT (10 classes)",
                "description": "Domain-adapted ResNet18 classifier fine-tuned on EuroSAT Sentinel-2 patches to classify Land Use/Land Cover (LULC)."
            },
            "ndwi_flood_detector": {
                "name": "Geospatial NDWI Analyzer",
                "version": "1.2.0",
                "framework": "Rasterio & NumPy",
                "description": "Geospatial tool that calculates Normalized Difference Water Index (NDWI) from green and NIR bands to extract water bodies."
            },
            "sar_flood_detector": {
                "name": "Geospatial SAR Backscatter Analyzer",
                "version": "1.1.0",
                "framework": "Rasterio & NumPy",
                "description": "Geospatial tool that thresholds Synthetic Aperture Radar (SAR) VV/VH backscatter values (in dB) to map surface water."
            },
            "change_detection_engine": {
                "name": "Bi-Temporal Change Analyzer",
                "version": "1.0.0",
                "framework": "NumPy Logic",
                "description": "Analyzes pixel-level expansion and recession between pre-event and post-event water masks."
            },
            "orchestrator_agent": {
                "name": "SatQuery Orchestrator (Gemini 3.5 Flash)",
                "version": "v1.5",
                "framework": "Google GenAI SDK",
                "description": "Cognitive agent responsible for query understanding, input validation, tool routing, and response synthesis."
            }
        }

    @property
    def lulc_model(self) -> EuroSATAdapter:
        """Lazy load the PyTorch EuroSAT model on first call."""
        if self._lulc_model is None:
            self._lulc_model = EuroSATAdapter(weights_path=self.adapted_model_path)
        return self._lulc_model

    def get_model_info(self, model_key: str) -> dict:
        """Returns details for a registered model/tool."""
        return self.registry.get(model_key, {"error": "Model not found"})

    def list_models(self) -> list:
        """Returns a list of all active models and tools."""
        return [{"id": k, **v} for k, v in self.registry.items()]
