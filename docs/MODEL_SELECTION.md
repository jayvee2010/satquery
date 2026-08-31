# Model Selection Document - SatQuery AI

This document details the selection process, evaluation, and rationale behind the models and specialist tools used in **SatQuery AI** for ISRO Problem Statement 26167.

---

## 1. Candidate Models Evaluated

| Model Candidate | Task Area | Size / Params | Pros | Cons | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GeoChat** | MLLM / VQA / Grounding | 7B | Strong conversational RS abilities. | Extremely heavy; requires >16GB VRAM; very high latency. | **Not Selected** (compute constraints) |
| **LocateAnything** | Grounding | 3B | Excellent referring expression accuracy. | Heavy; difficult to run locally on CPU/standard laptops. | **Not Selected** (deployment limits) |
| **EuroSAT-Adapted ResNet18** | Land Cover (LULC) | 11.7M | Tiny size; sub-second inference; trains in seconds; 10 LULC classes. | Does not support direct free-text generation natively. | **Selected** (for LULC extraction) |
| **Normalized Difference Water Index (NDWI)** | Optical Water | Formula | Georeferenced; 100% accurate spectral science; zero parameter. | Fails under cloud cover. | **Selected** (Geospatial Tool) |
| **SAR Backscatter Thresholding** | SAR Water | Formula | 100% cloud-penetrating; operates day/night; zero parameter. | Prone to wind-roughness noise. | **Selected** (Geospatial Tool) |
| **Gemini 3.5 Flash** | Orchestration & VLM | API | Multi-modal; high logical reasoning; ultra-low latency; handles synthesis. | Needs internet; requires API key. | **Selected** (Cognitive Orchestrator) |

---

## 2. Rationale & Expected Roles

### A. Cognitive Orchestrator: Gemini 3.5 Flash
- **Role**: Parses user queries, validates active imagery state, routes queries to the appropriate geospatial tools/adapted classifiers, fuses visual/statistical evidence, and synthesizes grounded natural-language responses.
- **License**: Proprietary (accessible via Google AI Studio API).
- **Hardware Requirement**: Low local footprint (API-based call).

### B. Domain-Adapted LULC Classifier: ResNet18 (adapted on EuroSAT)
- **Role**: Slices input optical images into 64x64 grids, classifies LULC classes (crop, pasture, forest, residential, water, industrial), and outputs a co-registered land-cover map.
- **License**: MIT / Apache 2.0 (PyTorch pre-trained weights).
- **Adaptation Strategy**: Legitimate PEFT/transfer learning on **EuroSAT RGB** (27,000 Sentinel-2 patches). Trains the classification head and final residual block for 1 epoch.
- **Hardware Requirement**: Ultra-low. Runs on CPU or Mac MPS (Apple Silicon).

### C. Optical & SAR Flood Detectors (NDWI & Low-Backscatter)
- **Role**: Performs pixel-level extraction of water bodies.
- **Formulas**:
  - **NDWI**: $(Green - NIR) / (Green + NIR)$ (threshold: 0.05).
  - **SAR**: $VH < -18 \text{ dB}$ (C-Band amplitude converted to relative decibels).
- **Hardware Requirement**: CPU (runs using NumPy vectorization in milliseconds).

---

## 3. Scientific Limitations

1. **Optical Cloud Obstruction**: NDWI cannot calculate water presence under clouds because cloud reflections obscure the ground. SatQuery flags this limitation and triggers SAR corroboration.
2. **SAR Surface Roughness**: High winds can create water surface roughness (waves), increasing SAR backscatter and causing false land classifications in water bodies.
3. **No Flood Depth**: Neither optical spectral indices nor single-frequency SAR backscatter can determine flood water depth. SatQuery explicitly reports this limitation if asked about depth, rather than generating fake values.
