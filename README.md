# SatQuery AI 🛰️
### Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis (ISRO PS 26167)

SatQuery AI is a technically credible remote-sensing workstation designed specifically for **satellite-based flood disaster investigation**. Built from scratch to satisfy **ISRO Problem Statement 26167**, the system isolates the conversational flexibility of Large Language Models (LLMs) from the deterministic accuracy of geospatial coordinate mapping and spectral calculations.

---

## 🗺️ Table of Contents
1. [Product Philosophy & LLM Boundaries](#1-product-philosophy--llm-boundaries)
2. [Geospatial & Remote Sensing Algorithms](#2-geospatial--remote-sensing-algorithms)
   * [Normalized Difference Water Index (NDWI)](#normalized-difference-water-index-ndwi)
   * [SAR Radar Backscatter Thresholding](#sar-radar-backscatter-thresholding)
   * [Bi-temporal Change Math](#bi-temporal-change-math)
   * [ResNet18 LULC Image Classification Loop](#resnet18-lulc-image-classification-loop)
   * [Centroid GPS Projection Derivation](#centroid-gps-projection-derivation)
3. [System Architecture](#3-system-architecture)
4. [Directory & Codebase Inventory](#4-directory--codebase-inventory)
5. [Step-by-Step Quick Start](#5-step-by-step-quick-start)
6. [Interactive Scenario Walkthroughs](#6-interactive-scenario-walkthroughs)
7. [Developer & Evaluator Guide](#7-developer--evaluator-guide)

---

## 1. Product Philosophy & LLM Boundaries

Traditional Vision-Language Models (VLMs) fail when deployed to geospatial analysis because they are prone to hallucinating coordinates, area calculations, and structural damage metrics. 

SatQuery AI introduces a strict boundary: **The LLM does not generate spatial data. It coordinates the tools that do.**

```text
User Question: "Where should my rescue team go first?"
  │
  ▼
Orchestrator Agent (FastAPI)
  │
  ├─► 1. Tasks routed using Gemini semantic parsing.
  ├─► 2. Runs NDWI, SAR, and PyTorch ResNet18 tools on numpy arrays.
  ├─► 3. Calculates flood intersections and computes spatial centroids (GPS).
  │
  ▼
Synthesis Engine (LLM or Template Fallback)
  │
  └─► Consumes raw numbers & coordinates, producing a grounded markdown briefing.
```

If the `GEMINI_API_KEY` is not present, the system runs locally using a deterministic fallback router and response compiler. No calculations, map layers, or statistics are degraded, guaranteeing that the dashboard remains functional even in offline environments.

---

## 2. Geospatial & Remote Sensing Algorithms

### Normalized Difference Water Index (NDWI)
Optical water extraction is done using the Normalized Difference Water Index (NDWI), which relies on the fact that liquid water absorbs Near-Infrared (NIR) radiation while reflecting visible green light.

$$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$

* **Smart Band Mapping**: The code in `backend/tools/flood_detector.py` scans the GeoTIFF's internal band descriptions (looking for keywords like "green", "nir", "infra", "b3", "b8"). If no descriptions are found, it uses standard defaults based on band counts:
  * **12 or 13 bands** (Standard Sentinel-2 products): Green = Band 3, NIR = Band 8.
  * **4 bands** (Typical PlanetScope/NAIP): Green = Band 2, NIR = Band 4.
  * **3 bands** (RGB): Falls back to the Normalized Green-Red Difference Index (NGRDI):
    $$\text{NGRDI} = \frac{\text{Green} - \text{Red}}{\text{Green} + \text{Red}}$$
    Combined with a dark-pixel threshold (average brightness $< 100$) to isolate dark water bodies.

### SAR Radar Backscatter Thresholding
Synthetic Aperture Radar (SAR) systems operating in the C-band (like Sentinel-1 or RISAT-1) penetrate cloud cover. Smooth standing water acts like a flat specular mirror, reflecting the radar beam away from the sensor. Consequently, flooded regions show up as dark pixels with extremely low backscatter coefficient ($\sigma^0$).

* **Logarithmic dB Scaling**: Real amplitude images are converted to relative backscatter in decibels:
  $$\text{Value (dB)} = 20 \times \log_{10}(\text{pixel\_value} + 1\text{e-5})$$
* **Thresholding**: Pixels with $\text{VH} < -18\text{ dB}$ (or $\text{VV} < -15\text{ dB}$) are flagged as water.

### Bi-temporal Change Math
The change analyzer (`backend/tools/change_analyzer.py`) compares pre-event and post-event binary water masks to map flood footprints and calculate dimensions:
* **Flood Expansion (New Inundation)**: 
  $$\text{Flooded Mask} = \text{PostMask} \text{ AND } (\text{NOT } \text{PreMask})$$
* **Stable Water (Permanent Channels)**:
  $$\text{Stable Mask} = \text{PostMask} \text{ AND } \text{PreMask}$$
* **Receded Water (Dry-up Zones)**:
  $$\text{Receded Mask} = (\text{NOT } \text{PostMask}) \text{ AND } \text{PreMask}$$
* **Metric Area Calculation**: The code multiplies the count of active pixels in each mask by the cell size ($X \times Y$) and divides by $1,000,000$ to get the area in square kilometers ($km^2$).

### ResNet18 LULC Image Classification Loop
Land Use / Land Cover (LULC) mapping is done using a PyTorch ResNet18 model fine-tuned on the 10-class EuroSAT Sentinel-2 dataset.
* **Patch-Based Processing**: Because the model is trained on $64 \times 64$ images, the engine slices large input GeoTIFFs into $64 \times 64$ tiles.
* **Batch Inference**: Runs a PyTorch forward pass over the tiles, predicting classes: *AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake*.
* **Upsampling**: The low-resolution class grid (e.g. $4 \times 4$ classes for a $256 \times 256$ input) is upsampled using nearest-neighbor interpolation to create a full-resolution $256 \times 256$ land cover grid.
* **Simplification**: The 10 EuroSAT classes are simplified into 4 operational categories for the emergency report:
  * `Built-up/Infra` (Residential, Industrial, Highway)
  * `Agriculture` (AnnualCrop, Pasture, PermanentCrop)
  * `Forest/Veg` (Forest, HerbaceousVegetation)
  * `Water Bodies` (River, SeaLake)

### Centroid GPS Projection Derivation
To provide coordinates for ground rescue teams, the engine calculates the center of mass (centroid) of the flooded pixels for each LULC class:
1. **Average Row/Col**: Finds the coordinates $(r_{avg}, c_{avg})$ where $(\text{flood\_mask} == 1) \text{ AND } (\text{lulc\_grid} == \text{target\_class})$.
2. **Affine Translation**: Maps pixel positions to projected map coordinates (Easting/Northing) using the image's affine coefficients:
   $$X_{proj} = a \times c_{avg} + b \times r_{avg} + c$$
   $$Y_{proj} = d \times c_{avg} + e \times r_{avg} + f$$
3. **Coordinate Transformation**: Reprojects the Easting/Northing $(X_{proj}, Y_{proj})$ from its local coordinate system (e.g., UTM Zone 45N) into global WGS84 GPS coordinates (Latitude/Longitude degrees) using `rasterio.warp.transform`.

---

## 3. System Architecture

```text
                   React SPA (Port 3000)
                             │
                             ▼  (REST API calls via /api/ proxy)
                    FastAPI (Port 8080)
                             │
                             ├─► [Orchestrator] Task Routing (Gemini or Fallback)
                             ├─► [Flood Detector] calculates NDWI & SAR dB
                             ├─► [Change Analyzer] subtracts pre & post masks
                             ├─► [ResNet18 Model] classifies LULC patches
                             ├─► [Land Cover Intersector] merges masks & computes centroids
                             │
                             ▼
                    JSON Response Payload
  (Answer text + Coordinate boundaries + 2D Mask data grids)
                             │
                             ▼
  Leaflet Map overlay maps rendered dynamically via HTML5 Canvas
```

---

## 4. Directory & Codebase Inventory

* `backend/`
  * `api/`
    * `main.py`: FastAPI server endpoints, static files mount, upload handling, CORS configurations.
  * `agents/`
    * `orchestrator.py`: Orchestrator agent that performs query classification, schedules geospatial tools, checks file compatibilities, and synthesizes answers.
  * `tools/`
    * `geospatial.py`: Reads GeoTIFF dimensions, CRS, bounds, and reprojects coordinate points.
    * `flood_detector.py`: Implements optical NDWI band mapping and SAR backscatter DB thresholding.
    * `change_analyzer.py`: Performs logical bi-temporal water subtraction and area calculation.
    * `landcover_intersect.py`: Intersects flood footprints with LULC and calculates coordinate centroids.
  * `models/`
    * `registry.py`: Model loader holding checkpoints and initializing Fallbacks.
    * `rs_adapter.py`: Slices rasters, loads ResNet18 weights, and runs patch-inference.
* `frontend/`
  * `src/`
    * `App.tsx`: Main React component implementing Leaflet map, opacity sliders, chat interfaces, execution trace panels, and canvas overlays.
* `scripts/`
  * `train_rs_model.py`: Fetches EuroSAT subsets and trains ResNet18.
  * `generate_demo_data.py`: Generates co-registered simulated GeoTIFF datasets.
* `tests/`
  * `test_backend.py`: Pytest suite verifying backend endpoints, NDWI thresholds, SAR backscatter, change math, and agent routing.

---

## 5. Step-by-Step Quick Start

### 📋 Prerequisites
* Python 3.10, 3.11, or 3.12 (must have headers installed, e.g. `python3-dev`)
* Node.js v18+ and npm
* C++ Compiler (GCC/Clang) for compiling rasterio/gdal bindings

### 🚀 Bootstrapping the System

1. **Clone and Configure Environment**:
   ```bash
   git clone git@github.com:jayvee2010/satquery.git
   cd satquery
   cp .env.example .env
   ```
   *Edit `.env` and set `GEMINI_API_KEY=your_key` (if omitted, the system will seamlessly run using its local rule-based engine).*

2. **Install Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies**:
   ```bash
   npm run install:frontend
   ```

4. **Train LULC Model & Create Demo Datasets**:
   ```bash
   # Trains ResNet18 classifier on EuroSAT (utilizes MPS/GPU if available)
   python3 scripts/train_rs_model.py
   
   # Generates simulated Assam Valley GeoTIFFs (spatial co-registered)
   python3 scripts/generate_demo_data.py
   ```

5. **Start Dev Servers**:
   ```bash
   npm start
   ```
   *Starts FastAPI backend on port `8080` and Vite dev server on port `3000`. Navigate your browser to `http://localhost:3000`.*

---

## 6. Interactive Scenario Walkthroughs

Once the workstation is open, click the **"Load Indian Flood Demo Data"** button to load the pre-generated Assam Valley GeoTIFFs. You can then copy-paste these queries to test the analysis pipelines:

### Scenario A: Single-Image VQA
* **Query**: `"Is there water in the post-event optical image?"`
* **Backend Flow**: Routes to `VQA` -> Runs NDWI on `optical_post.tif` -> Calculates water covers 22.5% of the scene -> Returns confirmation.

### Scenario B: Single-Image Grounding
* **Query**: `"Highlight the water body"`
* **Backend Flow**: Routes to `GROUNDING` -> Runs NDWI -> Generates binary mask -> Frontend overlays blue pixels on the Leaflet map overlay using HTML5 Canvas.

### Scenario C: Bi-Temporal Flood Change
* **Query**: `"What changed after the flood?"`
* **Backend Flow**: Routes to `CHANGE_ANALYSIS` -> Validates presence of pre- and post-event images -> Subtracts masks -> Reports **0.627 km²** of new flood expansion.

### Scenario D: Rescue Prioritization
* **Query**: `"Which area should my rescue team go to first?"`
* **Backend Flow**: Routes to `LAND_COVER_IMPACT` -> Runs ResNet18 LULC and NDWI -> Intersects agricultural and residential zones -> Calculates GPS centroids -> Outputs **High Priority (Residential)** and **Medium Priority (Crops)** navigation targets.

---

## 7. Developer & Evaluator Guide

### Running Automated Test Suites
Our tests verify that the mathematical algorithms, coordinate conversions, and agents remain stable. Run:
```bash
PYTHONPATH=. pytest tests/test_backend.py
```
*The test suite generates mock GeoTIFFs dynamically and runs them through the actual PyTorch weights and index tools, verifying 100% compliance with calculation expectations.*

### How to Add a New Geospatial Tool
1. Write your analysis function in `backend/tools/` (e.g. `vegetation_index.py`).
2. Register the tool in `backend/agents/orchestrator.py` by importing it.
3. Update `route_query_fallback` and `route_query` to recognize keywords related to your tool.
4. Add your tool output to `synthesize_response_fallback` and `synthesize_response` to include it in the final markdown summary.
