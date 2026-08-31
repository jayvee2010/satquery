# Architectural Design - SatQuery AI

This document provides a detailed breakdown of the system architecture, data flows, and routing algorithms of the **SatQuery AI** Geospatial Workstation.

---

## 1. System Block Diagram

```text
                  +-----------------------------------+
                  |      React Frontend Workstation   |
                  +-----------------+-----------------+
                                    |
                             /api/query & /api/upload
                                    |
                                    v
                  +-----------------+-----------------+
                  |      FastAPI Gateway / Router     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |      Orchestrator Agent (Gemini)  |
                  +--------+-----------------+--------+
                           |                 |
            [Routes Task]  |                 |  [Loads Config]
                           v                 v
                 +---------+---+     +-------+---------+
                 |  Model      |     |  Geospatial     |
                 |  Registry   |     |  Processing     |
                 +-----+-------+     +-------+---------+
                       |                     |
     +-----------------+-----------------+   |
     | Adapted ResNet LULC Model (PyTorch)|   |
     +-----------------------------------+   |
                       |                     |
                       v                     v
                 +-----+---------------------+---------+
                 |  Evidence Fusion & Stats Calculator |
                 +-----------------+-------------------+
                                   |
                         [Fuses and Synthesizes]
                                   v
                 +-----------------+-----------------+
                 |  Grounded Response Generator      |
                 +-----------------------------------+
```

---

## 2. Dynamic Workflow Execution

For a typical flood investigation request:

1. **Upload & Validate**:
   The user uploads a GeoTIFF. The backend opens it using `rasterio`, extracts spatial properties (CRS, resolution, affine transform), and translates UTM boundaries to WGS84 GPS boundaries (`latlon_bounds`) so they map directly onto Leaflet.
2. **Cognitive Routing**:
   The orchestrator parses the query.
   - If the task is **Change Analysis**, it checks that both pre-event and post-event images are present and spatially co-registered.
   - If the task is **Cross-Modal Fusion**, it verifies that both optical and SAR data are uploaded.
3. **Execution**:
   The agent schedules the appropriate tools.
   - Calculates **NDWI** for optical.
   - Calibrates and thresholds **SAR backscatter** ($<-18\text{ dB}$).
   - Runs patch-based **ResNet18 LULC classification**.
4. **Evidence Fusion & Geodesics**:
   - Calculates the IoU (Intersection over Union) between optical and SAR water masks.
   - Identifies discrepancies (e.g., cloud obscuring optical, SAR confirming water).
   - Multiplies pixel counts by pixel resolution area (in $km^2$) using UTM metrics or ellipsoidal degree adjustments to get actual affected areas.
5. **Grounded Response**:
   The structured statistics and limitations list are passed back to the LLM. It generates the final conversational response, ensuring that all numbers, locations, and confidence levels are tied 100% to the verified outputs (no hallucinations).
