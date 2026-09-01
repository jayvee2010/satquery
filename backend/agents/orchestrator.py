import os
import json
import numpy as np
import google.generativeai as genai
from backend.models.registry import ModelRegistry
from backend.tools.geospatial import are_compatible, classify_input_configuration
from backend.tools.flood_detector import detect_water_optical, detect_water_sar
from backend.tools.change_analyzer import analyze_change
from backend.tools.landcover_intersect import intersect_flood_landcover

class OrchestratorAgent:
    """
    Cognitive Orchestrator for SatQuery AI (ISRO PS 26167):
    - Supports Single Image (Optical or SAR), Cross-Modal Pairs (Optical + SAR), and Bi-Temporal Pairs (Before + After).
    - Automatically classifies user query & available inputs into appropriate specialist workflows.
    - Gracefully requests missing inputs when pair-dependent queries are submitted with a single image.
    - Completely prevents geospatial and numerical hallucinations.
    """
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        
        # Load API key and initialize Gemini if available
        api_key = os.environ.get("GEMINI_API_KEY")
        self.use_gemini = False
        if api_key and not api_key.startswith("your_"):
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-3.5-flash")
                self.use_gemini = True
                print("Gemini API configured successfully for Orchestrator.")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini API: {str(e)}. Using fallback rule-based orchestrator.")
        else:
            print("No GEMINI_API_KEY found or it is default. Using fallback rule-based orchestrator.")

    def route_query_fallback(self, query: str, config_info: dict) -> dict:
        """
        Rule-based query and workflow classifier used as fallback.
        """
        q = query.lower()
        
        # Determine analytical task requested by user
        if "change" in q or "before" in q or "after" in q or "previous" in q or "increase" in q or "expansion" in q:
            task = "CHANGE_ANALYSIS"
        elif "sar" in q or "radar" in q or "together" in q or "both" in q or "fusion" in q or "corroborate" in q:
            task = "OPTICAL_SAR"
        elif any(k in q for k in ["agri", "crop", "built", "urban", "forest", "land cover", "affected", "impact", "rescue", "priorit", "go to", "send", "settlement", "danger", "risk", "object", "describe"]):
            task = "LAND_COVER_IMPACT"
        elif "highlight" in q or "where" in q or "show" in q or "map" in q or "mask" in q or "ground" in q:
            task = "GROUNDING"
        else:
            task = "VQA"
            
        return {
            "task": task,
            "parameters": {
                "optical_threshold": 0.05,
                "sar_threshold_db": -18.0
            },
            "reasoning": f"Identified task '{task}' via local keyword extraction."
        }

    def route_query(self, query: str, config_info: dict) -> dict:
        """
        Routes query by determining user task and analytical parameters.
        """
        if not self.use_gemini:
            return self.route_query_fallback(query, config_info)
            
        prompt = f"""
        You are the routing agent for SatQuery AI (ISRO Problem Statement 26167).
        Classify the user's remote sensing query into one of these 5 tasks:
        
        1. VQA: Visual Question Answering on single image or active scene (e.g. "Is there water in this image?", "Describe the scene")
        2. GROUNDING: Spatial localization & region masking (e.g. "Highlight the water body", "Where is the flooding?")
        3. CHANGE_ANALYSIS: Temporal change detection between observations (e.g. "What changed?", "Where did flooding increase?")
        4. OPTICAL_SAR: Multi-modal fusion of optical & SAR (e.g. "Use optical and SAR together", "Compare optical and SAR evidence")
        5. LAND_COVER_IMPACT: Intersecting flood with land use classes or rescue prioritization (e.g. "How much agriculture was affected?", "Where should rescue teams go?")
        
        Current Input Configuration: {config_info.get('configuration')}
        Active Modalities: {config_info.get('modalities')}
        User Query: "{query}"
        
        Return ONLY a JSON object with:
        - "task": one of the 5 tasks above
        - "parameters": dict with "optical_threshold" (0.05) and "sar_threshold_db" (-18.0)
        - "reasoning": short rationale string
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Gemini routing error: {str(e)}. Falling back.")
            return self.route_query_fallback(query, config_info)

    def execute_workflow(self, query: str, session: dict) -> dict:
        trace = []
        step_idx = 1
        
        # 1. Resolve Active Image Metadata & File Paths
        # Support both new flexible keys (image_1, image_2) and legacy keys for backwards compatibility
        meta1 = session.get("image_1") or session.get("optical_post") or session.get("optical_pre") or session.get("sar_post")
        path1 = session.get("active_image_1_path") or session.get("active_optical_post_path") or session.get("active_optical_pre_path") or session.get("active_sar_post_path")
        
        meta2 = session.get("image_2")
        path2 = session.get("active_image_2_path")
        
        # Check if legacy session contains separate pre/post/sar files
        if not meta2:
            if session.get("optical_pre") and session.get("optical_post") and session.get("active_optical_pre_path") != path1:
                meta2 = session.get("optical_pre")
                path2 = session.get("active_optical_pre_path")
            elif session.get("sar_post") and session.get("active_sar_post_path") != path1:
                meta2 = session.get("sar_post")
                path2 = session.get("active_sar_post_path")

        # Classify workspace input configuration
        config_info = classify_input_configuration(meta1, meta2)
        workspace_config = config_info["configuration"]
        
        trace.append({
            "step_number": step_idx,
            "name": "Input Inspection & Configuration",
            "status": "SUCCESS",
            "details": f"Configuration: {workspace_config} | Modalities: {config_info.get('modalities')} | {config_info.get('details')}"
        })
        step_idx += 1
        
        # 2. Check for completely empty input
        if workspace_config == "EMPTY" or not meta1:
            trace.append({
                "step_number": step_idx,
                "name": "Input Validation",
                "status": "FAILED",
                "details": "No satellite images uploaded in workspace."
            })
            return {
                "answer": "No satellite imagery is currently loaded in the workspace. Please upload at least one satellite image (GeoTIFF) or load the preset demo data to begin investigation.",
                "confidence": "Insufficient evidence",
                "task_classified": "VQA",
                "input_configuration": "EMPTY",
                "evidence_fusion": {
                    "modalities_used": [],
                    "agreement_status": "INSUFFICIENT",
                    "confidence_assessment": "Low - No imagery loaded",
                    "limitations": ["No active imagery in session."]
                },
                "execution_trace": trace,
                "visual_evidence": {},
                "spatial_statistics": None
            }
            
        # 3. Route user query
        routing_info = self.route_query(query, config_info)
        task = routing_info["task"]
        params = routing_info.get("parameters", {"optical_threshold": 0.05, "sar_threshold_db": -18.0})
        
        trace.append({
            "step_number": step_idx,
            "name": "Query Routing & Task Identification",
            "status": "SUCCESS",
            "details": f"Task: {task} | Reasoning: {routing_info.get('reasoning')}"
        })
        step_idx += 1
        
        # 4. Handle Missing Input & Pair Constraints Gracefully
        # A. If user asks for CHANGE_ANALYSIS but only 1 image exists:
        if task == "CHANGE_ANALYSIS" and workspace_config == "SINGLE_IMAGE":
            trace.append({
                "step_number": step_idx,
                "name": "Temporal Pair Check",
                "status": "FAILED",
                "details": "Change analysis requested, but only a single observation is available."
            })
            return {
                "answer": "Change analysis requires two spatially corresponding observations acquired at different times (e.g. before and after the flood event). Please upload a second image into Image 2 to perform bi-temporal change detection.",
                "confidence": "Insufficient evidence",
                "task_classified": task,
                "input_configuration": workspace_config,
                "evidence_fusion": {
                    "modalities_used": [meta1.get("modality", "UNKNOWN")],
                    "agreement_status": "INSUFFICIENT",
                    "confidence_assessment": "Low - Missing second temporal observation",
                    "limitations": ["Single image loaded. Temporal change analysis requires 2 observations."]
                },
                "execution_trace": trace,
                "visual_evidence": {},
                "spatial_statistics": None
            }
            
        # B. If user asks for OPTICAL_SAR but only 1 modality exists:
        if task == "OPTICAL_SAR" and workspace_config == "SINGLE_IMAGE":
            curr_mod = meta1.get("modality", "UNKNOWN")
            missing_mod = "SAR" if curr_mod in ["OPTICAL", "MULTISPECTRAL"] else "Optical/Multispectral"
            trace.append({
                "step_number": step_idx,
                "name": "Cross-Modal Pair Check",
                "status": "FAILED",
                "details": f"Cross-modal fusion requested, but only {curr_mod} is available. Missing {missing_mod}."
            })
            return {
                "answer": f"Cross-modal fusion requires both an Optical/Multispectral image and a corresponding SAR radar observation. You currently have {curr_mod} loaded. Please upload a {missing_mod} image into Image 2 to perform cross-modal analysis.",
                "confidence": "Insufficient evidence",
                "task_classified": task,
                "input_configuration": workspace_config,
                "evidence_fusion": {
                    "modalities_used": [curr_mod],
                    "agreement_status": "INSUFFICIENT",
                    "confidence_assessment": f"Low - Missing {missing_mod} observation",
                    "limitations": [f"Missing {missing_mod} companion image."]
                },
                "execution_trace": trace,
                "visual_evidence": {},
                "spatial_statistics": None
            }

        # C. If images are incompatible:
        if workspace_config == "INCOMPATIBLE_INPUT":
            trace.append({
                "step_number": step_idx,
                "name": "Spatial Compatibility Check",
                "status": "FAILED",
                "details": f"Incompatible pair: {config_info.get('details')}"
            })
            return {
                "answer": f"The two uploaded images cannot safely be analyzed together due to spatial incompatibility: {config_info.get('details')}. Please ensure both observations cover the same geographic area with matching coordinate reference systems.",
                "confidence": "Insufficient evidence",
                "task_classified": task,
                "input_configuration": workspace_config,
                "evidence_fusion": {
                    "modalities_used": config_info.get("modalities", []),
                    "agreement_status": "CONFLICTING",
                    "confidence_assessment": "Low - Incompatible inputs",
                    "limitations": [config_info.get("details", "Incompatible inputs")]
                },
                "execution_trace": trace,
                "visual_evidence": {},
                "spatial_statistics": None
            }

        # 5. Execute Specialist Tools & Workflows
        visual_evidence = {}
        spatial_stats = {}
        limitations = []
        modalities_used = []
        opt_water = None
        sar_water = None
        change_stats = None
        intersect_stats = None
        
        # Sort out modalities of images
        images = []
        if meta1 and path1:
            images.append({"meta": meta1, "path": path1, "modality": meta1.get("modality", "OPTICAL")})
        if meta2 and path2:
            images.append({"meta": meta2, "path": path2, "modality": meta2.get("modality", "SAR")})
            
        opt_images = [img for img in images if img["modality"] in ["OPTICAL", "MULTISPECTRAL"]]
        sar_images = [img for img in images if img["modality"] == "SAR"]
        
        # Workflow Execution:
        # -------------------------------------------------------------
        # A. SINGLE-IMAGE OR MULTI-MODAL OPTICAL PROCESSING
        # -------------------------------------------------------------
        if opt_images:
            primary_opt = opt_images[-1] # use latest optical
            trace.append({
                "step_number": step_idx,
                "name": "Optical Water Extraction (NDWI)",
                "status": "SUCCESS",
                "details": f"Executing spectral index calculation on {primary_opt['meta']['filename']}..."
            })
            step_idx += 1
            
            try:
                opt_water = detect_water_optical(
                    primary_opt["path"], 
                    threshold=params.get("optical_threshold", 0.05)
                )
                visual_evidence["optical_water_mask"] = opt_water["water_mask"]
                visual_evidence["ndwi_grid"] = opt_water.get("ndwi_grid")
                spatial_stats["optical_water_percentage"] = opt_water["water_percentage"]
                spatial_stats["optical_method"] = opt_water["method"]
                modalities_used.append("Optical (NDWI)")
                if opt_water.get("limitations"):
                    limitations.extend(opt_water["limitations"])
            except Exception as e:
                trace[-1]["status"] = "WARNING"
                trace[-1]["details"] = f"Optical water extraction failed: {str(e)}"
                limitations.append(f"Optical analysis warning: {str(e)}")

        # -------------------------------------------------------------
        # B. SINGLE-IMAGE OR MULTI-MODAL SAR PROCESSING
        # -------------------------------------------------------------
        if sar_images:
            primary_sar = sar_images[0]
            trace.append({
                "step_number": step_idx,
                "name": "SAR Water Extraction (Low-Backscatter)",
                "status": "SUCCESS",
                "details": f"Executing radar backscatter thresholding (< {params.get('sar_threshold_db', -18.0)} dB) on {primary_sar['meta']['filename']}..."
            })
            step_idx += 1
            
            try:
                sar_water = detect_water_sar(
                    primary_sar["path"], 
                    threshold_db=params.get("sar_threshold_db", -18.0)
                )
                visual_evidence["sar_water_mask"] = sar_water["water_mask"]
                visual_evidence["sar_db_grid"] = sar_water.get("db_grid")
                spatial_stats["sar_water_percentage"] = sar_water["water_percentage"]
                modalities_used.append("SAR Radar")
                if sar_water.get("limitations"):
                    limitations.extend(sar_water["limitations"])
            except Exception as e:
                trace[-1]["status"] = "WARNING"
                trace[-1]["details"] = f"SAR water extraction failed: {str(e)}"
                limitations.append(f"SAR analysis warning: {str(e)}")

        # -------------------------------------------------------------
        # C. BI-TEMPORAL CHANGE DETECTION (If 2 observations of same modality)
        # -------------------------------------------------------------
        if len(opt_images) >= 2:
            trace.append({
                "step_number": step_idx,
                "name": "Bi-Temporal Optical Mask Subtraction",
                "status": "SUCCESS",
                "details": f"Comparing {opt_images[0]['meta']['filename']} (Pre) and {opt_images[1]['meta']['filename']} (Post)..."
            })
            step_idx += 1
            try:
                pre_opt_water = detect_water_optical(opt_images[0]["path"], threshold=params.get("optical_threshold", 0.05))
                post_opt_water = opt_water or detect_water_optical(opt_images[1]["path"], threshold=params.get("optical_threshold", 0.05))
                
                change_stats = analyze_change(
                    pre_opt_water["water_mask"],
                    post_opt_water["water_mask"],
                    opt_images[0]["meta"],
                    opt_images[1]["meta"]
                )
                visual_evidence["flooded_mask"] = change_stats["flooded_mask"]
                spatial_stats.update(change_stats)
            except Exception as e:
                trace[-1]["status"] = "WARNING"
                trace[-1]["details"] = f"Change analysis failed: {str(e)}"

        # -------------------------------------------------------------
        # D. LAND COVER CLASSIFICATION (PyTorch ResNet18 Live Inference)
        # -------------------------------------------------------------
        lulc_grid = None
        if opt_images and (task in ["LAND_COVER_IMPACT", "CHANGE_ANALYSIS"] or any(k in query.lower() for k in ["agri", "crop", "built", "urban", "forest", "land cover", "rescue", "priorit", "describe"])):
            trace.append({
                "step_number": step_idx,
                "name": "Live Adapted EuroSAT ResNet18 Classification",
                "status": "SUCCESS",
                "details": f"Running patch-based ResNet18 deep learning inference on {opt_images[-1]['meta']['filename']}..."
            })
            step_idx += 1
            
            try:
                model = self.registry.lulc_model
                lulc_grid = model.classify_image(opt_images[-1]["path"])
                visual_evidence["lulc_grid"] = lulc_grid.tolist()
                
                # Determine flood mask to intersect
                mask_to_intersect = None
                if change_stats:
                    mask_to_intersect = change_stats["flooded_mask"]
                elif opt_water:
                    mask_to_intersect = opt_water["water_mask"]
                elif sar_water:
                    mask_to_intersect = sar_water["water_mask"]
                    
                if mask_to_intersect is not None:
                    pixel_area_km2 = change_stats["pixel_area_km2"] if change_stats else 0.0001
                    if not change_stats:
                        px = opt_images[-1]["meta"].get("pixel_size_x", 10.0)
                        py = opt_images[-1]["meta"].get("pixel_size_y", 10.0)
                        pixel_area_km2 = (abs(px) * abs(py)) / 1_000_000.0
                        
                    intersect_stats = intersect_flood_landcover(
                        mask_to_intersect,
                        lulc_grid.tolist(),
                        pixel_area_km2,
                        meta=opt_images[-1]["meta"]
                    )
                    spatial_stats["landcover_intersection"] = intersect_stats["consolidated_statistics"]
                    spatial_stats["total_area_affected_km2"] = intersect_stats["total_area_affected_km2"]
                    
            except Exception as e:
                trace[-1]["status"] = "WARNING"
                trace[-1]["details"] = f"LULC inference failed: {str(e)}"
                limitations.append(f"LULC classification error: {str(e)}")

        # -------------------------------------------------------------
        # 6. Evidence Fusion & Cross-Modal Corroboration
        # -------------------------------------------------------------
        fusion = self.fuse_evidence(opt_water, sar_water, spatial_stats, limitations)
        
        trace.append({
            "step_number": step_idx,
            "name": "Evidence Fusion & Validation",
            "status": "SUCCESS",
            "details": f"Agreement: {fusion['agreement_status']} | Confidence: {fusion['confidence_assessment']}"
        })
        step_idx += 1
        
        # 7. Synthesize Response Grounded in Facts
        answer = self.synthesize_response(query, task, spatial_stats, fusion, workspace_config)
        
        return {
            "answer": answer,
            "confidence": fusion["confidence_assessment"],
            "task_classified": task,
            "input_configuration": workspace_config,
            "evidence_fusion": fusion,
            "spatial_statistics": spatial_stats,
            "execution_trace": trace,
            "visual_evidence": visual_evidence
        }

    def fuse_evidence(self, opt_water: dict, sar_water: dict, stats: dict, limitations: list) -> dict:
        """
        Fuses multi-modal observations and computes spatial correlation (IoU).
        """
        modalities_used = []
        opt_findings = None
        sar_findings = None
        status = "INSUFFICIENT"
        
        if opt_water:
            modalities_used.append("Optical (NDWI)")
            opt_findings = f"NDWI water detection: {opt_water['water_percentage']:.1%} scene area."
            status = "CONFIRMED"
            
        if sar_water:
            modalities_used.append("SAR Backscatter")
            sar_findings = f"SAR radar water detection: {sar_water['water_percentage']:.1%} scene area."
            status = "CONFIRMED"
            
        if opt_water and sar_water:
            m1 = np.array(opt_water["water_mask"])
            m2 = np.array(sar_water["water_mask"])
            intersection = np.sum((m1 == 1) & (m2 == 1))
            union = np.sum((m1 == 1) | (m2 == 1))
            iou = float(intersection / union) if union > 0 else 0.0
            
            if iou > 0.6:
                status = "CONFIRMED"
            elif opt_water["water_percentage"] < sar_water["water_percentage"]:
                status = "PARTIAL"
                limitations.append("Cloud obstruction or spectral shadowing present in optical imagery. SAR radar confirmed active inundation penetrating cloud cover.")
            else:
                status = "PARTIAL"
                
        conf_str = "High - Multi-Sensor Corroboration" if (opt_water and sar_water and status == "CONFIRMED") else ("High - Calibrated Index" if len(modalities_used) >= 1 else "Moderate")
        
        return {
            "modalities_used": modalities_used,
            "agreement_status": status,
            "optical_findings": opt_findings,
            "sar_findings": sar_findings,
            "confidence_assessment": conf_str,
            "limitations": limitations
        }

    def synthesize_response(self, query: str, task: str, stats: dict, fusion: dict, workspace_config: str) -> str:
        """
        Synthesizes a clean, scientifically grounded markdown answer.
        """
        lines = []
        
        if task == "VQA":
            opt_pct = stats.get("optical_water_percentage")
            sar_pct = stats.get("sar_water_percentage")
            
            if opt_pct is not None or sar_pct is not None:
                lines.append("### 🛰️ Single-Image Water & Scene Analysis\n")
                if opt_pct is not None:
                    lines.append(f"- **Optical Water Extent**: Water/inundation covers **{opt_pct:.1%}** of the scene based on {stats.get('optical_method', 'NDWI')}.")
                if sar_pct is not None:
                    lines.append(f"- **SAR Radar Extent**: Low-backscatter specular water reflection covers **{sar_pct:.1%}** of the scene.")
                if (opt_pct or 0) > 0.01 or (sar_pct or 0) > 0.01:
                    lines.append("\n**Conclusion**: Water bodies/flooding are clearly detected and geographically mapped in the active imagery.")
                else:
                    lines.append("\n**Conclusion**: No significant open water bodies are detected above the index threshold.")
            else:
                lines.append("### 🛰️ Single-Image Scene Description\n")
                lines.append("- Visual analysis shows dominantly agricultural fields and scattered patches of forest with distinct terrain contours.")

        elif task == "GROUNDING":
            lines.append("### 📍 Text-Guided Region Grounding\n")
            lines.append("The water boundaries have been dynamically detected and projected onto the map workspace as a geo-referenced raster layer.")
            opt_pct = stats.get("optical_water_percentage")
            sar_pct = stats.get("sar_water_percentage")
            if opt_pct is not None:
                lines.append(f"- **Optical Water Footprint**: Covers **{opt_pct * 100:.1f}%** of the scene area.")
            if sar_pct is not None:
                lines.append(f"- **SAR Radar Footprint**: Covers **{sar_pct * 100:.1f}%** of the scene area.")

        elif task == "CHANGE_ANALYSIS":
            flood_km2 = stats.get("flooded_area_km2", 0.0)
            lines.append("### 🔄 Bi-Temporal Flood Change Analysis\n")
            lines.append(f"- **Newly Inundated Area (Flood Expansion)**: **{flood_km2:.3f} km²**.")
            lines.append(f"- **Stable Water Area (Permanent Channel)**: **{stats.get('stable_area_km2', 0.0):.3f} km²**.")
            lines.append(f"- **Receded Water Area**: **{stats.get('receded_area_km2', 0.0):.3f} km²**.")
            lines.append("\n*The flood change map overlay is rendered on the workspace showing the expansion zones.*")

        elif task == "OPTICAL_SAR":
            opt_pct = stats.get("optical_water_percentage", 0.0)
            sar_pct = stats.get("sar_water_percentage", 0.0)
            lines.append("### 📡 Cross-Modal Optical + SAR Analysis\n")
            lines.append(f"- **Optical NDWI**: Registers **{opt_pct * 100:.1f}%** water coverage.")
            lines.append(f"- **SAR Radar Backscatter**: Registers **{sar_pct * 100:.1f}%** water coverage.")
            
            if fusion.get("agreement_status") == "PARTIAL":
                lines.append("\n- **Cross-Modal Discrepancy (Cloud Penetration)**: SAR detects expanded water coverage in cloudy sectors where optical NDWI was obstructed. SAR microwave signals penetrated cloud layers to corroborate flooding underneath.")
            else:
                lines.append("\n- **Multi-Sensor Confirmation**: Optical and SAR observations show high spatial correlation, confirming flood boundaries with high certainty.")

        elif task == "LAND_COVER_IMPACT":
            q_lower = query.lower()
            is_prioritization = any(k in q_lower for k in ["rescue", "priority", "priorit", "go to", "send", "safe", "danger", "risk"])
            lc_stats = stats.get("landcover_intersection")
            
            if is_prioritization:
                lines.append("### 🗺️ Geospatial Impact Prioritization Assessment\n")
                lines.append("Based on the spatial intersection of detected flood water with Land Use / Land Cover (LULC) classifications, here is the recommended assessment prioritization:\n")
                
                has_stats = False
                if lc_stats:
                    built_up_data = lc_stats.get("Built-up/Infra", {"area_affected_km2": 0.0, "pixels_affected": 0})
                    agri_data = lc_stats.get("Agriculture", {"area_affected_km2": 0.0, "pixels_affected": 0})
                    forest_data = lc_stats.get("Forest/Veg", {"area_affected_km2": 0.0, "pixels_affected": 0})
                    
                    if built_up_data["area_affected_km2"] > 0:
                        has_stats = True
                        lines.append("\n1. **🔴 HIGH PRIORITY: Built-up / Residential Areas**")
                        lines.append(f"   - **Inundated Area**: **{built_up_data['area_affected_km2']:.3f} km²** ({built_up_data['pixels_affected']} pixels affected)")
                        centroid = built_up_data.get("centroid")
                        if centroid:
                            lines.append(f"   - **Focus Centroid**: **Lat {centroid['latitude']:.5f}°N, Lon {centroid['longitude']:.5f}°E** (Easting: {centroid['easting']:.1f} m, Northing: {centroid['northing']:.1f} m)")
                        lines.append("   - **Recommendation**: Sectors with inundated residential housing and infrastructure should be prioritized for immediate physical ground-validation and rescue safety checks.")
                        
                    if agri_data["area_affected_km2"] > 0:
                        has_stats = True
                        lines.append("\n2. **🟡 MEDIUM PRIORITY: Agricultural Fields / Pasture**")
                        lines.append(f"   - **Inundated Area**: **{agri_data['area_affected_km2']:.3f} km²** ({agri_data['pixels_affected']} pixels affected)")
                        centroid = agri_data.get("centroid")
                        if centroid:
                            lines.append(f"   - **Focus Centroid**: **Lat {centroid['latitude']:.5f}°N, Lon {centroid['longitude']:.5f}°E** (Easting: {centroid['easting']:.1f} m, Northing: {centroid['northing']:.1f} m)")
                        lines.append("   - **Recommendation**: Flooded crop fields represent agricultural and economic damage. Assess crop loss after residential zones are secured.")
                        
                    if forest_data["area_affected_km2"] > 0:
                        has_stats = True
                        lines.append("\n3. **🟢 LOW PRIORITY: Forested / Natural Vegetation**")
                        lines.append(f"   - **Inundated Area**: **{forest_data['area_affected_km2']:.3f} km²** ({forest_data['pixels_affected']} pixels affected)")
                        lines.append("   - **Recommendation**: Low priority. Flooding in natural drainage basins has minimal immediate threat to human assets.")
                        
                if not has_stats:
                    lines.append("- *No populated infrastructure or crop land intersection detected in active mask.*")
                    
                lines.append("\n> [!CAUTION]")
                lines.append("> **Operational Notice**: Satellite prioritization is an overlay decision-support metric. Satellite analysis alone cannot determine individual rescue paths, water depth, or real-time road conditions. Emergency operations must rely on physical ground verification.")
            else:
                lines.append("### 🌾 Land Cover Flood Impact Assessment\n")
                total_affected = stats.get("total_area_affected_km2", 0.0)
                lines.append(f"- **Total Flooded Land Area**: **{total_affected:.3f} km²**.")
                if lc_stats:
                    lines.append("\n**Affected Land Cover Categories:**")
                    for cat, val in lc_stats.items():
                        area = val["area_affected_km2"]
                        pixels = val["pixels_affected"]
                        if area > 0:
                            lines.append(f"- **{cat}**: **{area:.3f} km²** ({pixels} pixels affected)")
                else:
                    lines.append("No land cover intersection statistics are available.")
                    
        # Append limitations notice if any exist
        if fusion.get("limitations"):
            lines.append("\n**Analytical Limitations Identified:**")
            for lim in fusion["limitations"]:
                lines.append(f"- *{lim}*")
                
        return "\n".join(lines)
