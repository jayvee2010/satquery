import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, Upload, ShieldAlert, Cpu, 
  Download, RefreshCw, Compass, 
  AlertTriangle, ChevronRight, Info
} from 'lucide-react';
import { MapContainer, TileLayer, ImageOverlay, useMap } from 'react-leaflet';
import L from 'leaflet';

// CSS fix for default Leaflet icon paths in React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Types definitions
interface ImageMetadata {
  filename: string;
  is_geotiff: boolean;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  crs: string;
  transform: number[];
  bounds: { left: number; bottom: number; right: number; top: number };
  latlon_bounds: { min_lat: number; min_lon: number; max_lat: number; max_lon: number; center: [number, number] };
  pixel_size_x: number;
  pixel_size_y: number;
}

interface Message {
  sender: 'user' | 'system';
  text: string;
  timestamp: string;
}

interface ExecutionStep {
  step_number: number;
  name: string;
  status: string;
  details: string;
}

interface EvidenceFusion {
  modalities_used: string[];
  agreement_status: string;
  optical_findings?: string;
  sar_findings?: string;
  confidence_assessment: string;
  limitations: string[];
}

interface AnalysisResult {
  answer: string;
  confidence: string;
  task_classified: string;
  evidence_fusion: EvidenceFusion;
  spatial_statistics?: any;
  execution_trace: ExecutionStep[];
  visual_evidence: any;
}

// Map center controller helper
function MapRecenter({ center }: { center: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.setView(center, 13);
    }
  }, [center, map]);
  return null;
}

export default function App() {
  // Session ID state
  const [sessionId] = useState(() => `session_${uuidv4()}`);
  
  // Image metadata states
  const [opticalPre, setOpticalPre] = useState<ImageMetadata | null>(null);
  const [opticalPost, setOpticalPost] = useState<ImageMetadata | null>(null);
  const [sarPost, setSarPost] = useState<ImageMetadata | null>(null);
  
  // App UI states
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'system',
      text: "Welcome to SatQuery AI Geospatial Workstation. Upload satellite images (GeoTIFF/TIFF) or load the pre-configured Indian flood demo data. Ask natural-language questions to analyze water expansion, flood impact on land covers, and verify cross-modal evidence.",
      timestamp: new Date().toLocaleTimeString()
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  
  // Map display settings
  const [selectedLayer, setSelectedLayer] = useState<string>('satellite'); // satellite, pre_optical, post_optical, flood_mask, change_mask, lulc_map
  const [opacity, setOpacity] = useState<number>(75);
  const [mapCenter, setMapCenter] = useState<[number, number] | null>([26.24, 93.02]); // Centered near Assam, India
  const [activeOverlayUrl, setActiveOverlayUrl] = useState<string | null>(null);
  const [overlayBounds, setOverlayBounds] = useState<[[number, number], [number, number]] | null>(null);
  const [modelRegistry, setModelRegistry] = useState<any[]>([]);
  const [showRegistry, setShowRegistry] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadingModality, setUploadingModality] = useState<string | null>(null);

  // Fetch model registry on mount
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => setModelRegistry(data))
      .catch(err => console.error("Error fetching models:", err));
  }, []);

  // Recenter map when metadata changes
  useEffect(() => {
    if (opticalPost?.latlon_bounds?.center) {
      setMapCenter(opticalPost.latlon_bounds.center);
      setOverlayBounds([
        [opticalPost.latlon_bounds.min_lat, opticalPost.latlon_bounds.min_lon],
        [opticalPost.latlon_bounds.max_lat, opticalPost.latlon_bounds.max_lon]
      ]);
    } else if (opticalPre?.latlon_bounds?.center) {
      setMapCenter(opticalPre.latlon_bounds.center);
      setOverlayBounds([
        [opticalPre.latlon_bounds.min_lat, opticalPre.latlon_bounds.min_lon],
        [opticalPre.latlon_bounds.max_lat, opticalPre.latlon_bounds.max_lon]
      ]);
    }
  }, [opticalPre, opticalPost]);

  // Canvas Grid Renderer to Base64 Overlay
  useEffect(() => {
    if (!analysis || !analysis.visual_evidence || !opticalPost) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const imgData = ctx.createImageData(256, 256);
    
    // Render layers
    if (selectedLayer === 'flood_mask' && analysis.visual_evidence.optical_water_mask) {
      const mask = analysis.visual_evidence.optical_water_mask;
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (mask[i][j] === 1) {
            // Blue for water
            imgData.data[idx] = 0;     // R
            imgData.data[idx+1] = 120; // G
            imgData.data[idx+2] = 255; // B
            imgData.data[idx+3] = 220; // A
          } else {
            // Transparent
            imgData.data[idx+3] = 0;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    } 
    else if (selectedLayer === 'sar_water_mask' && analysis.visual_evidence.sar_water_mask) {
      const mask = analysis.visual_evidence.sar_water_mask;
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (mask[i][j] === 1) {
            // Cyan for SAR water
            imgData.data[idx] = 0;     // R
            imgData.data[idx+1] = 220; // G
            imgData.data[idx+2] = 220; // B
            imgData.data[idx+3] = 220; // A
          } else {
            imgData.data[idx+3] = 0;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    }
    else if (selectedLayer === 'change_mask' && analysis.visual_evidence.flooded_mask) {
      const flooded = analysis.visual_evidence.flooded_mask;
      const receded = analysis.visual_evidence.receded_mask;
      
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (flooded && flooded[i][j] === 1) {
            // Red for new flood expansion
            imgData.data[idx] = 248;   // R
            imgData.data[idx+1] = 81;  // G
            imgData.data[idx+2] = 73;  // B
            imgData.data[idx+3] = 230; // A
          } else if (receded && receded[i][j] === 1) {
            // Green for receded water
            imgData.data[idx] = 46;    // R
            imgData.data[idx+1] = 164; // G
            imgData.data[idx+2] = 79;  // B
            imgData.data[idx+3] = 230; // A
          } else {
            imgData.data[idx+3] = 0;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    }
    else if (selectedLayer === 'lulc_map' && analysis.visual_evidence.lulc_grid) {
      const lulc = analysis.visual_evidence.lulc_grid;
      // EuroSAT class colors:
      // Agri (0, 5, 6): Greenish yellow
      // Forest (1, 2): Dark Green
      // Urban (3, 4, 7): Orange/Red
      // Water (8, 9): Dark Blue
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          const val = lulc[i][j];
          
          if ([0, 5, 6].includes(val)) { // Agriculture
            imgData.data[idx] = 180;   // R
            imgData.data[idx+1] = 210; // G
            imgData.data[idx+2] = 100; // B
            imgData.data[idx+3] = 180; // A
          } else if ([1, 2].includes(val)) { // Forest/Veg
            imgData.data[idx] = 34;    // R
            imgData.data[idx+1] = 139; // G
            imgData.data[idx+2] = 34;  // B
            imgData.data[idx+3] = 180; // A
          } else if ([3, 4, 7].includes(val)) { // Urban
            imgData.data[idx] = 218;   // R
            imgData.data[idx+1] = 118; // G
            imgData.data[idx+2] = 11;  // B
            imgData.data[idx+3] = 180; // A
          } else if ([8, 9].includes(val)) { // Water
            imgData.data[idx] = 12;    // R
            imgData.data[idx+1] = 45;  // G
            imgData.data[idx+2] = 90;  // B
            imgData.data[idx+3] = 180; // A
          } else {
            imgData.data[idx] = 100;   // Gray
            imgData.data[idx+1] = 100;
            imgData.data[idx+2] = 100;
            imgData.data[idx+3] = 120;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    } else {
      setActiveOverlayUrl(null);
    }
  }, [selectedLayer, analysis, opticalPost]);

  // Load preset demo files (optical_pre, optical_post, sar_post)
  const handleLoadDemo = async () => {
    setLoading(true);
    setMessages(prev => [...prev, {
      sender: 'system',
      text: "🔄 Contacting backend to ingest Indian flood demo GeoTIFFs (Assam Valley, UTM Zone 45N)...",
      timestamp: new Date().toLocaleTimeString()
    }]);
    
    try {
      // Direct file loads from local backend folders
      const loadDemoFile = async (filename: string, modality: string) => {
        // Fetch files generated on local system and post them
        const response = await fetch(`/api/demo/${filename}`);
        const blob = await response.blob();
        const file = new File([blob], filename, { type: "image/tiff" });
        
        const formData = new FormData();
        formData.append("file", file);
        formData.append("modality", modality);
        formData.append("session_id", sessionId);
        
        const uploadRes = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        return uploadRes.json();
      };
      
      const resPre = await loadDemoFile("optical_pre.tif", "optical_pre");
      setOpticalPre(resPre.metadata);
      
      const resPost = await loadDemoFile("optical_post.tif", "optical_post");
      setOpticalPost(resPost.metadata);
      
      const resSar = await loadDemoFile("sar_post.tif", "sar_post");
      setSarPost(resSar.metadata);
      
      setMessages(prev => [...prev, {
        sender: 'system',
        text: "✅ Loaded Indian Flood demo files: pre-event optical, post-event optical (with cloud cover), and post-event SAR. Telemetry checks validated CRS and geotransforms match. You can now execute flood change queries.",
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Error loading demo files: ${err.message}. Please check if backend is running.`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Upload handler
  const triggerUpload = (modality: string) => {
    setUploadingModality(modality);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !uploadingModality) return;
    
    setLoading(true);
    setMessages(prev => [...prev, {
      sender: 'system',
      text: `Uploading and extracting metadata for ${file.name}...`,
      timestamp: new Date().toLocaleTimeString()
    }]);
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("modality", uploadingModality);
    formData.append("session_id", sessionId);
    
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      
      if (uploadingModality === 'optical_pre') setOpticalPre(data.metadata);
      if (uploadingModality === 'optical_post') setOpticalPost(data.metadata);
      if (uploadingModality === 'sar_post') setSarPost(data.metadata);
      
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `✅ Uploaded ${uploadingModality.replace('_', ' ')}: ${data.metadata.filename}. CRS is ${data.metadata.crs}, shape ${data.metadata.width}x{data.metadata.height}.`,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
    } catch (err: any) {
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Error uploading file: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
      setUploadingModality(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Query Submit
  const handleQuery = async (presetQuery?: string) => {
    const textToSend = presetQuery || query;
    if (!textToSend.trim()) return;
    
    setMessages(prev => [...prev, {
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString()
    }]);
    
    if (!presetQuery) setQuery('');
    setLoading(true);
    
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: textToSend,
          session_id: sessionId
        })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to execute query");
      
      setAnalysis(data);
      
      // Add response message
      setMessages(prev => [...prev, {
        sender: 'system',
        text: data.answer,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      // Set default map view layer based on task
      if (data.task_classified === 'CHANGE_ANALYSIS') {
        setSelectedLayer('change_mask');
      } else if (data.task_classified === 'OPTICAL_SAR') {
        setSelectedLayer('sar_water_mask');
      } else if (data.task_classified === 'LAND_COVER_IMPACT') {
        setSelectedLayer('lulc_map');
      } else {
        setSelectedLayer('flood_mask');
      }
      
    } catch (err: any) {
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Query analysis failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Download Report
  const handleDownloadReport = () => {
    window.open(`/api/report/${sessionId}`, '_blank');
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0d0f12] text-slate-100">
      
      {/* Hidden file input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
        accept=".tif,.tiff" 
      />

      {/* Main Grid: Left Uploads Panel (1/4) | Map Center (2/4) | Chat/Stats Right (1/4) */}
      <div className="flex h-full w-full">
        
        {/* Left Side: Uploads & Telemetry */}
        <div className="flex w-[320px] flex-col border-r border-[#30363d] bg-[#161b22] p-4 text-sm overflow-y-auto">
          <div className="mb-4 flex items-center justify-between">
            <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
              🛰️ SatQuery AI
            </h1>
            <span className="rounded bg-brand-darkBlue px-1.5 py-0.5 text-xs text-brand-cyan border border-brand-blue/30">
              PS 26167
            </span>
          </div>
          
          <p className="text-xs text-slate-400 mb-4 leading-relaxed">
            Multimodal remote-sensing intelligence specialized for flood investigation.
          </p>

          <button 
            onClick={handleLoadDemo}
            disabled={loading}
            className="mb-6 flex w-full items-center justify-center gap-2 rounded-md bg-brand-blue/20 border border-brand-blue py-2.5 px-4 text-xs font-semibold text-brand-cyan hover:bg-brand-blue/30 transition disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Load Indian Flood Demo Data
          </button>

          {/* Upload Cards */}
          <div className="space-y-4 mb-6">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Layers className="h-3 w-3 text-brand-cyan" /> Imagery Inputs
            </h2>
            
            {/* Optical Pre */}
            <div className="rounded-md border border-[#30363d] bg-[#0d0f12] p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-300">OPTICAL PRE-EVENT</span>
                {opticalPre ? (
                  <span className="text-[10px] text-green-400 bg-green-500/10 px-1 border border-green-500/20 rounded">Active</span>
                ) : (
                  <span className="text-[10px] text-slate-500">Empty</span>
                )}
              </div>
              {opticalPre ? (
                <div className="text-[11px] text-slate-400 space-y-1">
                  <div className="truncate font-semibold text-slate-300">{opticalPre.filename}</div>
                  <div>Dim: {opticalPre.width}x{opticalPre.height} ({opticalPre.bands} bands)</div>
                  <div>CRS: {opticalPre.crs}</div>
                </div>
              ) : (
                <button 
                  onClick={() => triggerUpload('optical_pre')}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-1 border border-dashed border-slate-700 hover:border-slate-500 rounded py-2 text-xs text-slate-400"
                >
                  <Upload className="h-3 w-3" /> Upload Optical Pre
                </button>
              )}
            </div>

            {/* Optical Post */}
            <div className="rounded-md border border-[#30363d] bg-[#0d0f12] p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-300">OPTICAL POST-EVENT</span>
                {opticalPost ? (
                  <span className="text-[10px] text-green-400 bg-green-500/10 px-1 border border-green-500/20 rounded">Active</span>
                ) : (
                  <span className="text-[10px] text-slate-500">Empty</span>
                )}
              </div>
              {opticalPost ? (
                <div className="text-[11px] text-slate-400 space-y-1">
                  <div className="truncate font-semibold text-slate-300">{opticalPost.filename}</div>
                  <div>Dim: {opticalPost.width}x{opticalPost.height} ({opticalPost.bands} bands)</div>
                  <div>CRS: {opticalPost.crs}</div>
                  <div className="text-brand-orange flex items-center gap-1 text-[10px] mt-1 bg-brand-orange/10 p-1 rounded border border-brand-orange/20">
                    <AlertTriangle className="h-3 w-3 shrink-0" /> Clouds in top-right
                  </div>
                </div>
              ) : (
                <button 
                  onClick={() => triggerUpload('optical_post')}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-1 border border-dashed border-slate-700 hover:border-slate-500 rounded py-2 text-xs text-slate-400"
                >
                  <Upload className="h-3 w-3" /> Upload Optical Post
                </button>
              )}
            </div>

            {/* SAR Post */}
            <div className="rounded-md border border-[#30363d] bg-[#0d0f12] p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-300">SAR POST-EVENT</span>
                {sarPost ? (
                  <span className="text-[10px] text-green-400 bg-green-500/10 px-1 border border-green-500/20 rounded">Active</span>
                ) : (
                  <span className="text-[10px] text-slate-500">Empty</span>
                )}
              </div>
              {sarPost ? (
                <div className="text-[11px] text-slate-400 space-y-1">
                  <div className="truncate font-semibold text-slate-300">{sarPost.filename}</div>
                  <div>Dim: {sarPost.width}x{sarPost.height} ({sarPost.bands} bands)</div>
                  <div>CRS: {sarPost.crs}</div>
                  <div className="text-brand-cyan flex items-center gap-1 text-[10px] mt-1 bg-brand-blue/10 p-1 rounded border border-brand-blue/20">
                    <Compass className="h-3 w-3 shrink-0" /> C-Band penetrative SAR
                  </div>
                </div>
              ) : (
                <button 
                  onClick={() => triggerUpload('sar_post')}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-1 border border-dashed border-slate-700 hover:border-slate-500 rounded py-2 text-xs text-slate-400"
                >
                  <Upload className="h-3 w-3" /> Upload SAR Post
                </button>
              )}
            </div>
          </div>

          {/* Model Registry display trigger */}
          <button 
            onClick={() => setShowRegistry(!showRegistry)}
            className="flex w-full items-center justify-between border-t border-[#30363d] pt-4 text-xs font-semibold text-slate-300 hover:text-white"
          >
            <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5 text-brand-cyan" /> Model Registry & Info</span>
            <ChevronRight className={`h-4 w-4 transform transition ${showRegistry ? 'rotate-90' : ''}`} />
          </button>

          {showRegistry && modelRegistry.length > 0 && (
            <div className="mt-3 space-y-3 bg-[#0d0f12] p-2.5 rounded border border-[#30363d] text-[10px]">
              {modelRegistry.map((m: any) => (
                <div key={m.id} className="border-b border-[#30363d] pb-2 last:border-b-0 last:pb-0">
                  <div className="font-bold text-slate-300">{m.name} <span className="text-[8px] text-brand-cyan font-normal bg-brand-cyan/10 border border-brand-cyan/20 px-1 rounded">v{m.version}</span></div>
                  <div className="text-slate-400 leading-snug mt-1">{m.description}</div>
                  <div className="text-slate-500 mt-0.5">Framework: {m.framework}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Center Panel: Map Component (Leaves remaining space) */}
        <div className="relative flex flex-1 flex-col h-full bg-[#090b0d]">
          {/* Map Layer Controls overlay */}
          <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-2 rounded bg-brand-bg/90 border border-brand-border p-3 shadow-lg max-w-[200px] text-xs">
            <div className="font-bold mb-1 flex items-center gap-1 text-white">
              <Layers className="h-3.5 w-3.5 text-brand-cyan animate-pulse" /> MAP GRID LAYERS
            </div>
            <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
              <input type="radio" name="layer" checked={selectedLayer === 'satellite'} onChange={() => setSelectedLayer('satellite')} />
              <span>Base Map (Satellite)</span>
            </label>
            {analysis?.visual_evidence?.optical_water_mask && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'flood_mask'} onChange={() => setSelectedLayer('flood_mask')} />
                <span className="text-brand-blue font-semibold">Post-Optical Mask</span>
              </label>
            )}
            {analysis?.visual_evidence?.sar_water_mask && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'sar_water_mask'} onChange={() => setSelectedLayer('sar_water_mask')} />
                <span className="text-brand-cyan font-semibold">Post-SAR Mask</span>
              </label>
            )}
            {analysis?.visual_evidence?.flooded_mask && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'change_mask'} onChange={() => setSelectedLayer('change_mask')} />
                <span className="text-brand-red font-semibold">Flood Change Map</span>
              </label>
            )}
            {analysis?.visual_evidence?.lulc_grid && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'lulc_map'} onChange={() => setSelectedLayer('lulc_map')} />
                <span className="text-brand-green font-semibold">Adapted LULC map</span>
              </label>
            )}
            
            {activeOverlayUrl && (
              <div className="mt-2 border-t border-[#30363d] pt-2">
                <div className="flex justify-between mb-1">
                  <span>Layer Opacity:</span>
                  <span>{opacity}%</span>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="100" 
                  value={opacity} 
                  onChange={(e) => setOpacity(parseInt(e.target.value))}
                  className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}
          </div>

          {/* Coordinate telemetry overlays */}
          <div className="absolute bottom-4 left-4 z-[1000] rounded hud-panel p-2 text-[10px] space-y-1 text-slate-300 font-mono">
            <div>TARGET CRS: {opticalPost ? opticalPost.crs : 'Waiting for telemetry...'}</div>
            {opticalPost && (
              <>
                <div>EXTENT MIN: {opticalPost.bounds.left.toFixed(2)}, {opticalPost.bounds.bottom.toFixed(2)}</div>
                <div>EXTENT MAX: {opticalPost.bounds.right.toFixed(2)}, {opticalPost.bounds.top.toFixed(2)}</div>
              </>
            )}
          </div>

          <MapContainer 
            center={mapCenter || [26.24, 93.02]} 
            zoom={12} 
            className="w-full h-full"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {activeOverlayUrl && overlayBounds && (
              <ImageOverlay
                url={activeOverlayUrl}
                bounds={overlayBounds}
                opacity={opacity / 100}
              />
            )}
            <MapRecenter center={mapCenter} />
          </MapContainer>
        </div>

        {/* Right Side: Chat Sidebar & Telemetry Stats */}
        <div className="flex w-[400px] flex-col border-l border-[#30363d] bg-[#161b22] h-full overflow-hidden">
          
          {/* Top Panel: Statistics & Evidence Summary */}
          {analysis && (
            <div className="p-4 border-b border-[#30363d] bg-[#0d0f12] text-xs shrink-0 max-h-[50%] overflow-y-auto">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold tracking-wide uppercase text-slate-400">Analysis Telemetry</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysis.confidence.includes('High') ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                  analysis.confidence.includes('Moderate') ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' :
                  'bg-red-500/10 text-red-400 border border-red-500/20'
                }`}>
                  Confidence: {analysis.confidence.split('-')[0]}
                </span>
              </div>
              
              {/* Evidence fusion status */}
              <div className="rounded p-2 bg-[#161b22] border border-[#30363d] mb-3">
                <div className="flex items-center gap-1.5 font-semibold text-white mb-1">
                  <ShieldAlert className="h-3.5 w-3.5 text-brand-cyan" /> 
                  Evidence Fusion: {analysis.evidence_fusion.agreement_status}
                </div>
                <div className="text-[10px] text-slate-400 space-y-1">
                  {analysis.evidence_fusion.optical_findings && <div>• Optical: {analysis.evidence_fusion.optical_findings}</div>}
                  {analysis.evidence_fusion.sar_findings && <div>• SAR: {analysis.evidence_fusion.sar_findings}</div>}
                </div>
              </div>

              {/* Spatial metrics */}
              {analysis.spatial_statistics && (
                <div className="space-y-2 mb-3">
                  <div className="font-bold text-slate-300">Geospatial Measurements:</div>
                  <table className="w-full text-[11px] text-slate-400 border border-[#30363d] rounded">
                    <tbody>
                      {analysis.spatial_statistics.flooded_area_km2 !== undefined && (
                        <tr className="border-b border-[#30363d] bg-[#161b22]/50">
                          <td className="p-1 font-semibold text-slate-300">Flooded Expansion</td>
                          <td className="p-1 font-bold text-brand-red">{analysis.spatial_statistics.flooded_area_km2.toFixed(3)} km²</td>
                        </tr>
                      )}
                      {analysis.spatial_statistics.stable_area_km2 !== undefined && (
                        <tr className="border-b border-[#30363d]">
                          <td className="p-1">Stable Water (Channel)</td>
                          <td className="p-1">{analysis.spatial_statistics.stable_area_km2.toFixed(3)} km²</td>
                        </tr>
                      )}
                      {analysis.spatial_statistics.landcover_intersection && (
                        <tr className="bg-[#161b22]/30">
                          <td className="p-1" colSpan={2}>
                            <div className="font-bold text-slate-300 mb-1">Inundated Land Cover:</div>
                            <div className="space-y-1 pl-2">
                              {Object.entries(analysis.spatial_statistics.landcover_intersection).map(([cat, data]: any) => (
                                <div key={cat} className="flex justify-between text-[10px]">
                                  <span>{cat}</span>
                                  <span className="font-semibold text-slate-200">{data.area_affected_km2.toFixed(3)} km²</span>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Action buttons */}
              <div className="flex gap-2">
                <button 
                  onClick={handleDownloadReport}
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-brand-blue border border-brand-cyan/20 hover:bg-brand-blue/80 transition rounded text-white font-bold"
                >
                  <Download className="h-3 w-3" /> Download Report
                </button>
              </div>
            </div>
          )}

          {/* Chat Messages Panel */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((m, idx) => (
              <div 
                key={idx} 
                className={`flex flex-col max-w-[90%] rounded-lg p-3 ${
                  m.sender === 'user' 
                    ? 'bg-brand-blue/15 border border-brand-blue/30 text-slate-100 self-end ml-auto' 
                    : 'bg-[#161b22] border border-[#30363d] text-slate-300 self-start mr-auto'
                }`}
              >
                <div className="text-[10px] text-slate-500 mb-1 self-start">{m.timestamp}</div>
                <div className="text-xs whitespace-pre-line leading-relaxed">{m.text}</div>
              </div>
            ))}
            {loading && (
              <div className="bg-[#161b22] border border-[#30363d] text-slate-300 self-start mr-auto rounded-lg p-3 flex items-center gap-2 text-xs">
                <RefreshCw className="h-4 w-4 animate-spin text-brand-cyan" />
                Orchestrating specialist models & analyzing coordinates...
              </div>
            )}
          </div>

          {/* Quick-Prompt Scenario Grid */}
          <div className="p-3 bg-[#0d0f12] border-t border-[#30363d] shrink-0 space-y-2">
            <div className="text-[10px] font-bold text-slate-400 tracking-wide uppercase flex items-center gap-1">
              <Info className="h-3 w-3 text-brand-cyan" /> SCENARIO INVESTIGATION SHORTCUTS
            </div>
            <div className="grid grid-cols-2 gap-1.5 text-[10px]">
              <button 
                onClick={() => handleQuery("Is there a water body in this image?")}
                disabled={loading || !opticalPost}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40"
              >
                Demo 1: Single-Image VQA
              </button>
              <button 
                onClick={() => handleQuery("Highlight the water body")}
                disabled={loading || !opticalPost}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40"
              >
                Demo 2: Flood Grounding
              </button>
              <button 
                onClick={() => handleQuery("What changed after the flood?")}
                disabled={loading || !opticalPre || !opticalPost}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40 col-span-2"
              >
                Demo 3: Bi-Temporal Flood Change
              </button>
              <button 
                onClick={() => handleQuery("Use the optical and SAR images together to identify inundated regions and calculate affected agriculture")}
                disabled={loading || !opticalPost || !sarPost}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40 col-span-2"
              >
                Demo 4: Optical + SAR Fusion & LULC
              </button>
            </div>
          </div>

          {/* User Input Bar */}
          <div className="p-3 border-t border-[#30363d] bg-[#161b22] shrink-0">
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                placeholder="Ask what happened, where, or what was affected..."
                className="flex-1 rounded border border-[#30363d] bg-[#0d0f12] px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:border-brand-blue focus:outline-none"
              />
              <button
                onClick={() => handleQuery()}
                className="rounded bg-brand-blue px-4 py-1.5 text-xs font-semibold text-white hover:bg-brand-blue/90"
              >
                Query
              </button>
            </div>
          </div>

        </div>

      </div>

      {/* Execution Trace overlay at bottom of Map if analysis is active */}
      {analysis && analysis.execution_trace && (
        <div className="absolute bottom-4 right-[420px] z-[1000] w-[350px] max-h-[220px] rounded bg-brand-bg/95 border border-brand-border p-3 shadow-lg flex flex-col text-[10px]">
          <div className="font-bold border-b border-[#30363d] pb-1.5 mb-1.5 text-slate-300 flex items-center justify-between">
            <span>✓ AUDITABLE EXECUTION TRACE</span>
            <span className="text-[8px] bg-brand-cyan/15 text-brand-cyan px-1 rounded uppercase tracking-wider">Secured</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-1.5 scrollbar-thin">
            {analysis.execution_trace.map((step) => (
              <div key={step.step_number} className="flex gap-1.5">
                <span className={`h-1.5 w-1.5 rounded-full mt-1 shrink-0 ${
                  step.status === 'SUCCESS' ? 'bg-green-500' :
                  step.status === 'WARNING' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <div className="leading-snug">
                  <div className="font-bold text-slate-200">{step.step_number}. {step.name}</div>
                  <div className="text-slate-400">{step.details}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

// Simple uuid generation fallback since we aren't importing crypto directly
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
