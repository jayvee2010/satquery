import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, Upload, ShieldAlert, Cpu, 
  Download, RefreshCw, Compass, 
  AlertTriangle, ChevronRight, Info, CheckCircle2
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
  modality: string; // "OPTICAL", "SAR", "MULTISPECTRAL", "UNKNOWN"
  acquisition_date?: string;
  band_descriptions?: string[];
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
  input_configuration?: string;
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
  const [sessionId] = useState(() => `session_${uuidv4()}`);
  
  // Flexible Image Inputs
  const [image1, setImage1] = useState<ImageMetadata | null>(null);
  const [image2, setImage2] = useState<ImageMetadata | null>(null);
  const [inputConfig, setInputConfig] = useState<string>("EMPTY");
  const [detectedModalities, setDetectedModalities] = useState<string[]>([]);
  const [spatialAligned, setSpatialAligned] = useState<boolean>(false);
  
  // App UI states
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'system',
      text: "🛰️ Welcome to SatQuery AI (ISRO PS 26167). Upload a single satellite image (Optical or SAR) to begin VQA and grounding, or add a second observation for cross-modal fusion or temporal change detection.",
      timestamp: new Date().toLocaleTimeString()
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  
  // Map display settings
  const [selectedLayer, setSelectedLayer] = useState<string>('satellite');
  const [opacity, setOpacity] = useState<number>(75);
  const [mapCenter, setMapCenter] = useState<[number, number] | null>([26.24, 93.02]); // Assam, India
  const [activeOverlayUrl, setActiveOverlayUrl] = useState<string | null>(null);
  const [overlayBounds, setOverlayBounds] = useState<[[number, number], [number, number]] | null>(null);
  const [modelRegistry, setModelRegistry] = useState<any[]>([]);
  const [showRegistry, setShowRegistry] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadingSlot, setUploadingSlot] = useState<string | null>(null);

  // Fetch model registry on mount
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => setModelRegistry(data))
      .catch(err => console.error("Error fetching models:", err));
  }, []);

  // Update session configuration state
  const refreshSessionConfig = async () => {
    try {
      const res = await fetch(`/api/session/${sessionId}`);
      const data = await res.json();
      setInputConfig(data.configuration || "EMPTY");
      setDetectedModalities(data.detected_modalities || []);
      setSpatialAligned(data.spatial_alignment || false);
      if (data.image_1) setImage1(data.image_1);
      if (data.image_2) setImage2(data.image_2);
    } catch (e) {
      console.error("Failed to sync session config", e);
    }
  };

  // Recenter map when metadata changes
  useEffect(() => {
    const activeMeta = image1 || image2;
    if (activeMeta?.latlon_bounds?.center) {
      setMapCenter(activeMeta.latlon_bounds.center);
      setOverlayBounds([
        [activeMeta.latlon_bounds.min_lat, activeMeta.latlon_bounds.min_lon],
        [activeMeta.latlon_bounds.max_lat, activeMeta.latlon_bounds.max_lon]
      ]);
    }
  }, [image1, image2]);

  // Canvas Grid Renderer to Base64 Overlay
  useEffect(() => {
    if (!analysis || !analysis.visual_evidence) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const imgData = ctx.createImageData(256, 256);
    
    // 1. Optical Flood Mask (Blue)
    if (selectedLayer === 'flood_mask' && analysis.visual_evidence.optical_water_mask) {
      const mask = analysis.visual_evidence.optical_water_mask;
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (mask[i][j] === 1) {
            imgData.data[idx] = 0;
            imgData.data[idx+1] = 130;
            imgData.data[idx+2] = 255;
            imgData.data[idx+3] = 220;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    } 
    // 2. SAR Water Mask (Cyan)
    else if (selectedLayer === 'sar_water_mask' && analysis.visual_evidence.sar_water_mask) {
      const mask = analysis.visual_evidence.sar_water_mask;
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (mask[i][j] === 1) {
            imgData.data[idx] = 0;
            imgData.data[idx+1] = 230;
            imgData.data[idx+2] = 230;
            imgData.data[idx+3] = 220;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    }
    // 3. Bi-temporal Change Map (Red for new water, Blue for stable)
    else if (selectedLayer === 'change_mask' && analysis.visual_evidence.flooded_mask) {
      const flood = analysis.visual_evidence.flooded_mask;
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          if (flood[i][j] === 1) {
            imgData.data[idx] = 255;
            imgData.data[idx+1] = 50;
            imgData.data[idx+2] = 50;
            imgData.data[idx+3] = 230;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    }
    // 4. Adapted LULC Map
    else if (selectedLayer === 'lulc_map' && analysis.visual_evidence.lulc_grid) {
      const lulc = analysis.visual_evidence.lulc_grid;
      const colors: Record<number, [number, number, number]> = {
        0: [240, 220, 100], // Crops (Yellow)
        1: [34, 139, 34],   // Forest (Dark green)
        2: [124, 252, 0],   // Vegetation (Light green)
        3: [128, 128, 128], // Highway (Grey)
        4: [169, 169, 169], // Industrial
        5: [144, 238, 144], // Pasture
        6: [218, 165, 32],  // Perm Crop
        7: [220, 20, 60],   // Residential (Red-Pink)
        8: [30, 144, 255],  // River (Blue)
        9: [0, 0, 139]      // Sea
      };
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          const idx = (i * 256 + j) * 4;
          const cls = lulc[i][j];
          const rgb = colors[cls] || [200, 200, 200];
          imgData.data[idx] = rgb[0];
          imgData.data[idx+1] = rgb[1];
          imgData.data[idx+2] = rgb[2];
          imgData.data[idx+3] = 160;
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setActiveOverlayUrl(canvas.toDataURL());
    } else {
      setActiveOverlayUrl(null);
    }
  }, [selectedLayer, analysis]);

  // Load preset demo files
  const handleLoadDemo = async (presetType: 'cross_modal' | 'temporal' | 'single_optical' | 'single_sar') => {
    setLoading(true);
    setMessages(prev => [...prev, {
      sender: 'system',
      text: `🔄 Ingesting ${presetType.replace('_', ' ').toUpperCase()} demo observation(s)...`,
      timestamp: new Date().toLocaleTimeString()
    }]);
    
    try {
      const loadDemoFile = async (filename: string, slot: string) => {
        const response = await fetch(`/api/demo/${filename}`);
        const blob = await response.blob();
        const file = new File([blob], filename, { type: "image/tiff" });
        
        const formData = new FormData();
        formData.append("file", file);
        formData.append("slot", slot);
        formData.append("session_id", sessionId);
        
        const uploadRes = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        return uploadRes.json();
      };
      
      if (presetType === 'single_optical') {
        const r1 = await loadDemoFile("optical_post.tif", "image_1");
        setImage1(r1.metadata);
        setImage2(null);
      } else if (presetType === 'single_sar') {
        const r1 = await loadDemoFile("sar_post.tif", "image_1");
        setImage1(r1.metadata);
        setImage2(null);
      } else if (presetType === 'cross_modal') {
        const r1 = await loadDemoFile("optical_post.tif", "image_1");
        const r2 = await loadDemoFile("sar_post.tif", "image_2");
        setImage1(r1.metadata);
        setImage2(r2.metadata);
      } else if (presetType === 'temporal') {
        const r1 = await loadDemoFile("optical_pre.tif", "image_1");
        const r2 = await loadDemoFile("optical_post.tif", "image_2");
        setImage1(r1.metadata);
        setImage2(r2.metadata);
      }
      
      await refreshSessionConfig();
      
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `✅ Loaded ${presetType.replace('_', ' ').toUpperCase()} scenario data. Ready for investigation.`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Error loading demo data: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Upload handler
  const triggerUpload = (slot: string) => {
    setUploadingSlot(slot);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !uploadingSlot) return;
    
    setLoading(true);
    setMessages(prev => [...prev, {
      sender: 'system',
      text: `Uploading and detecting modality for ${file.name}...`,
      timestamp: new Date().toLocaleTimeString()
    }]);
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("slot", uploadingSlot);
    formData.append("session_id", sessionId);
    
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      
      if (uploadingSlot === 'image_1') setImage1(data.metadata);
      if (uploadingSlot === 'image_2') setImage2(data.metadata);
      
      await refreshSessionConfig();
      
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `✅ Ingested ${uploadingSlot.toUpperCase()}: ${data.metadata.filename} (Detected: ${data.metadata.modality}, ${data.metadata.bands} bands). Configuration: ${data.configuration?.configuration}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
    } catch (err: any) {
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Upload failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
      setUploadingSlot(null);
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
      
      setMessages(prev => [...prev, {
        sender: 'system',
        text: data.answer,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      // Auto-switch map layer
      if (data.task_classified === 'CHANGE_ANALYSIS') {
        setSelectedLayer('change_mask');
      } else if (data.task_classified === 'OPTICAL_SAR') {
        setSelectedLayer('sar_water_mask');
      } else if (data.task_classified === 'LAND_COVER_IMPACT') {
        setSelectedLayer('lulc_map');
      } else if (data.visual_evidence?.optical_water_mask) {
        setSelectedLayer('flood_mask');
      } else if (data.visual_evidence?.sar_water_mask) {
        setSelectedLayer('sar_water_mask');
      }
      
    } catch (err: any) {
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Investigation failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    window.open(`/api/report/${sessionId}`, '_blank');
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0d0f12] text-slate-100 font-sans">
      
      {/* Hidden file input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
        accept=".tif,.tiff" 
      />

      {/* Main Layout: Left Upload/Config Panel | Center Leaflet Map | Right Chat & Telemetry */}
      <div className="flex h-full w-full">
        
        {/* Left Side: Uploads & Telemetry */}
        <div className="flex w-[330px] flex-col border-r border-[#30363d] bg-[#161b22] p-4 text-sm overflow-y-auto shrink-0">
          <div className="mb-3 flex items-center justify-between">
            <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
              🛰️ SatQuery AI
            </h1>
            <span className="rounded bg-brand-darkBlue px-1.5 py-0.5 text-[10px] font-bold text-brand-cyan border border-brand-blue/30">
              ISRO PS 26167
            </span>
          </div>
          
          <p className="text-xs text-slate-400 mb-4 leading-relaxed">
            Agentic Vision-Language Assistant for Multimodal Remote Sensing.
          </p>

          {/* Quick Scenario Pre-sets */}
          <div className="mb-5 space-y-1.5">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
              <RefreshCw className="h-3 w-3 text-brand-cyan" /> Load Demo Presets
            </div>
            <div className="grid grid-cols-2 gap-1.5 text-[11px]">
              <button 
                onClick={() => handleLoadDemo('single_optical')}
                disabled={loading}
                className="p-1.5 rounded bg-[#0d0f12] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate font-medium text-left"
              >
                1. Single Optical
              </button>
              <button 
                onClick={() => handleLoadDemo('single_sar')}
                disabled={loading}
                className="p-1.5 rounded bg-[#0d0f12] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate font-medium text-left"
              >
                2. Single SAR
              </button>
              <button 
                onClick={() => handleLoadDemo('cross_modal')}
                disabled={loading}
                className="p-1.5 rounded bg-[#0d0f12] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate font-medium text-left col-span-2"
              >
                3. Optical + SAR Cross-Modal Pair
              </button>
              <button 
                onClick={() => handleLoadDemo('temporal')}
                disabled={loading}
                className="p-1.5 rounded bg-[#0d0f12] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate font-medium text-left col-span-2"
              >
                4. Pre + Post Bi-Temporal Pair
              </button>
            </div>
          </div>

          {/* Flexible Upload Cards */}
          <div className="space-y-3 mb-5">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Layers className="h-3.5 w-3.5 text-brand-cyan" /> Imagery Inputs
            </h2>
            
            {/* Image 1 Card (Required) */}
            <div className="rounded-md border border-[#30363d] bg-[#0d0f12] p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-bold text-xs text-slate-200">IMAGE 1 (PRIMARY)</span>
                {image1 ? (
                  <span className="text-[10px] text-green-400 bg-green-500/10 px-1.5 py-0.5 border border-green-500/20 rounded font-bold">
                    {image1.modality}
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 border border-amber-500/20 rounded">
                    Required
                  </span>
                )}
              </div>
              {image1 ? (
                <div className="text-[11px] text-slate-400 space-y-1 mt-2">
                  <div className="truncate font-semibold text-slate-200">{image1.filename}</div>
                  <div>Dim: {image1.width}x{image1.height} ({image1.bands} band{image1.bands > 1 ? 's' : ''})</div>
                  <div>CRS: {image1.crs}</div>
                </div>
              ) : (
                <button 
                  onClick={() => triggerUpload('image_1')}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-1.5 border border-dashed border-slate-700 hover:border-slate-500 rounded py-2 mt-2 text-xs text-slate-300 bg-[#161b22]/50 hover:bg-[#161b22]"
                >
                  <Upload className="h-3 w-3 text-brand-cyan" /> Upload Satellite GeoTIFF
                </button>
              )}
            </div>

            {/* Image 2 Card (Optional) */}
            <div className="rounded-md border border-[#30363d] bg-[#0d0f12] p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-bold text-xs text-slate-200">IMAGE 2 (COMPANION)</span>
                {image2 ? (
                  <span className="text-[10px] text-green-400 bg-green-500/10 px-1.5 py-0.5 border border-green-500/20 rounded font-bold">
                    {image2.modality}
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-500 bg-slate-800/50 px-1.5 py-0.5 border border-slate-700 rounded">
                    Optional
                  </span>
                )}
              </div>
              {image2 ? (
                <div className="text-[11px] text-slate-400 space-y-1 mt-2">
                  <div className="truncate font-semibold text-slate-200">{image2.filename}</div>
                  <div>Dim: {image2.width}x{image2.height} ({image2.bands} band{image2.bands > 1 ? 's' : ''})</div>
                  <div>CRS: {image2.crs}</div>
                </div>
              ) : (
                <button 
                  onClick={() => triggerUpload('image_2')}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-1.5 border border-dashed border-slate-700 hover:border-slate-500 rounded py-2 mt-2 text-xs text-slate-400 bg-[#161b22]/30 hover:bg-[#161b22]"
                >
                  <Upload className="h-3 w-3 text-slate-400" /> Upload 2nd Image (Optional)
                </button>
              )}
            </div>

            <div className="text-[10px] text-slate-500 leading-snug px-1">
              💡 <em>Single-image analysis works with Image 1 alone. Add Image 2 only for cross-modal (Optical+SAR) or bi-temporal change comparison.</em>
            </div>
          </div>

          {/* Model Registry trigger */}
          <button 
            onClick={() => setShowRegistry(!showRegistry)}
            className="flex w-full items-center justify-between border-t border-[#30363d] pt-3 text-xs font-semibold text-slate-300 hover:text-white"
          >
            <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5 text-brand-cyan" /> Model Registry & Adapters</span>
            <ChevronRight className={`h-4 w-4 transform transition ${showRegistry ? 'rotate-90' : ''}`} />
          </button>

          {showRegistry && modelRegistry.length > 0 && (
            <div className="mt-3 space-y-2.5 bg-[#0d0f12] p-2.5 rounded border border-[#30363d] text-[10px]">
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

        {/* Center Panel: Leaflet Map Component */}
        <div className="relative flex flex-1 flex-col h-full bg-[#090b0d]">
          {/* Map Layer Controls */}
          <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-1.5 rounded bg-brand-bg/90 border border-brand-border p-3 shadow-lg max-w-[210px] text-xs">
            <div className="font-bold mb-1 flex items-center gap-1.5 text-white">
              <Layers className="h-3.5 w-3.5 text-brand-cyan" /> MAP GRID LAYERS
            </div>
            <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
              <input type="radio" name="layer" checked={selectedLayer === 'satellite'} onChange={() => setSelectedLayer('satellite')} />
              <span>Base Map (Satellite)</span>
            </label>
            {analysis?.visual_evidence?.optical_water_mask && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'flood_mask'} onChange={() => setSelectedLayer('flood_mask')} />
                <span className="text-brand-blue font-semibold">Optical Water Mask</span>
              </label>
            )}
            {analysis?.visual_evidence?.sar_water_mask && (
              <label className="flex items-center gap-2 hover:bg-slate-800 p-1 rounded cursor-pointer">
                <input type="radio" name="layer" checked={selectedLayer === 'sar_water_mask'} onChange={() => setSelectedLayer('sar_water_mask')} />
                <span className="text-brand-cyan font-semibold">SAR Water Mask</span>
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
                <span className="text-brand-green font-semibold">Adapted LULC Map</span>
              </label>
            )}
            
            {activeOverlayUrl && (
              <div className="mt-2 border-t border-[#30363d] pt-2">
                <div className="flex justify-between mb-1 text-[11px]">
                  <span>Opacity:</span>
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

          {/* Input Status & Telemetry HUD */}
          <div className="absolute bottom-4 left-4 z-[1000] rounded hud-panel p-3 text-[11px] space-y-1 text-slate-300 font-mono border border-brand-border bg-brand-bg/95 shadow-lg">
            <div className="font-bold text-white mb-1 flex items-center gap-1.5">
              <Compass className="h-3.5 w-3.5 text-brand-cyan" /> WORKSPACE TELEMETRY
            </div>
            <div>CONFIG: <span className="text-brand-cyan font-bold">{inputConfig}</span></div>
            <div>MODALITIES: {detectedModalities.length > 0 ? detectedModalities.join(' + ') : 'None'}</div>
            <div>SPATIAL ALIGNMENT: {spatialAligned ? <span className="text-green-400 font-bold">YES</span> : <span className="text-slate-500">N/A</span>}</div>
            {image1 && (
              <div className="text-[10px] text-slate-400 mt-1 pt-1 border-t border-slate-700">
                CRS: {image1.crs} | Res: {image1.pixel_size_x.toFixed(1)}m
              </div>
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

        {/* Right Side: Chat & Telemetry */}
        <div className="flex w-[420px] flex-col border-l border-[#30363d] bg-[#161b22] h-full overflow-hidden shrink-0">
          
          {/* Top Panel: Statistics & Evidence Summary */}
          {analysis && (
            <div className="p-4 border-b border-[#30363d] bg-[#0d0f12] text-xs shrink-0 max-h-[45%] overflow-y-auto">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold tracking-wide uppercase text-slate-400">Analysis Telemetry</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysis.confidence.includes('High') ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                  analysis.confidence.includes('Moderate') ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' :
                  'bg-red-500/10 text-red-400 border border-red-500/20'
                }`}>
                  {analysis.confidence}
                </span>
              </div>
              
              {/* Evidence fusion status */}
              <div className="rounded p-2 bg-[#161b22] border border-[#30363d] mb-3">
                <div className="flex items-center gap-1.5 font-semibold text-white mb-1 text-[11px]">
                  <ShieldAlert className="h-3.5 w-3.5 text-brand-cyan" /> 
                  Evidence Status: {analysis.evidence_fusion.agreement_status}
                </div>
                <div className="text-[10px] text-slate-400 space-y-1">
                  {analysis.evidence_fusion.optical_findings && <div>• Optical: {analysis.evidence_fusion.optical_findings}</div>}
                  {analysis.evidence_fusion.sar_findings && <div>• SAR: {analysis.evidence_fusion.sar_findings}</div>}
                </div>
              </div>

              {/* Spatial metrics */}
              {analysis.spatial_statistics && (
                <div className="space-y-2 mb-3">
                  <div className="font-bold text-slate-300">Geospatial Metrics:</div>
                  <table className="w-full text-[11px] text-slate-400 border border-[#30363d] rounded">
                    <tbody>
                      {analysis.spatial_statistics.flooded_area_km2 !== undefined && (
                        <tr className="border-b border-[#30363d] bg-[#161b22]/50">
                          <td className="p-1 font-semibold text-slate-300">Flood Expansion</td>
                          <td className="p-1 font-bold text-brand-red">{analysis.spatial_statistics.flooded_area_km2.toFixed(3)} km²</td>
                        </tr>
                      )}
                      {analysis.spatial_statistics.stable_area_km2 !== undefined && (
                        <tr className="border-b border-[#30363d]">
                          <td className="p-1">Stable Water</td>
                          <td className="p-1">{analysis.spatial_statistics.stable_area_km2.toFixed(3)} km²</td>
                        </tr>
                      )}
                      {analysis.spatial_statistics.optical_water_percentage !== undefined && (
                        <tr className="border-b border-[#30363d]">
                          <td className="p-1">Optical Water Extent</td>
                          <td className="p-1 font-semibold text-brand-blue">{(analysis.spatial_statistics.optical_water_percentage * 100).toFixed(1)}%</td>
                        </tr>
                      )}
                      {analysis.spatial_statistics.sar_water_percentage !== undefined && (
                        <tr className="border-b border-[#30363d]">
                          <td className="p-1">SAR Water Extent</td>
                          <td className="p-1 font-semibold text-brand-cyan">{(analysis.spatial_statistics.sar_water_percentage * 100).toFixed(1)}%</td>
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
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-brand-blue border border-brand-cyan/20 hover:bg-brand-blue/80 transition rounded text-white font-bold text-xs"
                >
                  <Download className="h-3.5 w-3.5" /> Export Investigation Report
                </button>
              </div>
            </div>
          )}

          {/* Chat Messages Panel */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((m, idx) => (
              <div 
                key={idx} 
                className={`flex flex-col max-w-[92%] rounded-lg p-3 ${
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

          {/* Quick-Prompt Scenario Matrix */}
          <div className="p-3 bg-[#0d0f12] border-t border-[#30363d] shrink-0 space-y-2">
            <div className="text-[10px] font-bold text-slate-400 tracking-wide uppercase flex items-center gap-1">
              <Info className="h-3 w-3 text-brand-cyan" /> PS 26167 SCENARIO SHORTCUTS
            </div>
            <div className="grid grid-cols-2 gap-1.5 text-[10px]">
              <button 
                onClick={() => handleQuery("Is there water in this image?")}
                disabled={loading || !image1}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40"
              >
                1. Single-Image VQA
              </button>
              <button 
                onClick={() => handleQuery("Highlight the water body")}
                disabled={loading || !image1}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40"
              >
                2. Flood Grounding
              </button>
              <button 
                onClick={() => handleQuery("What changed after the flood?")}
                disabled={loading || !image1}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40 col-span-2"
              >
                3. Bi-Temporal Flood Change
              </button>
              <button 
                onClick={() => handleQuery("Use the optical and SAR images together to identify inundated regions and calculate affected agriculture")}
                disabled={loading || !image1}
                className="text-left p-1.5 rounded bg-[#161b22] hover:bg-slate-800 border border-[#30363d] text-slate-300 hover:text-white truncate disabled:opacity-40 col-span-2"
              >
                4. Optical + SAR Cross-Modal Fusion
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
                disabled={loading || !image1}
                className="rounded bg-brand-blue px-4 py-1.5 text-xs font-semibold text-white hover:bg-brand-blue/90 disabled:opacity-50"
              >
                Query
              </button>
            </div>
          </div>

        </div>

      </div>

      {/* Execution Trace HUD */}
      {analysis && analysis.execution_trace && (
        <div className="absolute bottom-4 right-[440px] z-[1000] w-[360px] max-h-[220px] rounded bg-brand-bg/95 border border-brand-border p-3 shadow-lg flex flex-col text-[10px]">
          <div className="font-bold border-b border-[#30363d] pb-1.5 mb-1.5 text-slate-300 flex items-center justify-between">
            <span>✓ AUDITABLE WORKFLOW TRACE</span>
            <span className="text-[8px] bg-brand-cyan/15 text-brand-cyan px-1 rounded uppercase tracking-wider font-bold">
              {analysis.input_configuration || 'Verified'}
            </span>
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

function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
