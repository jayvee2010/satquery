# Setup and Quick Start Guide - SatQuery AI

This guide will walk you through setting up and running the **SatQuery AI** Geospatial Workstation on your local machine.

---

## 1. Prerequisites
- **Python**: Version 3.10 to 3.12.
- **Node.js**: Version 18 or above, along with npm.
- **System Packages**: `gdal` (optional, as Python package compiles standard bindings).

---

## 2. Environment Configuration
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and set your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=your_actual_api_key_from_google_ai_studio
```

---

## 3. Installation

From the project root:

1. **Install Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Install Frontend Packages**:
   ```bash
   npm install && npm run install:frontend
   ```

---

## 4. Ingest and Train Models (One-Time Setup)

To demonstrate the **ISRO remote-sensing model adaptation** and generate the local data:

1. **Run Model Adaptation**:
   This downloads a subset of EuroSAT, trains the ResNet18 classifier for LULC, and saves the adapted weight file.
   ```bash
   python3 scripts/train_rs_model.py
   ```
2. **Generate Demo GeoTIFFs**:
   Generates spatially co-registered, multiband optical and SAR images located in the Assam Valley, India.
   ```bash
   python3 scripts/generate_demo_data.py
   ```

---

## 5. Run the Application
You can run the entire workstation (backend + frontend) using a single command:
```bash
npm start
```
- **Backend API**: Running on `http://localhost:8000` (docs available at `/docs`).
- **Frontend Workstation**: Running on `http://localhost:3000`.

Open your browser and navigate to `http://localhost:3000`.

---

## 6. Main API Endpoints

- **`POST /api/upload`**: Uploads and validates GeoTIFF images.
- **`POST /api/query`**: Submits a natural-language question.
- **`GET /api/report/{session_id}`**: Exports a markdown investigation report.
- **`GET /api/models`**: Queries the registry metadata.
