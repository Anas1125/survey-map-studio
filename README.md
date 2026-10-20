# Survey Map Studio

A local-first web application for uploading cadastral/village overview maps and individual survey sheets, creating review previews, and later cleaning annotations and assembling sheets into a larger map.

## Current implementation

- Upload one primary map and multiple individual survey sheets.
- Supported image formats: PNG, JPG/JPEG, TIFF, BMP and WEBP.
- 40 MB maximum per uploaded file.
- Live transfer progress bar during uploads.
- Background preview processing with a polled stage/progress/status bar and elapsed time; the UI reconnects to a running job after revisiting the page and retries only after a real job failure.
- Original files are stored separately and are never overwritten.
- Browser-safe PNG originals and thumbnails generated at ingestion, including TIFF files; uploaded bytes remain unchanged.
- A conservative OCR-guided candidate cleaner. When `pytesseract` and the Tesseract executable are installed, high-confidence decimal measurements are inpainted; ambiguous short numeric labels and candidate short lines are retained and listed for inspection.
- Generic color, connected-component, text-geometry and OCR passes support different text values, colors and orientations; detector results are cached by source/configuration. Value-specific examples are confined to regression tests.
- Review-only candidates for periodic dashed guides and box-shaped symbols; these are not automatically erased because cadastral maps need local validation.
- Versioned manual editing for brush/eraser/background strokes, repair/clone, line redraw, region selection, undo/redo, zoom, rotation and original/candidate comparison. Edits are applied to a separate candidate and clear prior approval.
- Approval-gated PNG/PDF downloads for cleaned candidates.
- Manual assembly canvas with draggable sheet overlays, scale/rotation controls, and persistent layout JSON.

**Important:** Automated cadastral interpretation is not validated across a representative 200-map ground-truth set. Inspect each candidate against its original, especially unresolved text, boundaries and symbols, before approval. OCR-assisted removal requires `pytesseract` plus the system Tesseract binary; without it the candidate retains annotations. Originals are preserved separately and are never edited.

## Requirements

- Python 3.10 or newer (Python 3.12 is a good default for broad package compatibility)
- Node.js 20 or newer
- npm
- Optional OCR: `pip install pytesseract` and install the Tesseract OCR executable for your operating system.

## Start on macOS/Linux

Open Terminal and enter these commands in order.

### 1. Start the API

```bash
cd survey-map-studio/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API health check: http://127.0.0.1:8000/api/health
Interactive API docs: http://127.0.0.1:8000/docs

### 2. Start the web app in a second Terminal tab/window

```bash
cd survey-map-studio/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

Cleaning runs return a job ID immediately and continue in FastAPI's background task pool. The status endpoint reports the active stage, elapsed time, per-stage timings, peak process memory and a user-facing error; full exceptions are logged server-side. Job and asset records are saved under `backend/data/jobs/` and `backend/data/asset_catalog.json`; interrupted jobs are restored as failed and can be retried. Detector OCR results are cached under `backend/data/ocr_cache/` using source metadata and detector version. Active jobs are deduplicated per source.

Full-resolution cleaners are bounded to one active job by default to avoid multiplying memory use. Set `SURVEY_MAP_CLEANING_WORKERS` to an explicitly measured safe worker count when running on hardware with enough memory; queued jobs remain visible in status.

## Cleaning and assembly workflow

Upload the primary reference map and individual sheets. Select a sheet and generate a candidate. Compare the original with the candidate, inspect uncertain labels and line/symbol proposals, or open **Edit candidate** for manual correction. Manual edits are versioned and invalidate approval. Inspect the exact final candidate, then choose **Approve Cleaned Sheet** to enable PNG/PDF downloads. The assembly workspace allows sheet layers to be dragged, scaled and rotated over the primary map; save the layout to persist transforms. Placement is manual and must be checked against real map evidence.

Run backend checks with `cd backend && python -m unittest discover -s tests -v`, syntax-check with `python -m compileall -q app tests`, and build the frontend with `cd frontend && npm run build`.

## Project roadmap

1. Upload, TIFF-compatible previews, conservative cleaning candidate, and approval-gated sheet exports.
2. Improve OCR/model quality and cadastral boundary QA on a representative labeled dataset.
3. Add control-point registration, seam diagnostics, full assembly preview and assembled-map exports.
4. Package the shared React UI and Python processing API as a desktop application.

## Local storage

Uploaded originals and generated preview images are stored in `backend/data/` by default. Set `SURVEY_MAP_DATA_DIR` to move them to another directory. The file catalogue and job state are persisted locally. Candidate PNGs, manual-edit manifests, project layouts, OCR cache and originals remain distinct artifacts under the data directory.
