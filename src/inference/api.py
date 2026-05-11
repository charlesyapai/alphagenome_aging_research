"""
FastAPI backend for the webapp.

Endpoints:
    GET  /api/gallery               → precomputed predictions for featured variants
    GET  /api/predict/{rsid}        → live AlphaGenome call + classifier inference
    POST /api/predict               → {variant: "chr:pos:ref:alt"} → inference
    GET  /api/compare?a={rsid1}&b={rsid2} → side-by-side
    GET  /api/healthz               → health check

Run:
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inference.predict_variant import predict


app = FastAPI(title="AlphaGenome × Aging Variant Study",
              description="Inference API for aging-variant classification + "
                          "tissue-level regulatory effects.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "data" / "alphagenome_scores" / "aging"
API_KEY_FILE = ROOT / "env" / "api_key.txt"
GALLERY_FILE = ROOT / "webapp" / "gallery.json"

# Resolve API key from: (1) ALPHAGENOME_API_KEY env var [Hugging Face Space
# secret], or (2) local env/api_key.txt [local dev]. Never logged.
_api_key = os.environ.get("ALPHAGENOME_API_KEY")
if not _api_key and API_KEY_FILE.exists():
    _api_key = API_KEY_FILE.read_text().strip()


class VariantRequest(BaseModel):
    rsid: str | None = None
    variant: str | None = None


@app.get("/api/healthz")
def healthz():
    return {
        "status": "ok",
        "models_loaded": (MODELS_DIR / "binary_rf.joblib").exists(),
        "has_alphagenome_key": bool(_api_key),
    }


@app.get("/api/gallery")
def gallery():
    if GALLERY_FILE.exists():
        return json.loads(GALLERY_FILE.read_text())
    return {"variants": [], "note": "Gallery not yet built. Run build_gallery.py."}


@app.get("/api/predict/{rsid}")
def predict_rsid(rsid: str):
    if not (MODELS_DIR / "binary_rf.joblib").exists():
        raise HTTPException(503, "Models not yet trained")
    result = predict(rsid=rsid, api_key=_api_key,
                     models_dir=str(MODELS_DIR),
                     cache_scores_dir=str(CACHE_DIR))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/predict")
def predict_post(req: VariantRequest):
    if not (MODELS_DIR / "binary_rf.joblib").exists():
        raise HTTPException(503, "Models not yet trained")
    result = predict(rsid=req.rsid, variant_str=req.variant,
                     api_key=_api_key, models_dir=str(MODELS_DIR),
                     cache_scores_dir=str(CACHE_DIR))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/compare")
def compare(a: str, b: str):
    if not (MODELS_DIR / "binary_rf.joblib").exists():
        raise HTTPException(503, "Models not yet trained")
    result_a = predict(rsid=a, api_key=_api_key, models_dir=str(MODELS_DIR),
                       cache_scores_dir=str(CACHE_DIR))
    result_b = predict(rsid=b, api_key=_api_key, models_dir=str(MODELS_DIR),
                       cache_scores_dir=str(CACHE_DIR))
    if "error" in result_a:
        raise HTTPException(400, f"A: {result_a['error']}")
    if "error" in result_b:
        raise HTTPException(400, f"B: {result_b['error']}")

    # Delta = A - B per tissue per output type
    delta = {}
    for ot in result_a.get("tissue_heatmap", {}):
        delta[ot] = {}
        for tissue, va in result_a["tissue_heatmap"][ot].items():
            vb = result_b["tissue_heatmap"][ot].get(tissue)
            if va is None or vb is None:
                delta[ot][tissue] = None
            else:
                delta[ot][tissue] = float(va - vb)

    return {"a": result_a, "b": result_b, "delta": delta}


# Static webapp (served at /)
WEBAPP_DIR = ROOT / "webapp"
DOCS_DIR = ROOT / "docs"
if WEBAPP_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEBAPP_DIR / "static"),
              name="static")
    if DOCS_DIR.exists():
        app.mount("/docs", StaticFiles(directory=DOCS_DIR, html=True),
                  name="docs")

    @app.get("/")
    def root():
        idx = WEBAPP_DIR / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return {"message": "Webapp not built"}
