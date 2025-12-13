from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from curvature_subspace_compression.api.tiny_transformer_loader import TinyTransformerPair, generate_text


DEFAULT_BASE = Path("curvature_subspace_compression/artifacts/v1")
DEFAULT_MODEL = "tiny_dim320"


def _default_paths(model_name: str) -> tuple[Path, Path, Path]:
    comp_json = DEFAULT_BASE / f"{model_name}_comp020.json"
    baseline_ckpt = DEFAULT_BASE / f"{model_name}.pt"
    compressed_ckpt = DEFAULT_BASE / f"{model_name}_comp020.pt"  # may not exist yet
    return comp_json, baseline_ckpt, compressed_ckpt


comp_json_path, baseline_ckpt_path, compressed_ckpt_path = _default_paths(DEFAULT_MODEL)
pair = TinyTransformerPair(
    comp_json_path=comp_json_path,
    baseline_ckpt_path=baseline_ckpt_path,
    compressed_ckpt_path=compressed_ckpt_path,
    cache_dir=Path("curvature_subspace_compression/data"),
)

app = FastAPI(title="Curvature Subspace Compression API", version="0.1.0")


class GenerateRequest(BaseModel):
    model: Literal["baseline", "compressed"] = "baseline"
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=50000)


class GenerateResponse(BaseModel):
    model: Literal["baseline", "compressed"]
    text: str


class CompareResponse(BaseModel):
    baseline: str
    compressed: str


@app.get("/health")
def health() -> dict:
    # Load baseline eagerly; compressed stays lazy unless requested.
    try:
        pair.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok": True,
        "model_name": DEFAULT_MODEL,
        "baseline_ckpt": str(baseline_ckpt_path),
        "comp020_json": str(comp_json_path),
        "comp020_ckpt_cached": str(compressed_ckpt_path) if compressed_ckpt_path.exists() else None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    pair.load()
    tokenizer = pair.tokenizer
    if req.model == "baseline":
        model = pair.baseline
    else:
        model = pair.compressed()
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return GenerateResponse(model=req.model, text=text)


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=50000)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    pair.load()
    tokenizer = pair.tokenizer
    baseline_text = generate_text(
        model=pair.baseline,
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    compressed_text = generate_text(
        model=pair.compressed(),
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return CompareResponse(baseline=baseline_text, compressed=compressed_text)


