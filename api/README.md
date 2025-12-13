### Curvature Subspace Compression API (TinyTransformer v1)

This API serves **two models side-by-side**:
- **baseline**: the v1 `TinyTransformer` checkpoint (e.g. `tiny_dim320.pt`)
- **compressed**: the **keep-20%** curvature-subspace-compressed version of that checkpoint

Because the v1 `*_comp020.json` files store compression *stats* (not weights), the server can either:
- **Load a pre-exported compressed checkpoint** (fast startup), or
- **Build it lazily on first use** by re-collecting curvature stats **without weight updates**.

#### Files used (default)
- Baseline: `curvature_subspace_compression/artifacts/v1/tiny_dim320.pt`
- Keep-20 metadata: `curvature_subspace_compression/artifacts/v1/tiny_dim320_comp020.json`
- Cached compressed (optional): `curvature_subspace_compression/artifacts/v1/tiny_dim320_comp020.pt`

#### Run the server

```bash
python -m uvicorn curvature_subspace_compression.api.server:app --host 0.0.0.0 --port 8000
```

#### Optional: export the compressed checkpoint once (recommended)

This avoids paying curvature-collection cost at serving time.

```bash
python -m curvature_subspace_compression.api.export_compressed_checkpoint \
  --comp-json curvature_subspace_compression/artifacts/v1/tiny_dim320_comp020.json \
  --out curvature_subspace_compression/artifacts/v1/tiny_dim320_comp020.pt
```

#### Example requests

Health:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Generate from one model:

```bash
curl -s http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"baseline","prompt":"Once upon a time","max_new_tokens":64,"temperature":0.8,"top_k":50}' \
  | python -m json.tool
```

Compare baseline vs compressed:

```bash
curl -s http://localhost:8000/compare \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":64,"temperature":0.8,"top_k":50}' \
  | python -m json.tool
```


