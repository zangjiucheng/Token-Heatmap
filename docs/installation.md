# Installation

## Requirements

- Python 3.10+
- PyTorch (CUDA optional; CPU works for small models)
- Node.js 20+ — only needed to *run the Vite dev server* or *build the frontend*; not required on HPC/servers

## Option A — pip + venv (recommended for development)

```bash
git clone <repo-url> llm-token-heatmap
cd llm-token-heatmap

./scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` is idempotent and does all of the following:

1. Creates `.venv` if it doesn't already exist.
2. Installs the core package in editable mode (`pip install -e ".[dev,models]"`).
3. Runs `npm install` in `web/frontend` — but only if `npm` is on `PATH`. If you don't have Node, that step is skipped with a friendly message.

After it finishes, the `token-heatmap` CLI is on your `PATH` (see [`cli.md`](cli.md)).

## Option B — conda (HPC / shared environments)

```bash
conda env create -f environment.yml
conda activate token-heatmap
```

`environment.yml` installs PyTorch from the `pytorch` + `nvidia` channels (GPU-enabled by default), then installs `transformers`, `accelerate`, both Python packages, and all dev deps via pip.

For CPU-only machines, edit `environment.yml` and replace the pytorch line with:

```yaml
- pytorch::pytorch>=2.1
- pytorch::cpuonly
```

## Installing optional extras

| Extra | Command | When you need it |
|---|---|---|
| Model extras | `pip install ".[models]"` or `pip install tiktoken einops` | Some tokenizers / model families (tiktoken, einops) |
| Gated models (Llama, Gemma, …) | Set `HF_TOKEN=hf_...` (or `HUGGINGFACE_HUB_TOKEN`) | Models that require accepting a licence on HF Hub |

YAML config (`--config`) works out of the box — `pyyaml` is now a core dependency.

## Running the web app

The web app is a static, file-based viewer — there is no backend to run. Traces
load from a dropped JSON file, a `?trace=<url>` URL, or the bundled sample.

### Local machine (with Node.js)

```bash
source .venv/bin/activate     # or: conda activate token-heatmap
cd web/frontend && npm run dev    # http://localhost:5173
```

Then drop a trace JSON onto the page, or open with `?trace=<url>`. Override the
dev-server port with `VITE_DEV_PORT=5180 npm run dev`.

The easiest path lets the CLI produce a trace and open the viewer in one go:

```bash
token-heatmap trace --config configs/example.yaml --serve --frontend
```

| Service             | URL                        |
| ------------------- | -------------------------- |
| Viewer (Vite dev)   | http://localhost:5173      |

### HPC / server without Node.js

The `--serve` flag on `token-heatmap trace` starts a zero-dependency Python file server after generation — no uvicorn, no npm, no extra installs.

```bash
# Generate and immediately serve
token-heatmap trace --config configs/example.yaml --serve

# Custom port (useful when 8000 is taken)
token-heatmap trace --config configs/example.yaml --serve --port 9000

# Tell --serve which frontend port to put in the printed URL
token-heatmap trace --config configs/example.yaml \
  --serve --port 9000 --frontend-url http://localhost:3000
```

On your laptop, SSH port-forward the chosen port, then open the printed URL:

```bash
ssh -L 9000:localhost:9000 user@hpc
# open http://localhost:3000/?trace=http://localhost:9000/adaptive_token_trace.json
```

To host the viewer itself without Node.js on the host, build the static `dist/`
once on any machine with Node.js and serve it with any static file server:

1. Build the static viewer:
   ```bash
   token-heatmap web build          # output: web/frontend/dist/
   rsync -av web/frontend/dist/ user@hpc:…/web/frontend/dist/
   ```
2. On HPC, serve `dist/` with any static file server:
   ```bash
   python -m http.server -d web/frontend/dist 8080
   # open http://localhost:8080/?trace=<trace-url> (after SSH port-forward)
   ```

See [`web-app.md`](web-app.md) for a detailed breakdown.
