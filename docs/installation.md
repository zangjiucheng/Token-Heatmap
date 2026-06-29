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
3. Runs `npm install` in `app` — but only if `npm` is on `PATH`. If you don't have Node, that step is skipped with a friendly message.

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

The web app is a static, file-based viewer — there is no backend to run. The CLI
just generates the trace to disk; you then open the JSON in the viewer:

```bash
token-heatmap trace --config configs/example.yaml   # writes outputs/example-run/
cd app && npm run dev                       # http://localhost:5173
# then drag outputs/example-run/adaptive_token_trace.json onto the page
```

Everything else — manual file drop, the bundled sample, hosting a prebuilt
`dist/`, and the native desktop app — is in [`web-app.md`](web-app.md).
