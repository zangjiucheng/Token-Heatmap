# Manifold reproduction study

Can this toolkit reproduce Anthropic's
[*"When Models Manipulate Manifolds: The Geometry of a Counting Task"*](https://transformer-circuits.pub/2025/linebreaks/index.html)
on an open model? This is the protocol we used, what we found on
**Qwen2.5‑7B‑Instruct**, and how to re‑run it on a bigger model.

## TL;DR — the verdict (7B)

| Claim from the paper | On Qwen2.5‑7B | Evidence |
| --- | --- | --- |
| The model **tracks line position / count** | ✅ **yes** | `line_position` linearly decodes from the residual stream at **CV R² ≈ 0.88** (mid layers), and it survives decorrelating content from column. |
| It encodes that count **on a helix** (periodic manifold) | ❌ **not detectably** | After removing the linear ramp, the circular coordinate is only weakly decodable (residual R² ≈ 0.25) and at a period equal to the line width (a single bend, not a repeating coil). |

So the 7B **counts** (linearly), but doesn't appear to use a **helix** to do it.
The paper's model is Claude 3.5 Haiku — far larger; helical/Fourier
representations may need that scale to emerge.

## The instruments (what the toolkit measures)

Run after generating a trace with `--capture-activations --capture-full-activations`:

```bash
token-heatmap manifold --trace <trace>.json --components 6 --probe line_position
```

Per `(layer, submodule)` cloud this writes into the trace JSON:

- **`probe.r2_cv`** — supervised linear probe: how well the scalar (e.g.
  `line_position` = chars since the last newline) decodes from the activations.
  Surfaces structure unsupervised PCA misses. (`llm_token_heatmap/probe.py`)
- **`probe.circular`** — the **helix test**: after projecting out the linear‑scalar
  direction, how decodable is the circular coordinate `cos/sin(2π·s/p)` at the
  best period `p`. High residual circular R² at an *interior* period (together
  with a high linear probe) is the helix signature; a plain ramp aliases onto a
  period≈range cosine and is rejected by the residualization.
- The unsupervised geometry (`participation_ratio`, TwoNN `intrinsic_dimension`,
  `trajectory_curvature`, `periodicity`) and the 2‑D/3‑D projection.

The web **Manifold tab** shows the 3‑D rotatable cloud (colour by step *or* by the
scalar), the probe R², and the Helix R².

## The study (three traces, increasing rigor)

The lesson is that **you must decorrelate the scalar from token content**, and
use a **long** trace, or you get a confounded or aliased "helix".

| Trace (`configs/…`) | content vs column | linear R² | helix |
| --- | --- | --- | --- |
| `linebreak.yaml` (count 1–80, ~3 lines) | decorrelated but **too short** | 0.83 (counting region) | artifacts (period 3 = token rhythm; period 29 = range) |
| `count-clean.yaml` (`0123456789` ×13) | **confounded** (column ≡ digit) | 0.99 | period‑8, residual R² 0.85–0.99 — but this is the **digit‑token manifold**, not counting |
| `wrap-text.yaml` (hard‑wrapped prose, ~40 lines) | **decorrelated + long** | **0.88** | residual R² ≈ 0.25 at period≈range → **no counting helix** |

Once content is decorrelated (`wrap-text`), the gorgeous period‑8 helix from the
repeating‑digit trace **vanishes** — confirming it was the digit manifold.

## Re‑run the protocol

```bash
# 1. generate a decorrelated, fixed-width, many-line trace (activations only)
CONFIG=configs/wrap-text.yaml OUT=outputs/wrap-text CAPTURE=activations \
  MANIFOLD_EXTRA="--components 6 --probe line_position" \
  sbatch --export=ALL,CONFIG,OUT,CAPTURE,MANIFOLD_EXTRA scripts/hpc-gen.slurm

# 2. read the per-layer linear + residual-circular table + verdict
python3 scripts/helix-report.py outputs/wrap-text/adaptive_token_trace.json

# 3. (optional) look at it: ./scripts/hpc-serve.sh outputs/wrap-text  → Manifold tab
```

Always sanity‑check `outputs/<run>/generated.txt`: the model must actually
produce many fixed‑width lines of **varied** content, or `line_position` is
confounded.

## Running a bigger model (GPU + Slurm)

### Step 0 — use the GPU env (built + verified ✅)

The default CLI (`/work/j7zang/.local/bin/token-heatmap`) runs everything on
**CPU**: its torch is `2.12.1+cu130` (CUDA 13.0) but the GPU nodes' driver is
**550.90.07 = CUDA 12.4**, too old, so `torch.cuda.is_available()` is `False`
(the log prints `device=cpu` + a "driver too old" warning). Tolerable for 7B at
short lengths (~6 min) but hopeless for anything bigger/longer.

A **dedicated GPU venv** is already set up so it can't disturb your other
research (`bridge-routing`, etc.): **`/work/j7zang/th-gpu`** with
`torch 2.6.0+cu124`. Verified on an L40S — `device=cuda`, real matmul works,
7B loads on GPU. **Use its CLI for GPU runs:**

```bash
BIN=/work/j7zang/th-gpu/bin/token-heatmap
```

How it was built (for reproduction / rebuilding):

```bash
/opt/uw/anaconda3/2025.06.1/bin/python3.13 -m venv /work/j7zang/th-gpu
source /work/j7zang/th-gpu/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e /work/j7zang/Token-Heatmap
# verify ON A GPU NODE (the login node has no GPU):
#   srun --account=normal --qos=normal --gres=gpu:l40s:1 --mem=8G --time=00:05:00 \
#     /work/j7zang/th-gpu/bin/python -c "import torch; print(torch.cuda.is_available())"
```

With CUDA live, generation is fast (model load ~1 min dominates), so longer
`wrap-text` traces and bigger models become practical.

### Slurm — RTX‑6000

Account `normal`, two relevant QOS:

| QOS | GPUs allowed | mem cap | walltime | note |
| --- | --- | --- | --- | --- |
| `qos_rtx6000_max` | rtx6000 | 200 G | 1 day | **gpu=1/user** — queues behind an existing rtx6000 job |
| `normal` | any (incl. l40s, rtx6000) | 30 G | 12 h | l40s nodes are usually idle |

```bash
# rtx6000 with headroom (queues if another rtx6000 job is running):
BIN=/work/j7zang/th-gpu/bin/token-heatmap \
CONFIG=configs/wrap-text.yaml OUT=outputs/wrap-14b CAPTURE=activations \
MANIFOLD_EXTRA="--components 6 --probe line_position" MODEL=Qwen/Qwen2.5-14B-Instruct \
  sbatch --account=normal --qos=qos_rtx6000_max --gres=gpu:rtx6000:1 --mem=64G --time=02:00:00 \
         --export=ALL,BIN,CONFIG,OUT,CAPTURE,MANIFOLD_EXTRA,MODEL scripts/hpc-gen.slurm
```

### Model size vs GPU (fp16)

| GPU | VRAM | fits (fp16) |
| --- | --- | --- |
| rtx6000 | 24 G | ≤ ~7B |
| l40s | 48 G | ≤ ~14B |

**14B+ does not fit an rtx6000 in fp16** — use an l40s (`--qos=normal
--gres=gpu:l40s:1 --mem=28G`, but that QOS caps mem at 30 G), or 4‑bit
quantization (needs a working bitsandbytes for the installed CUDA — only viable
once Step 0 gives a non‑cu130 torch). Larger models need a longer `wrap-text`
generation (raise `max_new_tokens`) so the helix test has enough line cycles.

Cached on the HPC (no download): Qwen2.5‑0.5B‑Instruct, 3B (base), **7B‑Instruct**,
Qwen3‑0.6B. Anything else downloads to `/work/j7zang/.cache/huggingface` (login
node has internet; compute nodes load from cache).

## Open gaps toward a full reproduction

- **Causal interventions** — steering / activation patching along the manifold
  (the part that proves the geometry is *used*). The toolkit only captures, it
  doesn't intervene.
- **Attention‑circuit analysis** — which heads "twist" the count manifold to
  estimate distance‑to‑boundary. Raw material (per‑head Q/K/V + weights) is
  captured; the analysis isn't built.
- **Bigger model** — re‑run this exact protocol on 14B+ (per above) and see
  whether the residual circular R² rises at an interior period (a real helix)
  rather than collapsing to the token rhythm.
