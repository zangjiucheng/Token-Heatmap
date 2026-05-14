# LLM Token Heatmap — Documentation

This is the project documentation index. The repository root [README](https://github.com/zangjiucheng/Token-Heatmap#readme)
has a short overview and quick start; the pages below cover each surface in depth.

| Page                                       | Topic                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| [`installation.md`](installation.md)       | Setting up the Python environment and (optionally) the web app               |
| [`cli.md`](cli.md)                         | The `token-heatmap` command-line interface                                   |
| [`python-api.md`](python-api.md)           | Using the library from Python: probe, generation, sampling, export, plotting |
| [`web-app.md`](web-app.md)                 | Running and using the FastAPI backend + React frontend                       |
| [`schema.md`](schema.md)                   | The on-disk trace JSON format and its attention/activation sidecars          |
| [`interpreting.md`](interpreting.md)       | What the recorded metrics mean and how to read the plots                     |
| [`troubleshooting.md`](troubleshooting.md) | Common issues and how to fix them                                            |

The canonical JSON Schemas are also under this folder and are imported by the
runtime — do not move them:

- [`web/trace.schema.json`](web/trace.schema.json) — full trace payload (loaded by the frontend validator and served by the backend's `GET /schema`)
- [`web/attention-sidecar.schema.json`](web/attention-sidecar.schema.json) — Tier-2 per-step attention sidecar
- [`web/activation.schema.json`](web/activation.schema.json) — projected activation trace payload
- [`web/activation-sidecar.schema.json`](web/activation-sidecar.schema.json) — Tier-2 per-step activation sidecar
- [`web/activation-diff.schema.json`](web/activation-diff.schema.json) — activation-diff payload written by `token-heatmap diff`

## Build the documentation site

The docs can be compiled into a static MkDocs site:

```bash
pip install -e ".[docs]"
mkdocs build
```

The generated HTML is written to `site/`.
