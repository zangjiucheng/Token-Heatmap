"""Heatmap and metric visualizations for adaptive token traces."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager, ft2font

from llm_token_heatmap.attention_stats import AttentionDerivedStats

CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "Noto Sans CJK KR",
    "Noto Sans CJK",
    "Source Han Sans SC",
    "Source Han Sans",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "PingFang TC",
    "PingFang HK",
    "Hiragino Sans GB",
    "Hiragino Sans",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Kaiti SC",
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "Yu Gothic",
    "Meiryo",
    "Malgun Gothic",
    "AppleGothic",
    "Arial Unicode MS",
)

CJK_CODEPOINT_RANGES = (
    (0x2E80, 0x2EFF),  # CJK radicals supplement
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x3130, 0x318F),  # Hangul compatibility jamo
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2A6DF),  # CJK unified ideographs extension B
    (0x2A700, 0x2B73F),  # CJK unified ideographs extension C
    (0x2B740, 0x2B81F),  # CJK unified ideographs extension D
    (0x2B820, 0x2CEAF),  # CJK unified ideographs extension E-F
)


def _contains_cjk(text: str) -> bool:
    return any(
        start <= ord(ch) <= end
        for ch in text
        for start, end in CJK_CODEPOINT_RANGES
    )


def _font_supports_text(font_path: str, text: str) -> bool:
    try:
        font = ft2font.FT2Font(font_path)
    except Exception:  # noqa: BLE001 — corrupt/inaccessible font; try next.
        return False
    return all(
        font.get_char_index(ord(ch)) != 0
        for ch in text
        if any(start <= ord(ch) <= end for start, end in CJK_CODEPOINT_RANGES)
    )


@lru_cache(maxsize=256)
def _cjk_font_for_text(text: str) -> font_manager.FontProperties | None:
    if not _contains_cjk(text):
        return None

    try:
        fonts = list(font_manager.fontManager.ttflist)
    except Exception:  # noqa: BLE001 — font cache rebuild can raise; degrade silently.
        return None

    for candidate in CJK_FONT_CANDIDATES:
        for font in fonts:
            if font.name == candidate and _font_supports_text(font.fname, text):
                return font_manager.FontProperties(fname=font.fname)

    # Last resort: use any installed font that covers the CJK glyphs in this
    # exact token. This is slower, but cached and only runs for CJK text.
    for font in fonts:
        if _font_supports_text(font.fname, text):
            return font_manager.FontProperties(fname=font.fname)

    return None


def _text_font_kwargs(text: str) -> dict[str, font_manager.FontProperties]:
    font_prop = _cjk_font_for_text(text)
    return {"fontproperties": font_prop} if font_prop is not None else {}


def _configure_cjk_fallback() -> None:
    """Append CJK-capable fonts (if any are installed) to matplotlib's sans-serif
    fallback list.

    Token strings often contain CJK / non-Latin glyphs (e.g. when the prompt
    or generation goes through a Qwen / Llama tokenizer). matplotlib's default
    DejaVu Sans has no CJK coverage, which produces a stream of
    ``UserWarning: Glyph ... missing from font(s) DejaVu Sans`` warnings and
    blank-box renderings.

    This global fallback helps normal labels. Token annotations additionally
    call ``_text_font_kwargs`` because matplotlib can still bind an individual
    text object to DejaVu Sans before fallback gets a chance to help.
    """
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001 — font cache rebuild can raise; degrade silently.
        return

    available = [name for name in CJK_FONT_CANDIDATES if name in installed]
    if not available:
        return

    current = list(matplotlib.rcParams.get("font.sans-serif", []))
    # Keep DejaVu Sans (or whatever's already first) as the primary; append
    # CJK fonts so matplotlib's fallback uses them for missing glyphs only.
    for name in available:
        if name not in current:
            current.append(name)
    matplotlib.rcParams["font.sans-serif"] = current


_configure_cjk_fallback()


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _filter_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "source" not in df.columns:
        return df
    return df[df["source"] == source]


def _grid_from_source(
    df: pd.DataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    steps = sorted(df["step"].unique())
    max_rank = int(df["rank"].max())

    grid = np.full((max_rank, len(steps)), np.nan, dtype=float)
    tokens = np.full((max_rank, len(steps)), "", dtype=object)

    step_index = {s: i for i, s in enumerate(steps)}
    for _, row in df.iterrows():
        col = step_index[row["step"]]
        rank = int(row["rank"]) - 1
        grid[rank, col] = row[value_col]
        tokens[rank, col] = str(row["token"])

    return grid, tokens, steps, max_rank


def plot_adaptive_heatmap(
    df: pd.DataFrame,
    value_col: str = "logprob",
    save_path: str | Path | None = None,
    annotate: bool = True,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    source: str = "raw",
) -> plt.Figure:
    """Render an adaptive token-probability heatmap.

    Args:
        df: DataFrame from `trace_to_dataframe`.
        value_col: Column to use for the color value (`prob` or `logprob`).
        save_path: Optional path to save the figure as PNG.
        annotate: If True, write the candidate token text inside each cell.
        cmap: Matplotlib colormap name.
        figsize: Optional explicit figure size; otherwise scaled from data shape.
        source: Which distribution to plot when the DataFrame has a ``source`` column.

    Returns:
        The matplotlib Figure.
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not in DataFrame columns")

    df = _filter_source(df, source)
    grid, tokens, steps, max_rank = _grid_from_source(df, value_col)

    if figsize is None:
        figsize = (max(6.0, 0.4 * len(steps)), max(4.0, 0.3 * max_rank))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper")

    ax.set_xlabel("Generation step")
    ax.set_ylabel("Adaptive token rank")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=0, fontsize=8)
    ax.set_yticks(np.arange(max_rank))
    ax.set_yticklabels([str(r + 1) for r in range(max_rank)], fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    if annotate:
        for r in range(max_rank):
            for c in range(len(steps)):
                token = tokens[r, c]
                if token:
                    text = token.replace("\n", "\\n")
                    ax.text(
                        c,
                        r,
                        text,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                        **_text_font_kwargs(text),
                    )

    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_selected_probability(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 3.5),
    source: str = "raw",
) -> plt.Figure:
    """Plot the per-step probability of the generated token."""
    df = _filter_source(df, source)
    per_step = df.drop_duplicates(subset=["step"]).sort_values("step")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(per_step["step"], per_step["selected_prob"], marker="o")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Selected token probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_entropy(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 3.5),
    source: str = "raw",
) -> plt.Figure:
    """Plot the entropy of the next-token distribution over generation steps."""
    df = _filter_source(df, source)
    per_step = df.drop_duplicates(subset=["step"]).sort_values("step")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(per_step["step"], per_step["entropy"], marker="o", color="tab:red")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_raw_vs_processed_heatmap(
    df: pd.DataFrame,
    value_col: str = "logprob",
    save_path: str | Path | None = None,
    annotate: bool = True,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render raw and sampling-processed heatmaps side by side with a shared color bar.

    Args:
        df: DataFrame from ``trace_to_dataframe`` containing a ``source`` column with
            both ``"raw"`` and ``"processed"`` rows.
        value_col: Column to use for the color value (``prob`` or ``logprob``).
        save_path: Optional path to save the figure as PNG.
        annotate: If True, write the candidate token text inside each cell.
        cmap: Matplotlib colormap name.
        figsize: Optional explicit figure size; otherwise scaled from data shape.

    Returns:
        The matplotlib Figure.
    """
    if "source" not in df.columns:
        raise ValueError("DataFrame must contain a 'source' column")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not in DataFrame columns")

    raw_df = df[df["source"] == "raw"]
    processed_df = df[df["source"] == "processed"]

    raw_grid, raw_tokens, raw_steps, raw_max_rank = _grid_from_source(raw_df, value_col)
    proc_grid, proc_tokens, proc_steps, proc_max_rank = _grid_from_source(processed_df, value_col)

    n_steps = max(len(raw_steps), len(proc_steps))
    max_rank = max(raw_max_rank, proc_max_rank)

    if figsize is None:
        figsize = (max(10.0, 0.8 * n_steps), max(4.0, 0.3 * max_rank))

    finite_values = np.concatenate(
        [raw_grid[np.isfinite(raw_grid)], proc_grid[np.isfinite(proc_grid)]]
    )
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, grid, tokens, steps, title in [
        (axes[0], raw_grid, raw_tokens, raw_steps, "raw"),
        (axes[1], proc_grid, proc_tokens, proc_steps, "processed"),
    ]:
        im = ax.imshow(
            grid,
            aspect="auto",
            cmap=cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Generation step")
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=0, fontsize=8)
        if annotate:
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    token = tokens[r, c]
                    if token:
                        text = token.replace("\n", "\\n")
                        ax.text(
                            c,
                            r,
                            text,
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white",
                            **_text_font_kwargs(text),
                        )

    axes[0].set_ylabel("Adaptive token rank")
    axes[0].set_yticks(np.arange(max_rank))
    axes[0].set_yticklabels([str(r + 1) for r in range(max_rank)], fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(value_col)

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def _decode_single(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except TypeError:
        decoded = tokenizer.decode(int(token_id))
    return decoded.replace("\n", "\\n")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _normalize_lens_layers(trace_step: dict[str, Any]) -> list[dict[str, Any]]:
    if "logit_lens" not in trace_step:
        raise ValueError("trace_step has no 'logit_lens' entry")
    layers = trace_step["logit_lens"]
    if not isinstance(layers, list):
        raise ValueError("trace_step['logit_lens'] must be a list of per-layer entries")
    return layers


def plot_logit_lens(
    trace_step: dict[str, Any],
    tokenizer: Any,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render a per-layer logit-lens top-k table for one generation step.

    Rows are layers (ordered from earliest to latest); columns are the top-k
    candidates per layer. Cell color encodes the candidate probability; cell
    text shows the decoded token string (NOT the raw token id).

    Args:
        trace_step: A single entry from a trace produced by
            ``generate_with_adaptive_probe`` with ``logit_lens`` attached.
        tokenizer: Tokenizer used to decode token ids into strings.
        save_path: Optional PNG output path.
        cmap: Matplotlib colormap name.
        figsize: Optional figure size override.

    Returns:
        The matplotlib Figure.
    """

    layers = _normalize_lens_layers(trace_step)
    if not layers:
        raise ValueError("logit_lens entry is empty")

    layer_indices = [int(layer["layer_idx"]) for layer in layers]
    top_k = max(len(_as_list(layer["top_k_token_ids"])) for layer in layers)

    grid = np.zeros((len(layers), top_k), dtype=float)
    token_texts = np.full((len(layers), top_k), "", dtype=object)

    for row, layer in enumerate(layers):
        ids = _as_list(layer["top_k_token_ids"])
        probs = _as_list(layer["top_k_probs"])
        for col, (token_id, prob) in enumerate(zip(ids, probs, strict=False)):
            grid[row, col] = float(prob)
            token_texts[row, col] = _decode_single(tokenizer, int(token_id))

    if figsize is None:
        figsize = (max(6.0, 0.9 * top_k), max(3.0, 0.35 * len(layers)))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)

    ax.set_xlabel("Top-k rank")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels([str(k + 1) for k in range(top_k)], fontsize=8)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(idx) for idx in layer_indices], fontsize=8)

    for r in range(len(layers)):
        for c in range(top_k):
            text = token_texts[r, c]
            if text:
                ax.text(
                    c,
                    r,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    **_text_font_kwargs(text),
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_logit_lens_selected_rank(
    trace: list[dict[str, Any]],
    tokenizer: Any,  # noqa: ARG001 — kept for API symmetry with plot_logit_lens
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis_r",
) -> plt.Figure:
    """Render selected-token rank by layer × step as a heatmap.

    Low ranks indicate the model has already "decided" on the selected token
    at that depth. By construction the final-layer row is all ones.

    Args:
        trace: List of step dicts with ``logit_lens`` entries.
        tokenizer: Tokenizer (unused; accepted for API symmetry).
        save_path: Optional PNG output path.
        figsize: Optional figure size override.
        cmap: Matplotlib colormap name (defaults to reversed so low rank = bright).

    Returns:
        The matplotlib Figure.
    """

    if not trace:
        raise ValueError("trace is empty")

    steps_with_lens = [entry for entry in trace if "logit_lens" in entry]
    if not steps_with_lens:
        raise ValueError("no steps in trace contain a 'logit_lens' entry")

    layer_indices: list[int] = []
    seen: set[int] = set()
    for entry in steps_with_lens:
        for layer in entry["logit_lens"]:
            idx = int(layer["layer_idx"])
            if idx not in seen:
                seen.add(idx)
                layer_indices.append(idx)
    layer_indices.sort()
    layer_position = {idx: pos for pos, idx in enumerate(layer_indices)}

    n_layers = len(layer_indices)
    n_steps = len(steps_with_lens)
    grid = np.full((n_layers, n_steps), np.nan, dtype=float)

    for col, entry in enumerate(steps_with_lens):
        for layer in entry["logit_lens"]:
            row = layer_position[int(layer["layer_idx"])]
            grid[row, col] = float(layer["selected_token_rank"])

    if figsize is None:
        figsize = (max(6.0, 0.4 * n_steps), max(3.0, 0.35 * n_layers))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper")

    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    step_labels = [str(int(entry.get("step", i))) for i, entry in enumerate(steps_with_lens)]
    ax.set_xticks(np.arange(n_steps))
    ax.set_xticklabels(step_labels, fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(idx) for idx in layer_indices], fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selected-token rank")

    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


_ATTENTION_HEAD_METRICS = {
    "entropy",
    "self_weight",
    "bos_weight",
    "q_norm",
    "k_norm",
    "v_norm",
    "qk_alignment_angle_deg",
    "effective_attention_span",
}


def _attention_grid(stats: AttentionDerivedStats, value: str) -> tuple[np.ndarray, list[int]]:
    if value not in _ATTENTION_HEAD_METRICS:
        raise ValueError(
            f"value '{value}' is not a supported per-head metric; "
            f"expected one of {sorted(_ATTENTION_HEAD_METRICS)}."
        )
    layer_indices = sorted(stats.layers.keys())
    if not layer_indices:
        return np.zeros((0, 0), dtype=float), layer_indices

    num_heads = stats.num_attention_heads
    grid = np.full((len(layer_indices), num_heads), np.nan, dtype=float)
    for row, layer_idx in enumerate(layer_indices):
        layer = stats.layers[layer_idx]
        for col, head in enumerate(layer.heads):
            grid[row, col] = float(getattr(head, value))
    return grid, layer_indices


def plot_attention_layer_head_grid(
    stats: AttentionDerivedStats,
    value: str = "entropy",
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render a (layer x head) heatmap colored by a per-head metric.

    Args:
        stats: Output of :func:`compute_attention_stats` for one step.
        value: Per-head metric name (e.g. ``"entropy"``, ``"self_weight"``).
        save_path: Optional PNG output path.
        cmap: Matplotlib colormap name.
        figsize: Optional figure size override.

    Returns:
        The matplotlib Figure. The colorbar is labelled with ``value``.
    """

    grid, layer_indices = _attention_grid(stats, value)

    if figsize is None:
        figsize = (
            max(6.0, 0.5 * max(1, stats.num_attention_heads)),
            max(3.0, 0.35 * max(1, len(layer_indices))),
        )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper")

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(stats.num_attention_heads))
    ax.set_xticklabels([str(h) for h in range(stats.num_attention_heads)], fontsize=8)
    ax.set_yticks(np.arange(len(layer_indices)))
    ax.set_yticklabels([str(idx) for idx in layer_indices], fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value)

    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_attention_pattern(
    stats: AttentionDerivedStats,
    layer: int,
    head: int,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 3.0),
) -> plt.Figure:
    """Plot a single head's attention pattern over previous source positions.

    Uses the ``top_k_positions`` captured for the requested (layer, head); the
    x-axis is source position and the y-axis is attention weight.
    """

    if layer not in stats.layers:
        raise ValueError(f"layer {layer} not present in stats; have {sorted(stats.layers)}.")
    layer_stats = stats.layers[layer]
    if head < 0 or head >= len(layer_stats.heads):
        raise ValueError(
            f"head {head} out of range for layer {layer} "
            f"(has {len(layer_stats.heads)} heads)."
        )

    head_stats = layer_stats.heads[head]
    positions = [p for p, _ in head_stats.top_k_positions]
    weights = [w for _, w in head_stats.top_k_positions]

    fig, ax = plt.subplots(figsize=figsize)
    if positions:
        ax.bar(positions, weights, color="tab:blue", width=0.8)
    ax.set_xlabel("Source position")
    ax.set_ylabel("Attention weight")
    ax.set_ylim(0.0, max(1.0, max(weights) if weights else 1.0))
    ax.set_title(f"Layer {layer}, head {head}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_activation_delta(
    diff: dict[str, Any],
    save_path: str | Path | None = None,
    metric: str = "l2",
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render a layer × step heatmap of per-(layer, submodule) activation deltas.

    Consumes the dict returned by ``compare_activations`` (matches
    ``docs/web/activation-diff.schema.json``). Each captured submodule gets
    its own stacked subplot, since the comparator can iterate over multiple
    submodules per step and packing them into a single grid would erase the
    submodule axis.

    Args:
        diff: Output of ``compare_activations``.
        save_path: Optional PNG output path.
        metric: Which per-(layer, submodule) field to colour by; one of
            ``"l2"`` or ``"cosine"``. The diff schema always populates both.
        cmap: Matplotlib colormap name.
        figsize: Optional figure size override.

    Returns:
        The matplotlib Figure. When ``diff`` has no aligned steps the figure
        is empty but still returned (the caller can decide whether to save).
    """

    if metric not in ("l2", "cosine"):
        raise ValueError(f"metric must be 'l2' or 'cosine'; got {metric!r}.")

    steps = diff.get("steps", []) or []
    submodules: list[str] = []
    seen_submodules: set[str] = set()
    layers_per_submodule: dict[str, list[int]] = {}
    for step in steps:
        for entry in step.get("delta", []):
            submodule = str(entry["submodule"])
            if submodule not in seen_submodules:
                seen_submodules.add(submodule)
                submodules.append(submodule)
                layers_per_submodule[submodule] = []
            layer = int(entry["layer"])
            if layer not in layers_per_submodule[submodule]:
                layers_per_submodule[submodule].append(layer)
    for submodule in submodules:
        layers_per_submodule[submodule].sort()

    n_submodules = max(1, len(submodules))
    n_steps = len(steps)

    if figsize is None:
        max_layers = max((len(layers_per_submodule[s]) for s in submodules), default=1)
        figsize = (
            max(6.0, 0.5 * max(1, n_steps)),
            max(3.0, 0.45 * max_layers * n_submodules + 0.5),
        )

    fig, axes = plt.subplots(
        n_submodules,
        1,
        figsize=figsize,
        squeeze=False,
    )

    for ax_row, submodule in zip(axes, submodules or [None], strict=False):
        ax = ax_row[0]
        if submodule is None or n_steps == 0:
            ax.set_axis_off()
            ax.set_title(f"submodule={submodule or '(no aligned steps)'} — empty")
            continue
        layer_indices = layers_per_submodule[submodule]
        layer_position = {idx: row for row, idx in enumerate(layer_indices)}
        grid = np.full((len(layer_indices), n_steps), np.nan, dtype=float)
        for col, step in enumerate(steps):
            for entry in step.get("delta", []):
                if str(entry["submodule"]) != submodule:
                    continue
                row = layer_position[int(entry["layer"])]
                grid[row, col] = float(entry[metric])
        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper")
        ax.set_title(f"submodule={submodule}")
        ax.set_xlabel("Aligned step")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(n_steps))
        ax.set_xticklabels([str(int(step["step"])) for step in steps], fontsize=8)
        ax.set_yticks(np.arange(len(layer_indices)))
        ax.set_yticklabels([str(idx) for idx in layer_indices], fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)

    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig


def plot_raw_vs_processed_selected_prob(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 3.5),
) -> plt.Figure:
    """Overlay the selected-token probability under the raw vs processed distributions.

    Args:
        df: DataFrame from ``trace_to_dataframe`` with a ``source`` column.
        save_path: Optional path to save the figure as PNG.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib Figure.
    """
    if "source" not in df.columns:
        raise ValueError("DataFrame must contain a 'source' column")

    fig, ax = plt.subplots(figsize=figsize)
    for source, color in [("raw", "tab:blue"), ("processed", "tab:orange")]:
        per_step = df[df["source"] == source].drop_duplicates(subset=["step"]).sort_values("step")
        ax.plot(
            per_step["step"],
            per_step["selected_prob"],
            marker="o",
            color=color,
            label=source,
        )
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Selected token probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        out = _ensure_parent(save_path)
        fig.savefig(out, dpi=150)

    return fig
