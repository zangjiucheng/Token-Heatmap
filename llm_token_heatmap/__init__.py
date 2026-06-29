"""LLM Token Heatmap.

A PyTorch component for analyzing LLM inference-time token probability
distributions, with adaptive top-k tracing, CSV export, and heatmap
visualization.
"""

from llm_token_heatmap.analysis.attention_stats import (
    AttentionDerivedStats,
    AttentionHeadStats,
    AttentionLayerAggregates,
    AttentionLayerDerivedStats,
    compute_attention_stats,
)
from llm_token_heatmap.analysis.diff import DIFF_SCHEMA_VERSION, compare_activations
from llm_token_heatmap.probes.activation_probe import (
    ActivationFullStats,
    ActivationLayerEntry,
    ActivationProbe,
    ActivationProbeConfig,
    ActivationProbeError,
    TopNeuron,
    tokenizer_fingerprint,
)
from llm_token_heatmap.probes.attention_probe import (
    AttentionProbe,
    AttentionProbeConfig,
    AttentionProbeError,
    AttentionStats,
)
from llm_token_heatmap.probes.logit_lens import (
    LogitLens,
    LogitLensConfig,
    LogitLensError,
    LogitLensLayerStats,
    LogitLensStats,
)
from llm_token_heatmap.serialize.attention_serializer import (
    attention_stats_to_payload,
    read_sidecar,
    write_sidecar,
)
from llm_token_heatmap.serialize.export import trace_to_dataframe
from llm_token_heatmap.tracing.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.tracing.generation import generate_with_adaptive_probe
from llm_token_heatmap.tracing.sampling import apply_sampling_filters, sample_next_token
from llm_token_heatmap.viz.plotting import (
    plot_activation_delta,
    plot_adaptive_heatmap,
    plot_attention_layer_head_grid,
    plot_attention_pattern,
    plot_entropy,
    plot_logit_lens,
    plot_logit_lens_selected_rank,
    plot_raw_vs_processed_heatmap,
    plot_raw_vs_processed_selected_prob,
    plot_selected_probability,
)

SCHEMA_VERSION = "2.0.0"

__all__ = [
    "ActivationFullStats",
    "ActivationLayerEntry",
    "ActivationProbe",
    "ActivationProbeConfig",
    "ActivationProbeError",
    "AdaptiveProbeConfig",
    "AdaptiveTokenProbe",
    "AttentionDerivedStats",
    "AttentionHeadStats",
    "AttentionLayerAggregates",
    "AttentionLayerDerivedStats",
    "AttentionProbe",
    "AttentionProbeConfig",
    "AttentionProbeError",
    "AttentionStats",
    "DIFF_SCHEMA_VERSION",
    "LogitLens",
    "LogitLensConfig",
    "LogitLensError",
    "LogitLensLayerStats",
    "LogitLensStats",
    "SCHEMA_VERSION",
    "TopNeuron",
    "apply_sampling_filters",
    "attention_stats_to_payload",
    "compare_activations",
    "compute_attention_stats",
    "generate_with_adaptive_probe",
    "plot_activation_delta",
    "plot_adaptive_heatmap",
    "plot_attention_layer_head_grid",
    "plot_attention_pattern",
    "plot_entropy",
    "plot_logit_lens",
    "plot_logit_lens_selected_rank",
    "plot_raw_vs_processed_heatmap",
    "plot_raw_vs_processed_selected_prob",
    "plot_selected_probability",
    "read_sidecar",
    "sample_next_token",
    "tokenizer_fingerprint",
    "trace_to_dataframe",
    "write_sidecar",
]

__version__ = "0.1.0"
