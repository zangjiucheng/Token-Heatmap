"""Operational sub-commands for the ``token-heatmap`` CLI.

Each module here registers one (or a small group of) sub-command(s) on the
top-level parser built in :mod:`llm_token_heatmap.cli`. They replace the old
``scripts/*.sh`` so ``token-heatmap`` is the single entry point:

- :mod:`.web`  → ``token-heatmap web build``  (build the frontend dist/)
- :mod:`.hpc`  → ``token-heatmap hpc {setup,run,serve}`` (Slurm round-trip)

They use only the standard library, so importing them while building the parser
stays cheap (no torch / numpy pulled in for ``--help``).
"""
