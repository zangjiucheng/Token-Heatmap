import { useEffect, useMemo, useRef, useState } from 'react';
import type { Trace, DirectLogitAttributionStep } from '@/types/trace';
import { VizModal } from '@/components/VizModal';
import { GraphFigure, type NodeKind, type PlacedNode } from './GraphFigure';
import './AttributionGraphTab.css';

export interface AttributionGraphTabProps {
  trace: Trace;
  selectedStep: number | null;
}

interface GNode {
  id: string;
  label: string;
  value: number;
  layer: number; // -1 for embed
  kind: NodeKind;
  head?: number;
}

const TOP_K = 16;
const DEFAULT_W = 900;

function pickStep(
  steps: DirectLogitAttributionStep[],
  selectedStep: number | null,
): DirectLogitAttributionStep {
  if (selectedStep != null) {
    const m = steps.find((s) => s.step === selectedStep);
    if (m) return m;
  }
  return steps[0];
}

/**
 * Attribution graph lens — renders the Direct Logit Attribution of the selected
 * step as a pruned, layer-ordered node-link graph: the target token (right) is
 * built from its top contributors (attention heads / MLP blocks / embedding),
 * positioned by depth, sized + coloured by signed contribution. Frontend-only:
 * it reuses the DLA data already in the trace.
 */
export function AttributionGraphTab({
  trace,
  selectedStep,
}: AttributionGraphTabProps) {
  const dla = trace.direct_logit_attribution;
  const [zoomOpen, setZoomOpen] = useState(false);

  // Lay out at the frame's real pixel width (not a fixed 920 scaled down), so
  // columns spread to fill, labels don't overlap, and text stays readable.
  const svgWrapRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(DEFAULT_W);
  useEffect(() => {
    const el = svgWrapRef.current;
    if (!el) return undefined;
    const measure = () =>
      setWidth(Math.max(320, Math.floor(el.getBoundingClientRect().width)));
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const view = useMemo(() => {
    const steps = dla?.steps ?? [];
    if (steps.length === 0) return null;
    const step = pickStep(steps, selectedStep);

    const nodes: GNode[] = [
      {
        id: 'embed',
        label: 'embed',
        value: step.embed ?? 0,
        layer: -1,
        kind: 'embed',
      },
    ];
    for (const l of step.layers) {
      if (l.heads && l.heads.length > 0) {
        for (const h of l.heads) {
          nodes.push({
            id: `L${l.layer}h${h.head}`,
            label: `L${l.layer}·h${h.head}`,
            value: h.attn,
            layer: l.layer,
            kind: 'head',
            head: h.head,
          });
        }
      } else {
        nodes.push({
          id: `L${l.layer}attn`,
          label: `L${l.layer}·attn`,
          value: l.attn,
          layer: l.layer,
          kind: 'attn',
        });
      }
      nodes.push({
        id: `L${l.layer}mlp`,
        label: `L${l.layer}·mlp`,
        value: l.mlp,
        layer: l.layer,
        kind: 'mlp',
      });
    }

    const totalN = nodes.length;
    const top = [...nodes]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, TOP_K);
    const error = step.error ?? 0;
    const maxAbs = Math.max(
      1e-9,
      ...top.map((n) => Math.abs(n.value)),
      Math.abs(error),
    );

    const layersSorted = Array.from(new Set(top.map((n) => n.layer))).sort(
      (a, b) => a - b,
    );
    const C = layersSorted.length;
    const leftM = 80;
    const rightM = 160;
    const COL_GAP = 190;
    const usableW = Math.max(0, width - leftM - rightM);
    // Cap how far the layer columns spread: a few layers should sit a readable
    // ~190px apart (not flung to opposite edges), while many layers still
    // expand to use the pane. The graph is left-anchored and the pane scrolls.
    const colSpan = C <= 1 ? 0 : Math.min(usableW, (C - 1) * COL_GAP);
    const colX = (layer: number) => {
      const idx = layersSorted.indexOf(layer);
      if (C <= 1) return leftM;
      return leftM + (idx / (C - 1)) * colSpan;
    };

    const byLayer = new Map<number, GNode[]>();
    for (const n of top) {
      const arr = byLayer.get(n.layer) ?? [];
      arr.push(n);
      byLayer.set(n.layer, arr);
    }
    for (const arr of byLayer.values()) {
      arr.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    }
    const maxCol = Math.max(1, ...[...byLayer.values()].map((a) => a.length));
    const rowH = 48;
    const topM = 44;
    const H = Math.max(220, maxCol * rowH + 2 * topM + 30);
    const cy = H / 2;

    const placed: PlacedNode[] = top.map((n) => {
      const arr = byLayer.get(n.layer) as GNode[];
      const i = arr.indexOf(n);
      const m = arr.length;
      const r = 7 + Math.sqrt(Math.abs(n.value) / maxAbs) * 13;
      return {
        ...n,
        x: colX(n.layer),
        y: cy + (i - (m - 1) / 2) * rowH,
        r,
      };
    });

    const output = {
      // Sit a fixed gap to the right of the last column instead of pinned to
      // the far edge, so the contributors → token flow reads tightly.
      x: leftM + colSpan + 150,
      y: cy,
      token:
        trace.steps.find((s) => s.step === step.step)?.selected.token ?? '',
    };

    // The drawing extends just past the output node (+ room for its label); the
    // SVG is sized to this, so it stays compact and the pane handles scrolling.
    const contentW = output.x + 160;

    return {
      step,
      placed,
      output,
      error,
      maxAbs,
      totalN,
      H,
      contentW,
      total: step.total_logit,
    };
  }, [dla, selectedStep, trace.steps, width]);

  if (!view) {
    return (
      <div
        className="graph-tab graph-tab--empty"
        data-testid="attribution-graph-tab-empty"
      >
        <p>
          No attribution data in this trace. Re-run the CLI with{' '}
          <code>--capture-full-activations</code> to build the attribution
          graph.
        </p>
      </div>
    );
  }

  const { placed, output, error, maxAbs, totalN, H, contentW, total } = view;

  return (
    <div className="graph-tab" data-testid="attribution-graph-tab-content">
      <header className="graph-tab__header">
        <div>
          <p className="eyebrow">Attribution graph</p>
          <h3 className="graph-tab__title">
            How{' '}
            <code className="graph-tab__token">
              {JSON.stringify(output.token)}
            </code>{' '}
            is built
          </h3>
        </div>
        <dl className="graph-tab__summary">
          <div>
            <dt>Logit</dt>
            <dd data-numeric>{total.toFixed(3)}</dd>
          </div>
        </dl>
      </header>

      <p className="graph-tab__note">
        Top {placed.length} of {totalN} contributors flow into the token
        (right). Node size and edge width track the contribution magnitude;
        orange promotes, blue suppresses. Columns are ordered by layer depth.
      </p>

      <div ref={svgWrapRef} className="graph-tab__svg-wrap">
        <button
          type="button"
          className="viz-expand-btn"
          onClick={() => setZoomOpen(true)}
          data-testid="graph-expand"
        >
          ⤢ Expand
        </button>
        <GraphFigure
          placed={placed}
          output={output}
          error={error}
          maxAbs={maxAbs}
          contentW={contentW}
          H={H}
        />
      </div>

      <p className="graph-tab__ablation-note">
        Interactive ablation returns via the CLI precomputing ablations into the
        trace.
      </p>

      <VizModal
        open={zoomOpen}
        onClose={() => setZoomOpen(false)}
        title={`How ${JSON.stringify(output.token)} is built`}
        aspect={contentW / H}
      >
        <GraphFigure
          placed={placed}
          output={output}
          error={error}
          maxAbs={maxAbs}
          contentW={contentW}
          H={H}
          testId="attribution-graph-svg-modal"
        />
      </VizModal>
    </div>
  );
}
