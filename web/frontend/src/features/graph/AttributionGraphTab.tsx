import { useEffect, useMemo, useRef, useState } from 'react';
import type { Trace, DirectLogitAttributionStep } from '@/types/trace';
import { InterventionPanel, type PresetTarget } from '@/features/dla';
import './AttributionGraphTab.css';

export interface AttributionGraphTabProps {
  trace: Trace;
  selectedStep: number | null;
}

type NodeKind = 'embed' | 'attn' | 'mlp' | 'head';

interface GNode {
  id: string;
  label: string;
  value: number;
  layer: number; // -1 for embed
  kind: NodeKind;
  head?: number;
}

interface PlacedNode extends GNode {
  x: number;
  y: number;
  r: number;
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

function edgePath(x1: number, y1: number, x2: number, y2: number): string {
  const mx = (x1 + x2) / 2;
  return `M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

function isAblatable(kind: NodeKind): boolean {
  return kind === 'attn' || kind === 'mlp' || kind === 'head';
}

/**
 * Attribution graph lens — renders the Direct Logit Attribution of the selected
 * step as a pruned, layer-ordered node-link graph: the target token (right) is
 * built from its top contributors (attention heads / MLP blocks / embedding),
 * positioned by depth, sized + coloured by signed contribution. Clicking a node
 * ablates that component (causal validation) in the panel below. Frontend-only:
 * it reuses the DLA data and the intervention engine.
 */
export function AttributionGraphTab({
  trace,
  selectedStep,
}: AttributionGraphTabProps) {
  const dla = trace.direct_logit_attribution;
  const [preset, setPreset] = useState<PresetTarget | null>(null);

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
    const usableW = width - leftM - rightM;
    const colX = (layer: number) => {
      const idx = layersSorted.indexOf(layer);
      if (C <= 1) return leftM + usableW / 2;
      return leftM + (idx / (C - 1)) * usableW;
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
      x: width - rightM / 2,
      y: cy,
      token:
        trace.steps.find((s) => s.step === step.step)?.selected.token ?? '',
    };

    return {
      step,
      placed,
      output,
      error,
      maxAbs,
      totalN,
      H,
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

  const { step, placed, output, error, maxAbs, totalN, H, total } = view;
  const outR = 24;

  const ablate = (n: PlacedNode) => {
    if (n.kind === 'head')
      setPreset({ layer: n.layer, component: 'head', head: n.head });
    else if (n.kind === 'attn')
      setPreset({ layer: n.layer, component: 'attn' });
    else if (n.kind === 'mlp') setPreset({ layer: n.layer, component: 'mlp' });
  };

  const edgeWidth = (v: number) => 1 + (Math.abs(v) / maxAbs) * 5;

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
        Click a node to ablate it and validate the edge below.
      </p>

      <div ref={svgWrapRef} className="graph-tab__svg-wrap">
        <svg
          className="graph-tab__svg"
          viewBox={`0 0 ${width} ${H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Attribution graph"
          data-testid="attribution-graph-svg"
        >
          {/* Edges */}
          <g className="graph-tab__edges">
            {placed.map((n) => (
              <path
                key={`e-${n.id}`}
                className="graph-tab__edge"
                data-sign={n.value >= 0 ? 'pos' : 'neg'}
                d={edgePath(n.x + n.r, n.y, output.x - outR, output.y)}
                style={{ strokeWidth: edgeWidth(n.value) }}
              />
            ))}
            {Math.abs(error) > 1e-6 ? (
              <path
                className="graph-tab__edge graph-tab__edge--error"
                d={edgePath(output.x, output.y + 70, output.x, output.y + outR)}
                style={{ strokeWidth: edgeWidth(error) }}
              />
            ) : null}
          </g>

          {/* Contributor nodes */}
          {placed.map((n) => {
            const ablatable = isAblatable(n.kind);
            const title = `${n.label}: ${n.value >= 0 ? '+' : ''}${n.value.toFixed(3)}`;
            return (
              <g
                key={n.id}
                className="graph-tab__node"
                data-ablatable={ablatable ? 'true' : 'false'}
                data-testid={`graph-node-${n.id}`}
                role={ablatable ? 'button' : undefined}
                tabIndex={ablatable ? 0 : undefined}
                aria-label={ablatable ? `Ablate ${n.label}` : n.label}
                onClick={ablatable ? () => ablate(n) : undefined}
                onKeyDown={
                  ablatable
                    ? (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          ablate(n);
                        }
                      }
                    : undefined
                }
              >
                <title>{title}</title>
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={n.r}
                  className="graph-tab__dot"
                  data-sign={n.value >= 0 ? 'pos' : 'neg'}
                  data-kind={n.kind}
                />
                <text
                  className="graph-tab__label"
                  x={n.x}
                  y={n.y + n.r + 13}
                  textAnchor="middle"
                >
                  {n.label}
                </text>
              </g>
            );
          })}

          {/* Error node */}
          {Math.abs(error) > 1e-6 ? (
            <g className="graph-tab__node">
              <title>{`unexplained (error): ${error.toFixed(3)}`}</title>
              <circle
                cx={output.x}
                cy={output.y + 70}
                r={9}
                className="graph-tab__dot graph-tab__dot--error"
              />
              <text
                className="graph-tab__label"
                x={output.x}
                y={output.y + 70 + 22}
                textAnchor="middle"
              >
                error
              </text>
            </g>
          ) : null}

          {/* Output (target token) */}
          <g className="graph-tab__output">
            <circle
              cx={output.x}
              cy={output.y}
              r={outR}
              className="graph-tab__output-dot"
            />
            <text
              className="graph-tab__output-label"
              x={output.x}
              y={output.y - outR - 8}
              textAnchor="middle"
            >
              {JSON.stringify(output.token)}
            </text>
          </g>
        </svg>
      </div>

      <InterventionPanel
        trace={trace}
        step={step}
        preset={preset}
        onPresetConsumed={() => setPreset(null)}
      />
    </div>
  );
}
