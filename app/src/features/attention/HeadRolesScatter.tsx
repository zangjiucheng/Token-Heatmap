import { useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import './HeadRolesScatter.css';

interface HeadRole {
  layer: number;
  head: number;
  role: string;
  dla_mean: number;
  dla_absmean: number;
  bos_weight: number;
  self_weight: number;
  induction: number;
}

interface HeadRoles {
  heads?: HeadRole[];
  summary?: { counts?: Record<string, number> };
}

export interface HeadRolesScatterProps {
  trace: Trace;
}

type XAxis = 'bos_weight' | 'self_weight' | 'induction';

const X_LABEL: Record<XAxis, string> = {
  bos_weight: 'BOS attention',
  self_weight: 'Self / local attention',
  induction: 'Induction score',
};

// Role palette — worker/induction (writers) pop; sinks are muted (they attend
// hard yet write ~nothing, which is the whole point).
const ROLE_COLOR: Record<string, string> = {
  worker: '#d97757',
  induction: '#b072d0',
  local: '#31688e',
  sink: '#6b7280',
  minor: '#8a8f98',
  other: '#4b5563',
};
const ROLE_ORDER = ['worker', 'induction', 'local', 'sink', 'minor', 'other'];

const W = 560;
const H = 300;
const PAD_L = 46;
const PAD_B = 34;
const PAD_T = 12;
const PAD_R = 128;

/**
 * Head roles — every attention head plotted as contribution (|direct logit
 * attribution|, y) against an attention-magnitude signal (x), coloured by its
 * fused role. The point: attention and contribution are *anti-correlated* —
 * "attention sinks" park on BOS (far right) yet write ~nothing (bottom), while
 * the load-bearing "workers" attend little but sit high. Uses trace.head_roles
 * (attention + DLA fusion already in the trace) — no new capture.
 */
export function HeadRolesScatter({ trace }: HeadRolesScatterProps) {
  const [xAxis, setXAxis] = useState<XAxis>('bos_weight');
  const [hover, setHover] = useState<HeadRole | null>(null);

  const hr = (trace as { head_roles?: HeadRoles }).head_roles;
  const heads = useMemo(() => hr?.heads ?? [], [hr]);
  const counts = hr?.summary?.counts ?? {};

  const maxY = useMemo(
    () => Math.max(0.01, ...heads.map((h) => h.dla_absmean)),
    [heads],
  );
  const maxX = useMemo(
    () => Math.max(0.01, ...heads.map((h) => h[xAxis])),
    [heads, xAxis],
  );

  if (heads.length === 0) return null;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xOf = (v: number) => PAD_L + (v / maxX) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v / maxY) * plotH;

  const rolesPresent = ROLE_ORDER.filter((r) => (counts[r] ?? 0) > 0);

  return (
    <section
      className="head-roles"
      data-testid="head-roles-scatter"
      aria-label="Head roles: attention vs contribution"
    >
      <div className="head-roles__head">
        <h3 className="head-roles__title">
          Attention <span>≠</span> contribution{' '}
          <span className="head-roles__subtitle">· per head</span>
        </h3>
        <div
          className="head-roles__toggle"
          role="group"
          aria-label="Attention axis"
        >
          {(Object.keys(X_LABEL) as XAxis[]).map((k) => (
            <button
              key={k}
              type="button"
              data-selected={xAxis === k}
              data-testid={`head-roles-x-${k}`}
              onClick={() => setXAxis(k)}
            >
              {k === 'bos_weight' ? 'BOS' : k === 'self_weight' ? 'Self' : 'Induction'}
            </button>
          ))}
        </div>
      </div>
      <p className="head-roles__hint">
        Each dot is a head: y = |direct logit attribution| (how much it writes to
        the answer), x = {X_LABEL[xAxis].toLowerCase()}. Sinks attend hard yet
        contribute ~nothing; workers write the answer.
      </p>

      <svg
        className="head-roles__svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Head contribution versus ${X_LABEL[xAxis]}, coloured by role`}
      >
        {/* axes */}
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} className="head-roles__axis" />
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={PAD_L + plotW}
          y2={PAD_T + plotH}
          className="head-roles__axis"
        />
        <text
          className="head-roles__axis-title"
          transform={`translate(12 ${PAD_T + plotH / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          |DLA|
        </text>
        <text
          className="head-roles__axis-title"
          x={PAD_L + plotW / 2}
          y={H - 4}
          textAnchor="middle"
        >
          {X_LABEL[xAxis]}
        </text>
        {/* y ticks */}
        <text x={PAD_L - 6} y={PAD_T + 4} className="head-roles__tick" textAnchor="end">
          {maxY.toFixed(2)}
        </text>
        <text x={PAD_L - 6} y={PAD_T + plotH + 3} className="head-roles__tick" textAnchor="end">
          0
        </text>
        {/* x ticks */}
        <text x={PAD_L + plotW} y={PAD_T + plotH + 16} className="head-roles__tick" textAnchor="end">
          {maxX.toFixed(2)}
        </text>

        {/* points */}
        {heads.map((h) => (
          <circle
            key={`${h.layer}-${h.head}`}
            cx={xOf(h[xAxis])}
            cy={yOf(h.dla_absmean)}
            r={hover === h ? 4.5 : 2.6}
            fill={ROLE_COLOR[h.role] ?? ROLE_COLOR.other}
            fillOpacity={0.82}
            data-testid={`head-role-point-${h.layer}-${h.head}`}
            onMouseEnter={() => setHover(h)}
            onMouseLeave={() => setHover((c) => (c === h ? null : c))}
            style={{ cursor: 'pointer' }}
          />
        ))}

        {/* legend with counts */}
        <g transform={`translate(${PAD_L + plotW + 16}, ${PAD_T + 2})`}>
          {rolesPresent.map((r, i) => (
            <g key={r} transform={`translate(0, ${i * 17})`}>
              <circle cx={5} cy={-3} r={4} fill={ROLE_COLOR[r]} />
              <text x={15} y={0} className="head-roles__legend">
                {r} <tspan className="head-roles__legend-count">{counts[r]}</tspan>
              </text>
            </g>
          ))}
        </g>
      </svg>

      {hover && (
        <div className="head-roles__tooltip" data-testid="head-roles-tooltip">
          <strong>
            L{hover.layer} · H{hover.head}
          </strong>{' '}
          — {hover.role}
          <br />
          |DLA| {hover.dla_absmean.toFixed(3)} · BOS {hover.bos_weight.toFixed(2)} ·
          self {hover.self_weight.toFixed(2)}
        </div>
      )}
    </section>
  );
}
