export type NodeKind = 'embed' | 'attn' | 'mlp' | 'head';

export interface PlacedNode {
  id: string;
  label: string;
  value: number;
  layer: number; // -1 for embed
  kind: NodeKind;
  head?: number;
  x: number;
  y: number;
  r: number;
}

export interface GraphOutput {
  x: number;
  y: number;
  token: string;
}

function edgePath(x1: number, y1: number, x2: number, y2: number): string {
  const mx = (x1 + x2) / 2;
  return `M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

const OUT_R = 24;

export interface GraphFigureProps {
  placed: PlacedNode[];
  output: GraphOutput;
  error: number;
  maxAbs: number;
  /** viewBox dimensions. */
  contentW: number;
  H: number;
  className?: string;
  testId?: string;
}

/**
 * The attribution-graph drawing itself, factored out of the tab so it can be
 * rendered both inline (scaled to fit the lens) and inside the zoom lightbox at
 * full size. Pure presentation: layout is computed by the caller.
 */
export function GraphFigure({
  placed,
  output,
  error,
  maxAbs,
  contentW,
  H,
  className = 'graph-tab__svg',
  testId = 'attribution-graph-svg',
}: GraphFigureProps) {
  const edgeWidth = (v: number) => 1 + (Math.abs(v) / maxAbs) * 5;

  return (
    <svg
      className={className}
      viewBox={`0 0 ${contentW} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Attribution graph"
      data-testid={testId}
    >
      {/* Edges */}
      <g className="graph-tab__edges">
        {placed.map((n) => (
          <path
            key={`e-${n.id}`}
            className="graph-tab__edge"
            data-sign={n.value >= 0 ? 'pos' : 'neg'}
            d={edgePath(n.x + n.r, n.y, output.x - OUT_R, output.y)}
            style={{ strokeWidth: edgeWidth(n.value) }}
          />
        ))}
        {Math.abs(error) > 1e-6 ? (
          <path
            className="graph-tab__edge graph-tab__edge--error"
            d={edgePath(output.x, output.y + 70, output.x, output.y + OUT_R)}
            style={{ strokeWidth: edgeWidth(error) }}
          />
        ) : null}
      </g>

      {/* Contributor nodes (static — sized + coloured by signed contribution) */}
      {placed.map((n) => {
        const title = `${n.label}: ${n.value >= 0 ? '+' : ''}${n.value.toFixed(3)}`;
        return (
          <g
            key={n.id}
            className="graph-tab__node"
            data-testid={`graph-node-${n.id}`}
            aria-label={n.label}
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
          r={OUT_R}
          className="graph-tab__output-dot"
        />
        <text
          className="graph-tab__output-label"
          x={output.x}
          y={output.y - OUT_R - 8}
          textAnchor="middle"
        >
          {JSON.stringify(output.token)}
        </text>
      </g>
    </svg>
  );
}
