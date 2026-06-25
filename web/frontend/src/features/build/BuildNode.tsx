import type { ReactNode } from 'react';

export interface BuildNodeProps {
  /** Short title shown in the node header. */
  title: string;
  /** 1-based index rendered in the node's step badge. */
  step: number;
  /** Short subtitle / role description. */
  subtitle?: string;
  /** Render a left-edge input port (false for the first node). */
  hasInput?: boolean;
  /** Render a right-edge output port (false for the last node). */
  hasOutput?: boolean;
  children: ReactNode;
}

/**
 * A single card in the config pipeline. Renders a titled, badged container with
 * optional left/right connection ports so a row of nodes reads as a wired
 * graph. Purely presentational — all field state lives in the parent graph.
 */
export function BuildNode({
  title,
  step,
  subtitle,
  hasInput = true,
  hasOutput = true,
  children,
}: BuildNodeProps) {
  return (
    <div className="build-node" data-testid={`build-node-${step}`}>
      {hasInput ? (
        <span
          className="build-node__port build-node__port--in"
          aria-hidden="true"
        />
      ) : null}
      {hasOutput ? (
        <span
          className="build-node__port build-node__port--out"
          aria-hidden="true"
        />
      ) : null}
      <header className="build-node__header">
        <span className="build-node__badge">{step}</span>
        <div className="build-node__heading">
          <h3 className="build-node__title">{title}</h3>
          {subtitle ? (
            <p className="build-node__subtitle">{subtitle}</p>
          ) : null}
        </div>
      </header>
      <div className="build-node__body">{children}</div>
    </div>
  );
}

/**
 * Directional connector drawn between two nodes. Horizontal (arrow points
 * right) on wide layouts; a CSS media query rotates it to point down when the
 * pipeline stacks vertically. Decorative.
 */
export function PipelineConnector() {
  return (
    <div className="build-connector" aria-hidden="true">
      <svg
        className="build-connector__svg"
        viewBox="0 0 48 16"
        preserveAspectRatio="none"
        focusable="false"
      >
        <line
          className="build-connector__line"
          x1="0"
          y1="8"
          x2="40"
          y2="8"
        />
        <path className="build-connector__head" d="M40 3 L47 8 L40 13 Z" />
      </svg>
    </div>
  );
}
