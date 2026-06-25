import type { ViewerTab } from '@/hooks/useViewState';

/**
 * Tiny inline line icons. The project intentionally ships no icon dependency
 * (only `ajv` + React), so each glyph is a hand-rolled 24×24 stroke path that
 * inherits `currentColor`. Kept deliberately minimal — one mark per lens.
 */

export interface IconProps {
  size?: number;
  className?: string;
}

function baseProps(size: number) {
  return {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.7,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
    focusable: false,
  };
}

/** Per-lens glyph paths, keyed by the lens id. */
const LENS_PATHS: Record<ViewerTab, React.ReactNode> = {
  heatmap: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="1.5" />
      <path d="M9 3v18M15 3v18M3 9h18M3 15h18" />
    </>
  ),
  output: (
    <>
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <path d="M8 8h8M8 12h8M8 16h5" />
    </>
  ),
  model: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
    </>
  ),
  attention: (
    <>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="2.5" />
    </>
  ),
  'logit-lens': (
    <>
      <path d="M12 3 3 8l9 5 9-5-9-5Z" />
      <path d="M3 12.5 12 17.5l9-5M3 16.5 12 21.5l9-5" />
    </>
  ),
  activations: <path d="M3 12h3l2.5-7 4 14 2.5-7H21" />,
  manifold: (
    <>
      <circle cx="6" cy="8" r="1.4" />
      <circle cx="12" cy="5.5" r="1.4" />
      <circle cx="17.5" cy="9" r="1.4" />
      <circle cx="8.5" cy="15" r="1.4" />
      <circle cx="15" cy="17" r="1.4" />
      <path d="M6 8l6-2.5M12 5.5 17.5 9M8.5 15 6 8M15 17l2.5-8M8.5 15 15 17" />
    </>
  ),
};

export function LensIcon({
  lens,
  size = 16,
  className,
}: IconProps & { lens: ViewerTab }) {
  return (
    <svg {...baseProps(size)} className={className}>
      {LENS_PATHS[lens]}
    </svg>
  );
}

export function ChevronIcon({
  direction = 'left',
  size = 16,
  className,
}: IconProps & { direction?: 'left' | 'right' }) {
  const d = direction === 'left' ? 'M15 6l-6 6 6 6' : 'M9 6l6 6-6 6';
  return (
    <svg {...baseProps(size)} className={className}>
      <path d={d} />
    </svg>
  );
}

export function InspectorIcon({ size = 16, className }: IconProps) {
  return (
    <svg {...baseProps(size)} className={className}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M14 4v16" />
    </svg>
  );
}

export function HelpIcon({ size = 16, className }: IconProps) {
  return (
    <svg {...baseProps(size)} className={className}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7" />
      <path d="M12 17h.01" />
    </svg>
  );
}

export function SunIcon({ size = 16, className }: IconProps) {
  return (
    <svg {...baseProps(size)} className={className}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

export function MoonIcon({ size = 16, className }: IconProps) {
  return (
    <svg {...baseProps(size)} className={className}>
      <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.5 6.5 0 0 0 9.8 9.8Z" />
    </svg>
  );
}

export function GithubIcon({ size = 16, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
      focusable={false}
      className={className}
    >
      <path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-.88-.01-1.73-2.78.62-3.37-1.21-3.37-1.21-.46-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.62.07-.62 1 .07 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.85.09-.66.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.06 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05a9.3 9.3 0 0 1 5 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.93-2.34 4.79-4.57 5.05.36.32.68.94.68 1.9 0 1.37-.01 2.48-.01 2.82 0 .27.18.6.69.49A10.26 10.26 0 0 0 22 12.25C22 6.58 17.52 2 12 2Z" />
    </svg>
  );
}
