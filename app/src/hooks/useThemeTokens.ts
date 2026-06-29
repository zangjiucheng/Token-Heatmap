import { useEffect, useState } from 'react';

/**
 * Canvas-drawn components (heatmaps) can't reference CSS custom properties
 * directly, so they read the resolved token values here. The hook re-reads
 * whenever the `data-theme` attribute on <html> flips, so canvases repaint
 * with the active theme's palette (the theme hook itself is local-state per
 * instance and doesn't broadcast, hence the MutationObserver).
 */
export interface ThemeTokens {
  bg: string;
  surface: string;
  bgMuted: string;
  text: string;
  textMuted: string;
  border: string;
  borderStrong: string;
  accent: string;
  selected: string;
}

const TOKEN_VARS: Record<keyof ThemeTokens, string> = {
  bg: '--color-bg',
  surface: '--color-surface',
  bgMuted: '--color-bg-muted',
  text: '--color-text',
  textMuted: '--color-text-muted',
  border: '--color-border',
  borderStrong: '--color-border-strong',
  accent: '--color-accent',
  selected: '--color-selected',
};

const FALLBACK: ThemeTokens = {
  bg: '#f3f1e9',
  surface: '#ffffff',
  bgMuted: '#eae7db',
  text: '#2b2823',
  textMuted: '#655f51',
  border: '#e6e2d4',
  borderStrong: '#cbc4b1',
  accent: '#d97757',
  selected: '#b26b05',
};

function readTokens(): ThemeTokens {
  if (
    typeof document === 'undefined' ||
    typeof getComputedStyle !== 'function'
  ) {
    return FALLBACK;
  }
  const cs = getComputedStyle(document.documentElement);
  const out = {} as ThemeTokens;
  (Object.keys(TOKEN_VARS) as (keyof ThemeTokens)[]).forEach((key) => {
    out[key] = cs.getPropertyValue(TOKEN_VARS[key]).trim() || FALLBACK[key];
  });
  return out;
}

export function useThemeTokens(): ThemeTokens {
  const [tokens, setTokens] = useState<ThemeTokens>(readTokens);

  useEffect(() => {
    // Re-read once on mount (covers SSR/hydration and the initial paint).
    setTokens(readTokens());
    if (
      typeof MutationObserver === 'undefined' ||
      typeof document === 'undefined'
    ) {
      return;
    }
    const observer = new MutationObserver(() => setTokens(readTokens()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    return () => observer.disconnect();
  }, []);

  return tokens;
}
