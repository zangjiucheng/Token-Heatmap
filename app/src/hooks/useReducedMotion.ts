import { useEffect, useState } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

/**
 * Tracks `prefers-reduced-motion: reduce`. Sets `data-reduced-motion="true"`
 * on <body> while active so CSS can disable transitions/animations via
 * attribute selectors as well as the @media query.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() => {
    if (
      typeof window === 'undefined' ||
      typeof window.matchMedia !== 'function'
    ) {
      return false;
    }
    return window.matchMedia(QUERY).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return undefined;
    }
    const mql = window.matchMedia(QUERY);
    const apply = (matches: boolean) => {
      setReduced(matches);
      if (typeof document !== 'undefined' && document.body) {
        if (matches) {
          document.body.setAttribute('data-reduced-motion', 'true');
        } else {
          document.body.removeAttribute('data-reduced-motion');
        }
      }
    };
    apply(mql.matches);
    const handler = (event: MediaQueryListEvent) => apply(event.matches);
    if (typeof mql.addEventListener === 'function') {
      mql.addEventListener('change', handler);
      return () => mql.removeEventListener('change', handler);
    }
    mql.addListener(handler);
    return () => mql.removeListener(handler);
  }, []);

  return reduced;
}
