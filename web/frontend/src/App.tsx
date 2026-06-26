import { useCallback, useEffect, useState } from 'react';
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { OPEN_KEYMAP_HELP_EVENT } from '@/components/layout/Header';
import { ErrorBoundary } from '@/components/feedback/ErrorBoundary';
import { KeymapHelpDialog } from '@/components/help/KeymapHelpDialog';
import { useKeymap } from '@/hooks/useKeymap';
import { useReducedMotion } from '@/hooks/useReducedMotion';
import { useTheme } from '@/hooks/useTheme';
import { useViewState } from '@/hooks/useViewState';
import { DiffViewerPage } from '@/pages/DiffViewerPage';
import { LandingPage } from '@/pages/LandingPage';
import { TraceViewerPage } from '@/pages/TraceViewerPage';

/**
 * Moves focus to the first heading inside <main> on each route change so
 * screen-reader users land at the top of the new page without having to
 * tab past the chrome.
 */
function RouteFocusManager() {
  const location = useLocation();

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const main = document.getElementById('main-content');
    if (!main) return;
    const heading = main.querySelector<HTMLElement>('h1, h2');
    const target = heading ?? main;
    if (!target.hasAttribute('tabindex')) {
      target.setAttribute('tabindex', '-1');
    }
    target.focus({ preventScroll: false });
  }, [location.pathname]);

  return null;
}

function GlobalShortcuts() {
  const { toggleTheme } = useTheme();
  const [helpOpen, setHelpOpen] = useState(false);
  const { setTab } = useViewState();

  useReducedMotion();

  const openHelp = useCallback(() => setHelpOpen(true), []);
  const closeHelp = useCallback(() => setHelpOpen(false), []);

  // The header's "?" button lives outside this component's tree, so it asks for
  // the dialog via a window event rather than a shared context.
  useEffect(() => {
    window.addEventListener(OPEN_KEYMAP_HELP_EVENT, openHelp);
    return () => window.removeEventListener(OPEN_KEYMAP_HELP_EVENT, openHelp);
  }, [openHelp]);

  useKeymap({
    'help.open': openHelp,
    'view.toggleTheme': () => toggleTheme(),
    'navigation.gotoHeatmap': () => {
      setTab('heatmap');
      // Defer focus so the tab content has a chance to mount.
      window.setTimeout(() => {
        const plot = document.querySelector<HTMLElement>(
          '[data-testid="token-heatmap-plot"]',
        );
        plot?.focus();
      }, 0);
    },
    'navigation.gotoAttention': () => {
      setTab('attention');
      window.setTimeout(() => {
        const tab = document.querySelector<HTMLElement>(
          '[data-testid="attention-tab"]',
        );
        tab?.focus();
      }, 0);
    },
    'navigation.gotoLogitLens': () => {
      setTab('logit-lens');
      window.setTimeout(() => {
        const tab = document.querySelector<HTMLElement>(
          '[data-testid="logit-lens-tab"]',
        );
        tab?.focus();
      }, 0);
    },
    'navigation.gotoDetail': () => {
      const panel = document.querySelector<HTMLElement>(
        '[data-testid="step-detail-panel"]',
      );
      panel?.focus();
    },
    'selection.clear': () => {
      if (!helpOpen) return;
      closeHelp();
    },
  });

  return <KeymapHelpDialog open={helpOpen} onClose={closeHelp} />;
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <RouteFocusManager />
        <GlobalShortcuts />
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<LandingPage />} />
            <Route path="trace" element={<TraceViewerPage />} />
            <Route path="trace/:id" element={<TraceViewerPage />} />
            <Route path="diff/:id" element={<DiffViewerPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
