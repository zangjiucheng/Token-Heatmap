import { NavLink, useLocation } from 'react-router-dom';
import { useTheme } from '@/hooks/useTheme';
import {
  GithubIcon,
  HelpIcon,
  MoonIcon,
  OpenTraceIcon,
  SunIcon,
} from '@/features/workspace/icons';
import './Header.css';

const GITHUB_URL = 'https://github.com/zangjiucheng/Token-Heatmap';

/** Open the keyboard-shortcut dialog owned by App's GlobalShortcuts. */
export const OPEN_KEYMAP_HELP_EVENT = 'token-heatmap:open-keymap-help';

function openKeymapHelp() {
  window.dispatchEvent(new Event(OPEN_KEYMAP_HELP_EVENT));
}

export function Header() {
  const { theme, toggleTheme } = useTheme();
  const { pathname } = useLocation();
  // Only meaningful once a trace is open; on the landing page the dropzone is
  // already the whole screen.
  const viewingTrace = pathname.startsWith('/trace');

  return (
    <header className="app-header" role="banner">
      <NavLink
        to="/"
        className="app-header__brand"
        aria-label="Token Heatmap — home"
      >
        <span className="app-header__mark" aria-hidden="true" />
        <span className="app-header__brand-text">Token Heatmap</span>
      </NavLink>

      <div className="app-header__utils">
        {/*
          The desktop app has no browser Back button, so without an explicit
          control here the viewer is a dead end: once a trace is open there is no
          discoverable way to pick a different one. The wordmark does navigate
          home, but nobody reads a logo as "open another file".
        */}
        {viewingTrace && (
          <NavLink
            to="/"
            className="app-header__icon-button app-header__open-trace"
            aria-label="Open another trace"
            title="Open another trace"
          >
            <OpenTraceIcon />
            <span className="app-header__open-trace-label">Open trace</span>
          </NavLink>
        )}
        <button
          type="button"
          className="app-header__icon-button"
          onClick={openKeymapHelp}
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          <HelpIcon />
        </button>
        <button
          type="button"
          className="app-header__icon-button"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          aria-pressed={theme === 'dark'}
          title="Toggle theme (T)"
          onClick={toggleTheme}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
        <a
          className="app-header__icon-button"
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer noopener"
          aria-label="GitHub repository"
          title="GitHub"
        >
          <GithubIcon />
        </a>
      </div>
    </header>
  );
}
