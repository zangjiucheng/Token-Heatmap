import { NavLink } from 'react-router-dom';
import { useTheme } from '@/hooks/useTheme';
import {
  GithubIcon,
  HelpIcon,
  MoonIcon,
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
