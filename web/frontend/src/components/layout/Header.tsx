import { Link, useNavigate } from 'react-router-dom';
import { useTheme } from '@/hooks/useTheme';
import './Header.css';

const GITHUB_URL = 'https://github.com/zangjiucheng/Token-Heatmap';

export function Header() {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  return (
    <header className="app-header" role="banner">
      <Link to="/" className="app-header__brand">
        LLM Token Heatmap
      </Link>
      <nav className="app-header__nav" aria-label="Primary">
        <button
          type="button"
          className="app-header__button"
          onClick={() => navigate('/')}
        >
          Load trace
        </button>
        <button
          type="button"
          className="app-header__button"
          onClick={() => navigate('/build')}
        >
          Build trace
        </button>
        <button
          type="button"
          className="app-header__button"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          aria-pressed={theme === 'dark'}
          onClick={toggleTheme}
        >
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
        <a
          className="app-header__link"
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer noopener"
        >
          GitHub
        </a>
      </nav>
    </header>
  );
}
