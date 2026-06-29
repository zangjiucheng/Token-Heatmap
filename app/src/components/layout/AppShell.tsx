import { Outlet } from 'react-router-dom';
import { useTheme } from '@/hooks/useTheme';
import { Header } from './Header';
import './AppShell.css';

export function AppShell() {
  // Calling useTheme here ensures the data-theme attribute is applied at
  // the very root of the rendered tree, before any children paint.
  useTheme();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <Header />
      <main id="main-content" className="app-shell__main" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
