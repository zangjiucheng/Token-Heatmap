# LLM Token Heatmap — Desktop App (Tauri + React)

React 18 + TypeScript + Vite SPA for loading and exploring LLM token
probability traces. It is a static, file-based **viewer** with no backend:
traces load from a dropped JSON file, a `?trace=<url>` URL, or the bundled
sample. The app supports sample data, raw/processed comparison,
attention/logit-lens inspection, activation views, diff-mode, and CSV/PNG
export.

## Prerequisites

- Node.js 20+
- npm 10+

## Install

```bash
cd app
npm install
```

## Common commands

| Command                | What it does                                                                      |
| ---------------------- | --------------------------------------------------------------------------------- |
| `npm run dev`          | Start the Vite dev server (default port `5173`; override with `VITE_DEV_PORT`).   |
| `npm run build`        | Type-check then produce a production bundle in `dist/`.                           |
| `npm run preview`      | Serve the production build locally.                                               |
| `npm run lint`         | Lint the codebase with ESLint (`--max-warnings 0`).                               |
| `npm run format`       | Format the codebase with Prettier.                                                |
| `npm run format:check` | Verify formatting without writing changes.                                        |
| `npm run test`         | Run the Vitest suite once.                                                        |
| `npm run test:watch`   | Run Vitest in watch mode.                                                         |
| `npm run codegen`      | Regenerate the trace schema's TypeScript types from `docs/web/trace.schema.json`. |

## Project layout

```
app/
├── src/
│   ├── components/  # Reusable presentational components
│   ├── features/    # Feature-scoped modules (heatmap, attention, activations, ...)
│   ├── hooks/       # Cross-cutting React hooks
│   ├── lib/         # Trace loaders, validators, sample data, and pure utilities
│   ├── pages/       # Route-level components
│   ├── styles/      # Global styles & CSS variables
│   ├── types/       # Shared TypeScript trace and activation types
│   ├── App.tsx      # Root routes and global shortcuts
│   ├── App.test.tsx # Smoke test
│   └── main.tsx     # Entry point
├── index.html
├── vite.config.ts
├── vitest.config.ts
├── tsconfig.json
└── package.json
```

## Path alias

`@/` resolves to `src/`. Import via:

```ts
import App from '@/App';
import { something } from '@/lib';
```

The alias is configured in both `tsconfig.json` (for type checking) and `vite.config.ts` (for the bundler).

## Dev server port

The dev server defaults to `5173`. Override with the `VITE_DEV_PORT` environment variable, e.g. `VITE_DEV_PORT=4000 npm run dev`.

## Keyboard shortcuts

Press `?` at any time to open the in-app keyboard shortcut help dialog. The full keymap is defined in `src/lib/keymap.ts`.

### Selection

| Key    | Action                              |
| ------ | ----------------------------------- |
| `←`    | Move selection to the previous step |
| `→`    | Move selection to the next step     |
| `Home` | Jump to the first step              |
| `End`  | Jump to the last step               |
| `Esc`  | Clear tooltip / selection           |

### View

| Key | Action                  |
| --- | ----------------------- |
| `T` | Toggle light/dark theme |
| `R` | Reset zoom and pan      |

### Comparison

| Key | Action                                            |
| --- | ------------------------------------------------- |
| `C` | Cycle distribution mode (raw → processed → split) |

### Navigation

| Key   | Action                      |
| ----- | --------------------------- |
| `G D` | Focus the step detail panel |
| `G H` | Focus the heatmap           |

### Help

| Key | Action                                 |
| --- | -------------------------------------- |
| `?` | Open the keyboard shortcut help dialog |

Shortcuts are ignored while an `<input>`, `<textarea>`, or contentEditable element is focused so the user can type freely.

## Accessibility

The frontend is built to meet WCAG 2.1 AA. Concretely:

- All interactive elements (buttons, links, sliders, radio groups) carry an accessible name and an appropriate ARIA role.
- Focus is moved to the page heading on every route transition; a "Skip to content" link is the first focusable element.
- A single `prefers-reduced-motion: reduce` block in `src/styles/theme.css` disables transitions, animations, and JS-driven inertia. The `useReducedMotion` hook mirrors the preference to a `data-reduced-motion="true"` attribute on `<body>`.
- A top-level `<ErrorBoundary>` catches render-time exceptions and renders a recoverable fallback with a "Reload" button.
- Live-region announcements (`aria-live="polite"`) are emitted for step-change events via `src/lib/a11y/announceLiveRegion.ts`.

### Verification

- Run `npm run test` — the suite under `src/a11y/` runs `axe-core` against the landing page, trace viewer (loading), and trace viewer (with sample trace loaded), and asserts every interactive element has a non-empty accessible name.
- A Lighthouse CI budget is checked into `lighthouserc.json`: accessibility = 100, performance ≥ 85, best-practices ≥ 90 against `dist/` of `npm run build`.

## Loading traces

The viewer has no backend. Produce traces with the CLI (`token-heatmap trace`),
then load one of three ways:

- drop (or pick) a JSON trace file — two files at once opens diff-mode;
- open with a `?trace=<url>` query param, which auto-fetches on page load
  (`token-heatmap trace … --serve --frontend` wires this for you);
- click **Try sample data** to load the bundled sample.

To serve a trace produced elsewhere over a CORS-enabled static file server (for
the `?trace=<url>` path), use `token-heatmap serve <output-dir>`.
