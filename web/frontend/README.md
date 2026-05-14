# LLM Token Heatmap — Frontend

React 18 + TypeScript + Vite SPA for loading and exploring LLM token
probability traces. The app supports trace upload, sample data, raw/processed
comparison, attention/logit-lens inspection, activation views, diff-mode, and
CSV/PNG export.

## Prerequisites

- Node.js 20+
- npm 10+

## Install

```bash
cd web/frontend
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
| `npm run codegen:api`  | Regenerate `src/api/generated/` from `openapi.json` (backend OpenAPI snapshot).   |

## Project layout

```
web/frontend/
├── src/
│   ├── api/         # Backend client and generated OpenAPI client
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

## Backend integration

The frontend talks to the FastAPI backend in `web/backend/` for trace generation, schema discovery, and CSV conversion. Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
# .env contains:
#   VITE_API_BASE_URL=http://localhost:8000
#   VITE_DEV_PORT=5173
```

### End-to-end dev loop

The fastest way to run both services together is the `dev` script at the repo root:

```bash
# from the repo root
./scripts/dev.sh
# backend → http://localhost:8000
# frontend → http://localhost:5173 (open this in your browser)
```

The script terminates both processes cleanly on `Ctrl+C`. Override ports with `BACKEND_PORT=8765 FRONTEND_PORT=4000 ./scripts/dev.sh`.

### Regenerating the API client

`src/api/generated/` is committed but autogenerated. When the backend's OpenAPI surface changes:

```bash
# 1. Refresh the OpenAPI snapshot (backend package must be importable).
python -c "import json; from llm_token_heatmap_api.main import create_app; json.dump(create_app().openapi(), open('web/frontend/openapi.json','w'), indent=2, sort_keys=True)"

# 2. Regenerate the typed client.
cd web/frontend && npm run codegen:api
```

The codegen is deterministic — running it twice yields a zero diff. CI guards against drift.
