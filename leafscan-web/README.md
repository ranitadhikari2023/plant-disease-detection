# LeafScan V3 Web Client — Botanical Field Console

Production frontend architecture for hierarchical foliar screening of Solanaceous crops (Potato & Tomato).

## Architectural Principles

- **Deterministic Backend Diagnostic Authority:** Machine learning models and the client never independently diagnose plant diseases or recommend treatment recipes.
- **Contract-First API Integration:** Strictly typed against OpenAPI specifications and validated at runtime using Zod.
- **Botanical Field Console Design System:** Tailored for outdoor legibility with centralized tokens (`--color-leaf-600`, `--color-leaf-50`, `--color-bark-900`, etc.).
- **Progressive Web App:** Offline-capable caching with Serwist, IndexedDB local ledger for field scans, and auto-syncing offline queues.
- **WCAG 2.2 AA Accessibility:** Screen reader announcements, non-color visual hierarchy, accessible meter semantics, and reduced-motion enforcement.

## Technology Stack

- **Framework:** Next.js 15 (App Router, Standalone Mode)
- **UI:** React 19
- **Styling:** Tailwind CSS v4 + `@theme` CSS tokens
- **State Management:** TanStack Query v5 + Zustand v5
- **Persistence:** IndexedDB via `idb`
- **Validation:** Zod + React Hook Form
- **PWA:** Serwist + Service Worker
- **Testing:** Vitest, Testing Library, MSW, Playwright
- **Linting & Formatting:** Biome

## Commands

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run unit, component, and contract tests
npm test

# Run tests with coverage
npm run test:coverage

# Lint and format check
npm run lint

# Production build
npm run build

# Start production server
npm start
```

## Environment Variables

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
SENTRY_DSN=
OTEL_EXPORTER_OTLP_ENDPOINT=
```
