# Treeloom Operator UI

A standalone single-page app (SPA) for operating a Treeloom indexer: viewing
indexing **Jobs**, browsing indexed **Sources**, and **Submit**ting new repos /
directories for indexing.

This is the v2 operator UI foundation. At this stage it is a
**scaffold only** — the three tabs are placeholder pages and the top-bar health
indicator is static. API wiring lands in later issues.

## Authentication

The app authenticates to the indexer with a **bearer token** — a personal access
token (PAT) or API key pasted into the login screen and stored in `localStorage`
(`src/api/token.ts`). The API client attaches it as `Authorization: Bearer …` on
every request.

- **AUTH_ENABLED-off works as-is.** The app renders normally with *no* token; the
  login screen is **never** shown pre-emptively. It is forced **reactively** only
  when the server actually returns **401** — wired once in
  `src/lib/queryClient.ts` (`QueryCache`/`MutationCache` `onError` →
  `shouldForceLogin` → the `AuthProvider`'s 401 bus), so a 401 from *anywhere* in
  the app clears the token and drops to the login screen with a
  "session expired / re-authenticate" banner.
- **Logout** lives in the header (`AuthControls`, next to the health indicator):
  it clears the token and returns to the login screen.
- **"Logged in as"** is best-effort: `AuthControls` calls `GET /auth/me` and shows
  the username when authenticated, tolerating a failure/401 as anonymous.

### Secondary cookie path (supported, not default)

The indexer also supports **username/password cookie login**
(`POST /auth/login` → a `SameSite=None; Secure` session cookie; see the backend
local-auth docs). That path is **not** the default for this UI and is **not**
implemented here — v1 uses bearer tokens only. A username/password form may be
added later for installs that prefer cookie sessions.

## Stack

- **Vite 6** + **React 18** + **TypeScript** (strict)
- **react-router-dom** v6 — tab routing (`/jobs`, `/sources`, `/submit`)
- **@tanstack/react-query** — data-fetching provider mounted at the app root
- **Tailwind CSS 3** + **shadcn/ui** components, dark theme

### shadcn/ui

The shadcn components were **hand-created** (the "copy components in" model)
rather than via `npx shadcn@latest init`. The init flow makes network calls and
runs an interactive prompt, which is unsuitable for an offline/CI scaffold. The
equivalents live under `src/components/ui/` (`button.tsx`, `tabs.tsx`) alongside
the `cn()` helper in `src/lib/utils.ts`, with the standard shadcn Tailwind theme
tokens in `tailwind.config.js` + `src/index.css`. Add more components later with
`npx shadcn@latest add <component>` once network is available — the config is
already shadcn-compatible.

The dark palette is loosely matched to the legacy operator dashboard: deep
`#0d1117` background, `#161b22` panels, brand green `#3fb950` primary/accent,
amber `#d29922` warning, red `#ef4444` danger.

## Commands

```bash
# Install dependencies
npm install

# Start the dev server (default port 5173)
npm run dev
#   ...or pick a port:  npm run dev -- --port 5199

# Type-check + production build -> ./dist
npm run build

# Preview the production build locally
npm run preview

# Type-check only (no emit)
npm run typecheck
```

## Layout

```
ui/
├─ index.html
├─ package.json
├─ vite.config.ts            # @ -> ./src alias
├─ tailwind.config.js        # dark theme + brand tokens
├─ postcss.config.js
├─ tsconfig*.json
└─ src/
   ├─ main.tsx               # React root
   ├─ App.tsx                # QueryClientProvider + RouterProvider
   ├─ index.css             # Tailwind layers + theme CSS vars
   ├─ lib/
   │  ├─ utils.ts            # cn()
   │  └─ queryClient.ts      # TanStack Query client
   ├─ routes/router.tsx      # react-router route table
   ├─ components/
   │  ├─ ui/                 # shadcn components (button, tabs)
   │  └─ layout/             # AppShell, HealthIndicator
   └─ pages/                 # JobsPage, SourcesPage, SubmitPage
```
