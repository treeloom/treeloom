import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AuthControls } from "@/auth/AuthControls";
import { HealthIndicator } from "@/components/layout/HealthIndicator";
import { cn } from "@/lib/utils";

const TABS = [
  { value: "jobs", label: "Jobs", path: "/jobs" },
  { value: "sources", label: "Sources", path: "/sources" },
  { value: "submit", label: "Submit", path: "/submit" },
  { value: "backends", label: "Backends", path: "/backends" },
  { value: "tokens", label: "Tokens", path: "/tokens" },
  { value: "users", label: "Users", path: "/users" },
] as const;

export function AppShell() {
  const location = useLocation();
  const navigate = useNavigate();

  const active =
    TABS.find((t) => location.pathname.startsWith(t.path))?.value ?? "jobs";

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card px-8 py-4">
        <div className="flex items-center gap-3">
          <img
            src="/brand/treeloom-logo-horizontal.svg"
            alt="Treeloom"
            className="h-8 w-auto select-none"
          />
          <span className="rounded-full border border-border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Operator
          </span>
        </div>
        <div className="flex items-center gap-4">
          <HealthIndicator />
          <AuthControls />
        </div>
      </header>

      {/* Tab navigation (react-router driven) */}
      <nav className="sticky top-[65px] z-[9] border-b border-border bg-card px-8">
        <Tabs
          value={active}
          onValueChange={(v) => {
            const tab = TABS.find((t) => t.value === v);
            if (tab) navigate(tab.path);
          }}
        >
          <TabsList className="bg-transparent p-0">
            {TABS.map((t) => (
              <TabsTrigger key={t.value} value={t.value} asChild>
                <NavLink
                  to={t.path}
                  className={({ isActive }) =>
                    cn(
                      "-mb-px border-b-2 px-4 py-3 text-sm font-medium transition-colors",
                      isActive
                        ? "border-[hsl(var(--bright-green))] text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground",
                    )
                  }
                >
                  {t.label}
                </NavLink>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </nav>

      {/* Routed page content */}
      <main className="mx-auto max-w-[1200px] px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}
