import { createBrowserRouter, Navigate } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { JobsPage } from "@/pages/JobsPage";
import { JobDetailPage } from "@/pages/JobDetailPage";
import { SourcesPage } from "@/pages/SourcesPage";
import { SubmitPage } from "@/pages/SubmitPage";
import { BackendsPage } from "@/pages/BackendsPage";
import { TokensPage } from "@/pages/TokensPage";
import { UsersPage } from "@/pages/UsersPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/jobs" replace /> },
      { path: "jobs", element: <JobsPage /> },
      { path: "jobs/:id", element: <JobDetailPage /> },
      { path: "sources", element: <SourcesPage /> },
      { path: "submit", element: <SubmitPage /> },
      { path: "backends", element: <BackendsPage /> },
      { path: "tokens", element: <TokensPage /> },
      { path: "users", element: <UsersPage /> },
      { path: "*", element: <Navigate to="/jobs" replace /> },
    ],
  },
]);
