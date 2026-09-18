import { useState } from "react";

import { Button } from "@/components/ui/button";

/**
 * Reveal-once secret box: shows a freshly-created token/key with a copy
 * affordance and a prominent "won't be shown again" warning. Used by the
 * Access Tokens tab for both PATs and API keys.
 */
export function SecretReveal({
  secret,
  label = "Secret",
  onDismiss,
}: {
  secret: string;
  label?: string;
  onDismiss?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard?.writeText(secret);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard may be unavailable (insecure context / test env) — no-op.
    }
  }

  return (
    <div
      role="alert"
      className="mt-4 rounded-md border border-success/40 bg-success/10 p-4"
    >
      <p className="text-sm font-medium text-success">{label} created</p>
      <p className="mt-1 text-xs text-muted-foreground">
        Copy it now — it won't be shown again.
      </p>
      <div className="mt-3 flex items-center gap-2">
        <code
          data-testid="secret-value"
          className="flex-1 truncate rounded bg-muted px-2 py-1 font-mono text-xs"
        >
          {secret}
        </code>
        <Button type="button" size="sm" variant="outline" onClick={() => void copy()}>
          {copied ? "Copied!" : "Copy"}
        </Button>
        {onDismiss && (
          <Button type="button" size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        )}
      </div>
    </div>
  );
}
