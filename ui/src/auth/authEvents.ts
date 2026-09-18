/**
 * A tiny synchronous event bus bridging the (non-React) TanStack Query client
 * to the React {@link AuthProvider}.
 *
 * The query client is constructed once at module load — outside the React tree
 * — so its `QueryCache`/`MutationCache` `onError` hooks can't call a hook or a
 * setState directly. They call {@link notifyAuthRequired} instead; the provider
 * subscribes via {@link onAuthRequired} on mount. Pure module state, no React.
 */
type Listener = () => void;

const listeners = new Set<Listener>();

/** Subscribe to "a 401 happened, force re-auth" events. Returns an unsubscribe. */
export function onAuthRequired(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Fire the "auth required" signal to every current subscriber. */
export function notifyAuthRequired(): void {
  for (const listener of [...listeners]) {
    listener();
  }
}
