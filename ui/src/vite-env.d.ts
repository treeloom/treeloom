/// <reference types="vite/client" />

declare global {
  interface Window {
    /**
     * Runtime API base URL, injected by `public/config.js` (overwritten by the
     * deploy container). May be undefined if config.js failed to load — the API
     * client falls back to the local-dev default.
     */
    __TREELOOM_API_BASE__?: string;
  }
}

export {};
