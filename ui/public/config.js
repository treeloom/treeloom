// Runtime API-base configuration.
//
// This file is loaded as a plain (non-module) script BEFORE the app bundle,
// so `window.__TREELOOM_API_BASE__` is available to the typed API client at
// runtime. The deploy container overwrites this file (later issue) to point
// the SPA at the real indexer; the value below is the local-dev default.
window.__TREELOOM_API_BASE__ = "http://localhost:8001";
