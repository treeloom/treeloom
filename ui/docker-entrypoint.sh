#!/bin/sh
# Generate the SPA's runtime config before nginx starts.
#
# TREELOOM_API_BASE must be reachable FROM THE BROWSER. The indexer runs on the
# host (published at localhost:8001), so this is a host URL, NOT an in-compose
# service name. The indexer's CORS allow-list (TREELOOM_CORS_ORIGINS) must
# include this UI's origin for cross-origin calls to succeed.
set -eu
: "${TREELOOM_API_BASE:=http://localhost:8001}"
cat > /usr/share/nginx/html/config.js <<EOF
window.__TREELOOM_API_BASE__ = "${TREELOOM_API_BASE}";
EOF
echo "treeloom-ui: wrote config.js (TREELOOM_API_BASE=${TREELOOM_API_BASE})"
