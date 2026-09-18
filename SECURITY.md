# Security Policy

## Supported versions

Only the latest release on `main` is supported.

## Reporting a vulnerability

Please do NOT open a public issue for security vulnerabilities. Use GitHub's
private vulnerability reporting ("Security" tab → "Report a vulnerability")
on https://github.com/treeloom/treeloom. You will receive a response within
a week. Please include a proof of concept if possible.

Areas of particular interest: the indexer HTTP API auth model (PATs, API
keys, sessions), per-repo search authorization, git-URL validation on
`/index-repo`, and the webhook signature verification path.
