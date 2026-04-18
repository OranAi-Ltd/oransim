# Security Policy

## Supported Versions

Oransim is currently in alpha. We support security fixes on the latest minor version only:

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Do not open public issues for security vulnerabilities.**

Instead, email `cto@orannai.com` with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact (availability / confidentiality / integrity)
- Your suggested fix if you have one

You will receive an acknowledgment within 3 business days. We aim to issue a patch or mitigation within 30 days for critical findings, 90 days otherwise.

We follow responsible disclosure: please allow us to ship a fix before publishing details. Contributors who report valid vulnerabilities will be credited in our `SECURITY-HALL-OF-FAME.md` (opt-in).

## Scope

In-scope:
- The `oransim` Python package and bundled scripts
- Official Docker images (when available)
- Official documentation site

Out-of-scope:
- Third-party platform adapters in the community plugin registry (report to the plugin author)
- Self-hosted deployments with custom modifications

## Dependencies

We use Dependabot for automated dependency security updates. If you find a vulnerable dependency not yet flagged, please report it as described above.
