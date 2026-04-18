# Contributing to Oransim

Thank you for your interest in contributing! This document explains how to propose changes.

## Quick Start

1. Fork the repo on GitHub
2. Clone your fork: `git clone https://github.com/<your-handle>/oransim.git`
3. Create a feature branch: `git checkout -b feat/my-change`
4. Install dev dependencies: `pip install -e '.[dev]'`
5. Make your changes, add tests
6. Run tests: `pytest`
7. Run linters: `ruff check . && black --check .`
8. Commit with DCO sign-off (see below)
9. Push and open a PR

## Developer Certificate of Origin (DCO)

All contributions must be signed off per the [Developer Certificate of Origin](https://developercertificate.org/). This replaces a CLA — by signing off, you confirm you have the right to submit the change under the project's Apache-2.0 license.

Add `Signed-off-by: Your Name <your.email@example.com>` to each commit message. The easiest way is `git commit -s`.

Unsigned commits will block the PR. If you forget, you can fix with:
```bash
git commit --amend -s --no-edit
# or for multiple commits:
git rebase --signoff HEAD~N
```

## Pull Request Checklist

- [ ] Commits are signed off (`git commit -s`)
- [ ] Tests added/updated for new behavior
- [ ] `pytest` passes locally
- [ ] `ruff check .` and `black --check .` pass
- [ ] Documentation updated if applicable (user-facing API, README, docs/)
- [ ] `CHANGELOG.md` "Unreleased" section updated
- [ ] PR title follows Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, etc.)

## Types of Contributions We Welcome

- **Platform adapters** — see `docs/en/platforms/writing-an-adapter.md`
- **Data providers** — see `docs/en/platforms/writing-a-provider.md`
- **World model improvements** — new algorithms, calibration, benchmarks
- **Documentation** — fixes, translations, tutorials, architecture explainers
- **Examples** — new Jupyter notebooks in `examples/`
- **Bug fixes** — always welcome
- **Benchmark scenarios** — add to `OrancBench`

## Code Style

- **Python**: black + ruff (configured in `pyproject.toml`). Line length 100.
- **Comments**: English only (keep the international contributor barrier low).
- **User-facing text (UI / error messages)**: English + Chinese (中英双语).
- **Docstrings**: Google style, English only.
- **Type hints**: required on public APIs.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` refactoring without behavior change
- `test:` add/update tests
- `chore:` tooling, deps, build
- `perf:` performance improvement

## Running Locally

```bash
# Install
pip install -e '.[dev]'

# Run backend (mock LLM mode for offline dev)
LLM_MODE=mock python -m uvicorn oransim.api:app --port 8001

# Run frontend
python -m http.server 8090 --directory frontend
# → http://localhost:8090
```

## Reporting Bugs

Use the [Bug Report issue template](https://github.com/OranAi-Ltd/oransim/issues/new?template=bug_report.yml).

## Proposing Features

For significant features, open a [Feature Request](https://github.com/OranAi-Ltd/oransim/issues/new?template=feature_request.yml) or start a [Discussion](https://github.com/OranAi-Ltd/oransim/discussions) first to align on approach before coding.

## Platform Adapter Requests

Want support for a platform we haven't covered yet (e.g., LinkedIn, Pinterest)? File an [Adapter Request](https://github.com/OranAi-Ltd/oransim/issues/new?template=adapter_request.yml).

## Community

- **Discussions**: https://github.com/OranAi-Ltd/oransim/discussions
- **Issues**: https://github.com/OranAi-Ltd/oransim/issues
- **Email**: cto@orannai.com (maintainer)

## License

By contributing, you agree your contribution will be licensed under Apache-2.0.
