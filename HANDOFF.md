# Phase 1 Handoff

**Status:** ✅ Phase 1 complete and ready to push.

## What's ready

Local git repo at `/root/projects/oransim/` with **13 clean commits** on `main`:

| Category | Files | Lines |
|---|---|---|
| Docs (EN+ZH) | README (EN+ZH) / ROADMAP / CHANGELOG / CITATION / SECURITY / CONTRIBUTING / COC | ~1375 |
| Legal | LICENSE (Apache-2.0) / NOTICE | 215 |
| GitHub meta | ISSUE_TEMPLATE × 3 + PR template + FUNDING + CODEOWNERS + config.yml | 204 |
| SVG assets | logo / logo-dark / wordmark / social-preview / architecture + assets/README | 300 |
| Package skeleton | `backend/oransim/` with 5 platforms + 8 submodules (all `__init__.py` stubs) | 185 |
| Config | pyproject.toml / .gitignore / .gitattributes | 160 |

**Verified working:**
- `PYTHONPATH=backend python3 -c "import oransim; print(oransim.__version__)"` → `0.1.0a0` ✓
- TikTok/Instagram/YouTube Shorts/Douyin stubs raise `NotImplementedError` with ROADMAP.md reference ✓
- EN/ZH READMEs both have exactly 15 `##` sections (parity) ✓
- All 5 SVGs are valid `SVG Scalable Vector Graphics image` ✓
- Apache-2.0 LICENSE is 201 lines (full canonical text) ✓

## What you need to do

Read `GITHUB_SETUP.md` — seven steps take ~30 minutes:

1. Create GitHub org `oranai` (or choose your own handle/org)
2. `git remote add origin` + `git push`
3. Configure repo settings (topics, website, social preview upload, branch protection, security)
4. Tag `v0.1.0-alpha` release on GitHub
5. Grep-replace remaining `[TBD: ...]` placeholders (your GitHub handle, etc.)
6. Announce (HN, X, LinkedIn, etc.)
7. Monitor first 48 hours

## Deferred to Phase 2

Full list in the design spec at `/home/projects/sim/docs/superpowers/specs/2026-04-18-oransim-oss-design.md`:

- Code desensitization from `/home/projects/sim/`
- `AUDIT_LOG.md` with before/after hashes for all scrubbed content
- pkl safety verification (PCA feature names, no real KOL IDs)
- Prompt template scan for real brand names
- `sim/` → `oransim/` code migration (rename package `causaltwin` → `oransim`)

## Deferred to Phase 3

- `PlatformAdapter` abstract base + XHS reference implementation
- `CanonicalKOL / Note / FanProfile` Pydantic schemas
- DataProvider plugins (Synthetic / CSV / JSON / OpenAPI)
- `scripts/gen_synthetic_data.py` + synthetic-data-trained pkl
- Test suite (pytest + `LLM_MODE=mock`)
- Docker + docker-compose
- CI (GitHub Actions — `ci.yml` / `release.yml` / `docs.yml`)
- MkDocs site deployment

## Open questions parked (spec §0)

Placeholders remaining in repo files, grep for them with:
```bash
cd /root/projects/oransim && grep -rn "\[TBD:" .
```

Current TBDs (all optional):
- `[TBD: GITHUB_ORG]` in `.github/FUNDING.yml` — when GitHub Sponsors is live
- `[TBD: BUSINESS_EMAIL]` / `[TBD: CAREERS_EMAIL]` — if splitting beyond `cto@orannai.com`
- Discord / Twitter — Phase 2 community infrastructure

Note: `ORAN-cgsj` is already the active handle throughout the repo — no substitution needed.

## Next actions after GitHub push

Once live, consider:
1. Posting `Show HN: Oransim` to Hacker News
2. Seeding 2–3 GitHub Discussions to warm up the Q&A category
3. Coordinating a LinkedIn announce with OranAI team
4. Following the repo from your personal handle to prime the social graph

---

Delete this file after reading:
```bash
cd /root/projects/oransim && rm HANDOFF.md && git commit -am "chore: remove handoff note" -s
```
