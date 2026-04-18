# GitHub Setup Checklist (for Fakong)

This document walks you through the steps to publish the `oransim` repo and configure it for maximum visibility and star collection. All of these are **one-time setup steps** — complete once per repository.

**Local repo location:** `/root/projects/oransim`
**Target GitHub repo:** `https://github.com/ORAN-cgsj/oransim`

---

## Step 1: Create the GitHub remote

The live repo is hosted at **`https://github.com/ORAN-cgsj/oransim`** (OranAI GitHub org handle `ORAN-cgsj`). Early drafts of this doc referenced `oranai` as a placeholder before the org handle was finalized — if you see that in any archived file, treat it as historical.

If you're forking or bootstrapping a mirror under a different handle:

1. Go to https://github.com/organizations/new
2. Create the free org (or reuse a user account)
3. Confirm the URL: `https://github.com/<your-handle>`

Then create the repo:
1. https://github.com/organizations/<your-handle>/repositories/new
2. **Repository name**: `oransim`
3. **Description**: `🧠 Causal Digital Twin for Marketing at Scale · Predict any marketing decision before you spend a dollar.`
4. **Public** ✅
5. **Do NOT** initialize with README / LICENSE / .gitignore (we already have them)
6. Click **Create repository**

---

## Step 2: Push the local repo

```bash
cd /root/projects/oransim
git remote add origin https://github.com/ORAN-cgsj/oransim.git
git branch -M main
git push -u origin main
```

If you used a different org/user, replace `ORAN-cgsj` above and run a find-replace on the repo:

```bash
cd /root/projects/oransim
grep -rIl "ORAN-cgsj/oransim" . | xargs sed -i 's|ORAN-cgsj/oransim|YOUR_HANDLE/oransim|g'
grep -rIl "\[TBD: GITHUB_HANDLE\]" . | xargs sed -i 's|\[TBD: GITHUB_HANDLE\]|YOUR_HANDLE|g'
git add -A && git commit -m "chore: point at real GitHub org/handle" -s
git push
```

---

## Step 3: Configure repo settings

Go to https://github.com/ORAN-cgsj/oransim/settings and set:

### General
- [ ] **Features**: ✅ Issues · ✅ Discussions · ✅ Projects (optional) · ❌ Wiki (use `docs/` instead)
- [ ] **Pull Requests**: ✅ Allow squash merging (only) · ❌ Allow merge commits · ❌ Allow rebase merging · ✅ Automatically delete head branches
- [ ] **Archives**: leave defaults

### Topics
Add these topics (comma-separated, via the ⚙️ gear next to "About" on the repo home):
```
causal-inference
digital-twin
marketing-ai
agent-based-modeling
counterfactual-reasoning
hawkes-process
structural-causal-model
llm-agents
gpt
recommendation-systems
social-media
simulation
python
fastapi
```

### Website
In the "About" section (right sidebar of the repo home), click the ⚙️ and set:
- **Website**: `https://oran.cn/oransim`
- **Release tab**: ✅ Include in home page
- **Packages tab**: ✅ (optional)

### Social Preview
1. Export `assets/social-preview.svg` to PNG (1280×640):
   ```bash
   cd /root/projects/oransim
   # Option A (rsvg-convert):
   apt install -y librsvg2-bin
   rsvg-convert -w 1280 -h 640 assets/social-preview.svg > assets/social-preview.png

   # Option B (Inkscape):
   inkscape --export-type=png --export-filename=assets/social-preview.png --export-width=1280 assets/social-preview.svg
   ```
2. Upload `assets/social-preview.png` at **Settings → Social preview → Upload an image**.

### Branch Protection (Branches → main)
- [ ] **Require a pull request before merging**: ✅
- [ ] **Require approvals**: 1
- [ ] **Dismiss stale reviews when new commits are pushed**: ✅
- [ ] **Require status checks to pass before merging**: ✅ (after CI is live; skip now)
- [ ] **Require conversation resolution before merging**: ✅
- [ ] **Require signed commits**: ❌ (we use DCO, not GPG)
- [ ] **Require linear history**: ✅ (squash-only merge enforces this)

### Security
- Settings → Code security and analysis → enable:
  - [ ] Dependabot alerts: ✅
  - [ ] Dependabot security updates: ✅
  - [ ] Code scanning (CodeQL): ✅
  - [ ] Secret scanning: ✅

### Discussions
- Settings → Features → Discussions: ✅
- Once enabled, go to Discussions and create categories:
  - 📢 Announcements
  - 💡 Ideas / Adapter Requests
  - 🙋 Q&A
  - 🎉 Show & Tell

---

## Step 4: Tag the first release

```bash
cd /root/projects/oransim
git tag -a v0.1.0-alpha -m "Initial public release — skeleton + flagship README + ambitious roadmap"
git push origin v0.1.0-alpha
```

Then on GitHub:
1. Go to https://github.com/ORAN-cgsj/oransim/releases/new
2. Choose the tag `v0.1.0-alpha`
3. **Release title**: `v0.1.0-alpha — Oransim goes public 🚀`
4. **Describe this release**: copy from `CHANGELOG.md` → `## [0.1.0-alpha]` section
5. ✅ **Set as a pre-release**
6. Publish

---

## Step 5: Remaining placeholders (optional, post-launch)

The GitHub handle `ORAN-cgsj` is already substituted throughout the repo (Phase 1 close-out). A few optional placeholders remain for later:

```bash
cd /root/projects/oransim
grep -rn "\[TBD:" . --include="*.md" --include="*.yml"
```

Expected remaining TBDs (all optional, can be left for Phase 2):
- `[TBD: GITHUB_ORG]` in `.github/FUNDING.yml` — once Sponsors / org is set up
- `[TBD: BUSINESS_EMAIL]` / `[TBD: CAREERS_EMAIL]` — if you split contact beyond `cto@orannai.com`
- `[TBD: DISCORD_INVITE]` / `[TBD: TWITTER_HANDLE]` — Phase 2 community infrastructure

Replace each via `sed` or editor, then commit:

```bash
git commit -am "chore: resolve TBD placeholders" -s
git push
```

---

## Step 6: Announce

- [ ] Post on your social channels (X, LinkedIn) linking to the repo
- [ ] Submit to Hacker News (`Show HN: Oransim — causal digital twin for marketing`)
- [ ] Post on r/MachineLearning (tag: [Project])
- [ ] Submit to Product Hunt (optional — may want to wait for v0.2 when backend is real)
- [ ] Post in relevant Discord / Slack communities (MLOps, Causal Inference, Marketing Tech)
- [ ] Consider a launch blog post at `https://oran.cn/oransim/blog/hello-world`

---

## Step 7: Monitor early signal

First 48 hours, watch:
- Star count — target for launch week: 50+ (strong), 200+ (viral)
- Issues — respond within 24 hours
- Discussions — seed 2–3 questions yourself to warm up the category
- Fork count — often precedes stars by an hour or two

---

## Troubleshooting

### "Social preview is not updating"
GitHub caches social previews aggressively. Force a refresh:
- Post the URL in a Slack / Discord with `-og:force-reload` (doesn't actually do anything, but tricks OG scrapers)
- Wait 24 hours

### "Squash merge loses DCO sign-off"
Enable the `dco` GitHub App or configure the branch protection to require sign-off on the final commit message. Alternatively, add `Signed-off-by:` to the squash commit message manually when merging.

### "ROADMAP feels too ambitious"
That's the point. The spec was written with "更有野心更有想象空间" as an explicit requirement. Investors and contributors are attracted to ambitious projects; the roadmap signals that Oransim is building toward a major platform, not a weekend hack. Keep it.
