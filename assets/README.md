# Assets

Visual identity files for Oransim.

| File | Purpose | Dimensions |
|------|---------|------------|
| `logo.svg` | Primary square logo (light-theme-friendly) | 512×512 |
| `logo-dark.svg` | Dark-theme variant | 512×512 |
| `wordmark.svg` | Horizontal wordmark + icon | 800×200 |
| `social-preview.svg` | GitHub Social Preview (exported as PNG) | 1280×640 |
| `architecture.svg` | Main architecture diagram (used in README) | 1200×500 |

## Converting SVG → PNG

For GitHub's Social Preview upload, export the SVG to PNG:

```bash
# Using rsvg-convert (apt install librsvg2-bin)
rsvg-convert -w 1280 -h 640 social-preview.svg > social-preview.png

# Or using Inkscape CLI
inkscape --export-type=png --export-filename=social-preview.png --export-width=1280 social-preview.svg
```

Then upload `social-preview.png` at GitHub → repo `Settings → Social preview`.

## Placeholder Status

These SVGs are **initial placeholders** hand-coded for Phase 1 ship. A designer-finalized visual identity (logo refinements, brand-consistent color palette, illustration style) is on the Phase 2+ roadmap — replace these files when ready.

Brand colors used in placeholders:
- Primary orange: `#FF6B35`
- Primary orange (lighter): `#FF8A3C`
- Deep blue: `#004E89`
- Night navy: `#1a2a4a`
- Background dark: `#0d1117`
