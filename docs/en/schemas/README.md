# Schema Reference

> **Status:** Coming soon (v0.2: JSON Schema auto-export from Pydantic models).

For v0.1.0-alpha, see the root README [Schemas section](https://github.com/ORAN-cgsj/oransim/blob/main/README.md#-what-you-get--14-to-19-schemas) for a human-readable list of the 14–19 output schemas.

## Canonical Input Schemas (Phase 3)

Pydantic definitions live in `backend/oransim/data/schema/`. JSON Schema exports will land in this directory alongside Sphinx-generated HTML.

- `CanonicalKOL` — unified KOL representation across platforms
- `CanonicalNote` — unified post/creative representation
- `CanonicalFanProfile` — fan demographics (age / gender / region / income distribution)
- `CanonicalCreative` — the creative being evaluated (text + images + video)
- `CanonicalScenario` — a full prediction request bundle

## Output Schemas (all 19)

See README. Each schema will have:
- JSON Schema file: `<schema_name>.schema.json`
- Example payload: `<schema_name>.example.json`
- Field-level documentation in Sphinx

Full schema documentation ships with v0.2.
