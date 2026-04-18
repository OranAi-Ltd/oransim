"""Phase V · persona_card 向量化 enrichment

User's locked-in design: do NOT touch the static SYSTEM prompt (would invalidate
prompt cache + risk JSON schema drift). Instead, enrich the per-persona
`full_card()` string that is injected into the LLM's user message, using content
derived from the persona's existing interest + bigfive vectors.

4 enrichments on top of the existing {one_liner, interests, psych} card:

1. Archetype label — derived from dot-product match against 8 preset archetype
   anchors (interest-vector mixtures), deterministic per persona.
2. Consumption anchors (top-3 content types this persona would click).
3. Anti-anchors (bottom-3 content types this persona would skip).
4. Multi-bullet psych (bigfive facets expanded from 1 line to multiple).

These tests pin the card structure so a future refactor can't silently drop
one of the four fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------------- fixtures


def _make_pop(N: int = 50, seed: int = 7):
    from oransim.data.population import generate_population

    return generate_population(N=N, seed=seed)


def _make_persona(idx: int = 0, seed: int = 7):
    import numpy as np
    from oransim.agents.soul import build_persona

    pop = _make_pop(seed=seed)
    rng = np.random.default_rng(seed)
    return build_persona(pop, idx, rng)


# ----------------------------------------------------- archetype assignment


def test_persona_has_archetype_label():
    p = _make_persona()
    card = p.full_card()
    assert "原型画像" in card, (
        "persona_card must include an archetype label derived from the "
        "interest vector (one of 8 preset archetypes). Got:\n" + card
    )


def test_archetype_is_one_of_the_preset_labels():
    from oransim.agents.soul import ARCHETYPE_LABELS

    p = _make_persona()
    matched = [lbl for lbl in ARCHETYPE_LABELS if lbl in p.full_card()]
    assert len(matched) == 1, (
        f"exactly one archetype should match, got {matched}. Card:\n" + p.full_card()
    )


def test_archetype_deterministic_for_same_persona():
    p1 = _make_persona(idx=3, seed=11)
    p2 = _make_persona(idx=3, seed=11)
    assert p1.archetype == p2.archetype
    assert p1.full_card() == p2.full_card()


# ---------------------------------------------- consumption anchors / anti


def test_persona_has_consumption_anchors():
    p = _make_persona()
    card = p.full_card()
    assert "常看内容" in card, (
        "persona_card must include top-3 consumption anchors derived from "
        "the persona's interest-vector top dims. Got:\n" + card
    )


def test_persona_has_anti_anchors():
    p = _make_persona()
    card = p.full_card()
    assert "不爱看" in card, (
        "persona_card must include bottom-3 anti-anchors (content types the "
        "persona would skip) derived from interest-vector bottom dims. Got:\n" + card
    )


def test_anchors_and_anti_anchors_are_disjoint():
    """A persona should not have the same tag in both 常看 and 不爱看."""
    p = _make_persona(idx=5, seed=13)
    assert isinstance(p.anchors, list) and len(p.anchors) >= 1
    assert isinstance(p.anti_anchors, list) and len(p.anti_anchors) >= 1
    overlap = set(p.anchors) & set(p.anti_anchors)
    assert (
        not overlap
    ), f"anchors {p.anchors} and anti-anchors {p.anti_anchors} overlap on {overlap}"


# ----------------------------------------------------- multi-bullet psych


def test_psych_bullets_list_exposed():
    """`psych_bullets` field lists individual personality facets (bigfive-derived)."""
    p = _make_persona()
    assert hasattr(p, "psych_bullets"), "persona missing psych_bullets attribute"
    assert isinstance(p.psych_bullets, list)
    # bigfive is 5-D; current rules fire ≤5 but target at least 2-3 salient bullets
    assert 1 <= len(p.psych_bullets) <= 6


def test_card_includes_psych_bullets_as_separate_lines():
    p = _make_persona(idx=2, seed=17)
    card = p.full_card()
    # new card structure should have at least 5 lines
    # (one_liner / 兴趣 / 原型 / 常看 / 不爱 / 性格… bullets)
    lines = [l for l in card.splitlines() if l.strip()]
    assert len(lines) >= 5, f"expected ≥5 non-empty lines, got {len(lines)}:\n" + card


# -------------------------------------------------- backward compatibility


def test_existing_fields_preserved():
    """Original one_liner + interests + psych fields must still be present."""
    p = _make_persona()
    card = p.full_card()
    assert p.one_liner() in card, "one_liner must remain in the card"
    assert "兴趣" in card
    assert "性格" in card


def test_soul_agent_pool_still_builds():
    """SoulAgentPool integration — the enriched personas must still serialize."""
    from oransim.agents.soul import SoulAgentPool

    pop = _make_pop(N=80, seed=42)
    pool = SoulAgentPool(pop, n=20, seed=42)
    assert len(pool.personas) == 20
    sample = next(iter(pool.personas.values()))
    card = sample.full_card()
    assert "原型画像" in card and "常看内容" in card and "不爱看" in card
