"""A. Social-discourse simulator + SCM mediator.

The killer feature: after a first-wave exposure, a subset of engaged LLM souls
write **comments** (via GPT); another LLM summarizes the "dominant sentiment"
of the comment section; that sentiment becomes a **mediator node** in the SCM
that modulates how second-wave viewers react.

This lets us answer Pearl-L3 counterfactuals like:
  "If the top comments had been positive instead of negative, what's the ROI?"
  → do(comment_sentiment = +0.5) → rerun pipeline → CATE over swap

Beats MiroFish because:
  - MiroFish agents debate, but you can't mathematically intervene on
    collective sentiment. We can (via SCM + do-operator).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from ..data.creatives import Creative
from ..data.kols import KOL
from .soul import Persona, SoulAgentPool

COMMENT_SYSTEM = """你是社媒用户，刚刚在广告下发评论。根据 persona 和素材内容，写一条真实、口语化、带情绪的评论。只输出 JSON。"""

COMMENT_PROMPT = """persona: {persona}
我刚看了这条广告：{caption}
达人：{kol}
平台：{platform}

写一条 10-30 字的评论（真实口语化）。要反映我的 persona（年龄 / 职业 / 态度）。
严格 JSON：{{
  "comment": "...",
  "sentiment": -1 到 1 之间的小数（-1 极负面，0 中立，+1 极正面）,
  "tone": "种草/劝退/吐槽/求链接/求助/抬杠/酸/阴阳/真诚推荐 中选一个",
  "key_signal": "评论里最关键的一个信号词（比如 '太贵' / '真的好用' / '翻车' / '广子')"
}}"""

SUMMARY_SYSTEM = "你是舆情分析师。收到一批评论，提炼主导情绪、主要争议点、品牌信号。只输出 JSON。"

SUMMARY_PROMPT = """以下是 {n} 条广告评论：

{comments}

严格 JSON：
{{
  "dominant_sentiment": -1 到 1 之间的小数,
  "sentiment_variance": 0 到 1 之间（分歧度，0=一致，1=两极撕裂）,
  "top_objections": ["最负面的 3 个关切", ...],
  "top_praises": ["最正面的 3 个亮点", ...],
  "viral_tone": "会 viral 的 tone 比如 '大家都在求链接' 或 '骂翻了' 或 '无感划过' ",
  "expected_second_wave_impact": -0.3 到 +0.3（对二轮曝光者的点击概率影响）
}}"""


@dataclass
class CommentVerdict:
    persona_id: int
    persona_oneliner: str
    comment: str
    sentiment: float
    tone: str
    key_signal: str
    source: str = "mock"


@dataclass
class DiscourseReport:
    comments: list[CommentVerdict]
    dominant_sentiment: float
    sentiment_variance: float
    top_objections: list[str]
    top_praises: list[str]
    viral_tone: str
    second_wave_impact: float  # -0.3 .. +0.3, added to click logit
    source: str  # "llm" or "mock"
    cost_cny: float
    tokens_in: int
    tokens_out: int


MOCK_COMMENTS = {
    "positive": [
        ("求链接", 0.8, "求链接", "求链接"),
        ("已下单 期待一下", 0.7, "种草", "已下单"),
        ("测评看起来真实 比其他家真诚", 0.6, "真诚推荐", "真实"),
        ("这个价位还能接受 买了", 0.55, "种草", "价位接受"),
    ],
    "negative": [
        ("广子+1 跳过", -0.6, "劝退", "广子"),
        ("一看就贵 劝退", -0.7, "劝退", "太贵"),
        ("达人以前翻过车 不太信了", -0.65, "吐槽", "翻车"),
        ("又是套路 不点", -0.5, "阴阳", "套路"),
    ],
    "neutral": [
        ("看起来还行等降价", 0.1, "吐槽", "等降价"),
        ("有机会试试", 0.15, "真诚推荐", "可能试试"),
    ],
}


def _mock_comment_for_persona(p: Persona, creative: Creative, rng) -> CommentVerdict:
    # positive if persona.interests overlaps with creative caption words
    positive_bias = 0.4 if any(i in creative.caption for i in p.interests) else 0.0
    bucket = (
        "positive"
        if rng.random() < 0.4 + positive_bias
        else "negative" if rng.random() < 0.35 else "neutral"
    )
    c, s, t, k = MOCK_COMMENTS[bucket][rng.randint(0, len(MOCK_COMMENTS[bucket]) - 1)]
    return CommentVerdict(
        persona_id=p.id,
        persona_oneliner=p.one_liner(),
        comment=c,
        sentiment=s + rng.uniform(-0.1, 0.1),
        tone=t,
        key_signal=k,
        source="mock",
    )


def _mock_summary(comments: list[CommentVerdict]) -> DiscourseReport:
    sent = np.array([c.sentiment for c in comments], dtype=np.float32) if comments else np.zeros(1)
    dom = float(sent.mean()) if comments else 0.0
    var = float(sent.std()) if len(comments) > 1 else 0.0
    pos = [c.comment for c in comments if c.sentiment > 0.3][:3]
    neg = [c.comment for c in comments if c.sentiment < -0.3][:3]
    viral = (
        "大家都在求链接" if dom > 0.4 else "评论区吐槽为主" if dom < -0.2 else "路人划过 没啥情绪"
    )
    return DiscourseReport(
        comments=comments,
        dominant_sentiment=dom,
        sentiment_variance=var,
        top_objections=neg[:3] or ["无突出负面"],
        top_praises=pos[:3] or ["无突出正面"],
        viral_tone=viral,
        second_wave_impact=float(np.clip(dom * 0.25, -0.3, 0.3)),
        source="mock",
        cost_cny=0.0,
        tokens_in=0,
        tokens_out=0,
    )


def simulate_discourse_mock(
    creative: Creative,
    kol: KOL | None,
    platform: str,
    souls: SoulAgentPool,
    n_commenters: int = 15,
    seed: int = 7,
) -> DiscourseReport:
    import random

    rng = random.Random(seed)
    chosen = rng.sample(list(souls.personas.keys()), min(n_commenters, len(souls.personas)))
    comments = [_mock_comment_for_persona(souls.personas[pid], creative, rng) for pid in chosen]
    return _mock_summary(comments)


def simulate_discourse_llm(
    creative: Creative,
    kol: KOL | None,
    platform: str,
    souls: SoulAgentPool,
    n_commenters: int = 15,
    seed: int = 7,
) -> DiscourseReport:
    """LLM version: parallel comments → aggregated summary."""
    import os
    import random

    from .soul_llm import (
        MODEL,
        call_llm_json_with_retry,
        estimate_cost_cny,
        llm_available,
    )

    if not llm_available():
        return simulate_discourse_mock(creative, kol, platform, souls, n_commenters, seed)

    rng = random.Random(seed)
    chosen = rng.sample(list(souls.personas.keys()), min(n_commenters, len(souls.personas)))

    workers = int(os.environ.get("LLM_CONCURRENCY", "15"))
    kol_name = f"{kol.name}({kol.niche})" if kol else "(无达人)"

    tok_in = tok_out = 0

    def call_one(pid):
        p = souls.personas[pid]
        body = {
            "model": MODEL,
            "temperature": 0.8,
            "max_tokens": 150,
            "messages": [
                {"role": "system", "content": COMMENT_SYSTEM},
                {
                    "role": "user",
                    "content": COMMENT_PROMPT.format(
                        persona=p.full_card(),
                        caption=creative.caption,
                        kol=kol_name,
                        platform=platform,
                    ),
                },
            ],
        }
        try:
            # Retry wrapper handles network flake + malformed JSON
            r, usage = call_llm_json_with_retry(body, max_retries=2)
            return (
                pid,
                CommentVerdict(
                    persona_id=pid,
                    persona_oneliner=p.one_liner(),
                    comment=r.get("comment", ""),
                    sentiment=float(r.get("sentiment", 0)),
                    tone=r.get("tone", "?"),
                    key_signal=r.get("key_signal", ""),
                    source="llm",
                ),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
        except Exception:
            return pid, _mock_comment_for_persona(p, creative, rng), 0, 0

    comments: list[CommentVerdict] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(chosen))) as ex:
        futs = [ex.submit(call_one, pid) for pid in chosen]
        for f in as_completed(futs):
            pid, cv, t_in, t_out = f.result()
            comments.append(cv)
            tok_in += t_in
            tok_out += t_out

    # Summarize via 1 more LLM call
    summary = None
    if len(comments) >= 3:
        comments_text = "\n".join([f"- [{c.tone}] {c.comment}" for c in comments])
        body = {
            "model": MODEL,
            "temperature": 0.3,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(n=len(comments), comments=comments_text),
                },
            ],
        }
        try:
            summary, usage = call_llm_json_with_retry(body, max_retries=2)
            tok_in += usage.get("prompt_tokens", 0)
            tok_out += usage.get("completion_tokens", 0)
        except Exception:
            summary = None

    if summary:
        return DiscourseReport(
            comments=comments,
            dominant_sentiment=float(summary.get("dominant_sentiment", 0)),
            sentiment_variance=float(summary.get("sentiment_variance", 0)),
            top_objections=summary.get("top_objections", [])[:3] or ["(无)"],
            top_praises=summary.get("top_praises", [])[:3] or ["(无)"],
            viral_tone=summary.get("viral_tone", "?"),
            second_wave_impact=float(
                np.clip(summary.get("expected_second_wave_impact", 0), -0.3, 0.3)
            ),
            source="llm",
            cost_cny=estimate_cost_cny(tok_in, tok_out),
            tokens_in=tok_in,
            tokens_out=tok_out,
        )
    # fallback summarization
    rep = _mock_summary(comments)
    rep.source = "llm+mock_summary"
    rep.cost_cny = estimate_cost_cny(tok_in, tok_out)
    rep.tokens_in = tok_in
    rep.tokens_out = tok_out
    return rep


def apply_discourse_to_second_wave(
    click_logit: np.ndarray,
    discourse: DiscourseReport,
    second_wave_fraction: float = 0.5,
) -> np.ndarray:
    """Modulate click_logit of second-wave viewers by discourse sentiment.

    We assume `second_wave_fraction` of the given logits belong to the second
    wave (sequential exposure); we add the discourse impact to those only.

    For simplicity we apply to all (in prod, split by impression timestamp).
    """
    return click_logit + discourse.second_wave_impact * second_wave_fraction


def discourse_to_dict(d: DiscourseReport) -> dict:
    return {
        "source": d.source,
        "n_comments": len(d.comments),
        "dominant_sentiment": round(d.dominant_sentiment, 3),
        "sentiment_variance": round(d.sentiment_variance, 3),
        "top_objections": d.top_objections,
        "top_praises": d.top_praises,
        "viral_tone": d.viral_tone,
        "second_wave_click_delta": round(d.second_wave_impact, 3),
        "cost_cny": round(d.cost_cny, 4),
        "tokens": {"in": d.tokens_in, "out": d.tokens_out},
        "comments": [
            {
                "persona": c.persona_oneliner,
                "text": c.comment,
                "sentiment": round(c.sentiment, 2),
                "tone": c.tone,
                "signal": c.key_signal,
                "source": c.source,
            }
            for c in d.comments
        ],
    }
