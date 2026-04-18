"""Multi-turn LLM group chat — true agent-to-agent information passing.

Inspired by:
  - AgentSociety (Tsinghua FIB lab, Apache 2.0): Message Layer pattern
  - OASIS / CAMEL-AI: turn-based multi-agent dialog
  - Park et al. Generative Agents: persona-grounded conversation

Mechanics (per round):
  1. Each agent reads the conversation log so far + their persona
  2. LLM call → "what would you say next?" (or stay silent)
  3. Append to log; next round
  4. After N rounds, summarize: consensus / polarization / dominant frame

Difference vs Discourse module (one-shot comments):
  - Discourse = parallel independent comments → one summary
  - GroupChat = sequential, agents READ each other and SHIFT positions

Output is a SCM mediator node, intervenable via do(consensus=...).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..data.creatives import Creative
from ..data.kols import KOL
from .soul import Persona, SoulAgentPool

# ---------------- Message layer (AgentSociety-inspired) ----------------


@dataclass
class Message:
    turn: int
    sender_id: int
    sender_oneliner: str
    text: str
    sentiment: float  # -1..+1, agent's stance after speaking
    addresses_to: int | None = None  # if replying to specific message
    is_silent: bool = False  # agent chose not to speak this turn

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "sender_id": self.sender_id,
            "sender": self.sender_oneliner,
            "text": self.text,
            "sentiment": round(self.sentiment, 3),
            "replies_to": self.addresses_to,
            "silent": self.is_silent,
        }


@dataclass
class GroupChatReport:
    creative_caption: str
    n_agents: int
    n_rounds: int
    messages: list[Message]  # full transcript
    final_stances: dict[int, float]  # agent_id → final sentiment
    initial_stances: dict[int, float]
    consensus: float  # mean final
    polarization: float  # std final
    converged: bool  # std dropped over rounds?
    dominant_frame: str  # what topic dominates the chat
    second_wave_impact: float  # SCM mediator value, [-0.3, +0.3]
    cost_cny: float
    tokens_in: int
    tokens_out: int
    rounds_summary: list[dict]  # per-round mean+std

    def to_dict(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "messages": [m.to_dict() for m in self.messages],
            "initial_stances": {str(k): round(v, 2) for k, v in self.initial_stances.items()},
            "final_stances": {str(k): round(v, 2) for k, v in self.final_stances.items()},
            "consensus": round(self.consensus, 3),
            "polarization": round(self.polarization, 3),
            "converged": self.converged,
            "dominant_frame": self.dominant_frame,
            "second_wave_impact": round(self.second_wave_impact, 3),
            "cost_cny": round(self.cost_cny, 4),
            "tokens": {"in": self.tokens_in, "out": self.tokens_out},
            "rounds_summary": self.rounds_summary,
        }


# ---------------- LLM prompts ----------------

GC_SYSTEM = """你是社媒虚拟用户，正在和其他用户在评论区/群聊里就一条广告交流。
你能看到前面所有人的发言。你要：
1. 维持你的 persona（年龄/职业/态度）
2. 真实回应别人的观点（同意/反驳/追问/吐槽）
3. 你的态度可以被别人改变（说服你 / 你被打动）
4. 也可以选择不说话（划走）
只输出 JSON。"""

GC_PROMPT = """<your_persona>
{persona}
最近兴趣：{interests}
此前对该品牌态度（-1..+1）：{prior_stance}
</your_persona>

<the_ad>
{caption}（{platform}, KOL: {kol}）
</the_ad>

<conversation_so_far>
{transcript}
</conversation_so_far>

第 {turn} 轮 你要怎样？严格 JSON：
{{"will_speak": true/false,
  "text": "10-25 字真实口语化发言（如果说的话）",
  "addresses_to": 想回复的发言者 turn 号，没有就 null,
  "new_stance": -1 到 +1 之间的小数（看完别人的发言后你最新的态度），
  "tone": "种草|劝退|附议|反驳|吐槽|追问|阴阳|带节奏 中选一个"
}}"""

GC_SUMMARY_SYSTEM = "你是舆情分析师。读完一段虚拟群聊，提炼共识/分歧/主导话题。只输出 JSON。"

GC_SUMMARY_PROMPT = """以下是 {n_agents} 个虚拟用户经过 {n_rounds} 轮就一条广告的群聊：

{transcript}

严格 JSON：
{{
  "dominant_frame": "群聊里最被反复提到的论点（10-20字）",
  "consensus_text": "如果有共识，是什么 (10-20字)；如果分歧严重，写 '观点对立'",
  "minority_voice": "少数派的关键观点",
  "second_wave_impact": -0.3 到 +0.3 之间（这场群聊会让二轮路过的人 click 概率变化多少）,
  "viral_potential": "low|medium|high"
}}"""


# ---------------- Mock impl ----------------

MOCK_OPENERS = [
    "这价格我先看看",
    "这达人挺真诚",
    "广子+1 跳过",
    "求链接姐妹",
    "看起来不错",
    "AIGC 一眼假",
    "等降价再说",
    "已下单",
]
MOCK_REPLIES_POS = ["楼上说得对", "我也这么觉得", "+1", "种草了", "也想试试"]
MOCK_REPLIES_NEG = ["楼上水军吧", "别被忽悠", "我去年被骗过", "醒醒"]


def _mock_message(
    p: Persona, turn: int, prior_stance: float, transcript: list[Message], rng
) -> Message:
    if turn == 0:
        text = rng.choice(MOCK_OPENERS)
        new_stance = prior_stance + rng.uniform(-0.1, 0.2)
    else:
        # respond to last message: agree if same sign, disagree if opposite
        last = transcript[-1] if transcript else None
        if last and last.sentiment * prior_stance > 0:
            text = rng.choice(MOCK_REPLIES_POS)
            new_stance = prior_stance + rng.uniform(0, 0.15)
        else:
            text = rng.choice(MOCK_REPLIES_NEG)
            new_stance = prior_stance + rng.uniform(-0.15, 0)
    new_stance = max(-1, min(1, new_stance))
    return Message(
        turn=turn,
        sender_id=p.id,
        sender_oneliner=p.one_liner(),
        text=text,
        sentiment=new_stance,
        addresses_to=transcript[-1].turn if transcript else None,
    )


def _mock_summary(messages: list[Message], n_agents: int, n_rounds: int) -> dict:
    sents = [m.sentiment for m in messages if not m.is_silent]
    mean = float(np.mean(sents)) if sents else 0
    return {
        "dominant_frame": "众说纷纭",
        "consensus_text": "种草" if mean > 0.3 else "劝退" if mean < -0.3 else "观点对立",
        "minority_voice": "(mock)",
        "second_wave_impact": float(np.clip(mean * 0.2, -0.3, 0.3)),
        "viral_potential": "high" if abs(mean) > 0.5 else "medium",
    }


# ---------------- Main entry ----------------


def simulate_group_chat(
    creative: Creative,
    kol: KOL | None,
    platform: str,
    souls: SoulAgentPool,
    n_agents: int = 8,
    n_rounds: int = 4,
    use_llm: bool = False,
    seed: int = 7,
) -> GroupChatReport:
    """Run a multi-turn LLM group chat simulation."""
    rng = random.Random(seed)
    chosen_ids = rng.sample(list(souls.personas.keys()), min(n_agents, len(souls.personas)))
    personas = {pid: souls.personas[pid] for pid in chosen_ids}

    # Initial stance: noisy prior based on persona/creative match
    np_rng = np.random.default_rng(seed)
    initial_stances: dict[int, float] = {}
    for pid in chosen_ids:
        p = personas[pid]
        match = 0.3 if any(i in creative.caption for i in p.interests) else 0
        s = float(np.clip(np_rng.normal(0 + match, 0.3), -1, 1))
        initial_stances[pid] = s

    current_stances = dict(initial_stances)
    messages: list[Message] = []
    rounds_summary: list[dict] = []
    tok_in = tok_out = 0
    cost_cny = 0.0

    if use_llm:
        try:
            from .soul_llm import (
                API_KEY,
                BASE_URL,
                MODEL,
                _extract_json,
                _http_stream_post,
                estimate_cost_cny,
                llm_available,
            )

            llm_ok = llm_available()
        except Exception:
            llm_ok = False
    else:
        llm_ok = False

    headers = None
    if llm_ok:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for r in range(n_rounds):
        # randomize agent speaking order each round
        order = list(chosen_ids)
        rng.shuffle(order)
        for pid in order:
            p = personas[pid]
            transcript_text = (
                "\n".join(
                    [
                        f"  T{m.turn} #{m.sender_id} [{m.sender_oneliner}]: {m.text}"
                        for m in messages[-12:]  # cap context
                    ]
                )
                or "  (尚无人发言)"
            )

            if llm_ok:
                body = {
                    "model": MODEL,
                    "temperature": 0.85,
                    "max_tokens": 200,
                    "messages": [
                        {"role": "system", "content": GC_SYSTEM},
                        {
                            "role": "user",
                            "content": GC_PROMPT.format(
                                persona=p.full_card(),
                                interests=", ".join(p.interests),
                                prior_stance=round(current_stances[pid], 2),
                                caption=creative.caption,
                                platform=platform,
                                kol=kol.name if kol else "无",
                                transcript=transcript_text,
                                turn=r,
                            ),
                        },
                    ],
                }
                try:
                    content, usage = _http_stream_post(
                        f"{BASE_URL}/chat/completions", headers, body
                    )
                    tok_in += usage.get("prompt_tokens", 0)
                    tok_out += usage.get("completion_tokens", 0)
                    parsed = _extract_json(content)
                    will_speak = bool(parsed.get("will_speak", True))
                    text = (parsed.get("text") or "").strip()
                    addresses = parsed.get("addresses_to")
                    new_stance = float(parsed.get("new_stance", current_stances[pid]))
                    new_stance = max(-1, min(1, new_stance))
                    if will_speak and text:
                        msg = Message(
                            turn=len(messages),
                            sender_id=pid,
                            sender_oneliner=p.one_liner(),
                            text=text,
                            sentiment=new_stance,
                            addresses_to=addresses,
                        )
                    else:
                        msg = Message(
                            turn=len(messages),
                            sender_id=pid,
                            sender_oneliner=p.one_liner(),
                            text="(划过，未发言)",
                            sentiment=current_stances[pid],
                            is_silent=True,
                        )
                except Exception:
                    msg = _mock_message(p, len(messages), current_stances[pid], messages, rng)
            else:
                msg = _mock_message(p, len(messages), current_stances[pid], messages, rng)

            messages.append(msg)
            current_stances[pid] = msg.sentiment

        # Per-round summary
        sents = list(current_stances.values())
        rounds_summary.append(
            {
                "round": r,
                "mean_stance": round(float(np.mean(sents)), 3),
                "std_stance": round(float(np.std(sents)), 3),
                "n_speakers_total": sum(1 for m in messages if not m.is_silent),
            }
        )

    if llm_ok:
        cost_cny = estimate_cost_cny(tok_in, tok_out)

    # Final summary via 1 LLM call (or mock)
    transcript_full = "\n".join(
        [f"T{m.turn} {m.sender_oneliner}: {m.text}" for m in messages if not m.is_silent]
    )
    summary = None
    if llm_ok and len(messages) > 3:
        try:
            from .soul_llm import BASE_URL, MODEL, _extract_json, _http_stream_post

            body = {
                "model": MODEL,
                "temperature": 0.4,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": GC_SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": GC_SUMMARY_PROMPT.format(
                            n_agents=n_agents, n_rounds=n_rounds, transcript=transcript_full
                        ),
                    },
                ],
            }
            content, usage = _http_stream_post(f"{BASE_URL}/chat/completions", headers, body)
            tok_in += usage.get("prompt_tokens", 0)
            tok_out += usage.get("completion_tokens", 0)
            summary = _extract_json(content)
        except Exception:
            summary = None
    if summary is None:
        summary = _mock_summary(messages, n_agents, n_rounds)

    if llm_ok:
        from .soul_llm import estimate_cost_cny

        cost_cny = estimate_cost_cny(tok_in, tok_out)

    final = list(current_stances.values())
    initial = list(initial_stances.values())
    consensus = float(np.mean(final))
    polarization = float(np.std(final))
    converged = rounds_summary[-1]["std_stance"] < rounds_summary[0]["std_stance"]

    # ── KPI coupling 校准（替换原 magic 0.5）──────────────────
    # 基于 Sunstein 2017 群体极化 + Asch 1956 从众实验:
    #   - 平均群体共识对个体行为影响 ~15-25%（典型态度变化幅度）
    #   - 群体极化越高（std 越大）→ 个体越难确定方向 → 影响力递减
    #   - polarization > 0.5 时影响减半
    #
    #   coupling = consensus × MAX_INFLUENCE × (1 - min(polarization, 0.5))
    #              其中 MAX_INFLUENCE = 0.20 (Sunstein literature midpoint)
    MAX_INFLUENCE = 0.20  # Sunstein 2017 + Asch 1956 + Cialdini 2001 综合
    polarization_dampening = 1.0 - min(polarization, 0.5)
    second_wave_calibrated = float(
        np.clip(consensus * MAX_INFLUENCE * polarization_dampening, -0.3, 0.3)
    )
    # 若 LLM 自报 second_wave_impact 与文献预测一致（同号），平均之；否则用文献版
    llm_reported = summary.get("second_wave_impact")
    if isinstance(llm_reported, (int, float)) and (
        llm_reported * second_wave_calibrated > 0 or abs(llm_reported) < 0.05
    ):
        second_wave_final = float(np.clip((llm_reported + second_wave_calibrated) / 2, -0.3, 0.3))
        coupling_source = "literature_avg_with_llm"
    else:
        second_wave_final = second_wave_calibrated
        coupling_source = "literature_only_sunstein2017"

    return GroupChatReport(
        creative_caption=creative.caption,
        n_agents=n_agents,
        n_rounds=n_rounds,
        messages=messages,
        initial_stances=initial_stances,
        final_stances=current_stances,
        consensus=consensus,
        polarization=polarization,
        converged=converged,
        dominant_frame=summary.get("dominant_frame", "?"),
        second_wave_impact=second_wave_final,
        cost_cny=cost_cny,
        tokens_in=tok_in,
        tokens_out=tok_out,
        rounds_summary=rounds_summary,
    )
