"""L2c 'Soul' agents — template-based mock LLM producing human-like reasoning.

Each soul agent has a persona card (inspired by Park et al. 2023 Generative Agents).
Given an impression, they output click/skip decision + a natural-language reason
+ a mock comment. In production, replace `SoulAgent.infer()` with a vLLM call
to Qwen3-4B-Instruct.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..data.creatives import Creative
from ..data.kols import KOL
from ..data.population import AGE_BUCKETS, CITY_TIER, OCCUPATION, Population

# City name pool per tier for richer personas
CITY_NAMES = {
    0: ["北京", "上海", "深圳", "广州"],
    1: ["杭州", "成都", "南京", "武汉", "重庆", "西安"],
    2: ["苏州", "厦门", "福州", "合肥", "济南", "长沙"],
    3: ["淮安", "九江", "绵阳", "烟台", "保定"],
    4: ["宿迁", "周口", "玉林", "达州", "驻马店"],
}


POSITIVE_REASONS = [
    "BGM 很抓耳",
    "封面颜色挺清爽",
    "达人之前买过 她推的不踩雷",
    "正好最近在看这个",
    "有真人试色 / 实拍",
    "文案挺真诚 不像硬广",
    "价格看着能接受",
    "剧情有点意思 想看完",
]
NEGATIVE_REASONS = [
    "看封面就像硬广 跳过",
    "BGM 吵死了",
    "这个达人以前翻过车 不太信",
    "价格一看就贵 不是我这个价位",
    "又是美颜滤镜 真货肯定没这效果",
    "老年感重 不是我的风格",
    "Z世代味太重 看不懂",
    "剧情拉胯",
    "AIGC 一眼假",
    "上周刷到过同款 腻了",
]
COMMENT_TEMPLATES_POS = [
    "求链接姐妹",
    "这个色我真的爱",
    "已下单",
    "种草了",
    "看起来不错等降价",
    "跟我家娃挺配",
]
COMMENT_TEMPLATES_NEG = [
    "广子+1",
    "又是恰饭",
    "看封面就跳过",
    "价格劝退",
]


# ── Phase V · persona_card enrichment ─────────────────────────────────────
# 把 persona_card 从 3 行扩到 ~6 行，多出来的内容全部从**同一份** interest /
# bigfive 向量派生。SYSTEM 提示保持不变（cache 折扣 + JSON schema 稳定）。
# LLM 拿到的 user message 因此更 grounded，级联出更贴真人的反应。

TAG_POOL: list[str] = [
    "美妆",
    "母婴",
    "数码",
    "美食",
    "穿搭",
    "健身",
    "理财",
    "旅行",
    "宠物",
    "汽车",
    "游戏",
    "读书",
    "咖啡",
    "家居",
]

# 8 个 archetype 原型。每个是 tag → 权重字典；对 persona 的 per-tag 激活做
# 加权打分，最高分对应的 label 写进 persona card。"通用中间派" 作为打分
# 差距不显著时的 fallback（最低阈值分）。
ARCHETYPE_ANCHORS: dict[str, dict[str, float]] = {
    "Z 世代美妆早鸟": {"美妆": 1.0, "穿搭": 0.8, "咖啡": 0.3, "旅行": 0.2},
    "职场妈妈价格敏感型": {"母婴": 1.0, "美食": 0.6, "家居": 0.5, "理财": 0.3},
    "理性科技中产": {"数码": 1.0, "理财": 0.7, "汽车": 0.5, "读书": 0.3},
    "健身精致生活派": {"健身": 1.0, "美食": 0.5, "美妆": 0.3, "旅行": 0.4},
    "文艺旅行向往者": {"旅行": 1.0, "读书": 0.7, "咖啡": 0.5, "美食": 0.3},
    "游戏宅数字土著": {"游戏": 1.0, "数码": 0.7, "汽车": 0.2},
    "宠物家居温柔派": {"宠物": 1.0, "家居": 0.8, "美食": 0.3},
    "通用中间派": {},  # fallback
}
ARCHETYPE_LABELS: list[str] = list(ARCHETYPE_ANCHORS.keys())


def _tag_activations(interest_vec: np.ndarray, signed: bool = False) -> dict[str, float]:
    """Collapse 64-d interest vector onto TAG_POOL via mod-grouping.

    signed=True keeps direction (用于 anchor/anti-anchor 区分偏好 / 反感)；
    signed=False 取绝对值（用于 archetype 打分，方向不相关）。
    """
    activations = {tag: 0.0 for tag in TAG_POOL}
    for d, v in enumerate(np.asarray(interest_vec).ravel()):
        activations[TAG_POOL[d % len(TAG_POOL)]] += float(v if signed else abs(v))
    return activations


def _pick_archetype(abs_activations: dict[str, float]) -> str:
    total_act = sum(abs_activations.values())
    # Zero / near-zero vector → 显式 fallback（避免任何 score=0 都排第一名的
    # edge case）。真实 population 不会走这支，但生产数据清洗时有 zero vec 的
    # 可能，不能静默分配成「Z 世代美妆早鸟」。
    if total_act < 0.05:
        return "通用中间派"

    # L1 归一化权重后比较：每个 archetype 的 score 是加权**平均**，档次拉平。
    # 没这一步时 tag 覆盖多的 archetype（比如理性科技中产 4 个 tag）会系统性
    # 压过 tag 少的（游戏宅 3 个 tag），分布上出现 31% vs 0.2% 的不平衡。
    best_label = "通用中间派"
    best_score = -1.0
    scores: dict[str, float] = {}
    for label, weights in ARCHETYPE_ANCHORS.items():
        if not weights:
            continue
        total_w = sum(abs(w) for w in weights.values())
        raw = sum(abs_activations.get(t, 0.0) * w for t, w in weights.items())
        score = raw / (total_w + 1e-8)
        scores[label] = score
        if score > best_score:
            best_score = score
            best_label = label

    # 如果第一名和第二名差距太小（< 2% 相对差），说明 persona 不典型，
    # 回退到通用中间派。阈值是 bench 过的——5% 时通用占 36% 太多，
    # 1% 时几乎没人 fallback，2% 是 15-20% 甜点。
    if len(scores) >= 2:
        ranked = sorted(scores.values(), reverse=True)
        if ranked[1] > 0 and (ranked[0] - ranked[1]) / ranked[1] < 0.02:
            return "通用中间派"
    return best_label


def _pick_anchors(signed_activations: dict[str, float], n: int = 3) -> list[str]:
    """最想看的 n 个 tag（signed activation 最高）。"""
    return [t for t, _ in sorted(signed_activations.items(), key=lambda kv: -kv[1])[:n]]


def _pick_anti_anchors(
    signed_activations: dict[str, float], n: int = 3, exclude: set = None
) -> list[str]:
    """最不想看的 n 个 tag（signed activation 最低且不在 exclude 集合里）。"""
    exclude = exclude or set()
    result: list[str] = []
    for t, _ in sorted(signed_activations.items(), key=lambda kv: kv[1]):
        if t in exclude:
            continue
        result.append(t)
        if len(result) >= n:
            break
    return result


def _psych_bullets(bigfive: np.ndarray) -> list[str]:
    """Bigfive 5D → 最多 4 个显著 facets。阈值对称（0.65 / 0.35）避免 Persona
    只显示「情绪稳定」这种单 facet 空洞输出。"""
    O, C, E, A, N = (float(bigfive[i]) for i in range(5))
    bits: list[str] = []
    if O > 0.65:
        bits.append("对新鲜事物特别敏感")
    elif O < 0.35:
        bits.append("倾向经过验证的选择")
    if C > 0.65:
        bits.append("注重细节 · 反复比价")
    elif C < 0.35:
        bits.append("冲动消费多")
    if E > 0.65:
        bits.append("爱分享 · 晒图评论频繁")
    elif E < 0.35:
        bits.append("沉默围观 · 很少留言")
    if A > 0.70:
        bits.append("容易被真诚文案打动")
    elif A < 0.30:
        bits.append("对广告本能反感")
    if N > 0.65:
        bits.append("容易被焦虑痛点驱动")
    elif N < 0.35:
        bits.append("情绪稳定 · 不追热点")
    if not bits:
        bits.append("典型佛系 · 信息密度敏感")
    return bits[:4]


@dataclass
class Persona:
    id: int  # population index
    age: int
    gender: str
    city: str
    city_tier: str
    occupation: str
    income_tier: int
    interests: list[str]
    psych: str
    # Phase V · 向量派生的四项 enrichment
    archetype: str = "通用中间派"
    anchors: list[str] = field(default_factory=list)
    anti_anchors: list[str] = field(default_factory=list)
    psych_bullets: list[str] = field(default_factory=list)

    def one_liner(self) -> str:
        return (
            f"{self.age}岁{self.gender}·{self.city}·{self.occupation}·月入分位{self.income_tier}/10"
        )

    def full_card(self) -> str:
        lines = [
            self.one_liner(),
            f"兴趣：{', '.join(self.interests)}",
            f"原型画像：{self.archetype}",
            f"常看内容：{' / '.join(self.anchors) if self.anchors else '（无显著偏好）'}",
            f"不爱看：{' / '.join(self.anti_anchors) if self.anti_anchors else '（无显著反感）'}",
            f"性格：{self.psych}",
        ]
        if self.psych_bullets:
            lines.append("性格特征：\n  • " + "\n  • ".join(self.psych_bullets))
        return "\n".join(lines)


def build_persona(pop: Population, idx: int, rng: np.random.Generator) -> Persona:
    age_bucket = AGE_BUCKETS[pop.age_idx[idx]]
    # map bucket → fake concrete age
    lo, hi = {
        "15-24": (17, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65+": (65, 72),
    }[age_bucket]
    age = int(rng.integers(lo, hi + 1))
    gender = "女" if pop.gender_idx[idx] == 0 else "男"
    city_tier = CITY_TIER[pop.city_idx[idx]]
    city_list = CITY_NAMES.get(pop.city_idx[idx], ["某县城"])
    city = str(rng.choice(city_list))
    occ = OCCUPATION[pop.occ_idx[idx]]

    interest_vec = pop.interest[idx]
    # Phase V · 向量派生（tag-level 统一口径，避免 dim-level 和 tag-level 打架）
    abs_act = _tag_activations(interest_vec, signed=False)
    signed_act = _tag_activations(interest_vec, signed=True)

    # interests: top-5 tags by **signed tag-level activation**（把 64 dim 按
    # mod-14 归到 tag 上加和后排序）。放弃原先的 dim-level argsort，避免一个
    # tag 的净活跃度为负、却因为某个大正 dim 挤进 interests、然后又在 tag 净和
    # 上进 anti_anchors 的矛盾。downstream infer_one 的 niche match 因此自然地
    # 变成「正向品类匹配」。
    interests = [t for t, _ in sorted(signed_act.items(), key=lambda kv: -kv[1])[:5]]

    # psych from big five（保留原 one-line 语义用于 backward compat）
    bf = pop.bigfive[idx]
    psych_bits = []
    if bf[0] > 0.7:
        psych_bits.append("喜欢尝新")
    if bf[0] < 0.3:
        psych_bits.append("偏保守")
    if bf[2] > 0.7:
        psych_bits.append("外向爱分享")
    if bf[4] > 0.7:
        psych_bits.append("容易焦虑")
    if bf[4] < 0.3:
        psych_bits.append("情绪稳定")
    if not psych_bits:
        psych_bits = ["佛系"]

    archetype = _pick_archetype(abs_act)
    anchors = _pick_anchors(signed_act, n=3)
    anti_anchors = _pick_anti_anchors(signed_act, n=3, exclude=set(anchors))
    psych_bullets = _psych_bullets(bf)

    return Persona(
        id=int(idx),
        age=age,
        gender=gender,
        city=city,
        city_tier=city_tier,
        occupation=occ,
        income_tier=int(pop.income[idx]),
        interests=interests,
        psych="，".join(psych_bits),
        archetype=archetype,
        anchors=anchors,
        anti_anchors=anti_anchors,
        psych_bullets=psych_bullets,
    )


class SoulAgentPool:
    """A small set of persona-backed agents that produce qualitative reasoning."""

    def __init__(self, population: Population, n: int = 30, seed: int = 123):
        self.pop = population
        rng = np.random.default_rng(seed)
        # Sample stratified by gender × city tier to get diverse personas
        self.idx = rng.choice(population.N, size=n, replace=False)
        self.personas: dict[int, Persona] = {
            int(i): build_persona(population, int(i), rng) for i in self.idx
        }

    def infer_one(
        self,
        persona_id: int,
        creative: Creative,
        click_prob: float,
        kol: KOL | None,
        platform: str,
        rng: random.Random,
    ) -> dict:
        p = self.personas[persona_id]
        # Decide click: sample from click_prob, but soul agent can veto/accept
        base = click_prob
        if kol and any(
            n in p.interests
            for n in [
                kol.niche,
                {
                    "beauty": "美妆",
                    "mom": "母婴",
                    "tech": "数码",
                    "food": "美食",
                    "fashion": "穿搭",
                    "fitness": "健身",
                    "finance": "理财",
                    "travel": "旅行",
                }.get(kol.niche, ""),
            ]
        ):
            base = min(1.0, base * 1.4)
        will_click = rng.random() < base

        if will_click:
            reason = rng.choice(POSITIVE_REASONS)
            comment = rng.choice(COMMENT_TEMPLATES_POS)
            feel = rng.choice(["好奇", "心动", "购买冲动"])
            intent = round(min(0.95, 0.3 + base + rng.uniform(-0.1, 0.2)), 2)
        else:
            reason = rng.choice(NEGATIVE_REASONS)
            comment = rng.choice(COMMENT_TEMPLATES_NEG) if rng.random() < 0.3 else ""
            feel = rng.choice(["无感", "厌恶", "无感"])
            intent = round(max(0.02, 0.15 - (1 - base)), 2)

        return {
            "persona_id": persona_id,
            "persona_oneliner": p.one_liner(),
            "persona_card": p.full_card(),
            "will_click": will_click,
            "reason": reason,
            "comment": comment,
            "feel": feel,
            "purchase_intent_7d": intent,
        }

    def infer_batch(
        self,
        creative: Creative,
        outcome_click_probs: dict[int, float],
        kol: KOL | None,
        platform: str,
        n_sample: int = 10,
        seed: int = 7,
        use_llm: bool = False,
    ) -> list[dict]:
        """Pick n_sample souls and get their reasoning.

        If use_llm=True and env LLM_MODE=api with LLM_API_KEY set, calls a real LLM
        (DeepSeek / Qwen / GPT / local vLLM — OpenAI-compatible endpoint).
        Otherwise falls back to template mock.
        """
        rng = random.Random(seed)
        candidates = [pid for pid in self.personas if pid in outcome_click_probs]
        if len(candidates) < n_sample:
            candidates = list(self.personas.keys())
        chosen = rng.sample(candidates, min(n_sample, len(candidates)))

        if use_llm:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from . import async_pool, llm_dedup, stream_memory
            from .soul_llm import estimate_cost_cny, llm_available, soul_infer_llm

            if llm_available():
                kol_kwargs = dict(
                    caption=creative.caption,
                    platform=platform,
                    kol_name=kol.name if kol else "无",
                    kol_niche=kol.niche if kol else "通用",
                    kol_fans=kol.fan_count if kol else 0,
                    visual=creative.visual_style,
                    music=creative.music_mood,
                    duration=creative.duration_sec,
                )
                memstore = stream_memory.store()

                def _post(pid, r):
                    p = self.personas[pid]
                    if "_error" in r:
                        r = self.infer_one(
                            pid, creative, outcome_click_probs.get(pid, 0.05), kol, platform, rng
                        )
                        r["source"] = "mock-fallback"
                    else:
                        r["source"] = "llm"
                        r["persona_id"] = pid
                        r["persona_oneliner"] = p.one_liner()
                        r["persona_card"] = p.full_card()
                    if stream_memory.enabled():
                        memstore.record_event(
                            pid,
                            kind="ad_exposure",
                            content=f"看到{platform}广告:{creative.caption[:24]}",
                            metadata={
                                "platform": platform,
                                "will_click": bool(r.get("will_click")),
                                "feel": r.get("feel"),
                                "intent": r.get("purchase_intent_7d"),
                            },
                            profile={
                                "age": p.age,
                                "gender": p.gender,
                                "city": p.city,
                                "occupation": p.occupation,
                                "interests": p.interests,
                            },
                        )
                        if r.get("reason"):
                            memstore.record_perception(pid, thought=str(r["reason"]))
                    return pid, r

                # ── Path A: async pool (env ASYNC_POOL=1) ────────────
                if async_pool.enabled():
                    jobs = []
                    for pid in chosen:
                        p = self.personas[pid]
                        hint = ""
                        if stream_memory.enabled():
                            mem = memstore.get(pid)
                            if mem is not None:
                                hint = mem.summary_for_prompt()
                        jobs.append({"persona": p, "memory_hint": hint, **kol_kwargs})
                    raw = async_pool.run_batch_sync(jobs)
                    results_map = {}
                    total_in = total_out = 0
                    for pid, r in zip(chosen, raw, strict=False):
                        pid, r = _post(pid, r)
                        results_map[pid] = r
                        total_in += r.get("_tokens_in", 0)
                        total_out += r.get("_tokens_out", 0)
                    if stream_memory.enabled():
                        memstore.flush(list(results_map.keys()))
                    results = [results_map[pid] for pid in chosen]
                    cost_cny = estimate_cost_cny(total_in, total_out)
                    if results:
                        results[0]["_batch_cost_cny"] = round(cost_cny, 4)
                        results[0]["_batch_tokens"] = {"in": total_in, "out": total_out}
                        results[0]["_batch_engine"] = "async_pool"
                        results[0]["_dedup"] = (
                            llm_dedup.dedup_stats() if llm_dedup.enabled() else None
                        )
                    return results

                # ── Path B: thread pool + optional dedup ─────────────
                def call_one(pid):
                    p = self.personas[pid]
                    if llm_dedup.enabled():
                        key = llm_dedup.make_key(
                            p.full_card(),
                            creative.caption,
                            platform,
                            kol_kwargs["kol_name"],
                            creative.visual_style,
                            creative.music_mood,
                            creative.duration_sec,
                        )
                        r = llm_dedup.dedup_call(
                            key,
                            lambda: soul_infer_llm(persona=p, **kol_kwargs),
                        )
                    else:
                        r = soul_infer_llm(persona=p, **kol_kwargs)
                    return _post(pid, r)

                workers = min(
                    int(__import__("os").environ.get("LLM_CONCURRENCY", "15")), len(chosen)
                )
                results_map = {}
                total_in = total_out = 0
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(call_one, pid) for pid in chosen]
                    for f in as_completed(futs):
                        pid, r = f.result()
                        results_map[pid] = r
                        total_in += r.get("_tokens_in", 0)
                        total_out += r.get("_tokens_out", 0)
                if stream_memory.enabled():
                    memstore.flush(list(results_map.keys()))
                results = [results_map[pid] for pid in chosen]
                cost_cny = estimate_cost_cny(total_in, total_out)
                if results:
                    results[0]["_batch_cost_cny"] = round(cost_cny, 4)
                    results[0]["_batch_tokens"] = {"in": total_in, "out": total_out}
                    results[0]["_batch_workers"] = workers
                    results[0]["_batch_engine"] = "thread_pool"
                    results[0]["_dedup"] = llm_dedup.dedup_stats() if llm_dedup.enabled() else None
                return results

        # mock path
        results = []
        for pid in chosen:
            cp = outcome_click_probs.get(pid, 0.05)
            r = self.infer_one(pid, creative, cp, kol, platform, rng)
            r["source"] = "mock"
            results.append(r)
        return results
