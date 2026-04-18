"""L2c 'Soul' agents — template-based mock LLM producing human-like reasoning.

Each soul agent has a persona card (inspired by Park et al. 2023 Generative Agents).
Given an impression, they output click/skip decision + a natural-language reason
+ a mock comment. In production, replace `SoulAgent.infer()` with a vLLM call
to Qwen3-4B-Instruct.
"""
from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List

from ..data.population import Population, AGE_BUCKETS, GENDER, CITY_TIER, OCCUPATION
from ..data.creatives import Creative
from ..data.kols import KOL


# City name pool per tier for richer personas
CITY_NAMES = {
    0: ["北京", "上海", "深圳", "广州"],
    1: ["杭州", "成都", "南京", "武汉", "重庆", "西安"],
    2: ["苏州", "厦门", "福州", "合肥", "济南", "长沙"],
    3: ["淮安", "九江", "绵阳", "烟台", "保定"],
    4: ["宿迁", "周口", "玉林", "达州", "驻马店"],
}


POSITIVE_REASONS = [
    "BGM 很抓耳", "封面颜色挺清爽", "达人之前买过 她推的不踩雷",
    "正好最近在看这个", "有真人试色 / 实拍", "文案挺真诚 不像硬广",
    "价格看着能接受", "剧情有点意思 想看完",
]
NEGATIVE_REASONS = [
    "看封面就像硬广 跳过", "BGM 吵死了", "这个达人以前翻过车 不太信",
    "价格一看就贵 不是我这个价位", "又是美颜滤镜 真货肯定没这效果",
    "老年感重 不是我的风格", "Z世代味太重 看不懂", "剧情拉胯",
    "AIGC 一眼假", "上周刷到过同款 腻了",
]
COMMENT_TEMPLATES_POS = [
    "求链接姐妹", "这个色我真的爱", "已下单", "种草了",
    "看起来不错等降价", "跟我家娃挺配",
]
COMMENT_TEMPLATES_NEG = [
    "广子+1", "又是恰饭", "看封面就跳过", "价格劝退",
]


@dataclass
class Persona:
    id: int                 # population index
    age: int
    gender: str
    city: str
    city_tier: str
    occupation: str
    income_tier: int
    interests: List[str]
    psych: str

    def one_liner(self) -> str:
        return f"{self.age}岁{self.gender}·{self.city}·{self.occupation}·月入分位{self.income_tier}/10"

    def full_card(self) -> str:
        return (
            f"{self.one_liner()}\n"
            f"兴趣：{', '.join(self.interests)}\n"
            f"性格：{self.psych}"
        )


def build_persona(pop: Population, idx: int, rng: np.random.Generator) -> Persona:
    age_bucket = AGE_BUCKETS[pop.age_idx[idx]]
    # map bucket → fake concrete age
    lo, hi = {
        "15-24":(17,24),"25-34":(25,34),"35-44":(35,44),
        "45-54":(45,54),"55-64":(55,64),"65+":(65,72)
    }[age_bucket]
    age = int(rng.integers(lo, hi+1))
    gender = "女" if pop.gender_idx[idx] == 0 else "男"
    city_tier = CITY_TIER[pop.city_idx[idx]]
    city_list = CITY_NAMES.get(pop.city_idx[idx], ["某县城"])
    city = str(rng.choice(city_list))
    occ = OCCUPATION[pop.occ_idx[idx]]

    # interests: top-5 dims of interest vector → symbolic tags
    top_dims = np.argsort(-np.abs(pop.interest[idx]))[:5]
    tag_pool = ["美妆", "母婴", "数码", "美食", "穿搭", "健身", "理财",
                "旅行", "宠物", "汽车", "游戏", "读书", "咖啡", "家居"]
    interests = [tag_pool[d % len(tag_pool)] for d in top_dims]

    # psych from big five
    bf = pop.bigfive[idx]
    psych_bits = []
    if bf[0] > 0.7: psych_bits.append("喜欢尝新")
    if bf[0] < 0.3: psych_bits.append("偏保守")
    if bf[2] > 0.7: psych_bits.append("外向爱分享")
    if bf[4] > 0.7: psych_bits.append("容易焦虑")
    if bf[4] < 0.3: psych_bits.append("情绪稳定")
    if not psych_bits: psych_bits = ["佛系"]

    return Persona(
        id=int(idx), age=age, gender=gender, city=city,
        city_tier=city_tier, occupation=occ,
        income_tier=int(pop.income[idx]),
        interests=interests, psych="，".join(psych_bits),
    )


class SoulAgentPool:
    """A small set of persona-backed agents that produce qualitative reasoning."""

    def __init__(self, population: Population, n: int = 30, seed: int = 123):
        self.pop = population
        rng = np.random.default_rng(seed)
        # Sample stratified by gender × city tier to get diverse personas
        self.idx = rng.choice(population.N, size=n, replace=False)
        self.personas: Dict[int, Persona] = {
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
    ) -> Dict:
        p = self.personas[persona_id]
        # Decide click: sample from click_prob, but soul agent can veto/accept
        base = click_prob
        if kol and any(n in p.interests for n in [kol.niche, {"beauty":"美妆","mom":"母婴","tech":"数码","food":"美食","fashion":"穿搭","fitness":"健身","finance":"理财","travel":"旅行"}.get(kol.niche,"")]):
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
        outcome_click_probs: Dict[int, float],
        kol: KOL | None,
        platform: str,
        n_sample: int = 10,
        seed: int = 7,
        use_llm: bool = False,
    ) -> List[Dict]:
        """Pick n_sample souls and get their reasoning.

        If use_llm=True and env LLM_MODE=api with LLM_API_KEY set, calls a real LLM
        (DeepSeek / Qwen / GPT / local vLLM — OpenAI-compatible endpoint).
        Otherwise falls back to template mock.
        """
        rng = random.Random(seed)
        candidates = [pid for pid in self.personas.keys() if pid in outcome_click_probs]
        if len(candidates) < n_sample:
            candidates = list(self.personas.keys())
        chosen = rng.sample(candidates, min(n_sample, len(candidates)))

        if use_llm:
            from .soul_llm import llm_available, soul_infer_llm, estimate_cost_cny
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from . import llm_dedup, async_pool, stream_memory
            if llm_available():
                kol_kwargs = dict(
                    caption=creative.caption, platform=platform,
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
                        r = self.infer_one(pid, creative,
                                           outcome_click_probs.get(pid, 0.05),
                                           kol, platform, rng)
                        r["source"] = "mock-fallback"
                    else:
                        r["source"] = "llm"
                        r["persona_id"] = pid
                        r["persona_oneliner"] = p.one_liner()
                        r["persona_card"] = p.full_card()
                    if stream_memory.enabled():
                        memstore.record_event(
                            pid, kind="ad_exposure",
                            content=f"看到{platform}广告:{creative.caption[:24]}",
                            metadata={
                                "platform": platform,
                                "will_click": bool(r.get("will_click")),
                                "feel": r.get("feel"),
                                "intent": r.get("purchase_intent_7d"),
                            },
                            profile={
                                "age": p.age, "gender": p.gender, "city": p.city,
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
                    for pid, r in zip(chosen, raw):
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
                        results[0]["_dedup"] = llm_dedup.dedup_stats() if llm_dedup.enabled() else None
                    return results

                # ── Path B: thread pool + optional dedup ─────────────
                def call_one(pid):
                    p = self.personas[pid]
                    if llm_dedup.enabled():
                        key = llm_dedup.make_key(
                            p.full_card(), creative.caption, platform,
                            kol_kwargs["kol_name"], creative.visual_style,
                            creative.music_mood, creative.duration_sec,
                        )
                        r = llm_dedup.dedup_call(
                            key,
                            lambda: soul_infer_llm(persona=p, **kol_kwargs),
                        )
                    else:
                        r = soul_infer_llm(persona=p, **kol_kwargs)
                    return _post(pid, r)

                workers = min(int(__import__("os").environ.get("LLM_CONCURRENCY", "15")), len(chosen))
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
