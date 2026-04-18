"""Sandbox session + incremental recompute."""
from __future__ import annotations
import uuid
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..causal.counterfactual import Scenario, ScenarioResult, ScenarioRunner


@dataclass
class SandboxSession:
    id: str
    baseline: Scenario
    baseline_result: ScenarioResult
    current: Scenario
    current_result: ScenarioResult
    history: List[Scenario] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def snapshot(self) -> Dict:
        def kpi_round(d):
            return {k: round(float(v), 4) for k, v in d.items()}
        return {
            "id": self.id,
            "baseline_kpis": kpi_round(self.baseline_result.total_kpis),
            "current_kpis": kpi_round(self.current_result.total_kpis),
            "delta": {
                k: round(self.current_result.total_kpis.get(k, 0)
                         - self.baseline_result.total_kpis.get(k, 0), 4)
                for k in self.current_result.total_kpis
            },
            "current_scenario": {
                "total_budget": self.current.total_budget,
                "platform_alloc": self.current.platform_alloc,
            },
            "per_platform": {
                p: kpi_round(d["kpi"])
                for p, d in self.current_result.per_platform.items()
            },
        }


class SandboxStore:
    """In-memory session store with incremental recompute."""

    def __init__(self, runner: ScenarioRunner):
        self.runner = runner
        self.sessions: Dict[str, SandboxSession] = {}

    def create(self, baseline: Scenario) -> SandboxSession:
        baseline_result = self.runner.run(baseline, n_monte_carlo=5)
        sid = str(uuid.uuid4())[:8]
        sess = SandboxSession(
            id=sid,
            baseline=baseline,
            baseline_result=baseline_result,
            current=copy.copy(baseline),
            current_result=baseline_result,
        )
        self.sessions[sid] = sess
        return sess

    def get(self, sid: str) -> Optional[SandboxSession]:
        return self.sessions.get(sid)

    def update(self, sid: str, patch: Dict) -> SandboxSession:
        """Apply a patch to current scenario and incrementally recompute.

        Decision tree:
        - only `total_budget` changed → linear scale (fastest)
        - `platform_alloc` / `kol_per_platform` / `audience_filter` → rerun world+agents
          but preserve abducted U (counterfactual semantics)
        - `creative` changed → full rerun (new abduction)
        """
        sess = self.sessions[sid]
        prev = sess.current
        sess.history.append(copy.copy(prev))
        new = copy.copy(prev)

        creative_changed = False
        if "creative" in patch:
            new.creative = patch["creative"]
            creative_changed = True

        alloc_changed = False
        if "platform_alloc" in patch:
            # normalize
            a = patch["platform_alloc"]
            s = sum(a.values()) or 1
            new.platform_alloc = {k: v/s for k,v in a.items() if v > 0}
            alloc_changed = True

        budget_changed = False
        if "total_budget" in patch:
            new.total_budget = float(patch["total_budget"])
            budget_changed = True

        kol_changed = False
        if "kol_per_platform" in patch:
            new.kol_per_platform = patch["kol_per_platform"]
            kol_changed = True

        if "audience_filter" in patch:
            new.audience_filter = patch["audience_filter"]
            alloc_changed = True   # treat as dist change

        # Dispatch to right compute path
        if creative_changed:
            new_result = self.runner.run(new, n_monte_carlo=5)
        elif alloc_changed or kol_changed:
            # counterfactual from baseline with new allocation, preserving baseline U
            intervention = {
                "platform_alloc": new.platform_alloc,
                "kol_per_platform": new.kol_per_platform,
                "audience_filter": new.audience_filter,
                "total_budget": new.total_budget,
            }
            new_result = self.runner.counterfactual(
                sess.baseline, sess.baseline_result, intervention
            )
        elif budget_changed:
            # Non-linear budget scaling (saturation + frequency fatigue):
            #   - cost = 预算本身线性（投多少花多少 — 平台按实际花费计费）
            #   - impressions = Hill saturation curve (diminishing returns)
            #         impr = impr_base × (B/B_base) × S(B, B_audience_cap)
            #       S(B) = K / (K + B/B_base)  Michaelis-Menten 风格
            #       K=1 时: 2x 预算 → 1.33x 曝光 (不是 2x)
            #   - CTR 随频次上升微降 (频次疲劳 Ebbinghaus-like)
            #         CTR = CTR_base × (1 - 0.08 × ln(B/B_base))  (每翻倍 CTR 降 5.5%)
            #   - CVR 也有轻度疲劳 (广告词已看过)
            #   - revenue 跟着 impressions × CTR × CVR × AOV 算
            # 学术根据: Dubé & Manchanda 2005 (ad saturation); Naik & Raman 2003 (Adstock)
            import math as _m
            ratio = new.total_budget / max(prev.total_budget, 1e-6)
            scaled = copy.deepcopy(sess.current_result)

            # Hill saturation: Y = Y_max × B / (K + B)，归一化到 ratio=1 时返回 1
            #   effective_ratio = (1+K) × ratio / (K + ratio)
            #   K=1.0 → ratio=2.0x 预算 → 1.33x 曝光 (Y_max 递减边际)
            #           ratio=0.5x → 0.67x 曝光 (削减预算损失 <50%，有规模效应下限)
            #           ratio=4.0x → 1.60x 曝光 (预算翻 4 倍只有 1.6 倍效果)
            #   K 小 = 饱和快 (小预算就触顶); K 大 = 饱和慢 (资源充裕)
            import os as _os
            K_sat = float(_os.environ.get("BUDGET_SATURATION_K", "1.0"))
            # 经验值：中小品牌 K≈0.5-1.5 (饱和快), 大品牌 K≈2-5 (资源充裕可持续投)
            # PoC 客户可按历史真实投放曲线回归出自己的 K
            effective_impr_ratio = (1 + K_sat) * ratio / (K_sat + ratio)

            # 频次疲劳 CTR 衰减（只在增预算时生效）
            ctr_decay = 1.0 - 0.08 * max(0, _m.log2(ratio))  # 翻倍 → CTR × 0.92
            ctr_decay = max(0.5, ctr_decay)  # 下限 50%
            cvr_decay = 1.0 - 0.04 * max(0, _m.log2(ratio))  # 翻倍 → CVR × 0.96
            cvr_decay = max(0.7, cvr_decay)

            effective_click_ratio = effective_impr_ratio * ctr_decay
            effective_conv_ratio = effective_click_ratio * cvr_decay
            effective_rev_ratio = effective_conv_ratio  # revenue = conversions × AOV (AOV 不变)

            def scale_kpi(d):
                if "impressions" in d: d["impressions"] *= effective_impr_ratio
                if "clicks" in d:      d["clicks"]      *= effective_click_ratio
                if "conversions" in d: d["conversions"] *= effective_conv_ratio
                if "cost" in d:        d["cost"]        *= ratio  # 预算就是花费
                if "revenue" in d:     d["revenue"]     *= effective_rev_ratio
                # 重算 rate 类指标
                if d.get("impressions",0) > 0:
                    d["ctr"] = d.get("clicks",0) / d["impressions"]
                if d.get("clicks",0) > 0:
                    d["cvr"] = d.get("conversions",0) / d["clicks"]

            for plat, d in scaled.per_platform.items():
                if "kpi" in d: scale_kpi(d["kpi"])
            scale_kpi(scaled.total_kpis)

            # ROI 会因为曝光/点击率下降而**下降**（规模不经济）
            scaled.total_kpis["roi"] = (scaled.total_kpis.get("revenue", 0)
                                        - scaled.total_kpis.get("cost", 0)) \
                                       / max(scaled.total_kpis.get("cost", 1), 1)

            # 附加 audit 元数据方便前端展示
            scaled.total_kpis["_budget_scaling_note"] = (
                f"Hill 饱和 + 频次疲劳：预算 {ratio:.2f}x → "
                f"曝光 {effective_impr_ratio:.2f}x / 点击 {effective_click_ratio:.2f}x / "
                f"ROI 从 {prev.total_budget and sess.current_result.total_kpis.get('roi',0):.2f} → "
                f"{scaled.total_kpis['roi']:.2f}"
            )
            new_result = scaled
        else:
            new_result = sess.current_result

        sess.current = new
        sess.current_result = new_result
        return sess

    def undo(self, sid: str) -> SandboxSession:
        sess = self.sessions[sid]
        if not sess.history:
            return sess
        prev = sess.history.pop()
        sess.current = prev
        sess.current_result = self.runner.run(prev, n_monte_carlo=5)
        return sess
