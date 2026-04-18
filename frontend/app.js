"use strict";

// API lives on the same host as the page, on port 8001.
// This way it works whether you open via localhost, 127.0.0.1, or a remote IP.
let SESSION_ID = null;
let LAST_BASELINE = null;

// ─── Hero autoplay (T-A) ─────────────────────────────────────────────────
// 生成 4500 个合成粒子（25 个高斯集群 + 背景均匀分布），作为 hero 背景数据。
// 不依赖 backend，页面一打开就能跑动画；即使 API 挂了 hero 也不卡。
function _generateHeroData(n = 4500, nClusters = 25) {
  const points = new Array(n);
  const centers = new Array(nClusters);
  for (let c = 0; c < nClusters; c++) {
    centers[c] = { x: 0.08 + Math.random() * 0.84, y: 0.12 + Math.random() * 0.76, size: 0.04 + Math.random() * 0.09 };
  }
  const gauss = () => {  // Box-Muller
    const u = 1 - Math.random(), v = 1 - Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  for (let i = 0; i < n; i++) {
    // 85% 聚类分布，15% 均匀背景
    if (Math.random() < 0.85) {
      const c = centers[(Math.random() * nClusters) | 0];
      const x = Math.max(0.02, Math.min(0.98, c.x + gauss() * c.size));
      const y = Math.max(0.02, Math.min(0.98, c.y + gauss() * c.size));
      // 8% 是 soul agent（级联传播的种子），优先来自 cluster 中心附近
      const is_soul = Math.random() < 0.08;
      points[i] = { x, y, is_soul, tier: (Math.random() * 5) | 0 };
    } else {
      points[i] = { x: Math.random(), y: Math.random(), is_soul: false, tier: (Math.random() * 5) | 0 };
    }
  }
  return { points };
}

function _startHeroAmbient() {
  const c = document.getElementById("hero-canvas");
  if (!c) return;
  // 把 canvas 实际像素尺寸对齐 devicePixelRatio 保持清晰
  const rect = c.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  c.width = Math.floor(rect.width * dpr);
  c.height = Math.floor(rect.height * dpr);
  // T-C2: 1200 粒子 + 漂移 + 稀疏连线，既像 constellation 又保留 T-C1 波的叙事
  const data = window._HERO_DATA || (window._HERO_DATA = _generateHeroData(1200, 14));
  runOpinionCascade(c, data, {
    initialWaves: 4,
    primaryBurst: 2,
    primaryIntervalMs: 1900,
    waveSpeed: 0.13,
    waveThick: 0.035,
    waveTau: 3.2,
    cascadeProb: 0.004,
    maxWaves: 32,
    trailFill: 'rgba(6,8,13,0.14)',
    lowCpu: true,
    drift: true,
    driftSpeed: 0.010,
    connect: {
      threshold: 0.055,
      alphaMax: 0.18,
      rgb: "125,160,220",  // 冷蓝调 connect 线，与 T-C1 波形成对比
    },
  });
}

// 窗口尺寸变化 / DPR 变化时重新对齐 canvas
window.addEventListener('resize', () => {
  const c = document.getElementById("hero-canvas");
  if (!c) return;
  const rect = c.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  c.width = Math.floor(rect.width * dpr);
  c.height = Math.floor(rect.height * dpr);
});

function heroScrollToForm() {
  const m = document.querySelector('main');
  if (m) m.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function heroLaunch() {
  const heroCap = document.getElementById("hero-caption");
  const mainCap = document.getElementById("caption");
  const text = (heroCap && heroCap.value || "").trim();
  if (text && mainCap) mainCap.value = text;
  heroScrollToForm();
  // 稍等滚动完触发「极速」预设再预测，节奏自然
  setTimeout(() => {
    try { applyPreset('quick'); } catch(e) {}
    setTimeout(() => {
      try { runPredict(); } catch(e) {}
    }, 350);
  }, 700);
}


// 页面就绪后启动 hero ambient（放 setTimeout 让浏览器先完成首帧布局）
window.addEventListener('load', () => {
  setTimeout(_startHeroAmbient, 80);
});

// One-click preset configurations
function applyPreset(kind) {
  const set = (id, v) => { const el = document.getElementById(id); if (!el) return;
    if (el.type === 'checkbox') el.checked = v;
    else el.value = v;
    el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change'));
  };
  // common reset
  set("aud_gender", ""); set("sentiment", "neutral"); set("temp", 20); set("rainy", false);
  if (kind === "quick") {
    // Multi-agent baseline: 100k stat agents + small LLM sample for soul-quote viz
    set("use_llm", true); set("llm_calibrate", true); set("nsouls", 8);
    set("enable_crossplat", true);
    set("enable_recsys_rl", false); set("enable_discourse", false);
    set("enable_groupchat", false); set("enable_brand_memory", false);
    set("daypart", "auto"); set("overlap", 25);
    log("⚡ 轻量模式：100k 统计智能体 + 8 LLM Oracles + SLOC 校准 + Hawkes 时序，约 10-15s");
  } else if (kind === "viral") {
    set("use_llm", true); set("llm_calibrate", true); set("nsouls", 50);
    set("enable_crossplat", true); set("enable_recsys_rl", true);
    set("enable_discourse", true); set("ndisc", 12);
    set("enable_groupchat", true); set("ngc_a", 10); set("ngc_r", 3);
    set("enable_brand_memory", false);
    set("daypart", "evening"); set("music", "upbeat"); set("visual", "bright");
    set("niche", "food"); set("overlap", 30);
    log("🔥 爆款模式：100k 统计 + 50 LLM 灵魂 + 12-agent 多轮群聊 + 5 轮 RecSys RL，约 60-90s");
  } else if (kind === "crisis") {
    set("use_llm", true); set("llm_calibrate", true); set("nsouls", 80);
    set("enable_crossplat", true); set("enable_recsys_rl", false);
    set("enable_discourse", true); set("ndisc", 20);
    set("enable_groupchat", true); set("ngc_a", 15); set("ngc_r", 4);
    set("enable_brand_memory", true); set("nbm", 90);
    set("sentiment", "crisis"); set("daypart", "late");
    log("⚠️ 危机模式：100k 统计 + 80 LLM 灵魂 + 15-agent 群聊 + 20 条 LLM 评论 + 90 天纵向品牌追踪，约 2-3min");
  } else if (kind === "expert") {
    set("use_llm", true); set("llm_calibrate", true); set("nsouls", 100);
    set("enable_crossplat", true); set("enable_recsys_rl", true);
    set("enable_discourse", true); set("ndisc", 20);
    set("enable_groupchat", true); set("ngc_a", 15); set("ngc_r", 4);
    set("enable_brand_memory", true); set("nbm", 90);
    log("🧠 全开模式：100k 统计 + 100 LLM 灵魂 + ABCDE 全栈（跨平台/RecSys RL/评论辩论/群聊/90天品牌），约 2-3min");
  } else if (kind === "mega") {
    set("use_llm", true); set("llm_calibrate", true); set("nsouls", 10000);
    set("enable_crossplat", true); set("enable_recsys_rl", true);
    set("enable_discourse", true); set("ndisc", 30);
    set("enable_groupchat", true); set("ngc_a", 20); set("ngc_r", 4);
    set("enable_brand_memory", true); set("nbm", 90);
    set("daypart", "evening"); set("overlap", 35);
    window._megaMode = true;
    log("🌍 极限壮观：1M 虚拟消费者 + 全量 10000 AI 灵魂真 LLM + 五项高级分析全开，约 6-10 分钟、成本 ¥4-6。预测完自动跳到'🌍 100万智能体社会' tab。");
    // Pre-render the society canvas now so users see something while predict runs
    try { setTab('society'); renderSociety(30000); } catch(e) {}
  }
}

function buildRequest() {
  const alloc = { douyin: +document.getElementById("douyin_slider").value,
                  xhs:    +document.getElementById("xhs_slider").value };
  const sum = alloc.douyin + alloc.xhs || 1;
  const ages = [...document.getElementById("aud_age").selectedOptions].map(o=>+o.value);
  const cities = [...document.getElementById("aud_city").selectedOptions].map(o=>+o.value);
  const g = document.getElementById("aud_gender").value;
  return {
    creative: {
      caption: document.getElementById("caption").value,
      duration_sec: 15,
      visual_style: document.getElementById("visual").value,
      music_mood: document.getElementById("music").value,
      has_celeb: document.getElementById("celeb").checked,
    },
    total_budget: +document.getElementById("budget").value,
    platform_alloc: { douyin: alloc.douyin/sum, xhs: alloc.xhs/sum },
    kol_niche: document.getElementById("niche").value || null,
    use_llm: document.getElementById("use_llm").checked,
    llm_calibrate: document.getElementById("llm_calibrate").checked,
    n_souls: +document.getElementById("nsouls").value,
    today: document.getElementById("today").value || null,
    daypart: document.getElementById("daypart").value,
    audience_age_buckets: ages.length ? ages : null,
    audience_gender: g === "" ? null : +g,
    audience_city_tiers: cities.length ? cities : null,
    cross_platform_overlap: (+document.getElementById("overlap").value) / 100,
    sentiment: document.getElementById("sentiment").value,
    weather_temp_c: +document.getElementById("temp").value,
    rainy: document.getElementById("rainy").checked,
    enable_crossplat: document.getElementById("enable_crossplat").checked,
    enable_recsys_rl: document.getElementById("enable_recsys_rl").checked,
    enable_discourse: document.getElementById("enable_discourse").checked,
    discourse_n_comments: +document.getElementById("ndisc").value,
    enable_brand_memory: document.getElementById("enable_brand_memory").checked,
    brand_memory_days: +document.getElementById("nbm").value,
    enable_groupchat: document.getElementById("enable_groupchat").checked,
    groupchat_n_agents: +document.getElementById("ngc_a").value,
    groupchat_n_rounds: +document.getElementById("ngc_r").value,
    own_brand: document.getElementById("own_brand")?.value || null,
    category: document.getElementById("category")?.value || null,
    competitors: (document.getElementById("competitors")?.value || "").split(/[,，\s]+/).filter(x=>x.trim()).slice(0,5) || null,
    target_niches: document.getElementById("niche").value ? [{"beauty":"美妆","mom":"母婴","tech":"数码","food":"美食","fashion":"穿搭","fitness":"健身","finance":"理财","travel":"旅行"}[document.getElementById("niche").value]].filter(Boolean) : null,
    enable_kol_ilp: document.getElementById("enable_kol_ilp")?.checked,
    enable_search_elasticity: document.getElementById("enable_search_elasticity")?.checked,
  };
}

async function runPredict() {
  const t0 = Date.now();
  log("⏳ 预测中…");
  const btn = document.getElementById("btn-predict");
  let hb = null;
  const restoreBtn = () => { if (hb) clearInterval(hb); if (btn) { btn.disabled=false; btn.textContent="🚀 预测 & 开启沙盘"; }};
  try {
  if (btn) { btn.disabled = true; btn.textContent = "⏳ 预测中…(可能 30-60s)"; }
  hb = setInterval(() => {
    const elapsed = ((Date.now()-t0)/1000).toFixed(0);
    if (btn) btn.textContent = `⏳ 预测中… ${elapsed}s`;
  }, 2000);
  const body = buildRequest();
  // 1. full predict — gets macro + lifecycle + souls + KPIs
  const r1 = await fetch(API + "/api/predict", {
    method:"POST", headers:{"content-type":"application/json"},
    body: JSON.stringify(body)
  });
  const pred = await r1.json();
  // Also call V1 (CCG) — captures trace for ∞ V1 tab
  fetch(API + "/api/predict_v1", {method:"POST", headers:{"content-type":"application/json"},
    body: JSON.stringify(body)}).then(r=>r.json()).then(v1 => {
    window.LAST_TRACE = v1.trace;
    if (document.querySelector('.tab[data-tab=v1]').classList.contains('active'))
      renderTrace(v1.trace);
  }).catch(()=>{});
  renderMacro(pred.macro);
  renderPredictedSentiment(pred.predicted_sentiment);
  window.LAST_SCHEMA = pred.schema_outputs;
  renderSchemaOutputs(pred.schema_outputs);
  renderSouls(pred.soul_quotes);
  if (pred.lifecycle) drawLifecycle(pred.lifecycle);
  renderFrontier(pred.extras || {});
  log(`KPI: CTR ${(pred.kpis.ctr*100).toFixed(2)}% · CVR ${(pred.kpis.cvr*100).toFixed(2)}% · ROI ${pred.kpis.roi.toFixed(2)}x`);
  if (window._megaMode) {
    try { setTab('society'); renderSociety(30000); animateSociety(); } catch(e) {}
    window._megaMode = false;
  }

  // 2. create sandbox session (baseline = same plan, no llm)
  const r2 = await fetch(API + "/api/sandbox/session", {
    method:"POST", headers:{"content-type":"application/json"},
    body: JSON.stringify(body)
  });
  const s = await r2.json();
  SESSION_ID = s.id;
  LAST_BASELINE = s.baseline_kpis;
  log(`✅ 完成 ${((Date.now()-t0)/1000).toFixed(1)}s · 沙盘 ${SESSION_ID} · 基线 ROI ${s.baseline_kpis.roi.toFixed(2)}x`);
  renderSnapshot(s);
  drawDAG();
  } catch (e) {
    log("❌ 出错: " + (e.message||e));
    console.error(e);
  } finally { restoreBtn(); }
}

function renderMacro(m) {
  if (!m) return;
  const el = document.getElementById("macro-panel");
  let html = `
    📅 ${m.today} · ${m.holiday.label}(${m.holiday.lift}x) · ${m.daypart.label} · DOW×${m.dow_mult} · 季节×${m.season_mult}<br>
    🌍 世界事件类目 lift × ${m.world_category_lift||1.0}<br>
    🎯 CTR macro lift = <b style="color:var(--good)">${m.ctr_macro_lift}x</b> · CVR macro lift = <b style="color:var(--good)">${m.cvr_macro_lift}x</b><br>
    ⚠️ 素材 audit_risk=${m.creative_audit_risk} · aigc=${m.creative_aigc_score} · 类目=${m.creative_category}`;
  const cal = m.llm_calibration;
  if (cal) {
    const f = cal.global_factor;
    const color = f<0.8?'var(--bad)':f>1.2?'var(--good)':'var(--warn)';
    html += `<br>🧠 <b>SLOC (Sparse LLM Oracle Calibration)</b>: global × <b style="color:${color}">${f}</b>
      (${cal.n_souls_active}/${cal.n_souls_total} souls active · 单 soul 最多代表 ${Math.round(cal.max_territory_weight*100000)} 真实用户)
      <br>&nbsp;&nbsp;&nbsp;&nbsp; factor 分布 p10=${cal.factor_p10.toFixed(2)} · p50=${cal.factor_p50.toFixed(2)} · p90=${cal.factor_p90.toFixed(2)}`;
  }
  el.innerHTML = html;
  // Also refresh world panel from m.world (cheap, no API call)
  if (m.world) renderWorld({events: [], ...m.world}, true);
}

function renderFrontier(extras) {
  // D: cross-platform
  const cpEl = document.getElementById("frontier-d");
  if (extras.cross_platform) {
    const c = extras.cross_platform;
    cpEl.innerHTML = `<h4 style="color:var(--accent); margin:0 0 6px; font-size:13px">D · 跨平台身份 (unique reach / cannibalization)</h4>
      <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="kpi"><div class="v">${(c.unique_reach/10000).toFixed(1)}万</div><div class="l">unique reach</div></div>
        <div class="kpi"><div class="v">${c.avg_frequency}</div><div class="l">avg frequency</div></div>
        <div class="kpi"><div class="v" style="color:var(--bad)">${(c.cannibalization_pct*100).toFixed(1)}%</div><div class="l">cannibalization</div></div>
        <div class="kpi"><div class="v">${c.max_frequency}</div><div class="l">max freq</div></div>
      </div>
      <div class="hint">incremental 每平台新用户：${Object.entries(c.per_platform_incremental).map(([p,v])=>`${p} +${v.toLocaleString()}`).join(' · ')}</div>`;
  } else cpEl.innerHTML = "";

  // C: recsys RL
  const rlEl = document.getElementById("frontier-c");
  if (extras.recsys_rl) {
    const r = extras.recsys_rl;
    rlEl.innerHTML = `<h4 style="color:var(--accent); margin:0 0 6px; font-size:13px">C · RecSys RL 冷启破圈 (${r.n_rounds} 轮迭代)</h4>
      <div class="hint">破圈: <b style="color:${r.break_out?'var(--good)':'var(--muted)'}">${r.break_out?'✅ YES':'❌ NO'}</b> · peak round: R${r.peak_round} · 最终权重: ${Object.entries(r.final_weights).map(([k,v])=>`${k}:${v}`).join(' · ')}</div>
      <table style="width:100%; font-size:11px; border-collapse:collapse; margin-top:6px;">
        <thead style="color:var(--muted);"><tr><th>round</th><th>CTR</th><th>engage</th><th>reach</th><th>broke?</th></tr></thead>
        <tbody>${r.per_round.map(x=>`<tr style="border-top:1px solid var(--border);">
          <td style="padding:3px;">R${x.round}</td>
          <td style="color:${x.click_rate>0.035?'var(--good)':'var(--fg)'}">${(x.click_rate*100).toFixed(2)}%</td>
          <td>${(x.engage_rate*100).toFixed(2)}%</td>
          <td>${x.reach.toLocaleString()}</td>
          <td>${x.broke_out?'🚀':'·'}</td></tr>`).join('')}</tbody>
      </table>`;
  } else rlEl.innerHTML = "";

  // A: discourse
  const dEl = document.getElementById("frontier-a");
  if (extras.discourse) {
    const d = extras.discourse;
    const sentColor = d.dominant_sentiment > 0.3 ? 'var(--good)' : d.dominant_sentiment < -0.3 ? 'var(--bad)' : 'var(--warn)';
    dEl.innerHTML = `<h4 style="color:var(--accent); margin:0 0 6px; font-size:13px">A · LLM 评论区辩论 (SCM mediator) <span class="badge">${d.source}</span></h4>
      <div class="hint">
        主导情绪 <b style="color:${sentColor}">${d.dominant_sentiment.toFixed(2)}</b> · 分歧度 ${d.sentiment_variance.toFixed(2)} ·
        应用 CTR 乘子 <b style="color:var(--good)">${d.applied_ctr_multiplier}x</b> ·
        成本 ¥${d.cost_cny}
      </div>
      <div style="margin:6px 0; padding:6px; background:#0f141c; border-radius:4px;">
        <div><b style="color:var(--warn)">Viral tone:</b> ${d.viral_tone}</div>
        <div style="margin-top:4px;"><b style="color:var(--good)">👍 赞:</b> ${d.top_praises.map(p=>`<div style="font-size:11px; margin:2px 0 0 10px;">• ${p}</div>`).join('')}</div>
        <div style="margin-top:4px;"><b style="color:var(--bad)">👎 黑:</b> ${d.top_objections.map(p=>`<div style="font-size:11px; margin:2px 0 0 10px;">• ${p}</div>`).join('')}</div>
      </div>
      <div style="max-height:200px; overflow:auto; font-size:11px;">${d.comments.map(c =>
        `<div style="margin:3px 0; padding:4px 6px; background:#0f141c; border-radius:4px; border-left:3px solid ${c.sentiment>0.2?'var(--good)':c.sentiment<-0.2?'var(--bad)':'var(--warn)'};">
          <div class="oneliner">${c.persona} <span class="badge">${c.tone}</span></div>
          <div>${c.text}</div>
        </div>`).join('')}</div>`;
  } else dEl.innerHTML = "";

  // E: group chat (multi-turn LLM)
  const eEl = document.getElementById("frontier-e");
  if (extras.group_chat) {
    window.LAST_GROUPCHAT = extras.group_chat;
    renderGroupChat(extras.group_chat);
    const gc = extras.group_chat;
    const consColor = gc.consensus > 0.3 ? 'var(--good)' : gc.consensus < -0.3 ? 'var(--bad)' : 'var(--warn)';
    const drift = (id) => {
      const init = gc.initial_stances[id], fin = gc.final_stances[id];
      const delta = fin - init;
      return `<span style="color:${delta>0.1?'var(--good)':delta<-0.1?'var(--bad)':'var(--muted)'}">${init>=0?'+':''}${init} → ${fin>=0?'+':''}${fin}</span>`;
    };
    eEl.innerHTML = `<h4 style="color:var(--accent); margin:0 0 6px; font-size:13px">E · 多轮 LLM 群聊（peer-to-peer 信息传递 · ${gc.n_agents}×${gc.n_rounds})</h4>
      <div class="hint">
        共识 <b style="color:${consColor}">${gc.consensus.toFixed(2)}</b> · 极化度 ${gc.polarization.toFixed(2)} ·
        ${gc.converged?'✅ 收敛':'⚠️ 群体极化（越聊越散/同向加深）'} · 二轮影响 <b>${gc.second_wave_impact}</b> · ¥${gc.cost_cny}
      </div>
      <div style="margin:6px 0; padding:6px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--warn)">主导话题:</b> ${gc.dominant_frame}
      </div>
      <div style="font-size:11px; margin:6px 0;">
        <b>立场漂移：</b>
        ${Object.keys(gc.initial_stances).map(id => `<div class="seg"><span>agent ${id}</span><span>${drift(id)}</span></div>`).join('')}
      </div>
      <div style="font-size:11px; margin-top:8px;"><b>群聊节选：</b></div>
      <div style="max-height:280px; overflow:auto; font-size:11px; margin-top:4px;">${gc.messages.map(m =>
        m.silent ? `<div style="margin:2px 0; color:var(--muted); font-size:10px;">T${m.turn} ${m.sender}: 划过</div>` :
        `<div style="margin:3px 0; padding:4px 6px; background:#0f141c; border-radius:4px; border-left:3px solid ${m.sentiment>0.2?'var(--good)':m.sentiment<-0.2?'var(--bad)':'var(--warn)'};">
          <div style="display:flex; justify-content:space-between;">
            <span class="oneliner">T${String(m.turn).padStart(2,'0')} ${m.sender}</span>
            <span style="color:${m.sentiment>0?'var(--good)':'var(--bad)'};">${m.sentiment>=0?'+':''}${m.sentiment}</span>
          </div>
          <div>${m.text}</div>
        </div>`).join('')}</div>
      <div style="margin-top:6px; font-size:11px;"><b>每轮均值/方差曲线：</b>
        ${gc.rounds_summary.map(r => `<div class="seg"><span>R${r.round}</span><span>mean ${r.mean_stance>=0?'+':''}${r.mean_stance.toFixed(2)} · std ${r.std_stance.toFixed(2)} · n_speakers ${r.n_speakers_total}</span></div>`).join('')}
      </div>`;
  } else eEl.innerHTML = "";

  // B: brand memory
  const bEl = document.getElementById("frontier-b");
  if (extras.brand_memory) {
    const bm = extras.brand_memory, f = bm.final;
    bEl.innerHTML = `<h4 style="color:var(--accent); margin:0 0 6px; font-size:13px">B · ${bm.days} 天纵向品牌记忆</h4>
      <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="kpi"><div class="v">${(f.n_reached/10000).toFixed(1)}万</div><div class="l">reached</div></div>
        <div class="kpi"><div class="v">${(f.brand_recall_pct*100).toFixed(1)}%</div><div class="l">recall</div></div>
        <div class="kpi"><div class="v" style="color:var(--good)">${(f.brand_favor_pct*100).toFixed(1)}%</div><div class="l">favor</div></div>
        <div class="kpi"><div class="v" style="color:${f.brand_aversion_pct>0.1?'var(--bad)':'var(--fg)'}">${(f.brand_aversion_pct*100).toFixed(1)}%</div><div class="l">aversion</div></div>
      </div>
      <canvas id="bm-canvas" width="700" height="160" style="width:100%; background:#0f141c; border-radius:6px; margin-top:8px;"></canvas>`;
    drawBrandMemory(bm.timeline);
  } else bEl.innerHTML = "";
}

function renderGroupChat(gc) {
  if (!gc) return;
  const hint = document.getElementById("chat-empty-hint");
  if (hint) hint.style.display = "none";
  // ---- summary ----
  const consColor = gc.consensus > 0.3 ? 'var(--good)' : gc.consensus < -0.3 ? 'var(--bad)' : 'var(--warn)';
  document.getElementById("chat-summary").innerHTML = `
    <div class="kpi-grid" style="grid-template-columns:repeat(5,1fr);">
      <div class="kpi"><div class="v" style="color:${consColor}">${gc.consensus>=0?'+':''}${gc.consensus.toFixed(2)}</div><div class="l">最终共识</div></div>
      <div class="kpi"><div class="v">${gc.polarization.toFixed(2)}</div><div class="l">极化度</div></div>
      <div class="kpi"><div class="v" style="color:${gc.converged?'var(--good)':'var(--warn)'}">${gc.converged?'✅':'⚠️'}</div><div class="l">${gc.converged?'收敛':'群体极化'}</div></div>
      <div class="kpi"><div class="v">${gc.second_wave_impact>=0?'+':''}${gc.second_wave_impact}</div><div class="l">2轮 CTR Δ</div></div>
      <div class="kpi"><div class="v">¥${gc.cost_cny}</div><div class="l">LLM 成本</div></div>
    </div>
    <div style="margin-top:8px; padding:8px; background:#0f141c; border-radius:4px;">
      <span style="color:var(--warn)"><b>主导话题:</b></span> ${gc.dominant_frame}
    </div>`;

  // ---- per-agent stance trajectory (MiroFish-style line chart) ----
  drawStanceTrajectory(gc);

  // ---- conversation stream ----
  document.getElementById("chat-stream").innerHTML = gc.messages.map(m =>
    m.silent
      ? `<div style="margin:2px 0; padding:2px 6px; color:var(--muted); font-size:10px;">T${m.turn} ${m.sender}: 划过</div>`
      : `<div style="margin:4px 0; padding:6px 8px; background:#0f141c; border-radius:4px; border-left:3px solid ${m.sentiment>0.2?'var(--good)':m.sentiment<-0.2?'var(--bad)':'var(--warn)'};">
          <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted);">
            <span>T${String(m.turn).padStart(2,'0')} #${m.sender_id} · ${m.sender}</span>
            <span style="color:${m.sentiment>0?'var(--good)':'var(--bad)'};">${m.sentiment>=0?'+':''}${m.sentiment}</span>
          </div>
          <div style="margin-top:3px;">${m.text}</div>
          ${m.replies_to !== null && m.replies_to !== undefined ? `<div style="font-size:10px; color:var(--muted);">↳ 回复 T${String(m.replies_to).padStart(2,'0')}</div>` : ''}
        </div>`).join("");

  // ---- influence network ----
  drawInfluenceNetwork(gc);

  // ---- final stances list ----
  const stanceList = Object.entries(gc.final_stances)
    .sort((a,b) => b[1] - a[1])
    .map(([id, s]) => {
      const init = gc.initial_stances[id] || 0;
      const delta = s - init;
      const arrow = Math.abs(delta) < 0.1 ? '→' : delta > 0 ? '↑' : '↓';
      const color = delta > 0.1 ? 'var(--good)' : delta < -0.1 ? 'var(--bad)' : 'var(--muted)';
      return `<div class="seg"><span>#${id}</span><span style="color:${color}">${init>=0?'+':''}${init.toFixed(2)} ${arrow} ${s>=0?'+':''}${s.toFixed(2)}</span></div>`;
    }).join("");
  document.getElementById("chat-final-stances").innerHTML =
    "<b style='font-size:11px;'>立场漂移排序:</b>" + stanceList;
}

function drawStanceTrajectory(gc) {
  const c = document.getElementById("chat-stance-canvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, padL = 50, padR = 20, padT = 30, padB = 40;
  ctx.clearRect(0, 0, W, H);

  const agentIds = Object.keys(gc.initial_stances);
  const N = agentIds.length;
  // Build per-agent timeline: stance after each speaking turn
  const timelines = {};
  agentIds.forEach(id => { timelines[id] = [{x: 0, y: gc.initial_stances[id]}]; });
  gc.messages.forEach((m, i) => {
    if (!m.silent) {
      const id = String(m.sender_id);
      if (timelines[id]) {
        timelines[id].push({x: i + 1, y: m.sentiment});
      }
    }
  });
  // also add final point for each
  agentIds.forEach(id => {
    timelines[id].push({x: gc.messages.length, y: gc.final_stances[id]});
  });

  const xMax = gc.messages.length;
  const yToCanvas = y => padT + (1 - y) / 2 * (H - padT - padB);
  const xToCanvas = x => padL + x / Math.max(xMax, 1) * (W - padL - padR);

  // Background grid + zero line
  ctx.strokeStyle = "#222a38"; ctx.lineWidth = 1;
  // y axis: -1 .. +1 with gridlines at every 0.25
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace";
  for (let v = -1; v <= 1.001; v += 0.25) {
    const yp = yToCanvas(v);
    ctx.beginPath(); ctx.moveTo(padL, yp); ctx.lineTo(W - padR, yp);
    ctx.strokeStyle = v === 0 ? "#444a5c" : "#1a212e"; ctx.stroke();
    ctx.fillText(v.toFixed(2), 8, yp + 3);
  }
  // round dividers (each round has agentIds.length speaking turns)
  for (let r = 1; r <= gc.n_rounds; r++) {
    const x = xToCanvas(r * N);
    ctx.strokeStyle = "#1a212e"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#666"; ctx.fillText("R" + r, x - 8, H - padB + 14);
  }

  // Mean ± std band (compute at each tick)
  const ticks = Array.from({length: xMax + 1}, (_, t) => {
    const stances = agentIds.map(id => {
      const tl = timelines[id];
      // most recent value at or before tick t
      let v = gc.initial_stances[id];
      for (const p of tl) { if (p.x <= t) v = p.y; else break; }
      return v;
    });
    const mean = stances.reduce((a,b)=>a+b,0) / stances.length;
    const std = Math.sqrt(stances.reduce((s,v)=>s + (v-mean)**2, 0) / stances.length);
    return {x: t, mean, std, stances};
  });
  // ±std fill band
  ctx.fillStyle = "rgba(110,168,254,0.08)";
  ctx.beginPath();
  ticks.forEach((t,i) => {
    const x = xToCanvas(t.x), y = yToCanvas(t.mean + t.std);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  for (let i = ticks.length - 1; i >= 0; i--) {
    ctx.lineTo(xToCanvas(ticks[i].x), yToCanvas(ticks[i].mean - ticks[i].std));
  }
  ctx.closePath(); ctx.fill();

  // per-agent lines (color by agent id hash)
  const palette = ["#6ea8fe","#5ed39b","#ffc857","#ff7a85","#bd93f9","#8be9fd","#ff79c6","#50fa7b",
                    "#ffb86c","#f1fa8c","#ff6b6b","#a3be8c","#88c0d0","#ebcb8b","#d08770"];
  agentIds.forEach((id, idx) => {
    const tl = timelines[id];
    const color = palette[idx % palette.length];
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    tl.forEach((p, i) => {
      const x = xToCanvas(p.x), y = yToCanvas(p.y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // dot at speaking points
    tl.forEach((p, i) => {
      if (i > 0 && i < tl.length - 1) {
        ctx.fillStyle = color; ctx.beginPath();
        ctx.arc(xToCanvas(p.x), yToCanvas(p.y), 2.5, 0, 2*Math.PI); ctx.fill();
      }
    });
  });
  ctx.globalAlpha = 1;

  // mean line bold
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath();
  ticks.forEach((t, i) => {
    const x = xToCanvas(t.x), y = yToCanvas(t.mean);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // labels
  ctx.fillStyle = "#fff"; ctx.font = "bold 11px monospace";
  ctx.fillText("立场漂移轨迹（每条彩线 = 一个 agent · 白线 = 群体均值 · 蓝带 = ±1σ）", padL, 18);
  ctx.fillStyle = "#5ed39b"; ctx.fillText("正面 +1", W - padR - 50, padT + 10);
  ctx.fillStyle = "#ff7a85"; ctx.fillText("负面 -1", W - padR - 50, H - padB - 4);
}

function drawInfluenceNetwork(gc) {
  const svg = document.getElementById("chat-network");
  if (!svg) return;
  const W = 400, H = 380;
  const agentIds = Object.keys(gc.initial_stances);
  const N = agentIds.length;
  // Lay out in a circle
  const cx = W/2, cy = H/2, R = Math.min(W, H) * 0.36;
  const pos = {};
  agentIds.forEach((id, i) => {
    const a = (i / N) * 2 * Math.PI - Math.PI/2;
    pos[id] = {x: cx + R * Math.cos(a), y: cy + R * Math.sin(a)};
  });
  // Build influence edges from messages: m.replies_to → m.sender_id
  const edgeCounts = {};
  gc.messages.forEach(m => {
    if (m.silent) return;
    if (m.replies_to !== null && m.replies_to !== undefined) {
      // find the replier (turn = replies_to)
      const target = gc.messages.find(x => x.turn === m.replies_to);
      if (!target) return;
      const k = `${target.sender_id}->${m.sender_id}`;
      edgeCounts[k] = (edgeCounts[k] || 0) + 1;
    }
  });

  let svgContent = `<defs><marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker></defs>`;
  // edges
  Object.entries(edgeCounts).forEach(([k, count]) => {
    const [s, t] = k.split('->');
    const a = pos[s], b = pos[t];
    if (!a || !b) return;
    const w = Math.min(3, 0.5 + count * 0.5);
    svgContent += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#888" stroke-width="${w}" opacity="0.5" marker-end="url(#arr3)"/>`;
  });
  // nodes (color by final stance: red=neg, gray=neutral, green=pos)
  agentIds.forEach((id, i) => {
    const p = pos[id];
    const stance = gc.final_stances[id];
    const fill = stance > 0.2 ? "#5ed39b" : stance < -0.2 ? "#ff7a85" : "#8b93a7";
    const r = 14;
    // selection ring (blue) appears when selected
    svgContent += `<circle id="ring-${id}" cx="${p.x}" cy="${p.y}" r="${r+5}" fill="none" stroke="#6ea8fe" stroke-width="2.5" opacity="0"/>`;
    svgContent += `<circle data-aid="${id}" cx="${p.x}" cy="${p.y}" r="${r}" fill="${fill}" stroke="#0f141c" stroke-width="2" style="cursor:pointer"/>`;
    svgContent += `<text data-aid="${id}" x="${p.x}" y="${p.y+4}" text-anchor="middle" font-size="10" fill="#0f141c" font-weight="bold" style="pointer-events:none">${id.slice(-3)}</text>`;
    svgContent += `<text x="${p.x}" y="${p.y + (p.y>cy?28:-18)}" text-anchor="middle" font-size="9" fill="#e6e8ee" style="pointer-events:none">${stance>=0?'+':''}${stance.toFixed(2)}</text>`;
  });
  svg.innerHTML = svgContent;
  // attach handlers (delegate)
  svg.querySelectorAll("[data-aid]").forEach(el => {
    el.addEventListener("click", e => {
      const id = el.getAttribute("data-aid");
      selectAgent(id, gc);
    });
    el.addEventListener("contextmenu", e => {
      e.preventDefault();
      const id = el.getAttribute("data-aid");
      selectAgent(id, gc, /*showInterveneSlider*/true);
    });
  });
}

let SELECTED_AGENT = null;
function selectAgent(id, gc, showSlider=false) {
  // hide all rings
  document.querySelectorAll("#chat-network [id^='ring-']").forEach(r => r.setAttribute("opacity", "0"));
  // show this one
  const ring = document.getElementById(`ring-${id}`);
  if (ring) ring.setAttribute("opacity", "1");
  SELECTED_AGENT = id;

  // find agent's messages
  const msgs = (gc.messages||[]).filter(m => String(m.sender_id) === id);
  const init = gc.initial_stances[id], fin = gc.final_stances[id];
  const drift = fin - init;
  const driftColor = drift > 0.1 ? "var(--good)" : drift < -0.1 ? "var(--bad)" : "var(--muted)";
  const personaLine = msgs[0]?.sender || `agent ${id}`;

  let html = `<div style="border-bottom:1px solid var(--border); padding-bottom:6px; margin-bottom:6px;">
    <b style="color:var(--accent)">agent #${id}</b> · ${personaLine}
    <div>初始 <b>${init>=0?'+':''}${init.toFixed(2)}</b> → 最终 <b style="color:${driftColor}">${fin>=0?'+':''}${fin.toFixed(2)}</b> (Δ ${drift>=0?'+':''}${drift.toFixed(2)})</div>
  </div>`;
  if (msgs.length) {
    html += `<div style="font-size:10px; color:var(--muted); margin-bottom:4px;">这个 agent 说过 ${msgs.length} 次:</div>`;
    msgs.slice(0,5).forEach(m => {
      html += `<div style="padding:3px 6px; margin:2px 0; background:#0b0e14; border-radius:3px; border-left:2px solid ${m.sentiment>0?'var(--good)':'var(--bad)'};">
        <span style="color:var(--muted)">T${m.turn} [${m.sentiment>=0?'+':''}${m.sentiment}]</span> ${m.silent?'(划过)':m.text}
      </div>`;
    });
  }
  if (showSlider) {
    html += `<div style="margin-top:8px; padding:6px; border:1px dashed var(--accent); border-radius:4px;">
      <div style="font-size:11px; color:var(--accent)"><b>do()</b> 干预：把 agent #${id} 的初始立场覆盖为 ↓</div>
      <input type="range" id="intervene-slider" min="-1" max="1" step="0.1" value="${init.toFixed(1)}" oninput="document.getElementById('intervene-val').textContent=parseFloat(this.value).toFixed(1)" style="width:100%">
      <div style="display:flex; justify-content:space-between;">
        <span>-1.0 (强负)</span><b id="intervene-val" style="color:var(--accent)">${init.toFixed(1)}</b><span>+1.0 (强正)</span>
      </div>
      <button class="secondary" style="margin-top:4px; font-size:11px; padding:4px 8px;" onclick="applyIntervention('${id}')">🔮 重跑群聊（30-60s）看反事实</button>
      <div class="hint" style="margin-top:4px;">⚠️ 完整重跑当前用 backend 同样的 LLM 调用，约一分钟。</div>
    </div>`;
  } else {
    html += `<div class="hint" style="margin-top:6px;">右键节点 → 试因果干预 do(stance=...)</div>`;
  }
  document.getElementById("chat-agent-panel").innerHTML = html;
}

async function applyIntervention(agentId) {
  const slider = document.getElementById("intervene-slider");
  if (!slider) return;
  const newStance = parseFloat(slider.value);
  log(`🔮 do(agent ${agentId}.initial_stance = ${newStance.toFixed(2)})... 重跑群聊`);
  // For MVP we just re-run predict with same params — the real do() would
  // require backend support for groupchat_overrides. Show this as a TODO.
  alert(`已记录干预: agent ${agentId} 初始立场 → ${newStance.toFixed(2)}\n\n` +
        `MVP 版本：重跑群聊会重新随机种子，不保证差异是干预带来的。\n` +
        `生产版需后端支持 groupchat_overrides 字段（已在 roadmap）。\n\n` +
        `点 🚀 预测重跑看新群聊轨迹。`);
}

function drawBrandMemory(timeline) {
  const c = document.getElementById("bm-canvas");
  if (!c || !timeline) return;
  const ctx = c.getContext("2d"), W=c.width, H=c.height, pad=26;
  ctx.clearRect(0,0,W,H);
  const N = timeline.length;
  const xs = i => pad + i*(W-pad*2)/Math.max(N-1,1);
  const y = v => H-pad - v*(H-pad*2);
  ctx.strokeStyle="#222a38"; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  const metrics = [["brand_favor_pct","#5ed39b","favor"], ["brand_recall_pct","#6ea8fe","recall"], ["purchase_intent_pct","#ffc857","purch"], ["brand_aversion_pct","#ff7a85","aversion"]];
  metrics.forEach(([k,color,label],idx) => {
    ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
    timeline.forEach((m,i) => { const v=m[k]||0; if (i===0) ctx.moveTo(xs(i),y(v)); else ctx.lineTo(xs(i),y(v)); });
    ctx.stroke();
    ctx.fillStyle=color; ctx.font="10px monospace"; ctx.fillText(label, W-pad-50, 12+idx*12);
  });
  ctx.fillStyle="#8b93a7"; ctx.font="10px monospace"; ctx.fillText("D0", pad-8, H-8);
  ctx.fillText("D"+(N-1), W-pad-20, H-8);
}

async function refreshWorld() {
  document.getElementById("world-panel").innerHTML = "<div class='hint'>GPT 拉取中（10-20s）…</div>";
  const r = await fetch(API + "/api/world/refresh", {method:"POST"});
  const w = await r.json();
  renderWorld(w);
}

async function loadWorld() {
  try {
    const r = await fetch(API + "/api/world");
    renderWorld(await r.json());
  } catch(e) {}
}

function renderWorld(w, summaryOnly=false) {
  const el = document.getElementById("world-panel");
  if (!w) { el.innerHTML = "<div class='hint'>暂无</div>"; return; }
  let html = `<div class="hint">📡 ${w.source||'?'} · 抓取于 ${w.fetched_at||'?'} · sentiment: <b>${w.sentiment}</b> · avg_impact ${w.avg_consumer_impact||0}</div>`;
  const evs = w.events || w.top_events || [];
  if (summaryOnly) {
    html += evs.slice(0,3).map(e => `<div style="margin:4px 0; padding:4px; background:#0f141c; border-radius:4px;">• ${typeof e==='string'?e:e.title}</div>`).join('');
  } else {
    html += evs.map(e => `
      <div style="margin:5px 0; padding:6px; background:#0f141c; border-radius:4px;">
        <div><span class="badge">${e.category||'?'}</span> ${e.title}</div>
        <div class="hint" style="margin-top:3px;">impact ${e.consumer_impact>=0?'+':''}${e.consumer_impact} · attn ${(e.attention_share*100||0).toFixed(0)}% · cat: ${(e.affected_categories||[]).slice(0,3).join('/')}</div>
      </div>`).join('');
  }
  el.innerHTML = html;
}
loadWorld();

// Lightweight Markdown → HTML (headings, bold, tables, lists, code, HR)
function mdToHtml(md) {
  if (!md) return "";
  let html = md;
  // Escape
  html = html.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Headings
  html = html.replace(/^#### (.*)$/gm, '<h4 style="color:#a855f7; margin:10px 0 4px;">$1</h4>');
  html = html.replace(/^### (.*)$/gm, '<h3 style="color:#a855f7; margin:12px 0 6px;">$1</h3>');
  html = html.replace(/^## (.*)$/gm, '<h2 style="color:var(--accent); margin:16px 0 8px; border-bottom:1px solid #2a3040; padding-bottom:4px;">$1</h2>');
  html = html.replace(/^# (.*)$/gm, '<h1 style="color:#e6e8ee; margin:6px 0 10px;">$1</h1>');
  // Tables (simple pipe)
  html = html.replace(/((?:^\|.*\|\n?)+)/gm, (block) => {
    const lines = block.trim().split('\n');
    if (lines.length < 2) return block;
    const head = lines[0].split('|').slice(1,-1).map(c=>c.trim());
    const rows = lines.slice(2).map(l=>l.split('|').slice(1,-1).map(c=>c.trim()));
    return `<table style="border-collapse:collapse; margin:8px 0; font-size:12px;"><thead><tr>${head.map(h=>`<th style="padding:4px 8px; text-align:left; background:#1a2030;">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td style="padding:3px 8px; border-top:1px solid #1a2030;">${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  });
  // Lists
  html = html.replace(/^(?:- |\* )(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]+?<\/li>(?:\n<li>[\s\S]+?<\/li>)*)/g, '<ul style="margin:4px 0 8px 20px;">$1</ul>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  // HR
  html = html.replace(/^---+$/gm, '<hr style="border-color:#2a3040; margin:12px 0;">');
  // Paragraphs
  html = html.split(/\n\n+/).map(p => p.match(/^<(h\d|ul|table|hr)/) ? p : `<p style="margin:6px 0;">${p.replace(/\n/g,'<br>')}</p>`).join('');
  return html;
}

function downloadReport(reportId) {
  const md = document.getElementById('final-report-md')?.textContent || '';
  if (!md) return alert('无报告内容');
  const blob = new Blob([md], {type: 'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `oransim_${reportId || 'report'}.md`;
  a.click(); URL.revokeObjectURL(url);
}

function printReport() {
  const html = document.getElementById('final-report-rendered')?.innerHTML || '';
  if (!html) return alert('无报告内容');
  const w = window.open('', '_blank');
  w.document.write(`<html><head><title>Oransim 预测报告</title><style>
    body{font-family:system-ui,sans-serif;max-width:780px;margin:30px auto;padding:20px;color:#222;line-height:1.7;}
    h1,h2,h3,h4{color:#6b21a8;} table{border-collapse:collapse;margin:10px 0;} th{background:#f3f0ff;padding:6px 10px;text-align:left;} td{padding:5px 10px;border-top:1px solid #ddd;} hr{border:0;border-top:1px solid #ddd;margin:16px 0;} ul{margin:6px 0 10px 24px;}
  </style></head><body>${html}</body></html>`);
  w.document.close(); setTimeout(()=>w.print(), 300);
}

function renderSchemaOutputs(so) {
  const el = document.getElementById("schema-content");
  if (!el) return;
  if (!so || so._error) {
    el.innerHTML = so?._error ? `<div class="hint">生成失败: ${so._error}</div>` : "";
    return;
  }
  const f = so.T1_A2_mc_funnel_prediction || {};
  const b = so.T1_A1_funnel_beta_fit || [];
  const d = so.T2_A4_ugc_diffusion_simulation || {};
  const em = so.T3_A3_emergent_metrics || [];
  const sens = so.T3_A5_sensitivity_analysis || {};
  const pts = so.T3_A2_platform_simulation_ts || [];
  const personas = so.T3_A1_agent_persona || [];
  const rpt = so.report_market_insight || {};
  const compRoi = so.T1_A3_competitor_audience_roi;
  const kolMix = so.T2_A1_kol_mix_optimization;
  const kolReinvest = so.T2_A5_kol_reinvest_ranking || [];
  const searchElast = so.T3_A6_search_elasticity;
  const tagLift = so.T2_A3_tag_lift_ranking;
  const kolMatch = so.T2_A2_kol_content_match;
  const ctCoef = so.T3_A7_content_type_coefficient;
  const scenComp = so.T3_A4_scenario_comparison;
  const finalRpt = so.report_strategy_case;

  const fn = (v) => v == null ? "?" : Number(v).toLocaleString('zh-CN', {maximumFractionDigits:1});

  const funnelRow = (stage, label) => {
    const x = f[stage]; if (!x) return "";
    return `<tr><td>${label}</td><td>${fn(x.p25)}</td><td><b>${fn(x.p50)}</b></td><td>${fn(x.p75)}</td></tr>`;
  };

  el.innerHTML = `
    ${finalRpt ? `
    <div style="padding:14px; background:linear-gradient(135deg, #1a1030 0%, #0f141c 100%); border-radius:6px; border:2px solid #a855f7; margin-top:8px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <div>
          <b style="color:#a855f7; font-size:16px;">📄 最终预测报告</b>
          <span class="badge" style="margin-left:8px; background:${finalRpt.source==='gpt-5.4'?'#a855f7':'#555'};">${finalRpt.source}</span>
          ${finalRpt.tokens_out ? `<span class="hint" style="margin-left:8px;">${finalRpt.tokens_in}/${finalRpt.tokens_out} tok · ¥${finalRpt.cost_cny} · ${finalRpt.generation_ms}ms</span>` : `<span class="hint" style="margin-left:8px;">模板生成 · ${finalRpt.generation_ms}ms</span>`}
        </div>
        <div style="display:flex; gap:6px;">
          <button class="secondary" style="font-size:11px; padding:4px 10px;" onclick="navigator.clipboard.writeText(document.getElementById('final-report-md').textContent); this.textContent='✅ 已复制'">📋 复制 MD</button>
          <button class="secondary" style="font-size:11px; padding:4px 10px;" onclick="downloadReport(${JSON.stringify(finalRpt.report_id)})">⬇ 下载 MD</button>
          <button class="secondary" style="font-size:11px; padding:4px 10px;" onclick="printReport()">🖨 打印/PDF</button>
        </div>
      </div>
      <div id="final-report-rendered" style="background:#0a0e16; padding:14px; border-radius:4px; max-height:520px; overflow:auto; font-size:13px; line-height:1.7;"></div>
      <pre id="final-report-md" style="display:none;">${(finalRpt.report_content||'').replace(/</g,'&lt;')}</pre>
    </div>
    ` : ''}

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px;">

      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T1-A2 · 五阶漏斗 (P25/50/75)</b>
        <table style="width:100%; margin-top:6px; border-collapse:collapse;">
          <thead><tr style="color:#888;"><th style="text-align:left;">阶段</th><th>P25 悲观</th><th>P50</th><th>P75 乐观</th></tr></thead>
          <tbody>
            ${funnelRow('A1_awareness','A1 曝光')}
            ${funnelRow('A2_interest','A2 兴趣')}
            ${funnelRow('A3_engagement','A3 互动')}
            ${funnelRow('A4_conversion','A4 转化')}
            ${funnelRow('A5_loyalty','A5 复购')}
          </tbody>
        </table>
        <div class="hint" style="margin-top:4px">spread ${f.spread_pct}% · 50% CI</div>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T1-A1 · Beta 分布拟合</b>
        <table style="width:100%; margin-top:6px; border-collapse:collapse;">
          <thead><tr style="color:#888;"><th style="text-align:left">转化对</th><th>α</th><th>β</th><th>均值</th><th>方差</th></tr></thead>
          <tbody>
            ${b.map(x=>`<tr><td>${x.funnel_transition}</td><td>${x.alpha_param}</td><td>${x.beta_param}</td><td>${(x.mean_rate*100).toFixed(2)}%</td><td>${x.variance.toExponential(2)}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T2-A4 · UGC 扩散模拟</b>
        <div style="margin-top:6px; line-height:1.7;">
          · 峰值日：Day <b>${d.peak_day}</b><br>
          · 峰值 UGC：<b>${fn(d.peak_ugc_count)}</b><br>
          · 半衰期 t½：<b>${d.half_life_days}</b> 天<br>
          · 14 天总量：<b>${fn(d.total_ugc_predicted)}</b><br>
          <span class="hint">曲线: ${(d.diffusion_curve||[]).slice(0,14).map(fn).join(' → ')}</span>
        </div>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T3-A3 · 涌现指标</b>
        <table style="width:100%; margin-top:6px; border-collapse:collapse;">
          <thead><tr style="color:#888;"><th style="text-align:left">指标</th><th>实测</th><th>基准</th><th>偏差</th></tr></thead>
          <tbody>
            ${em.map(x=>`<tr><td>${x.metric_name}</td><td><b>${fn(x.metric_value)}</b></td><td>${fn(x.chain_formula_value)}</td><td style="color:${x.deviation_rate>0.2?'var(--bad)':'var(--good)'}">${(x.deviation_rate*100).toFixed(1)}%</td></tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T3-A5 · 敏感性分析（龙卷风图 Top 8，±20% 扰动）</b>
        <div style="margin-top:6px;">
          ${(sens.parameters||[]).slice(0,8).map(p => {
            const amp = p.gmv_change_amplitude;
            const max = Math.max(...(sens.parameters||[]).map(x=>x.gmv_change_amplitude));
            const w = max ? (amp/max*100).toFixed(0) : 0;
            const col = p.elasticity > 0 ? 'var(--good)' : 'var(--bad)';
            return `<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
              <span style="width:110px; font-size:11px;">${p.parameter_name}</span>
              <div style="flex:1; height:12px; background:#1a1f2a;"><div style="width:${w}%; height:100%; background:${col};"></div></div>
              <span style="width:100px; text-align:right; font-size:11px;">¥${fn(amp)} <span style="color:${col};">${p.elasticity>0?'+':''}${p.elasticity.toFixed(2)}</span></span>
            </div>`;
          }).join('')}
        </div>
        <div class="hint" style="margin-top:4px">越靠上 = 改动该参数对 GMV 影响越大；颜色：绿=正向 / 红=负向</div>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T3-A2 · 平台日级时序（展示前 12 行）</b>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:10px;">
          <thead><tr style="color:#888;"><th>Day</th><th>平台</th><th>帖子</th><th>评论</th><th>分享</th><th>正面%</th><th>负面%</th></tr></thead>
          <tbody>
            ${pts.slice(0,12).map(x=>`<tr><td>${x.day}</td><td>${x.platform}</td><td>${fn(x.post_count)}</td><td>${fn(x.comment_count)}</td><td>${fn(x.share_count)}</td><td>${(x.emotion_positive*100).toFixed(0)}%</td><td>${(x.emotion_negative*100).toFixed(0)}%</td></tr>`).join('')}
          </tbody>
        </table>
      </div>

      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T3-A1 · Agent 人设 (Top ${Math.min(personas.length,10)}/${personas.length})</b>
        <div style="margin-top:6px; max-height:200px; overflow:auto; font-size:11px;">
          ${personas.slice(0,10).map(p=>`
            <div style="padding:4px; border-bottom:1px solid #1a2030;">
              <b>${p.persona_name}</b> <span class="badge">${p.crowd_segment}</span>
              <span class="hint" style="display:inline;">${p.verdict.reason||''}</span>
            </div>
          `).join('')}
        </div>
      </div>

      ${compRoi && compRoi.rows && compRoi.rows.length ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T1-A3 · 竞品受众抢夺 ROI (GPT-5.4 估算)</b>
        <span class="hint" style="margin-left:8px">${compRoi.llm_available?'真 LLM':'mock'} · 成本 ¥${compRoi.total_cost_cny||0}</span>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:11px;">
          <thead><tr style="color:#888;"><th style="text-align:left">竞品</th><th>重合率</th><th>重合粉丝</th><th>预估 ROI</th><th>可抢夺转化</th><th>TGI 标签</th></tr></thead>
          <tbody>
            ${compRoi.rows.map(r=>{
              const tgi = r.tgi_top_tags ? Object.entries(r.tgi_top_tags).slice(0,3).map(([k,v])=>`${k}(${v})`).join(' ') : '';
              return `<tr><td><b>${r.competitor_name}</b></td><td>${r.overlap_ratio?(r.overlap_ratio*100).toFixed(1)+'%':'-'}</td><td>${fn(r.overlap_fans_count)}</td><td style="color:${r.estimated_roi>2?'var(--good)':'var(--warn)'}">${r.estimated_roi||'-'}</td><td>${fn(r.estimated_conversion)}</td><td style="color:#a855f7">${tgi}</td></tr>${r.reasoning?`<tr><td colspan="6" style="padding:2px 10px; color:#888; font-size:10px;">💭 ${r.reasoning}</td></tr>`:''}`;
            }).join('')}
          </tbody>
        </table>
      </div>` : ''}

      ${kolMix && !kolMix._error ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T2-A1 · KOL 组合优化 (${kolMix.solver_status})</b>
        <span class="hint" style="margin-left:8px">从 ${kolMix.candidate_pool_size} KOL 池选出 ${kolMix.total_selected} 人</span>
        <div style="display:flex; gap:14px; margin:6px 0; font-size:11px; flex-wrap:wrap;">
          <div>KOL:KOC <b>${kolMix.kol_koc_ratio}</b></div>
          <div>预算利用 <b>${(kolMix.budget_utilization*100).toFixed(1)}%</b> (¥${fn(kolMix.estimated_cost)}/${fn(kolMix.budget)})</div>
          <div>总触达 <b style="color:var(--good)">${fn(kolMix.estimated_total_reach)}</b></div>
          <div>总互动 <b>${fn(kolMix.estimated_total_engagement)}</b></div>
          <div>平均 ROI <b style="color:var(--accent)">${kolMix.estimated_roi}</b></div>
        </div>
        <details style="font-size:10px;"><summary style="cursor:pointer; color:var(--accent);">▾ 入选 ${kolMix.total_selected} 位 KOL 明细</summary>
          <table style="width:100%; margin-top:4px; border-collapse:collapse;">
            <thead><tr style="color:#888;"><th style="text-align:left">达人</th><th>赛道</th><th>层级</th><th>粉丝</th><th>成本</th><th>触达</th><th>ROI</th><th>复投</th></tr></thead>
            <tbody>
              ${(kolMix.selected_kols||[]).slice(0,30).map(k=>{
                const reinv = kolReinvest.find(r=>r.kol_id===k.kol_id);
                const rec = reinv?.recommendation || '';
                const recCol = rec==='优先复投'?'var(--good)':rec==='替换'?'var(--bad)':'var(--warn)';
                return `<tr><td>${k.name}</td><td>${k.niche}</td><td><span class="badge">${k.tier}</span></td><td>${fn(k.fans)}</td><td>¥${fn(k.cost)}</td><td>${fn(k.reach)}</td><td>${k.roi}</td><td style="color:${recCol}">${rec}</td></tr>`;
              }).join('')}
            </tbody>
          </table>
        </details>
      </div>` : ''}

      ${kolMatch && !kolMatch._error && kolMatch.ranked_kols ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T2-A2 · KOL 内容匹配 (LLM Brief + Top10 排序)</b>
        <span class="hint" style="margin-left:8px">从 ${fn(kolMatch.candidate_pool_size)} KOL 候选池 · 模型: ${kolMatch.agent_model} · ¥${kolMatch.total_cost_cny}</span>
        <div style="margin:6px 0; padding:6px; background:#070a0f; border-radius:3px;">
          <b style="color:#a855f7">📋 GPT 生成 Brief：</b>${kolMatch.brief_full?.creative_angle || kolMatch.brief_content || '?'}
          <div style="font-size:10px; color:var(--muted); margin-top:3px;">
            人群标签: ${(kolMatch.target_crowd_tags||[]).join(' · ')} · 形式: ${kolMatch.brief_full?.content_format_hint||'?'} · 调性: ${kolMatch.brief_full?.tone||'?'}
            ${kolMatch.brief_full?.key_selling_points ? '<br>核心卖点: ' + kolMatch.brief_full.key_selling_points.join(' / ') : ''}
            ${kolMatch.brief_full?.ideal_kol_profile ? '<br>理想达人: ' + kolMatch.brief_full.ideal_kol_profile : ''}
          </div>
        </div>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:11px;">
          <thead><tr style="color:#888;"><th style="text-align:left">排名</th><th style="text-align:left">达人</th><th>赛道</th><th>层级</th><th>粉丝</th><th>互动率</th><th>匹配分</th><th>匹配理由</th></tr></thead>
          <tbody>
            ${kolMatch.ranked_kols.slice(0,10).map((k,i)=>{
              const isTop3 = i<3;
              const rowBg = isTop3 ? 'background:#1a1030' : '';
              return `<tr style="${rowBg}"><td><b>${i+1}</b>${isTop3?' 🌟':''}</td><td>${k.name}</td><td>${k.niche}</td><td><span class="badge">${k.tier}</span></td><td>${fn(k.fans)}</td><td>${(k.interaction_rate*100).toFixed(2)}%</td><td><b style="color:${k.match_score>0.7?'var(--good)':k.match_score>0.5?'var(--warn)':'var(--muted)'}">${k.match_score}</b></td><td style="color:${isTop3?'#a855f7':'var(--muted)'}; font-size:10px;">${k.match_reason||'-'}${k.concern?` <span style="color:var(--bad)">⚠ ${k.concern}</span>`:''}</td></tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>` : ''}

      ${tagLift && !tagLift._error && tagLift.rows && tagLift.rows.length ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T2-A3 · 话题标签 Lift 排序（${tagLift.target_niche}）</b>
        <span class="hint" style="margin-left:8px">从 ${fn(tagLift.n_notes_in_niche)} 条笔记 / ${fn(tagLift.n_notes_global)} 条全局 / 共 ${fn(tagLift.n_unique_tags)} 个 unique tags 计算 prevalence Lift</span>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:11px;">
          <thead><tr style="color:#888;"><th style="text-align:left">话题/关键词</th><th>Lift</th><th>支持度</th><th>置信度</th><th>样本</th><th>趋势</th></tr></thead>
          <tbody>
            ${tagLift.rows.slice(0,15).map(r=>`<tr><td>${r.hashtag}</td><td><b style="color:${r.lift_score>2?'var(--good)':r.lift_score>1?'var(--warn)':'var(--muted)'}">${r.lift_score}x</b></td><td>${(r.support*100).toFixed(2)}%</td><td>${(r.confidence*100).toFixed(2)}%</td><td>${r.sample_count}</td><td>${r.is_trending?'🔥':''}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>` : ''}

      ${ctCoef && !ctCoef._error && ctCoef.rows && ctCoef.rows.length ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T3-A7 · 内容类型效果系数（${ctCoef.target_niche}）</b>
        <span class="hint" style="margin-left:6px">${ctCoef.method}</span>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:11px;">
          <thead><tr style="color:#888;"><th style="text-align:left">类型</th><th>CPM 系数</th><th>互动系数</th><th>转化系数</th><th>样本 n</th><th>p 值</th></tr></thead>
          <tbody>
            ${ctCoef.rows.map(r=>`<tr><td><b>${r.content_type}</b></td><td>${r.cpm_coefficient}x</td><td style="color:${r.engagement_coefficient>1?'var(--good)':'var(--bad)'}">${r.engagement_coefficient}x</td><td>${r.cvr_coefficient}x</td><td>${r.sample_count}</td><td>${r.is_significant?'<b style="color:var(--good)">'+r.significance_p+' ✓</b>':r.significance_p}</td></tr>`).join('')}
          </tbody>
        </table>
        <div class="hint" style="margin-top:4px">基线互动 ${(ctCoef.baseline_engagement*100).toFixed(2)}% · 系数 > 1 = 该格式胜过赛道均值</div>
      </div>` : ''}

      ${scenComp && !scenComp._error && scenComp.kpi_comparison ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px;">
        <b style="color:var(--accent)">T3-A4 · 情景对比 + Wilcoxon 显著性</b>
        <div style="font-size:11px; margin:4px 0; color:var(--muted)">A: ${scenComp.scenario_a_desc} / B: ${scenComp.scenario_b_desc} · n=${scenComp.n_samples}</div>
        <table style="width:100%; margin-top:6px; border-collapse:collapse; font-size:11px;">
          <thead><tr style="color:#888;"><th style="text-align:left">KPI</th><th>A 均值</th><th>B 均值</th><th>Δ%</th><th>p 值</th><th>显著</th></tr></thead>
          <tbody>
            ${scenComp.kpi_comparison.map(r=>`<tr><td>${r.kpi}</td><td>${r.scenario_a_mean}</td><td>${r.scenario_b_mean}</td><td style="color:${(r.delta_pct||0)>0?'var(--good)':'var(--bad)'}">${r.delta_pct?(r.delta_pct>0?'+':'')+r.delta_pct+'%':'-'}</td><td>${r.p_value}</td><td>${r.is_significant_05?'<b style="color:var(--good)">✓ p<0.05</b>':'—'}</td></tr>`).join('')}
          </tbody>
        </table>
        <div style="margin-top:6px; padding:6px; background:#070a0f; border-radius:3px; font-size:11px;">
          🎯 <b>推荐方案：${scenComp.recommended_scenario}</b> · ${scenComp.recommendation_reason}
        </div>
      </div>` : ''}

      ${searchElast && !searchElast._error ? `
      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">T3-A6 · 搜索弹性回归 (log-log OLS)</b>
        <span class="hint" style="margin-left:8px">数据源: ${searchElast.data_source||'?'}</span>
        <div style="display:flex; gap:14px; margin:6px 0; font-size:11px; flex-wrap:wrap;">
          <div>弹性 ε <b style="color:var(--accent)">${searchElast.elasticity_coeff}</b></div>
          <div>95% CI [<b>${searchElast.confidence_lower}, ${searchElast.confidence_upper}</b>]</div>
          <div>R² <b style="color:${searchElast.r_squared>0.5?'var(--good)':'var(--warn)'}">${searchElast.r_squared}</b></div>
          <div>样本 n=<b>${searchElast.sample_size}</b></div>
          <div>DW <b>${searchElast.dw_statistic}</b></div>
          <div>残差正态性 <b style="color:${searchElast.residual_normality==='pass'?'var(--good)':'var(--warn)'}">${searchElast.residual_normality}</b></div>
        </div>
        <div style="font-size:11px; color:#ccc; padding:6px; background:#070a0f; border-radius:3px;">
          💡 ${searchElast.interpretation}
        </div>
      </div>` : ''}

      <div style="padding:8px; background:#0f141c; border-radius:4px; grid-column:span 2;">
        <b style="color:var(--accent)">report_market_insight · 市场洞察报告 (MD · T1 模块版)</b>
        <button class="secondary" style="font-size:11px; padding:3px 8px; margin-left:8px;" onclick="navigator.clipboard.writeText(document.getElementById('schema-report-md').textContent); this.textContent='✅ 已复制'">📋 复制 MD</button>
        <pre id="schema-report-md" style="margin-top:6px; padding:8px; background:#070a0f; border-radius:3px; font-size:10px; max-height:260px; overflow:auto; white-space:pre-wrap;">${(rpt.report_content||'').replace(/</g,'&lt;')}</pre>
      </div>

    </div>
  `;
  // Render the final report MD to HTML
  if (finalRpt && finalRpt.report_content) {
    const rendered = document.getElementById('final-report-rendered');
    if (rendered) rendered.innerHTML = mdToHtml(finalRpt.report_content);
  }
}

function renderPredictedSentiment(ps) {
  const el = document.getElementById("predicted-sentiment");
  if (!el) return;
  if (!ps) { el.innerHTML = ""; return; }
  const dist = ps.sentiment_distribution || {};
  const pos = dist.positive || 0, neu = dist.neutral || 0, neg = dist.negative || 0;
  const net = ps.net_sentiment_score || 0;
  const netCol = net > 0.2 ? 'var(--good)' : net < -0.2 ? 'var(--bad)' : 'var(--warn)';
  const themes = (ps.key_opinion_themes || []).map(t =>
    `<span class="badge" style="background:#1a2030; margin:2px;">${t.theme} <b>${t.count}</b></span>`).join('');
  const bar = (pct, col) => `<div style="flex:${pct}; background:${col}; height:12px;" title="${(pct*100).toFixed(1)}%"></div>`;
  el.innerHTML = `
    <div style="padding:8px; background:#0f141c; border-radius:5px; border-left:3px solid ${netCol};">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b style="color:var(--accent); font-size:12px;">🎯 本广告 AI 投票情感预测</b>
        <span style="font-size:10px; color:var(--muted);">${ps.agent_count} AI (${ps.llm_backed} LLM 真问)</span>
      </div>
      <div style="display:flex; height:12px; border-radius:3px; overflow:hidden; margin:6px 0;">
        ${pos>0?bar(pos,'var(--good)'):''}${neu>0?bar(neu,'#666'):''}${neg>0?bar(neg,'var(--bad)'):''}
      </div>
      <div style="font-size:11px; display:flex; gap:12px; flex-wrap:wrap;">
        <span style="color:var(--good)">正面 ${(pos*100).toFixed(1)}%</span>
        <span style="color:#aaa">中性 ${(neu*100).toFixed(1)}%</span>
        <span style="color:var(--bad)">负面 ${(neg*100).toFixed(1)}%</span>
        <span style="color:${netCol}">净情感 <b>${net.toFixed(2)}</b></span>
        <span style="color:#a855f7">高购意 ${(ps.high_intent_pct*100).toFixed(1)}%</span>
        <span class="hint" style="display:inline">平均购意 ${ps.avg_purchase_intent_7d}</span>
      </div>
      ${themes ? `<div style="margin-top:6px; font-size:11px;"><span class="hint">高频关注：</span>${themes}</div>` : ''}
    </div>`;
}

let SOUL_QUOTES_ALL = [];
let SOUL_FILTER = "all";
let SOUL_PAGE = 0;
const SOUL_PAGE_SIZE = 40;

function renderSouls(quotes) {
  if (!quotes) { document.getElementById("souls").innerHTML=""; return; }
  SOUL_QUOTES_ALL = quotes;
  SOUL_FILTER = "all";
  SOUL_PAGE = 0;
  renderSoulView();
}

function renderSoulView() {
  const quotes = SOUL_QUOTES_ALL || [];
  const cost = quotes[0]?._batch_cost_cny;
  const tk = quotes[0]?._batch_tokens;
  const wk = quotes[0]?._batch_workers;
  const engine = quotes[0]?._batch_engine || 'thread_pool';

  // Aggregate stats on the FULL set (not filtered)
  const n = quotes.length;
  const clicked = quotes.filter(q=>q.will_click).length;
  const skipped = n - clicked;
  const feelCount = {};
  quotes.forEach(q => { const f=q.feel||'无'; feelCount[f]=(feelCount[f]||0)+1; });
  const avgIntent = n ? (quotes.reduce((s,q)=>s+(+q.purchase_intent_7d||0),0)/n) : 0;
  const highIntent = quotes.filter(q=>(+q.purchase_intent_7d||0) > 0.6).length;
  // Top reasons (top-8)
  const reasonCount = {};
  quotes.forEach(q => { const r=q.reason||'?'; reasonCount[r]=(reasonCount[r]||0)+1; });
  const topReasons = Object.entries(reasonCount).sort((a,b)=>b[1]-a[1]).slice(0,8);

  // Filter selection
  let shown = quotes;
  if (SOUL_FILTER === "click") shown = quotes.filter(q=>q.will_click);
  else if (SOUL_FILTER === "skip") shown = quotes.filter(q=>!q.will_click);
  else if (SOUL_FILTER.startsWith("feel:")) {
    const f = SOUL_FILTER.slice(5);
    shown = quotes.filter(q=>q.feel===f);
  }
  const totalPages = Math.max(1, Math.ceil(shown.length / SOUL_PAGE_SIZE));
  SOUL_PAGE = Math.min(SOUL_PAGE, totalPages-1);
  const page = shown.slice(SOUL_PAGE*SOUL_PAGE_SIZE, (SOUL_PAGE+1)*SOUL_PAGE_SIZE);

  const feelBar = Object.entries(feelCount).sort((a,b)=>b[1]-a[1]).map(([f,c]) => {
    const pct = n ? (c*100/n).toFixed(0) : 0;
    return `<span class="badge" style="cursor:pointer; ${SOUL_FILTER===('feel:'+f)?'background:#a855f7;color:#fff':''}" onclick="SOUL_FILTER='feel:${f}'; SOUL_PAGE=0; renderSoulView();">${f} ${c} (${pct}%)</span>`;
  }).join(' ');

  document.getElementById("souls").innerHTML = `
    ${cost !== undefined ? `<div class="hint" style="padding:4px 6px; background:#0f141c; border-radius:3px;">真 LLM · ${quotes.filter(q=>q.source==='llm').length}/${n} agents · engine:<b>${engine}</b> · tokens ${tk?.in}/${tk?.out} · 成本 <b style="color:var(--accent)">¥${cost}</b></div>` : ""}

    <div style="margin:8px 0; padding:8px; background:#0f141c; border-radius:5px;">
      <div style="display:flex; gap:16px; font-size:12px; flex-wrap:wrap;">
        <div><b style="color:var(--good)">👆 点了</b> ${clicked} <span class="hint" style="display:inline">(${n?(clicked*100/n).toFixed(1):0}%)</span></div>
        <div><b style="color:var(--muted)">✖️ 跳过</b> ${skipped} <span class="hint" style="display:inline">(${n?(skipped*100/n).toFixed(1):0}%)</span></div>
        <div><b style="color:#a855f7">💓 7日购意 ≥0.6</b> ${highIntent} <span class="hint" style="display:inline">(${n?(highIntent*100/n).toFixed(1):0}%)</span></div>
        <div><b>平均购意</b> ${avgIntent.toFixed(2)}</div>
      </div>
      <div style="margin-top:6px; font-size:11px;">情绪分布：${feelBar}</div>
      ${topReasons.length>1 ? `<details style="margin-top:6px"><summary class="hint" style="cursor:pointer; font-size:11px;">▾ 高频理由 Top 8</summary><div style="font-size:11px; margin-top:4px;">${topReasons.map(([r,c])=>`<div style="padding:2px 0">• ${r} <span class="badge">${c}</span></div>`).join('')}</div></details>` : ''}
    </div>

    <div style="display:flex; gap:6px; margin-bottom:6px; font-size:11px; align-items:center;">
      <span class="hint">筛选:</span>
      <span class="badge" style="cursor:pointer; ${SOUL_FILTER==='all'?'background:#a855f7;color:#fff':''}" onclick="SOUL_FILTER='all'; SOUL_PAGE=0; renderSoulView();">全部 ${n}</span>
      <span class="badge" style="cursor:pointer; ${SOUL_FILTER==='click'?'background:var(--good);color:#000':''}" onclick="SOUL_FILTER='click'; SOUL_PAGE=0; renderSoulView();">点了 ${clicked}</span>
      <span class="badge" style="cursor:pointer; ${SOUL_FILTER==='skip'?'background:#555;color:#fff':''}" onclick="SOUL_FILTER='skip'; SOUL_PAGE=0; renderSoulView();">跳过 ${skipped}</span>
      <span style="margin-left:auto; color:var(--muted); font-size:11px;">第 ${SOUL_PAGE+1}/${totalPages} 页 · 共 ${shown.length} 条</span>
    </div>

    <div>${page.map(s =>
      `<div class="soul ${s.will_click?'':'skip'}">
        <div class="oneliner">${s.persona_oneliner||'(no persona)'} <span class="badge">${s.source||'mock'}</span> <span class="badge" style="background:#333">${s.feel||''}</span></div>
        <div class="reason"><b>${s.will_click?'👆 点了':'✖️ 跳过'}</b> · ${s.reason||''}
        <span class="badge">7 日购意 ${s.purchase_intent_7d}</span></div>
        ${s.comment?`<div class="comment">"${s.comment}"</div>`:""}
      </div>`).join("")}</div>

    ${totalPages>1 ? `
    <div style="display:flex; gap:6px; margin-top:8px;">
      <button class="secondary" style="font-size:11px; padding:4px 10px;" ${SOUL_PAGE===0?'disabled':''} onclick="SOUL_PAGE--; renderSoulView();">← 上一页</button>
      <button class="secondary" style="font-size:11px; padding:4px 10px;" ${SOUL_PAGE>=totalPages-1?'disabled':''} onclick="SOUL_PAGE++; renderSoulView();">下一页 →</button>
      <button class="secondary" style="font-size:11px; padding:4px 10px;" onclick="SOUL_PAGE=totalPages-1; renderSoulView();">末页</button>
      <span class="hint" style="align-self:center;">跳页: <input type="number" min="1" max="${totalPages}" value="${SOUL_PAGE+1}" style="width:50px; padding:2px;" onchange="SOUL_PAGE=Math.max(0,Math.min(${totalPages-1},+this.value-1)); renderSoulView();"></span>
    </div>` : ''}
  `;
}

async function onSlide() {
  const dv = +document.getElementById("douyin_slider").value;
  const xv = +document.getElementById("xhs_slider").value;
  const bv = +document.getElementById("budget_slider").value;
  document.getElementById("douyin_v").textContent = dv + "%";
  document.getElementById("xhs_v").textContent = xv + "%";
  document.getElementById("budget_v").textContent = (bv/10000).toFixed(1) + "万";
  document.getElementById("budget").value = bv;
  if (!SESSION_ID) return;
  const sum = dv + xv || 1;
  const patch = {
    total_budget: bv,
    platform_alloc: { douyin: dv/sum, xhs: xv/sum },
  };
  const r = await fetch(API + "/api/sandbox/session/" + SESSION_ID, {
    method:"PATCH", headers:{"content-type":"application/json"},
    body: JSON.stringify(patch)
  });
  const s = await r.json();
  renderSnapshot(s);

  // CATE from counterfactual
  const cfr = await fetch(API + "/api/sandbox/session/" + SESSION_ID + "/counterfactual", {
    method:"POST", headers:{"content-type":"application/json"},
    body: JSON.stringify(patch)
  });
  const cf = await cfr.json();
  renderCATE(cf.cate);
}

function renderSnapshot(s) {
  const k = s.current_kpis, b = s.baseline_kpis, d = s.delta;
  const keys = [["roi","ROI",x=>x.toFixed(2)+"x"],
                ["ctr","CTR",x=>(x*100).toFixed(2)+"%"],
                ["cvr","CVR",x=>(x*100).toFixed(2)+"%"],
                ["impressions","曝光",x=>(x/10000).toFixed(1)+"万"],
                ["clicks","点击",x=>Math.round(x).toLocaleString()],
                ["conversions","转化",x=>Math.round(x).toLocaleString()]];
  const grid = keys.map(([key,label,fmt]) => {
    const cur = k[key] ?? 0, base = b[key] ?? 0, delta = d[key] ?? 0;
    let deltaPct = base!==0 ? (delta/Math.abs(base)*100).toFixed(1)+"%" : "—";
    let cls = delta>0 ? "delta-up" : delta<0 ? "delta-down" : "";
    return `<div class="kpi"><div class="v">${fmt(cur)}</div><div class="l">${label}</div>
            <div class="d ${cls}">Δ ${delta>=0?"+":""}${deltaPct}</div></div>`;
  }).join("");
  document.getElementById("kpis").innerHTML = grid;

  document.getElementById("per-plat").innerHTML = Object.entries(s.per_platform || {}).map(([p,kk]) =>
    `<div class="seg"><span>${p}</span><span>CTR ${(kk.ctr*100).toFixed(2)}% · CVR ${(kk.cvr*100).toFixed(2)}% · ROI ${kk.roi.toFixed(2)}x</span></div>`
  ).join("");
}

function renderCATE(cate) {
  const el = document.getElementById("cate-content");
  const hint = document.getElementById("cate-empty-hint");
  if (cate && hint) hint.style.display = "none";
  if (!cate || !cate.length) {
    el.innerHTML = `<div class="hint">⏳ 暂无 CATE 数据。<br><br>
      <b>触发方式</b>：拖动左侧"抖音 %" / "小红书 %" / "总预算"滑块 → 系统会做反事实模拟 → 自动算 CATE。<br><br>
      <b>什么是 CATE</b>：同一个干预（比如挪预算）对不同人群效果不一样——25 岁女一线 +1.5%，55 岁男下沉 -0.8%。CATE 告诉你"该往哪挪"，比单一平均数有用得多。</div>`;
    return;
  }
  const imp = cate.find(x=>x.importances)?.importances;
  const seg = cate.find(x=>x.top_segments)?.top_segments;
  const allZero = imp && Object.values(imp).every(v => v < 1e-6);
  if (allZero) {
    el.innerHTML = `<div class="hint">⚠️ 当前干预幅度太小，因果森林在所有人群上都得到接近 0 的处理效应——这是<b>诚实的</b>结果，不是 bug。<br><br>
      试试拖滑块到极端值（例如抖音 0% / 小红书 100%）再看。</div>`;
    return;
  }
  let html = "";
  if (imp) {
    html += '<div style="margin:8px 0"><b>特征重要度</b></div>';
    html += Object.entries(imp).sort((a,b)=>b[1]-a[1]).map(([k,v]) => {
      const w = Math.round(v*100);
      return `<div class="seg"><span>${k}</span><span style="flex:1; margin:0 10px; background:#0b0e14; border-radius:3px; height:12px; position:relative"><span style="position:absolute; left:0; top:0; bottom:0; width:${w}%; background:var(--accent); border-radius:3px"></span></span><span>${w}%</span></div>`;
    }).join("");
  }
  if (seg) {
    html += '<div style="margin:12px 0 4px"><b>Top 敏感人群（Δ点击倾向）</b></div>';
    html += seg.map(s => `<div class="seg"><span>${s.segment} <span class="badge">n=${s.n}</span></span><span class="${s.delta>0?'delta-up':'delta-down'}">${s.delta>0?'+':''}${(s.delta*100).toFixed(2)}pp</span></div>`).join("");
  }
  el.innerHTML = html;
}

async function explain() {
  if (!SESSION_ID) return;
  const useLlm = document.getElementById("use_llm").checked;
  const n = +document.getElementById("nsouls").value;
  const r = await fetch(API + `/api/sandbox/session/${SESSION_ID}/explain?n=${n}&use_llm=${useLlm}`, {method:"POST"});
  const j = await r.json();
  renderSouls(j.soul_quotes);
}

async function undo() {
  if (!SESSION_ID) return;
  const r = await fetch(API + "/api/sandbox/session/" + SESSION_ID + "/undo", {method:"POST"});
  const s = await r.json();
  renderSnapshot(s);
  log("已撤销");
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active", t.dataset.tab===name));
  ["kpi","life","chat","front","v1","society","dag","cate","schema"].forEach(n=>{
    const el = document.getElementById("tab-"+n);
    if (el) el.style.display = (n===name) ? "" : "none";
  });
  if (name === "life" && SESSION_ID) refreshLifecycle();
  if (name === "v1") refreshUEB();
  if (name === "chat" && window.LAST_GROUPCHAT) renderGroupChat(window.LAST_GROUPCHAT);
  if (name === "society" && !window.SOCIETY_LOADED) renderSociety(30000);
  if (name === "schema" && window.LAST_SCHEMA) renderSchemaOutputs(window.LAST_SCHEMA);
}

function toggleTabMore(ev) {
  if (ev) ev.stopPropagation();
  const m = document.getElementById("tab-more-menu");
  if (!m) return;
  m.style.display = (m.style.display === "none" || !m.style.display) ? "block" : "none";
  // click-outside 关闭
  if (m.style.display === "block" && !m._bound) {
    m._bound = true;
    setTimeout(() => {
      const close = (e) => {
        if (!m.contains(e.target) && e.target.id !== "tab-more-btn") {
          m.style.display = "none";
          document.removeEventListener("click", close);
          m._bound = false;
        }
      };
      document.addEventListener("click", close);
    }, 0);
  }
}

async function refreshUEB() {
  const [statsR, srcR, gR] = await Promise.all([
    fetch(API+"/api/ueb/stats").then(r=>r.json()),
    fetch(API+"/api/ueb/sources").then(r=>r.json()),
    fetch(API+"/api/graph/inspect").then(r=>r.json()),
  ]);
  const sl = statsR.scaling_law_estimate;
  const errPct = (sl.estimated_generalization_err_upper_bound*100).toFixed(2);
  const accPct = (100 - sl.estimated_generalization_err_upper_bound*100).toFixed(2);
  document.getElementById("ueb-stats").innerHTML = `
    <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
      <div class="kpi"><div class="v">${statsR.n_sources}</div><div class="l">数据源</div></div>
      <div class="kpi"><div class="v">${statsR.total_items.toLocaleString()}</div><div class="l">已索引 items</div></div>
      <div class="kpi"><div class="v" style="color:var(--bad)">${errPct}%</div><div class="l">误差上界</div></div>
      <div class="kpi"><div class="v" style="color:var(--good)">${accPct}%</div><div class="l">理论准确率</div></div>
    </div>
    <div class="hint" style="margin-top:6px;">下次减半需累计到 N=${sl.halving_at_n.toLocaleString()}（当前 ${statsR.total_items.toLocaleString()} → ×4 = ${(statsR.total_items*4).toLocaleString()}）</div>
  `;
  document.getElementById("ueb-sources").innerHTML =
    "<div class='hint'>注册的 16+ 数据源（任意时刻可加新）：</div>" +
    srcR.sources.map(s => `
      <div class="seg">
        <span><span class="badge">${s.modality}</span> <b>${s.source}</b> <span class="hint" style="display:inline">${s.notes}</span></span>
        <span>${s.n_items.toLocaleString()} items · dim ${s.output_dim}</span>
      </div>`).join("");
  // CCG graph
  const nodes = gR.nodes;
  const edges = nodes.flatMap(n => n.deps.map(d => [d, n.name]));
  const allNodes = Array.from(new Set([...nodes.map(n=>n.name), ...edges.flat()]));
  // simple horizontal layout by topological depth
  const depth = {};
  function getD(n) {
    if (depth[n] !== undefined) return depth[n];
    const incoming = edges.filter(([_,t]) => t===n).map(([s,_]) => s);
    return depth[n] = incoming.length === 0 ? 0 : Math.max(...incoming.map(getD)) + 1;
  }
  allNodes.forEach(getD);
  const maxD = Math.max(...Object.values(depth));
  const nodeMap = {};
  const cols = {};
  allNodes.forEach(n => { (cols[depth[n]] = cols[depth[n]] || []).push(n); });
  let svg = `<svg viewBox="0 0 800 280" style="width:100%; background:#0f141c; border-radius:6px;">
    <defs><marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#556"/></marker></defs>`;
  Object.entries(cols).forEach(([d, ns]) => {
    ns.forEach((n, i) => {
      const x = 60 + (+d) * 130;
      const y = 30 + i * 60;
      nodeMap[n] = {x, y};
      const isInput = !nodes.find(nn => nn.name === n);
      svg += `<g transform="translate(${x},${y})">
        <rect width="110" height="34" rx="4" fill="${isInput?'#23304a':'#0f141c'}" stroke="${isInput?'var(--warn)':'var(--accent)'}" stroke-width="1"/>
        <text x="55" y="16" text-anchor="middle" fill="var(--fg)" font-size="11">${n}</text>
        <text x="55" y="28" text-anchor="middle" fill="var(--muted)" font-size="9">${isInput?'(input)':''}</text>
      </g>`;
    });
  });
  edges.forEach(([s,t]) => {
    const a = nodeMap[s], b = nodeMap[t];
    if (!a||!b) return;
    svg += `<path d="M${a.x+110} ${a.y+17} C${a.x+150} ${a.y+17} ${b.x-20} ${b.y+17} ${b.x} ${b.y+17}" stroke="#556" fill="none" marker-end="url(#arr2)"/>`;
  });
  svg += "</svg>";
  document.getElementById("ccg-graph").innerHTML = svg;
  // CCG trace from last predict if available
  if (window.LAST_TRACE) renderTrace(window.LAST_TRACE);
}

function renderTrace(trace) {
  const total = trace.total_ms || 1;
  const html = `<div class="hint">CCG run total <b>${total.toFixed(1)}ms</b> · cache ${trace.cache_stats.hits} hit / ${trace.cache_stats.misses} miss</div>` +
    trace.traces.map(t => {
      const w = Math.min(100, (t.duration_ms / total) * 100);
      const color = t.cache_hit ? "var(--good)" : t.intervened ? "var(--warn)" : "var(--accent)";
      return `<div class="seg">
        <span style="width:130px"><b>${t.name}</b></span>
        <span style="flex:1; margin:0 10px; background:#0b0e14; border-radius:3px; height:14px; position:relative">
          <span style="position:absolute; left:0; top:0; bottom:0; width:${w}%; background:${color}; border-radius:3px"></span>
        </span>
        <span>${t.duration_ms.toFixed(1)}ms ${t.cache_hit?'(cache)':''}</span>
      </div>`;
    }).join("");
  document.getElementById("ccg-trace").innerHTML = html;
}

async function indexMockData() {
  const items = Array.from({length:1000}, (_,i)=>`mock comment #${i} 评论 ${i}`);
  const r = await fetch(API+"/api/ueb/index", {method:"POST",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({source:"comment_text", items})});
  const d = await r.json();
  log("UEB +1000 items: scaling law → " + JSON.stringify(d.scaling_law_now));
  refreshUEB();
}

async function registerCustomSource() {
  const name = prompt("新数据源名（如 weibo_brand_mentions）");
  if (!name) return;
  const modality = prompt("modality (text/categorical/timeseries/geo/event/tabular)", "text");
  const r = await fetch(API+"/api/ueb/register", {method:"POST",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({source:name, modality, notes:"用户动态注册"})});
  log("registered " + name);
  refreshUEB();
}

async function refreshLifecycle() {
  if (!SESSION_ID) return;
  const r = await fetch(API + "/api/sandbox/session/" + SESSION_ID + "/lifecycle?days=14");
  const lc = await r.json();
  drawLifecycle(lc);
}

function drawLifecycle(lc) {
  const c = document.getElementById("life-canvas");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 36;
  ctx.clearRect(0,0,W,H);
  if (!lc || !lc.total_daily) return;
  const days = lc.day_axis, paid = lc.paid_daily, org = lc.organic_daily, tot = lc.total_daily;
  const maxV = Math.max(...tot) || 1;
  const xs = days.map((_,i)=> pad + i * (W - pad*2) / Math.max(days.length-1,1));
  const y = v => H - pad - v/maxV * (H - pad*2);
  // axes
  ctx.strokeStyle = "#222a38"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();
  ctx.fillStyle = "#8b93a7"; ctx.font = "11px monospace";
  for (let i = 0; i < days.length; i++) if (i % 2 === 0)
    ctx.fillText("D"+i, xs[i]-8, H-pad+14);
  for (let i=0;i<=4;i++) {
    const v = maxV * i/4;
    ctx.fillText(v>=1000?(v/1000).toFixed(0)+"k":v.toFixed(0), 4, y(v)+3);
  }
  // stacked fill: organic on top of paid
  function drawArea(data, color, base = null) {
    ctx.beginPath();
    ctx.moveTo(xs[0], base ? y(base[0]) : H-pad);
    for (let i = 0; i < data.length; i++) {
      ctx.lineTo(xs[i], y((base?base[i]:0) + data[i]));
    }
    for (let i = data.length - 1; i >= 0; i--) {
      ctx.lineTo(xs[i], y(base?base[i]:0));
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
  // paid area
  drawArea(paid, "rgba(110,168,254,0.55)");
  // organic stacked on paid
  drawArea(org, "rgba(94,211,155,0.55)", paid);
  // total line
  ctx.strokeStyle = "#e6e8ee"; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i=0;i<tot.length;i++) {
    if (i===0) ctx.moveTo(xs[i], y(tot[i])); else ctx.lineTo(xs[i], y(tot[i]));
  }
  ctx.stroke();
  // legend
  ctx.fillStyle = "rgba(110,168,254,0.7)"; ctx.fillRect(W-140, 10, 10, 10);
  ctx.fillStyle = "#e6e8ee"; ctx.fillText("Paid", W-124, 19);
  ctx.fillStyle = "rgba(94,211,155,0.7)"; ctx.fillRect(W-80, 10, 10, 10);
  ctx.fillStyle = "#e6e8ee"; ctx.fillText("Organic", W-64, 19);
  document.getElementById("life-stats").textContent =
    `peak day: D${lc.peak_day.toFixed(1)} · organic share ${(lc.organic_share*100).toFixed(1)}% · R₀=${lc.branching_ratio}`;
}

async function drawDAG() {
  const r = await fetch(API + "/api/dag");
  const dag = await r.json();
  const svg = document.getElementById("dag-svg");
  // Layered layout: 8 columns (L1..L8), nodes stacked vertically within each
  const layerOrder = ["L1","L2","L3","L4","L5","L6","L7","L8"];
  const NODE_W = 145, NODE_H = 30, GAP_Y = 12, COL_GAP = 25;
  const COL_W = NODE_W + COL_GAP;
  const W = layerOrder.length * COL_W + 30;
  const layerNodes = {};
  layerOrder.forEach(L => layerNodes[L] = dag.nodes.filter(n => n.layer === L));
  const maxRows = Math.max(...layerOrder.map(L => layerNodes[L].length));
  const H = maxRows * (NODE_H + GAP_Y) + 100;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.minHeight = H + "px";
  svg.style.minWidth = W + "px";
  const pos = {};
  layerOrder.forEach((L, ci) => {
    const xs = ci * COL_W + 10;
    layerNodes[L].forEach((n, ri) => {
      const ys = 50 + ri * (NODE_H + GAP_Y);
      pos[n.name] = {x: xs, y: ys, color: n.color};
    });
  });
  let content = "";
  // layer headers
  layerOrder.forEach((L, ci) => {
    const layerInfo = dag.layers.find(x => x.id === L);
    const x = ci * COL_W + 10;
    content += `<text x="${x + NODE_W/2}" y="20" text-anchor="middle" fill="${dag.nodes.find(n=>n.layer===L)?.color || '#888'}" font-size="13" font-weight="bold">${L} · ${layerInfo.label}</text>`;
    content += `<text x="${x + NODE_W/2}" y="32" text-anchor="middle" fill="#666" font-size="10">${layerInfo.nodes.length} 节点</text>`;
  });
  // edges (curved Bézier from right side of source to left side of target)
  for (const [s, t] of dag.edges) {
    const a = pos[s], b = pos[t];
    if (!a || !b) continue;
    const x1 = a.x + NODE_W, y1 = a.y + NODE_H/2;
    const x2 = b.x, y2 = b.y + NODE_H/2;
    const cx = (x1 + x2) / 2;
    const opacity = 0.35;
    content += `<path d="M${x1} ${y1} C${cx} ${y1} ${cx} ${y2} ${x2} ${y2}" stroke="#556" stroke-width="0.8" fill="none" opacity="${opacity}" marker-end="url(#arr)"/>`;
  }
  // nodes — clickable
  for (const n of dag.nodes) {
    const p = pos[n.name];
    if (!p) continue;
    const isInt = n.intervenable;
    const isTV = n.time_varying;
    const stroke = isInt ? "#fff" : p.color;
    const sw = isInt ? 2 : 1;
    const dash = isTV ? "stroke-dasharray='3,2'" : "";
    content += `<g transform="translate(${p.x},${p.y})" data-scm-node="${n.name}" style="cursor:pointer">
      <rect width="${NODE_W}" height="${NODE_H}" rx="3" fill="${p.color}22" stroke="${stroke}" stroke-width="${sw}" ${dash}/>
      <text x="${NODE_W/2}" y="${NODE_H/2+4}" text-anchor="middle" fill="#e6e8ee" font-size="10" style="pointer-events:none">${n.label_zh}${n.computed_by?' ✓':''}</text>
    </g>`;
  }
  // legend
  const legendY = H - 30;
  content += `<g transform="translate(20,${legendY})">
    <text fill="#e6e8ee" font-size="11">${dag.n_nodes} 节点 · ${dag.n_edges} 因果边 · ${dag.stats.intervenable_count} 可干预 · ${dag.stats.time_varying_count} 时变 · ${dag.stats.computed_count} 已实现</text>
  </g>`;
  svg.innerHTML = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#556"/></marker></defs>` + content;
  // attach SCM node handlers
  svg.querySelectorAll("[data-scm-node]").forEach(g => {
    const name = g.getAttribute("data-scm-node");
    const node = dag.nodes.find(n => n.name === name);
    g.addEventListener("click", e => showSCMNode(node, false));
    g.addEventListener("contextmenu", e => {
      e.preventDefault(); e.stopPropagation();
      showSCMNode(node, true);
    });
  });
}

function showSCMNode(node, isRightClick) {
  if (!node) return;
  // Highlight the node
  document.querySelectorAll("#dag-svg [data-scm-node] rect").forEach(r => r.removeAttribute("filter"));
  const g = document.querySelector(`#dag-svg [data-scm-node="${node.name}"] rect`);
  if (g) g.setAttribute("filter", "url(#glow)");
  // ensure glow filter exists
  if (!document.getElementById("glow")) {
    const defs = document.querySelector("#dag-svg defs");
    if (defs) defs.insertAdjacentHTML("beforeend",
      '<filter id="glow"><feGaussianBlur stdDeviation="3"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>');
  }
  // Find or create floating panel
  let panel = document.getElementById("scm-panel");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "scm-panel";
    panel.style.cssText = "position:fixed; top:90px; right:24px; width:340px; background:#141922; border:2px solid var(--accent); border-radius:8px; padding:14px; box-shadow:0 4px 20px rgba(0,0,0,0.6); z-index:9999;";
    document.body.appendChild(panel);
  }
  let html = `<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); padding-bottom:6px; margin-bottom:8px;">
    <b style="color:${node.color}; font-size:14px;">${node.label_zh}</b>
    <button class="secondary" onclick="document.getElementById('scm-panel').remove()" style="font-size:11px; padding:2px 6px;">✖</button>
  </div>
  <div style="font-size:11px; line-height:1.7;">
    <div><span style="color:var(--muted)">name:</span> <code>${node.name}</code></div>
    <div><span style="color:var(--muted)">layer:</span> ${node.layer} · ${node.layer_label}</div>
    <div><span style="color:var(--muted)">category:</span> ${node.category}</div>
    <div><span style="color:var(--muted)">可干预:</span> ${node.intervenable?'<b style="color:var(--accent)">✅ 是</b>':'❌ 否（中间/产出节点）'}</div>
    <div><span style="color:var(--muted)">时变:</span> ${node.time_varying?'✅ 是':'❌ 否'}</div>
    <div><span style="color:var(--muted)">已实现:</span> ${node.computed_by?`<b style="color:var(--good)">✓ ${node.computed_by}</b>`:'<span style="color:var(--warn)">⏳ 占位</span>'}</div>
  </div>`;
  if (isRightClick) {
    if (node.intervenable) {
      // Map SCM node name → PredictRequest field path (for the whitelisted scalar nodes)
      const fieldMap = {
        budget: "total_budget", total_budget: "total_budget",
        sentiment: "sentiment", weather_temp_c: "weather_temp_c",
        daypart: "daypart", rainy: "rainy",
        visual: "creative.visual_style", visual_style: "creative.visual_style",
        music: "creative.music_mood", music_mood: "creative.music_mood",
        duration: "creative.duration_sec", duration_sec: "creative.duration_sec",
        caption: "creative.caption", has_celeb: "creative.has_celeb",
        platform_weight: "platform_alloc", kol_niche: "kol_niche",
        overlap: "cross_platform_overlap",
      };
      const fieldPath = fieldMap[node.name];
      const supported = !!fieldPath;
      html += `<div style="margin-top:10px; padding:8px; border:1px dashed var(--accent); border-radius:4px;">
        <div style="color:var(--accent); font-size:12px;"><b>🔮 do() 反事实干预</b></div>
        <div class="hint">把 <b>${node.label_zh}</b> 强制设为新值，重跑预测、对比基线。${supported?'':'<b style="color:var(--warn)">⚠️ 此节点需结构化参数，当前 UI 只支持顶层标量节点</b>'}</div>
        <input id="scm-int-val" placeholder="数值 / 字符串 / true/false" style="width:100%; margin:4px 0;">
        <button ${supported?'':'disabled'} onclick="doInterveneSCM('${node.name}','${fieldPath || ''}')">🚀 应用并重跑预测</button>
        <div id="scm-int-result" style="margin-top:6px; font-size:11px;"></div>
      </div>`;
    } else {
      html += `<div style="margin-top:10px; padding:8px; background:#0f141c; border-radius:4px; font-size:11px; color:var(--muted)">
        ⛔ 这个节点是 <b>${node.category}</b>（不可直接干预），但你可以干预它的<b>父节点</b>（影响它的因变量）：
      </div>`;
    }
  } else {
    html += `<div class="hint" style="margin-top:8px;">右键查看 do() 干预选项</div>`;
  }
  panel.innerHTML = html;
}

async function doInterveneSCM(nodeName, fieldPath) {
  const raw = document.getElementById('scm-int-val').value.trim();
  if (!raw) { alert('请输入新值'); return; }
  let val = raw;
  if (/^-?\d+(\.\d+)?$/.test(raw)) val = +raw;
  else if (raw === 'true') val = true;
  else if (raw === 'false') val = false;
  const req = buildRequest();
  const parts = fieldPath.split('.');
  let cur = req;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]] || typeof cur[parts[i]] !== 'object') cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length-1]] = val;
  const result = document.getElementById('scm-int-result');
  result.innerHTML = '⏳ 反事实运行中...';
  try {
    const r = await fetch(API + '/api/predict', {
      method: 'POST', headers: {'content-type':'application/json'},
      body: JSON.stringify(req),
    });
    const j = await r.json();
    const base = LAST_BASELINE || {};
    const after = j.kpis || {};
    const rowFmt = (label, key, fmt) => {
      const b = base[key], c = after[key];
      if (b == null || c == null) return '';
      const d = c - b;
      const dPct = b ? (d/b*100).toFixed(1) : '?';
      const col = d > 0 ? 'var(--good)' : d < 0 ? 'var(--bad)' : 'var(--muted)';
      return `<tr><td>${label}</td><td>${fmt(b)}</td><td><b>${fmt(c)}</b></td><td style="color:${col}">${d>=0?'+':''}${fmt(d)} (${dPct}%)</td></tr>`;
    };
    result.innerHTML = `
      <div style="color:var(--accent); margin:6px 0;">✅ do(${nodeName} = ${JSON.stringify(val)}) 反事实结果</div>
      <table style="width:100%; font-size:10px; border-collapse:collapse;">
        <thead><tr style="color:#888;"><th style="text-align:left">指标</th><th>基线</th><th>干预后</th><th>变化</th></tr></thead>
        <tbody>
          ${rowFmt('曝光', 'impressions', v=>Math.round(v).toLocaleString())}
          ${rowFmt('点击', 'clicks', v=>Math.round(v).toLocaleString())}
          ${rowFmt('转化', 'conversions', v=>Math.round(v).toLocaleString())}
          ${rowFmt('ROI', 'roi', v=>v.toFixed(2)+'x')}
          ${rowFmt('CTR', 'ctr', v=>(v*100).toFixed(2)+'%')}
          ${rowFmt('CVR', 'cvr', v=>(v*100).toFixed(2)+'%')}
        </tbody>
      </table>`;
  } catch(e) {
    result.innerHTML = `<span style="color:var(--bad)">❌ ${e.message}</span>`;
  }
}

function resetDemo() {
  SESSION_ID = null; LAST_BASELINE = null;
  document.getElementById("kpis").innerHTML = "";
  document.getElementById("per-plat").innerHTML = "";
  document.getElementById("souls").innerHTML = "";
  document.getElementById("cate-content").innerHTML = "";
  log("已重置");
}
