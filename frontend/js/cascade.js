"use strict";

// ─── T-C1 Opinion Propagation Engine ────────────────────────────────────
// 多源径向波 + 按情感极性染色 + 命中粒子触发级联二级波 + 持续 RAF 循环
// 同一引擎既给「🌊 舆论波」按钮用（定时版），也给 hero 背景用（持续 + 帧跳）。

const CASCADE_HUES = {
  positive:    [0x22, 0xc5, 0x5e],  // 绿 · 点击 / 好感
  highIntent:  [0xa8, 0x55, 0xf7],  // 紫 · 强购意
  negative:    [0xef, 0x44, 0x44],  // 红 · 跳过 / 厌恶
  neutral:     [0x6e, 0xa8, 0xfe],  // 蓝 · 好奇 / 无倾向
};
const CASCADE_KINDS = Object.keys(CASCADE_HUES);

const CASCADE_INSTANCES = new Map();  // canvasId → runtime state

function _cascadePickSeedIdx(pts) {
  // 70% 从 soul agent 抽（高影响力种子），30% 全量抽
  const soulIdx = [];
  for (let i = 0; i < pts.length; i++) if (pts[i].is_soul) soulIdx.push(i);
  if (soulIdx.length > 0 && Math.random() < 0.7)
    return soulIdx[(Math.random() * soulIdx.length) | 0];
  return (Math.random() * pts.length) | 0;
}

function _cascadeSpawnWave(state, seedIdx, kind, amplitude, isSecondary, data) {
  const p = data.points[seedIdx];
  if (!p) return;
  state.waves.push({
    x: p.x, y: p.y, startT: performance.now(),
    kind, amplitude, isSecondary, seedIdx,
  });
}

function _cascadeSpawnPrimary(state, n, data) {
  for (let i = 0; i < n; i++) {
    const k = CASCADE_KINDS[(Math.random() * CASCADE_KINDS.length) | 0];
    _cascadeSpawnWave(state, _cascadePickSeedIdx(data.points), k, 1.0, false, data);
  }
}

function _cascadeFrame(state, now) {
  if (!state.active) return;
  const c = state.canvas;
  if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  const data = state.data;
  const pts = data.points;

  // Low-CPU 帧跳（hero 背景用）
  if (state.lowCpu) {
    state.skipToggle = !state.skipToggle;
    if (state.skipToggle) {
      state.rafId = requestAnimationFrame((t) => _cascadeFrame(state, t));
      return;
    }
  }

  // Persistence-of-vision 轨迹
  ctx.fillStyle = state.trailFill;
  ctx.fillRect(0, 0, W, H);

  // T-C2: drift + sparse connections substrate（放在波之前，让波光在连线之上）
  if (state.drift || state.connect) _cascadeDriftAndConnect(state, now);

  const WAVE_SPEED = state.waveSpeed;      // normalized/s
  const WAVE_THICK = state.waveThick;      // normalized
  const WAVE_TAU   = state.waveTau;        // seconds

  const N = pts.length;
  const glow = state._glowBuf || new Float32Array(N);
  glow.fill(0);
  state._glowBuf = glow;
  const kindIdx = state._kindBuf || new Int8Array(N);
  state._kindBuf = kindIdx;

  const aliveWaves = [];
  for (const w of state.waves) {
    const age = (now - w.startT) / 1000;
    if (age > WAVE_TAU) continue;
    const r = age * WAVE_SPEED;
    const envelope = Math.exp(-age / (WAVE_TAU * 0.5)) * w.amplitude;
    if (envelope < 0.02) continue;
    aliveWaves.push(w);

    // 环带：只扫 |d - r| < WAVE_THICK 的粒子
    const rLo2 = Math.max(0, r - WAVE_THICK);
    const rHi  = r + WAVE_THICK;
    const rHi2 = rHi * rHi;
    const rLoSq = rLo2 * rLo2;
    const kindOrd = CASCADE_KINDS.indexOf(w.kind);

    for (let i = 0; i < N; i++) {
      const p = pts[i];
      const dx = p.x - w.x, dy = p.y - w.y;
      const d2 = dx*dx + dy*dy;
      if (d2 > rHi2 || d2 < rLoSq) continue;
      const d = Math.sqrt(d2);
      const ringDist = Math.abs(d - r);
      if (ringDist > WAVE_THICK) continue;
      const proximity = 1 - ringDist / WAVE_THICK;
      const g = proximity * envelope;
      if (g > glow[i]) {
        glow[i] = g;
        kindIdx[i] = kindOrd;
      }
      // 级联：0.2% 几率晋级为二级源（amplitude 减半）
      if (!w.isSecondary && proximity > 0.92 && state.waves.length < state.maxWaves
          && Math.random() < state.cascadeProb) {
        _cascadeSpawnWave(state, i, w.kind, 0.55, true, data);
      }
    }
  }
  state.waves = aliveWaves;

  // 渲染发光粒子
  for (let i = 0; i < N; i++) {
    const g = glow[i];
    if (g < 0.04) continue;
    const p = pts[i];
    const x = p.x * (W - 20) + 10;
    const y = p.y * (H - 20) + 10;
    const rgb = CASCADE_HUES[CASCADE_KINDS[kindIdx[i]]] || CASCADE_HUES.neutral;
    const rad = 1.6 + g * 4.2;
    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${Math.min(1, g*1.25)})`;
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fill();
  }

  // 按节奏补发主波
  if (now - state.lastPrimaryAt > state.primaryIntervalMs) {
    _cascadeSpawnPrimary(state, state.primaryBurst, data);
    state.lastPrimaryAt = now;
  }

  state.rafId = requestAnimationFrame((t) => _cascadeFrame(state, t));
}

/**
 * 启动意见传播动画。
 * @param {HTMLCanvasElement} canvas - 目标 2D canvas（如 society-canvas 或 hero-canvas）
 * @param {Object} data - { points: [{x,y,is_soul,...}] }，归一化 [0,1]² 坐标
 * @param {Object} opts - {
 *   initialWaves=5, primaryBurst=3, primaryIntervalMs=2000,
 *   waveSpeed=0.14, waveThick=0.03, waveTau=2.8, maxWaves=40,
 *   cascadeProb=0.002, trailFill='rgba(10,14,22,0.22)', lowCpu=false
 * }
 * @returns {string} instance key (canvas id) 用于 stop
 */
// ─── T-C2: drift + sparse connections ────────────────────────────────
// 每帧：粒子按 vx/vy 漂移（边界 wrap）+ 用 spatial grid 找邻居画稀疏连线。
// 复杂度 O(N · k)，k = 邻格平均粒子数。4500 粒子 threshold=0.045 ≈ 400k 检查/帧。
function _cascadeDriftAndConnect(state, now) {
  const data = state.data;
  const pts = data.points;
  const N = pts.length;
  const ctx = state.canvas.getContext("2d");
  const W = state.canvas.width, H = state.canvas.height;

  // dt 计算（首帧 dt=0）
  const lastT = state._lastFrameT || now;
  const dt = Math.min(0.05, (now - lastT) / 1000);
  state._lastFrameT = now;

  // Drift：更新位置
  if (state.drift) {
    const speedScale = state.driftSpeed || 0.012;
    for (let i = 0; i < N; i++) {
      const p = pts[i];
      if (p.vx === undefined) {
        p.vx = (Math.random() - 0.5) * speedScale;
        p.vy = (Math.random() - 0.5) * speedScale;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      // Wrap around
      if (p.x < 0) p.x += 1; else if (p.x > 1) p.x -= 1;
      if (p.y < 0) p.y += 1; else if (p.y > 1) p.y -= 1;
    }
  }

  // Connections：spatial grid 找邻居画线
  if (state.connect) {
    const thr   = state.connect.threshold ?? 0.045;
    const aMax  = state.connect.alphaMax ?? 0.22;
    const lineRGB = state.connect.rgb || "140,170,220";
    const gSize = Math.max(2, Math.ceil(1 / thr));
    const grid = new Array(gSize * gSize);
    for (let i = 0; i < grid.length; i++) grid[i] = null;

    for (let i = 0; i < N; i++) {
      const p = pts[i];
      const gx = Math.min(gSize - 1, Math.max(0, Math.floor(p.x * gSize)));
      const gy = Math.min(gSize - 1, Math.max(0, Math.floor(p.y * gSize)));
      const cell = gy * gSize + gx;
      if (grid[cell] === null) grid[cell] = [i];
      else grid[cell].push(i);
    }

    const t2 = thr * thr;
    ctx.lineWidth = 0.5;

    // 遍历每格 × 右下邻格（避免双重计数）
    for (let gy = 0; gy < gSize; gy++) {
      for (let gx = 0; gx < gSize; gx++) {
        const cellPts = grid[gy * gSize + gx];
        if (!cellPts) continue;
        // 本格内两两 + 右 + 下 + 右下 + 左下
        const neighborOffsets = [[0,0],[1,0],[0,1],[1,1],[-1,1]];
        for (const [dgx, dgy] of neighborOffsets) {
          const ngx = gx + dgx, ngy = gy + dgy;
          if (ngx < 0 || ngx >= gSize || ngy < 0 || ngy >= gSize) continue;
          const nbrPts = (dgx === 0 && dgy === 0) ? cellPts : grid[ngy * gSize + ngx];
          if (!nbrPts) continue;
          for (let aIdx = 0; aIdx < cellPts.length; aIdx++) {
            const a = cellPts[aIdx];
            const pa = pts[a];
            const startJ = (dgx === 0 && dgy === 0) ? aIdx + 1 : 0;
            for (let bIdx = startJ; bIdx < nbrPts.length; bIdx++) {
              const b = nbrPts[bIdx];
              if (b === a) continue;
              const pb = pts[b];
              const ddx = pa.x - pb.x, ddy = pa.y - pb.y;
              const d2 = ddx*ddx + ddy*ddy;
              if (d2 > t2) continue;
              const alpha = aMax * (1 - Math.sqrt(d2) / thr);
              if (alpha < 0.02) continue;
              ctx.strokeStyle = `rgba(${lineRGB},${alpha.toFixed(3)})`;
              ctx.beginPath();
              ctx.moveTo(pa.x * (W - 20) + 10, pa.y * (H - 20) + 10);
              ctx.lineTo(pb.x * (W - 20) + 10, pb.y * (H - 20) + 10);
              ctx.stroke();
            }
          }
        }
      }
    }
  }
}

function runOpinionCascade(canvas, data, opts) {
  opts = opts || {};
  if (!canvas || !data || !data.points) return null;
  const key = canvas.id || "_cascade_default";
  stopOpinionCascade(key);
  const state = {
    active: true, rafId: null, skipToggle: false,
    canvas, data,
    waves: [],
    lastPrimaryAt: performance.now() - 9999,
    waveSpeed:         opts.waveSpeed         ?? 0.14,
    waveThick:         opts.waveThick         ?? 0.03,
    waveTau:           opts.waveTau           ?? 2.8,
    maxWaves:          opts.maxWaves          ?? 40,
    primaryBurst:      opts.primaryBurst      ?? 3,
    primaryIntervalMs: opts.primaryIntervalMs ?? 2000,
    cascadeProb:       opts.cascadeProb       ?? 0.002,
    trailFill:         opts.trailFill         || 'rgba(10,14,22,0.22)',
    lowCpu:            !!opts.lowCpu,
    drift:             !!opts.drift,
    driftSpeed:        opts.driftSpeed        ?? 0.012,
    connect:           opts.connect           || null,
  };
  CASCADE_INSTANCES.set(key, state);
  _cascadeSpawnPrimary(state, opts.initialWaves ?? 5, data);
  state.rafId = requestAnimationFrame((t) => _cascadeFrame(state, t));
  return key;
}

function stopOpinionCascade(key) {
  const k = key || "_cascade_default";
  const state = CASCADE_INSTANCES.get(k);
  if (!state) return;
  state.active = false;
  if (state.rafId) cancelAnimationFrame(state.rafId);
  CASCADE_INSTANCES.delete(k);
}

// ─── 「🌊 舆论波」按钮：12 秒三幕剧叙事版 ─────────────────────────────
// KOL 种子发声 → 一级扩散 → 级联触发，每一幕都说清楚在做什么 + 实时
// 触达计数。粒子同时 drift + sparse connect，观感更"活"。
function animateSociety() {
  if (!SOCIETY_DATA) { renderSociety(30000); return; }
  const c = document.getElementById("society-canvas");
  const infected = new Set();
  const t0 = performance.now();

  const key = runOpinionCascade(c, SOCIETY_DATA, {
    initialWaves: 5, primaryBurst: 2, primaryIntervalMs: 2200,
    waveSpeed: 0.16, waveThick: 0.04, waveTau: 3.5,
    cascadeProb: 0.005, maxWaves: 60,
    trailFill: 'rgba(10,14,22,0.14)',
    drift: true, driftSpeed: 0.008,
    connect: { threshold: 0.048, alphaMax: 0.15, rgb: "110,150,200" },
  });

  const state = CASCADE_INSTANCES.get(key);
  const statsEl = document.getElementById("society-stats");
  const phases = [
    { t: 0,    text: "🌊 <b>Act I</b> · 5 个 KOL 种子发声（四种立场按色）· <span style='color:#22c55e'>绿</span>=好感 <span style='color:#a855f7'>紫</span>=强购意 <span style='color:#ef4444'>红</span>=厌恶 <span style='color:#6ea8fe'>蓝</span>=观望" },
    { t: 2500, text: "⚡ <b>Act II</b> · 一级扩散 · 波前碰到粉丝就点亮立场色 · 粒子漂移模拟 persona 在 interest 空间流动" },
    { t: 5500, text: "🔥 <b>Act III</b> · 级联触发 · 被感染的粉丝以 0.5% 概率晋升为二级波源（黑产式二次传播）" },
    { t: 9000, text: "🎯 <b>收束</b> · 观察四色倾向在星图里的地缘分布，以及谁被级联到、谁一直没被说服" },
  ];

  const heartbeat = setInterval(() => {
    if (state?._glowBuf) {
      for (let i = 0; i < state._glowBuf.length; i++) {
        if (state._glowBuf[i] > 0.3) infected.add(i);
      }
    }
    const elapsed = performance.now() - t0;
    for (let i = phases.length - 1; i >= 0; i--) {
      if (elapsed >= phases[i].t) {
        const total = SOCIETY_DATA.points.length;
        const pct = (infected.size / total * 100).toFixed(1);
        statsEl.innerHTML = phases[i].text + `<br><span style="color:var(--muted); font-size:11px">已触达 ${infected.size.toLocaleString()} / ${total.toLocaleString()} 粒子（${pct}%） · 活跃波源 ${state?.waves?.length ?? 0}</span>`;
        break;
      }
    }
  }, 200);

  setTimeout(() => {
    clearInterval(heartbeat);
    stopOpinionCascade(key);
    const total = SOCIETY_DATA.points.length;
    const pct = (infected.size / total * 100).toFixed(1);
    statsEl.innerHTML = `✅ <b>12 秒传播结束</b>：${infected.size.toLocaleString()} / ${total.toLocaleString()} 粒子被触达（${pct}%）· 级联贡献约 ${Math.max(0, infected.size - 5 * 400)} 个额外触达。<br><span class="hint">这就是为什么预算集中投少数 KOL，触达面远超粉丝数之和——网络效应。</span>`;
  }, 12000);
}

