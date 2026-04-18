"use strict";

// ─── Hero autoplay section (extracted A2) ─────────────

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
