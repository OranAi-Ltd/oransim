"use strict";

let SOCIETY_DATA = null;
let SOCIETY_MODE = "tier"; // tier | verdict | reach
let GL_CTX = null, GL_PROG = null, GL_BUF_POS = null, GL_BUF_COL = null, GL_BUF_SIZE = null, GL_N = 0;

function initGL() {
  if (GL_CTX) return GL_CTX;
  const cv = document.getElementById("society-canvas-gl");
  if (!cv) return null;
  const gl = cv.getContext("webgl", { antialias: true, premultipliedAlpha: false });
  if (!gl) return null;
  const vsrc = `attribute vec2 a_pos; attribute vec4 a_col; attribute float a_size;
    varying vec4 v_col;
    void main() {
      gl_Position = vec4((a_pos * 2.0 - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);
      gl_PointSize = a_size;
      v_col = a_col;
    }`;
  const fsrc = `precision mediump float; varying vec4 v_col;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      float r2 = dot(d, d);
      if (r2 > 0.25) discard;
      float a = 1.0 - smoothstep(0.18, 0.25, r2);
      gl_FragColor = vec4(v_col.rgb, v_col.a * a);
    }`;
  const mkShader = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); return null; }
    return s;
  };
  const vs = mkShader(gl.VERTEX_SHADER, vsrc);
  const fs = mkShader(gl.FRAGMENT_SHADER, fsrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(prog)); return null; }
  gl.useProgram(prog);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  GL_CTX = gl; GL_PROG = prog;
  GL_BUF_POS  = gl.createBuffer();
  GL_BUF_COL  = gl.createBuffer();
  GL_BUF_SIZE = gl.createBuffer();
  return gl;
}

const TIER_RGB = [
  [0x6e/255, 0xa8/255, 0xfe/255],  // 一线
  [0x5e/255, 0xd3/255, 0x9b/255],  // 新一线
  [0xff/255, 0xc8/255, 0x57/255],  // 二线
  [0xff/255, 0x9a/255, 0x55/255],  // 三四线
  [0xff/255, 0x7a/255, 0x85/255],  // 五线+
];

function uploadGLBuffers() {
  const gl = GL_CTX; if (!gl || !SOCIETY_DATA) return;
  const pts = SOCIETY_DATA.points;
  const N = pts.length;
  GL_N = N;

  const pos = new Float32Array(N * 2);
  const col = new Float32Array(N * 4);
  const size = new Float32Array(N);

  const verdictByPid = {};
  if (SOCIETY_MODE === "verdict" && SOUL_QUOTES_ALL) {
    for (const q of SOUL_QUOTES_ALL) if (q.persona_id != null) verdictByPid[q.persona_id] = q;
  }

  for (let i = 0; i < N; i++) {
    const p = pts[i];
    pos[i*2]   = p.x;
    pos[i*2+1] = p.y;
    let r, g, b, a, sz;
    if (p.is_soul) {
      sz = 7;
      a = 0.95;
      if (SOCIETY_MODE === "verdict") {
        const v = verdictByPid[p.pid];
        if (v) {
          const intent = +v.purchase_intent_7d || 0;
          if (intent > 0.6) { r = 0xa8/255; g = 0x55/255; b = 0xf7/255; sz = 12; }
          else if (v.will_click) { r = 0x22/255; g = 0xc5/255; b = 0x5e/255; sz = 9; }
          else { r = 0xef/255; g = 0x44/255; b = 0x44/255; sz = 9; }
        } else { r = 0.33; g = 0.33; b = 0.36; a = 0.55; sz = 4; }
      } else {
        [r, g, b] = TIER_RGB[p.tier] || [1, 1, 1];
      }
    } else {
      // background population
      if (SOCIETY_MODE === "verdict") { r = 0.16; g = 0.19; b = 0.25; a = 0.55; }
      else { const rgb = TIER_RGB[p.tier] || [0.5,0.5,0.5]; r = rgb[0]; g = rgb[1]; b = rgb[2]; a = 0.40; }
      sz = Math.max(1.2, Math.sqrt(300000 / Math.max(1, N)) * 1.6);  // scale with density
    }
    col[i*4]   = r; col[i*4+1] = g; col[i*4+2] = b; col[i*4+3] = a;
    size[i] = sz;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_POS);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_COL);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_SIZE);
  gl.bufferData(gl.ARRAY_BUFFER, size, gl.DYNAMIC_DRAW);
}

function drawGL() {
  const gl = GL_CTX; if (!gl || !GL_N) return;
  const cv = document.getElementById("society-canvas-gl");
  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.04, 0.055, 0.087, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(GL_PROG);

  const posLoc  = gl.getAttribLocation(GL_PROG, "a_pos");
  const colLoc  = gl.getAttribLocation(GL_PROG, "a_col");
  const sizeLoc = gl.getAttribLocation(GL_PROG, "a_size");

  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_POS);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_COL);
  gl.enableVertexAttribArray(colLoc);
  gl.vertexAttribPointer(colLoc, 4, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, GL_BUF_SIZE);
  gl.enableVertexAttribArray(sizeLoc);
  gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, 0, 0);

  gl.drawArrays(gl.POINTS, 0, GL_N);
}

async function renderSociety(n = 30000) {
  const el = document.getElementById("society-stats");
  el.innerHTML = `⏳ 加载 ${n.toLocaleString()} 粒子中${n>=300000?'（大数据量，可能 10-30s）':''}...`;
  const t0 = performance.now();
  let r, d;
  try {
    r = await fetch(API + "/api/society/sample?n=" + n);
    d = await r.json();
  } catch(e) { el.innerHTML = "❌ 加载失败: " + e.message; return; }
  const dt = Math.round(performance.now() - t0);
  SOCIETY_DATA = d;
  window.SOCIETY_LOADED = true;
  if (SOUL_QUOTES_ALL && SOUL_QUOTES_ALL.length > 0) SOCIETY_MODE = "verdict";

  const gl = initGL();
  if (gl) {
    uploadGLBuffers();
    drawGL();
    drawSocietyOverlay();  // 2D layer on top: title + legend
  } else {
    drawSocietyFrame();  // fallback pure 2D
  }
  updateSocietyStats(dt);
}

function drawSocietyOverlay() {
  // Only title + legend; transparent background so the GL layer shows through.
  const d = SOCIETY_DATA;
  if (!d) return;
  const c = document.getElementById("society-canvas");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const tierColors = ["#6ea8fe", "#5ed39b", "#ffc857", "#ff9a55", "#ff7a85"];
  const verdictByPid = {};
  let clickedCount=0, skippedCount=0, highIntentCount=0;
  if (SOCIETY_MODE === "verdict" && SOUL_QUOTES_ALL) {
    for (const q of SOUL_QUOTES_ALL) if (q.persona_id != null) verdictByPid[q.persona_id] = q;
    for (const p of d.points) {
      if (!p.is_soul) continue;
      const v = verdictByPid[p.pid]; if (!v) continue;
      if ((+v.purchase_intent_7d||0) > 0.6) highIntentCount++;
      else if (v.will_click) clickedCount++;
      else skippedCount++;
    }
  }
  ctx.fillStyle = "rgba(10,14,22,0.7)";
  ctx.fillRect(0, 0, W, 36);
  ctx.fillStyle = "#e6e8ee";
  ctx.font = "bold 22px sans-serif";
  const titleTxt = SOCIETY_MODE === "verdict"
    ? `预测结果 · 人口 ${(d.n_total/10000).toFixed(0)} 万 · AI 投票 ${d.n_llm_souls_highlighted}`
    : `人口分布 · ${(d.n_total/10000).toFixed(0)} 万 · 渲染 ${(d.n_sampled/10000).toFixed(1)} 万`;
  ctx.fillText(titleTxt, 18, 28);

  ctx.font = "16px monospace";
  if (SOCIETY_MODE === "verdict") {
    const items = [
      ["#22c55e", `点了 ${clickedCount}`],
      ["#ef4444", `跳过 ${skippedCount}`],
      ["#a855f7", `高购意 ${highIntentCount}`],
      ["#2a3040", "背景人群"],
    ];
    ctx.fillStyle = "rgba(10,14,22,0.7)";
    ctx.fillRect(W - 260, 46, 240, items.length*22+12);
    items.forEach(([c,t],i) => {
      ctx.fillStyle = c; ctx.fillRect(W - 250, 56 + i*22, 14, 14);
      ctx.fillStyle = "#e6e8ee"; ctx.fillText(t, W - 230, 68 + i*22);
    });
  } else {
    const tierNames = ["一线", "新一线", "二线", "三四线", "五线+"];
    ctx.fillStyle = "rgba(10,14,22,0.7)";
    ctx.fillRect(W - 200, 46, 180, tierNames.length*20+12);
    for (let i = 0; i < 5; i++) {
      ctx.fillStyle = tierColors[i];
      ctx.fillRect(W - 190, 56 + i*20, 14, 14);
      ctx.fillStyle = "#e6e8ee"; ctx.fillText(tierNames[i], W - 170, 68 + i*20);
    }
  }
}

function drawSocietyFrame() {
  // Fallback: pure 2D, used only if WebGL init fails.
  const d = SOCIETY_DATA;
  if (!d) return;
  const c = document.getElementById("society-canvas");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.fillStyle = "#0a0e16";
  ctx.fillRect(0, 0, W, H);

  const tierColors = ["#6ea8fe", "#5ed39b", "#ffc857", "#ff9a55", "#ff7a85"];

  // Build verdict lookup from soul quotes if available
  const verdictByPid = {};
  if (SOCIETY_MODE === "verdict" && SOUL_QUOTES_ALL) {
    for (const q of SOUL_QUOTES_ALL) {
      if (q.persona_id != null) verdictByPid[q.persona_id] = q;
    }
  }

  const souls = [];
  for (const p of d.points) {
    if (p.is_soul) { souls.push(p); continue; }
    const x = p.x * (W - 20) + 10;
    const y = p.y * (H - 20) + 10;
    let col = tierColors[p.tier] || "#888";
    // In verdict mode, non-souls are dim gray (we didn't query them)
    if (SOCIETY_MODE === "verdict") col = "#2a3040";
    ctx.fillStyle = col + "60";
    ctx.fillRect(x, y, 1.2, 1.2);
  }

  // LLM souls — big and colorful
  let clickedCount=0, skippedCount=0, highIntentCount=0;
  for (const p of souls) {
    const x = p.x * (W - 20) + 10;
    const y = p.y * (H - 20) + 10;
    let col = tierColors[p.tier] || "#fff";
    let r = 6;
    if (SOCIETY_MODE === "verdict") {
      const v = verdictByPid[p.pid];
      if (v) {
        const intent = +v.purchase_intent_7d || 0;
        if (v.will_click) { col = "#22c55e"; clickedCount++; }
        else { col = "#ef4444"; skippedCount++; }
        if (intent > 0.6) { col = "#a855f7"; r = 9; highIntentCount++; }
      } else {
        col = "#555"; r = 3;
      }
    }
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, col + "ff");
    grad.addColorStop(0.6, col + "80");
    grad.addColorStop(1, col + "00");
    ctx.fillStyle = grad;
    ctx.fillRect(x - r, y - r, 2*r, 2*r);
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, r/3), 0, 2*Math.PI); ctx.fill();
  }

  // Title
  ctx.fillStyle = "#e6e8ee";
  ctx.font = "bold 13px sans-serif";
  const titleTxt = SOCIETY_MODE === "verdict"
    ? `预测结果可视化 · 人口 ${(d.n_total/10000).toFixed(0)} 万 · AI 用户 ${d.n_llm_souls_highlighted}`
    : `人口分布 · ${(d.n_total/10000).toFixed(0)} 万 · 渲染 ${(d.n_sampled/10000).toFixed(1)} 万`;
  ctx.fillText(titleTxt, 12, 20);

  // Legend
  ctx.font = "11px monospace";
  if (SOCIETY_MODE === "verdict") {
    const items = [
      ["#22c55e", `点了 ${clickedCount}`],
      ["#ef4444", `跳过 ${skippedCount}`],
      ["#a855f7", `高购意 ${highIntentCount}`],
      ["#2a3040", "其余背景人群 (未问)"],
    ];
    items.forEach(([c,t],i) => {
      ctx.fillStyle = c; ctx.fillRect(W - 160, 34 + i*16, 10, 10);
      ctx.fillStyle = "#aaa"; ctx.fillText(t, W - 145, 43 + i*16);
    });
  } else {
    const tierNames = ["一线", "新一线", "二线", "三四线", "五线+"];
    for (let i = 0; i < 5; i++) {
      ctx.fillStyle = tierColors[i];
      ctx.fillRect(W - 120, 34 + i*14, 10, 10);
      ctx.fillStyle = "#aaa"; ctx.fillText(tierNames[i], W - 105, 44 + i*14);
    }
  }
}

function toggleSocietyMode(m) {
  SOCIETY_MODE = m;
  if (!SOCIETY_DATA) return;
  if (GL_CTX) {
    uploadGLBuffers();
    drawGL();
    drawSocietyOverlay();
  } else {
    drawSocietyFrame();
  }
  updateSocietyStats();
  if (m === "verdict") pulseSocietyVerdict();
}

// MiroFish-style pulse: expanding rings over GL layer, rendered on the 2D overlay canvas.
function pulseSocietyVerdict() {
  if (!SOCIETY_DATA) return;
  const c = document.getElementById("society-canvas");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  const verdictByPid = {};
  for (const q of SOUL_QUOTES_ALL||[]) if (q.persona_id != null) verdictByPid[q.persona_id] = q;
  const souls = SOCIETY_DATA.points.filter(p=>p.is_soul && verdictByPid[p.pid]);
  if (!souls.length) return;
  let tick = 0;
  const FRAMES = 90;
  const anim = () => {
    tick++;
    // repaint the 2D overlay (transparent bg) each frame so rings don't accumulate
    drawSocietyOverlay();
    const phase = (tick % 40) / 40;
    for (const p of souls) {
      const x = p.x * W;
      const y = p.y * H;
      const v = verdictByPid[p.pid];
      const col = (+v.purchase_intent_7d > 0.6) ? "#a855f7" : (v.will_click ? "#22c55e" : "#ef4444");
      const radius = 14 + phase * 36;
      const alpha = 0.45 * (1 - phase);
      ctx.strokeStyle = col + Math.floor(alpha*255).toString(16).padStart(2,'0');
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, radius, 0, 2*Math.PI); ctx.stroke();
    }
    if (tick < FRAMES) requestAnimationFrame(anim);
  };
  requestAnimationFrame(anim);
}

// Hover inspector — nearest soul within 12 px gets tooltip
function setupSocietyHover() {
  const c = document.getElementById("society-canvas");
  if (!c || c._hoverBound) return;
  c._hoverBound = true;
  let tip = document.getElementById("society-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "society-tip";
    tip.style.cssText = "position:fixed; display:none; background:#1a2030; border:1px solid #a855f7; color:#e6e8ee; padding:8px 10px; border-radius:5px; font-size:11px; max-width:320px; pointer-events:none; z-index:9999; box-shadow:0 4px 16px rgba(0,0,0,0.5);";
    document.body.appendChild(tip);
  }
  c.addEventListener("mousemove", (ev) => {
    if (!SOCIETY_DATA) return;
    const rect = c.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * (c.width / rect.width);
    const my = (ev.clientY - rect.top) * (c.height / rect.height);
    const W = c.width, H = c.height;
    // hit threshold scales with device pixel resolution (canvas is 1800x1000 but rendered smaller)
    const thr = Math.max(14, W / 60);
    let best = null, bestD2 = thr * thr;
    for (const p of SOCIETY_DATA.points) {
      if (!p.is_soul) continue;
      const x = p.x * W;  // GL uses full 0..1 range directly
      const y = p.y * H;
      const d2 = (x-mx)*(x-mx) + (y-my)*(y-my);
      if (d2 < bestD2) { bestD2 = d2; best = p; }
    }
    if (!best) { tip.style.display = "none"; return; }
    const verdictByPid = {};
    for (const q of SOUL_QUOTES_ALL||[]) if (q.persona_id != null) verdictByPid[q.persona_id] = q;
    const v = verdictByPid[best.pid];
    const tierNames = ["一线","新一线","二线","三四线","五线+"];
    let html = `<div style="color:#a855f7; font-weight:bold; margin-bottom:4px;">AI 用户 #${best.pid}</div>
      <div style="font-size:10px; color:#aaa;">${tierNames[best.tier]||'?'} · ${best.gender===0?'女':'男'} · 年龄段 ${best.age}</div>`;
    if (v) {
      const verd = v.will_click ? '<span style="color:#22c55e">👆 点了</span>' : '<span style="color:#ef4444">✖️ 跳过</span>';
      html += `<hr style="border-color:#333; margin:6px 0;">
        <div>${verd} · 情绪 <b>${v.feel||'?'}</b> · 7日购意 <b>${v.purchase_intent_7d}</b></div>
        <div style="margin-top:4px; color:#ccc;">"${v.reason||''}"</div>
        ${v.comment?`<div style="margin-top:4px; font-style:italic; color:#8ab0e0;">评论："${v.comment}"</div>`:''}
        ${v.persona_oneliner?`<div style="margin-top:6px; font-size:10px; color:#888;">${v.persona_oneliner}</div>`:''}`;
    } else {
      html += `<div style="margin-top:4px; color:#888; font-style:italic;">(未参与本次预测)</div>`;
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 15) + "px";
    tip.style.top = (ev.clientY + 10) + "px";
  });
  c.addEventListener("mouseleave", () => {
    const tip = document.getElementById("society-tip"); if (tip) tip.style.display = "none";
  });
}
// attach once on load
if (document.readyState !== "loading") setupSocietyHover();
else document.addEventListener("DOMContentLoaded", setupSocietyHover);

function updateSocietyStats(loadMs) {
  const el = document.getElementById("society-stats");
  const d = SOCIETY_DATA;
  if (!d) { el.innerHTML = ""; return; }
  const hasPred = SOUL_QUOTES_ALL && SOUL_QUOTES_ALL.length > 0;
  const verdictByPid = {};
  if (hasPred) for (const q of SOUL_QUOTES_ALL) if (q.persona_id != null) verdictByPid[q.persona_id] = q;
  const matched = d.points.filter(p => p.is_soul && verdictByPid[p.pid]).length;
  const btns = `
    <div style="display:flex; gap:6px; margin-top:6px;">
      <button class="secondary" style="font-size:11px; padding:3px 8px; ${SOCIETY_MODE==='tier'?'background:#2a1a3a; border-color:#a855f7':''}" onclick="toggleSocietyMode('tier')">👥 人口分布</button>
      <button class="secondary" style="font-size:11px; padding:3px 8px; ${SOCIETY_MODE==='verdict'?'background:#2a1a3a; border-color:#a855f7':''}" onclick="toggleSocietyMode('verdict')" ${hasPred?'':'disabled title="先跑一次预测才能看结果"'}>🎯 预测结果着色 ${hasPred?'('+SOUL_QUOTES_ALL.length+' AI)':''}</button>
    </div>`;
  const renderer = GL_CTX ? '<span class="badge" style="background:#a855f7">⚡ WebGL</span>' : '<span class="badge">Canvas 2D</span>';
  const loadStr = loadMs!=null ? ` · 加载 ${(loadMs/1000).toFixed(1)}s` : '';
  el.innerHTML = `
    ${renderer} <b style="color:var(--accent)">已渲染 ${d.n_sampled.toLocaleString()} 粒子</b> · 后端真实人口 <b>${d.n_total.toLocaleString()}</b>
    · AI 灵魂 <b>${d.n_llm_souls_highlighted}</b> 个${hasPred?` · 其中 <b style="color:var(--good)">${matched}</b> 个已匹配到预测结果`:''}${loadStr}<br>
    ${hasPred ? '<span class="hint">切换模式 ↓ 看这批 AI 对你广告的投票分布 · 鼠标悬停 AI 点可看投票理由</span>' : '<span class="hint">跑一次预测后点"🎯 预测结果着色"看 AI 用户分别怎么投票</span>'}
    ${btns}
  `;
}

