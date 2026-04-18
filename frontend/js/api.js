"use strict";

// ── Core globals + API wrapper + health + mock banner ───────────
const API = `${location.protocol}//${location.hostname}:8001`;

function log(msg) {
  const el = document.getElementById("log");
  const now = new Date().toLocaleTimeString();
  el.innerHTML = `[${now}] ${msg}<br>` + el.innerHTML;
}

async function healthcheck() {
  try {
    const r = await fetch(API + "/api/health");
    const j = await r.json();
    document.getElementById("api-status").textContent = `OK · ${j.population} agents · ${j.souls} souls`;
    // 检查 LLM 模式 · mock / 无 key 时弹配置 banner
    const llm = j.llm || {};
    if (llm.mode !== "api" || !llm.api_key_set) showMockBanner(llm);
    else hideMockBanner();
  } catch (e) { document.getElementById("api-status").textContent = "❌ 连不上"; }
}
healthcheck();

function showMockBanner(llm) {
  if (sessionStorage.getItem("mockBannerDismissed") === "1") return;
  if (document.getElementById("mock-banner")) return;
  const bar = document.createElement("div");
  bar.id = "mock-banner";
  bar.style.cssText = `
    position:relative; margin:0 18px; margin-top:12px;
    padding:12px 44px 12px 16px;
    background:linear-gradient(90deg, rgba(245,181,66,.14), rgba(245,181,66,.06));
    border:1px solid rgba(245,181,66,.45); border-radius:10px;
    color:#ffd68a; font-size:13px; line-height:1.6;
    display:flex; gap:10px; align-items:flex-start;
  `;
  bar.innerHTML = `
    <span style="font-size:16px; line-height:1.3">⚠️</span>
    <div style="flex:1">
      <b style="color:#ffd68a">当前是 mock 模式</b>（LLM 预测走模板，soul agent 不会读你的文案）<br>
      <span style="color:#e7c48a; font-size:12px">切真 LLM：后端启动时加上 env 变量——</span>
      <code style="display:inline-block; margin-top:4px; padding:4px 8px; background:rgba(0,0,0,.35); border-radius:4px; font-size:11.5px; color:#ffe9b3; font-family:Menlo, monospace">
        LLM_MODE=api LLM_API_KEY=&lt;你的key&gt; LLM_MODEL=gpt-5.4
      </code>
      <span style="color:#e7c48a; font-size:12px"> ·
        provider 可选 <code style="background:rgba(0,0,0,.3); padding:1px 4px; border-radius:3px">LLM_PROVIDER=openai/anthropic/gemini/qwen</code>
        (默认 openai-compat)
      </span>
    </div>
    <span onclick="dismissMockBanner()" style="position:absolute; top:8px; right:12px; cursor:pointer; color:#d4a54a; font-size:16px; user-select:none" title="本次会话不再提示">✕</span>
  `;
  const anchor = document.querySelector("main");
  if (anchor) anchor.parentNode.insertBefore(bar, anchor);
  else document.body.insertBefore(bar, document.body.firstChild);
}
function hideMockBanner() {
  const b = document.getElementById("mock-banner");
  if (b) b.remove();
}
function dismissMockBanner() {
  sessionStorage.setItem("mockBannerDismissed", "1");
  hideMockBanner();
}
