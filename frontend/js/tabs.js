"use strict";

// ─── Tab switching + "更多 ›" dropdown (A3) ────────────

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
