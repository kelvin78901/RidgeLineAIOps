// Global helpers shared across pages: model picker + markdown render.

window.RG = window.RG || {};

// ----------------- Model picker (sidebar dropdown, persisted) -----------------

(async function initModelPicker() {
  const sel = document.getElementById("model-picker");
  if (!sel) return;
  try {
    const r = await fetch("/api/models");
    const data = await r.json();
    const stored = JSON.parse(localStorage.getItem("rg-model") || "null");
    const initialProvider = stored?.provider || data.active_provider;
    const initialModel = stored?.model || data.active_model;

    sel.innerHTML = "";
    for (const [provider, models] of Object.entries(data.providers)) {
      const og = document.createElement("optgroup");
      og.label = provider;
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = `${provider}::${m.id}`;
        opt.textContent = `${m.label} · ${m.tier}`;
        if (provider === initialProvider && m.id === initialModel) opt.selected = true;
        og.appendChild(opt);
      }
      sel.appendChild(og);
    }
    window.RG.getModelChoice = () => {
      const v = sel.value;
      if (!v) return {provider: data.active_provider, model: data.active_model};
      const [provider, model] = v.split("::");
      return {provider, model};
    };
    sel.addEventListener("change", () => {
      localStorage.setItem("rg-model", JSON.stringify(window.RG.getModelChoice()));
      window.dispatchEvent(new CustomEvent("rg:model-changed", {detail: window.RG.getModelChoice()}));
    });
  } catch (e) {
    console.warn("model picker init failed", e);
  }
})();

// ----------------- Markdown rendering -----------------

window.RG.renderMarkdown = function(text) {
  try {
    if (typeof marked === "undefined") return escapeHtml(text);
    marked.setOptions({breaks: true, gfm: true});
    const raw = marked.parse(text || "");
    return (typeof DOMPurify !== "undefined") ? DOMPurify.sanitize(raw) : raw;
  } catch (e) {
    return escapeHtml(text);
  }
};

function escapeHtml(s) {
  return (s||"").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
window.RG.escapeHtml = escapeHtml;
