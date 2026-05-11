// AlphaGenome × Aging Variant Study — webapp logic
// Backend: FastAPI at same origin (see src/inference/api.py)

const API_BASE = "";  // same origin

// ---------- helpers ----------

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "onclick") e.addEventListener("click", v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function fmt(n, d = 3) {
  if (n == null) return "–";
  return typeof n === "number" ? n.toFixed(d) : String(n);
}

// Diverging color for heatmap cells (blue for negative, red for positive)
function heatColor(v) {
  if (v == null || isNaN(v)) return "#0b1220";
  const mag = Math.min(1, Math.abs(v) / 3);
  if (v > 0) return `rgba(239, 68, 68, ${mag.toFixed(2)})`;
  return `rgba(59, 130, 246, ${mag.toFixed(2)})`;
}

// ---------- gallery ----------

async function loadGallery() {
  const grid = document.getElementById("gallery-grid");
  try {
    const r = await fetch(`${API_BASE}/api/gallery`);
    if (!r.ok) throw new Error("fetch failed");
    const data = await r.json();
    const variants = data.variants || [];
    grid.innerHTML = "";
    if (!variants.length) {
      grid.appendChild(el("div", { class: "gallery-loading" },
        data.note || "Gallery empty"));
      return;
    }
    for (const v of variants) {
      grid.appendChild(el("div", {
        class: "gallery-card",
        onclick: () => showPrediction(v.rsid),
      },
        el("div", { class: "rsid" }, v.rsid),
        el("div", { class: "trait" }, v.trait_category || "—"),
        el("div", { class: "gene" }, v.gene || v.nearest_gene || ""),
        el("div", { class: "prob-value" }, `P(aging)=${fmt(v.p_aging)}`),
      ));
    }
  } catch (e) {
    grid.innerHTML = "";
    grid.appendChild(el("div", { class: "gallery-loading" },
      "Gallery unavailable (backend may not be running)"));
  }
}

// ---------- predict ----------

async function showPrediction(rsid, targetId = "predict-result") {
  const target = document.getElementById(targetId);
  target.innerHTML = "";
  target.appendChild(el("div", { class: "loading" },
    `Scoring ${rsid} … (first call may take 30-60s via AlphaGenome)`));

  try {
    const r = await fetch(`${API_BASE}/api/predict/${rsid}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    renderPrediction(data, target);
  } catch (e) {
    target.innerHTML = "";
    target.appendChild(el("div", { class: "error" }, `Error: ${e.message}`));
  }
}

function renderPrediction(data, target) {
  target.innerHTML = "";
  const box = el("div", { class: "prediction-box" });

  // Variant info
  const v = data.variant || {};
  box.appendChild(el("h3", {},
    `${v.rsid || "?"} — ${v.chromosome || "?"}:${v.position || "?"} `
    + `${v.ref_allele || ""}>${v.alt_allele || ""}`));

  // Binary prediction + probabilities
  if (data.binary_probabilities) {
    box.appendChild(el("p", {},
      "Prediction: ", el("strong", {}, data.binary_prediction)));
    for (const [label, p] of Object.entries(data.binary_probabilities)) {
      const row = el("div", { class: "prob-row" },
        el("div", { class: "prob-label" }, label),
        el("div", { class: "prob-bar" },
          el("div", { class: "prob-fill" })),
        el("div", { class: "prob-value" }, fmt(p)));
      row.querySelector(".prob-fill").style.width = `${(p * 100).toFixed(0)}%`;
      box.appendChild(row);
    }
  }

  // Trait (multiclass) prediction
  if (data.trait_probabilities) {
    box.appendChild(el("h4", {}, "Trait category (if aging)"));
    box.appendChild(el("p", {},
      "Top prediction: ", el("strong", {}, data.trait_prediction)));
    const sorted = Object.entries(data.trait_probabilities)
      .sort((a, b) => b[1] - a[1]).slice(0, 6);
    for (const [label, p] of sorted) {
      const row = el("div", { class: "prob-row" },
        el("div", { class: "prob-label" }, label),
        el("div", { class: "prob-bar" },
          el("div", { class: "prob-fill" })),
        el("div", { class: "prob-value" }, fmt(p)));
      row.querySelector(".prob-fill").style.width = `${(p * 100).toFixed(0)}%`;
      box.appendChild(row);
    }
  }

  // Tissue × output heatmap
  if (data.tissue_heatmap) {
    box.appendChild(el("h4", {}, "AlphaGenome regulatory signature"));
    box.appendChild(el("p", { class: "prob-value" },
      "Cells show max |score| per (tissue, output type). "
      + "Red = positive (increase), blue = negative (decrease)."));
    const hm = el("div", { class: "heatmap" });
    const table = el("table");
    const outputTypes = Object.keys(data.tissue_heatmap);
    const tissues = outputTypes.length ? Object.keys(
      data.tissue_heatmap[outputTypes[0]]) : [];

    const head = el("tr", {}, el("th", {}, ""));
    for (const t of tissues) head.appendChild(el("th", {}, t));
    table.appendChild(head);

    for (const ot of outputTypes) {
      const row = el("tr", {}, el("td", { class: "row-label" }, ot));
      for (const t of tissues) {
        const v = data.tissue_heatmap[ot][t];
        const cell = el("td", {}, v == null ? "–" : fmt(v, 2));
        cell.style.background = heatColor(v);
        row.appendChild(cell);
      }
      table.appendChild(row);
    }
    hm.appendChild(table);
    box.appendChild(hm);
  }

  // Top features
  if (data.top_features && data.top_features.length) {
    box.appendChild(el("h4", {}, "Top contributing features"));
    const tbl = el("table", { class: "feat-table" });
    tbl.appendChild(el("tr", {},
      el("th", {}, "Feature"),
      el("th", {}, "Value"),
      el("th", {}, "Contribution")));
    for (const f of data.top_features.slice(0, 10)) {
      tbl.appendChild(el("tr", {},
        el("td", {}, f.feature),
        el("td", {}, fmt(f.value_raw, 2)),
        el("td", {}, fmt(f.contribution_score, 4))));
    }
    box.appendChild(tbl);
  }

  target.appendChild(box);
}

// ---------- compare ----------

async function compareVariants(a, b) {
  const target = document.getElementById("compare-result");
  target.innerHTML = "";
  target.appendChild(el("div", { class: "loading" },
    `Scoring ${a} and ${b} …`));

  try {
    const r = await fetch(`${API_BASE}/api/compare?a=${a}&b=${b}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    target.innerHTML = "";

    // A and B side-by-side
    const grid = el("div", { class: "gallery-grid" });
    const boxA = el("div"); renderPrediction(data.a, boxA);
    const boxB = el("div"); renderPrediction(data.b, boxB);
    grid.appendChild(boxA); grid.appendChild(boxB);
    target.appendChild(grid);

    // Delta heatmap
    if (data.delta) {
      const deltaBox = el("div", { class: "prediction-box" });
      deltaBox.appendChild(el("h3", {}, `Δ (${a} − ${b})`));
      const hm = el("div", { class: "heatmap" });
      const table = el("table");
      const outputTypes = Object.keys(data.delta);
      const tissues = outputTypes.length ? Object.keys(
        data.delta[outputTypes[0]]) : [];
      const head = el("tr", {}, el("th", {}, ""));
      for (const t of tissues) head.appendChild(el("th", {}, t));
      table.appendChild(head);
      for (const ot of outputTypes) {
        const row = el("tr", {}, el("td", { class: "row-label" }, ot));
        for (const t of tissues) {
          const v = data.delta[ot][t];
          const cell = el("td", {}, v == null ? "–" : fmt(v, 2));
          cell.style.background = heatColor(v);
          row.appendChild(cell);
        }
        table.appendChild(row);
      }
      hm.appendChild(table);
      deltaBox.appendChild(hm);
      target.appendChild(deltaBox);
    }
  } catch (e) {
    target.innerHTML = "";
    target.appendChild(el("div", { class: "error" }, `Error: ${e.message}`));
  }
}

// ---------- boot ----------

document.addEventListener("DOMContentLoaded", () => {
  loadGallery();
  document.getElementById("predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = document.getElementById("rsid-input").value.trim();
    if (v) showPrediction(v);
  });
  document.getElementById("compare-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const a = document.getElementById("var-a").value.trim();
    const b = document.getElementById("var-b").value.trim();
    if (a && b) compareVariants(a, b);
  });
});
