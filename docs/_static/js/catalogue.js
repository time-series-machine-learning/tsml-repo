/* Interactive TSML Dataset Catalogue client controller */
(function () {
  "use strict";

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let val = bytes;
    let idx = 0;
    while (val >= 1024 && idx < units.length - 1) {
      val /= 1024;
      idx += 1;
    }
    return idx === 0 ? `${Math.round(val)} B` : `${val.toFixed(1)} ${units[idx]}`;
  }

  function formatInt(val) {
    return val === null || val === undefined ? "—" : Number(val).toLocaleString("en-US");
  }

  function formatLength(row) {
    if (row.equal_length) {
      return formatInt(row.min_length);
    }
    return `${formatInt(row.min_length)}–${formatInt(row.max_length)}`;
  }

  function parseOptionalNumber(text) {
    if (!text || !String(text).trim()) return null;
    const num = Number(text);
    return Number.isFinite(num) ? num : null;
  }

  function triggerDownload(filename, mimeType, content) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function toCsv(rows) {
    const headers = [
      "dataset_name",
      "record_name",
      "variant",
      "task",
      "train_cases",
      "test_cases",
      "total_cases",
      "n_channels",
      "n_classes",
      "min_length",
      "max_length",
      "mean_length",
      "equal_length",
      "unequal_case_lengths",
      "unequal_channel_lengths",
      "has_missing",
      "total_bytes",
      "record_id",
      "record_url",
    ];
    const escapeCell = (val) => {
      if (val === null || val === undefined) return "";
      const str = String(val);
      return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    const lines = [headers.join(",")];
    for (const row of rows) {
      lines.push(headers.map((h) => escapeCell(row[h])).join(","));
    }
    return lines.join("\n") + "\n";
  }

  function initCatalogue(root) {
    const src = root.getAttribute("data-catalogue-src") || "_static/data/catalogue.json";
    fetch(src)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
      })
      .then((payload) => mountCatalogue(root, payload))
      .catch((err) => {
        root.innerHTML = `<div class="tsml-empty">Could not load dataset catalogue (${String(err)}).</div>`;
      });
  }

  function mountCatalogue(root, payload) {
    const allDatasets = payload.datasets || [];
    const benchmarks = payload.benchmarks || [];
    const collections = payload.collections || [];

    // Count available variants per base record so base view can show +eq/+nmv badges
    const variantsByRecord = new Map();
    for (const row of allDatasets) {
      if (!variantsByRecord.has(row.record_name)) {
        variantsByRecord.set(row.record_name, []);
      }
      if (row.variant !== "base") {
        variantsByRecord.get(row.record_name).push(row.variant);
      }
    }

    const validSorts = new Set([
      "dataset_name",
      "task",
      "train_cases",
      "test_cases",
      "total_cases",
      "n_channels",
      "n_classes",
      "max_length",
      "total_bytes",
    ]);

    const params = new URLSearchParams(window.location.search);
    const initialBenchmark = params.get("benchmark") || "";
    const rawSort = params.get("sort") || "dataset_name";
    const state = {
      q: params.get("q") || "",
      variant: params.get("variant") || (initialBenchmark ? "all" : "base"),
      task: params.get("task") || "all",
      dim: params.get("dim") || "all",
      length: params.get("length") || "all",
      missing: params.get("missing") || "all",
      benchmark: initialBenchmark,
      collection: params.get("collection") || "",
      minCases: params.get("min_cases") || "",
      maxCases: params.get("max_cases") || "",
      minCh: params.get("min_ch") || "",
      maxCh: params.get("max_ch") || "",
      minLen: params.get("min_len") || "",
      maxLen: params.get("max_len") || "",
      sort: validSorts.has(rawSort) ? rawSort : "dataset_name",
      dir: params.get("dir") || "asc",
    };

    const groupedBenchmarks = new Map();
    for (const b of benchmarks) {
      const group = b.category_label || b.category || "Benchmarks";
      if (!groupedBenchmarks.has(group)) groupedBenchmarks.set(group, []);
      groupedBenchmarks.get(group).push(b);
    }

    let benchmarkOptionsHtml = `<option value="">All datasets</option>`;
    for (const [groupLabel, items] of groupedBenchmarks.entries()) {
      benchmarkOptionsHtml += `<optgroup label="${groupLabel}">`;
      for (const b of items) {
        benchmarkOptionsHtml += `<option value="${b.id}">${b.id} (${b.count})</option>`;
      }
      benchmarkOptionsHtml += `</optgroup>`;
    }

    let collectionOptionsHtml = `<option value="">All collections</option>`;
    for (const c of collections) {
      collectionOptionsHtml += `<option value="${c.id}">${c.label} (${c.count})</option>`;
    }

    const taskMeta = {
      classification: { label: "Classification", short: "Cls", cls: "tsml-pill-cls" },
      regression: { label: "Extrinsic regression", short: "Reg", cls: "tsml-pill-reg" },
      clustering: { label: "Clustering", short: "Clu", cls: "" },
      forecasting: { label: "Forecasting", short: "Fct", cls: "" },
    };
    const seenTasks = Array.from(new Set(allDatasets.map((r) => r.task)));
    let taskOptionsHtml = `<option value="all">All tasks</option>`;
    for (const t of seenTasks) {
      const label = taskMeta[t]
        ? taskMeta[t].label
        : t.charAt(0).toUpperCase() + t.slice(1);
      taskOptionsHtml += `<option value="${t}">${label}</option>`;
    }

    const hasInitialRanges = Boolean(
      state.minCases || state.maxCases || state.minCh || state.maxCh || state.minLen || state.maxLen
    );

    root.innerHTML = `
      <div class="tsml-controls-panel">
        <div class="tsml-controls-grid">
          <div class="tsml-field">
            <label for="tsml-q">Search dataset or Zenodo ID</label>
            <input id="tsml-q" type="search" placeholder="e.g. GunPoint, PPGDalia, 24354112…" />
          </div>
          <div class="tsml-field">
            <label for="tsml-benchmark">Benchmark suite</label>
            <select id="tsml-benchmark">${benchmarkOptionsHtml}</select>
          </div>
          <div class="tsml-field">
            <label for="tsml-collection">Domain collection</label>
            <select id="tsml-collection">${collectionOptionsHtml}</select>
          </div>
          <div class="tsml-field">
            <label for="tsml-variant">Variant scope</label>
            <select id="tsml-variant">
              <option value="base">Base records (${payload.summary.dataset_record_count})</option>
              <option value="all">All variants (${payload.summary.dataset_variant_count})</option>
              <option value="variants">Preprocessed only (_eq / _nmv)</option>
            </select>
          </div>
          <div class="tsml-field">
            <label for="tsml-task">Task</label>
            <select id="tsml-task">${taskOptionsHtml}</select>
          </div>
          <div class="tsml-field">
            <label for="tsml-dim">Channels</label>
            <select id="tsml-dim">
              <option value="all">Univariate &amp; multivariate</option>
              <option value="univariate">Univariate (1 channel)</option>
              <option value="multivariate">Multivariate (&gt;1 channels)</option>
            </select>
          </div>
          <div class="tsml-field">
            <label for="tsml-length">Series length</label>
            <select id="tsml-length">
              <option value="all">Any length</option>
              <option value="equal">Equal length</option>
              <option value="unequal">Unequal length</option>
              <option value="unequal_case">Unequal across cases</option>
              <option value="unequal_channel">Unequal across channels</option>
            </select>
          </div>
          <div class="tsml-field">
            <label for="tsml-missing">Missing values</label>
            <select id="tsml-missing">
              <option value="all">Any</option>
              <option value="none">No missing values</option>
              <option value="has">Has missing values</option>
            </select>
          </div>
        </div>

        <details class="tsml-range-details" ${hasInitialRanges ? "open" : ""}>
          <summary>Numeric range filters (cases, channels, series length)</summary>
          <div class="tsml-range-grid">
            <div class="tsml-field">
              <label>Total cases (min / max)</label>
              <div class="tsml-range-pair">
                <input id="tsml-min-cases" type="number" min="1" placeholder="Min" />
                <input id="tsml-max-cases" type="number" min="1" placeholder="Max" />
              </div>
            </div>
            <div class="tsml-field">
              <label>Channels (min / max)</label>
              <div class="tsml-range-pair">
                <input id="tsml-min-ch" type="number" min="1" placeholder="Min" />
                <input id="tsml-max-ch" type="number" min="1" placeholder="Max" />
              </div>
            </div>
            <div class="tsml-field">
              <label>Series length (min / max)</label>
              <div class="tsml-range-pair">
                <input id="tsml-min-len" type="number" min="1" placeholder="Min" />
                <input id="tsml-max-len" type="number" min="1" placeholder="Max" />
              </div>
            </div>
          </div>
        </details>
      </div>

      <div class="tsml-toolbar">
        <div id="tsml-count" class="tsml-result-count" aria-live="polite"></div>
        <div class="tsml-toolbar-actions">
          <button type="button" id="tsml-reset" class="tsml-btn">Reset filters</button>
          <button type="button" id="tsml-export-csv" class="tsml-btn">Export CSV</button>
          <button type="button" id="tsml-export-json" class="tsml-btn">Export JSON</button>
        </div>
      </div>

      <div class="tsml-table-wrap">
        <table class="tsml-table">
          <thead>
            <tr>
              <th><button type="button" class="tsml-sort-btn" data-sort="dataset_name">Dataset <span class="tsml-sort-indicator"></span></button></th>
              <th><button type="button" class="tsml-sort-btn" data-sort="task">Task <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="train_cases">Train <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="test_cases">Test <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="total_cases">Total <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="n_channels">Channels <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="n_classes">Classes <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="max_length">Length <span class="tsml-sort-indicator"></span></button></th>
              <th class="tsml-num"><button type="button" class="tsml-sort-btn" data-sort="total_bytes">Size <span class="tsml-sort-indicator"></span></button></th>
              <th>Zenodo</th>
            </tr>
          </thead>
          <tbody id="tsml-tbody"></tbody>
        </table>
      </div>
    `;

    const els = {
      q: root.querySelector("#tsml-q"),
      benchmark: root.querySelector("#tsml-benchmark"),
      collection: root.querySelector("#tsml-collection"),
      task: root.querySelector("#tsml-task"),
      dim: root.querySelector("#tsml-dim"),
      length: root.querySelector("#tsml-length"),
      missing: root.querySelector("#tsml-missing"),
      variant: root.querySelector("#tsml-variant"),
      minCases: root.querySelector("#tsml-min-cases"),
      maxCases: root.querySelector("#tsml-max-cases"),
      minCh: root.querySelector("#tsml-min-ch"),
      maxCh: root.querySelector("#tsml-max-ch"),
      minLen: root.querySelector("#tsml-min-len"),
      maxLen: root.querySelector("#tsml-max-len"),
      count: root.querySelector("#tsml-count"),
      tbody: root.querySelector("#tsml-tbody"),
      reset: root.querySelector("#tsml-reset"),
      exportCsv: root.querySelector("#tsml-export-csv"),
      exportJson: root.querySelector("#tsml-export-json"),
      sortBtns: Array.from(root.querySelectorAll(".tsml-sort-btn")),
    };

    function syncInputsFromState() {
      els.q.value = state.q;
      els.benchmark.value = state.benchmark;
      els.collection.value = state.collection;
      els.task.value = state.task;
      els.dim.value = state.dim;
      els.length.value = state.length;
      els.missing.value = state.missing;
      els.variant.value = state.variant;
      els.minCases.value = state.minCases;
      els.maxCases.value = state.maxCases;
      els.minCh.value = state.minCh;
      els.maxCh.value = state.maxCh;
      els.minLen.value = state.minLen;
      els.maxLen.value = state.maxLen;
    }

    function syncUrlFromState() {
      const next = new URLSearchParams();
      if (state.q) next.set("q", state.q);
      if (state.benchmark) next.set("benchmark", state.benchmark);
      if (state.collection) next.set("collection", state.collection);
      if (state.task !== "all") next.set("task", state.task);
      if (state.dim !== "all") next.set("dim", state.dim);
      if (state.length !== "all") next.set("length", state.length);
      if (state.missing !== "all") next.set("missing", state.missing);
      const defaultVariant = state.benchmark ? "all" : "base";
      if (state.variant !== defaultVariant) next.set("variant", state.variant);
      if (state.minCases) next.set("min_cases", state.minCases);
      if (state.maxCases) next.set("max_cases", state.maxCases);
      if (state.minCh) next.set("min_ch", state.minCh);
      if (state.maxCh) next.set("max_ch", state.maxCh);
      if (state.minLen) next.set("min_len", state.minLen);
      if (state.maxLen) next.set("max_len", state.maxLen);
      if (state.sort !== "dataset_name") next.set("sort", state.sort);
      if (state.dir !== "asc") next.set("dir", state.dir);

      const qs = next.toString();
      const target = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
      window.history.replaceState(null, "", target);
    }

    let currentFiltered = [];

    function getFilteredAndSorted() {
      const query = state.q.trim().toLowerCase();
      const minCases = parseOptionalNumber(state.minCases);
      const maxCases = parseOptionalNumber(state.maxCases);
      const minCh = parseOptionalNumber(state.minCh);
      const maxCh = parseOptionalNumber(state.maxCh);
      const minLen = parseOptionalNumber(state.minLen);
      const maxLen = parseOptionalNumber(state.maxLen);

      const filtered = allDatasets.filter((row) => {
        if (state.variant === "base" && row.variant !== "base") return false;
        if (state.variant === "variants" && row.variant === "base") return false;
        if (state.benchmark && !row.benchmarks.includes(state.benchmark)) return false;
        if (state.collection && !row.collections.includes(state.collection)) return false;
        if (state.task !== "all" && row.task !== state.task) return false;
        if (state.dim === "univariate" && row.n_channels !== 1) return false;
        if (state.dim === "multivariate" && row.n_channels <= 1) return false;
        if (state.length === "equal" && !row.equal_length) return false;
        if (state.length === "unequal" && row.equal_length) return false;
        if (state.length === "unequal_case" && !row.unequal_case_lengths) return false;
        if (state.length === "unequal_channel" && !row.unequal_channel_lengths) return false;
        if (state.missing === "none" && row.has_missing) return false;
        if (state.missing === "has" && !row.has_missing) return false;

        if (minCases !== null && row.total_cases < minCases) return false;
        if (maxCases !== null && row.total_cases > maxCases) return false;
        if (minCh !== null && row.n_channels < minCh) return false;
        if (maxCh !== null && row.n_channels > maxCh) return false;
        if (minLen !== null && row.max_length < minLen) return false;
        if (maxLen !== null && row.min_length > maxLen) return false;

        if (query) {
          const hay = `${row.dataset_name} ${row.record_name} ${row.record_id}`.toLowerCase();
          if (!hay.includes(query)) return false;
        }
        return true;
      });

      const dirMul = state.dir === "desc" ? -1 : 1;
      const field = state.sort;
      filtered.sort((a, b) => {
        const av = a[field];
        const bv = b[field];
        if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
        if (bv === null || bv === undefined) return -1;
        if (typeof av === "string") {
          const cmp = av.localeCompare(bv, undefined, { sensitivity: "base" });
          return cmp !== 0 ? cmp * dirMul : a.dataset_name.localeCompare(b.dataset_name) * dirMul;
        }
        if (av !== bv) return (av - bv) * dirMul;
        return a.dataset_name.localeCompare(b.dataset_name);
      });

      return filtered;
    }

    function render() {
      syncUrlFromState();
      currentFiltered = getFilteredAndSorted();

      for (const btn of els.sortBtns) {
        const indicator = btn.querySelector(".tsml-sort-indicator");
        if (btn.dataset.sort === state.sort) {
          indicator.textContent = state.dir === "asc" ? "▲" : "▼";
        } else {
          indicator.textContent = "";
        }
      }

      const scopeTotal =
        state.variant === "base"
          ? payload.summary.dataset_record_count
          : state.variant === "variants"
          ? payload.summary.preprocessed_variant_count
          : payload.summary.dataset_variant_count;
      els.count.innerHTML = `Showing <strong>${formatInt(currentFiltered.length)}</strong> of ${formatInt(scopeTotal)} ${
        state.variant === "base" ? "base datasets" : "dataset variants"
      }`;

      if (!currentFiltered.length) {
        els.tbody.innerHTML = `<tr><td colspan="10" class="tsml-empty">No datasets match the active filters.</td></tr>`;
        return;
      }

      const htmlRows = currentFiltered.map((row) => {
        const extraVariants = variantsByRecord.get(row.record_name) || [];
        let pills = "";
        if (row.variant !== "base") {
          pills += `<span class="tsml-pill">_${row.variant}</span>`;
        } else if (state.variant === "base" && extraVariants.length) {
          for (const v of extraVariants) {
            pills += `<span class="tsml-pill" title="Includes _${v} variant">+${v}</span>`;
          }
        }
        if (row.unequal_channel_lengths) {
          pills += `<span class="tsml-pill tsml-pill-warn" title="Channels within a case have different lengths">unequal channels</span>`;
        }
        if (row.has_missing) {
          pills += `<span class="tsml-pill tsml-pill-warn" title="Contains missing values">missing values</span>`;
        }
        if (row.packaging === "zip") {
          pills += `<span class="tsml-pill" title="Published as ZIP archive">ZIP</span>`;
        }

        const tInfo = taskMeta[row.task] || {
          short: row.task.slice(0, 3).toUpperCase(),
          cls: "",
        };
        const taskPill = `<span class="tsml-pill ${tInfo.cls}">${tInfo.short}</span>`;

        const classesCell =
          row.n_classes !== null && row.n_classes !== undefined
            ? formatInt(row.n_classes)
            : `<span class="tsml-muted">—</span>`;

        return `<tr>
          <td><a class="tsml-dataset-link" href="${row.page_url}">${row.dataset_name}</a>${pills}</td>
          <td>${taskPill}</td>
          <td class="tsml-num">${formatInt(row.train_cases)}</td>
          <td class="tsml-num">${formatInt(row.test_cases)}</td>
          <td class="tsml-num">${formatInt(row.total_cases)}</td>
          <td class="tsml-num">${formatInt(row.n_channels)}</td>
          <td class="tsml-num">${classesCell}</td>
          <td class="tsml-num">${formatLength(row)}</td>
          <td class="tsml-num">${formatBytes(row.total_bytes)}</td>
          <td><a href="${row.record_url}" target="_blank" rel="noopener">${row.record_id}</a></td>
        </tr>`;
      });

      els.tbody.innerHTML = htmlRows.join("");
    }

    syncInputsFromState();
    render();

    els.q.addEventListener("input", () => {
      state.q = els.q.value;
      render();
    });

    els.benchmark.addEventListener("change", () => {
      state.benchmark = els.benchmark.value;
      if (state.benchmark && state.variant === "base") {
        state.variant = "all";
        els.variant.value = "all";
      } else if (!state.benchmark && state.variant === "all") {
        state.variant = "base";
        els.variant.value = "base";
      }
      render();
    });

    for (const [key, el] of [
      ["collection", els.collection],
      ["task", els.task],
      ["dim", els.dim],
      ["length", els.length],
      ["missing", els.missing],
      ["variant", els.variant],
    ]) {
      el.addEventListener("change", () => {
        state[key] = el.value;
        render();
      });
    }

    for (const [key, el] of [
      ["minCases", els.minCases],
      ["maxCases", els.maxCases],
      ["minCh", els.minCh],
      ["maxCh", els.maxCh],
      ["minLen", els.minLen],
      ["maxLen", els.maxLen],
    ]) {
      el.addEventListener("input", () => {
        state[key] = el.value;
        render();
      });
    }

    for (const btn of els.sortBtns) {
      btn.addEventListener("click", () => {
        const nextSort = btn.dataset.sort;
        if (state.sort === nextSort) {
          state.dir = state.dir === "asc" ? "desc" : "asc";
        } else {
          state.sort = nextSort;
          state.dir = nextSort === "dataset_name" || nextSort === "task" ? "asc" : "desc";
        }
        render();
      });
    }

    els.reset.addEventListener("click", () => {
      state.q = "";
      state.variant = "base";
      state.task = "all";
      state.dim = "all";
      state.length = "all";
      state.missing = "all";
      state.benchmark = "";
      state.collection = "";
      state.minCases = "";
      state.maxCases = "";
      state.minCh = "";
      state.maxCh = "";
      state.minLen = "";
      state.maxLen = "";
      state.sort = "dataset_name";
      state.dir = "asc";
      syncInputsFromState();
      render();
    });

    els.exportCsv.addEventListener("click", () => {
      triggerDownload("tsml-datasets.csv", "text/csv;charset=utf-8", toCsv(currentFiltered));
    });

    els.exportJson.addEventListener("click", () => {
      triggerDownload(
        "tsml-datasets.json",
        "application/json;charset=utf-8",
        JSON.stringify(currentFiltered, null, 2) + "\n"
      );
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("tsml-catalogue-root");
    if (root) initCatalogue(root);
  });
})();
