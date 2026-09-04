const state = { files: [], results: [] };

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const chooseButton = document.getElementById("chooseButton");
const queue = document.getElementById("queue");
const queueCount = document.getElementById("queueCount");
const fileList = document.getElementById("fileList");
const clearButton = document.getElementById("clearButton");
const extractButton = document.getElementById("extractButton");
const extractButtonText = document.getElementById("extractButtonText");
const spinner = document.getElementById("spinner");
const resultsSection = document.getElementById("resultsSection");
const resultsEl = document.getElementById("results");
const totalTiming = document.getElementById("totalTiming");
const resultTemplate = document.getElementById("resultTemplate");
const statusBadge = document.getElementById("statusBadge");

chooseButton.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (event) => {
  if (event.target !== chooseButton) fileInput.click();
});
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => addFiles([...fileInput.files]));

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});
dropZone.addEventListener("drop", (event) => addFiles([...event.dataTransfer.files]));

clearButton.addEventListener("click", () => {
  state.files = [];
  renderQueue();
});
extractButton.addEventListener("click", extractFiles);

function addFiles(files) {
  const allowed = /\.(jpe?g|png|webp|pdf)$/i;
  for (const file of files) {
    if (!allowed.test(file.name)) continue;
    if (state.files.length >= 20) break;

    const duplicate = state.files.some(
      (existing) =>
        existing.name === file.name &&
        existing.size === file.size &&
        existing.lastModified === file.lastModified
    );
    if (!duplicate) state.files.push(file);
  }
  fileInput.value = "";
  renderQueue();
}

function renderQueue() {
  queue.hidden = state.files.length === 0;
  queueCount.textContent = `${state.files.length} file${state.files.length === 1 ? "" : "s"}`;
  fileList.innerHTML = "";

  state.files.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = file.name;

    const size = document.createElement("span");
    size.className = "size";
    size.textContent = formatBytes(file.size);
    name.appendChild(size);

    const remove = document.createElement("button");
    remove.className = "remove-file";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      state.files.splice(index, 1);
      renderQueue();
    });

    row.append(name, remove);
    fileList.appendChild(row);
  });
}

async function extractFiles() {
  if (!state.files.length) return;

  const form = new FormData();
  state.files.forEach((file) => form.append("files", file, file.name));

  setLoading(true);
  resultsSection.hidden = true;

  try {
    const response = await fetch("/api/v1/mulkiya/extract/batch", {
      method: "POST",
      body: form,
    });

    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || "Extraction failed.");

    state.results = payload.results || [];
    renderResults();

    const failed = payload.failed || 0;
    totalTiming.textContent =
      `Total ${formatMs(payload.processing_time_ms)} · ${payload.total} file(s)` +
      (failed ? ` · ${failed} failed` : "");
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    alert(error.message);
  } finally {
    setLoading(false);
  }
}

function renderResults() {
  resultsEl.innerHTML = "";

  state.results.forEach((result) => {
    const node = resultTemplate.content.cloneNode(true);

    node.querySelector(".result-filename").textContent = result.filename;
    node.querySelector(".result-meta").textContent = `Processed in ${formatMs(result.processing_time_ms)}`;

    // A file that failed inside a batch: show why, and keep the other results.
    if (result.success === false) {
      const card = node.querySelector(".result-card");
      card.classList.add("failed");
      const box = document.createElement("div");
      box.className = "error-box";
      box.textContent = result.error || "This file could not be processed.";
      card.querySelector(".field-groups").replaceWith(box);
    }

    Object.entries(result.data || {}).forEach(([key, value]) => {
      const target = node.querySelector(`[data-field="${key}"]`);
      if (!target) return;

      if (value === null || value === undefined || value === "") {
        target.textContent = "Not detected";
        target.classList.add("empty");
      } else {
        target.textContent = displayValue(key, value);
      }

      const conf = result.confidence?.[key];
      if (conf !== null && conf !== undefined) {
        target.title = `OCR confidence: ${Math.round(conf * 100)}%`;
      }
    });

    const warningBox = node.querySelector(".warning-box");
    if (result.success !== false && result.warnings?.length) {
      warningBox.hidden = false;
      warningBox.textContent = result.warnings.join(" · ");
    }

    const jsonPanel = node.querySelector(".json-panel");
    jsonPanel.querySelector("pre").textContent = JSON.stringify(result, null, 2);

    node.querySelector(".toggle-json").addEventListener("click", (event) => {
      jsonPanel.hidden = !jsonPanel.hidden;
      event.currentTarget.textContent = jsonPanel.hidden ? "View JSON" : "Hide JSON";
    });

    // Raw OCR panel, present only when the API runs with INCLUDE_RAW_OCR=true.
    const ocrButton = node.querySelector(".toggle-ocr");
    const ocrPanel = node.querySelector(".ocr-panel");
    if (result.raw_ocr?.length) {
      ocrButton.hidden = false;
      fillOcrTable(ocrPanel.querySelector("tbody"), result.raw_ocr);
      ocrButton.addEventListener("click", (event) => {
        ocrPanel.hidden = !ocrPanel.hidden;
        event.currentTarget.textContent = ocrPanel.hidden ? "Raw OCR" : "Hide OCR";
      });
    }

    node.querySelector(".copy-json").addEventListener("click", async (event) => {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      const button = event.currentTarget;
      const old = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => (button.textContent = old), 1200);
    });

    resultsEl.appendChild(node);
  });
}

function fillOcrTable(tbody, lines) {
  tbody.innerHTML = "";
  const sorted = [...lines].sort((a, b) => a.cy - b.cy || a.cx - b.cx);

  for (const item of sorted) {
    const tr = document.createElement("tr");
    const cells = [
      Math.round(item.cy),
      Math.round(item.cx),
      item.lang,
      item.score.toFixed(2),
      item.text,
    ];

    cells.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index === 3 && item.score < 0.75) td.className = "low";
      if (index === 4) td.className = /[\u0600-\u06FF]/.test(item.text) ? "text ar" : "text";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

function displayValue(key, value) {
  const dateFields = new Set([
    "insurance_expiry",
    "registration_expiry",
    "registration_issuance",
  ]);

  if (dateFields.has(key) && /^\d{4}-\d{2}-\d{2}$/.test(String(value))) {
    const [year, month, day] = String(value).split("-");
    return `${day}-${month}-${year}`;
  }
  return String(value);
}

function setLoading(loading) {
  extractButton.disabled = loading;
  spinner.hidden = !loading;
  extractButtonText.textContent = loading ? "Reading Mulkiya…" : "Extract information";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatMs(ms) {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} sec`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    const data = await response.json();
    statusBadge.classList.add("ok");
    statusBadge.querySelector("span:last-child").textContent =
      `Ready · ${data.ocr_languages.join(" + ").toUpperCase()}`;
  } catch {
    statusBadge.classList.add("error");
    statusBadge.querySelector("span:last-child").textContent = "Service unavailable";
  }
}

checkHealth();
