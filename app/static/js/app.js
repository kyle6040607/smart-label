// 智慧分割標記助手 — 前端最小可用版
// 流程：上傳 → 選圖 → 自動/單點分割 → 標種子 → 審核紅色低信心片段 → 看統計
"use strict";

const state = {
  currentProjectId: null,
  currentImage: null,
  drawMode: false,
  drawing: false,
  points: [],
  segmenting: false,
  lastSegments: [],   // 目前畫布上的片段，供審核卡片 hover 加亮用
  imgBatchMode: false,
  segBatchMode: false,
  autoSegCompleted: false,
  undoStack: [],
  redoStack: [],
  tagColors: {},
};
const $ = (id) => document.getElementById(id);

// ---------- 類別色彩管理機制 (Tag Color Palette Manager) ----------
const DEFAULT_PALETTE = [
  "#4F9CFF", // Soft Blue
  "#36D399", // Emerald Green
  "#FFD166", // Amber Yellow
  "#A78BFA", // Violet Purple
  "#FF885B", // Coral Orange
  "#F472B6", // Hot Pink
  "#14B8A6", // Teal
  "#F59E0B", // Gold
  "#8B5CF6", // Purple
  "#3B82F6", // Royal Blue
  "#EC4899", // Magenta
  "#84CC16", // Lime Green
  "#06B6D4", // Cyan
  "#E11D48", // Crimson
  "#64748B"  // Slate Gray
];

function loadTagColors() {
  try {
    const saved = localStorage.getItem("smart_label_tag_colors");
    return saved ? JSON.parse(saved) : {};
  } catch (e) {
    return {};
  }
}

state.tagColors = loadTagColors();

function saveTagColors() {
  try {
    localStorage.setItem("smart_label_tag_colors", JSON.stringify(state.tagColors));
  } catch (e) { }
}

function getTagColor(label) {
  if (!label) return "#36d399";
  if (state.tagColors && state.tagColors[label]) {
    return state.tagColors[label];
  }
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  const idx = Math.abs(hash) % DEFAULT_PALETTE.length;
  return DEFAULT_PALETTE[idx];
}

function setTagColor(label, color) {
  if (!state.tagColors) state.tagColors = {};
  state.tagColors[label] = color;
  saveTagColors();
}

function updateThumbTagDots(label, newColor) {
  document.querySelectorAll(".thumb-tag-dot").forEach((dot) => {
    if (dot.dataset.tag === label) {
      dot.style.backgroundColor = newColor;
    }
  });
}

function hsvToHex(h, s, v) {
  s /= 100;
  v /= 100;
  let r = 0, g = 0, b = 0;
  let i = Math.floor((h / 60) % 6);
  let f = (h / 60) - Math.floor(h / 60);
  let p = v * (1 - s);
  let q = v * (1 - f * s);
  let t = v * (1 - (1 - f) * s);
  switch (i) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  const toHex = x => Math.round(x * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToHsv(hex) {
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  let r = parseInt(c.substring(0, 2), 16) / 255 || 0;
  let g = parseInt(c.substring(2, 4), 16) / 255 || 0;
  let b = parseInt(c.substring(4, 6), 16) / 255 || 0;

  let max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0, v = max;
  let d = max - min;
  s = max === 0 ? 0 : d / max;

  if (max === min) {
    h = 0;
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return { h: Math.round(h * 360), s: Math.round(s * 100), v: Math.round(v * 100) };
}

let activeColorPopover = null;

function closeColorPickerPopover() {
  if (activeColorPopover) {
    if (activeColorPopover.parentElement) {
      activeColorPopover.parentElement.removeChild(activeColorPopover);
    }
    activeColorPopover = null;
  }
}

function openColorPickerPopover(e, label) {
  e.stopPropagation();
  closeColorPickerPopover();

  const popover = document.createElement("div");
  popover.className = "color-picker-popover";

  const rect = e.target.getBoundingClientRect();
  popover.style.top = `${Math.min(window.innerHeight - 300, Math.max(10, rect.bottom + 6))}px`;
  popover.style.left = `${Math.min(window.innerWidth - 250, Math.max(10, rect.left - 80))}px`;

  const currentColor = getTagColor(label);
  const currentHsv = hexToHsv(currentColor);

  popover.innerHTML = `
    <div class="color-picker-header">
      <span>「${label}」標籤色彩</span>
      <button type="button" class="toast-close" style="font-size:14px;">✕</button>
    </div>
    <div class="color-presets-grid">
      ${DEFAULT_PALETTE.map((c) => `
        <div class="color-swatch ${c.toLowerCase() === currentColor.toLowerCase() ? "active" : ""}" style="background-color: ${c};" data-color="${c}"></div>
      `).join("")}
    </div>
    <div class="custom-picker-section">
      <div class="sv-box" id="svBox">
        <div class="sv-handle" id="svHandle"></div>
      </div>
      <div class="hue-bar" id="hueBar">
        <div class="hue-handle" id="hueHandle"></div>
      </div>
      <div class="color-inputs-row">
        <div class="color-preview-box" id="colorPreviewBox"></div>
        <div class="hex-input-wrap">
          <span style="opacity:0.6; font-size:11px; font-weight:600;">HEX</span>
          <input type="text" id="hexInput" class="hex-input" maxLength="7" value="${currentColor}" />
        </div>
      </div>
    </div>
  `;

  popover.querySelector(".toast-close").onclick = (evt) => {
    evt.stopPropagation();
    closeColorPickerPopover();
  };

  const svBox = popover.querySelector("#svBox");
  const svHandle = popover.querySelector("#svHandle");
  const hueBar = popover.querySelector("#hueBar");
  const hueHandle = popover.querySelector("#hueHandle");
  const previewBox = popover.querySelector("#colorPreviewBox");
  const hexInput = popover.querySelector("#hexInput");

  const applyColor = async (newColor, notify = true) => {
    setTagColor(label, newColor);
    updateThumbTagDots(label, newColor);
    if (state.currentImage && state.lastSegments) {
      await redraw(state.lastSegments);
    }
    await refreshSidebar();
    if (notify) {
      showToast(`已更新「${label}」的標籤色彩`, "success");
    }
  };

  const updateUIFromHsv = (skipApply = false) => {
    const hex = hsvToHex(currentHsv.h, currentHsv.s, currentHsv.v);
    svBox.style.backgroundColor = `hsl(${currentHsv.h}, 100%, 50%)`;
    svHandle.style.left = `${currentHsv.s}%`;
    svHandle.style.top = `${100 - currentHsv.v}%`;
    hueHandle.style.left = `${(currentHsv.h / 360) * 100}%`;
    previewBox.style.backgroundColor = hex;
    hexInput.value = hex;

    popover.querySelectorAll(".color-swatch").forEach((s) => {
      s.classList.toggle("active", s.dataset.color.toLowerCase() === hex.toLowerCase());
    });

    if (!skipApply) {
      applyColor(hex, false);
    }
  };

  let isDraggingSV = false;
  let isDraggingHue = false;

  const handleSVMove = (evt) => {
    const boxRect = svBox.getBoundingClientRect();
    const x = Math.max(0, Math.min(boxRect.width, evt.clientX - boxRect.left));
    const y = Math.max(0, Math.min(boxRect.height, evt.clientY - boxRect.top));
    currentHsv.s = Math.round((x / boxRect.width) * 100);
    currentHsv.v = Math.round((1 - y / boxRect.height) * 100);
    updateUIFromHsv();
  };

  const handleHueMove = (evt) => {
    const barRect = hueBar.getBoundingClientRect();
    const x = Math.max(0, Math.min(barRect.width, evt.clientX - barRect.left));
    currentHsv.h = Math.round((x / barRect.width) * 360);
    updateUIFromHsv();
  };

  svBox.addEventListener("mousedown", (evt) => {
    isDraggingSV = true;
    handleSVMove(evt);
    evt.preventDefault();
  });

  hueBar.addEventListener("mousedown", (evt) => {
    isDraggingHue = true;
    handleHueMove(evt);
    evt.preventDefault();
  });

  const onGlobalMouseMove = (evt) => {
    if (isDraggingSV) handleSVMove(evt);
    else if (isDraggingHue) handleHueMove(evt);
  };

  const onGlobalMouseUp = () => {
    if (isDraggingSV || isDraggingHue) {
      isDraggingSV = false;
      isDraggingHue = false;
      applyColor(hsvToHex(currentHsv.h, currentHsv.s, currentHsv.v), true);
    }
  };

  window.addEventListener("mousemove", onGlobalMouseMove);
  window.addEventListener("mouseup", onGlobalMouseUp);

  popover.querySelectorAll(".color-swatch").forEach((swatch) => {
    swatch.onclick = (evt) => {
      evt.stopPropagation();
      const col = swatch.dataset.color;
      const parsed = hexToHsv(col);
      currentHsv.h = parsed.h;
      currentHsv.s = parsed.s;
      currentHsv.v = parsed.v;
      updateUIFromHsv(false);
      applyColor(col, true);
    };
  });

  hexInput.oninput = (evt) => {
    let val = evt.target.value.trim();
    if (!val.startsWith("#")) val = "#" + val;
    if (/^#[0-9A-Fa-f]{6}$/.test(val)) {
      const parsed = hexToHsv(val);
      currentHsv.h = parsed.h;
      currentHsv.s = parsed.s;
      currentHsv.v = parsed.v;
      updateUIFromHsv(false);
      applyColor(val, false);
    }
  };

  updateUIFromHsv(true);
  document.body.appendChild(popover);
  activeColorPopover = popover;
}

document.addEventListener("click", (e) => {
  if (activeColorPopover && !activeColorPopover.contains(e.target)) {
    closeColorPickerPopover();
  }
});

// ---------- 右下角輕量 Toast 通知機制 ----------
function showToast(message, type = "info", duration = 3000) {
  const container = $("toastContainer");
  if (!container) return;

  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠️",
    info: "ℹ️"
  };

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">${message}</div>
    <button type="button" class="toast-close" title="關閉">✕</button>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  const dismiss = () => {
    toast.classList.remove("show");
    toast.style.transform = "translateY(10px)";
    toast.style.opacity = "0";
    setTimeout(() => {
      if (toast.parentElement) toast.parentElement.removeChild(toast);
    }, 300);
  };

  closeBtn.onclick = dismiss;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      toast.classList.add("show");
    });
  });

  if (duration > 0) {
    setTimeout(dismiss, duration);
  }
}

// 原圖快取：縮圖與重繪共用同一份，避免重複下載解碼
const imageCache = new Map();
function loadImage(imageId) {
  if (!imageCache.has(imageId)) {
    imageCache.set(imageId, new Promise((resolve, reject) => {
      const pic = new Image();
      pic.onload = () => resolve(pic);
      pic.onerror = reject;
      pic.src = `/api/images/${imageId}/file`;
    }));
  }
  return imageCache.get(imageId);
}

// 遮罩影像快取（同步快取 Image 物件，防止非同步 await 造成的時序交錯與閃爍）
const maskImages = {};

// 著色後的遮罩快取：key = "segId:color"，value = 裁到 bbox 的小 canvas
const tintedMaskCache = new Map();

// 取得（或建立）指定顏色的著色遮罩，尚未下載完成時回傳 null
function getTintedMask(s, color) {
  const key = `${s.id}:${color}`;
  const cached = tintedMaskCache.get(key);
  if (cached) return cached;

  const maskImg = maskImages[s.id];
  if (!maskImg || !maskImg.complete || maskImg.naturalWidth === 0) return null;

  const [x, y, w, h] = s.bbox;
  if (!w || !h) return null;

  // 只處理 bbox 範圍，不用開整張圖大小的 canvas
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tctx = tempCanvas.getContext("2d");
  tctx.drawImage(maskImg, x, y, w, h, 0, 0, w, h);

  // 將黑白遮罩的「亮度」映射為「透明度」，並填上目標色
  const imgData = tctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  for (let i = 0; i < data.length; i += 4) {
    // 亮度 × 原透明度，同時相容黑底白階與透明底白階的遮罩
    data[i + 3] = Math.round((data[i] / 255) * data[i + 3]);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
  tctx.putImageData(imgData, 0, 0);

  tintedMaskCache.set(key, tempCanvas);
  return tempCanvas;
}

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

let progressInterval = null;
let currentProgress = 0;

function startFakeProgress(startVal = 10, limitVal = 75) {
  stopFakeProgress();
  currentProgress = startVal;
  updateProgressBar(Math.round(currentProgress));

  progressInterval = setInterval(() => {
    if (currentProgress < limitVal) {
      const increment = (limitVal - currentProgress) * 0.04;
      currentProgress += Math.max(0.1, increment);
      updateProgressBar(Math.round(currentProgress));
    }
  }, 200);
}

function stopFakeProgress() {
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
}
function updateAutoSegBtn(keepText = false) {
  const btn = $("autoSegBtn");
  if (!btn) return;
  if (!state.currentImage) {
    btn.disabled = true;
    btn.textContent = "自動分割";
    return;
  }

  if (keepText) {
    btn.disabled = true;
    return;
  }

  if (state.autoSegCompleted) {
    btn.disabled = true;
    btn.textContent = "✓ 已完成分割";
  } else {
    btn.disabled = false;
    btn.textContent = "自動分割";
  }
}

function setSegmentationLoading(active, message = "分割中…", showProgress = false) {
  state.segmenting = active;
  $("segmentLoadingText").textContent = message;
  $("segmentLoading").hidden = !active;

  if (active) {
    if (showProgress) {
      $("progressBarContainer").style.display = "block";
      $("progressPercentText").style.display = "inline";
    } else {
      $("progressBarContainer").style.display = "none";
      $("progressPercentText").style.display = "none";
    }
  } else {
    stopFakeProgress();
  }

  canvas.closest(".canvas-wrap").classList.toggle("is-loading", active);
  canvas.setAttribute("aria-busy", String(active));

  if (active) {
    $("autoSegBtn").disabled = true;
  } else {
    updateAutoSegBtn();
  }

  $("drawBtn").disabled = active || !state.currentImage;
  $("textPromptInput").disabled = active;
  $("textSegBtn").disabled = active;
}

function updateProgressBar(percent) {
  $("progressBar").style.width = `${percent}%`;
  $("progressPercentText").textContent = `${percent}%`;
}

async function responseError(res, fallback) {
  const detail = await res.text();
  return new Error(detail ? `${fallback}：${detail}` : fallback);
}

async function fetchWithProgress(url, options, onProgress) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw await responseError(response, "請求失敗");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.trim()) {
        const data = JSON.parse(line);
        if (data.event === "progress") {
          onProgress(data);
        } else if (data.event === "done") {
          finalResult = data;
        } else if (data.event === "error") {
          throw new Error(data.message || "發生錯誤");
        }
      }
    }
  }

  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer);
      if (data.event === "progress") {
        onProgress(data);
      } else if (data.event === "done") {
        finalResult = data;
      } else if (data.event === "error") {
        throw new Error(data.message || "發生錯誤");
      }
    } catch (e) {
      // ignore
    }
  }

  if (!finalResult) {
    throw new Error("伺服器未回傳完成狀態");
  }
  return finalResult;
}

// 把滑鼠座標換算成 canvas（原圖）座標
function toImageXY(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((e.clientX - rect.left) * (canvas.width / rect.width)),
    y: Math.round((e.clientY - rect.top) * (canvas.height / rect.height)),
  };
}

// 輔助函式：從觸控事件中取得相對於視窗的座標
function getTouchPos(e) {
  const touch = e.touches[0] || e.changedTouches[0];
  return {
    clientX: touch.clientX,
    clientY: touch.clientY
  };
}

state.currentXhr = null;
state.isUploading = false;
state.isAborted = false;
state.allImages = [];
state.renderedImageCount = 0;
const GALLERY_PAGE_SIZE = 100;

function updateFileCountHint() {
  const fileInputEl = $("fileInput");
  const files = state.stagedFiles.length ? state.stagedFiles : Array.from(fileInputEl.files);
  const fileCountHint = $("fileCountHint");
  const uploadBtn = $("uploadBtn");

  if (state.isUploading) return;

  if (!files.length) {
    if (fileCountHint) fileCountHint.textContent = "尚未選擇檔案";
    if (uploadBtn) uploadBtn.style.display = "none";
    return;
  }

  if (uploadBtn) uploadBtn.style.display = "inline-block";

  if (!fileCountHint) return;

  const totalBytes = files.reduce((acc, f) => acc + f.size, 0);
  const mbStr = (totalBytes / (1024 * 1024)).toFixed(1);
  const archiveCount = files.filter((f) => /\.(zip|7z|tar|gz|tgz|bz2|tbz2|xz|txz)$/i.test(f.name)).length;

  if (files.length === 1) {
    fileCountHint.textContent = `已選擇：${files[0].name} (${mbStr} MB)`;
  } else {
    const archiveInfo = archiveCount > 0 ? ` · 含 ${archiveCount} 個壓縮包` : "";
    fileCountHint.textContent = `已選擇 ${files.length} 個檔案 (${mbStr} MB${archiveInfo})`;
  }
}

function initDropZone() {
  const dropZone = $("dropZone");
  const fileInputEl = $("fileInput");
  const cancelUploadBtn = $("cancelUploadBtn");
  if (!dropZone || !fileInputEl) return;

  if (cancelUploadBtn) {
    cancelUploadBtn.onclick = async (e) => {
      e.stopPropagation();
      state.isAborted = true;
      try {
        fetch("/api/images/cancel_upload", { method: "POST" });
      } catch (err) { }
      if (state.currentXhr) {
        try {
          state.currentXhr.abort();
        } catch (err) { }
        state.currentXhr = null;
      }
      await loadThumbs();
      expandGallery();
    };
  }

  dropZone.onclick = (e) => {
    if (e.target.closest("#uploadBtn") || e.target.closest("#cancelUploadBtn")) {
      return;
    }
    fileInputEl.click();
  };

  fileInputEl.onchange = () => {
    state.stagedFiles = Array.from(fileInputEl.files);
    updateFileCountHint();
  };

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      state.stagedFiles = Array.from(e.dataTransfer.files);
      updateFileCountHint();
    }
  });
}
initDropZone();

$("uploadBtn").onclick = async (e) => {
  if (e) e.stopPropagation();
  const fileInputEl = $("fileInput");
  const files = state.stagedFiles.length ? state.stagedFiles : Array.from(fileInputEl.files);
  if (!files.length) return alert("請先選擇照片或資料集壓縮檔");

  const uploadBtn = $("uploadBtn");
  const cancelUploadBtn = $("cancelUploadBtn");
  const fileCountHint = $("fileCountHint");

  uploadBtn.disabled = true;
  if (cancelUploadBtn) {
    cancelUploadBtn.style.display = "inline-block";
    cancelUploadBtn.disabled = false;
  }
  state.isUploading = true;
  state.isAborted = false;

  const BATCH_SIZE_LIMIT = 50 * 1024 * 1024; // 50 MB
  const BATCH_COUNT_LIMIT = 50;

  let uploadedCount = 0;
  let uploadedBytes = 0;
  const totalFiles = files.length;
  const totalBytes = files.reduce((acc, f) => acc + f.size, 0);

  const sendBatch = (batch) => {
    return new Promise((resolve, reject) => {
      if (state.isAborted) {
        return reject(new Error("使用者已取消上傳"));
      }

      const xhr = new XMLHttpRequest();
      state.currentXhr = xhr;
      const fd = new FormData();
      for (const f of batch) fd.append("files", f);
      if (state.currentProjectId) fd.append("project_id", state.currentProjectId);

      xhr.upload.onprogress = (e) => {
        if (state.isAborted) return;
        if (e.lengthComputable && fileCountHint) {
          const currentBatchUploaded = e.loaded;
          const totalProgressBytes = uploadedBytes + currentBatchUploaded;
          const pct = Math.min(100, Math.round((totalProgressBytes / totalBytes) * 100));
          const mbUploaded = (totalProgressBytes / (1024 * 1024)).toFixed(1);
          const mbTotal = (totalBytes / (1024 * 1024)).toFixed(1);

          if (e.loaded >= e.total) {
            fileCountHint.textContent = `檔案傳送完成 (100%) · 伺服器準備解壓照片…`;
          } else {
            fileCountHint.textContent = `正在傳送檔案… ${pct}% (${mbUploaded} / ${mbTotal} MB)`;
          }
        }
      };

      xhr.onprogress = () => {
        if (state.isAborted) return;
        const text = xhr.responseText;
        const lines = text.split("\n");
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line.trim());
            if (data.event === "progress") {
              state.lastProcessedCount = data.created_count || data.current;
              state.lastTotalCount = data.total;
              const pct = Math.round((data.current / data.total) * 100);
              if (fileCountHint) {
                fileCountHint.textContent = `正在解壓與處理照片… ${data.current.toLocaleString()} / ${data.total.toLocaleString()} 張 (${pct}%)`;
              }
              if (data.latest_image) {
                appendThumbs([data.latest_image]);
                expandGallery();
              }
            }
          } catch (err) { }
        }
      };

      xhr.onload = () => {
        state.currentXhr = null;
        if (state.isAborted) {
          return reject(new Error("使用者已取消上傳"));
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            let createdImages = [];
            const text = xhr.responseText.trim();
            if (text.startsWith("{")) {
              const lines = text.split("\n");
              for (const line of lines) {
                if (!line.trim()) continue;
                try {
                  const data = JSON.parse(line.trim());
                  if (data.event === "done" && Array.isArray(data.created)) {
                    createdImages = data.created;
                  }
                } catch (e) { }
              }
            } else {
              createdImages = JSON.parse(text);
            }
            uploadedCount += batch.length;
            uploadedBytes += batch.reduce((acc, f) => acc + f.size, 0);

            if (Array.isArray(createdImages) && createdImages.length > 0) {
              appendThumbs(createdImages);
              expandGallery();
            }
            resolve(createdImages);
          } catch (err) {
            reject(new Error("伺服器回傳格式錯誤"));
          }
        } else {
          let errorMsg = xhr.responseText;
          try {
            const json = JSON.parse(xhr.responseText);
            errorMsg = json.error || json.message || xhr.responseText;
          } catch (e) { }
          reject(new Error(errorMsg || `HTTP ${xhr.status}`));
        }
      };

      xhr.onerror = () => {
        state.currentXhr = null;
        reject(new Error(state.isAborted ? "使用者已取消上傳" : "網路傳送失敗"));
      };

      xhr.onabort = () => {
        state.currentXhr = null;
        reject(new Error("使用者已取消上傳"));
      };

      xhr.open("POST", "/api/images?stream=1");
      xhr.setRequestHeader("Accept", "application/x-ndjson");
      xhr.send(fd);
    });
  };

  try {
    let currentBatch = [];
    let currentBatchSize = 0;

    for (const f of files) {
      if (state.isAborted) break;
      if (currentBatch.length > 0 && (currentBatchSize + f.size > BATCH_SIZE_LIMIT || currentBatch.length >= BATCH_COUNT_LIMIT)) {
        await sendBatch(currentBatch);
        currentBatch = [];
        currentBatchSize = 0;
      }
      currentBatch.push(f);
      currentBatchSize += f.size;
    }

    if (currentBatch.length > 0 && !state.isAborted) {
      await sendBatch(currentBatch);
    }
  } catch (err) {
    if (state.isAborted) {
      if (fileCountHint) {
        const countInfo = state.lastProcessedCount
          ? `（已保留已解壓處理的 ${state.lastProcessedCount.toLocaleString()} / ${(state.lastTotalCount || '?').toLocaleString()} 張照片）`
          : "";
        fileCountHint.textContent = `已中途取消上傳${countInfo}`;
      }
    } else {
      alert(`上傳中斷：${err.message}`);
    }
  } finally {
    fileInputEl.value = "";
    state.stagedFiles = [];
    state.isUploading = false;
    state.currentXhr = null;
    uploadBtn.disabled = false;
    if (cancelUploadBtn) {
      cancelUploadBtn.style.display = "none";
      cancelUploadBtn.disabled = false;
    }
    if (state.isAborted) {
      await loadThumbs();
      expandGallery();
      setTimeout(() => {
        if (!state.isUploading) updateFileCountHint();
      }, 3000);
    } else {
      updateFileCountHint();
    }
  }
};

function createThumbElement(im) {
  const wrap = document.createElement("div");
  wrap.className = "thumb";

  const el = document.createElement("img");
  el.loading = "lazy";
  el.src = `/api/images/${im.id}/file`;
  el.title = im.filename;
  if (state.currentImage && state.currentImage.id === im.id) {
    el.classList.add("active");
  }
  el.onclick = () => {
    if (state.imgBatchMode) {
      chk.checked = !chk.checked;
      if (!state.selectedImageIds) state.selectedImageIds = new Set();
      if (chk.checked) state.selectedImageIds.add(im.id);
      else state.selectedImageIds.delete(im.id);
      updateImgBatchBtnState();
      return;
    }
    selectImage(im, el);
  };

  // 批次管理勾選框
  const chk = document.createElement("input");
  chk.type = "checkbox";
  chk.className = "thumb-chk";
  chk.dataset.id = im.id;
  chk.checked = state.selectedImageIds ? state.selectedImageIds.has(im.id) : false;
  chk.onclick = (e) => {
    e.stopPropagation();
    if (!state.selectedImageIds) state.selectedImageIds = new Set();
    if (chk.checked) state.selectedImageIds.add(im.id);
    else state.selectedImageIds.delete(im.id);
    updateImgBatchBtnState();
  };

  const del = document.createElement("button");
  del.className = "thumb-del";
  del.textContent = "×";
  del.title = "刪除這張";
  del.onclick = (e) => { e.stopPropagation(); deleteImage(im); };

  wrap.append(chk, el, del);

  // 💡 若照片已有分割紀錄，動態繪製左上角標記徽章與右下角類別顏色圓點
  const segCount = im.segment_count || 0;
  if (segCount > 0) {
    const badge = document.createElement("div");
    badge.className = "thumb-seg-badge";
    badge.title = `這張照片已有 ${segCount} 個遮罩片段`;
    badge.innerHTML = `✓ ${segCount}`;
    wrap.appendChild(badge);

    const tags = im.segment_tags || [];
    if (tags.length > 0) {
      const dotsContainer = document.createElement("div");
      dotsContainer.className = "thumb-tag-dots";
      dotsContainer.title = `包含類別：${tags.join("、")}`;
      tags.forEach((tag) => {
        const dot = document.createElement("span");
        dot.className = "thumb-tag-dot";
        dot.dataset.tag = tag;
        dot.style.backgroundColor = getTagColor(tag);
        dotsContainer.appendChild(dot);
      });
      wrap.appendChild(dotsContainer);
    }
  }

  return wrap;
}

// 照片庫折疊控制
function initGalleryCollapse() {
  const toggleBtn = $("toggleGalleryCollapseBtn");
  const wrapper = $("thumbsWrapper");
  if (!toggleBtn || !wrapper) return;

  toggleBtn.onclick = () => {
    const isCollapsed = wrapper.classList.toggle("collapsed");
    updateGalleryToggleUI(!isCollapsed);
  };

  // 滾動觸發無限載入
  wrapper.addEventListener("scroll", () => {
    if (wrapper.scrollTop + wrapper.clientHeight >= wrapper.scrollHeight - 120) {
      renderMoreThumbs();
    }
  });
}

function getFilteredImages() {
  if (!state.allImages) return [];
  const filter = state.imgFilter || "all";
  if (filter === "all") return state.allImages;
  if (filter === "unsegmented") {
    return state.allImages.filter((im) => (im.segment_count || 0) === 0);
  }
  if (filter === "segmented") {
    return state.allImages.filter((im) => (im.segment_count || 0) > 0);
  }
  if (filter.startsWith("tag:")) {
    const tagName = filter.substring(4);
    return state.allImages.filter((im) => (im.segment_tags || []).includes(tagName));
  }
  return state.allImages;
}

function updateImgFilterOptions() {
  const sel = $("imgFilterSelect");
  if (!sel) return;

  const currentValue = sel.value || state.imgFilter || "all";
  const tagsSet = new Set();
  (state.allImages || []).forEach((im) => {
    (im.segment_tags || []).forEach((t) => tagsSet.add(t));
  });

  let html = `
    <option value="all">全部照片</option>
    <option value="unsegmented">未標記/未分割 (0 遮罩)</option>
    <option value="segmented">已標記/已有遮罩 (≥1 遮罩)</option>
  `;

  if (tagsSet.size > 0) {
    html += `<optgroup label="依標籤類別篩選">`;
    Array.from(tagsSet).sort().forEach((tag) => {
      html += `<option value="tag:${tag}">類別：${tag}</option>`;
    });
    html += `</optgroup>`;
  }

  sel.innerHTML = html;
  if (sel.querySelector(`option[value="${currentValue}"]`)) {
    sel.value = currentValue;
  } else {
    sel.value = "all";
    state.imgFilter = "all";
  }
}

function updateGalleryToggleUI(isExpanded) {
  const icon = $("galleryToggleIcon");
  const text = $("galleryToggleText");
  const filtered = getFilteredImages();
  const totalCount = state.allImages ? state.allImages.length : 0;
  const filteredCount = filtered.length;
  const wrapper = $("thumbsWrapper");
  const isCurrentlyCollapsed = wrapper ? wrapper.classList.contains("collapsed") : true;

  const countStr = filteredCount === totalCount ? `${totalCount}` : `${filteredCount}/${totalCount}`;

  if (icon) icon.textContent = isCurrentlyCollapsed ? "▼" : "▲";
  if (text) text.textContent = isCurrentlyCollapsed ? `展開照片庫 (${countStr})` : `折疊照片庫 (${countStr})`;
}

function expandGallery() {
  const wrapper = $("thumbsWrapper");
  if (wrapper && wrapper.classList.contains("collapsed")) {
    wrapper.classList.remove("collapsed");
  }
  updateGalleryToggleUI(true);
}
initGalleryCollapse();

function renderMoreThumbs() {
  const images = getFilteredImages();
  if (!images || state.renderedImageCount >= images.length) return;
  const nextSlice = images.slice(state.renderedImageCount, state.renderedImageCount + GALLERY_PAGE_SIZE);
  state.renderedImageCount += nextSlice.length;

  const box = $("thumbs");
  if (!box) return;

  const fragment = document.createDocumentFragment();
  nextSlice.forEach((im) => {
    if (box.querySelector(`.thumb-chk[data-id="${im.id}"]`)) return;
    const wrap = createThumbElement(im);
    fragment.appendChild(wrap);
  });
  box.appendChild(fragment);

  if (!state.imgBatchMode) {
    const selectAll = $("selectAllImgs");
    if (selectAll) selectAll.checked = false;
    updateImgBatchBtnState();
  }
  updateGalleryToggleUI();
}

function appendThumbs(newImages) {
  if (!Array.isArray(newImages) || !newImages.length) return;

  newImages.forEach((im) => {
    if (!state.allImages.some((item) => item.id === im.id)) {
      state.allImages.unshift(im);
      state.renderedImageCount += 1;
    }
  });

  updateImgFilterOptions();

  const box = $("thumbs");
  if (!box) return;

  const fragment = document.createDocumentFragment();
  newImages.forEach((im) => {
    if (box.querySelector(`.thumb-chk[data-id="${im.id}"]`)) return;
    const wrap = createThumbElement(im);
    fragment.appendChild(wrap);
  });

  box.insertBefore(fragment, box.firstChild);

  if (!state.imgBatchMode) {
    const selectAll = $("selectAllImgs");
    if (selectAll) selectAll.checked = false;
    updateImgBatchBtnState();
  }
  updateGalleryToggleUI();
}

async function loadThumbs() {
  const imgs = await (await fetch("/api/images")).json();
  state.allImages = Array.isArray(imgs) ? imgs : [];
  state.renderedImageCount = 0;

  updateImgFilterOptions();

  const box = $("thumbs");
  if (box) {
    box.innerHTML = "";
    box.classList.toggle("batch-active", state.imgBatchMode);
  }

  renderMoreThumbs();
  updateGalleryToggleUI();
  updateImgNavUI();
}

async function deleteImage(im) {
  if (!confirm(`確定刪除「${im.filename}」？連同它的遮罩會一起清掉。`)) return;
  const res = await fetch(`/api/images/${im.id}`, { method: "DELETE" });
  if (!res.ok) return showToast("刪除失敗：" + (await res.text()), "error");
  // 若刪的是目前選中的圖，清空畫布
  if (state.currentImage && state.currentImage.id === im.id) {
    state.currentImage = null;
    updateAutoSegBtn();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  await loadThumbs();
  await refreshSidebar();
  updateImgNavUI();
  showToast("已刪除照片", "info");
}

// ---------- 照片切換導航 (上一張 / 下一張 + 鍵盤捷徑) ----------
function updateImgNavUI() {
  const prevBtn = $("prevImgBtn");
  const nextBtn = $("nextImgBtn");
  const counter = $("imgNavCounter");
  if (!prevBtn || !nextBtn || !counter) return;

  const total = state.allImages ? state.allImages.length : 0;
  if (total === 0) {
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    counter.textContent = "0 / 0";
    return;
  }

  prevBtn.disabled = total <= 1;
  nextBtn.disabled = total <= 1;

  const currentIdx = state.currentImage ? state.allImages.findIndex((img) => img.id === state.currentImage.id) : -1;
  if (currentIdx > -1) {
    counter.textContent = `${currentIdx + 1} / ${total}`;
  } else {
    counter.textContent = `0 / ${total}`;
  }
}

function switchPrevImage() {
  const images = getFilteredImages();
  if (!images || images.length === 0 || state.segmenting) return;
  const total = images.length;
  let currentIdx = state.currentImage ? images.findIndex((img) => img.id === state.currentImage.id) : -1;

  let targetIdx;
  if (currentIdx === -1) {
    targetIdx = 0;
  } else if (currentIdx === 0) {
    targetIdx = total - 1; // 循環到最後一張
  } else {
    targetIdx = currentIdx - 1;
  }

  const targetImg = images[targetIdx];
  if (targetImg) {
    const thumbImg = document.querySelector(`.thumb img[src*="/api/images/${targetImg.id}/file"]`) ||
      document.querySelector(`.thumb-chk[data-id="${targetImg.id}"]`)?.nextElementSibling;
    selectImage(targetImg, thumbImg);
  }
}

function switchNextImage() {
  const images = getFilteredImages();
  if (!images || images.length === 0 || state.segmenting) return;
  const total = images.length;
  let currentIdx = state.currentImage ? images.findIndex((img) => img.id === state.currentImage.id) : -1;

  let targetIdx;
  if (currentIdx === -1 || currentIdx >= total - 1) {
    targetIdx = 0; // 循環到第一張
  } else {
    targetIdx = currentIdx + 1;
  }

  const targetImg = images[targetIdx];
  if (targetImg) {
    const thumbImg = document.querySelector(`.thumb img[src*="/api/images/${targetImg.id}/file"]`) ||
      document.querySelector(`.thumb-chk[data-id="${targetImg.id}"]`)?.nextElementSibling;
    selectImage(targetImg, thumbImg);
  }
}

function pushUndoAction(action) {
  // 只記錄目前照片的操作
  if (!state.currentImage || action.imageId !== state.currentImage.id) return;
  state.undoStack.push(action);
  updateUndoUI();
}

function updateUndoUI() {
  const undoBtn = $("undoBtn");
  if (undoBtn) undoBtn.disabled = state.undoStack.length === 0;
}

async function performUndo() {
  if (state.undoStack.length === 0 || state.segmenting) return;
  const action = state.undoStack.pop();
  updateUndoUI();

  try {
    if (action.type === "REVIEW_SEGMENT") {
      if (action.prevLabel) {
        await fetch(`/api/segments/${action.segId}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label: action.prevLabel }),
        });
      } else {
        // 原本沒有人工審核標籤，復原即取消審核狀態並刪除範例數量
        await fetch(`/api/segments/${action.segId}/unreview`, { method: "POST" });
      }
      await refreshAfterSegChange();
    } else if (action.type === "CREATE_SEGMENTS") {
      // 復原新分割的區塊 (刪除那些 segments)
      if (Array.isArray(action.segIds) && action.segIds.length > 0) {
        for (const id of action.segIds) {
          try {
            await fetch(`/api/segments/${id}`, { method: "DELETE" });
          } catch (e) { }
        }
      }
      state.autoSegCompleted = false;
      updateAutoSegBtn();
      await refreshAfterSegChange();
    }
  } catch (err) {
    console.error("復原失敗:", err);
  } finally {
    updateUndoUI();
  }
}

function initImgNavEvents() {
  const prevBtn = $("prevImgBtn");
  const nextBtn = $("nextImgBtn");
  const undoBtn = $("undoBtn");

  if (prevBtn) prevBtn.onclick = () => switchPrevImage();
  if (nextBtn) nextBtn.onclick = () => switchNextImage();
  if (undoBtn) undoBtn.onclick = () => performUndo();

  window.addEventListener("keydown", (e) => {
    // 若正在輸入框打字、彈窗開啟中、或正進行 AI 分割，忽略快捷鍵
    const activeEl = document.activeElement;
    if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.tagName === "SELECT" || activeEl.isContentEditable)) {
      return;
    }
    const modalOverlay = $("modalOverlay");
    if (modalOverlay && modalOverlay.classList.contains("active")) {
      return;
    }

    // Ctrl+Z (Undo)
    if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
      e.preventDefault();
      performUndo();
      return;
    }

    if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") {
      e.preventDefault();
      switchPrevImage();
    } else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") {
      e.preventDefault();
      switchNextImage();
    }
  });
}
initImgNavEvents();

// ---------- 選圖並畫到 canvas ----------
function selectImage(im, el, targetSegId = null) {
  if (state.segmenting) return;
  state.currentImage = im;
  document.querySelectorAll(".thumb img").forEach((i) => i.classList.remove("active"));
  if (el) el.classList.add("active");

  updateImgNavUI();

  // 切換圖片時，清空歷史紀錄並停用復原按鈕
  state.undoStack = [];
  updateUndoUI();

  state.autoSegCompleted = false;
  updateAutoSegBtn(true);

  $("drawBtn").disabled = false;
  $("textPromptInput").disabled = false;
  $("textSegBtn").disabled = false;

  const pic = new Image();
  pic.onload = async () => {
    // 防呆防競態：確認加載完成時，使用者沒有切換到其他張圖
    if (!state.currentImage || state.currentImage.id !== im.id) return;
    canvas.width = pic.width;
    canvas.height = pic.height;
    ctx.drawImage(pic, 0, 0);

    // 載入並重繪該影像先前已有的所有標記區塊
    try {
      const res = await fetch(`/api/images/${im.id}/segments`);
      if (!res.ok) return;
      const segments = await res.json();
      if (!state.currentImage || state.currentImage.id !== im.id) return;
      await redraw(segments, targetSegId);

      state.autoSegCompleted = (segments.length > 0);
      updateAutoSegBtn();
    } catch (err) {
      console.error("載入已標記區塊失敗:", err);
    }
  };
  pic.src = `/api/images/${im.id}/file`;
}

async function selectImageById(imageId, targetSegId = null) {
  if (state.segmenting) return;
  try {
    const res = await fetch(`/api/images/${imageId}`);
    if (!res.ok) return;
    const im = await res.json();

    const thumbImg = document.querySelector(`.thumb img[src*="/api/images/${im.id}/file"]`) ||
      document.querySelector(`.thumb-chk[data-id="${im.id}"]`)?.nextElementSibling;

    selectImage(im, thumbImg, targetSegId);
  } catch (err) {
    console.error("選取照片失敗:", err);
  }
}

// ---------- 自動分割整張 ----------
$("autoSegBtn").onclick = async () => {
  if (!state.currentImage || state.segmenting) return;

  const imageId = state.currentImage.id;
  setSegmentationLoading(true, "自動分割中…", true);
  startFakeProgress(10, 75);
  try {
    const data = await fetchWithProgress(
      `/api/images/${imageId}/segment`,
      { method: "POST" },
      (progressData) => {
        if (progressData.stage === "classifying" || progressData.stage === "done") {
          stopFakeProgress();
          setSegmentationLoading(true, progressData.message, true);
          updateProgressBar(progressData.progress);
        } else {
          setSegmentationLoading(true, progressData.message, true);
        }
      }
    );



    await refreshAfterSegChange();

    if (Array.isArray(data.segments) && data.segments.length > 0) {
      pushUndoAction({ type: "CREATE_SEGMENTS", segIds: data.segments.map((s) => s.id), imageId });
    }

    state.autoSegCompleted = true;
    updateAutoSegBtn();
    showToast("整張自動分割完成！", "success");
  } catch (error) {
    console.error(error);
    showToast(error instanceof Error ? error.message : "自動分割失敗", "error");
  } finally {
    setSegmentationLoading(false);
  }
};

// ---------- 自然語言分割 (YOLO-World) ----------
$("textSegBtn").onclick = async () => {
  if (!state.currentImage) return showToast("請先從左側照片庫或導航按鈕選取一張照片", "warning");
  if (state.segmenting) return;
  const promptVal = $("textPromptInput").value.trim();
  if (!promptVal) return showToast("請輸入想搜尋的物件名稱（例如：飛機）", "warning");

  const imageId = state.currentImage.id;
  setSegmentationLoading(true, `正在搜尋「${promptVal}」並進行分割…`, true);
  startFakeProgress(10, 75);

  try {
    const data = await fetchWithProgress(
      `/api/images/${imageId}/segment_text`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptVal }),
      },
      (progressData) => {
        if (progressData.stage === "segmenting" || progressData.stage === "done") {
          stopFakeProgress();
          setSegmentationLoading(true, progressData.message, true);
          updateProgressBar(progressData.progress);
        } else {
          setSegmentationLoading(true, progressData.message, true);
        }
      }
    );

    await refreshAfterSegChange();

    if (Array.isArray(data.segments) && data.segments.length > 0) {
      pushUndoAction({ type: "CREATE_SEGMENTS", segIds: data.segments.map((s) => s.id), imageId });
      showToast(`搜尋「${promptVal}」完成，找到 ${data.segments.length} 個物件`, "success");
    } else {
      showToast(`搜尋「${promptVal}」未找到符合物件`, "info");
    }
  } catch (error) {
    console.error(error);
    showToast(error instanceof Error ? error.message : "文字分割失敗", "error");
  } finally {
    setSegmentationLoading(false);
  }
};

// 支援在文字輸入框按下 Enter 鍵直接進行分割
$("textPromptInput").onkeydown = (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    $("textSegBtn").click();
  }
};

// ---------- 模式切換：手動描邊 ----------
$("drawBtn").onclick = () => {
  state.drawMode = !state.drawMode;
  $("drawBtn").classList.toggle("on", state.drawMode);
  canvas.style.cursor = state.drawMode ? "crosshair" : "crosshair";
  $("modeHint").textContent = state.drawMode
    ? "按住滑鼠沿物件邊界拖曳，放開即完成描邊"
    : "點物件做單點分割";
};

// ---------- 單點分割（一般模式：點 canvas）----------
canvas.onclick = async (e) => {
  if (!state.currentImage || state.drawMode || state.segmenting) return;
  const { x, y } = toImageXY(e);
  const imageId = state.currentImage.id;
  setSegmentationLoading(true, "單點分割中…");
  try {
    const res = await fetch(`/api/images/${imageId}/segment_point`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y }),
    });
    if (!res.ok) throw await responseError(res, "單點分割失敗");
    const seg = await res.json();
    const listRes = await fetch(`/api/images/${imageId}/segments`);
    if (!listRes.ok) throw await responseError(listRes, "讀取分割結果失敗");
    const all = await listRes.json();
    await redraw(all);
    await refreshSidebar();

    pushUndoAction({ type: "CREATE_SEGMENTS", segIds: [seg.id], imageId });
    showToast("單點分割完成", "success");
    await promptLabel(seg, true);
  } catch (error) {
    console.error(error);
    showToast(error instanceof Error ? error.message : "單點分割失敗", "error");
  } finally {
    setSegmentationLoading(false);
  }
};

// ---------- 手動描邊（draw 模式：按住拖曳描邊界）----------
canvas.onmousedown = (e) => {
  if (!state.currentImage || !state.drawMode) return;
  state.drawing = true;
  state.points = [toImageXY(e)];
};

canvas.onmousemove = (e) => {
  if (!state.drawing) return;
  const p = toImageXY(e);
  state.points.push(p);
  // 即時畫出正在描的線
  ctx.strokeStyle = "#ffd93d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const a = state.points[state.points.length - 2];
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
};

canvas.onmouseup = async () => {
  if (!state.drawing) return;
  state.drawing = false;
  const points = state.points.map((p) => [p.x, p.y]);
  state.points = [];
  if (points.length < 3) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    return redraw(all); // 點太少，取消
  }
  const res = await fetch(`/api/images/${state.currentImage.id}/segment_polygon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!res.ok) return showToast("描邊失敗：" + (await res.text()), "error");
  const seg = await res.json();
  const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
  await redraw(all);
  await refreshSidebar();

  pushUndoAction({ type: "CREATE_SEGMENTS", segIds: [seg.id], imageId: state.currentImage.id });
  showToast("手動描邊分割完成", "success");
  await promptLabel(seg, true);
};

// 行動端觸控事件繪圖支援
canvas.addEventListener("touchstart", (e) => {
  if (!state.currentImage || !state.drawMode) return;
  e.preventDefault();
  state.drawing = true;
  state.points = [toImageXY(getTouchPos(e))];
}, { passive: false });

canvas.addEventListener("touchmove", (e) => {
  if (!state.drawing) return;
  e.preventDefault();
  const p = toImageXY(getTouchPos(e));
  state.points.push(p);
  // 即時畫出正在描的線
  ctx.strokeStyle = "#ffd93d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const a = state.points[state.points.length - 2];
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}, { passive: false });

canvas.addEventListener("touchend", async (e) => {
  if (!state.drawing) return;
  e.preventDefault();
  state.drawing = false;
  const points = state.points.map((p) => [p.x, p.y]);
  state.points = [];
  if (points.length < 3) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    return redraw(all); // 點太少，取消
  }
  setSegmentationLoading(true, "儲存描邊中…");
  try {
    const res = await fetch(`/api/images/${state.currentImage.id}/segment_polygon`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points }),
    });
    if (!res.ok) return showToast("描邊失敗：" + (await res.text()), "error");
    const seg = await res.json();
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    await redraw(all);
    await refreshSidebar();

    pushUndoAction({ type: "CREATE_SEGMENTS", segIds: [seg.id], imageId: state.currentImage.id });
    showToast("手動描邊分割完成", "success");
    await promptLabel(seg, true);
  } catch (err) {
    console.error(err);
    showToast("描邊分割失敗", "error");
  } finally {
    setSegmentationLoading(false);
  }
}, { passive: false });

// ---------- 把遮罩疊回圖上：高信心綠框、低信心紅框 ----------
// highlightId：審核卡片 hover 時，把對應的框加粗變黃
async function redraw(segments, highlightId = null) {
  const currentImageId = state.currentImage ? state.currentImage.id : null;
  if (!currentImageId) return;
  state.lastSegments = segments;
  const pic = await loadImage(currentImageId);
  if (!state.currentImage || state.currentImage.id !== currentImageId) return;
  ctx.drawImage(pic, 0, 0);

  // 💡 步驟 1：先畫所有不規則的 SAM 遮罩（Mask），著色結果按 (segId, color) 快取，hover 重繪只剩 drawImage
  for (const s of segments) {
    const hi = s.id === highlightId;
    const label = s.final_label || s.predicted_label;
    const tagColor = getTagColor(label);
    const color = hi ? "#ffd166" : (s.needs_review && !s.final_label ? "#ff5470" : tagColor);
    const tinted = getTintedMask(s, color);

    if (tinted) {
      const [x, y] = s.bbox;
      ctx.save();
      ctx.globalAlpha = 0.35; // 35% 半透明色塊
      ctx.drawImage(tinted, x, y);
      ctx.restore();
    } else if (!maskImages[s.id]) {
      // 若尚未下載，則啟動非同步下載，下載成功後觸發重繪
      const img = new Image();
      img.src = `/api/segments/${s.id}/mask`;
      img.onload = () => {
        // segments 若已被較新的 redraw 取代，就不要用舊資料蓋回去
        if (state.lastSegments === segments) redraw(segments, highlightId);
      };
      img.onerror = () => {
        console.warn("Mask 下載失敗:", s.id);
        delete maskImages[s.id]; // 移除失敗紀錄，讓下次 redraw 能重試
      };
      maskImages[s.id] = img;
    }
  }

  // 💡 步驟 2：只畫文字標籤（方框已移除）
  for (const s of segments) {
    const label = s.final_label || s.predicted_label;
    if (label) {
      const [x, y] = s.bbox;
      const hi = s.id === highlightId;
      const tagColor = getTagColor(label);
      ctx.fillStyle = hi ? "#ffd166" : (s.needs_review && !s.final_label ? "#ff5470" : tagColor);
      ctx.font = "14px sans-serif";
      ctx.fillText(`${label} ${s.confidence.toFixed(2)}`, x + 2, y + 14);
    }
  }
}

// 點完一塊後問使用者類別，存成種子範例
async function promptLabel(seg, isNewSegment = false) {
  const label = prompt("這塊是什麼類別？（留空跳過）");
  if (!label) {
    if (isNewSegment) {
      await refreshAfterSegChange();
    }
    return;
  }
  await fetch(`/api/segments/${seg.id}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!isNewSegment) {
    pushUndoAction({
      type: "REVIEW_SEGMENT",
      segId: seg.id,
      prevLabel: seg.final_label || "",
      predictedLabel: seg.predicted_label || "",
      newLabel: label,
      imageId: seg.image_id
    });
  }
  await refreshAfterSegChange();
}

// 刪掉類別（連同它的種子範例與關聯遮罩片段，並回訓）
async function deleteLabel(name) {
  if (!confirm(`確定刪除類別「${name}」？其所有範例與標有此類別的遮罩將會一併刪除。`)) return;
  const res = await fetch(`/api/labels/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) return showToast("刪除失敗：" + (await res.text()), "error");
  await refreshAfterSegChange();
  showToast(`已刪除類別「${name}」及其所有遮罩`, "info");
}

// 重新命名或合併類別標籤
async function renameLabel(name) {
  const newName = prompt(`請輸入「${name}」的新類別名稱：`, name);
  if (!newName || !newName.trim() || newName.trim() === name) return;
  const targetName = newName.trim();

  let res = await fetch("/api/labels/rename", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_label: name, new_label: targetName, combine: false }),
  });

  if (res.status === 409) {
    if (confirm(`類別「${targetName}」已存在！\n是否要將「${name}」的所有標記與範例合併至「${targetName}」？`)) {
      res = await fetch("/api/labels/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_label: name, new_label: targetName, combine: true }),
      });
      if (!res.ok) return showToast("合併類別失敗：" + (await res.text()), "error");
      await refreshAfterSegChange();
      showToast(`已成功將類別「${name}」合併至「${targetName}」`, "success");
      return;
    } else {
      return;
    }
  }

  if (!res.ok) return showToast("重命名失敗：" + (await res.text()), "error");
  if (state.tagColors && state.tagColors[name]) {
    state.tagColors[targetName] = state.tagColors[name];
    delete state.tagColors[name];
    saveTagColors();
  }
  await refreshAfterSegChange();
  showToast(`已將類別「${name}」重新命名為「${targetName}」`, "success");
}

// 片段有變動後重畫目前的圖 + 更新側欄
async function refreshAfterSegChange() {
  if (state.currentImage) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    await redraw(all);
  }
  await refreshSidebar();
  await loadThumbs();
}

// ---------- 右側：統計 + 審核佇列 ----------
async function refreshSidebar() {
  const stats = await (await fetch("/api/stats")).json();
  $("stats").innerHTML = `
    總片段：${stats.total_segments}<br>
    自動接受：<b>${stats.auto_accepted}</b>（省下工時 ≈ <b>${(stats.auto_ratio * 100).toFixed(0)}%</b>）<br>
    待審：${stats.need_review} · 已審：${stats.reviewed}<br>
    範例數：${stats.num_examples} · 類別數：${stats.num_labels}`;

  if (state.mode === "engineer") {
    updateCharts(stats);
  }

  const labels = await (await fetch("/api/labels")).json();
  const ll = $("labelList");
  ll.innerHTML = "";
  if (!labels.length) ll.innerHTML = "<li class='hint'>尚未建立任何類別</li>";
  labels.forEach((name) => {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.alignItems = "center";
    li.style.justifyContent = "space-between";
    li.style.gap = "6px";

    const leftGroup = document.createElement("div");
    leftGroup.style.display = "flex";
    leftGroup.style.alignItems = "center";
    leftGroup.style.gap = "8px";
    leftGroup.style.flex = "1";
    leftGroup.style.minWidth = "0";

    const colorDot = document.createElement("span");
    colorDot.className = "tag-color-dot";
    colorDot.style.width = "14px";
    colorDot.style.height = "14px";
    colorDot.style.borderRadius = "50%";
    colorDot.style.backgroundColor = getTagColor(name);
    colorDot.style.cursor = "pointer";
    colorDot.style.flexShrink = "0";
    colorDot.style.border = "1.5px solid rgba(255, 255, 255, 0.3)";
    colorDot.style.boxShadow = "0 1px 3px rgba(0,0,0,0.3)";
    colorDot.style.display = "inline-block";
    colorDot.title = `點擊更換「${name}」的標籤色彩`;
    colorDot.onclick = (e) => openColorPickerPopover(e, name);

    const span = document.createElement("span");
    span.textContent = name;
    span.style.overflow = "hidden";
    span.style.textOverflow = "ellipsis";
    span.style.whiteSpace = "nowrap";

    leftGroup.appendChild(colorDot);
    leftGroup.appendChild(span);
    li.appendChild(leftGroup);

    const btnGroup = document.createElement("div");
    btnGroup.style.display = "flex";
    btnGroup.style.gap = "4px";
    btnGroup.style.alignItems = "center";

    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "btn-secondary";
    editBtn.style.padding = "1px 5px";
    editBtn.style.fontSize = "11px";
    editBtn.textContent = "✏️";
    editBtn.title = "重命名或合併此類別";
    editBtn.onclick = () => renameLabel(name);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "label-del";
    del.textContent = "×";
    del.title = "刪除這個類別";
    del.onclick = () => deleteLabel(name);

    btnGroup.appendChild(editBtn);
    btnGroup.appendChild(del);
    li.appendChild(btnGroup);
    ll.appendChild(li);
  });

  const queue = await (await fetch("/api/review/queue")).json();
  const ul = $("reviewQueue");
  ul.innerHTML = "";

  // 依據批次管理狀態切換 CSS 類別
  ul.classList.toggle("batch-active", state.segBatchMode);

  queue.forEach((s) => {
    const li = document.createElement("li");
    const probs = Object.entries(s.probs)
      .map(([k, v]) => `${k}:${v.toFixed(2)}`)
      .join(" · ") || "（尚無範例可分類）";
    const pred = (s.predicted_label || "").trim();
    const placeholderText = pred ? `預設：${pred}` : "正確類別";
    const confirmTitle = pred ? `按「確認」或 Enter 將直接採納預測標籤：「${pred}」` : "確認標籤";

    li.innerHTML = `
      <div class="queue-item">
        <input type="checkbox" class="seg-chk" data-id="${s.id}" />
        <canvas class="seg-thumb" width="56" height="56" title="片段預覽"></canvas>
        <div class="queue-body">
          <div>預測：${s.predicted_label ?? "—"} · 信心 ${s.confidence.toFixed(2)}</div>
          <div class="probs">${probs}</div>
          <div style="margin-top:6px; display:flex; gap:6px; align-items:center;">
            <input placeholder="${placeholderText}" data-seg="${s.id}" style="flex:1; min-width:80px;" />
            <button class="confirm" title="${confirmTitle}">確認</button>
            <button class="seg-del" title="刪掉這個切壞的片段">刪除</button>
          </div>
        </div>
      </div>`;

    // 綁定批次勾選框事件
    const chk = li.querySelector(".seg-chk");
    chk.onclick = () => {
      updateSegBatchBtnState();
    };

    // 點擊片段預覽圖：批次模式下勾選，非批次模式下自動切換並選取該張照片 (並高亮該片段)
    const thumbCanvas = li.querySelector(".seg-thumb");
    thumbCanvas.style.cursor = "pointer";
    thumbCanvas.title = "點擊自動切換並選取這張照片";
    thumbCanvas.onclick = (e) => {
      e.stopPropagation();
      if (state.segBatchMode) {
        chk.checked = !chk.checked;
        updateSegBatchBtnState();
      } else {
        selectImageById(s.image_id, s.id);
      }
    };

    // 點擊整張待審卡片空白處亦可直接切換至該照片
    li.style.cursor = "pointer";
    li.onclick = (e) => {
      if (["INPUT", "BUTTON", "LABEL"].includes(e.target.tagName)) return;
      if (state.segBatchMode) return;
      selectImageById(s.image_id, s.id);
    };

    // 縮圖：以 bbox 為中心裁一塊正方形（外擴 15% 留點上下文）
    loadImage(s.image_id).then((pic) => {
      const [x, y, w, h] = s.bbox;
      const tc = li.querySelector(".seg-thumb");
      const tctx = tc.getContext("2d");
      let size = Math.max(w, h) * 1.3;
      size = Math.min(size, pic.width, pic.height);
      const sx = Math.max(0, Math.min(pic.width - size, x + w / 2 - size / 2));
      const sy = Math.max(0, Math.min(pic.height - size, y + h / 2 - size / 2));
      tctx.drawImage(pic, sx, sy, size, size, 0, 0, tc.width, tc.height);
    }).catch(() => { });
    // 滑過卡片 → 大圖上對應的框加亮（只對目前選中的圖有效）
    li.onmouseenter = () => {
      if (state.currentImage && s.image_id === state.currentImage.id) redraw(state.lastSegments, s.id);
    };
    li.onmouseleave = () => {
      if (state.currentImage && s.image_id === state.currentImage.id) redraw(state.lastSegments);
    };
    const inputEl = li.querySelector("input[data-seg]");
    const submitReview = async () => {
      let label = inputEl.value.trim();
      if (!label && pred) {
        label = pred; // 輸入框留白時，直接採納 AI 預測標籤！
      }
      if (!label) {
        showToast("請輸入類別標籤", "warning");
        return;
      }

      // 樂觀 UI (Optimistic UI)：立刻將卡片半透明並停用，消除網路延遲的遲滯感
      li.style.opacity = "0.3";
      li.style.pointerEvents = "none";

      try {
        const res = await fetch(`/api/segments/${s.id}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label }),
        });
        if (!res.ok) throw new Error("審核失敗");
        pushUndoAction({
          type: "REVIEW_SEGMENT",
          segId: s.id,
          prevLabel: s.final_label || "",
          predictedLabel: s.predicted_label || "",
          newLabel: label,
          imageId: s.image_id
        });
        await refreshAfterSegChange();
        showToast(`標籤審核完成：「${label}」`, "success");
      } catch (err) {
        console.error(err);
        li.style.opacity = "1";
        li.style.pointerEvents = "auto";
        showToast("審核失敗，請重試", "error");
      }
    };
    li.querySelector(".confirm").onclick = submitReview;
    inputEl.onkeydown = async (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        await submitReview();
      }
    };
    li.querySelector(".seg-del").onclick = async () => {
      try {
        const res = await fetch(`/api/segments/${s.id}`, { method: "DELETE" });
        if (!res.ok) {
          return showToast("刪除失敗：" + (await res.text()), "error");
        }
        state.autoSegCompleted = false;
        updateAutoSegBtn();
        await refreshAfterSegChange();
        showToast("已刪除遮罩片段", "info");
      } catch (error) {
        console.error(error);
        showToast("刪除時發生錯誤", "error");
      }
    };
    ul.appendChild(li);
  });

  // 非批次模式下，重設勾選狀態
  if (!state.segBatchMode) {
    $("selectAllSegs").checked = false;
    updateSegBatchBtnState();
  }
}

// ---------- 匯出資料集（專案的最終產出：圖 + 遮罩 + 標籤）----------
$("exportBtn").onclick = () => {
  const fmt = $("exportFormat").value;
  // 直接導向下載端點，瀏覽器自動存檔
  window.location = `/api/export?format=${encodeURIComponent(fmt)}`;
};

// ---------- 拖曳上傳與檔案選擇相關邏輯 ----------
const dropZone = $("dropZone");
const fileInput = $("fileInput");
const selectFileBtn = $("selectFileBtn");
const fileCountHint = $("fileCountHint");

if (dropZone && fileInput && selectFileBtn && fileCountHint) {
  selectFileBtn.onclick = (e) => {
    e.stopPropagation();
    fileInput.value = "";
    fileInput.click();
  };

  dropZone.onclick = (e) => {
    if (e.target.id !== "uploadBtn" && e.target.id !== "selectFileBtn") {
      fileInput.value = "";
      fileInput.click();
    }
  };

  fileInput.onchange = () => {
    const count = fileInput.files.length;
    fileCountHint.textContent = count > 0 ? `已選取 ${count} 個檔案` : "尚未選擇檔案";
  };

  dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  };

  dropZone.ondragleave = () => {
    dropZone.classList.remove("dragover");
  };

  dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      const imageFiles = files.filter((f) => f.type.startsWith("image/"));

      if (imageFiles.length === 0) {
        alert("拖入的檔案中沒有有效的影像檔！");
        return;
      }

      if (imageFiles.length < files.length) {
        alert(`已自動忽略其中的 ${files.length - imageFiles.length} 個非影像檔案。`);
      }

      const dt = new DataTransfer();
      imageFiles.forEach((f) => dt.items.add(f));
      fileInput.files = dt.files;

      const count = fileInput.files.length;
      fileCountHint.textContent = `已拖入 ${count} 個影像檔`;
    }
  };
}

// ---------- 批次管理輔助函式與事件監聽器 ----------

function updateImgBatchBtnState() {
  if (!state.selectedImageIds) state.selectedImageIds = new Set();
  const count = state.selectedImageIds.size;
  const filtered = getFilteredImages();
  const total = filtered.length;
  const btn = $("batchDelImgsBtn");
  if (btn) {
    btn.disabled = count === 0;
    btn.textContent = count > 0 ? `確認刪除 (${count})` : "確認刪除";
  }

  const selectAll = $("selectAllImgs");
  if (selectAll) {
    selectAll.checked = total > 0 && count === total;
  }
}

function updateSegBatchBtnState() {
  const chks = document.querySelectorAll(".seg-chk");
  const checked = document.querySelectorAll(".seg-chk:checked");
  const btn = $("batchDelSegsBtn");
  if (btn) btn.disabled = checked.length === 0;

  const selectAll = $("selectAllSegs");
  if (selectAll) {
    selectAll.checked = chks.length > 0 && checked.length === chks.length;
  }
}

// 照片批次管理切換
function toggleImgBatchUI(isBatch) {
  state.imgBatchMode = isBatch;
  if (!state.selectedImageIds) state.selectedImageIds = new Set();
  state.selectedImageIds.clear();
  $("toggleImgBatchModeBtn").style.display = isBatch ? "none" : "block";
  $("batchDelImgsBtn").style.display = isBatch ? "block" : "none";
  $("cancelImgBatchBtn").style.display = isBatch ? "block" : "none";
  $("selectAllImgsLabel").style.display = isBatch ? "flex" : "none";
  updateImgBatchBtnState();
  loadThumbs();
}

$("toggleImgBatchModeBtn").onclick = () => toggleImgBatchUI(true);
$("cancelImgBatchBtn").onclick = () => toggleImgBatchUI(false);

// 待審遮罩批次管理切換
function toggleSegBatchUI(isBatch) {
  state.segBatchMode = isBatch;
  $("toggleSegBatchModeBtn").style.display = isBatch ? "none" : "block";
  $("batchDelSegsBtn").style.display = isBatch ? "block" : "none";
  $("cancelSegBatchBtn").style.display = isBatch ? "block" : "none";
  $("selectAllSegsLabel").style.display = isBatch ? "flex" : "none";
  refreshSidebar();
}

$("toggleSegBatchModeBtn").onclick = () => toggleSegBatchUI(true);
$("cancelSegBatchBtn").onclick = () => toggleSegBatchUI(false);

// 照片篩選器變更事件
const filterSelectEl = $("imgFilterSelect");
if (filterSelectEl) {
  filterSelectEl.onchange = (e) => {
    state.imgFilter = e.target.value;
    state.renderedImageCount = 0;
    const box = $("thumbs");
    if (box) box.innerHTML = "";
    renderMoreThumbs();
    updateGalleryToggleUI();
  };
}

// 照片全選 (根據目前篩選結果進行全選)
$("selectAllImgs").onchange = (e) => {
  const isChecked = e.target.checked;
  if (!state.selectedImageIds) state.selectedImageIds = new Set();
  const images = getFilteredImages();

  if (isChecked) {
    (images || []).forEach((im) => state.selectedImageIds.add(im.id));
  } else {
    state.selectedImageIds.clear();
  }

  document.querySelectorAll(".thumb-chk").forEach((chk) => {
    chk.checked = isChecked;
  });
  updateImgBatchBtnState();
};

// 待審遮罩全選
$("selectAllSegs").onchange = (e) => {
  const isChecked = e.target.checked;
  document.querySelectorAll(".seg-chk").forEach((chk) => {
    chk.checked = isChecked;
  });
  updateSegBatchBtnState();
};

// 執行照片批次刪除 (支援大批次分段刪除)
$("batchDelImgsBtn").onclick = async () => {
  if (!state.selectedImageIds || state.selectedImageIds.size === 0) return;
  const ids = Array.from(state.selectedImageIds);

  if (!confirm(`確定要批次刪除選取的 ${ids.length} 張照片嗎？這會同時清除與其相關的遮罩。`)) return;

  const btn = $("batchDelImgsBtn");
  const originalText = btn.textContent;
  btn.disabled = true;

  const DELETE_BATCH_SIZE = 500;
  let deletedTotal = 0;

  try {
    for (let i = 0; i < ids.length; i += DELETE_BATCH_SIZE) {
      const batchIds = ids.slice(i, i + DELETE_BATCH_SIZE);
      btn.textContent = `刪除中 (${Math.min(i + batchIds.length, ids.length)} / ${ids.length})…`;

      const res = await fetch("/api/images/delete_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: batchIds }),
      });

      if (!res.ok) throw new Error(await res.text());
      deletedTotal += batchIds.length;
    }

    // 如果刪除的照片包含當前選擇的圖片，清空畫布
    if (state.currentImage && ids.includes(state.currentImage.id)) {
      state.currentImage = null;
      $("autoSegBtn").disabled = true;
      $("drawBtn").disabled = true;
      $("textPromptInput").disabled = true;
      $("textSegBtn").disabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    state.selectedImageIds.clear();
    state.imgBatchMode = false;
    $("toggleImgBatchModeBtn").style.display = "block";
    $("batchDelImgsBtn").style.display = "none";
    $("cancelImgBatchBtn").style.display = "none";
    $("selectAllImgsLabel").style.display = "none";
    $("thumbs").classList.remove("batch-active");
    await loadThumbs();
    await refreshSidebar();
    showToast(`成功刪除 ${deletedTotal} 張照片`, "success");
  } catch (err) {
    showToast("批次刪除失敗: " + err.message, "error");
    await loadThumbs();
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
};

// 執行遮罩批次刪除
$("batchDelSegsBtn").onclick = async () => {
  const checked = document.querySelectorAll(".seg-chk:checked");
  const ids = Array.from(checked).map((chk) => chk.dataset.id);
  if (ids.length === 0) return;

  if (!confirm(`確定要批次刪除選取的 ${ids.length} 個遮罩片段嗎？`)) return;

  const btn = $("batchDelSegsBtn");
  const originalText = btn.textContent;
  btn.textContent = "刪除中…";
  btn.disabled = true;

  try {
    const res = await fetch("/api/segments/delete_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segment_ids: ids }),
    });

    if (!res.ok) throw new Error(await res.text());

    // 增量 DOM 移除：直接移除清單項目
    ids.forEach((id) => {
      const chk = document.querySelector(`.seg-chk[data-id="${id}"]`);
      if (chk) {
        const li = chk.closest("li");
        if (li) li.remove();
      }
    });

    // 退出批次模式並重整
    state.autoSegCompleted = false;
    updateAutoSegBtn();
    state.segBatchMode = false;
    $("toggleSegBatchModeBtn").style.display = "block";
    $("batchDelSegsBtn").style.display = "none";
    $("cancelSegBatchBtn").style.display = "none";
    $("selectAllSegsLabel").style.display = "none";
    $("reviewQueue").classList.remove("batch-active");
    await refreshAfterSegChange();
    showToast(`成功刪除 ${ids.length} 個遮罩片段`, "success");
  } catch (err) {
    showToast("批次刪除失敗: " + err.message, "error");
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
};

// ---------- 專案管理邏輯 ----------
function clearCanvasAndCurrentState() {
  state.currentImage = null;
  state.autoSegCompleted = false;
  state.lastSegments = [];
  state.points = [];
  $("autoSegBtn").disabled = true;
  $("drawBtn").disabled = true;
  $("textPromptInput").disabled = true;
  $("textSegBtn").disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function fetchProjects() {
  try {
    const res = await fetch("/api/projects");
    if (!res.ok) return;
    const data = await res.json();
    state.currentProjectId = data.active_project_id;
    const select = $("projectSelect");
    if (!select) return;
    select.innerHTML = "";
    (data.projects || []).forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      if (p.id === data.active_project_id) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("載入專案清單失敗:", err);
  }
}

async function selectProject(projectId) {
  try {
    const res = await fetch(`/api/projects/${projectId}/select`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    state.currentProjectId = projectId;
    clearCanvasAndCurrentState();
    await loadThumbs();
    await refreshSidebar();
    if (state.mode === "engineer") await loadParameters();
  } catch (err) {
    alert("切換專案失敗: " + err.message);
  }
}

function showModal({ title, desc = "", showInput = false, inputLabel = "", inputValue = "", confirmText = "確認", isDanger = false }) {
  return new Promise((resolve) => {
    const overlay = $("modalOverlay");
    const titleEl = $("modalTitle");
    const descEl = $("modalDesc");
    const inputGroup = $("modalInputGroup");
    const inputLabelEl = $("modalInputLabel");
    const inputEl = $("modalInput");
    const cancelBtn = $("modalCancelBtn");
    const confirmBtn = $("modalConfirmBtn");

    titleEl.textContent = title;
    if (desc) {
      descEl.textContent = desc;
      descEl.style.display = "block";
    } else {
      descEl.style.display = "none";
    }

    if (showInput) {
      inputLabelEl.textContent = inputLabel || "名稱";
      inputEl.value = inputValue || "";
      inputGroup.style.display = "flex";
    } else {
      inputGroup.style.display = "none";
    }

    confirmBtn.textContent = confirmText;
    confirmBtn.className = isDanger ? "btn-danger" : "btn-primary";

    overlay.classList.add("active");

    if (showInput) {
      setTimeout(() => {
        inputEl.focus();
        inputEl.select();
      }, 50);
    }

    const close = (val) => {
      overlay.classList.remove("active");
      cancelBtn.onclick = null;
      confirmBtn.onclick = null;
      inputEl.onkeydown = null;
      resolve(val);
    };

    cancelBtn.onclick = () => close(null);
    confirmBtn.onclick = () => close(showInput ? inputEl.value : true);

    if (showInput) {
      inputEl.onkeydown = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          close(inputEl.value);
        } else if (e.key === "Escape") {
          close(null);
        }
      };
    }
  });
}

function initProjectControls() {
  const select = $("projectSelect");
  if (select) {
    select.onchange = async () => {
      const selectedId = select.value;
      if (selectedId && selectedId !== state.currentProjectId) {
        await selectProject(selectedId);
      }
    };
  }

  const createBtn = $("createProjectBtn");
  if (createBtn) {
    createBtn.onclick = async () => {
      const name = await showModal({
        title: "✨ 建立新專案",
        showInput: true,
        inputLabel: "專案名稱",
        inputValue: "新專案",
        confirmText: "建立專案"
      });
      if (name === null || !name.trim()) return;
      try {
        const res = await fetch("/api/projects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim(), mode: state.mode })
        });
        if (!res.ok) throw new Error(await res.text());
        clearCanvasAndCurrentState();
        await fetchProjects();
        await loadThumbs();
        await refreshSidebar();
      } catch (err) {
        alert("建立專案失敗: " + err.message);
      }
    };
  }

  const renameBtn = $("renameProjectBtn");
  if (renameBtn) {
    renameBtn.onclick = async () => {
      if (!state.currentProjectId) return;
      const selectEl = $("projectSelect");
      const currentOpt = selectEl ? selectEl.selectedOptions[0] : null;
      const oldName = currentOpt ? currentOpt.textContent : "";
      const newName = await showModal({
        title: "✏️ 重新命名專案",
        showInput: true,
        inputLabel: "新的專案名稱",
        inputValue: oldName,
        confirmText: "儲存修改"
      });
      if (newName === null || !newName.trim() || newName.trim() === oldName) return;
      try {
        const res = await fetch(`/api/projects/${state.currentProjectId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: newName.trim() })
        });
        if (!res.ok) throw new Error(await res.text());
        await fetchProjects();
      } catch (err) {
        alert("修改專案名稱失敗: " + err.message);
      }
    };
  }

  const deleteBtn = $("deleteProjectBtn");
  if (deleteBtn) {
    deleteBtn.onclick = async () => {
      if (!state.currentProjectId) return;
      const selectEl = $("projectSelect");
      const currentOpt = selectEl ? selectEl.selectedOptions[0] : null;
      const projName = currentOpt ? currentOpt.textContent : "此專案";
      const confirmed = await showModal({
        title: "⚠️ 確定要刪除專案嗎？",
        desc: `專案「${projName}」內的所有照片、遮罩檔及分類成果將會被永久刪除且無法復原！`,
        confirmText: "確認刪除",
        isDanger: true
      });
      if (!confirmed) return;

      try {
        const res = await fetch(`/api/projects/${state.currentProjectId}`, { method: "DELETE" });
        if (!res.ok) throw new Error(await res.text());
        clearCanvasAndCurrentState();
        await fetchProjects();
        await loadThumbs();
        await refreshSidebar();
      } catch (err) {
        alert("刪除專案失敗: " + err.message);
      }
    };
  }
}

// 初始載入
async function initApp() {
  initProjectControls();
  await fetchProjects();
  await loadThumbs();
}
initApp();

// ---------- 雙模式與參數調整初始化 ----------
state.mode = localStorage.getItem("mode") || "layman";

let laborSavingChartInstance = null;
let categoryDistributionChartInstance = null;
let reviewProgressChartInstance = null;

function getCssVar(varName, fallback = '') {
  if (typeof window === "undefined" || !document.documentElement) return fallback;
  const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  return val || fallback;
}

function applyMode(mode) {
  state.mode = mode;
  localStorage.setItem("mode", mode);

  const isEng = mode === "engineer";
  $("laymanModeBtn").classList.toggle("active", !isEng);
  $("engineerModeBtn").classList.toggle("active", isEng);

  const wrapper = document.querySelector(".mode-switch-wrapper");
  if (wrapper) {
    wrapper.classList.toggle("eng-active", isEng);
  }

  document.body.classList.toggle("layman-mode", !isEng);

  document.querySelectorAll(".engineer-only").forEach(el => {
    el.classList.toggle("show", isEng);
  });

  if (isEng) {
    loadParameters();
  }
  refreshSidebar();
}

function updateSliderFill(sliderEl) {
  if (!sliderEl) return;
  const min = parseFloat(sliderEl.min) || 0;
  const max = parseFloat(sliderEl.max) || 1;
  const val = parseFloat(sliderEl.value) || 0;
  const percent = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
  sliderEl.style.background = `linear-gradient(90deg, #4f9cff 0%, #36d399 ${percent}%, #1b2636 ${percent}%, #1b2636 100%)`;
}

function updateConfThresholdDisplay(val) {
  const num = Number(val);
  const inputVal = $("confThresholdValue");
  if (inputVal && document.activeElement !== inputVal) {
    inputVal.value = isNaN(num) ? "" : num.toFixed(2);
  }
  updateSliderFill($("confThresholdInput"));
  const hint = $("confThresholdHint");
  if (hint) {
    if (num === 0) {
      hint.textContent = "（全自動通關）";
      hint.style.color = getCssVar("--ok", "#36d399");
    } else if (num === 1) {
      hint.textContent = "（全人工審核）";
      hint.style.color = getCssVar("--accent", "#4f9cff");
    } else {
      hint.textContent = "";
    }
  }
}

async function loadParameters() {
  try {
    const res = await fetch("/api/parameters");
    if (res.ok) {
      const data = await res.json();
      $("confThresholdInput").value = data.confidence_threshold;
      updateConfThresholdDisplay(data.confidence_threshold);
      $("yoloConfInput").value = data.yolo_world_confidence;
      $("yoloConfValue").value = Number(data.yolo_world_confidence).toFixed(2);
      updateSliderFill($("yoloConfInput"));
      if (data.yolo_imgsz != null) {
        const imgszEl = $("yoloImgszInput");
        // 後端若回傳下拉選單沒有的值（例如用環境變數設的），補一個選項再選取
        if (!Array.from(imgszEl.options).some((o) => o.value === String(data.yolo_imgsz))) {
          imgszEl.add(new Option(String(data.yolo_imgsz), String(data.yolo_imgsz)));
        }
        imgszEl.value = String(data.yolo_imgsz);
      }
    }
  } catch (err) {
    console.error("載入參數失敗:", err);
  }
}

function updateCenterText(chartCanvasId, text, color) {
  const canvas = $(chartCanvasId);
  if (!canvas) return;
  const wrapper = canvas.parentElement;
  let overlay = wrapper.querySelector(".chart-center-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "chart-center-overlay";
    overlay.style.position = "absolute";
    overlay.style.top = "50%";
    overlay.style.left = "50%";
    overlay.style.transform = "translate(-50%, -50%)";
    overlay.style.fontSize = "20px";
    overlay.style.fontWeight = "bold";
    overlay.style.pointerEvents = "none";
    wrapper.appendChild(overlay);
  }
  overlay.textContent = text;
  overlay.style.color = color || getCssVar("--text", "#f3f4f6");
}

function updateCharts(stats) {
  if (typeof Chart === "undefined") return;

  // 1. 自動過審 vs 手動標籤
  const laborSavingCtx = $("laborSavingChart").getContext("2d");
  const totalLabeled = stats.auto_accepted + stats.reviewed;
  const autoRatioPercent = totalLabeled ? (stats.auto_accepted / totalLabeled * 100).toFixed(0) + "%" : "0%";
  $("laborSavingSub").innerHTML = `自動過審: <b>${stats.auto_accepted}</b> / 手動標籤: <b>${stats.reviewed}</b>`;

  if (laborSavingChartInstance) {
    laborSavingChartInstance.data.datasets[0].data = [stats.auto_accepted, stats.reviewed];
    laborSavingChartInstance.update();
  } else {
    laborSavingChartInstance = new Chart(laborSavingCtx, {
      type: 'doughnut',
      data: {
        labels: ['自動過審', '手動標籤'],
        datasets: [{
          data: [stats.auto_accepted, stats.reviewed],
          backgroundColor: ['#36d399', '#4f9cff'],
          borderWidth: 1,
          borderColor: '#28323f'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        cutout: '75%'
      }
    });
  }
  updateCenterText("laborSavingChart", autoRatioPercent, getCssVar("--ok", "#36d399"));

  // 2. 類別分布
  const categoryCtx = $("categoryDistributionChart").getContext("2d");
  const labelCounts = stats.label_counts || {};
  const labels = Object.keys(labelCounts);
  const counts = Object.values(labelCounts);
  $("categorySub").innerHTML = `已建立類別數: <b>${stats.num_labels}</b>`;

  const colors = labels.map((lbl) => getTagColor(lbl));

  const cType = state.categoryChartType || "bar";
  if (categoryDistributionChartInstance && categoryDistributionChartInstance.config.type !== cType) {
    categoryDistributionChartInstance.destroy();
    categoryDistributionChartInstance = null;
  }

  if (cType === "bar") {
    if (categoryDistributionChartInstance) {
      categoryDistributionChartInstance.data.labels = labels;
      categoryDistributionChartInstance.data.datasets[0].data = counts;
      categoryDistributionChartInstance.data.datasets[0].backgroundColor = colors;
      categoryDistributionChartInstance.update();
    } else {
      categoryDistributionChartInstance = new Chart(categoryCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: '數量',
            data: counts,
            backgroundColor: colors,
            borderWidth: 1,
            borderColor: 'var(--panel)'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: 'var(--muted)', stepSize: 1 }
            },
            x: {
              grid: { display: false },
              ticks: { color: 'var(--muted)' }
            }
          }
        }
      });
    }
  } else {
    // 圓餅圖
    if (categoryDistributionChartInstance) {
      categoryDistributionChartInstance.data.labels = labels;
      categoryDistributionChartInstance.data.datasets[0].data = counts;
      categoryDistributionChartInstance.data.datasets[0].backgroundColor = colors;
      categoryDistributionChartInstance.update();
    } else {
      categoryDistributionChartInstance = new Chart(categoryCtx, {
        type: 'pie',
        data: {
          labels: labels,
          datasets: [{
            data: counts,
            backgroundColor: colors,
            borderWidth: 1,
            borderColor: '#28323f'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } }
        }
      });
    }
  }

  // 更新類別分布圖的圖例
  const legendDiv = $("categoryLegend");
  if (legendDiv) {
    legendDiv.innerHTML = "";
    labels.forEach((label, idx) => {
      const color = colors[idx];
      const span = document.createElement("span");
      span.style.display = "flex";
      span.style.alignItems = "center";
      span.style.gap = "4px";
      span.innerHTML = `<span style="display: inline-block; width: 10px; height: 10px; background-color: ${color}; border-radius: 2px;"></span>${label}`;
      legendDiv.appendChild(span);
    });
  }

  // 3. 審核進度
  const reviewCtx = $("reviewProgressChart").getContext("2d");
  const totalToReview = stats.reviewed + stats.need_review;
  const progressRatio = totalToReview ? stats.reviewed / totalToReview : 0.0;
  const progressPercentText = (progressRatio * 100).toFixed(0) + "%";
  $("reviewSub").innerHTML = `已審核: <b>${stats.reviewed}</b> / 待審: <b>${stats.need_review}</b>`;

  if (reviewProgressChartInstance) {
    reviewProgressChartInstance.data.datasets[0].data = [stats.reviewed, stats.need_review];
    reviewProgressChartInstance.update();
  } else {
    reviewProgressChartInstance = new Chart(reviewCtx, {
      type: 'doughnut',
      data: {
        labels: ['已審核', '待人工審核'],
        datasets: [{
          data: [stats.reviewed, stats.need_review],
          backgroundColor: ['#4f9cff', '#ff5470'],
          borderWidth: 1,
          borderColor: '#28323f'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        cutout: '75%'
      }
    });
  }
  updateCenterText("reviewProgressChart", progressPercentText, "var(--accent)");
}

const switchWrapper = document.querySelector(".mode-switch-wrapper");
if (switchWrapper) {
  switchWrapper.onclick = () => {
    const mode = state.mode === "layman" ? "engineer" : "layman";
    applyMode(mode);
  };
}

// 深色 / 淺色主題切換控制 (Dark / Light Mode Toggle)
function initThemeToggle() {
  const themeBtn = document.getElementById("themeToggleBtn");
  const savedTheme = localStorage.getItem("app_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);

  if (themeBtn) {
    themeBtn.onclick = () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
      const nextTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("app_theme", nextTheme);
    };
  }
}
initThemeToggle();

function bindNumericInputGuard(inputEl, sliderEl, minVal, maxVal, onUpdate) {
  if (!inputEl) return;

  let lastValidValue = parseFloat(sliderEl ? sliderEl.value : inputEl.value);
  if (isNaN(lastValidValue)) lastValidValue = minVal;

  inputEl.addEventListener("keydown", (e) => {
    if (["-", "+", "e", "E"].includes(e.key)) {
      e.preventDefault();
    }
  });

  inputEl.addEventListener("focus", () => {
    const curVal = parseFloat(sliderEl ? sliderEl.value : inputEl.value);
    if (!isNaN(curVal) && curVal >= minVal && curVal <= maxVal) {
      lastValidValue = curVal;
    }
  });

  inputEl.addEventListener("input", () => {
    let raw = inputEl.value;
    if (raw === "") return;
    let num = parseFloat(raw);
    if (isNaN(num)) return;

    if (num > maxVal) {
      num = maxVal;
      inputEl.value = num;
    }
    if (num < 0) {
      num = 0;
      inputEl.value = num;
    }

    if (num >= minVal && num <= maxVal) {
      lastValidValue = num;
      onUpdate(num);
    }
  });

  inputEl.addEventListener("blur", () => {
    let raw = inputEl.value;
    let num = parseFloat(raw);
    if (isNaN(num) || num < minVal || num > maxVal) {
      num = lastValidValue;
    } else {
      lastValidValue = num;
    }
    inputEl.value = num.toFixed(2);
    onUpdate(num);
  });
}

$("confThresholdInput").oninput = (e) => {
  updateConfThresholdDisplay(e.target.value);
};

bindNumericInputGuard($("confThresholdValue"), $("confThresholdInput"), 0.0, 1.0, (num) => {
  const clamped = Math.max(0.0, Math.min(1.0, num));
  $("confThresholdInput").value = clamped;
  updateConfThresholdDisplay(clamped);
});

$("yoloConfInput").oninput = (e) => {
  const val = Number(e.target.value).toFixed(2);
  if (document.activeElement !== $("yoloConfValue")) {
    $("yoloConfValue").value = val;
  }
  updateSliderFill(e.target);
};

bindNumericInputGuard($("yoloConfValue"), $("yoloConfInput"), 0.1, 1.0, (num) => {
  const clamped = Math.max(0.1, Math.min(1.0, num));
  $("yoloConfInput").value = clamped;
  updateSliderFill($("yoloConfInput"));
});

$("saveParamsBtn").onclick = async () => {
  let confidence_threshold = parseFloat($("confThresholdValue").value);
  if (isNaN(confidence_threshold)) {
    confidence_threshold = parseFloat($("confThresholdInput").value);
  }
  let yolo_world_confidence = parseFloat($("yoloConfValue").value);
  if (isNaN(yolo_world_confidence)) {
    yolo_world_confidence = parseFloat($("yoloConfInput").value);
  }
  const yolo_imgsz = parseInt($("yoloImgszInput").value, 10);

  try {
    const res = await fetch("/api/parameters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence_threshold, yolo_world_confidence, yolo_imgsz })
    });
    if (res.ok) {
      alert("參數儲存與重新預測成功！");
      await refreshAfterSegChange();
    } else {
      alert("儲存失敗：" + (await res.text()));
    }
  } catch (err) {
    console.error(err);
    alert("儲存時發生錯誤");
  }
};

// 初始套用模式
state.categoryChartType = "bar"; // 預設為長條圖

const toggleBtn = $("toggleCategoryChartTypeBtn");
if (toggleBtn) {
  toggleBtn.onclick = () => {
    state.categoryChartType = state.categoryChartType === "bar" ? "pie" : "bar";
    toggleBtn.textContent = state.categoryChartType === "bar" ? "切換為圓餅圖" : "切換為長條圖";
    refreshSidebar();
  };
}

// ---------- 資料集匯出預覽 Modal (Dataset Export Preview Modal) ----------
function initExportPreviewModal() {
  const exportBtn = $("exportBtn");
  const modal = $("exportPreviewModal");
  const closeBtn = $("closeExportPreviewModalBtn");
  const cancelBtn = $("cancelExportPreviewBtn");
  const confirmBtn = $("confirmExportBtn");
  const formatSelect = $("exportFormat");

  if (!exportBtn || !modal) return;

  const toggleExportModal = (show) => {
    const isVisible = show !== undefined ? Boolean(show) : modal.style.display !== "flex";
    modal.style.display = isVisible ? "flex" : "none";
    if (isVisible) modal.classList.add("active");
    else modal.classList.remove("active");
  };

  if (closeBtn) closeBtn.onclick = () => toggleExportModal(false);
  if (cancelBtn) cancelBtn.onclick = () => toggleExportModal(false);
  modal.onclick = (e) => {
    if (e.target === modal) toggleExportModal(false);
  };

  exportBtn.onclick = async () => {
    const fmt = formatSelect ? formatSelect.value : "coco";
    try {
      const res = await fetch(`/api/export/preview?format=${fmt}`);
      if (!res.ok) throw new Error("讀取匯出資料預覽失敗");
      const data = await res.json();

      // 更新格式標籤與說明
      const fmtTitles = {
        coco: "COCO Instance Segmentation",
        yolo: "YOLOv8-seg / YOLO11 Format",
        mask: "Semantic Mask PNG Format"
      };
      const fmtDescs = {
        coco: "相容於 Detectron2, MMDetection, YOLOv8 格式",
        yolo: "包含 txt 多邊形點座標與 data.yaml 訓練設定檔",
        mask: "包含二值化語意分割 PNG 圖片檔與 classes.txt 對照表"
      };

      $("exportFmtTag").textContent = fmtTitles[fmt] || fmt.toUpperCase();
      $("exportFmtDesc").textContent = fmtDescs[fmt] || "";

      // 更新統計卡片數據
      $("exportStatImages").textContent = data.annotated_images || 0;
      $("exportStatSegments").textContent = data.total_segments || 0;
      $("exportStatLabels").textContent = data.num_labels || 0;

      // 更新類別分佈表格
      const tbody = $("exportClassTbody");
      if (tbody) {
        tbody.innerHTML = "";
        const labelCounts = data.label_counts || {};
        const totalSegs = data.total_segments || 1;
        const labels = Object.keys(labelCounts);

        if (labels.length === 0) {
          tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:var(--muted); padding:10px;">尚未有已完成標記的類別數據</td></tr>`;
        } else {
          labels.forEach((lbl) => {
            const cnt = labelCounts[lbl];
            const pct = ((cnt / totalSegs) * 100).toFixed(1);
            const color = getTagColor(lbl);
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td>
                <span class="tag-color-dot" style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:${color}; margin-right:6px;"></span>
                <b>${lbl}</b>
              </td>
              <td>${cnt} 個遮罩</td>
              <td>${pct}%</td>
            `;
            tbody.appendChild(tr);
          });
        }
      }

      // 更新 ZIP 目錄結構預覽
      const treeBox = $("exportFileTree");
      if (treeBox) {
        const treeLines = (data.format_trees && data.format_trees[fmt]) || [];
        treeBox.textContent = treeLines.join("\n");
      }

      toggleExportModal(true);
    } catch (err) {
      console.error(err);
      showToast("載入匯出預覽失敗", "error");
    }
  };

  if (confirmBtn) {
    confirmBtn.onclick = () => {
      const fmt = formatSelect ? formatSelect.value : "coco";
      toggleExportModal(false);
      window.location.href = `/api/export?format=${fmt}`;
      showToast(`已開始下載 ${fmt.toUpperCase()} 格式資料集 ZIP 壓縮檔`, "success");
    };
  }
}

initExportPreviewModal();
applyMode(state.mode);

// ==========================================
// ⚡ 多圖批次文字標註 (Batch Auto-Labeling with Prompt)
// ==========================================
const batchTextBtn = $("batchTextSegBtn");
if (batchTextBtn) {
  batchTextBtn.onclick = async () => {
    const promptInput = $("textPromptInput");
    const prompt = promptInput ? promptInput.value.trim() : "";
    if (!prompt) {
      alert("請先在搜尋框輸入要批次標註的物件名稱（例如：strawberry）！");
      if (promptInput) promptInput.focus();
      return;
    }

    // 🔍 精確抓取網頁圖庫中所有打藍色勾勾 (.thumb-chk:checked) 的圖片 ID
    const checkedChks = document.querySelectorAll(".thumb-chk:checked");
    const selectedIds = Array.from(checkedChks).map(chk => chk.dataset.id).filter(Boolean);
    const hasSelected = selectedIds.length > 0;
    const targetText = hasSelected ? `已勾選的 ${selectedIds.length} 張圖片` : "圖庫內的所有圖片";

    if (!confirm(`確定要對 ${targetText}，統一套用 Prompt: '${prompt}' 進行 YOLO-World 批次自動標註嗎？`)) {
      return;
    }

    batchTextBtn.disabled = true;
    batchTextBtn.textContent = "⚡ 標註處理中...";
    setSegmentationLoading(true, `正在對 ${targetText} 進行批次標註 (${prompt})...`, true);

    try {
      const res = await fetch("/api/images/batch_segment_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          image_ids: hasSelected ? selectedIds : []
        })
      });

      if (!res.ok) {
        const err = await res.json();
        alert("批次標註失敗：" + (err.error || "未知錯誤"));
        setSegmentationLoading(false);
        batchTextBtn.disabled = false;
        batchTextBtn.textContent = "⚡ 批次多圖標註";
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let successCnt = 0;
      let failCnt = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.event === "progress") {
              const pct = Math.round((msg.current / msg.total) * 100);
              updateProgressBar(pct);
              setSegmentationLoading(true, `[${msg.current}/${msg.total}] 正在分析 ${msg.filename} (Prompt: '${prompt}')...`, true);
            } else if (msg.event === "image_done") {
              successCnt++;
            } else if (msg.event === "image_error") {
              failCnt++;
              console.warn(`圖片 ${msg.filename} 標註失敗: ${msg.message}`);
            } else if (msg.event === "done") {
              const successNum = msg.success_images !== undefined ? msg.success_images : successCnt;
              const totalNum = msg.total_images !== undefined ? msg.total_images : (successNum + failCnt);
              const failedNum = totalNum - successNum;

              if (successNum === totalNum && totalNum > 0) {
                alert(`🎉 批次標註全數成功！共成功標註 ${successNum} 張圖片！`);
              } else if (successNum === 0) {
                alert(`❌ 批次標註全數失敗！共 ${totalNum} 張圖片皆無法處理（請檢查模型或圖片）。`);
              } else {
                alert(`⚠️ 批次標註部分完成！成功：${successNum} 張，失敗：${failedNum} 張。`);
              }
            } else if (msg.event === "error") {
              alert("批次標註發生錯誤：" + msg.message);
            }
          } catch (e) {
            console.error("解析進度失敗", e);
          }
        }
      }
      await refreshSidebar();
      if (state.currentImage && state.currentImage.id) {
        await selectImageById(state.currentImage.id);
      }
    } catch (err) {
      console.error("批次標註異常:", err);
      alert("批次標註通訊失敗：" + err.message);
    } finally {
      setSegmentationLoading(false);
      batchTextBtn.disabled = false;
      batchTextBtn.textContent = "⚡ 批次多圖標註";
    }
  };
}
