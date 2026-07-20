// 智慧分割標記助手 — 前端最小可用版
// 流程：上傳 → 選圖 → 自動/單點分割 → 標種子 → 審核紅色低信心片段 → 看統計
"use strict";

const state = {
  currentImage: null,
  drawMode: false,
  drawing: false,
  points: [],
  segmenting: false,
  lastSegments: [],   // 目前畫布上的片段，供審核卡片 hover 加亮用
  imgBatchMode: false,
  segBatchMode: false,
<<<<<<< HEAD
  jobRunning: false,  // 有批量分割進行中 → 停用批量按鈕（後端也會擋 409）
=======
  autoSegCompleted: false,
>>>>>>> origin/main
};
const $ = (id) => document.getElementById(id);

// 原圖快取：縮圖與重繪共用同一份，避免重複下載解碼
const imageCache = new Map();

// image_id → 檔名，給批量分割失敗清單顯示用（loadThumbs 時更新）
const imageNames = new Map();
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
function updateAutoSegBtn() {
  const btn = $("autoSegBtn");
  if (!btn) return;
  if (!state.currentImage) {
    btn.disabled = true;
    btn.textContent = "自動分割";
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
  $("textPromptInput").disabled = active || !state.currentImage;
  $("textSegBtn").disabled = active || !state.currentImage;
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

$("uploadBtn").onclick = async () => {
  const files = $("fileInput").files;
  if (!files.length) return alert("先選檔案");
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const res = await fetch("/api/images", { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    let errorMsg = text;
    try {
      const json = JSON.parse(text);
      errorMsg = json.error || json.message || text;
    } catch (e) { }
    return alert("上傳失敗：" + errorMsg);
  }
  // 清除選擇的檔案與提示
  $("fileInput").value = "";
  if ($("fileCountHint")) {
    $("fileCountHint").textContent = "尚未選擇檔案";
  }
  await loadThumbs();
};

async function loadThumbs() {
  const imgs = await (await fetch("/api/images")).json();
  const box = $("thumbs");
  box.innerHTML = "";
  
  // 根據批次管理狀態切換 CSS class
  box.classList.toggle("batch-active", state.imgBatchMode);

  imageNames.clear();
  imgs.forEach((im) => {
    imageNames.set(im.id, im.filename);
    const wrap = document.createElement("div");
    wrap.className = "thumb";

    const el = document.createElement("img");
    el.src = `/api/images/${im.id}/file`;
    el.title = im.filename;
    el.onclick = () => {
      if (state.imgBatchMode) {
        chk.checked = !chk.checked;
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
    chk.onclick = (e) => {
      e.stopPropagation();
      updateImgBatchBtnState();
    };

    const del = document.createElement("button");
    del.className = "thumb-del";
    del.textContent = "×";
    del.title = "刪除這張";
    del.onclick = (e) => { e.stopPropagation(); deleteImage(im); };

    wrap.append(chk, el, del);
    box.appendChild(wrap);
  });

  // 非批次模式下，重設勾選狀態
  if (!state.imgBatchMode) {
    $("selectAllImgs").checked = false;
    updateImgBatchBtnState();
  }
}

async function deleteImage(im) {
  if (!confirm(`確定刪除「${im.filename}」？連同它的遮罩會一起清掉。`)) return;
  const res = await fetch(`/api/images/${im.id}`, { method: "DELETE" });
  if (!res.ok) return alert("刪除失敗：" + (await res.text()));
  // 若刪的是目前選中的圖，清空畫布
  if (state.currentImage && state.currentImage.id === im.id) {
    state.currentImage = null;
    updateAutoSegBtn();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  await loadThumbs();
  await refreshSidebar();
}

// ---------- 選圖並畫到 canvas ----------
function selectImage(im, el) {
  if (state.segmenting) return;
  state.currentImage = im;
  document.querySelectorAll(".thumb img").forEach((i) => i.classList.remove("active"));
  el.classList.add("active");
  
  state.autoSegCompleted = false;
  updateAutoSegBtn();

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
      await redraw(segments);

      state.autoSegCompleted = (segments.length > 0);
      updateAutoSegBtn();
    } catch (err) {
      console.error("載入已標記區塊失敗:", err);
    }
  };
  pic.src = `/api/images/${im.id}/file`;
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



    await redraw(data.segments);
    await refreshSidebar();

    state.autoSegCompleted = true;
    updateAutoSegBtn();
  } catch (error) {
    console.error(error);
    alert(error instanceof Error ? error.message : "自動分割失敗");
  } finally {
    setSegmentationLoading(false);
  }
};

// ---------- 自然語言分割 (YOLO-World) ----------
$("textSegBtn").onclick = async () => {
  const promptVal = $("textPromptInput").value.trim();
  if (!promptVal) return alert("請輸入想搜尋的物件名稱（例如：飛機）");
  if (!state.currentImage || state.segmenting) return;

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

    await redraw(data.segments);
    await refreshSidebar();
  } catch (error) {
    console.error(error);
    alert(error instanceof Error ? error.message : "文字分割失敗");
  } finally {
    setSegmentationLoading(false);
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
    promptLabel(seg);
  } catch (error) {
    console.error(error);
    alert(error instanceof Error ? error.message : "單點分割失敗");
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
  if (!res.ok) return alert("描邊失敗：" + (await res.text()));
  const seg = await res.json();
  const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
  await redraw(all);
  await refreshSidebar();
  promptLabel(seg);
};

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
    const color = hi ? "#ffd166" : (s.needs_review ? "#ff5470" : "#36d399");
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
    if (s.final_label) {
      const [x, y] = s.bbox;
      const hi = s.id === highlightId;
      ctx.fillStyle = hi ? "#ffd166" : (s.needs_review ? "#ff5470" : "#36d399");
      ctx.font = "14px sans-serif";
      ctx.fillText(`${s.final_label} ${s.confidence.toFixed(2)}`, x + 2, y + 14);
    }
  }
}

// 點完一塊後問使用者類別，存成種子範例
async function promptLabel(seg) {
  const label = prompt("這塊是什麼類別？（留空跳過）");
  if (!label) return;
  await fetch(`/api/segments/${seg.id}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
  await redraw(all);
  await refreshSidebar();
}

// 刪掉建錯的類別（連同它的種子範例，並回訓）
async function deleteLabel(name) {
  if (!confirm(`刪除類別「${name}」？它的種子範例會一起清掉。`)) return;
  const res = await fetch(`/api/labels/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) return alert("刪除失敗：" + (await res.text()));
  if (state.currentImage) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    await redraw(all);
  }
  await refreshSidebar();
}

// 片段有變動後重畫目前的圖 + 更新側欄
async function refreshAfterSegChange() {
  if (state.currentImage) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    await redraw(all);
  }
  await refreshSidebar();
}

// ---------- 右側：統計 + 審核佇列 ----------
async function refreshSidebar() {
  const stats = await (await fetch("/api/stats")).json();
  $("stats").innerHTML = `
    總片段：${stats.total_segments}<br>
    自動接受：<b>${stats.auto_accepted}</b>（省下工時 ≈ <b>${(stats.auto_ratio * 100).toFixed(0)}%</b>）<br>
    待審：${stats.need_review} · 已審：${stats.reviewed}<br>
    範例數：${stats.num_examples} · 類別數：${stats.num_labels}`;

  const labels = await (await fetch("/api/labels")).json();
  const ll = $("labelList");
  ll.innerHTML = "";
  if (!labels.length) ll.innerHTML = "<li class='hint'>尚未建立任何類別</li>";
  labels.forEach((name) => {
    const li = document.createElement("li");
    li.textContent = name;
    const del = document.createElement("button");
    del.className = "label-del";
    del.textContent = "×";
    del.title = "刪除這個類別";
    del.onclick = () => deleteLabel(name);
    li.appendChild(del);
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
    li.innerHTML = `
      <div class="queue-item">
        <input type="checkbox" class="seg-chk" data-id="${s.id}" />
        <canvas class="seg-thumb" width="56" height="56" title="片段預覽"></canvas>
        <div class="queue-body">
          <div>預測：${s.predicted_label ?? "—"} · 信心 ${s.confidence.toFixed(2)}</div>
          <div class="probs">${probs}</div>
          <div style="margin-top:6px">
            <input placeholder="正確類別" data-seg="${s.id}" />
            <button class="confirm">確認</button>
            <button class="seg-del" title="刪掉這個切壞的片段">刪除</button>
          </div>
        </div>
      </div>`;
    
    // 綁定批次勾選框事件
    const chk = li.querySelector(".seg-chk");
    chk.onclick = () => {
      updateSegBatchBtnState();
    };

    // 點擊片段預覽圖也可切換勾選狀態
    const thumbCanvas = li.querySelector(".seg-thumb");
    thumbCanvas.onclick = () => {
      if (state.segBatchMode) {
        chk.checked = !chk.checked;
        updateSegBatchBtnState();
      }
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
    li.querySelector(".confirm").onclick = async () => {
      const label = li.querySelector("input[data-seg]").value.trim();
      if (!label) return;
      await fetch(`/api/segments/${s.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      await refreshAfterSegChange();
    };
    li.querySelector(".seg-del").onclick = async () => {
      try {
        const res = await fetch(`/api/segments/${s.id}`, { method: "DELETE" });
        if (!res.ok) {
          return alert("刪除失敗：" + (await res.text()));
        }
        state.autoSegCompleted = false;
        updateAutoSegBtn();
        await refreshAfterSegChange();
      } catch (error) {
        console.error(error);
        alert("刪除時發生錯誤");
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
  selectFileBtn.onclick = () => fileInput.click();

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
      fileInput.files = e.dataTransfer.files;
      const count = fileInput.files.length;
      fileCountHint.textContent = `已拖入 ${count} 個檔案`;
    }
  };
}

// ---------- 批次管理輔助函式與事件監聽器 ----------

function updateImgBatchBtnState() {
  const chks = document.querySelectorAll(".thumb-chk");
  const checked = document.querySelectorAll(".thumb-chk:checked");
  const btn = $("batchDelImgsBtn");
  if (btn) btn.disabled = checked.length === 0;
  const segBtn = $("batchSegImgsBtn");
  if (segBtn) segBtn.disabled = checked.length === 0 || state.jobRunning;

  const selectAll = $("selectAllImgs");
  if (selectAll) {
    selectAll.checked = chks.length > 0 && checked.length === chks.length;
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
  $("toggleImgBatchModeBtn").style.display = isBatch ? "none" : "block";
  $("segUnprocessedBtn").style.display = isBatch ? "none" : "block";
  $("batchSegImgsBtn").style.display = isBatch ? "block" : "none";
  $("batchDelImgsBtn").style.display = isBatch ? "block" : "none";
  $("cancelImgBatchBtn").style.display = isBatch ? "block" : "none";
  $("selectAllImgsLabel").style.display = isBatch ? "flex" : "none";
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

// 照片全選
$("selectAllImgs").onchange = (e) => {
  const isChecked = e.target.checked;
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

// 執行照片批次刪除
$("batchDelImgsBtn").onclick = async () => {
  const checked = document.querySelectorAll(".thumb-chk:checked");
  const ids = Array.from(checked).map((chk) => chk.dataset.id);
  if (ids.length === 0) return;

  if (!confirm(`確定要批次刪除選取的 ${ids.length} 張照片嗎？這會同時清除與其相關的遮罩。`)) return;

  try {
    const res = await fetch("/api/images/delete_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_ids: ids }),
    });

    if (!res.ok) throw new Error(await res.text());
    
    // 如果刪除的照片包含當前選擇的圖片，清空畫布
    if (state.currentImage && ids.includes(state.currentImage.id)) {
      state.currentImage = null;
      $("autoSegBtn").disabled = true;
      $("drawBtn").disabled = true;
      $("textPromptInput").disabled = true;
      $("textSegBtn").disabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // 退出批次模式並重整
    toggleImgBatchUI(false);
    await refreshSidebar();
  } catch (err) {
    alert("批次刪除失敗: " + err.message);
  }
};

// ---------- 批量分割 job：建立 → 輪詢進度 → 失敗清單 / 重試 ----------

let jobTimer = null;       // 輪詢計時器
let lastFinishedJob = null; // 結束的 job，給「重試失敗」用

function askBatchPrompt() {
  // 回傳 null = 使用者取消；空字串 = 自動分割整張；其他 = 文字分割 prompt
  const p = prompt("輸入文字提示做批量文字分割（留空 = 自動分割整張）", "");
  return p === null ? null : p.trim();
}

async function createSegmentJob(body) {
  const res = await fetch("/api/segment_jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await responseError(res, "建立批量分割失敗");
  return res.json();
}

function renderJob(job) {
  const panel = $("jobPanel");
  panel.hidden = false;

  const kind = job.prompt ? `批量文字分割「${job.prompt}」` : "批量自動分割";
  const running = job.status === "queued" || job.status === "running";
  const failCount = job.failed.length;

  state.jobRunning = running;
  $("segUnprocessedBtn").disabled = running;
  updateImgBatchBtnState();

  if (running) {
    $("jobTitle").textContent = `${kind}中…`;
  } else if (job.status === "interrupted") {
    $("jobTitle").textContent = `${kind}被中斷（伺服器重啟或內部錯誤），未完成的圖片請重新送出`;
  } else {
    $("jobTitle").textContent = failCount
      ? `${kind}完成，${failCount} 張失敗`
      : `${kind}完成 ✔`;
  }
  $("jobCount").textContent = `${job.done} / ${job.total}`;

  const fill = $("jobBarFill");
  fill.style.width = job.total ? `${Math.round((job.done / job.total) * 100)}%` : "0%";
  fill.classList.toggle("done", !running && !failCount && job.status === "done");
  fill.classList.toggle("has-fail", failCount > 0);

  const ul = $("jobFailList");
  ul.innerHTML = "";
  job.failed.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = `${imageNames.get(f.image_id) || f.image_id}：${f.error}`;
    ul.appendChild(li);
  });

  $("jobDismissBtn").hidden = running;
  $("jobRetryBtn").hidden = running || failCount === 0;
  lastFinishedJob = running ? null : job;
}

// job 結束後重整縮圖、側欄與目前畫布上的片段
async function refreshAfterJob() {
  await loadThumbs();
  await refreshSidebar();
  if (state.currentImage) {
    const all = await (await fetch(`/api/images/${state.currentImage.id}/segments`)).json();
    await redraw(all);
  }
}

function watchJob(jobId) {
  clearInterval(jobTimer);
  jobTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/segment_jobs/${jobId}`);
      if (!res.ok) return; // 網路暫時失敗，下一輪再試
      const job = await res.json();
      renderJob(job);
      if (job.status !== "queued" && job.status !== "running") {
        clearInterval(jobTimer);
        jobTimer = null;
        await refreshAfterJob();
      }
    } catch (e) { /* 下一輪再試 */ }
  }, 1000);
}

async function startJob(body) {
  try {
    const job = await createSegmentJob(body);
    renderJob(job);
    watchJob(job.id);
  } catch (err) {
    alert(err instanceof Error ? err.message : "建立批量分割失敗");
  }
}

// 批次模式：對勾選的圖批量分割
$("batchSegImgsBtn").onclick = async () => {
  const ids = Array.from(document.querySelectorAll(".thumb-chk:checked")).map((c) => c.dataset.id);
  if (!ids.length) return;
  const p = askBatchPrompt();
  if (p === null) return;
  toggleImgBatchUI(false);
  await startJob(p ? { image_ids: ids, prompt: p } : { image_ids: ids });
};

// 捷徑：分割所有還沒有任何片段的圖
$("segUnprocessedBtn").onclick = async () => {
  const p = askBatchPrompt();
  if (p === null) return;
  await startJob(p ? { scope: "unprocessed", prompt: p } : { scope: "unprocessed" });
};

// 重試：把失敗的那幾張組成新 job（沿用原 prompt）
$("jobRetryBtn").onclick = async () => {
  if (!lastFinishedJob || !lastFinishedJob.failed.length) return;
  const ids = lastFinishedJob.failed.map((f) => f.image_id);
  const body = lastFinishedJob.prompt
    ? { image_ids: ids, prompt: lastFinishedJob.prompt }
    : { image_ids: ids };
  await startJob(body);
};

$("jobDismissBtn").onclick = () => {
  $("jobPanel").hidden = true;
  lastFinishedJob = null;
};

// 頁面載入時找回進行中的批量工作（重整不斷線）
async function resumeActiveJob() {
  try {
    const jobs = await (await fetch("/api/segment_jobs")).json();
    const active = jobs.find((j) => j.status === "queued" || j.status === "running");
    if (active) {
      renderJob(active);
      watchJob(active.id);
    }
  } catch (e) { /* 沒有 job 或請求失敗都不影響頁面 */ }
}

// 執行遮罩批次刪除
$("batchDelSegsBtn").onclick = async () => {
  const checked = document.querySelectorAll(".seg-chk:checked");
  const ids = Array.from(checked).map((chk) => chk.dataset.id);
  if (ids.length === 0) return;

  if (!confirm(`確定要批次刪除選取的 ${ids.length} 個遮罩片段嗎？`)) return;

  try {
    const res = await fetch("/api/segments/delete_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segment_ids: ids }),
    });

    if (!res.ok) throw new Error(await res.text());

    // 退出批次模式並重整
    state.autoSegCompleted = false;
    updateAutoSegBtn();
    toggleSegBatchUI(false);
    await refreshAfterSegChange();
  } catch (err) {
    alert("批次刪除失敗: " + err.message);
  }
};

// 初始載入
loadThumbs().then(resumeActiveJob);
refreshSidebar();
