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
};
const $ = (id) => document.getElementById(id);

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

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

function setSegmentationLoading(active, message = "分割中…") {
  state.segmenting = active;
  $("segmentLoadingText").textContent = message;
  $("segmentLoading").hidden = !active;
  canvas.closest(".canvas-wrap").classList.toggle("is-loading", active);
  canvas.setAttribute("aria-busy", String(active));
  $("autoSegBtn").disabled = active || !state.currentImage;
  $("drawBtn").disabled = active || !state.currentImage;
  $("textPromptInput").disabled = active || !state.currentImage;
  $("textSegBtn").disabled = active || !state.currentImage;
}

async function responseError(res, fallback) {
  const detail = await res.text();
  return new Error(detail ? `${fallback}：${detail}` : fallback);
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
  imgs.forEach((im) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";

    const el = document.createElement("img");
    el.src = `/api/images/${im.id}/file`;
    el.title = im.filename;
    el.onclick = () => selectImage(im, el);

    const del = document.createElement("button");
    del.className = "thumb-del";
    del.textContent = "×";
    del.title = "刪除這張";
    del.onclick = (e) => { e.stopPropagation(); deleteImage(im); };

    wrap.append(el, del);
    box.appendChild(wrap);
  });
}

async function deleteImage(im) {
  if (!confirm(`確定刪除「${im.filename}」？連同它的遮罩會一起清掉。`)) return;
  const res = await fetch(`/api/images/${im.id}`, { method: "DELETE" });
  if (!res.ok) return alert("刪除失敗：" + (await res.text()));
  // 若刪的是目前選中的圖，清空畫布
  if (state.currentImage && state.currentImage.id === im.id) {
    state.currentImage = null;
    $("autoSegBtn").disabled = true;
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
  $("autoSegBtn").disabled = false;
  $("drawBtn").disabled = false;
  $("textPromptInput").disabled = false;
  $("textSegBtn").disabled = false;

  const pic = new Image();
  pic.onload = () => {
    canvas.width = pic.width;
    canvas.height = pic.height;
    ctx.drawImage(pic, 0, 0);
  };
  pic.src = `/api/images/${im.id}/file`;
}

// ---------- 自動分割整張 ----------
$("autoSegBtn").onclick = async () => {
  if (!state.currentImage || state.segmenting) return;
  const imageId = state.currentImage.id;
  setSegmentationLoading(true, "自動分割中…");
  try {
    const res = await fetch(`/api/images/${imageId}/segment`, { method: "POST" });
    if (!res.ok) throw await responseError(res, "自動分割失敗");
    const data = await res.json();

    // 如果全部區塊原本就已經存在（無缺失且已自動分割過），彈出完成提示
    if (data.status === "already_completed") {
      alert("已經完成自動分割");
    }

    await redraw(data.segments);
    await refreshSidebar();
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
  setSegmentationLoading(true, `正在搜尋「${promptVal}」並進行分割…`);

  try {
    const res = await fetch(`/api/images/${imageId}/segment_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptVal }),
    });

    if (!res.ok) throw await responseError(res, "文字分割失敗");
    const segs = await res.json();
    await redraw(segs);
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
  state.lastSegments = segments;
  const pic = await loadImage(state.currentImage.id);
  ctx.drawImage(pic, 0, 0);

  // 💡 步驟 1：先畫所有不規則的 SAM 遮罩（Mask），採取「同步加載快取 + 異步載入重繪」以防畫面閃爍
  for (const s of segments) {
    const maskImg = maskImages[s.id];
    const hi = s.id === highlightId;
    const color = hi ? "#ffd166" : (s.needs_review ? "#ff5470" : "#36d399");

    if (maskImg && maskImg.complete && maskImg.naturalWidth > 0) {
      // 若已經下載完畢，同步進行著色與繪製，避免 await 導致繪圖不同步
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = pic.width;
      tempCanvas.height = pic.height;
      const tctx = tempCanvas.getContext("2d");

      // 1. 先繪製黑白遮罩圖
      tctx.drawImage(maskImg, 0, 0);

      // 2. 取得畫素進行去背與著色：將黑白色彩的「亮度」直接映射為「透明度（Alpha）」
      const imgData = tctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const data = imgData.data;

      // 解析 hex 顏色為 RGB
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);

      for (let i = 0; i < data.length; i += 4) {
        const rVal = data[i];     // R 通道值 (0 ~ 255) 代表亮度
        const aVal = data[i + 3]; // 原本的 Alpha 通道值 (0 ~ 255) 代表透明度
        
        // 💡 融合公式：最終透明度 = (亮度/255) * 原本的透明度
        // 這能 100% 相容「黑底白階圖」與「透明底白階圖」，徹底解決整片染紅的 bug！
        const alphaVal = Math.round((rVal / 255) * aVal);
        
        data[i] = r;       // R 通道設為目標色
        data[i + 1] = g;   // G 通道設為目標色
        data[i + 2] = b;   // B 通道設為目標色
        data[i + 3] = alphaVal; // 寫入融合後的透明度
      }
      tctx.putImageData(imgData, 0, 0);

      // 3. 疊加到主 Canvas
      ctx.save();
      ctx.globalAlpha = 0.35; // 35% 半透明色塊
      ctx.drawImage(tempCanvas, 0, 0);
      ctx.restore();
    } else if (!maskImages[s.id]) {
      // 若尚未下載，則啟動非同步下載，下載成功後觸發重繪
      const img = new Image();
      img.src = `/api/segments/${s.id}/mask`;
      img.onload = () => {
        // 確保重新繪製時仍為同一張大圖，防止圖片切換後的時序干擾
        if (state.currentImage && state.currentImage.id === currentImageId) {
          redraw(segments, highlightId);
        }
      };
      img.onerror = () => {
        console.warn("Mask 下載失敗:", s.id);
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
  queue.forEach((s) => {
    const li = document.createElement("li");
    const probs = Object.entries(s.probs)
      .map(([k, v]) => `${k}:${v.toFixed(2)}`)
      .join(" · ") || "（尚無範例可分類）";
    li.innerHTML = `
      <div class="queue-item">
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
      const label = li.querySelector("input").value.trim();
      if (!label) return;
      await fetch(`/api/segments/${s.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      await refreshAfterSegChange();
    };
    li.querySelector(".seg-del").onclick = async () => {
      li.classList.add("deleting");
      await new Promise((resolve) => setTimeout(resolve, 800));
      try {
        const res = await fetch(`/api/segments/${s.id}`, { method: "DELETE" });
        if (!res.ok) {
          li.classList.remove("deleting");
          return alert("刪除失敗：" + (await res.text()));
        }
        await refreshAfterSegChange();
      } catch (error) {
        li.classList.remove("deleting");
        console.error(error);
        alert("刪除時發生錯誤");
      }
    };
    ul.appendChild(li);
  });
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

// 初始載入
loadThumbs();
refreshSidebar();
