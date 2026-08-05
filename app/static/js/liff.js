// 取得 HTML 傳入的 LIFF 設定
const liffDataElement = document.getElementById("liff-data");
const LIFF_ID = liffDataElement?.dataset.liffId?.trim() ?? "";
const APPEND_TARGET_TASK_ID = new URL(window.location.href)
    .searchParams.get("append_to")?.trim() ?? "";

// 取得頁面元素
const pageTitleElement = document.getElementById("page-title");
const statusElement = document.getElementById("status");
const profileElement =document.getElementById("profile");
const displayNameElement =document.getElementById("display-name");
const profileImageElement =document.getElementById("profile-image");
const taskFormElement =document.getElementById("task-form");
const imageInputElement =document.getElementById("image-input");
const imageCountElement =document.getElementById("image-count");
const promptElement =document.getElementById("prompt");
const promptCountElement =document.getElementById("prompt-count");
const promptGroupElement = document.getElementById("prompt-group");
const uploadButtonElement = document.getElementById("upload-button");
const submitButtonElement =document.getElementById("submit-button");
const uploadProgressContainerElement = document.getElementById(
    "upload-progress-container"
);
const uploadProgressElement = document.getElementById("upload-progress");
const uploadProgressTextElement = document.getElementById(
    "upload-progress-text"
);
const uploadSuccessModalElement = document.getElementById(
    "upload-success-modal"
);
const uploadSuccessMessageElement = document.getElementById(
    "upload-success-message"
);
const uploadSuccessCloseElement = document.getElementById(
    "upload-success-close"
);
const uploadSuccessViewListElement = document.getElementById(
    "upload-success-view-list"
);
const uploadSuccessCloseHelpElement = document.getElementById(
    "upload-success-close-help"
);
const uploadedImagesSectionElement = document.getElementById(
    "uploaded-images-section"
);
const uploadedImagesCountElement = document.getElementById(
    "uploaded-images-count"
);
const uploadedImagesListElement = document.getElementById(
    "uploaded-images-list"
);
const selectedImageFiles = new Map();
const uploadedImageRecords = new Map();
let uploadSessionId = "";
let uploadReady = false;
let uploadInProgress = false;
let taskCreated = false;


function getImageFileKey(file) {
    return [
        file.name,
        file.size,
        file.type,
        file.lastModified,
    ].join(":");
}


function showUploadSuccess(result) {
    uploadSuccessMessageElement.textContent = APPEND_TARGET_TASK_ID
        ? `已成功新增 ${result.added_image_count} 張圖片。`
        : `已成功上傳 ${result.image_count} 張圖片。`;
    uploadSuccessCloseHelpElement.hidden = true;
    uploadSuccessCloseElement.disabled = false;
    uploadSuccessCloseElement.textContent = "離開頁面";
    uploadSuccessModalElement.hidden = false;
    document.body.classList.add("modal-open");
    uploadSuccessViewListElement.focus();
}


function updateCreateButtonState() {
    const promptReady = APPEND_TARGET_TASK_ID || Boolean(promptElement.value.trim());
    submitButtonElement.disabled = !(
        uploadReady
        && uploadedImageRecords.size > 0
        && promptReady
    );
    uploadedImagesCountElement.textContent = `${uploadedImageRecords.size} 張`;
}


uploadSuccessViewListElement.addEventListener("click", () => {
    window.location.href = "/liff/tasks";
});


uploadSuccessCloseElement.addEventListener("click", () => {
    uploadSuccessCloseElement.disabled = true;
    uploadSuccessCloseElement.textContent = "正在關閉...";

    if (liff.isInClient()) {
        liff.closeWindow();
        return;
    }

    window.setTimeout(() => {
        if (document.visibilityState === "visible") {
            uploadSuccessCloseElement.disabled = false;
            uploadSuccessCloseElement.textContent = "離開頁面";
            uploadSuccessCloseHelpElement.hidden = false;
            uploadSuccessViewListElement.focus();
        }
    }, 300);
    window.close();
});


window.addEventListener("beforeunload", (event) => {
    if (taskCreated || (!uploadInProgress && !uploadReady)) {
        return;
    }
    event.preventDefault();
    event.returnValue = "";
});

// URL 與 LINE 登入輔助函式
function getCleanRedirectUri() {
    /*
     * 回傳乾淨的頁面網址。
     *
     * 不包含：
     * code
     * state
     * liffClientId
     * liffRedirectUri
     */
    const redirectUrl = new URL(
        window.location.origin + window.location.pathname
    );
    if (APPEND_TARGET_TASK_ID) {
        redirectUrl.searchParams.set("append_to", APPEND_TARGET_TASK_ID);
    }
    return redirectUrl.toString();
}


function cleanLineCallbackUrl() {
    /*
     * LINE Login 完成後，網址可能包含：
     *
     * ?code=...
     * &state=...
     * &liffClientId=...
     * &liffRedirectUri=...
     *
     * 這些參數只供 LIFF SDK 初始化使用。
     * 必須等 liff.init() 完成後才能清除。
     */

    const url = new URL(window.location.href);

    const callbackParameters = [
        "code",
        "state",
        "liffClientId",
        "liffRedirectUri",
        "friendship_status_changed",
    ];

    let changed = false;

    callbackParameters.forEach(
        (parameter) => {
            if (
                url.searchParams.has(parameter)
            ) {
                url.searchParams.delete(parameter);
                changed = true;
            }
        }
    );

    if (!changed) {
        return;
    }

    const queryString =
        url.searchParams.toString();

    const cleanUrl =
        url.pathname +
        (
            queryString
                ? `?${queryString}`
                : ""
        ) +
        url.hash;

    window.history.replaceState(
        null,
        document.title,
        cleanUrl
    );
}


function isCurrentIdTokenExpired() {
    /*
     * 前端只做過期時間的預先檢查。
     * 真正的身分安全驗證仍由 Flask 呼叫 LINE API。
     */

    const decodedToken =liff.getDecodedIDToken();

    if (
        !decodedToken ||
        !decodedToken.exp
    ) {
        return true;
    }

    const currentTimestamp =Math.floor(Date.now() / 1000);

    /*
     * 預留 30 秒。
     * 避免 Token 在送到 Flask 的途中剛好過期。
     */
    return (decodedToken.exp <=currentTimestamp + 30);
}


function restartLineLogin() {
    /*
     * 在真正的 LIFF Browser 內，
     * 不適合直接呼叫 liff.login()。
     *
     * 因此提示使用者關閉頁面後重新從 LINE 開啟。
     */
    if (liff.isInClient()) {
        statusElement.textContent ="LINE 登入資料已失效，請關閉此頁面後，從 LINE 選單重新開啟。";
        return;
    }

    /*
     * 外部瀏覽器可以先登出，
     * 再重新執行 LINE Login。
     */
    statusElement.textContent ="LINE 登入資料已失效，正在重新登入……";

    if (liff.isLoggedIn()) {liff.logout();}

    liff.login({redirectUri: getCleanRedirectUri(),});
}


// ========================================
// 初始化 LIFF
// ========================================

async function initializeLiff() {
    try {
        if (!LIFF_ID) {
            throw new Error(
                "找不到 LIFF_ID，請檢查 .env 與 Config 設定"
            );
        }

        statusElement.textContent =
            "正在連接 LINE……";

        await liff.init({
            liffId: LIFF_ID,

            // 外部瀏覽器開啟時，自動進入 LINE Login
            withLoginOnExternalBrowser: true,
        });

        /*
         * 一定要等 liff.init() 完成後，
         * 才能清除 callback 參數。
         */
        cleanLineCallbackUrl();

        if (!liff.isLoggedIn()) {
            if (liff.isInClient()) {
                throw new Error(
                    "LINE 登入狀態異常，請關閉頁面後重新開啟"
                );
            }

            liff.login({
                redirectUri: getCleanRedirectUri(),
            });

            return;
        }

        /*
         * Profile 只用來顯示名稱與頭像。
         * 真正身分以 Flask 驗證 ID Token 的結果為準。
         */
        const profile =
            await liff.getProfile();

        displayNameElement.textContent =
            profile.displayName;

        if (profile.pictureUrl) {
            profileImageElement.src =profile.pictureUrl;

            profileImageElement.style.display ="block";
        } else {
            profileImageElement.style.display ="none";
        }

        profileElement.style.display ="block";

        if (APPEND_TARGET_TASK_ID) {
            statusElement.textContent = "正在載入原標註任務……";
            const appendContext = await requestJson(
                `/liff/tasks/${encodeURIComponent(APPEND_TARGET_TASK_ID)}/append/context`,
                {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        id_token: liff.getIDToken(),
                    }),
                }
            );
            promptElement.value = appendContext.prompt;
            promptElement.readOnly = true;
            promptCountElement.textContent =
                `${appendContext.prompt.length} / 200`;
            pageTitleElement.textContent = "新增照片";
            submitButtonElement.textContent = "新增照片並重新標註";
        }

        taskFormElement.style.display ="block";

        statusElement.textContent = APPEND_TARGET_TASK_ID
            ? "請選擇要加入此任務的照片"
            : "歡迎使用標註系統";

    } catch (error) {
        console.error(
            "LIFF 初始化失敗：",
            error
        );

        statusElement.textContent =
            `LIFF 登入失敗：${error.message}`;
    }
}


// ========================================
// 圖片選擇
// ========================================

imageInputElement.addEventListener("change", () => {
    if (uploadInProgress || uploadSessionId) {
        return;
    }
    const newFiles = Array.from(imageInputElement.files);
    const allowedTypes = ["image/jpeg", "image/png"];

    if (newFiles.length === 0) {
        return;
    }

    const invalidFiles = newFiles.filter(
        (file) => !allowedTypes.includes(file.type)
    );

    if (invalidFiles.length > 0) {
        imageInputElement.value = "";
        imageCountElement.textContent =
            "檔案格式錯誤，只能選擇 JPG、JPEG 或 PNG";
        imageCountElement.classList.add("error-message");
        return;
    }

    for (const file of newFiles) {
        selectedImageFiles.set(
            getImageFileKey(file),
            file
        );
    }

    imageInputElement.value = "";
    imageCountElement.classList.remove("error-message");
    imageCountElement.textContent =
        `已累加選擇 ${selectedImageFiles.size} 張圖片`;
    uploadButtonElement.disabled = selectedImageFiles.size === 0;

    console.log(
        "目前累加的圖片：",
        Array.from(selectedImageFiles.values())
    );
});


// ========================================
// Prompt 字數統計
// ========================================

promptElement.addEventListener(
    "input",
    () => {
        const promptLength =
            promptElement.value.length;

        promptCountElement.textContent =
            `${promptLength} / 200`;
        updateCreateButtonState();
    }
);


const LIFF_UPLOAD_BATCH_MAX_IMAGES = Number.parseInt(
    liffDataElement?.dataset.uploadBatchMaxImages ?? "5",
    10
);
const LIFF_UPLOAD_BATCH_MAX_BYTES = Number.parseInt(
    liffDataElement?.dataset.uploadBatchMaxBytes ?? `${15 * 1024 * 1024}`,
    10
);
const LIFF_UPLOAD_MAX_TOTAL_BYTES = Number.parseInt(
    liffDataElement?.dataset.uploadMaxTotalBytes ?? "0",
    10
);
const LIFF_UPLOAD_MAX_RETRIES = 3;
const LIFF_UPLOAD_REQUEST_TIMEOUT_MS = 5 * 60 * 1000;


function waitForRetry(milliseconds) {
    return new Promise(
        (resolve) => window.setTimeout(resolve, milliseconds)
    );
}


async function requestJson(url, options) {
    const response = await fetch(url, options);
    const responseText = await response.text();
    let result = null;

    if (responseText) {
        try {
            result = JSON.parse(responseText);
        } catch {
            const error = new Error(
                `伺服器拒絕請求（HTTP ${response.status}）`
            );
            error.status = response.status;
            throw error;
        }
    }

    if (!response.ok) {
        const error = new Error(
            result?.message || `請求失敗（HTTP ${response.status}）`
        );
        error.status = response.status;
        error.code = result?.code || "";
        throw error;
    }

    if (!result) {
        const error = new Error("伺服器沒有回傳資料");
        error.status = response.status;
        throw error;
    }

    return result;
}


function addUploadedImageRecord(clientId, imageId, file) {
    if (!file || uploadedImageRecords.has(clientId)) {
        return;
    }

    const item = document.createElement("figure");
    item.className = "uploaded-image-item";
    item.dataset.clientId = clientId;
    item.dataset.imageId = imageId;

    const image = document.createElement("img");
    const objectUrl = URL.createObjectURL(file);
    image.alt = file.name;
    image.loading = "lazy";
    image.decoding = "async";
    image.addEventListener("load", () => {
        URL.revokeObjectURL(objectUrl);
    }, {once: true});
    image.addEventListener("error", () => {
        URL.revokeObjectURL(objectUrl);
    }, {once: true});
    image.src = objectUrl;

    const caption = document.createElement("figcaption");
    caption.textContent = file.name;

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "uploaded-image-remove";
    removeButton.setAttribute("aria-label", `移除 ${file.name}`);
    removeButton.textContent = "×";
    removeButton.disabled = !uploadReady;
    removeButton.addEventListener("click", async () => {
        if (!uploadReady || !uploadSessionId) {
            return;
        }
        removeButton.disabled = true;
        item.classList.add("is-removing");
        try {
            await requestJson(
                `/liff/uploads/${encodeURIComponent(uploadSessionId)}/images/${encodeURIComponent(imageId)}`,
                {
                    method: "DELETE",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        id_token: liff.getIDToken(),
                    }),
                }
            );
            uploadedImageRecords.delete(clientId);
            selectedImageFiles.delete(clientId);
            item.remove();
            imageCountElement.textContent = uploadedImageRecords.size > 0
                ? `已上傳並保留 ${uploadedImageRecords.size} 張圖片`
                : "已移除全部圖片，請重新整理後重新選擇";
            updateCreateButtonState();
        } catch (error) {
            item.classList.remove("is-removing");
            removeButton.disabled = false;
            statusElement.textContent = `移除圖片失敗：${error.message}`;
        }
    });

    item.append(image, caption, removeButton);
    uploadedImagesListElement.append(item);
    uploadedImageRecords.set(clientId, {
        clientId,
        imageId,
        file,
        item,
        removeButton,
    });
    uploadedImagesSectionElement.hidden = false;
    updateCreateButtonState();
}


function enableUploadedImageRemoval() {
    uploadReady = true;
    for (const record of uploadedImageRecords.values()) {
        record.removeButton.disabled = false;
    }
    updateCreateButtonState();
}


function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "0 B";
    }

    const units = ["B", "KiB", "MiB", "GiB"];
    const unitIndex = Math.min(
        Math.floor(Math.log(bytes) / Math.log(1024)),
        units.length - 1
    );
    const value = bytes / (1024 ** unitIndex);
    return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}


function updateUploadProgress(
    uploadedBytes,
    totalBytes,
    uploadedCount,
    totalCount,
    message = "正在上傳"
) {
    const normalizedBytes = Math.min(
        Math.max(uploadedBytes, 0),
        Math.max(totalBytes, 0)
    );
    const percent = totalBytes > 0
        ? Math.floor((normalizedBytes / totalBytes) * 100)
        : 0;

    uploadProgressContainerElement.hidden = false;
    uploadProgressElement.value = percent;
    uploadProgressTextElement.textContent =
        `${message} ${percent}%・${uploadedCount} / ${totalCount} 張・`
        + `${formatBytes(normalizedBytes)} / ${formatBytes(totalBytes)}`;
}


function createUploadBatches(files, batchMaxImages, batchMaxBytes) {
    const batches = [];
    let currentBatch = [];
    let currentBytes = 0;

    for (const file of files) {
        if (file.size > batchMaxBytes) {
            throw new Error(
                `${file.name} 超過 LIFF 單張 ${formatBytes(batchMaxBytes)} 上傳限制`
            );
        }

        const exceedsBatch =
            currentBatch.length >= batchMaxImages
            || currentBytes + file.size > batchMaxBytes;

        if (currentBatch.length > 0 && exceedsBatch) {
            batches.push(currentBatch);
            currentBatch = [];
            currentBytes = 0;
        }

        currentBatch.push(file);
        currentBytes += file.size;
    }

    if (currentBatch.length > 0) {
        batches.push(currentBatch);
    }

    return batches;
}


function uploadBatchOnce(
    sessionId,
    batchId,
    files,
    lineIdToken,
    onProgress
) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append("id_token", lineIdToken);
        formData.append("batch_id", batchId);
        files.forEach((file) => {
            formData.append("images", file);
            formData.append("client_ids", getImageFileKey(file));
        });

        const xhr = new XMLHttpRequest();
        xhr.open(
            "POST",
            `/liff/uploads/${encodeURIComponent(sessionId)}/batch`
        );
        xhr.timeout = LIFF_UPLOAD_REQUEST_TIMEOUT_MS;

        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable && event.total > 0) {
                onProgress(Math.min(event.loaded / event.total, 1));
            }
        };

        xhr.onload = () => {
            let result = null;
            if (xhr.responseText) {
                try {
                    result = JSON.parse(xhr.responseText);
                } catch {
                    const error = new Error(
                        `伺服器拒絕請求（HTTP ${xhr.status}）`
                    );
                    error.status = xhr.status;
                    reject(error);
                    return;
                }
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                const error = new Error(
                    result?.message || `請求失敗（HTTP ${xhr.status}）`
                );
                error.status = xhr.status;
                error.code = result?.code || "";
                reject(error);
                return;
            }

            if (!result) {
                const error = new Error("伺服器沒有回傳資料");
                error.status = xhr.status;
                reject(error);
                return;
            }

            resolve(result);
        };

        xhr.onerror = () => {
            const error = new Error("網路連線中斷");
            error.status = 0;
            reject(error);
        };
        xhr.ontimeout = () => {
            const error = new Error("圖片批次上傳逾時");
            error.status = 408;
            reject(error);
        };
        xhr.onabort = () => {
            const error = new Error("圖片批次上傳已取消");
            error.status = 0;
            reject(error);
        };

        xhr.send(formData);
    });
}


async function uploadBatchWithRetry(
    sessionId,
    batchId,
    files,
    lineIdToken,
    onProgress,
    onRetry
) {
    let lastError = null;

    for (let attempt = 1; attempt <= LIFF_UPLOAD_MAX_RETRIES; attempt += 1) {
        try {
            return await uploadBatchOnce(
                sessionId,
                batchId,
                files,
                lineIdToken,
                onProgress
            );
        } catch (error) {
            lastError = error;
            const retryable =
                !error.status
                || error.status === 408
                || error.status === 429
                || error.status >= 500;
            if (!retryable || attempt === LIFF_UPLOAD_MAX_RETRIES) {
                throw error;
            }
            onRetry(attempt + 1, LIFF_UPLOAD_MAX_RETRIES);
            await waitForRetry(1000 * (2 ** (attempt - 1)));
        }
    }

    throw lastError || new Error("圖片批次上傳失敗");
}


// ========================================
// 暫存圖片上傳
// ========================================

uploadButtonElement.addEventListener("click", async () => {
    const files = Array.from(selectedImageFiles.values());
    if (files.length === 0 || uploadInProgress || uploadSessionId) {
        return;
    }
    if (!liff.isLoggedIn() || isCurrentIdTokenExpired()) {
        restartLineLogin();
        return;
    }
    const lineIdToken = liff.getIDToken();
    if (!lineIdToken) {
        restartLineLogin();
        return;
    }

    uploadInProgress = true;
    uploadButtonElement.disabled = true;
    uploadButtonElement.textContent = "上傳中...";
    imageInputElement.disabled = true;

    try {
        const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
        if (
            LIFF_UPLOAD_MAX_TOTAL_BYTES > 0
            && totalBytes > LIFF_UPLOAD_MAX_TOTAL_BYTES
        ) {
            throw new Error(
                `所選圖片總大小超過 ${formatBytes(LIFF_UPLOAD_MAX_TOTAL_BYTES)}`
            );
        }
        const batches = createUploadBatches(
            files,
            LIFF_UPLOAD_BATCH_MAX_IMAGES,
            LIFF_UPLOAD_BATCH_MAX_BYTES
        );
        updateUploadProgress(0, totalBytes, 0, files.length, "準備上傳");
        statusElement.textContent = "正在建立上傳工作階段...";
        const uploadSession = await requestJson("/liff/uploads/init", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                id_token: lineIdToken,
                expected_image_count: files.length,
                expected_total_bytes: totalBytes,
                target_task_id: APPEND_TARGET_TASK_ID,
            }),
        });
        uploadSessionId = uploadSession.session_id;

        let uploadedCount = 0;
        let committedBytes = 0;
        for (let index = 0; index < batches.length; index += 1) {
            const batch = batches[index];
            const batchBytes = batch.reduce((sum, file) => sum + file.size, 0);
            statusElement.textContent =
                `正在上傳 ${uploadedCount} / ${files.length} 張圖片...`;
            const batchResult = await uploadBatchWithRetry(
                uploadSessionId,
                `batch-${index + 1}`,
                batch,
                lineIdToken,
                (batchRatio) => {
                    updateUploadProgress(
                        committedBytes + (batchBytes * batchRatio),
                        totalBytes,
                        uploadedCount,
                        files.length
                    );
                },
                (nextAttempt, maxAttempts) => {
                    statusElement.textContent =
                        `網路不穩，正在重試第 ${nextAttempt} / ${maxAttempts} 次...`;
                }
            );
            for (const item of batchResult.items || []) {
                addUploadedImageRecord(
                    item.client_id,
                    item.image_id,
                    selectedImageFiles.get(item.client_id)
                );
            }
            uploadedCount = batchResult.uploaded_count;
            const serverUploadedBytes = Number(batchResult.uploaded_bytes);
            committedBytes = Number.isFinite(serverUploadedBytes)
                ? serverUploadedBytes
                : committedBytes + batchBytes;
            updateUploadProgress(
                committedBytes,
                totalBytes,
                uploadedCount,
                files.length
            );
        }

        const readyResult = await requestJson(
            `/liff/uploads/${encodeURIComponent(uploadSessionId)}/finalize`,
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({id_token: lineIdToken}),
            }
        );
        if (readyResult.task_status !== "upload_ready") {
            throw new Error("圖片清單尚未進入確認狀態");
        }

        enableUploadedImageRemoval();
        promptGroupElement.hidden = false;
        submitButtonElement.hidden = false;
        uploadButtonElement.textContent = "圖片上傳完成";
        imageCountElement.textContent = `已上傳 ${uploadedImageRecords.size} 張圖片`;
        statusElement.textContent = "圖片上傳完成，請確認清單並建立標註任務";
        updateUploadProgress(
            totalBytes,
            totalBytes,
            files.length,
            files.length,
            "上傳完成"
        );
        updateCreateButtonState();
        uploadedImagesSectionElement.scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
    } catch (error) {
        console.error("上傳失敗：", error);
        if (error.status === 401) {
            statusElement.textContent = error.message || "LINE 登入資料已失效";
            restartLineLogin();
            return;
        }
        statusElement.textContent = `上傳失敗：${error.message}`;
        if (!uploadSessionId) {
            uploadButtonElement.disabled = false;
            imageInputElement.disabled = false;
        }
    } finally {
        uploadInProgress = false;
        if (!uploadReady && !uploadSessionId) {
            uploadButtonElement.textContent = "開始上傳";
        }
    }
});


// ========================================
// 正式建立標註任務
// ========================================

taskFormElement.addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = promptElement.value.trim();
    if (!uploadReady || !uploadSessionId || uploadedImageRecords.size === 0) {
        statusElement.textContent = "請先完成圖片上傳並至少保留一張圖片";
        return;
    }
    if (!APPEND_TARGET_TASK_ID && !prompt) {
        statusElement.textContent = "請輸入想要標註的物件";
        promptElement.focus();
        return;
    }
    if (!liff.isLoggedIn() || isCurrentIdTokenExpired()) {
        restartLineLogin();
        return;
    }
    const lineIdToken = liff.getIDToken();
    if (!lineIdToken) {
        restartLineLogin();
        return;
    }

    submitButtonElement.disabled = true;
    submitButtonElement.textContent = "正在建立任務...";
    statusElement.textContent = "正在建立標註任務...";
    try {
        const result = await requestJson(
            `/liff/uploads/${encodeURIComponent(uploadSessionId)}/create-task`,
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    id_token: lineIdToken,
                    prompt,
                }),
            }
        );
        if (!["pending", "processing"].includes(result.task_status)) {
            throw new Error("任務尚未進入排隊狀態，請稍後再試");
        }
        taskCreated = true;
        taskFormElement.style.display = "none";
        statusElement.textContent = APPEND_TARGET_TASK_ID
            ? "圖片已新增，任務正在背景重新排隊"
            : "標註任務已建立，正在背景執行";
        showUploadSuccess(result);
    } catch (error) {
        console.error("建立任務失敗：", error);
        if (error.status === 401) {
            restartLineLogin();
            return;
        }
        statusElement.textContent = `建立任務失敗：${error.message}`;
        submitButtonElement.disabled = false;
        submitButtonElement.textContent = APPEND_TARGET_TASK_ID
            ? "新增照片並重新標註"
            : "建立標註任務";
    }
});


// ========================================
// 啟動 LIFF
// ========================================

initializeLiff();
