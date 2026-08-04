// 取得 HTML 傳入的 LIFF 設定
const liffDataElement = document.getElementById("liff-data");
const LIFF_ID = liffDataElement?.dataset.liffId?.trim() ?? "";

// 取得頁面元素
const statusElement = document.getElementById("status");
const profileElement =document.getElementById("profile");
const displayNameElement =document.getElementById("display-name");
const profileImageElement =document.getElementById("profile-image");
const taskFormElement =document.getElementById("task-form");
const imageInputElement =document.getElementById("image-input");
const imageCountElement =document.getElementById("image-count");
const promptElement =document.getElementById("prompt");
const promptCountElement =document.getElementById("prompt-count");
const submitButtonElement =document.getElementById("submit-button");
const selectedImageFiles = new Map();


function getImageFileKey(file) {
    return [
        file.name,
        file.size,
        file.type,
        file.lastModified,
    ].join(":");
}

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
    return (
        window.location.origin +
        window.location.pathname
    );
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

        taskFormElement.style.display ="block";

        statusElement.textContent ="歡迎使用標註系統";

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
    }
);


const LIFF_UPLOAD_MAX_IMAGES = 30;
const LIFF_UPLOAD_BATCH_MAX_IMAGES = 5;
const LIFF_UPLOAD_BATCH_MAX_BYTES = 15 * 1024 * 1024;
const LIFF_UPLOAD_MAX_RETRIES = 3;


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


function createUploadBatches(files) {
    const batches = [];
    let currentBatch = [];
    let currentBytes = 0;

    for (const file of files) {
        if (file.size > LIFF_UPLOAD_BATCH_MAX_BYTES) {
            throw new Error(
                `${file.name} 超過 LIFF 單張 15 MiB 上傳限制`
            );
        }

        const exceedsBatch =
            currentBatch.length >= LIFF_UPLOAD_BATCH_MAX_IMAGES
            || currentBytes + file.size > LIFF_UPLOAD_BATCH_MAX_BYTES;

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


async function uploadBatchWithRetry(
    sessionId,
    batchId,
    files,
    lineIdToken
) {
    let lastError = null;

    for (let attempt = 1; attempt <= LIFF_UPLOAD_MAX_RETRIES; attempt += 1) {
        const formData = new FormData();
        formData.append("id_token", lineIdToken);
        formData.append("batch_id", batchId);
        files.forEach((file) => formData.append("images", file));

        try {
            return await requestJson(
                `/liff/uploads/${encodeURIComponent(sessionId)}/batch`,
                {
                    method: "POST",
                    body: formData,
                }
            );
        } catch (error) {
            lastError = error;
            const retryable = !error.status || error.status >= 500;
            if (!retryable || attempt === LIFF_UPLOAD_MAX_RETRIES) {
                throw error;
            }
            await waitForRetry(1000 * (2 ** (attempt - 1)));
        }
    }

    throw lastError || new Error("圖片批次上傳失敗");
}


// ========================================
// 表單送出
// ========================================

taskFormElement.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const files = Array.from(selectedImageFiles.values());

        const prompt =promptElement.value.trim();


        // ------------------------
        // 驗證圖片
        // ------------------------

        if (files.length === 0) {
            statusElement.textContent =
                "請至少選擇一張圖片";

            imageCountElement.textContent =
                "請先選擇圖片";

            imageCountElement.classList.add(
                "error-message"
            );

            return;
        }

        if (files.length > LIFF_UPLOAD_MAX_IMAGES) {
            statusElement.textContent =
                `每個任務最多選擇 ${LIFF_UPLOAD_MAX_IMAGES} 張圖片`;
            imageCountElement.classList.add("error-message");
            return;
        }


        // ------------------------
        // 驗證 Prompt
        // ------------------------

        if (!prompt) {
            statusElement.textContent =
                "請輸入想要標註的物件";

            promptElement.focus();

            return;
        }


        // ------------------------
        // 驗證 LINE 登入狀態
        // ------------------------

        if (!liff.isLoggedIn()) {
            restartLineLogin();
            return;
        }


        // ------------------------
        // 預先檢查 Token 是否過期
        // ------------------------

        if (isCurrentIdTokenExpired()) {
            statusElement.textContent =
                "LINE 登入資料已過期";

            restartLineLogin();

            return;
        }


        /*
         * 每次按送出時才取得目前的 ID Token。
         *
         * 不建立全域 lineIdToken，
         * 避免出現 lineIdToken is not defined。
         */
        const lineIdToken = liff.getIDToken();

        if (!lineIdToken) {
            restartLineLogin();
            return;
        }


        try {
            submitButtonElement.disabled =
                true;

            submitButtonElement.textContent =
                "上傳中...";

            statusElement.textContent =
                "正在建立上傳工作階段...";

            const batches = createUploadBatches(files);
            const uploadSession = await requestJson(
                "/liff/uploads/init",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        id_token: lineIdToken,
                        prompt,
                        expected_image_count: files.length,
                    }),
                }
            );

            let uploadedCount = 0;
            for (let index = 0; index < batches.length; index += 1) {
                const batch = batches[index];
                statusElement.textContent =
                    `正在上傳 ${uploadedCount} / ${files.length} 張圖片...`;

                const batchResult = await uploadBatchWithRetry(
                    uploadSession.session_id,
                    `batch-${index + 1}`,
                    batch,
                    lineIdToken
                );
                uploadedCount = batchResult.uploaded_count;
                statusElement.textContent =
                    `已上傳 ${uploadedCount} / ${files.length} 張圖片`;
            }

            statusElement.textContent =
                "圖片上傳完成，正在建立標註任務...";
            const result = await requestJson(
                `/liff/uploads/${encodeURIComponent(uploadSession.session_id)}/finalize`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        id_token: lineIdToken,
                    }),
                }
            );

            if (result.task_status !== "pending") {
                throw new Error(
                    "任務尚未進入排隊狀態，請稍後再試"
                );
            }


            // ------------------------
            // 上傳成功
            // ------------------------

            console.log(
                "後端回傳資料：",
                result
            );

            taskFormElement.style.display = "none";

            if (liff.isInClient()) {
                statusElement.textContent =
                    `已建立 ${result.image_count} 張圖片的標註任務，正在關閉頁面...`;

                window.setTimeout(() => {
                    liff.closeWindow();
                }, 1200);
            } else {
                statusElement.textContent =
                    `已建立 ${result.image_count} 張圖片的標註任務，可以關閉此頁面`;
            }

        } catch (error) {
            console.error(
                "上傳失敗：",
                error
            );

            if (error.status === 401) {
                statusElement.textContent =
                    error.message || "LINE 登入資料已失效";
                restartLineLogin();
                return;
            }

            statusElement.textContent =
                `上傳失敗：${error.message}`;

        } finally {
            submitButtonElement.disabled =
                false;

            submitButtonElement.textContent =
                "建立標註任務";
        }
    }
);


// ========================================
// 啟動 LIFF
// ========================================

initializeLiff();
