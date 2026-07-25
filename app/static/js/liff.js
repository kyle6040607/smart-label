
// 透過 dataset 抓取 HTML 上的變數
const liffData = document.getElementById("liff-data");
const LIFF_ID = liffData.dataset.liffId;

const statusElement = document.getElementById("status");
const profileElement = document.getElementById("profile");
const displayNameElement = document.getElementById("display-name");
const profileImageElement = document.getElementById("profile-image");

const taskFormElement = document.getElementById("task-form");
const imageInputElement = document.getElementById("image-input");
const imageCountElement = document.getElementById("image-count")

const promptElement = document.getElementById("prompt");
const promptCountElement = document.getElementById("prompt-count");
const submitButtonElement = document.getElementById("submit-button");

let lineIdToken = null;
async function initializeLiff() {
    try {
        if (!LIFF_ID) {
            throw new Error("找不到 LIFF_ID，請檢查 .env 設定");
        }

        await liff.init({
            liffId: LIFF_ID,
            // 使用一般瀏覽器開啟時，自動進入 LINE 登入流程
            withLoginOnExternalBrowser: true,
        });

        if(!liff.isLoggedIn()) {
            liff.login();
            return;
        }

        const profile = await liff.getProfile();
        lineIdToken = liff.getIDToken();

        if (!lineIdToken) {
            throw new Error(
                "無法取得 LINE ID Token，請確認 LIFF 已啟用 openid 權限"
            );
        }

        // 不要把完整 Token 印在 Console
        console.log("已取得 LINE ID Token");

        // 取得使用者資料
        displayNameElement.textContent = profile.displayName;
        // 取得使用者照片
        if(profile.pictureUrl) {
            profileImageElement.src = profile.pictureUrl;
            profileImageElement.style.display = "block";
        }

        profileElement.style.display = "block";
        taskFormElement.style.display = "block";

        statusElement.textContent = "歡迎使用標註系統";

    } catch (error) {
        console.error("LIFF 登入失敗:", error);
        statusElement.textContent = `LIFF 登入失敗:${error.message}`;
    }
}

// 選擇圖片按鈕
imageInputElement.addEventListener("change", () => {
    const files = Array.from(imageInputElement.files);

    if (files.length === 0) {
        imageCountElement.textContent = "尚未選擇圖片";
        return;
    }
    const allowedTypes = [
        "image/jpeg",
        "image/png",
    ];

    const invalidFiles = files.filter(
        file => !allowedTypes.includes(file.type)
    );
    // 
    if (invalidFiles.length > 0) {
        imageInputElement.value = "";
        imageCountElement.textContent = 
        "檔案格式錯誤，只能選擇JPG、JPEG 或 PNG";

        return;
    }

    imageCountElement.textContent =
        `已選 ${files.length} 張圖片`;
    
        console.log("選擇的圖片: ", files);
});

// prompt count
promptElement.addEventListener("input", () => {
    const promptLength = promptElement.value.length;

    promptCountElement.textContent = 
        `${promptLength} / 200`
});

// 表單檢查
taskFormElement.addEventListener("submit", async (event) => {
    event.preventDefault();

    const files = Array.from(imageInputElement.files);
    const prompt = promptElement.value.trim();

    if (files.length === 0) {
        statusElement.textContent = "請至少選擇一張圖片";
        return;        
    }

    if (!prompt) {
    statusElement.textContent = "請輸入想要標註的物件";
    promptElement.focus();
    return;
    }

    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("id_token", lineIdToken);

    files.forEach((file)=>{
        formData.append("images", file);
    });

    try {
        submitButtonElement.disabled = true;
        submitButtonElement.textContent = "上傳中...";
        statusElement.textContent =
            `正在上傳 ${files.length} 張圖片...`;

        const response = await fetch(window.location.pathname, {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.message || "圖片上傳失敗");
        }

        console.log("後端回傳資料：", result);

        statusElement.textContent =
            `上傳成功，收到 ${result.image_count} 張圖片`;

    } catch (error) {
        console.error("上傳失敗：", error);

        statusElement.textContent =
            `上傳失敗：${error.message}`;

    } finally {
        submitButtonElement.disabled = false;
        submitButtonElement.textContent = "建立標註任務";
    }
});

initializeLiff();
