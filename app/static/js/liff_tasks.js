const liffDataElement = document.getElementById("liff-data");
const LIFF_ID = liffDataElement?.dataset.liffId?.trim() ?? "";

const statusElement = document.getElementById("status");
const refreshButtonElement = document.getElementById("refresh-button");
const activeSectionElement = document.getElementById("active-section");
const historySectionElement = document.getElementById("history-section");
const emptySectionElement = document.getElementById("empty-section");
const activeTasksElement = document.getElementById("active-tasks");
const historyTasksElement = document.getElementById("history-tasks");
const taskCardTemplate = document.getElementById("task-card-template");

const TASK_STATUS_LABELS = {
    pending: "排隊中",
    processing: "處理中",
    completed: "已完成",
    failed: "失敗",
};


function formatUpdatedAt(timestamp) {
    return new Date(timestamp * 1000).toLocaleString("zh-TW");
}


function openDownloadInExternalBrowser(downloadUrl) {
    const url = new URL(downloadUrl, window.location.origin);
    url.searchParams.set("openExternalBrowser", "1");

    if (liff.isInClient()) {
        liff.openWindow({
            url: url.toString(),
            external: true,
        });
        return;
    }

    window.location.href = url.toString();
}


function createTaskCard(task) {
    const card = taskCardTemplate.content.firstElementChild.cloneNode(true);
    const promptElement = card.querySelector(".task-prompt");
    const taskStatusElement = card.querySelector(".task-status");
    const taskInfoElement = card.querySelector(".task-info");
    const taskErrorElement = card.querySelector(".task-error");
    const downloadElement = card.querySelector(".task-download");

    promptElement.textContent = task.prompt;
    taskStatusElement.textContent =
        TASK_STATUS_LABELS[task.task_status] ?? task.task_status;
    taskStatusElement.classList.add(`task-status-${task.task_status}`);

    const versionText = task.dataset_version > 0
        ? `資料集 v${task.dataset_version}`
        : "尚未產生資料集";

    taskInfoElement.textContent =
        `${task.image_count} 張圖片・${versionText}・`
        + `更新於 ${formatUpdatedAt(task.updated_at)}`;

    if (task.error_message) {
        taskErrorElement.textContent = task.error_message;
        taskErrorElement.hidden = false;
    }

    if (task.download_url) {
        downloadElement.href = task.download_url;
        downloadElement.hidden = false;
        downloadElement.addEventListener("click", (event) => {
            event.preventDefault();
            openDownloadInExternalBrowser(task.download_url);
        });
    }

    return card;
}

let isLoadingTasks = false;


function renderTasks(tasks) {
    activeTasksElement.replaceChildren();
    historyTasksElement.replaceChildren();

    const activeTasks = tasks.filter(
        (task) => ["pending", "processing"].includes(task.task_status)
    );
    const historyTasks = tasks.filter(
        (task) => !["pending", "processing"].includes(task.task_status)
    );

    for (const task of activeTasks) {
        activeTasksElement.appendChild(createTaskCard(task));
    }

    for (const task of historyTasks) {
        historyTasksElement.appendChild(createTaskCard(task));
    }

    activeSectionElement.hidden = activeTasks.length === 0;
    historySectionElement.hidden = historyTasks.length === 0;
    emptySectionElement.hidden = tasks.length !== 0;
}


async function loadTasks(showLoading = true) {
    if (isLoadingTasks) {
        return;
    }

    isLoadingTasks = true;
    refreshButtonElement.disabled = true;

    if (showLoading) {
        statusElement.textContent = "正在載入任務......";
    }

    try {
        const idToken = liff.getIDToken();

        if (!idToken) {
            throw new Error("無法取得 LINE ID Token");
        }

        const response = await fetch("/liff/tasks", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                id_token: idToken,
            }),
        });

        const result = await response.json();

        if (
            response.status === 401
            && result.code === "LINE_ID_TOKEN_EXPIRED"
        ) {
            liff.logout();
            liff.login({
                redirectUri: window.location.href,
            });
            return;
        }

        if (!response.ok) {
            throw new Error(result.message || "無法取得任務清單");
        }

        renderTasks(result.tasks);
        statusElement.textContent =
            `共有 ${result.task_count} 個標註任務`;
    } catch (error) {
        console.error(error);
        statusElement.textContent =
            error.message || "載入任務時發生錯誤";
    } finally {
        isLoadingTasks = false;
        refreshButtonElement.disabled = false;
    }
}


async function initializeTasksPage() {
    try {
        if (!LIFF_ID) {
            throw new Error("伺服器尚未設定 LIFF_ID");
        }

        await liff.init({
            liffId: LIFF_ID,
        });

        if (!liff.isLoggedIn()) {
            liff.login({
                redirectUri: window.location.href,
            });
            return;
        }

        refreshButtonElement.hidden = false;
        await loadTasks();

        window.setInterval(() => {
            if (document.visibilityState === "visible") {
                void loadTasks(false);
            }
        }, 5000);
    } catch (error) {
        console.error(error);
        statusElement.textContent =
            error.message || "LIFF 初始化失敗";
    }
}


refreshButtonElement.addEventListener("click", () => {
    void loadTasks();
});

void initializeTasksPage();
