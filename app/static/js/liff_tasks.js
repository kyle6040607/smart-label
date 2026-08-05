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
    retry_wait: "等待重試",
    completed: "已完成",
    failed: "失敗",
    deleting: "刪除中",
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


function getTaskRenderSignature(task) {
    return JSON.stringify([
        task.prompt,
        task.task_status,
        task.image_count,
        task.dataset_version,
        task.exported_count,
        task.excluded_count,
        task.no_detection_count,
        task.attempt_count,
        task.updated_at,
        task.error_message ?? "",
        task.download_url ?? "",
        Boolean(task.can_add_images),
        Boolean(task.can_delete),
    ]);
}


function createTaskCard(task) {
    const card = taskCardTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.taskId = task.task_id;
    card.dataset.renderSignature = getTaskRenderSignature(task);
    const promptElement = card.querySelector(".task-prompt");
    const taskStatusElement = card.querySelector(".task-status");
    const taskInfoElement = card.querySelector(".task-info");
    const taskErrorElement = card.querySelector(".task-error");
    const downloadElement = card.querySelector(".task-download");
    const addImagesElement = card.querySelector(".task-add-images");
    const excludedToggle = card.querySelector(".task-excluded-toggle");
    const excludedPanel = card.querySelector(".task-excluded-panel");
    const excludedList = card.querySelector(".task-excluded-list");
    const excludedMore = card.querySelector(".task-excluded-more");
    const deleteElement = card.querySelector(".task-delete");

    promptElement.textContent = task.prompt;
    taskStatusElement.textContent =
        TASK_STATUS_LABELS[task.task_status] ?? task.task_status;
    taskStatusElement.classList.add(`task-status-${task.task_status}`);

    const versionText = task.dataset_version > 0
        ? `資料集 v${task.dataset_version}`
        : "尚未產生資料集";

    const resultText = task.task_status === "completed"
        ? `匯出 ${task.exported_count}・未通過 ${task.excluded_count}・未偵測 ${task.no_detection_count}`
        : `第 ${task.attempt_count} 次嘗試`;

    taskInfoElement.textContent =
        `${task.image_count} 張圖片・${versionText}・${resultText}・`
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

    if (task.can_add_images) {
        addImagesElement.hidden = false;
        addImagesElement.addEventListener("click", () => {
            const uploadUrl = new URL("/liff/create", window.location.origin);
            uploadUrl.searchParams.set("append_to", task.task_id);
            window.location.href = uploadUrl.toString();
        });
    }

    if (task.can_delete) {
        deleteElement.hidden = false;
        deleteElement.textContent = task.task_status === "deleting"
            ? "重試刪除"
            : "刪除任務";
        deleteElement.addEventListener("click", async () => {
            const confirmed = window.confirm(
                "確定要永久刪除此標註任務嗎？圖片、標註、低信心縮圖與 ZIP 將一併刪除，且無法復原。"
            );
            if (!confirmed) {
                return;
            }

            const originalText = deleteElement.textContent;
            deleteElement.disabled = true;
            deleteElement.textContent = "刪除中...";
            try {
                const response = await fetch(`/liff/tasks/${task.task_id}`, {
                    method: "DELETE",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        id_token: liff.getIDToken(),
                    }),
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.message || "無法刪除標註任務");
                }
                card.remove();
                await loadTasks(false);
            } catch (error) {
                window.alert(error.message || "刪除任務時發生錯誤");
            } finally {
                if (deleteElement.isConnected) {
                    deleteElement.disabled = false;
                    deleteElement.textContent = originalText;
                }
            }
        });
    }

    if (task.excluded_count > 0) {
        excludedToggle.textContent = `查看未通過圖片（${task.excluded_count}）`;
        excludedToggle.hidden = false;
        let nextPage = 1;

        const loadPage = async () => {
            excludedMore.disabled = true;
            const response = await fetch(`/liff/tasks/${task.task_id}/excluded`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    id_token: liff.getIDToken(),
                    page: nextPage,
                }),
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.message || "無法載入未通過圖片");
            }
            for (const item of result.items) {
                const figure = document.createElement("figure");
                figure.className = "task-excluded-item";
                const image = document.createElement("img");
                image.src = item.thumbnail_url;
                image.alt = "低信心標註縮圖";
                image.loading = "lazy";
                const caption = document.createElement("figcaption");
                caption.textContent = `信心 ${(item.detection_confidence * 100).toFixed(1)}%`;
                figure.append(image, caption);
                excludedList.appendChild(figure);
            }
            nextPage += 1;
            excludedMore.hidden = !result.has_more;
            excludedMore.disabled = false;
        };

        excludedToggle.addEventListener("click", async () => {
            excludedPanel.hidden = !excludedPanel.hidden;
            if (!excludedPanel.hidden && excludedList.children.length === 0) {
                try {
                    await loadPage();
                } catch (error) {
                    excludedList.textContent = error.message;
                }
            }
        });
        excludedMore.addEventListener("click", async () => {
            try {
                await loadPage();
            } catch (error) {
                excludedList.textContent = error.message;
            }
        });
    }

    return card;
}

let isLoadingTasks = false;


function renderTasks(tasks) {
    const existingCards = new Map(
        Array.from(
            document.querySelectorAll(".task-card[data-task-id]")
        ).map((card) => [card.dataset.taskId, card])
    );

    const cardForTask = (task) => {
        const existingCard = existingCards.get(task.task_id);
        if (!existingCard) {
            return createTaskCard(task);
        }

        const excludedPanel = existingCard.querySelector(
            ".task-excluded-panel"
        );
        const isExcludedPanelOpen =
            excludedPanel && !excludedPanel.hidden;

        // 展開時保留同一個 DOM，避免輪詢清掉縮圖、分頁與展開狀態。
        if (isExcludedPanelOpen) {
            return existingCard;
        }

        if (
            existingCard.dataset.renderSignature
            === getTaskRenderSignature(task)
        ) {
            return existingCard;
        }

        return createTaskCard(task);
    };

    const activeTasks = tasks.filter(
        (task) => ["pending", "retry_wait", "processing", "deleting"].includes(task.task_status)
    );
    const historyTasks = tasks.filter(
        (task) => !["pending", "retry_wait", "processing", "deleting"].includes(task.task_status)
    );

    const activeFragment = document.createDocumentFragment();
    const historyFragment = document.createDocumentFragment();

    for (const task of activeTasks) {
        activeFragment.appendChild(cardForTask(task));
    }

    for (const task of historyTasks) {
        historyFragment.appendChild(cardForTask(task));
    }

    activeTasksElement.replaceChildren(activeFragment);
    historyTasksElement.replaceChildren(historyFragment);

    activeSectionElement.hidden = activeTasks.length === 0;
    historySectionElement.hidden = historyTasks.length === 0;
    emptySectionElement.hidden = tasks.length !== 0;
}


async function loadTasks(showLoading = true) {
    if (isLoadingTasks) {
        return;
    }

    isLoadingTasks = true;

    if (showLoading) {
        refreshButtonElement.disabled = true;
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
        if (showLoading) {
            refreshButtonElement.disabled = false;
        }
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
