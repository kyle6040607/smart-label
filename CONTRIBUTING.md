# 團隊協作指南（Git 新手版）

這份文件給 Smart Label 專案的四位組員看。**第一次用 Git 也沒關係**，照著一步步做就好。

---

## 分工建議（照模組分，最不會吵架）

| 組員 | 負責部分 | 主要檔案資料夾 |
|------|----------|----------------|
| A    | AI 影像切割 | `app/ml/` |
| B    | 後端網址路由 | `app/routes/` |
| C    | 服務與資料庫 | `app/services/`、`app/repository.py` |
| D    | 前端網頁畫面 | `app/static/`、`app/templates/` |

> **黃金守則：盡量不要兩個人同時改同一個檔案。**
> 各顧各的資料夾，合併時就幾乎不會出問題。

## 分支命名規則

格式：`類型/簡短英文描述`（小寫，用 `-` 連接）

| 類型 | 用途 | 範例 |
|------|------|------|
| `feature/` | 新功能 | `feature/sam-model`、`feature/upload-ui` |
| `fix/` | 修 bug | `fix/floodfill-error` |
| `docs/` | 改文件 | `docs/readme-update` |

❌ 不要取 `test`、`mybranch`、`abc` 這種看不出在做什麼的名字。

---

## 每天工作的標準流程（最常用，請背起來）

### 步驟 1：先更新 main（拿到別人最新的成果）
```bash
git checkout main          # 切換到 main 分支
git pull origin main       # 把 GitHub 上最新的 main 下載下來
```
> `checkout` = 切換分支。`pull` = 從 GitHub 下載最新版本。
> **每天開工第一件事就是這兩行**，不然你會用到過時的程式。

### 步驟 2：開一條自己的新分支
```bash
git checkout -b feature/sam-model
```
> `-b` = 建立並切換到新分支。名字換成你這次要做的任務。

### 步驟 3：寫程式
正常用 VS Code 改檔案、寫功能。改多久都行。

### 步驟 4：把改動記錄下來（commit）
```bash
git add .                          # 把所有改動加入「準備提交」清單
git commit -m "接上 SAM 模型載入"   # 提交，並寫一句說明做了什麼
```
> `add .` 的點代表「全部檔案」。
> `commit -m` 後面引號內寫**你這次做了什麼**，中文沒關係，但要講清楚。
> ✅ 好範例：`"新增圖片上傳功能"`
> ❌ 壞範例：`"改了一些東西"`、`"123"`、`"update"`

可以一邊做一邊 commit，分很多次小提交比一次大提交好。

### 步驟 5：推到 GitHub（push）
```bash
git push origin feature/sam-model
```
> `push` = 把你電腦上的分支上傳到 GitHub。分支名字要跟步驟 2 開的一樣。

### 步驟 6：開 Pull Request（在 GitHub 網頁操作）
1. 打開 https://github.com/kyle6040607/smart-label
2. 會看到黃色提示「Compare & pull request」，點它。
3. 寫標題和說明（做了什麼功能）。
4. 點「Create pull request」。
5. **找另一位組員看過、按 Merge**（不要自己合併自己的）。

### 步驟 7：合併後，回到 main 準備下一個任務
```bash
git checkout main
git pull origin main              # 拿到剛剛合併進去的新功能
git checkout -b feature/下一個任務  # 開新分支，重複步驟 3~6
```

---

## 常見問題（卡住時看這裡）

### Q1：出現 CONFLICT（衝突）怎麼辦？
代表你和別人改到同一段程式，Git 不知道要留誰的。

```bash
git checkout feature/你的分支
git pull origin main      # 把 main 最新的併進來，這時可能跳出 CONFLICT
```

打開 VS Code，被衝突的檔案會出現這種記號：
```
&lt;&lt;&lt;&lt;&lt;&lt;&lt; HEAD
你寫的內容
=== 分隔線 ===
別人寫的內容
&gt;&gt;&gt;&gt;&gt;&gt;&gt; main
```
1. 決定要留哪一段（或兩段都留），**手動刪掉 `<<<<<<<`、`=======`、`>>>>>>>` 這些記號**。
2. 存檔後執行：
```bash
git add .
git commit -m "解決衝突"
git push
```
> 不確定怎麼選的時候，**先問改到同一塊的那位組員**，不要亂刪。

### Q2：我不小心在 main 上改了東西怎麼辦？
還沒 commit 的話：
```bash
git stash                       # 把改動暫存起來
git checkout -b feature/補救分支  # 開正確的分支
git stash pop                   # 把改動拿回來
```
> 不確定就直接問組長，**不要硬 push 到 main**。

### Q3：怎麼看現在在哪個分支？
```bash
git status        # 會顯示 "On branch xxx"
git branch        # 列出所有分支，* 號代表你目前所在
```

---

## 四條鐵則（貼在牆上那種）

1. **永遠不要直接 push 到 `main`** —— 一定走分支 + PR。
2. **每天開工先 `git pull origin main`** —— 確保用的是最新版。
3. **一個任務一條分支，做完就合併** —— 不要一條分支拖兩週。
4. **commit 訊息寫清楚做了什麼** —— 未來的你會感謝現在的你。

---

## 指令小抄（剪下來貼桌面）

```bash
# === 開始新任務 ===
git checkout main
git pull origin main
git checkout -b feature/任務名稱

# === 存檔上傳 ===
git add .
git commit -m "說明你做了什麼"
git push origin feature/任務名稱
# 然後去 GitHub 開 Pull Request

# === 查看狀態 ===
git status        # 我在哪個分支？改了什麼？
git branch        # 列出所有分支
git log --oneline # 看歷史提交紀錄
```

有任何看不懂的，先在群組問，不要自己亂試導致專案壞掉。一起加油 💪
