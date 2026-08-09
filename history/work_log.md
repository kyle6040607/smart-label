# Work Log

## [2026-08-09]
- **Task**: 修正工程師模式下「自訂模式」按鈕受全域 `.engineer-only.show` 樣式干擾導致的尺寸高低不一致問題。
- **Changes**:
  - `app/templates/index.html`：為三個按鈕統一設定 `height: 52px; min-height: 52px; box-sizing: border-box; align-items: stretch;`。
  - `app/static/css/style.css`：新增 `.train-mode-btn.engineer-only.show` 專用 CSS 覆蓋補丁，重置 `margin-top: 0 !important` 與 `padding: 8px 4px !important`，解決因外邊距/內邊距被大 Panel 規則覆蓋造成的按鈕尺寸落差。
