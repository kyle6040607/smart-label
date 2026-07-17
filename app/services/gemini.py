import json
import urllib.request
import urllib.error

class GeminiService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 使用最新、速度極快的 gemini-3.5-flash 作為預設語意分析模型
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={self.api_key}"

    def parse_prompt(self, prompt: str) -> list[str]:
        """呼叫 Gemini 將複雜的中文或英文句子分析並翻譯成 YOLO-World 的英文單詞清單。
        
        若解析失敗或 API 金鑰未設定，會安全降級直接回傳 [prompt]。
        """
        if not self.api_key:
            print("💡 [Gemini] 未設定 GEMINI_API_KEY，跳過語意分析，直接使用原始 Prompt。")
            return [prompt]

        system_instruction = (
            "你是一個視覺物件偵測的語意解析專家。請將使用者輸入的中文或英文描述，分析並轉換成一組用於 YOLO-World 物件偵測的『英文具體名詞或短語清單』。\n\n"
            "請務必嚴格遵守以下規則：\n"
            "1. 只提取影像中可能出現的具體實體名詞（例如：'dog', 'red cup', 'person'），忽略虛詞、動作或修飾詞（例如不要輸出 'lying' 或 'beautiful'）。\n"
            "2. 將所有中文詞彙翻譯成高品質的英文名詞。\n"
            "3. 必須回傳合法的 JSON 格式 String Array，例如：[\"black cat\", \"sofa\", \"mug\"]。\n"
            "4. 不要回傳任何 Markdown 語法（例如 ```json）、解釋文字或問候語，只回傳 JSON 陣列本身。\n"
            "5. ⚠️【極為重要】如果使用者輸入的 Prompt 僅是一個簡單的物品名稱或單詞（例如：'草莓' 或 '貓'），請『僅僅』將其翻譯為最精準的單一英文單字即可（例如：['strawberry'] 或 ['cat']），千萬不可以擅自添加多餘的形容詞、顏色（例如不要自作聰明加上 'red strawberry'）或產生其他無關的延伸聯想詞！"
        )

        # 💡 使用 Gemini 官方推薦的 systemInstruction 與 contents 分離結構，對齊效果最好
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_instruction
                    }
                ]
            },
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            # 💡 將超時延長到 10 秒，防止網路波動導致 Socket Timeout
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
                # 提取 Gemini 回傳的文字內容
                candidates = res_data.get("candidates", [])
                if not candidates:
                    return [prompt]
                
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if not text:
                    return [prompt]
                
                # 💡 防禦性清理：移除非預期的 Markdown 標記，避免 json.loads 崩潰
                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()
                
                # 💡 防禦性清理：若大模型因重複生成輸出了多個 JSON 陣列（多行），僅保留第一行
                if "\n" in text:
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    if lines:
                        text = lines[0]
                
                # 解析 JSON 陣列
                parsed_list = json.loads(text)
                if isinstance(parsed_list, list) and all(isinstance(x, str) for x in parsed_list):
                    print(f"🎯 [Gemini] 成功將 Prompt '{prompt}' 解析為: {parsed_list}")
                    return parsed_list
                return [prompt]
        except Exception as e:
            print(f"⚠️ [Gemini] 語意解析失敗 ({e})。安全降級使用原始 Prompt。")
            return [prompt]
