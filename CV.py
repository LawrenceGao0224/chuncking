import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from pdf2image import convert_from_path
from PIL import Image
import re

class ImprovedCVDocumentChunker:
    def __init__(
        self,
        output_dpi: int = 300,
        tesseract_lang: str = 'chi_tra+eng',
        min_title_font_ratio: float = 1.3,
        debug_mode: bool = False
    ):
        self.output_dpi = output_dpi
        self.tesseract_lang = tesseract_lang
        self.min_title_font_ratio = min_title_font_ratio
        self.debug_mode = debug_mode
        self.avg_font_size = 12.0

    def preprocess_image_for_ocr(self, image_path: str) -> Image.Image:
        """
        [改進點 1] 影像預處理：增強對比度與二值化，大幅提升 OCR 準確率
        """
        # 讀取圖片 (OpenCV 格式)
        img = cv2.imread(image_path)
        
        # 1. 轉灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 放大圖片 (如果字太小，放大有助於辨識)
        # 如果解析度夠高(300dpi)通常不需要，但針對模糊文件可以開啟
        # gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        # 3. 自適應二值化 (去除陰影和雜訊，讓文字變純黑，背景變純白)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. 去除微小雜訊 (可選，視情況調整)
        # kernel = np.ones((1, 1), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 轉換回 PIL 格式供 Tesseract 使用
        return Image.fromarray(binary)

    def pdf_to_images(self, pdf_path: str, temp_dir: str = "./temp_images") -> List[str]:
        print(f"[1] 將 PDF 轉換為圖片: {pdf_path}")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        # 提高 DPI 到 350 或 400 可以進一步提升小字辨識率，但速度變慢
        images = convert_from_path(pdf_path, dpi=self.output_dpi, fmt='png')
        image_paths = []
        for i, img in enumerate(images):
            img_path = Path(temp_dir) / f"page_{i+1}.png"
            img.save(img_path, format='PNG')
            image_paths.append(str(img_path))
        return image_paths

    def extract_text_lines_with_layout(self, image_path: str) -> List[Dict]:
        """
        [改進點 2] 提取文字：移除信心度過濾，避免漏字
        """
        # 使用預處理後的圖片
        preprocessed_img = self.preprocess_image_for_ocr(image_path)
        
        ocr_data = pytesseract.image_to_data(
            preprocessed_img,
            lang=self.tesseract_lang,
            output_type=pytesseract.Output.DICT,
            # psm 6 假設是一個統一的文字塊，適合大多數段落
            # 如果是複雜多欄排版，保持預設 (psm 3)
            config='--psm 3' 
        )
        
        lines_dict = {}
        
        for i in range(len(ocr_data['text'])):
            # [重要修改] 移除信心度過濾 (conf < 30)，避免因為字跡模糊導致整句消失
            # 只過濾空字串
            word = ocr_data['text'][i].strip()
            if not word:
                continue
            
            # 使用 block_num 和 line_num 確保同一行的字在一起
            block_num = ocr_data['block_num'][i]
            line_num = ocr_data['line_num'][i]
            key = (block_num, line_num)
            
            if key not in lines_dict:
                lines_dict[key] = {
                    "words": [],
                    "x_coords": [],
                    "y_coords": [],
                    "heights": [],
                    "confidences": [] # 僅做紀錄，不過濾
                }
            
            lines_dict[key]["words"].append(word)
            lines_dict[key]["x_coords"].append(ocr_data['left'][i])
            lines_dict[key]["y_coords"].append(ocr_data['top'][i])
            lines_dict[key]["heights"].append(ocr_data['height'][i])
            lines_dict[key]["confidences"].append(ocr_data['conf'][i])
        
        text_lines = []
        for key in sorted(lines_dict.keys(), key=lambda k: (k[0], k[1])):
            line_data = lines_dict[key]
            
            # 處理文字拼接 (英文加空格，中文理論上不用但加了也沒關係，後續清洗可處理)
            text = ' '.join(line_data["words"])
            
            x = min(line_data["x_coords"])
            y = min(line_data["y_coords"])
            # 寬度計算修正
            w = max([x + w for x, w in zip(line_data["x_coords"], [ocr_data['width'][i] for i in range(len(ocr_data['text'])) if ocr_data['block_num'][i] == key[0] and ocr_data['line_num'][i] == key[1]])]) - x
            h = int(np.mean(line_data["heights"])) # 平均字高
            
            # 過濾極度過小的雜訊行 (例如只有 1-2 pixel 高)
            if h < 5: 
                continue

            text_lines.append({
                "text": text,
                "bbox": (x, y, w, h),
                "font_size": h,
                "line_num": len(text_lines)
            })
            
        return text_lines

    def group_lines_into_paragraphs(self, text_lines: List[Dict], page_height: int) -> List[Dict]:
        """
        [改進點 3] 段落合併：使用動態行高閾值，而非固定像素
        """
        if not text_lines:
            return []
        
        # 預先計算頁面平均字體大小，作為基準
        all_font_sizes = [l["font_size"] for l in text_lines]
        global_avg_font_size = np.mean(all_font_sizes) if all_font_sizes else 12
        self.avg_font_size = global_avg_font_size

        paragraphs = []
        current_para = {
            "lines": [],
            "type": "paragraph",
            "level": 0
        }
        
        for i, line in enumerate(text_lines):
            should_start_new = False
            
            if current_para["lines"]:
                prev_line = current_para["lines"][-1]
                
                # 計算垂直間距
                prev_bottom = prev_line["bbox"][1] + prev_line["bbox"][3]
                current_top = line["bbox"][1]
                vertical_gap = current_top - prev_bottom
                
                # 取得當前行的字高
                current_line_height = line["font_size"]
                
                # --- 動態判斷邏輯 ---
                
                # 1. 動態間距閾值：如果間距大於 1.5 倍行高，視為新段落 (比固定 30px 更準)
                # 對於標題，通常間距會更大，所以這也能抓到
                dynamic_threshold = max(current_line_height, prev_line["font_size"]) * 1.5
                if vertical_gap > dynamic_threshold:
                    should_start_new = True
                
                # 2. 字體大小變化顯著 (超過 20% 變化即視為不同區塊)
                font_ratio = line["font_size"] / prev_line["font_size"]
                if font_ratio > 1.2 or font_ratio < 0.8:
                    should_start_new = True
                
                # 3. 檢查是否為列表開頭 (如果前一行不是列表，這行是，則斷開)
                if self._is_list_item(line["text"]) and not self._is_list_item(prev_line["text"]):
                    should_start_new = True

            if should_start_new and current_para["lines"]:
                paragraph = self._finalize_paragraph(current_para, page_height)
                if paragraph:
                    paragraphs.append(paragraph)
                current_para = {"lines": [line], "type": "paragraph", "level": 0}
            else:
                current_para["lines"].append(line)
        
        # 處理最後一段
        if current_para["lines"]:
            paragraph = self._finalize_paragraph(current_para, page_height)
            if paragraph:
                paragraphs.append(paragraph)
        
        return paragraphs

    def _is_list_item(self, text: str) -> bool:
        """簡單判斷是否為列表項目"""
        return bool(re.match(r'^[-•●○■□▪▫\d+\.]\s', text.strip()))

    def _finalize_paragraph(self, para_data: Dict, page_height: int) -> Optional[Dict]:
        lines = para_data["lines"]
        if not lines: return None
        
        # 智慧文字合併：處理英文與中文的空格問題
        # 這裡簡化處理，統一用空格連接，後續可以用 LLM 清洗
        text = ' '.join(line["text"] for line in lines)
        
        # bbox 計算
        x_min = min(line["bbox"][0] for line in lines)
        y_min = min(line["bbox"][1] for line in lines)
        x_max = max(line["bbox"][0] + line["bbox"][2] for line in lines)
        y_max = max(line["bbox"][1] + line["bbox"][3] for line in lines)
        
        # 標題判斷
        avg_size = np.mean([line["font_size"] for line in lines])
        para_type = "paragraph"
        level = 0
        
        if avg_size > self.avg_font_size * self.min_title_font_ratio:
            para_type = "title"
            level = 1 if avg_size > self.avg_font_size * 1.8 else 2

        # 跨頁判斷 (邏輯不變)
        last_line = lines[-1]
        y_pos = last_line["bbox"][1] + last_line["bbox"][3]
        is_continued = (y_pos > page_height * 0.9) and not bool(re.search(r'[。.!?！？]$', text.strip()))

        return {
            "type": para_type,
            "level": level,
            "text": text.strip(),
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "is_continued": is_continued,
            "font_size": avg_size,
            "lines": lines # 保留原始行以便除錯或進階合併
        }

    # --- 以下保留原有的跨頁合併 (merge_cross_page_paragraphs) 和 Chunk 產生邏輯 (create_chunks) ---
    # 因為這些邏輯本身是正確的，問題出在上游的資料品質
    
    def merge_cross_page_paragraphs(self, all_pages_paragraphs: List[List[Dict]]) -> List[Dict]:
        # (與原程式碼相同，省略以節省篇幅)
        if not all_pages_paragraphs: return []
        merged = []
        pending = None
        for page_num, page_paras in enumerate(all_pages_paragraphs):
            if not page_paras: continue
            for i, para in enumerate(page_paras):
                if pending:
                    if i == 0 and para["type"] == pending["type"]:
                        para["text"] = pending["text"] + " " + para["text"]
                        para["pages"] = [pending.get("page", page_num), page_num + 1]
                        pending = None
                    else:
                        merged.append(pending)
                        pending = None
                
                if "pages" not in para: para["page"] = page_num + 1
                
                if para["is_continued"] and i == len(page_paras) - 1:
                    pending = para
                else:
                    merged.append(para)
        if pending: merged.append(pending)
        return merged

    def process_pdf(self, pdf_path: str, output_dir: str = "output"):
        # 簡化的執行流程
        image_paths = self.pdf_to_images(pdf_path)
        all_paragraphs = []
        
        print("\n[2] 開始 OCR 與段落分析...")
        for img_path in image_paths:
            lines = self.extract_text_lines_with_layout(img_path)
            # 讀取圖片高度
            h, w = cv2.imread(img_path).shape[:2]
            paras = self.group_lines_into_paragraphs(lines, h)
            all_paragraphs.append(paras)
            print(f"  - 頁面處理完成: {len(paras)} 個段落")
            
        merged_paras = self.merge_cross_page_paragraphs(all_paragraphs)
        
        # 簡單輸出 markdown
        Path(output_dir).mkdir(exist_ok=True)
        with open(f"{output_dir}/result.md", "w", encoding="utf-8") as f:
            for p in merged_paras:
                prefix = "#" * p["level"] + " " if p["type"] == "title" else ""
                f.write(f"{prefix}{p['text']}\n\n")
        
        print(f"\n✅ 完成！結果已保存至 {output_dir}/result.md")
        return merged_paras

# 使用方式
if __name__ == "__main__":
    chunker = ImprovedCVDocumentChunker()
    chunker.process_pdf("OCR.pdf")