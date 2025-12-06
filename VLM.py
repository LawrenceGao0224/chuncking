import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json
from pdf2image import convert_from_path
import re
from io import BytesIO

class PDFToMarkdownPipeline:
    """
    改進版 PDF → Image → VLM → Markdown → Chunked Blocks
    
    改進要點：
    1. 不使用 Base64（直接傳送圖片檔案）
    2. 先合併所有頁面的 Markdown，再進行全局 Chunking（避免跨頁切分）
    3. 輸出包含完整的 Chunking 內容
    """
    
    def __init__(
        self,
        vlm_model: str = "llama3.2-vision",
        vlm_api_url: str = "http://localhost:11434/api/generate",
        output_dpi: int = 300
    ):
        self.vlm_model = vlm_model
        self.vlm_api_url = vlm_api_url
        self.output_dpi = output_dpi
    
    def pdf_to_images(self, pdf_path: str, temp_dir: str = "./temp_images") -> List[str]:
        """
        Step 1: PDF 轉圖片並保存到臨時目錄
        
        Returns:
            圖片檔案路徑列表
        """
        print(f"[1] 將 PDF 轉換為圖片: {pdf_path}")
        
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        
        images = convert_from_path(
            pdf_path,
            dpi=self.output_dpi,
            fmt='png'
        )
        
        image_paths = []
        for i, img in enumerate(images):
            img_path = Path(temp_dir) / f"page_{i+1}.png"
            img.save(img_path, format='PNG')
            image_paths.append(str(img_path))
            print(f"  ✓ 第 {i+1}/{len(images)} 頁已保存: {img_path}")
        
        return image_paths
    
    def call_vlm(self, image_path: str) -> str:
        """
        Step 2: 調用 VLM 推論（直接傳送圖片檔案）
        
        Args:
            image_path: 圖片檔案路徑
            
        Returns:
            Markdown 格式的文本
        """
        prompt = """請將這張圖片的內容轉換為 Markdown 格式。

要求：
1. 保留所有標題層級（使用 #, ##, ### 等）
2. 將表格轉換為 Markdown 表格語法（|---|）
3. 保留列表結構（使用 - 或 1. 2. 3.）
4. 忽略頁碼、頁首、頁尾
5. 如果內容看起來在頁面邊緣被截斷，請標記 [繼續於下一頁]

不要添加任何解釋，直接輸出 Markdown 內容。"""
        
        if "localhost" in self.vlm_api_url or "127.0.0.1" in self.vlm_api_url:
            return self._call_ollama_vlm(image_path, prompt)
        else:
            return self._call_remote_vlm(image_path, prompt)
    
    def _call_ollama_vlm(self, image_path: str, prompt: str) -> str:
        """調用 Ollama 本地 VLM（直接使用檔案路徑）"""
        
        # 讀取圖片為二進制數據
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Ollama 需要 base64，但我們在這裡處理，不暴露給外部
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "model": self.vlm_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
        
        try:
            response = requests.post(
                self.vlm_api_url,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            print(f"  ✗ VLM 推論失敗: {e}")
            return ""
    
    def _call_remote_vlm(self, image_path: str, prompt: str) -> str:
        """
        調用遠端 VLM API（使用 multipart/form-data 上傳圖片）
        適用於支援檔案上傳的 API
        """
        api_key = os.getenv("VLM_API_KEY", "")
        if not api_key:
            print("  ✗ 缺少 VLM_API_KEY 環境變數")
            return ""
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'model': self.vlm_model,
                'prompt': prompt
            }
            
            try:
                response = requests.post(
                    self.vlm_api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300
                )
                response.raise_for_status()
                return response.json().get('content', '')
            except Exception as e:
                print(f"  ✗ 遠端 VLM 推論失敗: {e}")
                return ""
    
    def merge_cross_page_content(self, page_markdowns: List[str]) -> str:
        """
        合併跨頁內容
        
        處理邏輯：
        1. 檢測頁面結尾的 [繼續於下一頁] 標記
        2. 移除重複的標題
        3. 合併被截斷的段落
        """
        if not page_markdowns:
            return ""
        
        merged = []
        
        for i, md in enumerate(page_markdowns):
            lines = md.strip().split('\n')
            
            # 移除 [繼續於下一頁] 標記
            cleaned_lines = [
                line for line in lines 
                if '[繼續於下一頁]' not in line and '[continued]' not in line.lower()
            ]
            
            # 如果是最後一頁，直接添加
            if i == len(page_markdowns) - 1:
                merged.extend(cleaned_lines)
            else:
                # 檢查是否有未完成的段落（最後一行不是空行、標題或列表）
                if cleaned_lines and not re.match(r'^(#{1,6}\s|[-*]\s|\d+\.\s|$)', cleaned_lines[-1]):
                    # 標記為未完成，下一頁可能接續
                    merged.extend(cleaned_lines)
                    merged.append("")  # 添加分隔符
                else:
                    merged.extend(cleaned_lines)
                    merged.append("")
        
        return '\n'.join(merged)
    
    def split_markdown_by_headers(
        self,
        markdown_text: str,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ) -> List[Dict[str, any]]:
        """
        Step 3: 全局 Header-based Markdown Chunking
        
        改進策略：
        1. 按照標題層級切分（# > ## > ###）
        2. 保持每個 chunk 的語意完整性
        3. 支援最小/最大 chunk 大小控制
        
        Returns:
            Chunk 列表，每個包含：
            {
                "chunk_id": int,
                "level": int,
                "title": str,
                "content": str,
                "word_count": int,
                "headers_path": List[str]  # 標題路徑（麵包屑）
            }
        """
        chunks = []
        lines = markdown_text.split('\n')
        
        current_chunk = {
            "level": 0,
            "title": "Document Start",
            "content_lines": [],
            "headers_path": []
        }
        
        headers_stack = []  # 用於追蹤標題層級路徑
        
        for line in lines:
            # 檢測標題
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # 保存舊 chunk（如果有內容）
                if current_chunk["content_lines"]:
                    content = '\n'.join(current_chunk["content_lines"]).strip()
                    if len(content) >= min_chunk_size or current_chunk["level"] <= 2:
                        chunks.append({
                            "chunk_id": len(chunks) + 1,
                            "level": current_chunk["level"],
                            "title": current_chunk["title"],
                            "content": content,
                            "word_count": len(content),
                            "headers_path": current_chunk["headers_path"].copy()
                        })
                
                # 更新標題路徑
                while headers_stack and headers_stack[-1]["level"] >= level:
                    headers_stack.pop()
                
                headers_stack.append({"level": level, "title": title})
                headers_path = [h["title"] for h in headers_stack]
                
                # 開始新 chunk
                current_chunk = {
                    "level": level,
                    "title": title,
                    "content_lines": [line],
                    "headers_path": headers_path
                }
            else:
                current_chunk["content_lines"].append(line)
        
        # 保存最後一個 chunk
        if current_chunk["content_lines"]:
            content = '\n'.join(current_chunk["content_lines"]).strip()
            if content:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "level": current_chunk["level"],
                    "title": current_chunk["title"],
                    "content": content,
                    "word_count": len(content),
                    "headers_path": current_chunk["headers_path"].copy()
                })
        
        # 合併過小的 chunks
        chunks = self._merge_small_chunks(chunks, min_chunk_size, max_chunk_size)
        
        return chunks
    
    def _merge_small_chunks(
        self,
        chunks: List[Dict],
        min_size: int,
        max_size: int
    ) -> List[Dict]:
        """合併過小的 chunks，避免碎片化"""
        if not chunks:
            return []
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            size = chunk["word_count"]
            
            # 如果 chunk 太小且不是頂級標題
            if size < min_size and chunk["level"] > 2:
                if buffer is None:
                    buffer = chunk
                else:
                    # 合併到 buffer
                    buffer["content"] += "\n\n" + chunk["content"]
                    buffer["word_count"] += chunk["word_count"]
                    buffer["title"] += " + " + chunk["title"]
            else:
                # 先清空 buffer
                if buffer:
                    merged.append(buffer)
                    buffer = None
                
                # 如果 chunk 太大，嘗試拆分
                if size > max_size:
                    split_chunks = self._split_large_chunk(chunk, max_size)
                    merged.extend(split_chunks)
                else:
                    merged.append(chunk)
        
        # 處理剩餘的 buffer
        if buffer:
            merged.append(buffer)
        
        # 重新編號
        for i, chunk in enumerate(merged, 1):
            chunk["chunk_id"] = i
        
        return merged
    
    def _split_large_chunk(self, chunk: Dict, max_size: int) -> List[Dict]:
        """將過大的 chunk 按段落拆分"""
        paragraphs = chunk["content"].split('\n\n')
        sub_chunks = []
        current_content = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > max_size and current_content:
                # 保存當前 sub-chunk
                sub_chunks.append({
                    "chunk_id": 0,  # 稍後重新編號
                    "level": chunk["level"],
                    "title": f"{chunk['title']} (Part {len(sub_chunks) + 1})",
                    "content": '\n\n'.join(current_content),
                    "word_count": current_size,
                    "headers_path": chunk["headers_path"]
                })
                current_content = [para]
                current_size = para_size
            else:
                current_content.append(para)
                current_size += para_size
        
        # 保存最後一個 sub-chunk
        if current_content:
            sub_chunks.append({
                "chunk_id": 0,
                "level": chunk["level"],
                "title": f"{chunk['title']} (Part {len(sub_chunks) + 1})" if sub_chunks else chunk['title'],
                "content": '\n\n'.join(current_content),
                "word_count": current_size,
                "headers_path": chunk["headers_path"]
            })
        
        return sub_chunks if sub_chunks else [chunk]
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        keep_temp_images: bool = False
    ) -> Dict:
        """
        執行完整 Pipeline
        
        流程：
        1. PDF → Images
        2. Images → Markdown (逐頁)
        3. 合併所有 Markdown (處理跨頁)
        4. 全局 Chunking (避免跨頁切分)
        
        Returns:
            {
                "full_markdown": str,
                "chunks": [...],
                "metadata": {...}
            }
        """
        print("\n" + "="*70)
        print("🚀 開始 PDF → Markdown → Chunks Pipeline (改進版)")
        print("="*70 + "\n")
        
        temp_dir = "./temp_images"
        
        # Step 1: PDF → Images
        image_paths = self.pdf_to_images(pdf_path, temp_dir)
        
        # Step 2: Images → Markdown (逐頁處理)
        print(f"\n[2] VLM 推論中...")
        page_markdowns = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"  處理第 {i}/{len(image_paths)} 頁...")
            markdown = self.call_vlm(img_path)
            
            if markdown:
                page_markdowns.append(markdown)
                print(f"    ✓ 完成 ({len(markdown)} 字元)")
            else:
                print(f"    ✗ 失敗")
        
        # Step 3: 合併跨頁內容
        print(f"\n[3] 合併 {len(page_markdowns)} 頁的 Markdown...")
        full_markdown = self.merge_cross_page_content(page_markdowns)
        print(f"  ✓ 合併完成 (總計 {len(full_markdown)} 字元)")
        
        # Step 4: 全局 Chunking
        print(f"\n[4] 執行全局語意分塊...")
        chunks = self.split_markdown_by_headers(
            full_markdown,
            min_chunk_size=100,
            max_chunk_size=2000
        )
        print(f"  ✓ 生成 {len(chunks)} 個語意 Chunks")
        
        # 構建結果
        results = {
            "full_markdown": full_markdown,
            "chunks": chunks,
            "metadata": {
                "pdf_path": pdf_path,
                "total_pages": len(image_paths),
                "total_chunks": len(chunks),
                "model": self.vlm_model,
                "dpi": self.output_dpi,
                "avg_chunk_size": sum(c["word_count"] for c in chunks) // len(chunks) if chunks else 0
            }
        }
        
        # 保存輸出
        if output_dir:
            self._save_results(pdf_path, results, output_dir)
        
        # 清理臨時圖片
        if not keep_temp_images:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n🧹 已清理臨時圖片")
        
        print(f"\n" + "="*70)
        print(f"✅ 完成!")
        print(f"   • 總頁數: {results['metadata']['total_pages']}")
        print(f"   • Chunks: {results['metadata']['total_chunks']}")
        print(f"   • 平均大小: {results['metadata']['avg_chunk_size']} 字元/chunk")
        print("="*70 + "\n")
        
        return results
    
    def _save_results(self, pdf_path: str, results: Dict, output_dir: str):
        """保存結果"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdf_name = Path(pdf_path).stem
        
        # 1. 保存完整 Markdown
        md_path = Path(output_dir) / f"{pdf_name}_full.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(results["full_markdown"])
        print(f"  💾 完整 Markdown: {md_path}")
        
        # 2. 保存 Chunks (JSON)
        chunks_json_path = Path(output_dir) / f"{pdf_name}_chunks.json"
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  💾 Chunks JSON: {chunks_json_path}")
        
        # 3. 保存 Chunks (Markdown 格式，方便閱讀)
        chunks_md_path = Path(output_dir) / f"{pdf_name}_chunks.md"
        with open(chunks_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 文檔分塊結果\n\n")
            f.write(f"**來源:** {pdf_path}\n")
            f.write(f"**總 Chunks:** {len(results['chunks'])}\n\n")
            f.write("---\n\n")
            
            for chunk in results["chunks"]:
                f.write(f"## Chunk {chunk['chunk_id']}: {chunk['title']}\n\n")
                f.write(f"**層級:** {chunk['level']} | ")
                f.write(f"**字數:** {chunk['word_count']} | ")
                f.write(f"**路徑:** {' > '.join(chunk['headers_path'])}\n\n")
                f.write("```\n")
                f.write(chunk['content'][:500])  # 預覽前 500 字元
                if len(chunk['content']) > 500:
                    f.write("\n... (略)")
                f.write("\n```\n\n")
                f.write("---\n\n")
        
        print(f"  💾 Chunks Markdown: {chunks_md_path}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 初始化 Pipeline
    pipeline = PDFToMarkdownPipeline(
        vlm_model="llama3.2-vision",  # 使用 Llama 3.2 Vision
        vlm_api_url="http://localhost:11434/api/generate",
        output_dpi=300
    )
    
    # 執行處理
    results = pipeline.process_pdf(
        pdf_path="OCR.pdf",
        output_dir="./output",
        keep_temp_images=False  # 處理完成後刪除臨時圖片
    )
    
    # 查看結果統計
    print("\n📊 結果統計:")
    print(f"  • Markdown 總長度: {len(results['full_markdown'])} 字元")
    print(f"  • Chunks 數量: {len(results['chunks'])}")
    print(f"\n📄 前 3 個 Chunks:")
    
    for chunk in results["chunks"][:3]:
        print(f"\n  [{chunk['chunk_id']}] {chunk['title']}")
        print(f"      路徑: {' > '.join(chunk['headers_path'])}")
        print(f"      內容: {chunk['content'][:100]}...")