"""
RAG Document Chunking System
支援 PDF 和 Word 文件的智能分割系統
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import tiktoken
import PyPDF2
from docx import Document
import io
import os
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="RAG Chunking API", version="1.0.0")

# Pydantic 模型
class ChunkResult(BaseModel):
    chunk_id: int
    content: str
    token_count: int
    start_char: int
    end_char: int

class ChunkingResponse(BaseModel):
    filename: str
    total_chunks: int
    total_tokens: int
    chunk_size: int
    overlap: int
    chunks: List[ChunkResult]
    output_file: str

class DocumentChunker:
    """文件分割器"""
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 100):
        """
        初始化分割器
        
        Args:
            chunk_size: 每個 chunk 的 token 數量
            overlap: chunk 之間的重疊 token 數量
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        return len(self.encoding.encode(text))
    
    def read_pdf(self, file_bytes: bytes) -> str:
        """讀取 PDF 文件"""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def read_docx(self, file_bytes: bytes) -> str:
        """讀取 Word 文件"""
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def chunk_text(self, text: str) -> List[dict]:
        """
        將文本分割成 chunks
        
        Args:
            text: 要分割的文本
            
        Returns:
            包含 chunk 資訊的字典列表
        """
        # 按段落分割
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        char_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = self.count_tokens(paragraph)
            
            # 如果單個段落超過 chunk_size，需要進一步分割
            if paragraph_tokens > self.chunk_size:
                # 先保存當前 chunk
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'token_count': current_tokens,
                        'start_char': char_position - len(current_chunk),
                        'end_char': char_position
                    })
                    current_chunk = ""
                    current_tokens = 0
                
                # 分割長段落
                sentences = paragraph.split('。')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_tokens = self.count_tokens(sentence + '。')
                    
                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'token_count': current_tokens,
                                'start_char': char_position - len(current_chunk),
                                'end_char': char_position
                            })
                        current_chunk = sentence + '。'
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += sentence + '。'
                        current_tokens += sentence_tokens
                    
                    char_position += len(sentence) + 1
            
            # 如果加入此段落會超過 chunk_size
            elif current_tokens + paragraph_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'token_count': current_tokens,
                        'start_char': char_position - len(current_chunk),
                        'end_char': char_position
                    })
                
                # 處理 overlap
                if self.overlap > 0 and chunks:
                    # 從前一個 chunk 取最後幾個句子作為 overlap
                    last_chunk = chunks[-1]['content']
                    sentences = last_chunk.split('。')
                    overlap_text = '。'.join(sentences[-2:]) if len(sentences) > 1 else last_chunk
                    overlap_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_tokens <= self.overlap:
                        current_chunk = overlap_text + '\n' + paragraph
                        current_tokens = self.count_tokens(current_chunk)
                    else:
                        current_chunk = paragraph
                        current_tokens = paragraph_tokens
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                current_chunk += '\n' + paragraph
                current_tokens += paragraph_tokens
            
            char_position += len(paragraph) + 1
        
        # 添加最後一個 chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'start_char': char_position - len(current_chunk),
                'end_char': char_position
            })
        
        return chunks
    
    def process_file(self, file_bytes: bytes, filename: str) -> tuple[str, List[dict]]:
        """
        處理文件並分割
        
        Args:
            file_bytes: 文件字節
            filename: 文件名稱
            
        Returns:
            (文本內容, chunks 列表)
        """
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = self.read_pdf(file_bytes)
        elif file_ext in ['docx', 'doc']:
            text = self.read_docx(file_bytes)
        else:
            raise ValueError(f"不支援的文件格式: {file_ext}")
        
        chunks = self.chunk_text(text)
        return text, chunks
    
    def save_chunks_to_file(self, chunks: List[dict], filename: str, output_dir: str = "output") -> str:
        """
        將 chunks 保存到文本文件
        
        Args:
            chunks: chunk 列表
            filename: 原始文件名
            output_dir: 輸出目錄
            
        Returns:
            輸出文件路徑
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_chunks_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"文件: {filename}\n")
            f.write(f"分割時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Chunk 大小: {self.chunk_size} tokens\n")
            f.write(f"Overlap: {self.overlap} tokens\n")
            f.write(f"總 Chunks 數: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                f.write(f"{'='*80}\n")
                f.write(f"Chunk ID: {i}\n")
                f.write(f"Token 數量: {chunk['token_count']}\n")
                f.write(f"字符位置: {chunk['start_char']} - {chunk['end_char']}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{chunk['content']}\n")
                f.write(f"{'='*80}\n\n")
        
        return output_file


# FastAPI 端點
@app.get("/")
async def root():
    """API 根路徑"""
    return {
        "message": "RAG Document Chunking API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chunk": "上傳文件進行分割",
            "GET /health": "健康檢查"
        }
    }

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chunk", response_model=ChunkingResponse)
async def chunk_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1024),
    overlap: int = Form(100)
):
    """
    上傳文件並進行分割
    
    Args:
        file: 上傳的文件 (PDF 或 Word)
        chunk_size: 每個 chunk 的 token 數量 (預設: 1024)
        overlap: chunk 之間的重疊 token 數量 (預設: 100)
        
    Returns:
        分割結果
    """
    try:
        # 讀取文件
        file_bytes = await file.read()
        
        # 檢查文件格式
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['pdf', 'docx', 'doc']:
            raise HTTPException(
                status_code=400, 
                detail=f"不支援的文件格式: {file_ext}。僅支援 PDF 和 Word 文件。"
            )
        
        # 創建分割器
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        
        # 處理文件
        text, chunks = chunker.process_file(file_bytes, file.filename)
        
        # 保存結果
        output_file = chunker.save_chunks_to_file(chunks, file.filename)
        
        # 準備回應
        total_tokens = sum(chunk['token_count'] for chunk in chunks)
        chunk_results = [
            ChunkResult(
                chunk_id=i+1,
                content=chunk['content'],
                token_count=chunk['token_count'],
                start_char=chunk['start_char'],
                end_char=chunk['end_char']
            )
            for i, chunk in enumerate(chunks)
        ]
        
        return ChunkingResponse(
            filename=file.filename,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            overlap=overlap,
            chunks=chunk_results,
            output_file=output_file
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理文件時發生錯誤: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    下載生成的 chunks 文件
    
    Args:
        filename: 文件名稱
        
    Returns:
        文件下載
    """
    file_path = os.path.join("output", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/plain'
    )


if __name__ == "__main__":
    import uvicorn
    
    # 創建輸出目錄
    os.makedirs("output", exist_ok=True)
    
    print("啟動 RAG Document Chunking API...")
    print("API 文檔: http://localhost:8000/docs")
    print("ReDoc 文檔: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)