# Required libraries: Install them via pip if needed
# pip install pytesseract pillow opencv-python PyMuPDF sentence-transformers torch torchvision
# Also, install Tesseract OCR: https://github.com/tesseract-ocr/tesseract

import fitz  # PyMuPDF for PDF handling
import cv2
import numpy as np
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import urllib.error
import ssl

# Global flag to track if advanced model failed (avoid repeated download attempts)
_advanced_model_failed = False

# Step 1: Load PDF and extract pages as images for OCR and CV analysis
def load_pdf_as_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    doc.close()
    return images

# Step 2: Use OCR to extract text from each page image
def ocr_extract_text(image):
    # Preprocess image for better OCR: grayscale, threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang='eng')  # Change lang as needed
    return text.strip()

# Step 3: Use Computer Vision to detect chunks (e.g., paragraphs/sections via layout analysis)
def detect_chunks_with_cv(image):
    # Use OpenCV to detect contours or lines for layout
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find potential text blocks (paragraphs)
    text_blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Arbitrary thresholds for text blocks
            text_blocks.append((x, y, w, h))
    
    # Sort blocks top-to-bottom, left-to-right
    text_blocks.sort(key=lambda b: (b[1], b[0]))
    
    # Extract text for each block using OCR on cropped image
    chunks = []
    for x, y, w, h in text_blocks:
        crop = image[y:y+h, x:x+w]
        chunk_text = ocr_extract_text(crop)
        if chunk_text:
            chunks.append(chunk_text)
    
    # Fallback: if no chunks found, extract text from whole page
    if not chunks:
        full_text = ocr_extract_text(image)
        if full_text:
            # Split into paragraphs as chunks
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            chunks.extend(paragraphs)
            # If no paragraph breaks, split by newlines
            if not chunks:
                lines = [l.strip() for l in full_text.split('\n') if l.strip()]
                chunks.extend(lines)
    
    return chunks

# Advanced CV option: Use object detection for layout elements (requires torch)
def advanced_detect_chunks(image):
    global _advanced_model_failed
    
    # Skip trying to load model if it already failed
    if _advanced_model_failed:
        return detect_chunks_with_cv(image)
    
    try:
        # Load pre-trained Faster R-CNN for object detection (adapt for text regions if model fine-tuned)
        # Use weights parameter instead of deprecated pretrained parameter
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval()
        
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(Image.fromarray(image))
        with torch.no_grad():
            predictions = model([img_tensor])
        
        # Predictions include boxes for detected objects; filter for potential text areas
        # This is simplistic; in practice, use a layout model like LayoutLM or fine-tune
        boxes = predictions[0]['boxes'].cpu().numpy()
        chunks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            chunk_text = ocr_extract_text(crop)
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    except (urllib.error.URLError, ssl.SSLError, Exception) as e:
        # Mark model as failed and fall back to basic CV method
        _advanced_model_failed = True
        print(f"Warning: Advanced detection failed ({type(e).__name__}), falling back to basic CV method for all pages")
        return detect_chunks_with_cv(image)

# Step 4: Merge related chunks across pages using sentence similarity
def merge_related_chunks(all_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    merged_chunks = []
    current_chunk = ""
    
    for i, chunk in enumerate(all_chunks):
        if not current_chunk:
            current_chunk = chunk
            continue
        
        # Compute similarity between last sentence of current and first of next
        last_sent = current_chunk.split('.')[-1].strip()
        first_sent = chunk.split('.')[0].strip()
        
        if last_sent and first_sent:
            embeddings = model.encode([last_sent, first_sent])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            if similarity > 0.7:  # Threshold for relatedness
                current_chunk += " " + chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        else:
            current_chunk += " " + chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    return merged_chunks

# Step 5: RAG Pipeline (simplified: chunking + embedding + retrieval)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rag_pipeline(pdf_path, query):
    # Load and process pages
    images = load_pdf_as_images(pdf_path)
    
    # Extract chunks per page using CV
    all_chunks = []
    for img in images:
        # chunks = detect_chunks_with_cv(img)  # Basic CV
        chunks = advanced_detect_chunks(img)  # Advanced with torch
        all_chunks.extend(chunks)
    
    # Fallback: if no chunks found, extract text from whole pages
    if not all_chunks:
        print("Warning: No chunks detected, extracting text from full pages as fallback")
        for img in images:
            full_text = ocr_extract_text(img)
            if full_text:
                # Split into paragraphs as chunks
                paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
                all_chunks.extend(paragraphs)
    
    # If still no chunks, return empty list
    if not all_chunks:
        print("Error: No text could be extracted from the PDF")
        return []
    
    # Merge related cross-page chunks
    merged_chunks = merge_related_chunks(all_chunks)
    
    # Filter out empty chunks
    merged_chunks = [chunk for chunk in merged_chunks if chunk.strip()]
    
    if not merged_chunks:
        print("Error: All chunks are empty after merging")
        return []
    
    # Embed chunks (simple TF-IDF for demo; use embeddings in production)
    try:
        vectorizer = TfidfVectorizer(min_df=1, token_pattern=r'\b\w+\b')
        chunk_vectors = vectorizer.fit_transform(merged_chunks)
        
        # Embed query
        query_vector = vectorizer.transform([query])
        
        # Retrieve top chunks
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = np.argsort(similarities)[-3:]  # Top 3
        retrieved = [merged_chunks[i] for i in top_indices]
        
        return retrieved
    except ValueError as e:
        # Handle empty vocabulary error
        print(f"Warning: TF-IDF vectorization failed ({e}), returning all chunks")
        return merged_chunks[:3]  # Return first 3 chunks as fallback

# Example usage
pdf_path = "OCR.pdf"
query = "Your search query here"
retrieved_chunks = rag_pipeline(pdf_path, query)
print("Retrieved chunks for RAG:")
for chunk in retrieved_chunks:
    print(chunk)
    print("---")