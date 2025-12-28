from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import pytesseract
import shutil
import requests
import torch
import re
import gc
import os
from typing import Optional, Dict, List
from functools import lru_cache

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("Tesseract is not installed. OCR may not work.")

app = FastAPI(title="AI Image Metadata API", version="2.0")

@lru_cache(maxsize=1)
def load_models():
    gc.collect()
    try:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir="./model_cache"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir="./model_cache",
            torch_dtype=torch.float32
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        return processor, model
    except Exception as e:
        return None, None

blip_processor, blip_model = load_models()

def detect_document_type(text: str) -> str:
    if not text:
        return "unknown"
    text_lower = text.lower()
    if any(word in text_lower for word in ["invoice", "bill", "total", "amount", "due"]):
        return "invoice"
    if any(word in text_lower for word in ["sale", "offer", "discount", "promotion", "event"]):
        return "advertisement"
    if any(word in text_lower for word in ["post", "announcement", "news", "update"]):
        return "post"
    if any(word in text_lower for word in ["card", "business", "contact", "email"]):
        return "business_card"
    if any(word in text_lower for word in ["id", "passport", "license", "identification"]):
        return "id_card"
    return "unknown"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    text = re.sub(r"[^\w\s.,:;&()\-'\"?!]", " ", text)
    return text.strip()

def fix_joined_words(text: str) -> str:
    fixes = [
        ("sistersweeklyhalaqa", "Sisters Weekly Halaqa"),
        ("journeythroughthequran", "Journey Through The Quran"),
        ("atthenoor", "At The Noor"),
        ("startingseptember", "Starting September"),
        ("everysunday", "Every Sunday"),
        ("weeklyhalaqa", "Weekly Halaqa"),
    ]
    for bad, good in fixes:
        text = text.replace(bad, good)
        text = text.replace(bad.lower(), good.lower())
    return text

def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    if gray.width < 600 or gray.height < 400:
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
    gray = gray.point(lambda x: 0 if x < 128 else 255, "1")
    return gray

def extract_event_info(text: str) -> Dict[str, Optional[str]]:
    result = {
        "event_name": None,
        "event_date": None,
        "event_time": None,
        "location": None,
        "presenter": None
    }
    if not text:
        return result
    name_patterns = [
        r"sisters\s+weekly\s+halaqa",
        r"journey\s+through\s+the\s+quran",
        r"event\s*[:\-]?\s*([^\n]{5,50})",
        r"title\s*[:\-]?\s*([^\n]{5,50})",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) > 0:
                result["event_name"] = match.group(1).strip()
            else:
                result["event_name"] = match.group().strip()
            break
    date_patterns = [
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d{1,2}-\d{1,2}-\d{2,4}",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[,.]?\s+\d{4}",
        r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["event_date"] = match.group()
            break
    time_patterns = [
        r"\b(1[0-2]|0?[1-9]):[0-5][0-9]\s?[-to]\s?(1[0-2]|0?[1-9]):[0-5][0-9]\s?(am|pm)\b",
        r"\b(1[0-2]|0?[1-9]):[0-5][0-9]\s?(am|pm)\b",
        r"\b\d{1,2}\s?(am|pm)\b",
    ]
    for pattern in time_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["event_time"] = match.group()
            break
    location_match = re.search(r"(at|@|location)\s+([^\n]{3,50})", text, re.IGNORECASE)
    if location_match:
        result["location"] = location_match.group(2).strip()
    speaker_match = re.search(r"(with|by|speaker|presenter)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", text, re.IGNORECASE)
    if speaker_match:
        result["presenter"] = speaker_match.group(2).strip()
    return result

def generate_caption(image: Image.Image) -> str:
    if blip_processor is None or blip_model is None:
        return "AI model not available"
    try:
        if image.width > 512 or image.height > 512:
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
        inputs = blip_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50, num_beams=3)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return "Error generating caption"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            raise HTTPException(400, "File must be an image")
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 10MB)")
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        metadata = {
            "filename": file.filename,
            "dimensions": f"{image.width}x{image.height}",
            "format": image.format or "Unknown",
            "size_kb": len(image_data) // 1024
        }
        processed_image = preprocess_image(image)
        try:
            ocr_text = pytesseract.image_to_string(processed_image, lang='eng')
            ocr_text = fix_joined_words(clean_text(ocr_text))
        except:
            ocr_text = ""
        caption = generate_caption(image)
        event_info = extract_event_info(ocr_text)
        doc_type = detect_document_type(ocr_text)
        gc.collect()
        return {
            "status": "success",
            "metadata": metadata,
            "ocr_text": ocr_text,
            "ai_caption": caption,
            "document_type": doc_type,
            "event_info": event_info,
            "has_text": bool(ocr_text.strip())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/analyze-url")
async def analyze_image_url(url: str):
    try:
        if not url.startswith(('http://', 'https://')):
            url = "https://zaq23store.s3.amazonaws.com/" + url.lstrip('/')
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        if len(response.content) > 10 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 10MB)")
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        metadata = {
            "source_url": url,
            "dimensions": f"{image.width}x{image.height}",
            "format": image.format or "Unknown"
        }
        processed_image = preprocess_image(image)
        try:
            ocr_text = pytesseract.image_to_string(processed_image, lang='eng')
            ocr_text = fix_joined_words(clean_text(ocr_text))
        except:
            ocr_text = ""
        caption = generate_caption(image)
        event_info = extract_event_info(ocr_text)
        doc_type = detect_document_type(ocr_text)
        gc.collect()
        return {
            "status": "success",
            "metadata": metadata,
            "ocr_text": ocr_text,
            "ai_caption": caption,
            "document_type": doc_type,
            "event_info": event_info
        }
    except requests.RequestException:
        raise HTTPException(400, "Failed to download image")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            processed_image = preprocess_image(image)
            try:
                ocr_text = pytesseract.image_to_string(processed_image, lang='eng')
                ocr_text = fix_joined_words(clean_text(ocr_text))
            except:
                ocr_text = ""
            event_info = extract_event_info(ocr_text)
            results.append({
                "filename": file.filename,
                "ocr_text": ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text,
                "event_info": event_info,
                "status": "success"
            })
            gc.collect()
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    return {
        "processed_count": len(results),
        "results": results
    }

@app.get("/health")
async def health_check():
    import psutil
    process = psutil.Process()
    return {
        "status": "healthy",
        "service": "AI Image Metadata API",
        "version": "2.0",
        "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "ai_model_loaded": blip_processor is not None,
        "ocr_available": tesseract_path is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Image Metadata API",
        "version": "2.0",
        "endpoints": {
            "POST /analyze": "Analyze a single image",
            "POST /analyze-url": "Analyze image from URL",
            "POST /batch-analyze": "Analyze batch of images",
            "GET /health": "Health check",
            "GET /docs": "Interactive API docs (Swagger)"
        },
        "note": "Optimized for low-memory servers (2GB RAM)"
    }

@app.on_event("startup")
async def startup():
    pass

@app.on_event("shutdown")
async def shutdown():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )