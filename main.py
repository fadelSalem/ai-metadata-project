from fastapi import FastAPI, UploadFile, File
from transformers import BlipProcessor, BlipForConditionalGeneration, ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io
import pytesseract

import cv2
import numpy as np
from bs4 import BeautifulSoup

import shutil
import pytesseract

possible_paths = [
    "/usr/bin/tesseract",       # Linux Ubuntu
    "/opt/homebrew/bin/tesseract"  # Mac
]

for path in possible_paths:
    if shutil.which(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Tesseract path set to: {path}")
        break
else:
    raise EnvironmentError("Tesseract not found! Please install it and ensure it's in PATH.")
import requests
import torch
import re
from typing import Optional, Dict
from typing import List

_blip_model = None
_blip_processor = None

def get_blip():
    global _blip_model, _blip_processor

    if _blip_model is None or _blip_processor is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32
        )
        _blip_model.eval()

    return _blip_processor, _blip_model

# Lazy-loaded ViT classification model
_vit_extractor = None
_vit_model = None

def get_vit():
    global _vit_extractor, _vit_model

    if _vit_extractor is None or _vit_model is None:
        from transformers import ViTFeatureExtractor, ViTForImageClassification

        print("Loading Image Classification Model (ViT)...")
        _vit_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        _vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        _vit_model.eval()
        print("Classification Model Loaded Successfully!")

    return _vit_extractor, _vit_model

app = FastAPI(title="AI Image Metadata API", version="1.0")

print("OCR Engine Ready (pytesseract)")

def detect_document_type(text: str, main_object: str) -> str:
    text_lower = text.lower()

    if any(word in text_lower for word in ["invoice", "total", "amount", "due"]):
        return "invoice"

    if any(word in text_lower for word in ["sale", "offer", "event", "starting", "every"]):
        return "advertisement"

    if any(word in text_lower for word in ["post", "announcement", "update"]):
        return "post"

    if main_object in ["poster", "flyer", "brochure"]:
        return "advertisement"

    return "unknown"

def rewrite_text(text: str) -> str:
    """
    Lightweight rewrite/cleanup for OCR text without using extra AI models.
    - Normalizes spaces and newlines
    - Capitalizes sentences
    - Removes duplicate lines
    - Removes unwanted symbols
    """
    if not text:
        return ""

    # Remove unwanted symbols except common punctuation
    import re
    cleaned = re.sub(r"[^A-Za-z0-9\s.,:;\'\"\-\(\)]", "", text)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    # Split into sentences (simple heuristic)
    sentences = [s.strip().capitalize() for s in cleaned.split(".") if s.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    return ". ".join(unique_sentences)


def extract_event_data(text: str) -> Dict[str, Optional[str]]:
    if not text:
        return {
            "event_name": None,
            "event_date": None,
            "event_time": None,
            "presenter": None,
            "location": None,
            "address": None
        }

    data = {
        "event_name": None,
        "event_date": None,
        "event_time": None,
        "presenter": None,
        "location": None,
        "address": None
    }

    # Flexible regex to capture start dates without using dateparser
    start_match = re.search(
        r"(starting|starts|from|beginning|launching|commencing|held\s+on)\s+(?:\w+\s+){0,3}"
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|october|oct|november|nov|december|dec)\s+\d{1,2}(st|nd|rd|th)?",
        text,
        re.IGNORECASE
    )
    if start_match:
        # Extract only the month and day
        month_day_match = re.search(
            r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
            r"september|sep|october|oct|november|nov|december|dec)\s+\d{1,2}(st|nd|rd|th)?",
            start_match.group(0),
            re.IGNORECASE
        )
        if month_day_match:
            data["event_date"] = month_day_match.group()

    # Enhanced time extraction without duplication
    # First, try to match standard ranges like 11:00 am-12:00 pm
    time_range_match = re.search(
        r"\b\d{1,2}(:\d{2})?\s?(am|pm)?\s*[-–to]+\s*\d{1,2}(:\d{2})?\s?(am|pm)\b",
        text,
        re.IGNORECASE
    )
    if time_range_match:
        data["event_time"] = time_range_match.group()
    else:
        # Only use single times if no range is found
        single_times = re.findall(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b", text, re.IGNORECASE)
        if single_times:
            # Use only the first occurrence to avoid duplication
            times = ["".join(t) for t in single_times]
            data["event_time"] = times[0]

    # Presenter / Speaker (flexible extraction)
    presenter_match = re.search(
        r"(?:speaker|presented by|with instructor|with|hosted by|by)\s+([A-Za-z\.\-]+\s+[A-Za-z\.\-]+(?:\s+[A-Za-z\.\-]+)*)",
        text,
        re.IGNORECASE
    )
    if presenter_match:
        data["presenter"] = presenter_match.group(1).strip()

    # Location extraction (flexible)
    # Location extraction (improved)
    location_match = re.search(
        r"\b(?:at|at the)\s+([A-Za-z0-9\s\-&']+?)(?:\.|,|;|$|\swith\s|\sjoin\s)",
        text,
        re.IGNORECASE
    )
    if location_match:
        location_raw = location_match.group(1)
        # Remove trailing non-alphanumeric characters
        location_clean = re.sub(r"[^\w\s\-&']", "", location_raw).strip()
        data["location"] = location_clean.title()
    # Address (simple heuristic)
    address_match = re.search(r"(address)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if address_match:
        data["address"] = address_match.group(2).strip()

    # Clean extracted fields to remove non-ASCII and unwanted symbols
    def clean_field(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        # Remove any non-ASCII characters and unwanted symbols, keep letters, numbers, common punctuation and spaces
        cleaned = re.sub(r"[^\x00-\x7F]", "", value)  # remove non-ASCII
        cleaned = re.sub(r"[^A-Za-z0-9\s.,:;'\-&()]", "", cleaned)  # remove other unwanted symbols
        # Normalize whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned else None

    for key in ["event_name", "presenter", "location", "address"]:
        data[key] = clean_field(data.get(key))

    return data

def rewrite_location(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return " ".join(value.split()).title()

def extract_qr_data(image: Image.Image):
    """
    Extract QR code URLs from image using OpenCV only (no zbar dependency)
    Supports single or multiple QR codes
    """
    open_cv_image = np.array(image)

    detector = cv2.QRCodeDetector()

    # Try multi QR detection first (OpenCV >= 4.7)
    try:
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(open_cv_image)
        if retval and decoded_info:
            return [d for d in decoded_info if d]
    except Exception:
        pass

    # Fallback to single QR detection
    data, points, _ = detector.detectAndDecode(open_cv_image)
    if data:
        return [data]

    return []


def get_page_title(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.title.string.strip() if soup.title else None
    except Exception:
        return None
def extract_event_name_with_ai(image: Image.Image, text: str) -> Optional[str]:
    """
    Uses BLIP to infer the event title from the IMAGE.
    Includes strong validation and fallback to avoid prompt-echo issues.
    """
    if image is None:
        return None
    processor, model = get_blip()

    prompt = "What is the title of this event?"

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=15)
    result = processor.decode(output[0], skip_special_tokens=True)

    if not result:
        return None

    cleaned = result.strip()

    # Reject prompt-echo or garbage answers
    banned_phrases = [
        "what is",
        "answer with",
        "event title",
        "shown in this poster"
    ]
    if any(p in cleaned.lower() for p in banned_phrases):
        return None

    # Reject very long or sentence-like outputs
    if len(cleaned.split()) > 6:
        return None

    # Capitalize nicely
    return cleaned.title()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # ---- QR Code Extraction ----
    qr_links = extract_qr_data(image)
    qr_results = []
    for link in qr_links:
        qr_results.append({
            "url": link,
            "title": get_page_title(link)
        })
    
    exif_data = {}
    try:
        exif = image.getexif()
        if exif:
            exif_data = {
                str(k): str(v)
                for k, v in exif.items()
                if k in [271, 272, 306]
            }
    except Exception:
        exif_data = {}

    extracted_text = ""
    try:
        extracted_text = pytesseract.image_to_string(image, lang="eng")
        extracted_text = extracted_text.strip()
    except Exception:
        extracted_text = ""

    event_data = extract_event_data(extracted_text)
    ai_event_name = extract_event_name_with_ai(image, extracted_text)
    if ai_event_name:
        event_data["event_name"] = ai_event_name
    else:
        # Robust fallback: ignore lines with symbols, prefer event keywords
        lines = [l.strip() for l in extracted_text.splitlines() if len(l.strip()) > 3]

        # Filter out lines with excessive symbols
        symbol_filtered = [l for l in lines if not re.search(r"[>@{}\[\]\(\)\-]", l)]

        # Prefer lines with keywords
        keywords = ["workshop", "program", "event", "seminar", "conference", "class"]
        keyword_lines = [l for l in symbol_filtered if any(k in l.lower() for k in keywords)]

        if keyword_lines:
            event_data["event_name"] = keyword_lines[0].title()
        elif symbol_filtered:
            event_data["event_name"] = symbol_filtered[0].title()
        else:
            event_data["event_name"] = lines[0].title()

    event_data["location"] = rewrite_location(event_data.get("location"))

    processor, model = get_blip()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    ai_description = processor.decode(out[0], skip_special_tokens=True)

    with torch.no_grad():
        extractor, vit_model = get_vit()
        cls_inputs = extractor(images=image, return_tensors="pt")
        outputs = vit_model(**cls_inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        main_object = vit_model.config.id2label[predicted_class.item()]

    top_objects = [{"label": main_object, "confidence": round(confidence.item(), 3)}]

    document_type = detect_document_type(
        extracted_text,
        top_objects[0]["label"] if top_objects else ""
    )

    if top_objects:
        enriched_description = f"A photo of {top_objects[0]['label']}. {ai_description}"
    else:
        enriched_description = ai_description

    return {
        "filename": file.filename,
        "basic_metadata": {
            "top_objects": top_objects
        },
        "descriptive_metadata": {
            "ai_generated_caption": enriched_description
        },
        "technical_metadata": exif_data,
        "document_metadata": {
            "extracted_text": rewrite_text(extracted_text),
            "document_type": document_type,
            "has_text": bool(extracted_text)
        },
        "event_data": event_data,
        "qr_codes": qr_results,
        "status": "Process Completed"
    }

@app.post("/analyze-image-url")
async def analyze_image_url(image_path: str):
    BASE_URL = "https://zaq23store.s3.amazonaws.com/"
    image_url = BASE_URL + image_path.lstrip("/")

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return {
            "status": "error",
            "message": "Unable to download or read image from URL"
        }

    # ---- QR Code Extraction ----
    qr_links = extract_qr_data(image)
    qr_results = []
    for link in qr_links:
        qr_results.append({
            "url": link,
            "title": get_page_title(link)
        })

    exif_data = {}
    try:
        exif = image.getexif()
        if exif:
            exif_data = {
                str(k): str(v)
                for k, v in exif.items()
                if k in [271, 272, 306]
            }
    except Exception:
        exif_data = {}

    extracted_text = ""
    try:
        extracted_text = pytesseract.image_to_string(image, lang="eng").strip()
    except Exception:
        extracted_text = ""

    event_data = extract_event_data(extracted_text)
    ai_event_name = extract_event_name_with_ai(image, extracted_text)
    if ai_event_name:
        event_data["event_name"] = ai_event_name
    else:
        # Robust fallback: ignore lines with symbols, prefer event keywords
        lines = [l.strip() for l in extracted_text.splitlines() if len(l.strip()) > 3]

        # Filter out lines with excessive symbols
        symbol_filtered = [l for l in lines if not re.search(r"[>@{}\[\]\(\)\-]", l)]

        # Prefer lines with keywords
        keywords = ["workshop", "program", "event", "seminar", "conference", "class"]
        keyword_lines = [l for l in symbol_filtered if any(k in l.lower() for k in keywords)]

        if keyword_lines:
            event_data["event_name"] = keyword_lines[0].title()
        elif symbol_filtered:
            event_data["event_name"] = symbol_filtered[0].title()
        else:
            event_data["event_name"] = lines[0].title()

    event_data["location"] = rewrite_location(event_data.get("location"))

    processor, model = get_blip()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    ai_description = processor.decode(out[0], skip_special_tokens=True)

    with torch.no_grad():
        extractor, vit_model = get_vit()
        cls_inputs = extractor(images=image, return_tensors="pt")
        outputs = vit_model(**cls_inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        main_object = vit_model.config.id2label[predicted_class.item()]

    top_objects = [{"label": main_object, "confidence": round(confidence.item(), 3)}]

    document_type = detect_document_type(
        extracted_text,
        top_objects[0]["label"] if top_objects else ""
    )

    if top_objects:
        enriched_description = f"A photo of {top_objects[0]['label']}. {ai_description}"
    else:
        enriched_description = ai_description

    return {
        "image_path": image_path,
        "basic_metadata": {
            "top_objects": top_objects
        },
        "descriptive_metadata": {
            "ai_generated_caption": enriched_description
        },
        "technical_metadata": exif_data,
        "document_metadata": {
            "extracted_text": rewrite_text(extracted_text),
            "document_type": document_type,
            "has_text": bool(extracted_text)
        },
        "event_data": event_data,
        "qr_codes": qr_results,
        "status": "Process Completed"
    }

@app.post("/analyze-images")
async def analyze_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid image file"
            })
            continue

        extracted_text = ""
        try:
            extracted_text = pytesseract.image_to_string(image, lang="eng").strip()
        except Exception:
            extracted_text = ""

        event_data = extract_event_data(extracted_text)

        results.append({
            "filename": file.filename,
            "document_metadata": {
                "extracted_text": rewrite_text(extracted_text),
                "has_text": bool(extracted_text)
            },
            "event_data": event_data,
            "status": "processed"
        })

    return {
        "count": len(results),
        "results": results
    }

@app.get("/")
def home():
    return {"message": "AI Metadata API is running. Go to /docs for testing."}