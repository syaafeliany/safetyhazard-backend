import os
import httpx
from io import BytesIO
from PIL import Image
from app.services.severity_rules import get_severity
from app.services.area_rules import check_ppe_compliance, check_special_hazards

YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "https://compute-vision-safetyhazard-production.up.railway.app")
RAG_SERVICE_URL  = os.getenv("RAG_SERVICE_URL",  "https://mattel-ehss-rag-production-12a3.up.railway.app")


# Confidence threshold default untuk YOLO. Diupdate ke 0.25 (API minimum).
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
# Ukuran slice SAHI (pixel). Lebih kecil = lebih sensitif ke objek kecil.
YOLO_SLICE_SIZE = int(os.getenv("YOLO_SLICE_SIZE", "320"))
# Max dimension untuk resize image sebelum kirim ke YOLO (mengurangi beban CPU)
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1280"))


def resize_image_if_needed(image_bytes: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> bytes:
    """
    Downscale image jika lebih besar dari max_dimension, maintain aspect ratio.
    YOLO service running di CPU, image besar + SAHI bisa timeout/OOM.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        
        # Convert RGBA to RGB if needed
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        
        # If already small enough, return as-is
        if img.width <= max_dimension and img.height <= max_dimension:
            return image_bytes
        
        # Calculate new dimensions (maintain aspect ratio)
        ratio = min(max_dimension / img.width, max_dimension / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        # Resize using high-quality filter
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert back to bytes
        output = BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        # If resize fails, return original
        return image_bytes


async def call_yolo_bytes(image_bytes: bytes, confidence: float = YOLO_CONFIDENCE) -> list:
    """Deteksi dari bytes gambar langsung (tanpa download URL).

    Selalu pakai /detect-sahi — endpoint SAHI memotong gambar jadi slice kecil
    sehingga jauh lebih akurat mendeteksi objek kecil (helmet, vest, person
    jauh) dibanding /detect standar. Dipakai baik oleh live-preview maupun
    analisa penuh supaya keduanya konsisten & akurat.
    
    Retry logic: 500 errors bisa sementara (YOLO service restart/overload).
    Image resizing: Downscale ke 1280px untuk mengurangi beban CPU YOLO service.
    """
    # Resize image untuk mengurangi beban YOLO service (running di CPU)
    image_bytes = resize_image_if_needed(image_bytes)
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
                response = await client.post(
                    f"{YOLO_SERVICE_URL}/detect-sahi",
                    files=files,
                    params={
                        "confidence": confidence,
                        "slice_size": YOLO_SLICE_SIZE,
                        "is_walking": True,
                        "lane_start": 0.2,
                        "lane_end": 0.8,
                    },
                )
                response.raise_for_status()
                raw = response.json().get("detections", [])
                return [normalise_detection(d) for d in raw]
        except httpx.HTTPStatusError as e:
            # 500 errors could be transient, retry
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(retry_delay)
                continue
            # 4xx errors or final retry, raise
            raise
        except httpx.RequestError as e:
            # Network errors, retry
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(retry_delay)
                continue
            raise
    
    # Should not reach here, but return empty if all retries fail
    return []


async def call_yolo(image_url: str, confidence: float = YOLO_CONFIDENCE) -> list:
    async with httpx.AsyncClient(timeout=60.0) as client:
        img_res = await client.get(image_url)
        img_res.raise_for_status()
        image_bytes = img_res.content
    return await call_yolo_bytes(image_bytes, confidence)


async def call_ocr(image_url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            img_res = await client.get(image_url)
            img_res.raise_for_status()
            files = {"image": ("image.jpg", img_res.content, "image/jpeg")}
            response = await client.post(f"{YOLO_SERVICE_URL}/ocr", files=files)
            response.raise_for_status()
            return response.json().get("ocr_text", "")
    except Exception:
        # OCR opsional — kalau gagal, lanjut tanpa OCR
        return ""


async def call_rag(hazards: list) -> list:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{RAG_SERVICE_URL}/rag/generate-corrective-actions",
            json={"hazards": hazards}
        )
        response.raise_for_status()
        # Nisrina confirmed actual response shape: {"actions": [{"label": ..., "action_description": ...}]}
        return response.json().get("actions", [])


ENV_HAZARD_LABELS = {"wet_floor", "blocked_walkway", "exposed_cable", "chemical_spill"}


def normalise_detection(raw: dict) -> dict:
    """
    Convert YOLO v2.0.0 response to internal format.
    
    YOLO v2.0.0 returns bbox as dict: {"x1": 105.5, "y1": 200.0, "x2": 150.2, "y2": 320.8, "width": 44.7, "height": 120.8}
    This function normalizes to array format [x1, y1, x2, y2] for internal processing.
    """
    bbox_raw = raw.get("bbox", {})
    
    # Handle both object format (v2.0.0) and array format (legacy)
    if isinstance(bbox_raw, dict):
        x1 = float(bbox_raw.get("x1", 0))
        y1 = float(bbox_raw.get("y1", 0))
        x2 = float(bbox_raw.get("x2", 0))
        y2 = float(bbox_raw.get("y2", 0))
        bbox = [x1, y1, x2, y2]
    elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
        bbox = [float(v) for v in bbox_raw[:4]]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]
    
    return {
        "label": str(raw.get("label", "unknown")).lower(),
        "confidence": float(raw.get("confidence_score") or raw.get("confidence") or 0.0),
        "confidence_score": float(raw.get("confidence_score") or raw.get("confidence") or 0.0),
        "bbox": bbox,  # always [x1, y1, x2, y2] after this
    }


def get_analysis_dimensions(image_bytes: bytes) -> tuple:
    """
    Kembalikan (width, height) gambar yang dikirim ke YOLO SETELAH resize.
    Koordinat bbox dari YOLO selalu dalam skala dimensi ini — frontend
    perlu tahu untuk menghitung scale factor ke ukuran canvas/video asli.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        w, h = img.width, img.height
        ratio = min(MAX_IMAGE_DIMENSION / w, MAX_IMAGE_DIMENSION / h)
        if ratio < 1:
            return int(w * ratio), int(h * ratio)
        return w, h
    except Exception:
        return 0, 0


async def run_full_pipeline(image_url: str, area: str = "spray_decoration") -> tuple:
    """
    Return tuple: (raw_detections, enriched_hazards)
    - raw_detections: deteksi mentah dari YOLO (untuk summary stats)
    - enriched_hazards: hazard yang sudah diproses dengan RAG + severity
    """
    # 1. YOLO detection (pakai SAHI)
    detections = await call_yolo(image_url)

    if not detections:
        return ([], [])  # Return empty tuple

    detected_labels = {d.get("label", "").lower() for d in detections}
    person_detections = [d for d in detections if d.get("label", "").lower() == "person"]
    person_count = len(person_detections)

    # Jika YOLO tidak mendeteksi "person" tapi ada item PPE (helmet, boots,
    # glasses, gloves, apron) — item PPE hanya muncul di atas orang, jadi
    # anggap ada minimal 1 pekerja. Tanpa ini, PPE violations tidak pernah
    # di-generate dan risk selalu "safe" padahal ada pelanggaran.
    PPE_ITEM_LABELS = {"safety_helmet", "safety_glasses", "safety_gloves", "safety_boots", "apron"}
    if person_count == 0 and detected_labels.intersection(PPE_ITEM_LABELS):
        person_count = 1

    # a) Hazard lingkungan — setiap deteksi LANGSUNG jadi hazard
    hazard_detections = [
        d for d in detections if d.get("label", "").lower() in ENV_HAZARD_LABELS
    ]

    # b) Hazard PPE — area-based detection menggunakan dataset baru
    # Dataset baru punya: person, trolley, phone, apron, safety_glasses, 
    # safety_gloves, safety_boots, safety_helmet (bukan "helmet"/"safety_vest" lagi)
    if person_count > 0:
        # Gunakan area_rules untuk cek PPE compliance per area.
        # Teruskan detections penuh supaya matching per-person (IoU)
        # bisa berjalan — orang yang PPE-nya tidak lengkap tetap terdeteksi
        # walau pekerja lain di frame sudah lengkap.
        missing_ppe = check_ppe_compliance(detected_labels, area, person_count, detections)
        hazard_detections.extend(missing_ppe)
    
    # c) Special hazards (phone usage, trolley/person lane violations)
    special_hazards = check_special_hazards(detections, area)
    hazard_detections.extend(special_hazards)

    if not hazard_detections:
        return (detections, [])  # Ada deteksi tapi tidak ada hazard → area aman

    
    ocr_text = ""

    # 3. RAG — kirim semua hazard sekaligus (batch)
    hazard_inputs = [
        {
            "label":            d.get("label"),
            "confidence_score": d.get("confidence_score"),
            "ocr_text":         ocr_text,
        }
        for d in hazard_detections
    ]

    try:
        rag_results = await call_rag(hazard_inputs)
    except Exception:
        # Kalau RAG gagal, tetap lanjut dengan default action
        rag_results = []

    # 4. Gabungkan dengan severity rules
    rag_map = {r["label"]: r for r in rag_results}
    hazards = []

    for detection in hazard_detections:
        label      = detection.get("label")
        confidence = detection.get("confidence_score", 1.0)
        severity   = get_severity(label, confidence)
        rag        = rag_map.get(label, {})

        hazards.append({
            "yolo_label":       label,
            "category":         label.replace("_", " ").title(),
            "confidence_score": confidence,
            "risk_level":       severity["risk_level"],
            "ocr_text":         ocr_text,
            "corrective_action": {
                "action_description": rag.get("action_description", "Refer to EHSS guidelines"),
                "priority":           severity["priority"],
                "due_date":           severity["due_date"],
            }
        })

    return (detections, hazards)  # Return tuple: (raw_detections, enriched_hazards)