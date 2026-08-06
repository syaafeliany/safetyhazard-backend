"""
area_rules.py — PPE compliance rules per Mattel facility area.

Dipakai oleh ai_pipeline.py dan inspections.py untuk menentukan
hazard apa yang perlu di-generate berdasarkan area inspeksi dan
label yang dideteksi YOLO v2.0.0.

YOLO v2.0.0 classes:
    person, trolley, phone, apron,
    safety_glasses, safety_gloves, safety_boots, safety_helmet
"""

from __future__ import annotations
from datetime import datetime, timedelta

# ── Area configuration ──────────────────────────────────────────────────────
AREA_CONFIG: dict[str, dict] = {
    "spray_decoration": {
        "display_name": "Spray/Decoration Area",
        "required_ppe": ["safety_glasses", "safety_gloves", "apron"],
        "violation_labels": {
            "safety_glasses": "no_glasses",
            "safety_gloves":  "no_gloves",
            "apron":          "no_apron",
        },
    },
    "central_staging": {
        "display_name": "Central Staging Area",
        "required_ppe": ["safety_helmet", "safety_boots"],
        "violation_labels": {
            "safety_helmet": "no_helmet",
            "safety_boots":  "no_safety_shoes",
        },
    },
    "assembly": {
        "display_name": "Assembly Area",
        "required_ppe": [],
        "violation_labels": {},
    },
    "general": {
        "display_name": "General",
        "required_ppe": [],
        "violation_labels": {},
    },
}

# Normalise incoming area strings from the frontend
_AREA_ALIAS: dict[str, str] = {
    # exact keys
    "spray_decoration": "spray_decoration",
    "central_staging":  "central_staging",
    "assembly":         "assembly",
    "general":          "general",
    # display-name variants (what the frontend may send)
    "Spray/Decoration Area":  "spray_decoration",
    "Central Staging Area":   "central_staging",
    "Assembly Area":          "assembly",
    "General":                "general",
}

# Risk levels for inferred violations
VIOLATION_RISK: dict[str, str] = {
    "no_glasses":      "high",
    "no_gloves":       "high",
    "no_apron":        "medium",
    "no_helmet":       "high",
    "no_safety_shoes": "high",
    "phone_while_walking":    "medium",
    "trolley_out_of_lane":    "high",
    "person_out_of_lane":     "medium",
}

# Lane boundaries (fraction of image width) used for Assembly Area
LANE_START: float = 0.2
LANE_END:   float = 0.8


# ── Public helpers ──────────────────────────────────────────────────────────

def get_area_config(area: str) -> dict:
    """Return config dict for the given area string (normalised)."""
    key = _AREA_ALIAS.get(area, "general")
    return AREA_CONFIG.get(key, AREA_CONFIG["general"])


def _make_violation(label: str, bbox: list | None = None, confidence: float = 0.90,
                    inferred: bool = True) -> dict:
    return {
        "label":          label,
        "yolo_label":     label,
        "confidence_score": confidence,
        "bbox":           bbox or [],
        "risk_level":     VIOLATION_RISK.get(label, "medium"),
        "inferred":       inferred,
    }


def check_ppe_compliance(
    detected_labels: set[str],
    area: str,
    person_count: int,
    detections: list[dict] | None = None,
) -> list[dict]:
    """
    Return a list of missing-PPE violation dicts per person.

    Jika `detections` disertakan (list dict dengan bbox per deteksi), tiap
    orang diperiksa satu-per-satu apakah PPE area yang dibutuhkan benar memotong
    bbox orang tersebut (IoU). Dengan begitu, orang yang PPE-nya tidak lengkap
    tetap terdeteksi walau ada pekerja lain yang benar-benar lengkap.

    Jika `detections` None (legacy caller), fallback ke pendekatan set-label:
    satu violation per class PPE yang tidak terdeteksi sama sekali.

    Parameters
    ----------
    detected_labels : set of lowercased label strings from YOLO detections
    area            : raw area string from the frontend / DB
    person_count    : number of persons detected (violations only generated
                      when at least one person is present)
    detections      : optional full detection list (label+confidence_score+bbox)
    """
    if person_count == 0:
        return []

    config = get_area_config(area)
    violations: list[dict] = []

    if detections:
        # ── Per-person matching (akurat) ──────────────────────────
        persons = [d for d in detections if str(d.get("label", "")).lower() == "person"]
        for person in persons:
            bbox = person.get("bbox")
            if not bbox:
                continue
            b = bbox
            if isinstance(b, dict):
                b = [b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)]
            if not isinstance(b, (list, tuple)) or len(b) < 4:
                continue
            px1, py1, px2, py2 = (float(v) for v in b[:4])
            y_top, y_bot = min(py1, py2), max(py1, py2)
            x_left, x_right = min(px1, px2), max(px1, px2)
            height = max(1.0, y_bot - y_top)
            top_half = [x_left, y_top, x_right, y_top + height / 2.0]

            def _iou(a, bbox2):
                if not bbox2:
                    return 0.0
                if isinstance(bbox2, dict):
                    b2 = [bbox2.get("x1", 0), bbox2.get("y1", 0),
                          bbox2.get("x2", 0), bbox2.get("y2", 0)]
                else:
                    b2 = list(bbox2)[:4]
                try:
                    b2 = [float(v) for v in b2]
                except (TypeError, ValueError):
                    return 0.0
                ax1, ay1, ax2, ay2 = min(b2[0], b2[2]), min(b2[1], b2[3]), max(b2[0], b2[2]), max(b2[1], b2[3])
                ix1, iy1 = max(a[0], ax1), max(a[1], ay1)
                ix2, iy2 = min(a[2], ax2), min(a[3], ay2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (ax2 - ax1) * (ay2 - ay1)
                union = area_a + area_b - inter
                return inter / union if union > 0 else 0.0

            def _wearing(item_label: str) -> bool:
                return any(
                    _iou(top_half, d.get("bbox")) >= 0.05
                    for d in detections
                    if str(d.get("label", "")).lower() == item_label
                )

            for ppe_class, violation_label in config["violation_labels"].items():
                if not _wearing(ppe_class):
                    v = _make_violation(violation_label)
                    v["bbox"] = [x_left, y_top, x_right, y_top + height / 2.0]
                    violations.append(v)

    # ── Fallback / set-based ──────────────────────────────────────
    # Dipakai saat detections tidak disertakan, ATAU saat YOLO tidak
    # mendeteksi label "person" (person_count dari fallback PPE-items)
    # sehingga per-person matching tidak bisa berjalan.
    if not detections or not persons_matched(detections):
        for ppe_class, violation_label in config["violation_labels"].items():
            if ppe_class.lower() not in detected_labels:
                # missing PPE — generate one violation per missing class
                violations.append(_make_violation(violation_label))

    return violations


def persons_matched(detections: list[dict]) -> bool:
    """True jika ada deteksi person dengan bbox valid di list ini."""
    for d in detections:
        if str(d.get("label", "")).lower() != "person":
            continue
        b = d.get("bbox")
        if isinstance(b, dict):
            return all(k in b for k in ("x1", "y1", "x2", "y2"))
        if isinstance(b, (list, tuple)):
            return len(b) >= 4
    return False


def check_special_hazards(detections: list[dict], area: str) -> list[dict]:
    """
    Detect special hazards that require spatial or behavioural logic:
      - phone_while_walking  (universal — semua area, bukan cuma "general")
      - trolley_out_of_lane  (Assembly area: trolley centre outside lane)
      - person_out_of_lane   (Assembly area: person centre outside lane)

    Parameters
    ----------
    detections : list of raw YOLO detection dicts
                 Each dict: {"label": str, "confidence_score": float, "bbox": [x1,y1,x2,y2]}
    area       : raw area string
    """
    key = _AREA_ALIAS.get(area, "general")
    hazards: list[dict] = []

    labels_present = {d.get("label", "").lower() for d in detections}

    # Universal — jalan di SEMUA area, bukan hanya saat area == "general"
    if "phone" in labels_present and "person" in labels_present:
        for d in detections:
            if d.get("label", "").lower() == "phone":
                hazards.append(_make_violation(
                    "phone_while_walking",
                    bbox=d.get("bbox", []),
                    confidence=float(d.get("confidence_score", 0.90)),
                    inferred=False,
                ))

    if key == "assembly":
        for d in detections:
            label = d.get("label", "").lower()
            bbox  = d.get("bbox", [])
            if label not in ("trolley", "person") or len(bbox) < 4:
                continue

            x1, _y1, x2, _y2 = bbox
            centre_x_fraction = (x1 + x2) / 2
            if centre_x_fraction > 1:
                centre_x_fraction = centre_x_fraction / 640.0

            out_of_lane = (
                centre_x_fraction < LANE_START or centre_x_fraction > LANE_END
            )
            if out_of_lane:
                violation_label = (
                    "trolley_out_of_lane" if label == "trolley" else "person_out_of_lane"
                )
                hazards.append(_make_violation(
                    violation_label,
                    bbox=bbox,
                    confidence=float(d.get("confidence_score", 0.90)),
                    inferred=False,
                ))

    return hazards