"""
ingest.py — Image Ingestion, Processing, and Metadata Pipeline
---------------------------------------------------------------

This module handles the ingestion and preprocessing of RAW wildlife images.
It extracts metadata, enriches geolocation information, converts images to JPEG,
detects subjects using YOLO, generates CLIP embeddings, and inserts records into the database.

Key Features:
- EXIF metadata extraction using `exiftool`
- RAW to JPEG conversion with `rawpy` and `OpenCV`
- Automatic animal detection and cropping via YOLO
- Image embedding generation with CLIP (ViT-B/32)
- Database insertion of structured image, metadata, and embedding records
- Organized image storage in a data lake hierarchy by date

Dependencies:
- rawpy, OpenCV, PIL for image handling
- exiftool (external) for metadata extraction
- YOLO for animal detection
- CLIP for image embedding generation
- SQLAlchemy for database operations

Updated to support DigitalOcean Spaces:
- Downloads NEF files temporarily from Spaces staging folder
- Processes metadata, generates JPEG, detects animals, computes embedding
- Stores processed images to `raw/` and `jpg/` folders in Spaces
- Inserts structured metadata and embeddings into the database

"""


# --- Imports ---
import subprocess
import json
import cv2
import numpy as np
import rawpy
import re
from PIL import Image
from tools.embedding_utils import generate_openclip_image_embedding, DEVICE
from tools.geo_utils import get_location_metadata
from tools.yolo_detector import YOLODetector
from db.db import SessionLocal
from db.image_model import ImageHeader, ImageExif, ImageEmbedding, ImageFeature
from db.color_palette_model import ColorPalette
from collections import Counter
from sklearn.cluster import KMeans
from config.settings import LOCAL, STAGE_DIR, RAW_DIR, JPG_DIR
from datetime import datetime
from pathlib import Path
import shutil
from enum import Enum
from typing import Iterable, Union
from typing import Optional, Tuple
import time


# --- Configuration ---
EXPOSURE_ADJUSTMENT = 2.3  # Brightness adjustment for RAW to JPEG conversion

Item = Union[Path, str]  # Path in LOCAL, key in CLOUD

class CleanupMode(str, Enum):
    KEEP = "keep"       # leave items in stage
    MOVE = "move"       # LOCAL only: move to stage_processed
    DELETE = "delete"   # delete from stage (local unlink or cloud delete)


def list_stage_items() -> list[Path]:
    """Return NEFs in stage (LOCAL only)."""
    if not STAGE_DIR.exists():
        return []
    return [p for p in STAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".nef"]


def dms_to_decimal(dms_str: str) -> float:
    """
    Convert GPS coordinates from DMS (degrees, minutes, seconds) to decimal format.
    """
    match = re.match(r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])", dms_str)
    if not match:
        raise ValueError(f"Invalid DMS string format: {dms_str}")
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal


def parse_exposure_compensation(value):
    """
    Safely parse exposure compensation from EXIF metadata.
    Handles fraction formats or raw float values.
    """
    if isinstance(value, str) and "/" in value:
        try:
            numerator, denominator = value.replace("+", "").replace("-", "-").split("/")
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            pass
    return float(value) if value else None


def extract_nef_exif_with_exiftool(file_path):
    """
    Extract EXIF metadata from a NEF file using exiftool.
    Returns parsed JSON dictionary or None on failure.
    """
    try:
        result = subprocess.run([
            "exiftool",
            "-json",
            str(file_path)
        ], capture_output=True, text=True)
        metadata = json.loads(result.stdout)[0] if result.returncode == 0 else None
        return metadata
    except Exception as e:
        print(f"❌ Error extracting EXIF with exiftool from {file_path}: {e}")
        return None


def load_palette_from_db():
    session = SessionLocal()
    palette = session.query(ColorPalette).all()
    session.close()
    return palette


def rgb_to_palette_color(r, g, b, palette):
    avg = (r + g + b) / 3
    for entry in palette:
        if ((entry.min_r is None or r >= entry.min_r) and
            (entry.max_r is None or r <= entry.max_r) and
            (entry.min_g is None or g >= entry.min_g) and
            (entry.max_g is None or g <= entry.max_g) and
            (entry.min_b is None or b >= entry.min_b) and
            (entry.max_b is None or b <= entry.max_b) and
            (entry.min_avg is None or avg >= entry.min_avg) and
            (entry.max_avg is None or avg <= entry.max_avg)):
            return entry.color_name
    return "unknown"


def _pick_best_crop(crops_with_labels):
    """Pick the crop with the largest bbox area when confidences aren’t available."""
    if not crops_with_labels:
        return None
    def area(b):
        # assume (x, y, w, h)
        if len(b) != 4:
            return 0
        x, y, w, h = b
        return abs(w * h)
    return max(crops_with_labels, key=lambda t: area(t[2]))

def _maybe_denoise_rgb(rgb: np.ndarray, enable: bool) -> np.ndarray:
    """Single-pass mild denoise; skip for tiny crops to save time."""
    if not enable:
        return rgb
    h, w = rgb.shape[:2]
    if min(h, w) < 256:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def convert_nef_to_jpeg(raw_file, jpeg_path, detector=None):
    """
    Convert NEF RAW file to JPEG, apply YOLO animal detection and cropping.
    Applies mild denoising before saving JPEG.
    Saves JPEG to the given path and returns that path.
    """
    try:
        with rawpy.imread(str(raw_file)) as raw:
            rgb_image = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                bright=EXPOSURE_ADJUSTMENT
            )
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

        # Use yolo detector if available
        if detector is None:
            from tools.yolo_detector import YOLODetector  # or from wherever it's defined
            detector = YOLODetector()

        crops_with_labels = detector.detect_and_crop(pil_image)

        if not crops_with_labels:
            print("No animals detected by YOLO. Falling back to full image.")
            crops_with_labels = [(pil_image, "unknown", (0, 0, pil_image.width, pil_image.height))]
        else:
            print(f"Detected {len(crops_with_labels)} animal(s) with YOLO.")

        first_crop, yolo_label, bbox_tuple = crops_with_labels[0]
        np_crop = cv2.cvtColor(np.array(first_crop), cv2.COLOR_RGB2BGR)

        #print(f"YOLO broad label: {yolo_label}")
        #print(f"YOLO bbox: {bbox_tuple}")

        # === MILD DENOISING ONLY ===
        #np_crop = cv2.fastNlMeansDenoisingColored(
        #    np_crop, None, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21
        #)
        #np_crop = cv2.fastNlMeansDenoisingColored(
        #    np_crop, None, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21
        #)

        # Save as JPEG
        cv2.imwrite(str(jpeg_path), np_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if jpeg_path.exists():
            return jpeg_path, yolo_label, bbox_tuple
        else:
            return None, None, None

    except Exception as e:
        print(f"❌ Error converting {raw_file} to JPEG with rawpy and OpenCV: {e}")
        return None, None, None




def classify_subject_size(
    focal_length_mm: float = None,
    focus_distance_m: float = None,
    sensor_width_mm: float = 36.0,  # Default full frame
    image_width_px: int = None,
    image_height_px: int = None,
    bbox: tuple = None,  # (x_min, y_min, x_max, y_max)
    size_thresholds: dict = None,
    orientation: str = "horizontal"
):
    """
    Returns a dict with:
    - 'size_class': small, medium, large, extra_large
    - 'physical_width_m'
    - 'relative_bbox_width'
    - 'relative_bbox_area'
    - 'method'
    """
    default_thresholds = {
        "physical_width_m": [0.3, 1.0, 2.5],
        "relative_width": [0.15, 0.4, 0.7],
        "relative_area": [0.01, 0.18, 0.45],
    }
    thresholds = size_thresholds or default_thresholds

    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    if orientation == "horizontal":
        bbox_dim = bbox_width
        img_dim = image_width_px
    else:
        bbox_dim = bbox_height
        img_dim = image_height_px

    relative_bbox_width = bbox_dim / img_dim if img_dim else None
    relative_bbox_area = (bbox_width * bbox_height) / (image_width_px * image_height_px) if image_width_px and image_height_px else None

    physical_width_m = None
    size_class = None
    method = "relative_only"

    if focus_distance_m is not None and (focus_distance_m < 1 or focus_distance_m > 2500):
        focus_distance_m = None

    # --- Physical size primary ---
    if all([focal_length_mm, focus_distance_m, bbox_dim, img_dim]):
        scene_width_m = (sensor_width_mm / focal_length_mm) * focus_distance_m
        physical_width_m = (bbox_dim / img_dim) * scene_width_m

        # Physical size bin
        if physical_width_m < thresholds["physical_width_m"][0]:
            size_class = "small"
        elif physical_width_m < thresholds["physical_width_m"][1]:
            size_class = "medium"
        elif physical_width_m < thresholds["physical_width_m"][2]:
            size_class = "large"
        else:
            size_class = "extra_large"
        method = "physical_and_relative"

        # --- Hybrid downgrade: If 'large' but relative area is small, downgrade ---
        if size_class == "large" and relative_bbox_area is not None:
            if relative_bbox_area < thresholds["relative_area"][1]:  # e.g., < 0.18
                size_class = "medium"
                method += "_adj"

    # --- Fallback: relative width ---
    elif relative_bbox_width is not None:
        if relative_bbox_width < thresholds["relative_width"][0]:
            size_class = "small"
        elif relative_bbox_width < thresholds["relative_width"][1]:
            size_class = "medium"
        elif relative_bbox_width < thresholds["relative_width"][2]:
            size_class = "large"
        else:
            size_class = "extra_large"
        method = "relative_only"
    # --- Fallback: relative area ---
    elif relative_bbox_area is not None:
        if relative_bbox_area < thresholds["relative_area"][0]:
            size_class = "small"
        elif relative_bbox_area < thresholds["relative_area"][1]:
            size_class = "medium"
        elif relative_bbox_area < thresholds["relative_area"][2]:
            size_class = "large"
        else:
            size_class = "extra_large"
        method = "relative_only_area"

    return {
        "size_class": size_class,
        "physical_width_m": physical_width_m,
        "relative_bbox_width": relative_bbox_width,
        "relative_bbox_area": relative_bbox_area,
        "method": method
    }



def insert_metadata_to_db(
    metadata,
    raw_data_lake_path,
    jpeg_data_lake_path,
    raw_file,
    jpeg_path,
    update_status,
    slim_features=None,
    yolo_label=None,
    bbox_tuple=None,
    image_name=None,
    embedder=None,
    **props
):
    """
    Insert image metadata, location details, and image embedding into the database.
    """
    session = SessionLocal()

    capture_date = metadata.get("SubSecCreateDate")
    if not capture_date:
        print(f"❌ No capture date found in EXIF for {raw_file.name}. Skipping.")
        return

    clean_capture_date = capture_date.split(".")[0].split("-")[0]
    capture_date_obj = datetime.strptime(clean_capture_date, "%Y:%m:%d %H:%M:%S")

    lat, lon = metadata.get("GPSLatitude"), metadata.get("GPSLongitude")
    if lat and lon:
        loc = get_location_metadata(dms_to_decimal(lat), dms_to_decimal(lon))
        lat_val = loc.get("latitude")
        lon_val = loc.get("longitude")
        location_desc = loc.get("full_display_name")
    else:
        lat_val, lon_val = 0.0, 0.0
        location_desc = "Unknown"
        loc = {}

    # Use local file in LOCAL mode, else use data-lake key/URL
    path_for_embedding = str(jpeg_path) if LOCAL else jpeg_data_lake_path

    from tools.embedding_utils import generate_openclip_image_embedding
    pre_embedding = generate_openclip_image_embedding(path_for_embedding)

    # Generate image embedding
    #pre_embedding = generate_openclip_image_embedding(path_for_embedding)
    update_status(f"Pre-Embedding Generated (1024-D): {path_for_embedding}")

    # Prepare header
    image_header = ImageHeader(
        image_name=image_name,
        raw_path=str(raw_data_lake_path),
        jpeg_path=str(jpeg_data_lake_path),
        stage_processed_path=str(jpeg_path),
        capture_date=capture_date_obj,
        current_batch=True,
        latitude=lat_val,
        longitude=lon_val,
        location_description=location_desc,
        country_code=loc.get("country"),
        state_code=loc.get("state"),
        county=loc.get("county"),
        park=loc.get("park"),
        place=loc.get("place"),
        metadata_updated=False,
        species_id=-1
    )

    # Size and physical calculations
    size_info = classify_subject_size(
        focal_length_mm=float(str(metadata.get("FocalLength", "0 mm")).split()[0]),
        focus_distance_m=float(str(metadata.get("FocusDistance", "0 m")).split()[0]),
        sensor_width_mm=36.0,
        image_width_px=int(metadata.get("ImageWidth")),
        image_height_px=int(metadata.get("ImageHeight")),
        bbox=bbox_tuple
    )

    # EXIF
    image_exif = ImageExif(
        photographer=metadata.get("Artist"),
        camera_model=metadata.get("Model"),
        lens_model=metadata.get("LensModel"),
        focal_length=metadata.get("FocalLength"),
        exposure_time=metadata.get("ExposureTime"),
        aperture=metadata.get("FNumber"),
        iso=metadata.get("ISO"),
        shutter_speed=metadata.get("ShutterSpeed"),
        exposure_program=metadata.get("ExposureProgram"),
        exposure_compensation=parse_exposure_compensation(metadata.get("ExposureCompensation")),
        metering_mode=metadata.get("MeteringMode"),
        light_source=metadata.get("LightSource"),
        white_balance=metadata.get("WhiteBalance"),
        flash=metadata.get("Flash"),
        color_space=metadata.get("ColorSpace"),
        subject_detection=metadata.get("SubjectDetection"),
        autofocus=metadata.get("AutoFocus"),
        serial_number=metadata.get("SerialNumber"),
        software_version=metadata.get("Software"),
        exif_json=json.dumps(metadata),

        focal_length_mm=size_info.get("focal_length_mm") or float(str(metadata.get("FocalLength", "0 mm")).split()[0]),
        focus_distance_m=size_info.get("focus_distance_m") or float(str(metadata.get("FocusDistance", "0 m")).split()[0]),
        sensor_width_mm=36.0,
        image_width_px=int(metadata.get("ImageWidth")),
        image_height_px=int(metadata.get("ImageHeight")),
        relative_bbox_area=size_info.get("relative_bbox_area"),
        relative_bbox_width=size_info.get("relative_bbox_width"),
        physical_subject_width_m=size_info.get("physical_width_m"),
        physical_size_method=size_info.get("method"),
        size_class=size_info.get("size_class")
    )

    # Feature extraction (moved to image_feature)
    image_feature = ImageFeature(
        slim_features=slim_features,
        feature_description=props.get("feature_description"),
        color=props.get("color"),
        colors=props.get("colors"),
        size=props.get("size"),
        shape=props.get("shape"),
        yolo_label=yolo_label,
        color_proportion=props.get("color_proportion")
    )

    # Embedding
    image_embedding = ImageEmbedding(
        image_embedding=pre_embedding.tolist()
    )

    # Link relationships
    image_header.image_exif = image_exif
    image_header.image_embedding = image_embedding
    image_header.image_feature = image_feature

    session.add(image_header)
    session.commit()
    session.close()


def extract_slim_features(features):
    slim = {}
    # Shape info
    slim['aspect_ratio'] = features.get('aspect_ratio', 0)
    slim['contour_area'] = features.get('contour_area', 0)

    # Flatten top 3 dominant colors (RGB and percent)
    for idx, color in enumerate(features.get('dominant_colors', [])[:3]):
        slim[f'dom_color_{idx}_r'] = color['rgb'][0]
        slim[f'dom_color_{idx}_g'] = color['rgb'][1]
        slim[f'dom_color_{idx}_b'] = color['rgb'][2]
        slim[f'dom_color_{idx}_pct'] = color['percent']
    return slim


def extract_opencv_features(image_path):

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    features = {}

    # Example contour area/aspect ratio
    h, w = image_rgb.shape[:2]
    features["contour_area"] = h * w
    features["aspect_ratio"] = w / h if h != 0 else 0

    # --- Dominant Colors (KMeans) ---
    pixels = image_rgb.reshape(-1, 3)
    if pixels.shape[0] > 50000:
        pixels = pixels[np.random.choice(pixels.shape[0], 50000, replace=False)]
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        color_counts = Counter(kmeans.labels_)
        total_pixels = len(kmeans.labels_)

        dominant_colors = []
        for idx, color in enumerate(colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            pct = color_counts[idx] / total_pixels
            dominant_colors.append({
                'rgb': (r, g, b),
                'percent': pct
            })

        features['dominant_colors'] = dominant_colors

    except Exception as e:
        print("Dominant color extraction failed:", e)
        # Fallback: one color (black) with 100%
        features['dominant_colors'] = [{'rgb': (0, 0, 0), 'percent': 1.0}]

    return features


def features_to_description_and_properties(features):
    # Size
    area = features.get('contour_area', 0)
    if area < 1_000_000:
        size = "small"
    elif area < 3_000_000:
        size = "medium"
    else:
        size = "large"

    # Shape
    aspect = features.get('aspect_ratio', 0)
    if aspect > 2.5:
        shape = "very slender"
    elif aspect > 1.6:
        shape = "long and slender"
    elif aspect > 1.1:
        shape = "typical body shape"
    else:
        shape = "stout or compact"

    palette = load_palette_from_db()

    color_desc = []
    color_summary = {}

    for idx in range(3):
        pct = features.get(f'dom_color_{idx}_pct', 0)
        if pct > 0.18:
            r = features.get(f'dom_color_{idx}_r', 0)
            g = features.get(f'dom_color_{idx}_g', 0)
            b = features.get(f'dom_color_{idx}_b', 0)
            color_word = rgb_to_palette_color(r, g, b, palette)
            if color_word != "unknown":
                color_desc.append(color_word)
                color_summary[color_word] = color_summary.get(color_word, 0.0) + pct

    # Remove duplicates for description
    color_desc = list(dict.fromkeys(color_desc))
    color_main = color_desc[0] if color_desc else "unknown"

    # Natural language
    if color_desc:
        if len(color_desc) == 1 or (len(color_desc) > 1 and color_summary[color_main] > 0.75):
            color_part = f"mostly {color_main}"
        else:
            color_part = f"{color_main} with {color_desc[1]} highlights"
    else:
        color_part = "with indistinct coloring"

    feature_description = f"A {size} animal, {color_part}, and a {shape} body."

    return {
        "feature_description": feature_description,
        "color": color_main,
        "colors": color_summary,  # ← now includes percentages
        "size": size,
        "shape": shape
    }


def apply_gamma_correction(image, gamma=1.5):
    """
    Apply gamma correction to the input image.
    A gamma > 1 brightens the midtones.
    """
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)


def _find_stage_nefs_local() -> list[Path]:
    return [p for p in STAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".nef"]


def _fetch_nef_to_temp(nef_source: Path) -> Path:
    """LOCAL: use the original file directly."""
    return Path(nef_source)


def _derive_date_path_from_metadata(metadata: dict, filename: str, update_status) -> str | None:
    capture_date = metadata.get("SubSecCreateDate") or metadata.get("CreateDate") or metadata.get("DateTimeOriginal")
    if not capture_date:
        update_status(f"❌ No capture date found for {filename}. Skipping.")
        return None
    try:
        # Handle "YYYY:MM:DD ..." or "YYYY:MM:DD"
        date_token = capture_date.split()[0]
        dt = datetime.strptime(date_token, "%Y:%m:%d")
        return dt.strftime("%Y/%m/%d")
    except ValueError:
        update_status(f"❌ Invalid capture date format for {filename}: {capture_date}. Skipping.")
        return None


def _write_outputs(local_nef: Path, local_jpeg: Path, date_path: str, filename: str, update_status) -> tuple[str, str]:
    """
    Copy NEF and JPEG into local data lake mirrors and return 'keys' (relative paths).
    NEF source stays in stage until we move it after success.
    """
    raw_output_key = f"raw/{date_path}/{filename}"
    jpg_output_key = f"jpg/{date_path}/{local_jpeg.name}"

    raw_dest_dir = RAW_DIR / date_path
    jpg_dest_dir = JPG_DIR / date_path
    raw_dest_dir.mkdir(parents=True, exist_ok=True)
    jpg_dest_dir.mkdir(parents=True, exist_ok=True)

    # Always copy NEF
    shutil.copy2(local_nef, raw_dest_dir / filename)

    # Only copy JPEG if not already there
    jpeg_dest_path = jpg_dest_dir / local_jpeg.name
    if local_jpeg.resolve() != jpeg_dest_path.resolve():
        shutil.copy2(local_jpeg, jpeg_dest_path)
    else:
        print(f"Skipping JPEG copy — already in final location: {jpeg_dest_path}")

    update_status(f"Local data lake ready: {raw_dest_dir / filename}, {jpeg_dest_path}")
    return raw_output_key, jpg_output_key


def process_raw_images(update_status):
    stage_nefs = list_stage_items()
    if not stage_nefs:
        update_status("No RAW images found in staging.")
        return

    with SessionLocal() as session:
        existing_names = {name for (name,) in session.query(ImageHeader.image_name).all()}
    stage_nefs = [p for p in stage_nefs if p.name not in existing_names]
    if not stage_nefs:
        update_status("No new RAW images to process (all already ingested).")
        return

    update_status(f"Loading models....")

    with SessionLocal() as session:
        session.query(ImageHeader).update({ImageHeader.current_batch: False})
        session.commit()

    from tools.embedding_utils import ensure_model_loaded
    model, preprocess, _ = ensure_model_loaded()
    detector = YOLODetector("yolov8n.pt")

    def embedder(img_or_path):
        return generate_openclip_image_embedding(
            img_or_path, model=model, preprocess=preprocess, device=DEVICE
        )

    processed_items, failed_items = [], []

    for idx, raw_file in enumerate(stage_nefs, 1):
        filename = raw_file.name
        update_status(f"({idx}/{len(stage_nefs)}) Processing {filename}...")

        try:

            # --- EXIF ---
            update_status(f"Extracting EXIF metadata for {filename}")
            metadata = extract_nef_exif_with_exiftool(raw_file)

            if not metadata:
                update_status(f"❌ Failed to extract metadata for {filename}")
                failed_items.append(raw_file)
                continue

            # --- Date Path ---
            date_path = _derive_date_path_from_metadata(metadata, filename, update_status)

            if not date_path:
                failed_items.append(raw_file)
                continue

            # --- Convert NEF to JPEG ---
            jpeg_filename = raw_file.stem + ".jpg"
            jpeg_path = JPG_DIR / date_path / jpeg_filename
            jpeg_path.parent.mkdir(parents=True, exist_ok=True)

            update_status(f"Converting NEF and cropping for {filename}")
            jpeg_path, yolo_label, bbox_tuple = convert_nef_to_jpeg(
                raw_file, jpeg_path, detector=detector
            )
            update_status(f"NEF converted + cropped for {filename}")

            if not jpeg_path or not jpeg_path.exists():
                update_status(f"❌ Failed to convert {filename} to JPEG.")
                failed_items.append(raw_file)
                continue

            # --- Write to data lake ---
            raw_key, jpg_key = _write_outputs(raw_file, jpeg_path, date_path, filename, update_status)
            update_status(f"Written to data lake")

            # --- Feature Extraction ---
            features = extract_opencv_features(str(jpeg_path))
            slim_features = extract_slim_features(features)
            props = features_to_description_and_properties(slim_features)
            update_status(f"Features extracted")

            # --- Embedding + DB insert ---
            update_status("Generating embedding and writing metadata to DB")
            insert_metadata_to_db(
                metadata,
                raw_key,
                jpg_key,
                raw_file,
                jpeg_path,
                update_status,
                slim_features,
                yolo_label,
                bbox_tuple,
                filename,
                embedder=embedder,
                **props,
            )

            update_status(f"Metadata + embedding written for {filename}")

            # --- Clean up ---
            raw_file.unlink()
            update_status(f"All steps complete for {filename}")

            processed_items.append(raw_file)

        except Exception as e:
            update_status(f"❌ Unhandled error for {filename}: {e}")
            failed_items.append(raw_file)

    update_status(f"Run complete → ✅ {len(processed_items)} processed, ❌ {len(failed_items)} failed.")

