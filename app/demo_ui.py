import streamlit as st
from PIL import Image
from sqlalchemy import text
from db.db import SessionLocal
from tools.spaces import download_from_spaces_to_temp
import json
import traceback


st.markdown("""
<style>
img { pointer-events: none; user-select: none; }
</style>
""", unsafe_allow_html=True)

from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

def fetch_image_demo_rows():
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT image_id, jpeg_path, label_value, location_description, latitude, longitude
            FROM wildlife.image_demo
            ORDER BY image_id
        """)).mappings().all()
    return [dict(r) for r in rows]

def fetch_image_logs(image_id: int, limit: int = 20) -> list[dict]:
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT log_type, log_json
            FROM wildlife.image_log
            WHERE image_id = :image_id
            LIMIT :limit
        """), {"image_id": image_id, "limit": limit}).mappings().all()
    out = []
    for r in rows:
        entry = {"log_type": r["log_type"]}
        raw = r.get("log_json")
        if raw:
            try:
                entry["log_json"] = raw if isinstance(raw, dict) else json.loads(raw)
            except Exception:
                entry["log_json"] = {"raw": str(raw)[:2000]}
        out.append(entry)
    return out

try:
    st.write("Agentic Species Identification — Read-only Demo")
    st.write("Pick an image from the drop down to see the processing results for that image.")

    # --- Dropdown from image_demo only ---
    demo_rows = fetch_image_demo_rows()
    if not demo_rows:
        st.error("No images in `wildlife.image_demo`.")
        st.stop()

    image_data = {r["image_id"]: r for r in demo_rows}
    image_choices = [(r["image_id"], f"{r['label_value'] or 'Unlabeled'} ({r['image_id']})") for r in demo_rows]

    def _fmt(opt):
        try: return opt[1]
        except Exception: return str(opt)

    selected_pair = st.selectbox("Choose an image", image_choices, format_func=_fmt)
    if not isinstance(selected_pair, (list, tuple)) or not selected_pair:
        st.error("Invalid selection.")
        st.stop()

    image_id = selected_pair[0]
    selected = image_data.get(image_id)
    if not selected:
        st.error("Selected image not found.")
        st.stop()

    # --- Thumbnail from Spaces ---
    try:
        local_path = download_from_spaces_to_temp(selected["jpeg_path"])
        thumb = Image.open(local_path)
        st.image(thumb, caption=selected["label_value"] or "Unlabeled", width=320)
    except Exception as e:
        st.warning(f"Couldn't load image: {e}")

    # --- Minimal metadata ---
    with st.expander("Known Truth"):
        st.json({
            "image_id": selected["image_id"],
            "label_value": selected["label_value"],
            "location_description": selected["location_description"],
            "latitude": selected["latitude"],
            "longitude": selected["longitude"],
            "jpeg_path": selected["jpeg_path"],
        })

    # --- Show logs only (no model result parsing) ---
    with st.spinner("Loading logs..."):
        logs = fetch_image_logs(image_id, limit=20)

        st.markdown("### Species Identification Workflow")
        if not logs:
            st.info("No logs for this image.")
        else:
            for i, entry in enumerate(logs, start=1):
                st.markdown(f"{i}. {entry['log_type']}")
                with st.expander("JSON"):
                    st.json(entry.get("log_json", {}))

except Exception as e:
    first_line = traceback.format_exception_only(type(e), e)[-1].strip()
    st.error(first_line)
    st.stop()
