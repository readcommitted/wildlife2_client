"""
geocode_image_locations.py — Geocode & Update Image Locations
---------------------------------------------------------------

This Streamlit tool identifies images in `image_header` missing latitude/longitude
and attempts to geocode location details using OpenStreetMap's Nominatim service.

Features:
- Reverse geocodes based on `location_description`, or fallback to `place` + `park`
- Filters to only process images in the current batch (optional)
- Preview geocoding results before committing changes to the database
- Supports partial or full batch updates with flexible limit

Dependencies:
- SQLAlchemy for database interaction
- `geocode_place_name()` from tools.geo_utils
- Streamlit for interactive UI

"""

import streamlit as st
from sqlalchemy import select, update, or_, and_
from db.db import SessionLocal
from db.image_model import ImageHeader
from tools.geo_utils import geocode_place_name

# --- Page Header ---
st.write("Geocode & Update Image Locations")

# --- Sidebar Controls ---
limit = st.sidebar.slider("Max Images to Process", 10, 1000, 100)
filter_batch = st.sidebar.checkbox("Only process current batch", value=True)
commit_changes = st.sidebar.checkbox("Commit Updates to DB", value=False)

# --- Step 1: Query Images Missing lat/lon ---
with SessionLocal() as session:
    stmt = (
        select(
            ImageHeader.image_id,
            ImageHeader.location_description,
            ImageHeader.place,
            ImageHeader.park
        )
        .where(
            and_(
                ImageHeader.latitude == 0,
                or_(
                    ImageHeader.location_description != 'Unknown',
                    ImageHeader.park.isnot(None),
                    ImageHeader.place.isnot(None)
                )
            )
        )
    )
    if filter_batch:
        stmt = stmt.where(ImageHeader.current_batch == True)

    stmt = stmt.limit(limit)
    rows = session.execute(stmt).fetchall()

# --- Step 2: Display Input Table ---
st.write(f"Found {len(rows)} images missing lat/lon{' (current batch only)' if filter_batch else ''}.")
if not rows:
    st.stop()

st.subheader("📄 Records to Geocode")
df_input = [
    {
        "image_id": row.image_id,
        "location_description": row.location_description,
        "place": row.place,
        "park": row.park
    } for row in rows
]
st.dataframe(df_input)

# --- Step 3: Geocoding and Preview ---
if st.button("Run Geocode & Update"):
    updates = []
    preview_rows = []

    for image_id, location_description, place, park in rows:
        # Prefer location_description if valid
        if location_description and location_description.strip().lower() != "unknown":
            query = location_description
        elif place and park and (place.lower() != "unknown" or park.lower() != "unknown"):
            query = f"{place}, {park}"
        else:
            query = None

        if not query or query.strip().lower() == "unknown":
            continue

        geo = geocode_place_name(query)

        # Retry with park name if needed
        if not geo or geo.get("country") == "Unknown" or (
            geo.get("latitude", 0.0) == 0.0 and geo.get("longitude", 0.0) == 0.0
        ):
            query = park
            geo = geocode_place_name(query)

        lat = geo.get("latitude")
        lon = geo.get("longitude")

        if lat and lon and (lat != 0.0 or lon != 0.0):
            updates.append((image_id, lat, lon))
            preview_rows.append({
                "image_id": image_id,
                "location_description": location_description,
                "place": place,
                "park": park,
                "latitude": lat,
                "longitude": lon
            })

    st.write(f"✅ Ready to update {len(updates)} records.")
    st.dataframe(preview_rows)

    # --- Step 4: Optional Database Commit ---
    if commit_changes:
        with SessionLocal() as session:
            for image_id, lat, lon in updates:
                stmt = (
                    update(ImageHeader)
                    .where(ImageHeader.image_id == image_id)
                    .values(latitude=lat, longitude=lon)
                )
                session.execute(stmt)
            session.commit()
        st.success(f"✅ Updated {len(updates)} records in image_header.")
    else:
        st.info("Preview only — no changes saved unless 'Commit Updates to DB' is checked.")
