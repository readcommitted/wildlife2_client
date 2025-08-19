"""
load_metadata.py — Initialize and Seed Wildlife Database
---------------------------------------------------------

This Streamlit-enabled script loads seed metadata into a PostgreSQL
database for the Wildlife Vision System.

It fetches structured data from a remote API and loads it into tables
such as `species_embedding`, `species_region`, `species_color_profile`,
and `ecoregion_polygons`.

Features:
* Retrieves file manifest from a remote API
* Supports CSV (gzipped), Parquet, and FGB formats
* Dynamically infers schema/table targets
* Coerces data types based on DB metadata
* Efficiently inserts data using PostgreSQL COPY and ogr2ogr
* Streamlit button triggers the entire workflow

Used to populate the Wildlife DB from an externally hosted metadata bundle.

Dependencies:
- requests
- pandas
- numpy
- SQLAlchemy
- ogr2ogr (GDAL CLI tool)
- Streamlit

Environment Variables:
* `WILDLIFE_API`: Base URL of the metadata API (default: production endpoint)
* `SEED_API_TOKEN`: Optional Bearer token for authenticated requests

Sensitive Info:
* Uses API token securely via `Authorization` header (not printed)
* Temporary PostgreSQL connection string is built for `ogr2ogr`
     → Safe for local dev, but redact logs if exposed

File Types Supported:
* `.csv.gz` → flat tables
* `.parquet` → embedded vector tables
* `.fgb` → Fast GeoBuf for PostGIS (via ogr2ogr)

"""

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import os, io, json, gzip, tempfile, subprocess, ast
import numpy as np
import pandas as pd
import requests, streamlit as st
from sqlalchemy import text
from db.db import SessionLocal

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_BASE = os.getenv("WILDLIFE_API", "https://api.wildlife.readcommitted.com")
SEED_API_TOKEN = os.getenv("SEED_API_TOKEN")

# Tables loaded in order (parent → child)
LOAD_ORDER = [
    ("wildlife", "color_palette"),
    ("wildlife", "species_flattened"),
    ("wildlife", "regions_flattened"),
    ("public",   "ecoregion_polygons"),
    ("wildlife", "species_color_profile"),
    ("wildlife", "species_region"),
    ("wildlife", "species_ecoregion"),
    ("wildlife", "location_lookup"),
    ("wildlife", "species_embedding"),
]

# -------------------------------------------------------------------
# Remote helpers
# -------------------------------------------------------------------
def download(url: str) -> io.BytesIO:
    """
    Download a remote file into memory as BytesIO.
    """
    r = requests.get(url, stream=True, timeout=120); r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
    buf.seek(0)
    return buf

def fetch_manifest(version: str | None = None) -> dict:
    """
    Retrieve the full seed manifest from the metadata API.
    """
    params = {"version": version} if version else {}
    headers = {"Authorization": f"Bearer {SEED_API_TOKEN}"} if SEED_API_TOKEN else {}
    r = requests.get(f"{API_BASE}/seed/manifest", params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------------------
# Postgres introspection + coercion helpers
# -------------------------------------------------------------------
def get_table_columns(session, schema: str, table: str) -> list[str]:
    q = text("""
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema=:s AND table_name=:t
      ORDER BY ordinal_position
    """)
    return [r[0] for r in session.execute(q, {"s": schema, "t": table}).fetchall()]

def get_table_column_types(session, schema: str, table: str) -> dict[str, str]:
    q = text("""
      SELECT column_name, data_type, udt_name
      FROM information_schema.columns
      WHERE table_schema=:s AND table_name=:t
    """)
    out = {}
    for name, data_type, udt in session.execute(q, {"s": schema, "t": table}):
        t = (udt or "").lower() if (data_type or "").lower() == "user-defined" else (data_type or "").lower()
        out[name] = t
    return out

def coerce_to_json_text(v):
    """
    Ensures JSON strings are valid and safe for insertion.
    """
    if pd.isna(v): return None
    if isinstance(v, (dict, list, int, float, bool)): return json.dumps(v, ensure_ascii=False)
    if isinstance(v, str):
        s = v.strip()
        try:
            return json.dumps(json.loads(s), ensure_ascii=False)
        except Exception:
            try:
                return json.dumps(ast.literal_eval(s), ensure_ascii=False)
            except Exception:
                return None
    return None

def to_pgvector(val):
    """
    Converts input into Postgres vector format string.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        arr = np.asarray(val, dtype=float).ravel().tolist()
    except Exception:
        if not isinstance(val, str): return None
        s = val.strip()
        if "..." in s: return None
        try:
            arr = np.asarray(json.loads(s), dtype=float).ravel().tolist()
        except Exception:
            try:
                arr = np.asarray(ast.literal_eval(s), dtype=float).ravel().tolist()
            except Exception:
                s = s.strip("[]")
                parts = s.replace(",", " ").split()
                arr = [float(x) for x in parts] if parts else []
    return "[" + ",".join(f"{x:.7g}" for x in arr) + "]"

# -------------------------------------------------------------------
# COPY utilities
# -------------------------------------------------------------------
def df_to_copy_csv(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    return buf

def copy_dataframe(session, schema: str, table: str, df: pd.DataFrame):
    """
    COPYs a DataFrame to Postgres using inferred type coercion.
    """
    if df.empty:
        return

    table_cols = get_table_columns(session, schema, table)
    col_types  = get_table_column_types(session, schema, table)
    cols = [c for c in table_cols if c in df.columns]
    if not cols:
        raise RuntimeError(f"No overlapping columns for {schema}.{table}")

    data = df[cols].copy()

    for c in cols:
        dtype = col_types.get(c, "").lower()
        if dtype == "vector":
            data[c] = data[c].apply(to_pgvector)
        elif dtype in ("integer", "smallint", "bigint"):
            data[c] = pd.to_numeric(data[c], errors="coerce").round(0).astype("Int64")
        elif dtype in ("numeric", "decimal", "double precision", "real"):
            data[c] = pd.to_numeric(data[c], errors="coerce")
        elif dtype in ("boolean",):
            data[c] = (
                data[c]
                .astype(str).str.strip().str.lower()
                .map({"true": True, "t": True, "1": True, "yes": True, "y": True,
                      "false": False, "f": False, "0": False, "no": False, "n": False})
                .astype("boolean")
            )
        elif dtype in ("json", "jsonb"):
            data[c] = data[c].apply(coerce_to_json_text)
        else:
            if len(data) and isinstance(data[c].iloc[0], (list, tuple)):
                data[c] = data[c].apply(lambda v: None if v is None else json.dumps(list(v)))

    csv_buf = df_to_copy_csv(data)
    raw = session.get_bind().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(
                f"COPY {schema}.{table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT CSV)",
                csv_buf
            )
        raw.commit()
    finally:
        raw.close()

# -------------------------------------------------------------------
# File type routing
# -------------------------------------------------------------------
def infer_target(name: str) -> tuple[str, str] | None:
    base = name.lower()
    if base.endswith(".csv.gz"): return ("wildlife", base.replace(".csv.gz", ""))
    if base.endswith(".parquet") and "embedding" in base: return ("wildlife", "species_embedding")
    if base == "ecoregions.fgb": return ("public", "ecoregion_polygons")
    return None

def load_csv_gz(session, url: str, schema: str, table: str):
    st.write(f"→ {schema}.{table} (csv.gz)")
    with gzip.GzipFile(fileobj=download(url)) as gz:
        df = pd.read_csv(gz)
    copy_dataframe(session, schema, table, df)

def load_parquet(session, url: str, schema: str, table: str):
    st.write(f"→ {schema}.{table} (parquet)")
    df = pd.read_parquet(download(url))
    copy_dataframe(session, schema, table, df)

def load_fgb_with_ogr2ogr(session, url: str, schema: str, table: str):
    st.write(f"→ {schema}.{table} (fgb via ogr2ogr)")
    tmp = tempfile.NamedTemporaryFile(suffix=".fgb", delete=False)
    tmp.write(download(url).read()); tmp.flush(); tmp.close()

    eng_url = session.get_bind().url
    def q(v): return str(v).replace("'", "''") if v is not None else ""
    conn_str = f"PG:host='{q(eng_url.host)}' port='{q(eng_url.port or 5432)}' user='{q(eng_url.username)}' password='{q(eng_url.password)}' dbname='{q(eng_url.database)}'"
    cmd = [
        "ogr2ogr", "-f", "PostgreSQL", conn_str, tmp.name,
        "-nln", f"{schema}.{table}", "-nlt", "PROMOTE_TO_MULTI",
        "-t_srs", "EPSG:4326", "-lco", "GEOMETRY_NAME=geometry", "-overwrite"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)

# -------------------------------------------------------------------
# Streamlit Trigger
# -------------------------------------------------------------------
if st.button("Initialize database"):
    try:
        version = None
        manifest = fetch_manifest(version)
        st.write(f"Seed version: **{manifest['version']}")

        files_by_target = {}
        for f in manifest["files"]:
            name, url = f["name"], f["url"]
            if name in ("manifest.json", "checksums.sha256"):
                continue
            tgt = infer_target(name)
            if not tgt:
                st.write(f"Skipping unrecognized artifact: {name}")
                continue
            files_by_target.setdefault(tgt, []).append((name, url))

        session = SessionLocal()
        try:
            for tgt in LOAD_ORDER:
                schema, table = tgt
                for name, url in files_by_target.get(tgt, []):
                    if name.endswith(".csv.gz"):
                        load_csv_gz(session, url, schema, table)
                    elif name.endswith(".parquet"):
                        load_parquet(session, url, schema, table)
                    elif name.endswith(".fgb"):
                        load_fgb_with_ogr2ogr(session, url, schema, table)
            session.commit()
        finally:
            session.close()

        st.success("Initialization complete")

    except Exception as e:
        st.error(f"Failed: {e}")
