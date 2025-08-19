-- DROP TABLE wildlife.image_header
CREATE TABLE wildlife.image_header (
    image_id SERIAL PRIMARY KEY,
    image_name TEXT NOT NULL,
    raw_path TEXT,
    jpeg_path TEXT,
    stage_processed_path TEXT,
    capture_date TIMESTAMP,

    current_batch BOOLEAN DEFAULT FALSE,

    species_id INTEGER,
    speciesnet_raw TEXT,
    species_confidence FLOAT,
    species_detection_method TEXT,

    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    osm_latitude DOUBLE PRECISION,
    osm_longitude DOUBLE PRECISION,
    location_description TEXT,
    country_code TEXT,
    state_code TEXT,
    county TEXT,
    park TEXT,
    place TEXT,

    metadata_updated BOOLEAN DEFAULT FALSE,
    behavior_notes TEXT,
    tags TEXT[],

    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);