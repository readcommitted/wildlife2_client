CREATE TABLE wildlife.image_demo (
    image_id INTEGER PRIMARY KEY,
    image_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    raw_path TEXT,
    jpeg_path TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location_description TEXT,
    behavior_notes TEXT,
    tags TEXT[],
    common_name TEXT,
	label_value TEXT,
    image_embedding vector(1024)
);