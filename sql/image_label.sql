-- DROP TABLE wildlife.image_label
CREATE TABLE wildlife.image_label (
    image_label_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES wildlife.image_header(image_id) ON DELETE CASCADE,

    label_type TEXT,
    label_value TEXT,
    label_source TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT now()
);


