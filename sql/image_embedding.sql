-- DROP TABLE wildlife.image_embedding
CREATE TABLE wildlife.image_embedding (
    image_embedding_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES wildlife.image_header(image_id) ON DELETE CASCADE,

    image_embedding vector(1024),
    text_embedding vector(1536),
    embedding_method TEXT,
    embedding_date TIMESTAMP DEFAULT now()
);

ALTER TABLE wildlife.image_embedding
ADD COLUMN common_name TEXT;

ALTER TABLE wildlife.image_embedding
ADD COLUMN score FLOAT;