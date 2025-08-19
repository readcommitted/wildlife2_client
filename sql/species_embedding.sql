CREATE TABLE IF NOT EXISTS wildlife.species_embedding
(
    id integer,
    category text COLLATE pg_catalog."default",
    species text COLLATE pg_catalog."default",
    image_path text COLLATE pg_catalog."default",
    api_url text COLLATE pg_catalog."default",
    image_embedding vector(512),
    created_at timestamp without time zone,
    updated_at timestamp without time zone,
    status text COLLATE pg_catalog."default",
    text_embedding vector(512),
    combined_embedding vector(512),
    common_name text COLLATE pg_catalog."default",
    scientific_name_clean text COLLATE pg_catalog."default",
    common_name_clean text COLLATE pg_catalog."default",
    conservation_code text COLLATE pg_catalog."default",
    conservation_status text COLLATE pg_catalog."default",
    extinct boolean
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS wildlife.species_embedding
    OWNER to wildlife_user;

ALTER TABLE wildlife.species_embedding
    ADD PRIMARY KEY (id);

CREATE SEQUENCE wildlife.species_embedding_id_seq;
ALTER TABLE wildlife.species_embedding
ALTER COLUMN id SET DEFAULT nextval('wildlife.species_embedding_id_seq');
ALTER SEQUENCE wildlife.species_embedding_id_seq OWNED BY wildlife.species_embedding.id;
