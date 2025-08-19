CREATE TABLE IF NOT EXISTS wildlife.species_flattened
(
    id integer,
    species_id integer,
    kingdom text COLLATE pg_catalog."default",
    phylum text COLLATE pg_catalog."default",
    class text COLLATE pg_catalog."default",
    order_name text COLLATE pg_catalog."default",
    family text COLLATE pg_catalog."default",
    genus text COLLATE pg_catalog."default",
    species text COLLATE pg_catalog."default",
    subspecies text COLLATE pg_catalog."default",
    common_name text COLLATE pg_catalog."default",
    conservation_code text COLLATE pg_catalog."default",
    conservation_status text COLLATE pg_catalog."default",
    scientific_name text COLLATE pg_catalog."default",
    extinct boolean
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS wildlife.species_flattened
    OWNER to wildlife_user;

ALTER TABLE wildlife.species_flattened
ADD CONSTRAINT species_flattened_species_id_unique UNIQUE (species_id);

ALTER TABLE wildlife.species_flattened
ADD PRIMARY KEY (species_id);