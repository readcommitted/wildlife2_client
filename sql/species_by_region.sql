CREATE OR REPLACE VIEW wildlife.species_by_region
 AS
 SELECT rf.region_id,
    rf.realm_code,
    rf.realm_name,
    rf.ecoregion_code,
    rf.ecoregion_name,
    sf.scientific_name,
    sf.common_name,
    sf.class as class_name,
    sf.order_name,
    sf.family,
    sf.genus,
    sf.species,
    sf.conservation_code,
    sf.conservation_status,
    sf.extinct
   FROM wildlife.species_ecoregion se
     JOIN wildlife.species_flattened sf ON lower(se.scientific_name) = lower(sf.scientific_name)
     JOIN wildlife.regions_flattened rf ON se.eco_code = rf.ecoregion_code
  WHERE sf.extinct IS DISTINCT FROM true;

ALTER TABLE public.species_by_region
    OWNER TO wildlife_user;

INSERT INTO wildlife.species_ecoregion (
    species, eco_code, class, common_name, scientific_name
)
VALUES (
    'lupus', 'NA0528', 'Mammalia', 'Gray Wolf', 'Canis lupus'
);

INSERT INTO wildlife.species_ecoregion (
    species, eco_code, class, common_name, scientific_name
)
VALUES (
    'lupus', 'NA0528', 'Mammalia', 'Grey Wolf', 'Canis lupus'
);
