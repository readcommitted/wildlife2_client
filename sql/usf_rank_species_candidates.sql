
CREATE OR REPLACE FUNCTION wildlife.usf_rank_species_candidates(
	lat double precision,
	lon double precision,
	embedding vector,
	category text DEFAULT 'unknown'::text,
	top_n integer DEFAULT 5)
    RETURNS TABLE(species text, common_name text, image_path text, distance double precision, location_boosted boolean)
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
BEGIN
    RETURN QUERY

    WITH matched_ecoregion AS (
        SELECT eco_code
        FROM public.get_ecoregion_by_coords(lat, lon)
    ),

    loc_species AS (
        SELECT DISTINCT se.common_name
        FROM wildlife.species_ecoregion se
        JOIN matched_ecoregion me ON se.eco_code = me.eco_code
    )

    SELECT
        se.species,
        se.common_name,
        se.image_path,
        se.image_embedding <-> embedding AS distance,
        (se.common_name IN (SELECT ls.common_name FROM loc_species ls)) AS location_boosted
    FROM wildlife.species_embedding se
    WHERE se.image_embedding IS NOT NULL
    ORDER BY distance ASC
    LIMIT top_n;

END;
$BODY$;

ALTER FUNCTION wildlife.usf_rank_species_candidates(double precision, double precision, vector, text, integer)
    OWNER TO wildlife_user;
