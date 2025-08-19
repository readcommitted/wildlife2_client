-- ==========================================================
-- Function: get_ecoregion_by_coords
-- Purpose:  Returns the ecoregion containing a given lat/lon
--
-- Inputs:
--   input_lat  - Latitude in decimal degrees
--   input_lon  - Longitude in decimal degrees
--
-- Outputs:
--   eco_name   - Ecoregion descriptive name
--   eco_code   - Unique ecoregion code
--   realm      - Realm classification (e.g., Nearctic)
--   biome      - Biome code (numeric classification)
--
-- Requirements:
--   - PostGIS extension enabled
--   - Table `ecoregion_polygons` with `geometry` (Polygon) column
--
-- Notes:
--   - Uses EPSG:4326 for geographic coordinates
--   - Raises exception if point is invalid or no containing region is found
--
-- Example:
--   SELECT * FROM get_ecoregion_by_coords(44.6, -110.5);
--
-- ==========================================================

CREATE OR REPLACE FUNCTION public.get_ecoregion_by_coords(
    input_lat DOUBLE PRECISION,
    input_lon DOUBLE PRECISION
)
RETURNS TABLE (
    eco_name TEXT,
    eco_code TEXT,
    realm TEXT,
    biome DOUBLE PRECISION
)
LANGUAGE plpgsql
COST 100
VOLATILE
PARALLEL UNSAFE
AS $$
DECLARE
    geom GEOMETRY;
BEGIN
    -- Create geographic point from lat/lon
    SELECT ST_SetSRID(ST_MakePoint(input_lon, input_lat), 4326) INTO geom;

    IF geom IS NULL THEN
        RAISE EXCEPTION 'Invalid coordinates: %, %', input_lat, input_lon;
    END IF;

    -- Lookup containing ecoregion
    RETURN QUERY
    SELECT
        ep.eco_name,
        ep.eco_code,
        ep.realm,
        ep.biome
    FROM ecoregion_polygons ep
    WHERE ST_Contains(ep.geometry, geom)
    LIMIT 1;

    -- Raise error if no match found
    IF NOT FOUND THEN
        RAISE EXCEPTION 'No ecoregion found for coordinates: %, %', input_lat, input_lon;
    END IF;
END;
$$;

ALTER FUNCTION public.get_ecoregion_by_coords(DOUBLE PRECISION, DOUBLE PRECISION)
    OWNER TO wildlife_user;
