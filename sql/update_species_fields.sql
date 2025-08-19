CREATE OR REPLACE FUNCTION wildlife.update_species_fields(
	)
    RETURNS void
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
BEGIN
    UPDATE species_embedding se
    SET
        common_name = split_part(se.species, ',', 1),
        scientific_name_clean = TRIM(REGEXP_REPLACE(SPLIT_PART(species, ',', 2),E'\\[.*?\\]|\\s[A-Z]{2}$','','g')),
        common_name_clean = lower(trim(split_part(se.species, ',', 1))),
        conservation_code = REGEXP_MATCH(SPLIT_PART(species, ',', 2), E'([A-Z]{2,3})$')
    WHERE common_name is null;

    UPDATE species_embedding se
    SET
        conservation_code = REPLACE(REPLACE(conservation_code,'{',''),'}','')
    WHERE conservation_code IS NOT NULL and conservation_status is null;

    UPDATE species_embedding
    SET conservation_status = CASE conservation_code
        WHEN 'LC' THEN 'Least Concern'
        WHEN 'NT' THEN 'Near Threatened'
        WHEN 'VU' THEN 'Vulnerable'
        WHEN 'EN' THEN 'Endangered'
        WHEN 'CR' THEN 'Critically Endangered'
        WHEN 'EW' THEN 'Extinct in the Wild'
        WHEN 'EX' THEN 'Extinct'
        ELSE 'Unknown'
    END
    WHERE conservation_code IS NOT NULL and conservation_status is null;
END;
$BODY$;

ALTER FUNCTION wildlife.update_species_fields()
    OWNER TO wildlife_user;