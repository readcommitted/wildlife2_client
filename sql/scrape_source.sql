

CREATE TABLE IF NOT EXISTS wildlife.scrape_source (
    id              SERIAL PRIMARY KEY,
    category        TEXT      NOT NULL,
    url             TEXT      NOT NULL UNIQUE,
    region          TEXT,
    taxonomic_group TEXT,
    active          BOOLEAN   NOT NULL DEFAULT TRUE,
    last_scraped_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION trg_scrapesource_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS scrapesource_updated_at ON scrape_source;
CREATE TRIGGER scrapesource_updated_at
  BEFORE UPDATE ON scrape_source
  FOR EACH ROW EXECUTE FUNCTION trg_scrapesources_update_timestamp();


INSERT INTO wildlife.scrape_source (category, url, region, taxonomic_group)
VALUES
  (
    'mammals',
    'https://en.wikipedia.org/wiki/List_of_mammals_of_North_America',
    'North America',
    'Mammalia'
  )
ON CONFLICT (url) DO NOTHING;

INSERT INTO wildlife.scrape_source (category, url, region, taxonomic_group)
VALUES
  (
    'birds',
    'https://en.wikipedia.org/wiki/List_of_birds_of_North_America',
    'North America',
    'Aves'
  )
ON CONFLICT (url) DO NOTHING;

INSERT INTO wildlife.scrape_source (category, url, region, taxonomic_group)
VALUES
  (
    'reptiles',
    'https://en.wikipedia.org/wiki/List_of_reptiles_of_North_America',
    'North America',
    'Reptilia'
  )
ON CONFLICT (url) DO NOTHING;
