CREATE TABLE wildlife.location_lookup (
    location_lookup_id SERIAL PRIMARY KEY,
    location_description TEXT UNIQUE NOT NULL,
    place TEXT,
    park TEXT,
    state TEXT,
    county TEXT,
    country TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

-- Optional index for search optimization
CREATE INDEX idx_location_lookup_description ON wildlife.location_lookup(location_description);

TRUNCATE TABLE wildlife.location_lookup;

-- Yellowstone National Park
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Hayden Valley, Yellowstone National Park, Park County, Wyoming, United States', 'Hayden Valley', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.6331, -110.4167),
('Lamar Valley, Yellowstone National Park, Park County, Wyoming, United States', 'Lamar Valley', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9020, -110.1710),
('Mammoth Hot Springs, Yellowstone National Park, Park County, Wyoming, United States', 'Mammoth Hot Springs', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9769, -110.7013),
('Old Faithful, Yellowstone National Park, Teton County, Wyoming, United States', 'Old Faithful', 'Yellowstone National Park', 'WY', 'Teton County', 'USA', 44.4605, -110.8281);

-- Yellowstone National Park: Fishing Bridge to East Entrance

INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Fishing Bridge, Yellowstone National Park, Park County, Wyoming, United States', 'Fishing Bridge', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5648, -110.3731),
('Pelican Creek Nature Trail, Yellowstone National Park, Park County, Wyoming, United States', 'Pelican Creek Nature Trail', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5678, -110.3585),
('Lake Butte Overlook, Yellowstone National Park, Park County, Wyoming, United States', 'Lake Butte Overlook', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5536, -110.3041),
('Sylvan Lake, Yellowstone National Park, Park County, Wyoming, United States', 'Sylvan Lake', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5252, -110.1495),
('Sylvan Pass, Yellowstone National Park, Park County, Wyoming, United States', 'Sylvan Pass', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5153, -110.1048),
('East Entrance, Yellowstone National Park, Park County, Wyoming, United States', 'East Entrance', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.4887, -110.0027);

-- Lake Region
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Bridge Bay Marina, Yellowstone National Park, Park County, Wyoming, United States', 'Bridge Bay Marina', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5381, -110.3837),
('Lake Village, Yellowstone National Park, Park County, Wyoming, United States', 'Lake Village', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5533, -110.3950),
('Storm Point Trailhead, Yellowstone National Park, Park County, Wyoming, United States', 'Storm Point Trailhead', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.5508, -110.3413);

-- Canyon & Hayden Valley
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Grand Canyon of the Yellowstone, Yellowstone National Park, Park County, Wyoming, United States', 'Grand Canyon of the Yellowstone', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.7206, -110.4966),
('Artist Point, Yellowstone National Park, Park County, Wyoming, United States', 'Artist Point', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.7155, -110.4794),
('Grizzly Overlook, Yellowstone National Park, Park County, Wyoming, United States', 'Grizzly Overlook', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.6620, -110.4096);

-- Lamar Valley Region
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Slough Creek Road, Yellowstone National Park, Park County, Wyoming, United States', 'Slough Creek Road', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.8755, -110.2472),
('Soda Butte, Yellowstone National Park, Park County, Wyoming, United States', 'Soda Butte', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9090, -110.1107),
('Northeast Entrance, Yellowstone National Park, Park County, Wyoming, United States', 'Northeast Entrance', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9124, -109.9824);

-- Mammoth Region
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Undine Falls, Yellowstone National Park, Park County, Wyoming, United States', 'Undine Falls', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9771, -110.5785),
('Blacktail Plateau Drive, Yellowstone National Park, Park County, Wyoming, United States', 'Blacktail Plateau Drive', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.9151, -110.5870),
('North Entrance, Yellowstone National Park, Park County, Wyoming, United States', 'North Entrance', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 45.0266, -110.7057);

-- Norris & Geyser Basin
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Norris Geyser Basin, Yellowstone National Park, Park County, Wyoming, United States', 'Norris Geyser Basin', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.7313, -110.7037),
('Museum of the National Park Ranger, Yellowstone National Park, Park County, Wyoming, United States', 'Museum of the National Park Ranger', 'Yellowstone National Park', 'WY', 'Park County', 'USA', 44.7365, -110.7155);

-- West Side (Old Faithful & West Yellowstone)
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Grand Prismatic Spring, Yellowstone National Park, Teton County, Wyoming, United States', 'Grand Prismatic Spring', 'Yellowstone National Park', 'WY', 'Teton County', 'USA', 44.5250, -110.8382),
('Midway Geyser Basin, Yellowstone National Park, Teton County, Wyoming, United States', 'Midway Geyser Basin', 'Yellowstone National Park', 'WY', 'Teton County', 'USA', 44.5244, -110.8380),
('West Entrance, Yellowstone National Park, Gallatin County, Montana, United States', 'West Entrance', 'Yellowstone National Park', 'Montana', 'Gallatin County', 'USA', 44.6621, -111.0986);

-- Grand Teton National Park
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Jenny Lake, Grand Teton National Park, Teton County, Wyoming, United States', 'Jenny Lake', 'Grand Teton National Park', 'WY', 'Teton County', 'USA', 43.7545, -110.7208),
('Oxbow Bend, Grand Teton National Park, Teton County, Wyoming, United States', 'Oxbow Bend', 'Grand Teton National Park', 'WY', 'Teton County', 'USA', 43.8590, -110.5202),
('Schwabacher Landing, Grand Teton National Park, Teton County, Wyoming, United States', 'Schwabacher Landing', 'Grand Teton National Park', 'WY', 'Teton County', 'USA', 43.7536, -110.6017),
('Snake River Overlook, Grand Teton National Park, Teton County, Wyoming, United States', 'Snake River Overlook', 'Grand Teton National Park', 'WY', 'Teton County', 'USA', 43.7794, -110.5373);

-- Rocky Mountain National Park
INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Bear Lake, Rocky Mountain National Park, Larimer County, Colorado, United States', 'Bear Lake', 'Rocky Mountain National Park', 'CO', 'Larimer County', 'USA', 40.3135, -105.6458),
('Horseshoe Park, Rocky Mountain National Park, Larimer County, Colorado, United States', 'Horseshoe Park', 'Rocky Mountain National Park', 'CO', 'Larimer County', 'USA', 40.3886, -105.6027),
('Trail Ridge Road, Rocky Mountain National Park, Larimer County, Colorado, United States', 'Trail Ridge Road', 'Rocky Mountain National Park', 'CO', 'Larimer County', 'USA', 40.4361, -105.7132),
('Moraine Park, Rocky Mountain National Park, Larimer County, Colorado, United States', 'Moraine Park', 'Rocky Mountain National Park', 'CO', 'Larimer County', 'USA', 40.3523, -105.5998);

-- Boulder County, Colorado Open Spaces and Parks

INSERT INTO wildlife.location_lookup (location_description, place, park, state, county, country, latitude, longitude) VALUES
('Walden Ponds Wildlife Habitat, Boulder County, Colorado, United States', NULL, 'Walden Ponds Wildlife Habitat', 'CO', 'Boulder County', 'USA', 40.0376, -105.2036),
('Sawhill Ponds, Boulder County, Colorado, United States', NULL, 'Sawhill Ponds', 'CO', 'Boulder County', 'USA', 40.0355, -105.2007),
('Lagerman Reservoir, Boulder County, Colorado, United States', NULL, 'Lagerman Reservoir', 'CO', 'Boulder County', 'USA', 40.1412, -105.1944),
('Heil Valley Ranch, Boulder County, Colorado, United States', NULL, 'Heil Valley Ranch', 'CO', 'Boulder County', 'USA', 40.1360, -105.3051),
('Hall Ranch, Boulder County, Colorado, United States', NULL, 'Hall Ranch', 'CO', 'Boulder County', 'USA', 40.2004, -105.2962),
('Rabbit Mountain Open Space, Boulder County, Colorado, United States', NULL, 'Rabbit Mountain Open Space', 'CO', 'Boulder County', 'USA', 40.2448, -105.2321),
('Boulder Reservoir, Boulder County, Colorado, United States', NULL, 'Boulder Reservoir', 'CO', 'Boulder County', 'USA', 40.0704, -105.2262),
('South Boulder Creek Trail, Boulder County, Colorado, United States', NULL, 'South Boulder Creek Trail', 'CO', 'Boulder County', 'USA', 39.9583, -105.2250),
('White Rocks Trail, Boulder County, Colorado, United States', NULL, 'White Rocks Trail', 'CO', 'Boulder County', 'USA', 40.0557, -105.1540),
('Carolyn Holmberg Preserve at Rock Creek Farm, Boulder County, Colorado, United States', NULL, 'Carolyn Holmberg Preserve at Rock Creek Farm', 'CO', 'Boulder County', 'USA', 39.9614, -105.1370),
('Chautauqua Park, Boulder, Boulder County, Colorado, United States', NULL, 'Chautauqua Park', 'CO', 'Boulder County', 'USA', 39.9981, -105.2817),
('Betasso Preserve, Boulder County, Colorado, United States', NULL, 'Betasso Preserve', 'CO', 'Boulder County', 'USA', 40.0166, -105.3569);

-- Owl Nest at Legacy Church (approx. 95th Street, Boulder, CO)
INSERT INTO wildlife.location_lookup (
    location_description, place, park, state, county, country, latitude, longitude
) VALUES (
    'Owl Nest near Legacy Church, 95th Street, Boulder County, Colorado, United States',
    'Owl Nest',
    'Legacy Church Grounds',
    'CO',
    'Boulder County',
    'USA',
    40.0485,  -105.1789  -- approximate coordinates
);

-- Owl Nest on Centennial Drive, Louisville, CO
INSERT INTO wildlife.location_lookup (
    location_description, place, park, state, county, country, latitude, longitude
) VALUES (
    'Owl Nest on Centennial Drive, Louisville, Boulder County, Colorado, United States',
    'Owl Nest',
    NULL,
    'CO',
    'Boulder County',
    'USA',
    39.991676, -105.142923
);

INSERT INTO wildlife.location_lookup (
    location_description,
    place,
    park,
    state,
    county,
    country,
    latitude,
    longitude
) VALUES (
    'LeHardy Rapids, Yellowstone National Park, Wyoming, Park County, USA',
    'LeHardy Rapids',
    'Yellowstone National Park',
    'WY',
    'Park',
    'USA',
    44.6365,
    -110.4252
);


ALTER TABLE wildlife.location_lookup
ADD COLUMN eco_name text,
ADD COLUMN eco_code text,
ADD COLUMN realm text,
ADD COLUMN biome double precision;

UPDATE wildlife.location_lookup
SET
    eco_name = eco.eco_name,
    eco_code = eco.eco_code,
    realm = eco.realm,
    biome = eco.biome
FROM (
    SELECT
        ll.location_lookup_id,
        eco.eco_name,
        eco.eco_code,
        eco.realm,
        eco.biome
    FROM wildlife.location_lookup ll
    JOIN LATERAL get_ecoregion_by_coords(ll.latitude, ll.longitude) AS eco ON TRUE
    WHERE ll.latitude IS NOT NULL AND ll.longitude IS NOT NULL
) AS eco
WHERE wildlife.location_lookup.location_lookup_id = eco.location_lookup_id;

