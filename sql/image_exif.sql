-- DROP TABLE wildlife.image_exif
CREATE TABLE wildlife.image_exif (
    image_exif_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES wildlife.image_header(image_id) ON DELETE CASCADE,

    photographer TEXT,
    camera_model TEXT,
    lens_model TEXT,
    focal_length TEXT,
    exposure_time TEXT,
    aperture TEXT,
    iso INTEGER,
    shutter_speed TEXT,
    exposure_program TEXT,
    exposure_compensation FLOAT,
    metering_mode TEXT,
    light_source TEXT,
    white_balance TEXT,
    flash TEXT,
    color_space TEXT,
    subject_detection TEXT,
    autofocus TEXT,
    serial_number TEXT,
    software_version TEXT,
    exif_json JSON
);
