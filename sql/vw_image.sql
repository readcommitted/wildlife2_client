create or replace view wildlife.vw_image
as
select
	ih.image_id, ih.image_name, ih.raw_path, ih.jpeg_path, ih.stage_processed_path, ih.capture_date, ih.species_id,
	ih.species_confidence, ih.species_detection_method, ih.latitude, ih.longitude, ih.osm_latitude, ih.osm_longitude,
	ih.location_description, ih.country_code, ih.state_code, ih.county, ih.park, ih.place, ih.metadata_updated, ih.behavior_notes, ih.tags,
	ih.created_at, ih.updated_at, ih.speciesnet_raw, ih.current_batch, ih.features_json, ih.slim_features, ih.feature_description,
	ih.color, ih.colors, ih.size, ih.shape, ih.yolo_label,
	il.label_value,
	ie.photographer, ie.camera_model, ie.lens_model, ie.focal_length, ie.exposure_time, ie.aperture, ie.iso,
	ie.shutter_speed, ie.exposure_program, ie.exposure_compensation, ie.metering_mode, ie.light_source, ie.white_balance,
	ie.flash, ie.color_space, ie.subject_detection, ie.autofocus, ie.serial_number, ie.software_version,
	iee.score, iee.text_embedding, iee.image_embedding
from
	wildlife.image_header ih
join
	wildlife.image_label il
	on ih.image_id = il.image_id
join
	wildlife.image_exif ie
	on ih.image_id = ie.image_id
join
	wildlife.image_embedding iee
	on ih.image_id = iee.image_id;

ALTER VIEW wildlife.vw_image
    OWNER TO wildlife_user;