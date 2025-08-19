create table if not exists wildlife.image_model_result (
  image_model_id bigserial primary key,
  image_id       int not null
                 references wildlife.image_header(image_id) on delete cascade,
  model          text not null,            -- 'speciesnet' | 'embed' | 'yolo' ...
  model_version  text,                     -- optional (e.g., 'sn-v1.2.0')
  species_id     int,                      -- nullable; resolved match
  common_name    text,                     -- denormalized top-1 for quick reads
  prediction_score double precision,       -- model-native (prob or normalized sim)
  result         jsonb not null,           -- full structured payload (top-k, scores…)
  created_at     timestamptz default now(),
  updated_at     timestamptz default now()
);

create unique index if not exists image_model_result_uq
  on wildlife.image_model_result (image_id, model);

create or replace function wildlife.touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists image_model_result_touch on wildlife.image_model_result;
create trigger image_model_result_touch
before update on wildlife.image_model_result
for each row execute function wildlife.touch_updated_at();
