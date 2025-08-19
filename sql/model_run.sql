CREATE TABLE IF NOT EXISTS wildlife.model_run (
  model_run_id            SERIAL PRIMARY KEY,
  model_name              VARCHAR(50)  NOT NULL DEFAULT 'speciesnet',
  model_version           VARCHAR(50)  NOT NULL DEFAULT 'resnet18',
  tag                     VARCHAR(120),
  epochs                  INTEGER,
  lr                      DOUBLE PRECISION,
  batch_size              INTEGER,
  num_classes             INTEGER,
  num_train               INTEGER,
  num_val                 INTEGER,
  top1_accuracy           DOUBLE PRECISION CHECK (top1_accuracy BETWEEN 0 AND 1),
  top5_accuracy           DOUBLE PRECISION CHECK (top5_accuracy BETWEEN 0 AND 1),
  confusion_matrix        JSONB,              -- list[list[int]]
  classification_report   JSONB,              -- sklearn report dict
  model_path              TEXT,
  started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  finished_at             TIMESTAMPTZ
);

ALTER TABLE IF EXISTS wildlife.model_run
    OWNER to wildlife_user;

GRANT ALL ON TABLE wildlife.model_run TO wildlife_user;

-- indexes
CREATE INDEX IF NOT EXISTS idx_model_run_started_at
  ON wildlife.model_run (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_run_name_version
  ON wildlife.model_run (model_name, model_version);

-- 2) model_result
CREATE TABLE IF NOT EXISTS wildlife.model_result (
  model_result_id   SERIAL PRIMARY KEY,
  model_run_id      INTEGER NOT NULL REFERENCES wildlife.model_run(model_run_id) ON DELETE CASCADE,
  jpeg_path         TEXT NOT NULL,
  true_label        VARCHAR(120) NOT NULL,
  predicted_label   VARCHAR(120) NOT NULL,
  correct           BOOLEAN NOT NULL,
  top5              JSONB,                   -- e.g. [["Grey Wolf", 0.91], ...]
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


ALTER TABLE IF EXISTS wildlife.model_result
    OWNER to wildlife_user;

GRANT ALL ON TABLE wildlife.model_result TO wildlife_user;

-- indexes
CREATE INDEX IF NOT EXISTS idx_model_result_run
  ON wildlife.model_result (model_run_id);

CREATE INDEX IF NOT EXISTS idx_model_result_true_label
  ON wildlife.model_result (true_label);

CREATE INDEX IF NOT EXISTS idx_model_result_predicted_label
  ON wildlife.model_result (predicted_label);

CREATE INDEX IF NOT EXISTS idx_model_result_correct
  ON wildlife.model_result (correct);

CREATE INDEX IF NOT EXISTS idx_model_result_top5_gin
   ON wildlife.model_result USING GIN (top5);