-- 1. RSS sources
CREATE TABLE feed (
  feed_id      INTEGER PRIMARY KEY,
  feed_url     TEXT UNIQUE NOT NULL,
  title        TEXT,
  last_checked TIMESTAMP
);

-- 2. Articles (raw & metadata)
CREATE TABLE article (
  article_id      INTEGER PRIMARY KEY,
  feed_id         INTEGER REFERENCES feed(feed_id),
  guid            TEXT UNIQUE,          -- <guid> from RSS
  url             TEXT,
  title           TEXT,
  rss_description TEXT,
  pub_date        TIMESTAMP,
  language        TEXT,
  content_hash    TEXT,                 -- SHA-1 for dedup
  full_text       TEXT,
  vector_similarity_score FLOAT,
  linucb_score    FLOAT
);

-- 3. Vector embeddings (1 row ≡ 1 article vector)
CREATE TABLE embedding (
  article_id  INTEGER PRIMARY KEY REFERENCES article(article_id),
  vec         BLOB                        -- literal FLOAT32[dim] blob
);

-- Enable vector search (SQLite-vec example, 768-D)
CREATE VIRTUAL TABLE vss_embeddings USING vec0(vec FLOAT[768]);
-- Note: To use the index, you must populate it with data from the 'embedding' table, e.g.:
-- INSERT INTO vss_embeddings(rowid, vec) SELECT article_id, vec FROM embedding;

-- 4. User interactions (one table covers clicks, opens, ratings…)
CREATE TABLE interaction (
  int_id       INTEGER PRIMARY KEY,
  article_id   INTEGER REFERENCES article(article_id),
  event_ts     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  action_duration_ms     INTEGER,          -- NULL if not a "read" event
  action       TEXT              -- 'click' | 'dwell' | 'upvote' | 'downvote' | 'save' | 'meh' | 'skip' | 'click_archived'
);
CREATE INDEX ix_interaction_article_ts ON interaction(article_id, event_ts);

-- 5. topics (may come from LDA/zero-shot classifier, or assigned manually, or detected from embeddings)
CREATE TABLE topic (
  topic_data TEXT,
  topic_type INTEGER,
  topic_id INTEGER PRIMARY KEY,
  label    TEXT UNIQUE
);
CREATE TABLE article_topic (
  article_id INTEGER REFERENCES article(article_id),
  topic_id   INTEGER REFERENCES topic(topic_id),
  PRIMARY KEY (article_id, topic_id),
  confidence FLOAT
);

CREATE TABLE user_profile (
  profile_id INTEGER PRIMARY KEY,
  interest_vector_numerator  BLOB,
  interest_vector_denominator  BLOB,
  negative_interest_vector_numerator  BLOB,
  negative_interest_vector_denominator  BLOB,
  interest_vectors_updated_at TIMESTAMP
);

-- 6. LinUCB bandit algorithm state
CREATE TABLE linucb_state (
  user_id     INTEGER PRIMARY KEY,
  A           BLOB,       -- 16×16 matrix, row-major Float32 (feature-based)
  b           BLOB,       -- 16×1 vector (feature-based)
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);