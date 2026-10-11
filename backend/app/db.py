import psycopg

from .config import DATABASE_URL, EMBEDDING_DIM

SCHEMA = f"""
create extension if not exists vector;

create table if not exists documents (
  id serial primary key,
  title text unique not null,
  source text,
  created_at timestamptz default now()
);

create table if not exists chunks (
  id serial primary key,
  doc_id int not null references documents(id) on delete cascade,
  ord int not null,
  heading text not null,
  body text not null,
  embedding vector({EMBEDDING_DIM}) not null,
  tsv tsvector generated always as (to_tsvector('english', heading || ' ' || body)) stored
);

create index if not exists chunks_embedding_hnsw on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_tsv_gin on chunks using gin (tsv);

create table if not exists traces (
  id serial primary key,
  created_at timestamptz default now(),
  question text,
  mode text,
  stages jsonb,
  total_ms real,
  n_hits int,
  answered boolean,
  top_relevance real
);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
