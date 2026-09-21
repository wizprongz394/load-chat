-- V3 vector retrieval
-- Keeps V1/V2 retrieval untouched and gives V3 its own RPC.

create or replace function public.match_documents_gemini_v3(
  query_embedding vector(3072),
  match_threshold float,
  match_count int
)
returns table (
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents_gemini_v3 d
  where 1 - (d.embedding <=> query_embedding) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;