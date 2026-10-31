# Notes on PostgreSQL Indexing

These are study notes written for the DocuChat sample corpus. They summarise how the most common PostgreSQL index types work and when to use each one.

## B-tree indexes

B-tree is the default index type: `CREATE INDEX` without a `USING` clause builds a B-tree. It keeps keys in sorted order in a balanced tree, so it supports equality and range predicates (`=`, `<`, `<=`, `>`, `>=`, `BETWEEN`) and can also return rows already sorted for `ORDER BY`.

A multicolumn B-tree on `(a, b)` is most effective when the query constrains the leading column `a`. A query that filters only on `b` usually cannot use that index efficiently.

An index-only scan answers a query from the index alone, without visiting the table heap, when every referenced column is in the index and the visibility map shows the pages are all-visible. The `INCLUDE` clause adds non-key payload columns to make more queries index-only.

## Hash indexes

Hash indexes only support equality comparisons. Since PostgreSQL 10 they are WAL-logged and crash-safe, but in practice a B-tree is usually just as fast and far more flexible, so hash indexes are rarely the right choice.

## GIN indexes

GIN stands for Generalized Inverted Index. It maps each element (a word, an array element, a JSONB key) to the list of rows that contain it. GIN is the standard choice for full-text search on `tsvector` columns, for array containment with `@>`, and for JSONB queries.

GIN indexes are fast to search but relatively slow to update, because one inserted row can add many index entries. The `fastupdate` storage parameter buffers new entries in a pending list that is merged into the main index later, for example during VACUUM.

## GiST and BRIN indexes

GiST is a framework for balanced tree indexes over complex data types such as geometric shapes and ranges. It powers PostGIS spatial queries and exclusion constraints that prevent overlapping time ranges.

BRIN, the Block Range INdex, stores only a summary (such as the minimum and maximum value) for each range of physical table blocks. BRIN indexes are tiny and work well for very large, append-only tables where the column is correlated with physical order, like an event timestamp.

## Vector indexes with pgvector

The pgvector extension adds a `vector` column type and distance operators: `<->` for Euclidean distance, `<#>` for negative inner product and `<=>` for cosine distance. Cosine similarity is one minus the cosine distance.

pgvector offers two approximate nearest-neighbour index types. IVFFlat clusters vectors into lists and searches only the closest lists; it must be built after the table has data, and the `probes` setting trades recall for speed.

HNSW builds a multi-layer proximity graph. It has better speed-recall trade-offs than IVFFlat and can be created on an empty table, at the cost of slower builds and more memory. The `m` and `ef_construction` parameters control graph quality at build time, while `hnsw.ef_search` controls how many candidates are explored at query time (the default is 40).

Approximate indexes can miss true nearest neighbours. Filtering with a `WHERE` clause happens after the index scan, so a highly selective filter may return fewer rows than the `LIMIT` asks for.

## Full-text search basics

`to_tsvector('english', text)` parses a document into normalised lexemes: it lower-cases words, removes stop words and reduces words to their stems with the Snowball stemmer, so "indexes" and "indexing" both become "index".

A `tsquery` such as `plainto_tsquery('english', 'fast index')` joins terms with AND. The `@@` operator tests whether a tsvector matches a tsquery, and `ts_rank` or `ts_rank_cd` scores how well it matches; `ts_rank_cd` also rewards terms that appear close together (cover density).

Storing the tsvector in a generated column and indexing it with GIN avoids recomputing it on every query.

## Maintenance

Because of MVCC, updates and deletes leave dead tuples behind. VACUUM reclaims that space for reuse and updates the visibility map; autovacuum runs it automatically based on thresholds such as `autovacuum_vacuum_scale_factor`, which defaults to 20 percent of the table.

`ANALYZE` collects column statistics that the query planner uses to estimate row counts. Stale statistics are one of the most common causes of a bad query plan.

`REINDEX` rebuilds a bloated or corrupted index. `CREATE INDEX CONCURRENTLY` builds an index without taking a lock that blocks writes, at the cost of a slower build that cannot run inside a transaction block.

Use `EXPLAIN (ANALYZE, BUFFERS)` to see the actual plan, row counts, timing and buffer usage for a query.
