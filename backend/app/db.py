"""Connection pool + a minimal, transactional SQL migration runner."""

import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Arbitrary constant: serialises concurrent migration runs (e.g. two API replicas).
_MIGRATION_LOCK_ID = 7_301_001


def _connect_kwargs(schema: str | None) -> dict:
    kwargs: dict = {"row_factory": dict_row}
    if schema:
        if not _SCHEMA_NAME.match(schema):
            raise ValueError(f"invalid schema name: {schema!r}")
        # Unqualified table names resolve to `schema`; pgvector types stay in public.
        kwargs["options"] = f"-c search_path={schema},public"
    return kwargs


def create_pool(database_url: str, schema: str | None = None, max_size: int = 10) -> ConnectionPool:
    pool = ConnectionPool(database_url, min_size=1, max_size=max_size, kwargs=_connect_kwargs(schema), open=False)
    pool.open(wait=True, timeout=30)
    return pool


def connect(database_url: str, schema: str | None = None) -> psycopg.Connection:
    return psycopg.connect(database_url, **_connect_kwargs(schema))


def migrate(conn: psycopg.Connection, schema: str | None = None) -> list[str]:
    """Apply pending migrations/NNN_*.sql files in order. Returns applied versions."""
    applied: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        if schema:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.stem
            if version in done:
                continue
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    return applied


def main() -> None:
    from app.config import get_settings

    s = get_settings()
    with connect(s.database_url, s.db_schema) as conn:
        applied = migrate(conn, s.db_schema)
    print(f"migrations applied: {applied or 'none (up to date)'}")


if __name__ == "__main__":
    main()
