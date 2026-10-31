"""Integration-test fixtures: each test module gets its own throwaway Postgres schema.

Point TEST_DATABASE_URL at a pgvector-enabled Postgres (docker compose up -d db
gives you one on localhost:5433). Without a database the integration tests are
skipped, unless REQUIRE_DB=1 (set in CI) turns that into a failure.
"""

import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect, migrate
from app.main import create_app

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://docuchat:docuchat@localhost:5433/docuchat")


@pytest.fixture(scope="session")
def database_url() -> str:
    try:
        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=3):
            pass
    except psycopg.OperationalError as exc:
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(f"database required but unavailable: {exc}")
        pytest.skip(f"Postgres not reachable at {TEST_DATABASE_URL}")
    return TEST_DATABASE_URL


@pytest.fixture
def settings(database_url: str):
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with connect(database_url, schema) as conn:
        migrate(conn, schema)
    yield Settings(database_url=database_url, db_schema=schema, chunk_size=400, chunk_overlap=60)
    with connect(database_url) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        conn.commit()


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c
