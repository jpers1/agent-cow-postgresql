"""Regression coverage for composite FK pairing in get_cow_dependencies.

The dependency getter derives cross-table edges from database catalog
metadata.  A composite foreign key must be matched as a whole declared
tuple, paired by the constraint's own column mapping, and confined to the
session schema; per-column union branches create false edges or
undefined-operator failures for mixed-type keys.
"""

from __future__ import annotations

import uuid

import pytest

from agentcow.postgres import (
    apply_cow_variables,
    deploy_cow_functions,
    enable_cow_schema,
    get_operation_dependencies,
)

SITE_KEY = uuid.UUID("11111111-1111-1111-1111-111111111111")
BETA_KEY = uuid.UUID("22222222-2222-2222-2222-222222222222")

PARENT_CHILD_SQL = """
CREATE TABLE pairing_parent (
    id uuid PRIMARY KEY,
    site_key uuid NOT NULL,
    tag text NOT NULL,
    UNIQUE (site_key, tag)
);
CREATE TABLE pairing_child (
    id uuid PRIMARY KEY,
    site_key uuid NOT NULL,
    tag_ref text,
    FOREIGN KEY (site_key, tag_ref) REFERENCES pairing_parent (site_key, tag)
);
"""

# Referenced key permutation: the unique constraint is (alpha, beta) but the
# foreign key references (beta, alpha); pairing must follow declared FK
# ordinals, not unique-index or table order.
REVERSED_SQL = """
CREATE TABLE pairing_parent2 (
    id uuid PRIMARY KEY,
    alpha text NOT NULL,
    beta uuid NOT NULL,
    UNIQUE (alpha, beta)
);
CREATE TABLE pairing_child2 (
    id uuid PRIMARY KEY,
    x uuid NOT NULL,
    y text NOT NULL,
    FOREIGN KEY (x, y) REFERENCES pairing_parent2 (beta, alpha)
);
"""

# The same constraint name on two different child tables (legal in
# PostgreSQL); metadata pairing must bind the owning constraint, not just
# the constraint name.
DUPLICATE_NAME_SQL = """
CREATE TABLE pairing_parent (
    id uuid PRIMARY KEY,
    site_key uuid NOT NULL,
    tag text NOT NULL,
    UNIQUE (site_key, tag)
);
CREATE TABLE pairing_parent2 (
    id uuid PRIMARY KEY,
    alpha text NOT NULL,
    beta uuid NOT NULL,
    UNIQUE (alpha, beta)
);
CREATE TABLE pairing_child3 (
    id uuid PRIMARY KEY,
    site_key uuid NOT NULL,
    tag_ref text NOT NULL,
    CONSTRAINT dup_pairing_fk FOREIGN KEY (site_key, tag_ref)
        REFERENCES pairing_parent (site_key, tag)
);
CREATE TABLE pairing_child4 (
    id uuid PRIMARY KEY,
    alpha_ref text NOT NULL,
    beta_ref uuid NOT NULL,
    CONSTRAINT dup_pairing_fk FOREIGN KEY (alpha_ref, beta_ref)
        REFERENCES pairing_parent2 (alpha, beta)
);
"""

CROSS_SCHEMA_SQL = """
CREATE SCHEMA pairing_foreign;
CREATE TABLE pairing_foreign.pairing_parent2 (
    id uuid PRIMARY KEY,
    alpha text NOT NULL,
    beta uuid NOT NULL,
    UNIQUE (alpha, beta)
);
CREATE TABLE pairing_parent2 (
    id uuid PRIMARY KEY,
    solo text NOT NULL
);
CREATE TABLE pairing_child5 (
    id uuid PRIMARY KEY,
    alpha_ref text NOT NULL,
    beta_ref uuid NOT NULL,
    FOREIGN KEY (alpha_ref, beta_ref)
        REFERENCES pairing_foreign.pairing_parent2 (alpha, beta)
);
"""


async def _enable(executor) -> None:
    await deploy_cow_functions(executor)
    await enable_cow_schema(executor)
    executor.commit()
    await executor.execute("BEGIN")


async def _apply(executor, session_id, operation_id, statement: str) -> None:
    await apply_cow_variables(executor, session_id, operation_id)
    await executor.execute(statement)


async def _dependencies(executor, session_id) -> list[tuple[uuid.UUID, uuid.UUID]]:
    return await get_operation_dependencies(executor, session_id)


def _uuid() -> str:
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_pending_child_referencing_canonical_parent_returns_normally(executor):
    """A composite FK to a canonical (non-pending) parent must not error."""
    await executor.execute(PARENT_CHILD_SQL)
    executor.commit()
    await _enable(executor)
    await executor.execute(
        f"INSERT INTO pairing_parent_base (id, site_key, tag) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')"
    )
    executor.commit()
    await executor.execute("BEGIN")
    session_id = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        uuid.uuid4(),
        f"INSERT INTO pairing_child (id, site_key, tag_ref) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    assert await _dependencies(executor, session_id) == []


@pytest.mark.asyncio
async def test_matching_full_tuple_yields_exactly_one_directed_edge(executor):
    await executor.execute(PARENT_CHILD_SQL)
    executor.commit()
    await _enable(executor)
    session_id = uuid.uuid4()
    parent_op = uuid.uuid4()
    child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent (id, site_key, tag) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    await _apply(
        executor,
        session_id,
        child_op,
        f"INSERT INTO pairing_child (id, site_key, tag_ref) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    assert await _dependencies(executor, session_id) == [(parent_op, child_op)]


@pytest.mark.asyncio
async def test_shared_single_tuple_component_yields_no_edge(executor):
    await executor.execute(PARENT_CHILD_SQL)
    executor.commit()
    await _enable(executor)
    session_id = uuid.uuid4()
    parent_op = uuid.uuid4()
    child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent (id, site_key, tag) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    await _apply(
        executor,
        session_id,
        child_op,
        f"INSERT INTO pairing_child (id, site_key, tag_ref) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-b')",
    )
    assert await _dependencies(executor, session_id) == []


@pytest.mark.asyncio
async def test_reversed_referenced_key_full_tuple_yields_edge(executor):
    await executor.execute(REVERSED_SQL)
    executor.commit()
    await _enable(executor)
    session_id = uuid.uuid4()
    parent_op = uuid.uuid4()
    child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent2 (id, alpha, beta) "
        f"VALUES ('{_uuid()}', 'alpha-1', '{BETA_KEY}')",
    )
    await _apply(
        executor,
        session_id,
        child_op,
        f"INSERT INTO pairing_child2 (id, x, y) "
        f"VALUES ('{_uuid()}', '{BETA_KEY}', 'alpha-1')",
    )
    assert await _dependencies(executor, session_id) == [(parent_op, child_op)]


@pytest.mark.asyncio
async def test_reversed_referenced_key_partial_match_yields_no_edge(executor):
    await executor.execute(REVERSED_SQL)
    executor.commit()
    await _enable(executor)
    session_id = uuid.uuid4()
    parent_op = uuid.uuid4()
    child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent2 (id, alpha, beta) "
        f"VALUES ('{_uuid()}', 'alpha-1', '{BETA_KEY}')",
    )
    await _apply(
        executor,
        session_id,
        child_op,
        f"INSERT INTO pairing_child2 (id, x, y) "
        f"VALUES ('{_uuid()}', '{BETA_KEY}', 'alpha-2')",
    )
    assert await _dependencies(executor, session_id) == []


@pytest.mark.asyncio
async def test_nullable_tuple_component_yields_no_edge(executor):
    await executor.execute(PARENT_CHILD_SQL)
    executor.commit()
    await _enable(executor)
    session_id = uuid.uuid4()
    parent_op = uuid.uuid4()
    child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent (id, site_key, tag) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    await _apply(
        executor,
        session_id,
        child_op,
        f"INSERT INTO pairing_child (id, site_key, tag_ref) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', NULL)",
    )
    assert await _dependencies(executor, session_id) == []


@pytest.mark.asyncio
async def test_duplicate_fk_constraint_names_do_not_cross_tables(executor):
    await executor.execute(DUPLICATE_NAME_SQL)
    executor.commit()
    await _enable(executor)
    await executor.execute(
        f"INSERT INTO pairing_parent2_base (id, alpha, beta) "
        f"VALUES ('{_uuid()}', 'alpha-0', '{BETA_KEY}')"
    )
    executor.commit()
    await executor.execute("BEGIN")
    session_id = uuid.uuid4()
    canonical_child_op = uuid.uuid4()
    parent_op = uuid.uuid4()
    matching_child_op = uuid.uuid4()
    other_child_op = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        canonical_child_op,
        f"INSERT INTO pairing_child4 (id, alpha_ref, beta_ref) "
        f"VALUES ('{_uuid()}', 'alpha-0', '{BETA_KEY}')",
    )
    await _apply(
        executor,
        session_id,
        parent_op,
        f"INSERT INTO pairing_parent2 (id, alpha, beta) "
        f"VALUES ('{_uuid()}', 'alpha-2', '{BETA_KEY}')",
    )
    await _apply(
        executor,
        session_id,
        matching_child_op,
        f"INSERT INTO pairing_child4 (id, alpha_ref, beta_ref) "
        f"VALUES ('{_uuid()}', 'alpha-2', '{BETA_KEY}')",
    )
    await _apply(
        executor,
        session_id,
        other_child_op,
        f"INSERT INTO pairing_child3 (id, site_key, tag_ref) "
        f"VALUES ('{_uuid()}', '{SITE_KEY}', 'tag-a')",
    )
    assert await _dependencies(executor, session_id) == [(parent_op, matching_child_op)]


@pytest.mark.asyncio
async def test_cross_schema_fk_with_local_decoy_stays_confined(executor):
    """A foreign-schema reference is skipped (no edge, no error), and a
    same-named local table with different columns must not be substituted."""
    await executor.execute(CROSS_SCHEMA_SQL)
    executor.commit()
    await _enable(executor)
    assert await executor.execute(
        "SELECT to_regclass('public.pairing_parent2_changes') IS NOT NULL"
    ) == [(True,)]
    session_id = uuid.uuid4()
    await _apply(
        executor,
        session_id,
        uuid.uuid4(),
        f"INSERT INTO pairing_child5 (id, alpha_ref, beta_ref) "
        f"VALUES ('{_uuid()}', 'alpha-1', '{BETA_KEY}')",
    )
    assert await _dependencies(executor, session_id) == []
