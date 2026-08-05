"""SQLite work claiming for distributed LongMemEval execution."""

import sqlite3
from pathlib import Path


def initialize_queue(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "item INTEGER PRIMARY KEY, state INTEGER NOT NULL, owner TEXT)"
        )
        database.executemany(
            "INSERT OR IGNORE INTO items(item, state) VALUES (?, 0)",
            ((index,) for index in indices),
        )


def claim_item(path: Path, index: int, owner: str) -> bool:
    with sqlite3.connect(path, timeout=30, isolation_level=None) as database:
        database.execute("BEGIN IMMEDIATE")
        changed = database.execute(
            "UPDATE items SET state=1, owner=? WHERE item=? AND state=0",
            (owner, index),
        ).rowcount
        database.commit()
    return changed == 1


def finish_claim(path: Path, index: int) -> None:
    with sqlite3.connect(path, timeout=30) as database:
        database.execute("UPDATE items SET state=2 WHERE item=?", (index,))


def queue_states(path: Path) -> dict[int, int]:
    with sqlite3.connect(path, timeout=30) as database:
        return dict(
            database.execute("SELECT item, state FROM items ORDER BY item")
        )
