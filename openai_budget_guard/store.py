"""SQLite ledger of budgets and per-request usage. Safe across threads and processes."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

AGGREGATE = "*"   # budget row covering every task of a key
PERIODS = ("none", "hourly", "daily", "weekly", "monthly")

# dimension -> budgets column holding its limit (cost is checked first)
LIMITS = {
    "cost_usd": "max_cost_usd",
    "total_tokens": "max_total_tokens",
    "input_tokens": "max_input_tokens",
    "cached_tokens": "max_cached_tokens",
    "output_tokens": "max_output_tokens",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY, hint TEXT NOT NULL, label TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS budgets (
    key_id TEXT NOT NULL, task TEXT NOT NULL,
    max_cost_usd REAL, max_total_tokens INTEGER, max_input_tokens INTEGER,
    max_cached_tokens INTEGER, max_output_tokens INTEGER,
    period TEXT NOT NULL DEFAULT 'none', updated_at REAL NOT NULL,
    PRIMARY KEY (key_id, task));
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    key_id TEXT NOT NULL, task TEXT NOT NULL, model TEXT, endpoint TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    input_cost_usd REAL NOT NULL DEFAULT 0, cached_cost_usd REAL NOT NULL DEFAULT 0,
    output_cost_usd REAL NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL);  -- reserved | final | estimated
CREATE INDEX IF NOT EXISTS ix_ledger_key_ts ON ledger(key_id, ts);
"""

USAGE_COLS = ("input_tokens", "cached_tokens", "output_tokens",
              "input_cost_usd", "cached_cost_usd", "output_cost_usd", "cost_usd")


def default_db_path() -> Path:
    return Path(os.environ.get("OBG_DB") or Path.home() / ".openai_budget_guard" / "usage.db")


def window_start(period: str, now: float | None = None) -> float:
    """Start (epoch seconds, UTC) of the current budget window."""
    if period in (None, "", "none"):
        return 0.0
    t = datetime.fromtimestamp(now or time.time(), timezone.utc)
    if period == "hourly":
        t = t.replace(minute=0, second=0, microsecond=0)
    elif period == "daily":
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "weekly":
        t = t.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=t.weekday())
    elif period == "monthly":
        t = t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"unknown period {period!r}; use one of {PERIODS}")
    return t.timestamp()


class UsageStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # ---- keys & budgets -------------------------------------------------
    def register_key(self, key_id: str, hint: str, label: str | None = None) -> None:
        with self._db() as conn:
            conn.execute("INSERT INTO api_keys(key_id, hint, label, created_at) VALUES (?,?,?,?) "
                         "ON CONFLICT(key_id) DO UPDATE SET label=COALESCE(excluded.label, api_keys.label)",
                         (key_id, hint, label, time.time()))

    def set_budget(self, key_id: str, task: str, limits: dict, period: str = "none") -> None:
        if period not in PERIODS:
            raise ValueError(f"unknown period {period!r}; use one of {PERIODS}")
        cols = list(LIMITS.values())
        with self._db() as conn:
            conn.execute(
                f"INSERT INTO budgets(key_id, task, {', '.join(cols)}, period, updated_at) "
                f"VALUES (?,?,{','.join('?' * len(cols))},?,?) "
                f"ON CONFLICT(key_id, task) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in cols + ["period", "updated_at"]),
                (key_id, task, *[limits.get(c) for c in cols], period, time.time()))

    def remove_budget(self, key_id: str, task: str) -> bool:
        with self._db() as conn:
            return conn.execute("DELETE FROM budgets WHERE key_id=? AND task=?", (key_id, task)).rowcount > 0

    def budgets(self, key_id: str | None = None) -> list[dict]:
        sql = "SELECT b.*, k.hint, k.label FROM budgets b LEFT JOIN api_keys k USING(key_id)"
        args = ()
        if key_id:
            sql, args = sql + " WHERE b.key_id=?", (key_id,)
        with self._db() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY b.key_id, b.task != '*', b.task", args)]

    def keys(self) -> list[dict]:
        with self._db() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM api_keys ORDER BY created_at")]

    # ---- usage ----------------------------------------------------------
    @staticmethod
    def _usage(conn, key_id: str, task: str, since: float) -> dict:
        sql = (f"SELECT {', '.join(f'COALESCE(SUM({c}),0) AS {c}' for c in USAGE_COLS)}, COUNT(*) AS requests "
               "FROM ledger WHERE key_id=? AND ts>=?")
        args = [key_id, since]
        if task != AGGREGATE:
            sql += " AND task=?"
            args.append(task)
        u = dict(conn.execute(sql, args).fetchone())
        u["total_tokens"] = u["input_tokens"] + u["cached_tokens"] + u["output_tokens"]
        return u

    def usage(self, key_id: str, task: str = AGGREGATE, since: float = 0.0) -> dict:
        with self._db() as conn:
            return self._usage(conn, key_id, task, since)

    def reserve(self, key_id: str, task: str, model: str, endpoint: str, estimate: dict,
                preflight: bool, require_budget: bool = False) -> tuple[int | None, dict | None]:
        """Atomically check every applicable budget and, if all pass, insert a 'reserved' ledger row.

        Returns (row_id, None) on success or (None, violation) when a budget blocks the request.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            budgets = [dict(r) for r in conn.execute(
                "SELECT * FROM budgets WHERE key_id=? AND task IN (?, ?) ORDER BY task = ?",
                (key_id, task, AGGREGATE, AGGREGATE))]          # task-specific first, aggregate last
            if require_budget and not budgets:
                conn.execute("ROLLBACK")
                return None, {"dimension": None, "budget_task": task, "period": "none",
                              "used": 0, "limit": 0, "requested": 0, "exhausted": True}
            now = time.time()
            for b in budgets:
                used = self._usage(conn, key_id, b["task"], window_start(b["period"], now))
                for dim, col in LIMITS.items():
                    limit = b[col]
                    if limit is None:
                        continue
                    exhausted = used[dim] >= limit
                    if exhausted or (preflight and used[dim] + estimate[dim] > limit):
                        conn.execute("ROLLBACK")
                        return None, {"dimension": dim, "budget_task": b["task"], "period": b["period"],
                                      "used": used[dim], "limit": limit, "requested": estimate[dim],
                                      "exhausted": exhausted}
            cur = conn.execute(
                f"INSERT INTO ledger(ts, key_id, task, model, endpoint, {', '.join(USAGE_COLS)}, status) "
                f"VALUES (?,?,?,?,?,{','.join('?' * len(USAGE_COLS))},'reserved')",
                (now, key_id, task, model, endpoint, *[estimate[c] for c in USAGE_COLS]))
            conn.execute("COMMIT")
            return cur.lastrowid, None
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def finalize(self, row_id: int, values: dict, status: str = "final") -> None:
        with self._db() as conn:
            conn.execute(f"UPDATE ledger SET {', '.join(f'{c}=?' for c in USAGE_COLS)}, status=? WHERE id=?",
                         (*[values[c] for c in USAGE_COLS], status, row_id))

    def mark(self, row_id: int, status: str) -> None:
        with self._db() as conn:
            conn.execute("UPDATE ledger SET status=? WHERE id=?", (status, row_id))

    def cancel(self, row_id: int) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM ledger WHERE id=?", (row_id,))

    def reset(self, key_id: str, task: str | None = None) -> int:
        sql, args = "DELETE FROM ledger WHERE key_id=?", [key_id]
        if task and task != AGGREGATE:
            sql, args = sql + " AND task=?", args + [task]
        with self._db() as conn:
            return conn.execute(sql, args).rowcount

    def report(self, key_id: str | None = None, since: float = 0.0,
               group_by: tuple[str, ...] = ("key_id", "task", "model")) -> list[dict]:
        allowed = {"key_id", "task", "model", "endpoint", "status"}
        if not set(group_by) <= allowed:
            raise ValueError(f"group_by must be within {allowed}")
        g = ", ".join(f"l.{c}" for c in group_by)
        sql = (f"SELECT {g}, k.hint, k.label, COUNT(*) AS requests, "
               + ", ".join(f"SUM(l.{c}) AS {c}" for c in USAGE_COLS)
               + " FROM ledger l LEFT JOIN api_keys k USING(key_id) WHERE l.ts>=?")
        args: list = [since]
        if key_id:
            sql, args = sql + " AND l.key_id=?", args + [key_id]
        with self._db() as conn:
            return [dict(r) for r in conn.execute(sql + f" GROUP BY {g} ORDER BY cost_usd DESC", args)]

    def recent(self, limit: int = 20, key_id: str | None = None) -> list[dict]:
        sql, args = "SELECT l.*, k.hint FROM ledger l LEFT JOIN api_keys k USING(key_id)", []
        if key_id:
            sql, args = sql + " WHERE l.key_id=?", [key_id]
        with self._db() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY l.id DESC LIMIT ?", args + [limit])]
