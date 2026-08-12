"""Application-composed LangGraph checkpoint backends."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from harness.config import PersistenceConfig


_LOGGER = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Raised when the configured checkpoint backend cannot be created."""


def create_checkpointer(config: PersistenceConfig | None = None) -> Any:
    """Create the configured saver outside agent modules and graph nodes."""

    resolved = config or PersistenceConfig()
    if resolved.backend == "memory":
        return InMemorySaver()

    if resolved.backend != "sqlite" or resolved.sqlite_path is None:
        message = "SQLite persistence requires a configured sqlite_path"
        _LOGGER.error(message)
        raise PersistenceError(message)

    message = "SQLite persistence for the asynchronous graph requires create_async_checkpointer"
    _LOGGER.error(message)
    raise PersistenceError(message)


async def create_async_checkpointer(config: PersistenceConfig) -> Any:
    """Create an async-compatible durable saver for the harness's async graph API."""

    if config.backend == "memory":
        return InMemorySaver()
    if config.backend != "sqlite" or config.sqlite_path is None:
        message = "SQLite persistence requires a configured sqlite_path"
        _LOGGER.error(message)
        raise PersistenceError(message)

    path = Path(config.sqlite_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as error:
            message = "SQLite persistence requires the langgraph-checkpoint-sqlite package"
            _LOGGER.exception(message)
            raise PersistenceError(message) from error
        connection = await aiosqlite.connect(path)
        return AsyncSqliteSaver(connection)
    except (OSError, sqlite3.Error) as error:
        _LOGGER.exception("Unable to initialize configured SQLite checkpoint")
        raise PersistenceError(f"Unable to initialize SQLite checkpoint at {path}") from error
