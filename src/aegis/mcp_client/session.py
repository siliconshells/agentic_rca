"""MCP connection management.

One connection per run, opened and closed inside a single task. That constraint is not
stylistic: the MCP client is built on anyio task groups, and a cancel scope must be exited by the
task that entered it, so handing the session around between tasks corrupts the shutdown path.
``connect`` is therefore an async context manager the run body wraps itself in, and every agent
in that run shares the resulting client.

Three transports, all speaking the same protocol:

* ``streamable-http`` — the deployed path (docker compose).
* ``stdio`` — local development, server as a subprocess.
* ``in-memory`` — tests, no port and no process.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from aegis.logging import get_logger
from aegis.mcp_client.bridge import (
    mcp_tools_to_anthropic,
    prompt_to_messages,
    tool_result_to_text,
)

log = get_logger(__name__)


@dataclass
class ToolOutcome:
    text: str
    is_error: bool


class McpClient:
    """Thin async facade over ``ClientSession``, returning harness-shaped data."""

    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self._tool_cache: list[Any] | None = None

    async def list_tools(self, allow: set[str] | None = None) -> list[dict[str, Any]]:
        """Anthropic-shaped tool definitions, optionally scoped to ``allow``.

        Cached: the tool list is part of the prompt prefix, and re-fetching it mid-run risks a
        different ordering, which would invalidate the cache for every subsequent turn.
        """
        if self._tool_cache is None:
            self._tool_cache = list((await self.session.list_tools()).tools)
        return mcp_tools_to_anthropic(self._tool_cache, allow=allow)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Invoke a tool. Protocol errors are returned, not raised.

        A malformed call is something the agent can recover from by reading the message and
        trying again; killing the run over it would be strictly worse.
        """
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            log.warning("tool call failed", extra={"tool": name, "error": type(exc).__name__})
            return ToolOutcome(text=f"tool invocation failed: {exc}", is_error=True)
        text, is_error = tool_result_to_text(result)
        return ToolOutcome(text=text, is_error=is_error)

    async def read_resources(self, uris: list[str]) -> dict[str, str]:
        """Read resources for the cached prefix.

        A resource that cannot be read is skipped with a warning rather than failing the run:
        losing one runbook degrades the briefing, it does not invalidate the incident.
        """
        out: dict[str, str] = {}
        for uri in uris:
            try:
                result = await self.session.read_resource(uri)  # type: ignore[arg-type]
            except Exception as exc:
                log.warning("resource read failed", extra={"uri": uri, "error": str(exc)})
                continue
            contents = getattr(result, "contents", None) or []
            if contents:
                out[uri] = getattr(contents[0], "text", "") or ""
        return out

    async def list_prompts(self) -> list[dict[str, Any]]:
        """The server's workflow catalogue, for the dashboard's launcher."""
        result = await self.session.list_prompts()
        return [
            {
                "name": p.name,
                "description": p.description or "",
                "arguments": [
                    {
                        "name": a.name,
                        "description": a.description or "",
                        "required": bool(a.required),
                    }
                    for a in (p.arguments or [])
                ],
            }
            for p in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> list[dict[str, Any]]:
        """Render a server-side prompt into seed messages."""
        return prompt_to_messages(await self.session.get_prompt(name, arguments))


@asynccontextmanager
async def connect(
    url: str = "", stdio: bool = False, scenarios_path: str | None = None
) -> AsyncIterator[McpClient]:
    """Open an MCP connection for the lifetime of a run."""
    if stdio:
        env = {"AEGIS_SCENARIOS": scenarios_path} if scenarios_path else None
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_server.server"], env=env
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            log.info("mcp connected", extra={"transport": "stdio"})
            yield McpClient(session)
        return

    async with (
        streamablehttp_client(url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        log.info("mcp connected", extra={"transport": "streamable-http", "url": url})
        yield McpClient(session)


@asynccontextmanager
async def connect_in_memory(server: Any) -> AsyncIterator[McpClient]:
    """Connect straight to a server object. Tests and in-process eval runs only."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server) as session:
        yield McpClient(session)
