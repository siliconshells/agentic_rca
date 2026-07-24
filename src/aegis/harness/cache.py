"""Prompt-cache prefix construction.

Caching is a prefix match: one changed byte anywhere invalidates everything after it. The render
order is ``tools`` -> ``system`` -> ``messages``, so a single ``cache_control`` breakpoint on the
last system block caches the tool schemas and the system prompt together.

The rules this module exists to enforce:

* **Nothing volatile in the prefix.** No timestamps, no run ids, no per-agent identifiers. Those
  go in the first user message, after the breakpoint.
* **Deterministic ordering.** Tools are sorted by name and schemas are serialised with sorted
  keys, because dict iteration order is not a contract and an unstable prefix silently costs
  full price on every turn.
* **Resources belong here.** Runbooks, the catalog, and SLOs are incident-independent, so they
  are pinned once and read at ~0.1x thereafter rather than re-fetched per turn.

``fingerprint`` exists so tests can assert the prefix is byte-identical across builds — the
failure mode otherwise is invisible, showing up only as a cache-hit rate of zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Opus 4.8 will not cache a prefix below this size — it silently declines rather than erroring,
# so a small prefix looks like a broken cache. Surfaced via `meets_minimum` and asserted in tests.
MIN_CACHEABLE_TOKENS = 4096
# Chars per token. English prose runs ~4.0 and JSON tool schemas ~3.0; a mixed prefix sits
# between. We use the *high* end on purpose: dividing chars by a larger number yields a lower
# token estimate, so `meets_minimum` errs toward "too small" and warns, rather than claiming a
# prefix will cache when it is actually under the floor and silently gets zero hits.
_CHARS_PER_TOKEN = 4.0


@dataclass
class PromptPrefix:
    """The cached head of every request an agent makes.

    ``system_blocks`` and ``tools`` are the exact objects passed to the API; the last system
    block carries the breakpoint.
    """

    system_blocks: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(block.get("text", "") for block in self.system_blocks)

    def estimated_tokens(self) -> int:
        payload = self.text + json.dumps(self.tools, sort_keys=True)
        return int(len(payload) / _CHARS_PER_TOKEN)

    def meets_minimum(self) -> bool:
        """Whether the prefix is large enough for the model to actually cache it."""
        return self.estimated_tokens() >= MIN_CACHEABLE_TOKENS

    def fingerprint(self) -> str:
        """Content hash of everything before the breakpoint."""
        h = hashlib.sha256()
        h.update(json.dumps(self.tools, sort_keys=True).encode())
        for block in self.system_blocks:
            h.update(block.get("text", "").encode())
        return h.hexdigest()


def build_prefix(
    *,
    system_prompt: str,
    resources: dict[str, str] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> PromptPrefix:
    """Assemble a cacheable prefix.

    ``resources`` maps a URI to its content — MCP resources the harness read once at run start.
    They are emitted in sorted-URI order and wrapped in explicit markers so the model can tell
    reference material from instructions.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": system_prompt.strip()}]

    if resources:
        rendered = "\n\n".join(
            f'<resource uri="{uri}">\n{resources[uri].strip()}\n</resource>'
            for uri in sorted(resources)
        )
        blocks.append(
            {
                "type": "text",
                "text": (
                    "<reference_material>\n"
                    "Standing organisational knowledge, current as of this run. Authoritative.\n\n"
                    f"{rendered}\n"
                    "</reference_material>"
                ),
            }
        )

    # Sorted by name: the API does not care, but the cache key does, and `list_tools` ordering
    # is not guaranteed stable across MCP server restarts.
    sorted_tools = sorted(tools or [], key=lambda t: str(t.get("name", "")))

    # The breakpoint goes on the last system block, which covers tools + system in one entry.
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return PromptPrefix(system_blocks=blocks, tools=sorted_tools)


def assert_stable(*prefixes: PromptPrefix) -> None:
    """Raise if the given prefixes are not byte-identical. Used in tests and at startup."""
    fingerprints = {p.fingerprint() for p in prefixes}
    if len(fingerprints) > 1:
        raise AssertionError(
            f"prompt prefix is not stable across builds ({len(fingerprints)} distinct "
            "fingerprints); every turn will pay full input price"
        )
