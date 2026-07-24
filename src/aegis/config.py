"""Configuration.

Everything tunable lives here so that no module reaches for an env var directly. Loaded from
the environment (prefix ``AEGIS_``) with a ``.env`` fallback; see ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegis.types import Effort, Role, ToolPermission

REPO_ROOT = Path(__file__).resolve().parents[2]

# The model everything runs on. Effort — not model tier — is how we trade cost for quality.
DEFAULT_MODEL = "claude-opus-4-8"


class ModelPricing(BaseModel):
    """USD per million tokens.

    Cache writes bill at 1.25x base input (5-minute TTL) and reads at 0.1x. We only ever use the
    default 5-minute TTL, so those multipliers are fixed rather than configurable.
    """

    input_per_mtok: float
    output_per_mtok: float

    CACHE_WRITE_MULTIPLIER: float = 1.25
    CACHE_READ_MULTIPLIER: float = 0.1


# Sourced from the published price list. Unknown models fall back to Opus pricing so that a
# missing entry over-estimates rather than silently reporting a run as free.
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-7": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-sonnet-5": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0),
}


def pricing_for(model: str) -> ModelPricing:
    return PRICING.get(model, PRICING["claude-opus-4-8"])


class RoleConfig(BaseModel):
    """Per-agent model settings.

    ``max_tokens`` is a hard per-response ceiling the model never sees; ``task_budget_tokens`` is
    the soft, model-visible allowance it paces itself against. Both exist on purpose.
    """

    model: str = DEFAULT_MODEL
    effort: Effort
    max_tokens: int = 16_000
    max_turns: int = Field(default=20, description="Loop iterations before the agent is cut off.")
    task_budget_tokens: int | None = Field(
        default=None, description="Model-visible budget. Minimum 20000; None disables."
    )


class BudgetConfig(BaseModel):
    max_cost_usd: float = Field(default=2.0, description="Hard ceiling; the run aborts above it.")
    warn_fraction: float = Field(default=0.75, description="Emit budget.warning at this fraction.")
    max_wall_clock_s: float = 900.0


class RetryConfig(BaseModel):
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    request_timeout_s: float = 600.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- credentials -------------------------------------------------------------------
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # --- agent topology ----------------------------------------------------------------
    roles: dict[Role, RoleConfig] = Field(
        default_factory=lambda: {
            # Plans, adjudicates, and writes the RCA — the hardest reasoning in the run.
            Role.COORDINATOR: RoleConfig(
                effort=Effort.HIGH, max_tokens=32_000, max_turns=30, task_budget_tokens=80_000
            ),
            # Bounded, well-specified task over a scoped tool set: medium is the sweet spot.
            Role.INVESTIGATOR: RoleConfig(
                effort=Effort.MEDIUM, max_tokens=16_000, max_turns=15, task_budget_tokens=40_000
            ),
            # Adversarial checks are where under-thinking costs the most, so this stays high.
            Role.VERIFIER: RoleConfig(effort=Effort.HIGH, max_tokens=8_000, max_turns=8),
        }
    )
    max_parallel_investigators: int = 6

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # --- tool permission policy --------------------------------------------------------
    # Read-only tools run unattended; anything that mutates the world requires a human.
    # This map is the enforcement point — the model is never consulted about its own permissions.
    tool_policy: dict[str, ToolPermission] = Field(
        default_factory=lambda: {
            "propose_remediation": ToolPermission.ASK,
            "execute_remediation": ToolPermission.ASK,
        }
    )
    default_tool_permission: ToolPermission = ToolPermission.ALLOW
    approval_timeout_s: float = 600.0

    # --- MCP ---------------------------------------------------------------------------
    mcp_url: str = Field(
        default="http://localhost:8765/mcp",
        description="Streamable HTTP endpoint. Ignored when mcp_stdio is set.",
    )
    mcp_stdio: bool = Field(
        default=False, description="Launch the MCP server as a subprocess instead of over HTTP."
    )

    # --- scenarios ---------------------------------------------------------------------
    scenarios_path: Path = Field(
        default=REPO_ROOT / "evals" / "scenarios.jsonl",
        description="The scenario set the API and MCP server serve incidents from.",
    )

    # --- storage & telemetry -----------------------------------------------------------
    db_path: Path = REPO_ROOT / "data" / "aegis.db"
    otel_endpoint: str = Field(
        default="", description="OTLP/HTTP collector URL. Empty disables export."
    )
    otel_service_name: str = "aegis"
    log_level: str = "INFO"
    log_json: bool = False

    def role(self, role: Role) -> RoleConfig:
        return self.roles[role]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
