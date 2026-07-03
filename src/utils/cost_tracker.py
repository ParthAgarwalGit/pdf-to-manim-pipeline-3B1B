"""
Cost tracker for LLM API calls across the pipeline.

This module provides a singleton CostTracker class for tracking and reporting
on API call costs from various LLM providers. It supports multiple usage models
including direct instantiation, singleton access, and context manager usage.

Example:
    Direct usage:
    >>> tracker = CostTracker()
    >>> tracker.track_call("openai", "gpt-4o", 1000, 500, "chat")
    >>> print(tracker.get_total_cost())

    Singleton pattern:
    >>> CostTracker.reset_instance()
    >>> tracker = CostTracker.get_instance()
    >>> tracker.track_call("anthropic", "claude-sonnet", 2000, 1000, "completion")

    Context manager:
    >>> with CostTracker() as tracker:
    ...     tracker.track_call("openai", "gpt-4o-mini", 500, 200, "chat")
    ...     print(tracker.get_report())
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from enum import Enum


class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Model(Enum):
    """Supported models with their pricing tiers."""
    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    TTS = "tts"

    # Anthropic models
    CLAUDE_SONNET = "claude-sonnet"


# Pricing table: (input_price_per_1m_tokens, output_price_per_1m_tokens)
# Prices as of 2026 in USD
PRICING_TABLE: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-5"): (2.50, 10.00),    # Legacy/deprecated; kept for old runs
    # GPT-5.4 family (per-task tiering). Rates per OpenAI pricing, 2026-07.
    ("openai", "gpt-5.4"): (2.50, 15.00),       # Vision/OCR + unified-script fallback
    ("openai", "gpt-5.4-mini"): (0.75, 4.50),   # Per-segment beats + visual intent
    ("openai", "gpt-5.4-nano"): (0.20, 1.25),   # Storyboard grading + fidelity review
    ("openai", "gpt-5.5"): (5.00, 30.00),
    ("openai", "tts"): (15.00, 0.00),      # TTS pricing is per 1M characters, not tokens
    ("openai", "tts-1"): (15.00, 0.00),    # Standard TTS model
    ("openai", "tts-1-hd"): (30.00, 0.00), # HD TTS model
    ("anthropic", "claude-sonnet"): (3.00, 15.00),
    ("anthropic", "claude-sonnet-4-20250514"): (3.00, 15.00),
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
    # Sonnet 5: Manim code gen + unified script. Standard rate (intro $2/$10
    # through 2026-08-31 makes real spend lower; standard avoids under-billing).
    ("anthropic", "claude-sonnet-5"): (3.00, 15.00),
    ("anthropic", "claude-opus-4-7"): (5.00, 25.00),    # Corrected: Opus 4.7/4.8 = $5/$25
    ("anthropic", "claude-opus-4-8"): (5.00, 25.00),
    ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),    # Corrected: Haiku 4.5 = $1/$5
}

# Per-provider fallback rate when an exact (provider, model) pair is missing.
# Picked conservatively (highest reasonable rate) so cost is overestimated
# rather than underreported when a new model name appears mid-run.
PROVIDER_FALLBACK_RATES: Dict[str, Tuple[float, float]] = {
    "openai":    (2.50, 10.00),
    "anthropic": (3.00, 15.00),
}


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    call_type: str
    input_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CostBreakdown:
    """Cost breakdown by provider and model."""
    by_provider: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)
    by_call_type: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class CostTracker:
    """
    Thread-safe singleton for tracking LLM API call costs.

    This class tracks API calls from multiple providers and models,
    calculates costs based on token usage, and provides various
    reporting options.
    """

    _instance: Optional["CostTracker"] = None
    _lock: threading.Lock = threading.Lock()
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize the cost tracker."""
        self._calls: list[APICall] = []
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CostTracker":
        """
        Get or create the singleton instance.

        Returns:
            CostTracker: The singleton instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        with cls._instance_lock:
            cls._instance = None

    def __enter__(self) -> "CostTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass

    @staticmethod
    def _calculate_cost(
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[float, float, float]:
        """
        Calculate input, output, and total costs.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD

        Raises:
            ValueError: If provider/model combination is not found in pricing table.
        """
        key = (provider.lower(), model.lower())

        if key in PRICING_TABLE:
            input_price, output_price = PRICING_TABLE[key]
        elif provider.lower() in PROVIDER_FALLBACK_RATES:
            # Unknown model from a known provider — use the conservative
            # provider-level fallback so cost accounting never crashes a run.
            # The pipeline used to ValueError here, which made adding a new
            # model name a silent landmine.
            import logging
            logging.getLogger(__name__).warning(
                "CostTracker: unknown model %s/%s — falling back to %s "
                "provider rate. Add an entry to PRICING_TABLE for accurate "
                "accounting.", provider, model, provider.lower(),
            )
            input_price, output_price = PROVIDER_FALLBACK_RATES[provider.lower()]
        else:
            raise ValueError(
                f"Unknown provider/model combination: {provider}/{model}. "
                f"Available models: {list(PRICING_TABLE.keys())}"
            )

        # Convert token counts to per-1M-token basis
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def track_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_type: str
    ) -> None:
        """
        Track a single API call.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            call_type: Type of call (e.g., "chat", "completion", "tts")

        Raises:
            ValueError: If provider/model combination is unknown.
            TypeError: If token counts are not integers.
        """
        # Validate inputs
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            raise TypeError("Token counts must be integers")

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        # Calculate costs
        input_cost, output_cost, total_cost = self._calculate_cost(
            provider, model, input_tokens, output_tokens
        )

        # Create and store the call record
        call = APICall(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider.lower(),
            model=model.lower(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type=call_type.lower(),
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

        with self._lock:
            self._calls.append(call)

    def get_total_cost(self) -> float:
        """
        Get the total estimated cost in USD.

        Returns:
            Total cost as a float.
        """
        with self._lock:
            return sum(call.total_cost for call in self._calls)

    def get_breakdown(self) -> Dict:
        """
        Get cost breakdown by provider, model, and call type.

        Returns:
            Dictionary containing breakdown across multiple dimensions.
            Example:
            {
                'by_provider': {'openai': 12.50, 'anthropic': 5.25},
                'by_model': {'gpt-4o': 10.00, 'claude-sonnet': 5.25},
                'by_call_type': {'chat': 14.00, 'completion': 3.75},
                'total': 17.75
            }
        """
        breakdown = CostBreakdown()

        with self._lock:
            for call in self._calls:
                # By provider
                if call.provider not in breakdown.by_provider:
                    breakdown.by_provider[call.provider] = 0.0
                breakdown.by_provider[call.provider] += call.total_cost

                # By model
                if call.model not in breakdown.by_model:
                    breakdown.by_model[call.model] = 0.0
                breakdown.by_model[call.model] += call.total_cost

                # By call type
                if call.call_type not in breakdown.by_call_type:
                    breakdown.by_call_type[call.call_type] = 0.0
                breakdown.by_call_type[call.call_type] += call.total_cost

        breakdown.total = sum(breakdown.by_provider.values())

        # Round to 4 decimal places for currency
        breakdown.by_provider = {k: round(v, 4) for k, v in breakdown.by_provider.items()}
        breakdown.by_model = {k: round(v, 4) for k, v in breakdown.by_model.items()}
        breakdown.by_call_type = {k: round(v, 4) for k, v in breakdown.by_call_type.items()}
        breakdown.total = round(breakdown.total, 4)

        return breakdown.to_dict()

    def get_report(self) -> str:
        """
        Get a human-readable cost report.

        Returns:
            Formatted string with detailed cost information.
        """
        breakdown = self.get_breakdown()
        total_calls = len(self._calls)

        # Build report
        lines = [
            "=" * 60,
            "LLM API COST REPORT",
            "=" * 60,
            "",
            f"Total Calls: {total_calls}",
            f"Total Cost: ${breakdown['total']:.4f} USD",
            "",
        ]

        # By provider
        if breakdown['by_provider']:
            lines.append("Cost by Provider:")
            lines.append("-" * 40)
            for provider, cost in sorted(breakdown['by_provider'].items()):
                percentage = (cost / breakdown['total'] * 100) if breakdown['total'] > 0 else 0
                lines.append(f"  {provider:.<30} ${cost:>8.4f} ({percentage:>5.1f}%)")
            lines.append("")

        # By model
        if breakdown['by_model']:
            lines.append("Cost by Model:")
            lines.append("-" * 40)
            for model, cost in sorted(breakdown['by_model'].items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / breakdown['total'] * 100) if breakdown['total'] > 0 else 0
                lines.append(f"  {model:.<30} ${cost:>8.4f} ({percentage:>5.1f}%)")
            lines.append("")

        # By call type
        if breakdown['by_call_type']:
            lines.append("Cost by Call Type:")
            lines.append("-" * 40)
            for call_type, cost in sorted(breakdown['by_call_type'].items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / breakdown['total'] * 100) if breakdown['total'] > 0 else 0
                lines.append(f"  {call_type:.<30} ${cost:>8.4f} ({percentage:>5.1f}%)")
            lines.append("")

        # Summary statistics
        if total_calls > 0:
            with self._lock:
                avg_cost = breakdown['total'] / total_calls
                total_input_tokens = sum(call.input_tokens for call in self._calls)
                total_output_tokens = sum(call.output_tokens for call in self._calls)

            lines.append("Summary Statistics:")
            lines.append("-" * 40)
            lines.append(f"  Average Cost per Call: ${avg_cost:.4f}")
            lines.append(f"  Total Input Tokens: {total_input_tokens:,}")
            lines.append(f"  Total Output Tokens: {total_output_tokens:,}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracked calls."""
        with self._lock:
            self._calls.clear()

    def log_to_file(self, path: str) -> None:
        """
        Write the cost log to a JSON file.

        Args:
            path: File path where the log should be written.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            log_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_calls": len(self._calls),
                "total_cost": self.get_total_cost(),
                "breakdown": self.get_breakdown(),
                "calls": [call.to_dict() for call in self._calls]
            }

        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_calls(self) -> list[APICall]:
        """
        Get a copy of all tracked calls.

        Returns:
            List of APICall records.
        """
        with self._lock:
            return list(self._calls)

    def __repr__(self) -> str:
        """String representation."""
        total_cost = self.get_total_cost()
        call_count = len(self._calls)
        return f"CostTracker(calls={call_count}, total_cost=${total_cost:.4f})"
