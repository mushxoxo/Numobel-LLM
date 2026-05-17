"""ValidationResult dataclass — typed return value for validate_response()."""
from dataclasses import dataclass, field

__all__ = ["ValidationResult"]


@dataclass
class ValidationResult:
    valid: bool
    severity: str | None = None
    violations: list[str] = field(default_factory=list)
    fallback_content: str | None = None
    recovery_strategy: str | None = None
