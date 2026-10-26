"""Repository-local support closure for the CPU-S4 checkpoint entry."""

from .portable_binding import (
    ensure_support_imports,
    g2_adapter_path,
    load_g2_adapter,
    repository_root,
    support_root,
    support_module,
)

__all__ = [
    "ensure_support_imports",
    "g2_adapter_path",
    "load_g2_adapter",
    "repository_root",
    "support_root",
    "support_module",
]
