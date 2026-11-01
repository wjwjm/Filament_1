"""Named controller seam for the bounded full-size campaign.

The implementation stays in :mod:`hpc` so the batch entry and the parent
control script share exactly one launch/supervision path.
"""

from .hpc import launch, main, node_gate, preflight, run

__all__ = ["launch", "main", "node_gate", "preflight", "run"]
