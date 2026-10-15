"""CPU-Q2 B1/I1/I2/B2 adapter over the reviewed Q1 real-window runner."""
from __future__ import annotations

import sys

from tools.cpu_o1.q1_t0_real import main as _q1_real_main


def main(argv=None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if "--q2" in supplied:
        raise SystemExit("q2_t0_real supplies --q2; do not pass it twice")
    return _q1_real_main(["--q2", *supplied])


if __name__ == "__main__":
    raise SystemExit(main())
