# Phase 8B-R Task R1 preflight correction audit

- Status: **passed**
- Full 1.3 m jobs submitted: **0**
- Phase 8B-R Task R2 executed: **false**
- The original preflight was a false positive because the measured per-step p99 closure was not part of admission.
- Original p99: `0.01772617394104599`.
- Stable-difference-only p99: `0.004451706493273376`.
- Corrected p99: `0.00013349909517273764` (contract `<1e-3`).
- Corrected cumulative closure: `2.7029900593333878e-05` (contract `<5e-3`).
- Legacy Raman alpha maximum: `0.0`.
- Strict full configuration and summary now agree that Raman absorption is OFF.
- A separately authorized Task R2 is still required before any full Job 1 submission.
