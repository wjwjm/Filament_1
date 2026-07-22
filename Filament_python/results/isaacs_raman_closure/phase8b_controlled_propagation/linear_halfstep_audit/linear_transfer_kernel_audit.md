# Linear transfer kernel audit

- Production branch: `bk_nee -> step_linear_bk_nee_factorized`.
- The selected BK-NEE transfer is a pure phase by design; it has no physical linear absorption, high-k filter, evanescent deletion, mask, crop, or padding operation.
- complex64 `max(abs(abs(H)-1))`: `1.1920928955078125e-07`.
- float64 reference `max(abs(abs(H)-1))`: `2.220446049250313e-16`.
- complex64 zero bins: `0` of `100663296`.
- Per-checkpoint attenuated-bin energy fractions cannot be reconstructed from Job 179988 because it contains no complex field checkpoints.
