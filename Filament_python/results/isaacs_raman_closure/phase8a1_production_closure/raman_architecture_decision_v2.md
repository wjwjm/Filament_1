# Raman architecture decision v2

Selected architecture: `ready_full_operator`.

The actual legacy production split gate is `failed`; failed cases: 40fs_tl, 120fs_positive_chirp, 120fs_negative_chirp. The full Eq. (27) reference gate is `passed`. The candidate therefore uses the opt-in `full_isaacs_eq27` Heun operator, recomputes the Raman response at the intermediate stage, and rejects legacy Raman absorption. The analogous electronic-Kerr operator issue is recorded but is outside Phase 8A.1 and was not changed.
