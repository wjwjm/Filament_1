# HR-3A-R streaming diagnostic-storage decision

HR-3A-R preserves the complete microscopic-thermalization contract while
changing only data lifetime.

## Frozen storage contract

- Current-interval `q_ion`, `q_ib`, `q_raman`, and `q_thermal` are transient.
  Production propagation keeps no Python full-z map list and performs no
  full-z map stack.
- The authoritative full-z record is the O(K) scalar thermal ledger: channel
  and total interval energies, thermal-map statistics, T2/T3 residuals,
  pulse reductions, and the first failed canonical interval.
- Sparse `q_thermal` maps are diagnostic artifacts only. They are sampled by
  physical position, written directly to a NumPy memmap sidecar, and are not
  embedded in the main NPZ.
- Production sampling uses 5 mm outer targets and 1 mm focus targets, both
  mapped to the nearest interval midpoint using the absolute schedule. The
  archive also records the first/last interval, focus boundaries, and a
  finite focal-plane landmark when available.
- Production writes only sampled total `q_thermal`. A validation-configured
  sink can additionally stream sampled deposition components; it still does
  not create thermal channel duplicates.

## Authority boundary

`q_thermal` is not connected to the legacy `Q2D -> gamma_heat -> dn_gas`
compatibility path. Diagnostics explicitly state whether the HR-3A source is
authoritative, whether an authoritative HR slow-state update is active (false),
and whether the legacy slow-heat compatibility path is active (true).

Persistent `Delta T`, `delta rho`, `delta n`, interval-to-anchor projection,
slow-state storage, and interpulse disk-backed evolution remain HR-3B/HR-3C
responsibilities. HR-2E remains **DEFERRED** and the production longitudinal
schedule remains **NOT FROZEN**.

## Static scale estimate

For `K=16000`, `Nx=Ny=512`, and float32, one map is 1 MiB. The prior map-only
post-loop lower bound was about 156.25 GiB (and approximately 171.875 GiB with
the sum temporary). The streaming working set is four current maps, about
4 MiB, plus at most one 1 MiB host staging map. For a 1.3 m run with a 0.3 m
focus window, the nominal target union is approximately `261 + 301 - 61 =
501` maps, or about 501 MiB on disk. The actual archive count is always
`build_physical_sample_plan(...).count` after interval-midpoint snapping,
landmark inclusion, and interval-index de-duplication.
