# HR-3A thermalization contract

**Status:** CLOSED (2026-08-30, branch `HR-3`). The known unrelated HR-2E
strict-float targeted-test baseline failure is non-blocking for HR-3A closure.

## Authoritative transition

HR-3A is the explicit, mechanism-resolved conversion

\[
\mathbf q_{\rm deposition}
=\{q_{\rm ion},q_{\rm IB},q_{\rm Raman}\}
\longrightarrow
\mathbf q_{\rm thermal}
=\{q_{\rm th,ion},q_{\rm th,IB},q_{\rm th,Raman}\}.
\]

For the reference-compatible complete microscopic-thermalization model,

\[
q_{\rm th,c}=q_{\rm dep,c},\qquad
q_{\rm thermal}=\sum_c q_{\rm th,c}.
\]

The equal values do not make deposition and thermalization the same physical
object. HR-2 records optical-to-medium deposition during the pulse. HR-3A
records the later, reference-model approximation that this deposited energy is
available as thermal source after microscopic relaxation. It does not assert
instantaneous fs translational heating.

## Inputs and outputs

The implementation captures only the in-flight, HR-2 authoritative,
interval-average maps `q_ion`, `q_ib`, and `q_raman`, with the fixed
longitudinal schedule and transverse geometry already used for their HR-2
reductions. It emits

- full-z scalar `E_th_*_interval_J` and `E_th_*_pulse_J` using
  `sum(q_th[k]) * dx * dy * dz[k]`;
- independent T1/T2/T3 closure status, authority, source, scheme,
  active/inactive channel, unit, and schedule metadata.

HR-3A-R makes the maps transient and persists the full-z scalar ledger plus a
sparse physical-z `q_thermal` diagnostic sidecar. Full `[K, Ny, Nx]` thermal
or deposition map history is not a production output.

The source is fixed as `hr2_authoritative_deposition`. No thermal source is
constructed from field-energy loss, net electron-density change, recombination,
attachment, `Qacc`, `gamma_heat`, `Q_rot_vol`, `w_R`, `E_dep_rot_z`,
`Qacc_raman`, or signed Raman field loss.

## Closure and defensive rules

- **T1 identity/authority:** an active authoritative deposition channel is
  directly interpreted as its complete-microscopic thermal source without a
  duplicate map copy; inactive channels must be exact zero. An active
  non-authoritative channel is unavailable and fails closed.
- **T2 reduction:** interval reductions are recomputed from the thermal maps
  with the HR-2 schedule and geometry, then checked against the authoritative
  HR-2 interval energy ledger.
- **T3 sum:** `q_thermal` and `E_thermal` equal the three channel sums.
- T1, T2, and T3 are independent diagnostics. Overall authority is their
  conjunction; a T2 reduction failure, for example, leaves valid T1 and T3
  statuses unchanged.
- An inactive channel must be exact zero. Current Isaacs-compatible short-pulse
  configurations retain the inactive `IB` channel rather than removing it.
- A missing, non-finite, negative, shape-mismatched, schedule-mismatched, or
  non-authoritative active input cannot fall back to a legacy estimate. A
  non-authoritative active channel produces an explicitly non-authoritative
  thermal ledger with unavailable values.

## Stage boundary

HR-3A ends at microscopic thermalization. It does not calculate `Delta T`,
`delta rho`, `delta n`, a thermal-index screen, acoustic propagation,
post-acoustic/isobaric state, conduction, advection, or pulse-to-pulse slow
state. Those operations remain HR-3B/HR-3C or later.

HR-2E remains **DEFERRED** and the production longitudinal schedule remains
**NOT FROZEN**. HR-3A does not alter those conclusions.
