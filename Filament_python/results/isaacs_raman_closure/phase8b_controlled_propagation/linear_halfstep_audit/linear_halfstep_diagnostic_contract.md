# Linear-half-step diagnostic contract

This opt-in diagnostic is enabled only by
`propagation.diag_linear_halfstep_energy=true` on the BK-NEE production path.
It does not alter the field, the transfer multiplier, or any physical model.

For each accepted z step and for each linear half step it records float64
energy reductions before the operator, after the time FFT, after the spatial
transfer multiplier, after the inverse spatial FFT, and after the inverse time
FFT.  No mask/filter/crop stages exist on this selected path; their explicit
loss channels are therefore exactly zero rather than inferred.

Sign convention:

```text
field_delta_J = U_after - U_before       # negative means field loss
explicit_loss_J > 0                      # energy intentionally removed
unaccounted_residual_J = field_delta_J + sum(explicit_loss_J)
```

Thus a negative residual is a measured but unaccounted linear field loss.  It
must not be called physical absorption unless a distinct explicit operation
and physical/numerical design rationale have been identified.
