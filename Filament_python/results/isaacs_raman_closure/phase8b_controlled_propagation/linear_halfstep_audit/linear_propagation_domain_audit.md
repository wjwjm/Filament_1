# Linear propagation and energy-integration domains

| Array/domain | Shape or extent | Integration extent | Contains excluded nonzero energy? | Included in total-energy budget? |
|---|---|---|---|---|
| Production field `E` | `[Nt, Ny, Nx] = [384,512,512]` | Entire Cartesian field | No excluded region in BK-NEE path | Yes, through `U_z` and operator checkpoints |
| Time FFT `Ew` | Same shape | Not integrated separately | No padding | No separate channel |
| Per-frequency spatial FFT `S` | Same shape per time-frequency slice | Not integrated separately | No padding, mask, filter, or crop | No separate channel |
| BK-NEE transfer output | Same shape | Not integrated separately | No designed amplitude removal | No separate channel |
| Inverse FFT outputs | Same shape | Same physical samples as `E` | No crop/guard-cell removal | Yes, when converted to `U_z` |
| Runtime energy diagnostic | Scalar | `sum(I) * dt * dx * dy` over all `[Nt,Ny,Nx]` | No | Yes |
| Saved archive energy traces | `[Nz]`, `Nz=15000` | Already reduced runtime scalar | Not applicable | Yes |
| Saved `I_out_center_t` | `[Nt]` final centreline only | Cannot recover transverse energy | It omits most field energy, so is not used for energy accounting | No |

## Findings

The selected BK-NEE path does not use FFT padding, crop, guard cells,
absorbing boundaries, spatial/temporal masks, or spectral filters.  The
runtime energy reduction and the operator-energy checkpoints both use the
full Cartesian propagated field with `dt*dx*dy`; no cylindrical Jacobian is
used or appropriate.

The archive does not retain `E(z,t,y,x)` or any complex checkpoint fields.
Consequently, the stored Job 179988 data can verify scalar runtime-history
consistency but cannot independently reintegrate field energy or replay a
linear half step at its six selected positions.  This is an observability
limit, not evidence that the runtime integration domain is wrong.
