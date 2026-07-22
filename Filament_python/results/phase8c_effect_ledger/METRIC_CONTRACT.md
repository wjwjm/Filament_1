# Phase 8C-0 metric contract

Version: `phase8c.metric_contract.v1`.

This contract applies to all historical 120 fs curves indexed in `result_inventory.json`. It is intentionally a read-only post-processing contract: no source curve is smoothed, shifted, normalized again, repaired, or overwritten.

## Coordinate

All simulation curves use

```text
x_focus_cm = 100 * (z_m - 0.95)
```

The reference focus is `z = 0.95 m`. A stored `x_focus_cm` column is audited against this expression when `z_m` exists. The PyCAP digitized curve has no archived `z_m`, so its published `x_focus_cm` is used unchanged. Curves are never translated to align a peak or fit.

## Electron-density onset

The required thresholds are `1e19`, `1e20`, `1e21`, and `1e22 m^-3`. For each threshold, the onset is the first finite sample along increasing z that reaches or exceeds the threshold.

- Between two consecutive finite samples, the crossing uses linear interpolation.
- No interpolation crosses a `NaN`/`Inf`; no high-order fitting or smoothing is allowed.
- If the first finite point is already above the threshold, the returned position is that first point and its status is `left_censored_at_first_sample`.
- If a curve never crosses, the result is `null` with `not_crossed`.

The status is part of the metric. A left-censored PyCAP crossing may be plotted but is not eligible as the denominator of `fraction_of_total_pycap_offset`.

## Shape metrics

- `peak_density_m3`: finite global maximum.
- `peak_position_cm`: first position attaining that maximum.
- `peak_plateau_width_cm`: separation between the first and last exactly equal maximum sample.
- `left_halfmax_crossing_cm`, `right_halfmax_crossing_cm`, and `fwhm_cm`: linear half-maximum crossings; `fwhm_cm = null` if either side is absent.
- Rise: 10% and 90% of that curve's own peak on the ascending branch.
- Fall: 90%, 50%, and 10% of that curve's own peak on the descending branch.
- Tail integrals: direct trapezoidal integral in `m^-3 cm` from the peak, peak + 5 cm, and peak + 10 cm. A start outside the archived domain returns `null`.

## PyCAP comparison

For each complete non-PyCAP curve, linear-scale RMSE, log10-scale RMSE, and median absolute log10 error are evaluated only over the common finite x interval. The comparison grid is the merged native sample positions, joined by linear interpolation; this is comparison interpolation, not smoothing. The report must also carry peak-position and crossing-position differences.

`epsilon_x = 0.10 cm` is the fixed practical position tolerance. A contribution fraction is `null` if the total offset is absent, left-censored, changes sign, or has magnitude below this tolerance.

## Effect-pair rules

For a ledger comparison, `delta_* = comparison - baseline`; negative means the comparison turns on earlier. The total current-PyCAP offset uses the same sign. Only `strict_single_delta` pairs may have `high` confidence. Any other pair is capped at `medium`, and a missing curve is `not_interpretable`.

Each generated row carries source file, source array key, contract version, calculation status, and data-quality flags in `effect_metrics_by_result.json`/CSV.
