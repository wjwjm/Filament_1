# HR-2E Stage 1 preparation

Status: prepared locally; no HPC job had been submitted when this document was generated.

## Historical proposal evidence

- 40 fs legacy ionization-deposition proxy: current 0.85–1.05 m fine window covers 72.606%; proposed 0.75–1.05 m covers 99.999%.
- 120 fs proxy: current window covers 45.801%; proposed window covers 96.579%.
- 120 fs has the larger normalized longitudinal gradient and is the Stage 2 worst-case pulse width.
- These historical diagnostics are proposal-only and are not authoritative convergence evidence.
- The remaining 120 fs legacy tail beyond 1.05 m is broad and remains covered by the base-spacing part of every schedule; only new canonical candidate/fine results may decide whether that base spacing is adequate.

## Fixed schedule family

| Schedule | Base dz | Focus dz | Focus window | Intervals |
|---|---:|---:|---:|---:|
| coarse | 0.20 mm | 0.10 mm | 0.75–1.05 m | 8001 |
| candidate | 0.10 mm | 0.05 mm | 0.75–1.05 m | 16000 |
| fine | 0.05 mm | 0.025 mm | 0.75–1.05 m | 32000 |

The coarse schedule contains one approximately `1.11e-13 m` final clipping interval caused by the frozen schedule builder's floating-point accumulation. It is retained and explicitly marked; it is not used as representative spacing.

## Authoritative Raman diagnostic boundary

All six temporary configurations reuse the validated `full_isaacs_eq27`, Heun, Strang, exact-piecewise-linear Raman configuration. They do not change the production legacy Raman default. Official comparison additionally requires same execution Git SHA, config hashes bound to the manifest, fp32 for every case, full operator applied on every interval, Level-1/2 closure pass, authoritative total deposition, and finite Level-3 field-energy bookkeeping.

## Execution plan and estimate

- Stage 2: submit 120 fs coarse/candidate/fine as three independent one-GPU jobs.
- Stage 3: only after 120 fs candidate-vs-fine passes, submit 40 fs candidate/fine as two independent one-GPU jobs.
- Historical full-Raman timing was about 4.06 h for 15000 intervals on the validated GPU path. Linear interval-count estimates are approximately 2.2 h (coarse), 4.3 h (candidate), and 8.7 h (fine), excluding queue time. Stage 2 jobs are intended to run concurrently when scheduler capacity permits.
- Repeated identical jobs planned: 0.

Production configuration changed: no.

Full pytest planned: no.
