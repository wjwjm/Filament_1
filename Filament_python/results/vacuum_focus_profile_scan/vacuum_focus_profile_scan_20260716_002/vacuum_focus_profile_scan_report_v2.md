# FT90 vacuum-focus window closure and independent Fresnel verification

All coordinates use `x_focus = 100*(z-0.95) cm`; zero is permanently the 0.95 m geometric focus.

## Numerical status

- Resolution convergence (P1 512² vs 1024² at 8 mm): `True`; difference `-0.0789` cm.
- Window convergence (all P1--P6): `True`.  10→12 mm: `False`; 12→14 mm: `True`.
- Independent continuous Fresnel crosscheck at `14mm_896`: `True`.
- FFT `I_max` versus on-axis focus consistency: `True`.

## Physical evidence

- Fixed-density rising shifts: 120 fs `-2.589` cm; 40 fs `-3.270` cm.
- Final-window positive differential shifts: `[0.48742877479162017, 0.037989203843501684]` cm.
- Candidates closing both pulse widths within 1 cm: `[]`.

## Final classification

**not_supported**.

In the converged and independently verified window, the P1--P6 mathematical definitions do not supply a downstream shift capable of compensating the 2.6--3.3 cm density-rise advance.
