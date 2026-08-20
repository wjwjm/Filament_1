# historical_fr_mixture 120 fs causal comparison

- 1e22 onset: production=-16.411900580443586 cm, mixture=-13.395007613340857 cm, PyCAP=-14.027210012757275 cm.
- Mixture minus production shift at 1e22: **3.016892967102729 cm** (later (toward PyCAP), epsilon=0.100 cm).
- Peak rho: production=6.4609e+22 m^-3 at -14.440 cm; mixture=2.0350e+22 m^-3 at -9.855 cm; PyCAP=6.4546e+22 m^-3 at -12.184 cm.
- RMSE vs PyCAP (rho_max_z): production=1.8723e+22, mixture=2.4434e+22, Raman-OFF=2.1443e+22.

Core question: with every other model component frozen, does swapping the Raman phase operator to the pre-April f_R mixture move the 120 fs onset back toward PyCAP? See `historical_fr_mixture_comparison_summary.json` for the full effect-chain metrics.

