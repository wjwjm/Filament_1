# Complete Isaacs Eq. (27) C2 comparison

Classification: **electronic_eq27_operator_not_supported** (evidence gate passed).

- Candidate-current 1e22 onset shift: `0.03210538079646241` cm.
- Candidate improvement toward PyCAP at 1e22: `0.03210538079646241` cm.
- Candidate peak-density relative error to PyCAP: `0.0140648`.
- Full-axis RMSE current/candidate: `1.8351075059873155e+22` / `1.8162163471727834e+22` m^-3.

Fallback qualification: current full Eq.27 job 180748 and Raman-OFF job 180749 are `fallback_verified_non_strict` comparators. Their supplied audits and CSV path/SHA records passed, but this remains a non-strict fallback comparison and is not evidence of a strict same-run pair; those jobs used mixed_precision while the locked mother/candidate retains its baseline default linear precision.
Candidate provenance qualification: `verified_bundle_non_strict` (verified Git bundle after remote GitHub transport failure); this does not establish direct GitHub remote push/fetch verification.
Invalid jobs 179706 and 179988 are excluded from physical classification.

Causal interpretation: this classification covers the complete combined Eq.27 implementation, including electronic stage placement and electronic-rotational Heun coupling; it does not isolate the derivative algebra alone.

No coordinate shift, smoothing, renormalization, or replacement of the fixed PyCAP curve is applied.
