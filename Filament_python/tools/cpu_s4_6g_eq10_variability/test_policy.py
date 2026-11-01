"""Pure-stdlib policy-plan and scalar-analysis tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from . import controller
from .policy_analysis import analyze_policy, policy_plan


def _evidence(plan, metric_i=80.0, *, k2_slow=False, rr_delta=1.0, role_i=None):
    accepted = {}
    launches = {}
    for row in plan["matrix"]:
        case = row["case"]
        if row["kind"] == "RR_CONTROL":
            value = 100.0 if case.endswith("R1") else 100.0 + rr_delta
        elif row["policy"] == "I":
            value = 120.0 if k2_slow and case.startswith("K2") else (role_i or {}).get(row["role"], metric_i)
        else:
            value = 100.0
        accepted[case] = {
            "role": row["role"], "intervals": row["intervals"], "exact_gate_pass": True,
            "later_body_s": value, "application_service_s": value,
        }
        launches[case] = {
            "policy": row["policy"],
            "policy_evidence": {"valid_body_snapshot": True},
            "launch_scope_s": value * 1.1,
            "process_service_s": value * 1.2,
            "clock_metrics": {"propagation_s": value, "initialization_total_s": value * 0.9},
        }
    return accepted, launches


class PolicyTests(unittest.TestCase):
    def test_fixed_plan_order_and_prefix(self):
        plan = policy_plan()
        self.assertEqual(plan["order"], [row["case"] for row in plan["matrix"]])
        self.assertEqual(plan["expected_mems"], [0, 1, 2, 3])
        self.assertEqual(plan["policies"]["D"]["argv_prefix"], [])
        self.assertEqual(plan["policies"]["I"]["argv_prefix"], ["/usr/bin/numactl", "--interleave=all"])
        self.assertEqual(plan["policy_plan_revision"], "LOCAL_RESERVATION_EXCEPTION_V1")
        self.assertEqual(plan["local_reservation_exception"]["max_vmas"], 2)
        self.assertEqual(plan["local_reservation_exception"]["resident_pages_total"], 0)

    def test_faster_slower_and_undetermined_classifications(self):
        plan = policy_plan()
        accepted, launches = _evidence(plan, 80.0)
        self.assertEqual(analyze_policy(accepted, launches)["classification"], "OBSERVED_POLICY_FASTER")
        accepted, launches = _evidence(plan, 120.0)
        self.assertEqual(analyze_policy(accepted, launches)["classification"], "OBSERVED_POLICY_SLOWER")
        accepted, launches = _evidence(plan, 80.0, k2_slow=True)
        self.assertEqual(analyze_policy(accepted, launches)["classification"], "NO_DETERMINATE_POLICY_SIGNAL")

    def test_effect_below_default_rr_drift_is_undetermined(self):
        plan = policy_plan()
        accepted, launches = _evidence(plan, 99.0, rr_delta=2.0)
        result = analyze_policy(accepted, launches)
        self.assertEqual(result["classification"], "NO_DETERMINATE_POLICY_SIGNAL")
        self.assertGreater(result["rr_drift_max_abs_log"]["later_body"], 0.0)

    def test_contract_requires_same_nonzero_sign_and_mean_above_drift(self):
        plan = policy_plan()
        accepted, launches = _evidence(plan, 80.0)
        # K1 is faster while K2 is slower for role R: the role is unresolved.
        for key in ("later_body_s", "application_service_s"):
            accepted["K2_R1"][key] = 120.0
        for key in ("propagation_s", "initialization_total_s"):
            launches["K2_R1"]["clock_metrics"][key] = 120.0
        launches["K2_R1"]["launch_scope_s"] = 132.0
        launches["K2_R1"]["process_service_s"] = 144.0
        result = analyze_policy(accepted, launches)
        self.assertEqual(result["role_metric_classification"]["R"]["overall"]["later_body"], "NO_DETERMINATE_POLICY_SIGNAL")
        # Equality with RR drift is not sufficient because the contract uses >.
        accepted, launches = _evidence(plan, 80.0)
        threshold = 100.0 * math.exp(-0.01)
        for case in ("CTRL_LATE_R2", "K1_R2", "K1_X2", "K2_R1", "K2_X1"):
            accepted[case]["later_body_s"] = threshold
            accepted[case]["application_service_s"] = threshold
            launches[case]["clock_metrics"]["propagation_s"] = threshold
            launches[case]["clock_metrics"]["initialization_total_s"] = threshold
            launches[case]["launch_scope_s"] = threshold * 1.1
            launches[case]["process_service_s"] = threshold * 1.2
        result = analyze_policy(accepted, launches)
        self.assertEqual(result["role_metric_classification"]["R"]["overall"]["later_body"], "NO_DETERMINATE_POLICY_SIGNAL")
        # A zero block effect is unresolved even when the other block is faster.
        accepted, launches = _evidence(plan, 80.0)
        for key in ("later_body_s", "application_service_s"):
            accepted["K2_R1"][key] = 100.0
        for key in ("propagation_s", "initialization_total_s"):
            launches["K2_R1"]["clock_metrics"][key] = 100.0
        launches["K2_R1"]["launch_scope_s"] = 110.0
        launches["K2_R1"]["process_service_s"] = 120.0
        result = analyze_policy(accepted, launches)
        self.assertEqual(result["role_metric_classification"]["R"]["overall"]["later_body"], "NO_DETERMINATE_POLICY_SIGNAL")

    def test_role_reversal_is_not_hidden_by_balanced_average(self):
        plan = policy_plan()
        accepted, launches = _evidence(plan, role_i={"R": 70.0, "X": 120.0})
        result = analyze_policy(accepted, launches)
        self.assertEqual(result["role_metric_classification"]["R"]["overall"]["later_body"], "OBSERVED_POLICY_FASTER")
        self.assertEqual(result["role_metric_classification"]["X"]["overall"]["later_body"], "OBSERVED_POLICY_SLOWER")
        self.assertEqual(result["metric_classification"]["later_body"], "NO_DETERMINATE_POLICY_SIGNAL")
        self.assertEqual(result["policy_pair_table"][0]["actual_order"], ["K1_R1", "K1_X1", "K1_X2", "K1_R2"])
        pair = next(row for row in result["policy_pair_table"][0]["roles"] if row["role"] == "R")
        self.assertEqual(pair["default"]["case"], "K1_R1")
        self.assertEqual(pair["interleave"]["case"], "K1_R2")
        self.assertIn("ratio_I_over_D", pair["effect_I_over_D"]["launch_scope_s"])

    def test_policy_order_tamper_rejected(self):
        plan = policy_plan()
        accepted, launches = _evidence(plan)
        launches["K1_R1"]["policy"] = "I"
        with self.assertRaisesRegex(ValueError, "POLICY_ORDER_BINDING"):
            analyze_policy(accepted, launches)

    def test_plan_and_context_binding_tamper_rejected(self):
        plan = policy_plan()
        with tempfile.TemporaryDirectory() as root:
            report = Path(root)
            plan_path = report / "POLICY_PLAN.json"
            plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            context = {
                "diagnostic_kind": "MEMORY_POLICY_CROSSOVER",
                "policy_plan_raw_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            }
            self.assertEqual(controller._validate_policy_plan_binding(report, context), plan)
            context["policy_plan_raw_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "POLICY_PLAN_RAW_SHA256"):
                controller._validate_policy_plan_binding(report, context)
            context["policy_plan_raw_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            tampered = dict(plan, order=list(reversed(plan["order"])))
            plan_path.write_text(json.dumps(tampered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            context["policy_plan_raw_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "POLICY_PLAN_SOURCE_BINDING"):
                controller._validate_policy_plan_binding(report, context)


if __name__ == "__main__":
    unittest.main()
