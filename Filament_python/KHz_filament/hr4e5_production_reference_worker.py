"""Independent trajectory process for the E5-1B production runner.

The parent remains the only formal coordinator.  This process opens a read
view of its durable campaign and executes the existing R or final C method.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from .confio import load_all
from .hr4e5_evidence import atomic_json
from .hr4e5_formal_entry import build_prefix_schedule, resolve_storage_lock_timeout_s
from .hr4e5_paired_campaign import PairedCampaign
from .hr4e5_recovery_identity import capture_implementation_identity, require_same_implementation
from .hr4e5_storage import process_identity
from .hr4e5_production import (
    E5AProductionSpec, FORMAL_SITE_ADMISSION, _E5AProductionRunner,
    _LocalPolicyCapacityProvider, _ManifestCapacityProvider, _read_manifest,
    _schedule_from_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--pulse", type=int, required=True)
    parser.add_argument("--trajectory", choices=("R", "C"), default="R")
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()

    raw_spec = json.loads(args.spec_json)
    if not isinstance(raw_spec, dict):
        raise ValueError("Production subprocess specification must be an object")
    spec = E5AProductionSpec(**raw_spec)
    root = Path(spec.root).resolve()
    result_path = args.result_path.resolve()
    if result_path.is_relative_to(root) or root.is_relative_to(result_path):
        raise ValueError("Reference process result must be outside campaign storage")

    identity = json.loads((root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
    resolve_storage_lock_timeout_s(
        identity, explicit=spec.storage_lock_timeout_s,
    )
    implementation = json.loads((root / "E5_1B_IMPLEMENTATION_IDENTITY.json").read_text(encoding="utf-8"))
    require_same_implementation(implementation, capture_implementation_identity())
    if identity.get("campaign_id") != spec.campaign_id or int(identity.get("n_pulses", -1)) != int(spec.n_pulses):
        raise ValueError("Reference child campaign identity differs")
    if args.trajectory == "C" and int(args.pulse) != int(spec.n_pulses) - 1:
        raise ValueError("Candidate subprocess is restricted to the final pulse")

    components = load_all(str(spec.config_path))
    full_components = load_all(str(spec.full_config_path or spec.config_path))
    schedule = build_prefix_schedule(
        _schedule_from_config(full_components[2]), int(identity["schedule_identity"]["k"])
    )
    if json.loads(json.dumps(schedule.as_metadata())) != identity["schedule_identity"]["metadata"]:
        raise ValueError("Reference child schedule differs from admission identity")

    campaign = PairedCampaign.open(
        root, n_pulses=int(spec.n_pulses), campaign_id=str(spec.campaign_id),
        max_campaign_live_bytes=int(spec.max_campaign_live_bytes),
        final_output_budget_bytes=int(spec.final_output_budget_bytes),
        safety_margin_bytes=int(spec.safety_margin_bytes), require_quota=True,
        storage_lock_timeout_s=spec.storage_lock_timeout_s,
        admission_identity=identity, formal=True, require_intents=True,
    )
    if spec.admission_level == FORMAL_SITE_ADMISSION:
        if spec.site_resource_manifest_path is None:
            raise ValueError("formal Reference child lacks site resource evidence")
        campaign.storage.provider = _ManifestCapacityProvider(
            _read_manifest(Path(spec.site_resource_manifest_path), label="site resource")
        )
    else:
        campaign.storage.provider = _LocalPolicyCapacityProvider(root, spec.max_campaign_live_bytes)

    runner = _E5AProductionRunner(spec, identity, schedule, components, implementation)
    runner.driver = SimpleNamespace(campaign=campaign)
    started = time.monotonic()
    process = process_identity()
    result = runner._run_pulse(args.trajectory, int(args.pulse))
    if os.environ.get("HR4E5_GPU_ROLE"):
        from .hr4e5_gpu_bootstrap import update_final_metadata

        active = campaign.storage.active_writers(include_management=False)
        if args.trajectory == "C":
            active_c = [row for row in active if row.get("trajectory") == "C"]
            if active_c:
                raise RuntimeError("Candidate child left active scientific writers")
        else:
            active_r = [row for row in active
                        if row.get("trajectory") == "R" and int(row.get("pulse", -1)) == int(args.pulse)]
            if active_r:
                raise RuntimeError("Reference child left active scientific writers")
        update_final_metadata(
            science_output_status="PASS",
            writer_grant_status="NO_CHILD_GRANT_WRITERS_CLOSED",
        )
    atomic_json(result_path, {
        "schema": "khz_filament.hr4e5.reference_process_result.v1",
        "trajectory": args.trajectory, "pulse": int(args.pulse), "campaign_id": spec.campaign_id,
        "coordinator_epoch": str(campaign.active_coordinator_epoch()),
        "process_identity": process, "parent_pid": os.getppid(),
        "started_monotonic_s": started, "ended_monotonic_s": time.monotonic(),
        "result": result,
    }, overwrite=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
