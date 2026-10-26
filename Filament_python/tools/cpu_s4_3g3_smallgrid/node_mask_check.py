"""One actual-node W48 failure/recovery check on a declared micro grid."""
import argparse
import os
from pathlib import Path

from .benchmark import write
from .candidate import make_factory, unoptimized
from .scientific_context import context, frozen_tools, require


def run(report, work):
    import numba
    from KHz_filament.propagate import propagate_one_pulse
    unoptimized()
    require(os.environ.get("SLURM_JOB_ID") and numba.config.NUMBA_NUM_THREADS == 48, "ACTUAL_NODE_POOL48")
    report = Path(report)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    cases, exact = frozen_tools(report)
    ctx = context(report, shape=(384, 17, 19))
    numba.set_num_threads(24)
    prop, dispatch = make_factory()(propagate_one_pulse)
    state = dispatch.cpu_s4_3g_state
    def execute(label):
        field, kwargs, view = cases.prepare(ctx, work / label, intervals=1)
        try:
            result = prop(field, **kwargs)
            return cases.finish(result, view, work / label, intervals=1)
        finally:
            view.close()
    baseline = execute("baseline")
    oracle = exact.save_tree(work / "oracle", baseline, compressed=True)
    original = state._internal_stage
    seen = []
    def injected(call_context, field, params, *, stage2):
        result = original(call_context, field, params, stage2=stage2)
        if not stage2 and not seen:
            numba.set_num_threads(48)
            seen.append(numba.get_num_threads())
            raise RuntimeError("SG_NODE_INJECTED_MASK48_STAGE1_EXCEPTION")
        return result
    state._internal_stage = injected
    failure = None
    try:
        execute("specified_failure")
    except RuntimeError as error:
        failure = str(error)
    finally:
        state._internal_stage = original
    require(seen == [48] and failure == "SG_NODE_INJECTED_MASK48_STAGE1_EXCEPTION", "ACTUAL_MASK48_EXCEPTION_REACHED")
    require(numba.get_num_threads() == 24 and state._g_pending_cap is None and getattr(state._local, "context", None) is None and not state.metrics()["active_lifecycle_identity"], "ACTUAL_MASK24_AND_LIFETIME_EXCEPTION_RECOVERY")
    recovered = execute("recovered")
    comparison = exact.compare(Path(oracle), recovered, work / "RECOVERY_EXACT.jsonl", disk=True)
    require(comparison["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "NODE_RECOVERY_EXACT")
    result = dict(status="PASS_ACTUAL_W48_MASK_EXCEPTION_RESTORES24", shape=[384, 17, 19], baseline="PASS", injected_real_mask=seen,
                  exception=failure, mask_after=numba.get_num_threads(), recovery=comparison, scope="MICRO_CONTRACT_ONLY_NOT_MAIN_WORKER_QUALIFICATION")
    write(work / "NODE_MASK_CHECK.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()
    print(run(arguments.report, arguments.work))
