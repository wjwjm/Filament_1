"""Freeze a clean execution and the existing input/reference contracts locally."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from .entry import read, require
from .protocol import BASE, SCIENCE_TREE, TASK, plan


def prepare(report, *, inherited_manifest, reference_contract, input_plan):
    repo = Path(__file__).resolve().parents[3]
    pv._ensure_clean(repo)
    require(pv._git(repo, 'rev-parse', 'HEAD:Filament_python/KHz_filament') == SCIENCE_TREE, 'S46D_FROZEN_SCIENCE')
    require(pv._git(repo, 'merge-base', BASE, 'HEAD') == BASE, 'S46D_DESCENDANT_OF_APPROVED_BASE')
    report = Path(report).resolve()
    require(not report.exists(), 'S46D_NON_OVERWRITING_PREPARATION')
    inherited = read(inherited_manifest)
    paths = {row['path'] for row in pv.iter_records(inherited, require_hash_scope=True) if row['classification'] == 'tracked_text'}
    paths.update(path.relative_to(repo).as_posix() for path in Path(__file__).parent.iterdir() if path.suffix in ('.py', '.sbatch', '.md'))
    paths.update(('Filament_python/tools/run_local_tests.ps1', 'Filament_python/tools/hpc_ops/audit_batch_entry.py'))
    require(len(paths) >= 252, 'S46D_INHERITED_SOURCE_CLOSURE')
    report.mkdir(parents=True)
    sg.write(report / 'PLAN.json', plan())
    manifest = pv.create_manifest(repo, report / 'EXECUTION_SOURCE_MANIFEST.json', sorted(paths), [])
    pv.validate_manifest(repo, report / 'EXECUTION_SOURCE_MANIFEST.json', require_clean=True, require_hash_scope=True)
    historical = read(input_plan)
    sg.write(report / 'INPUT_CONTRACT.json', dict(input_contract=historical['input_contract'], expected_input_fingerprints=historical['expected_input_fingerprints'], source=str(input_plan), classification='external', hash_scope='raw_bytes', source_raw_sha256=pv.raw_sha256_file(input_plan)))
    shutil.copyfile(reference_contract, report / 'REFERENCE_INPUT_RAW_RECORDS.json')
    binding = dict(schema='filament.cpu_s4_6d.execution.v1', task=TASK, baseline_commit=BASE, science_tree=SCIENCE_TREE,
                   execution_commit=manifest['head'], execution_branch=manifest['branch'],
                   plan_raw_sha256=pv.raw_sha256_file(report / 'PLAN.json'),
                   source_manifest_raw_sha256=pv.raw_sha256_file(report / 'EXECUTION_SOURCE_MANIFEST.json'),
                   inherited_manifest_raw_sha256=pv.raw_sha256_file(inherited_manifest),
                   input_contract_raw_sha256=pv.raw_sha256_file(report / 'INPUT_CONTRACT.json'),
                   reference_contract_raw_sha256=pv.raw_sha256_file(report / 'REFERENCE_INPUT_RAW_RECORDS.json'),
                   submission_authorized=False, default='I0', production_qualified=False)
    sg.write(report / 'EXECUTION.json', binding)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'inherited-manifest', 'reference-contract', 'input-plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.report, inherited_manifest=args.inherited_manifest, reference_contract=args.reference_contract, input_plan=args.input_plan)
    print(result['execution_commit'])


if __name__ == '__main__':
    main()
