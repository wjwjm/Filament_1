"""Conservative storage bound for one serial S4 OFF/PROFILE allocation."""
from __future__ import annotations

import math
from numbers import Integral

GIB = 1024 ** 3
MIB = 1024 ** 2
FROZEN_CAP_MAX_BYTES = 11 * GIB
RESERVE_FRACTION = 0.20
EXPECTED_SOURCE_SHAPE = (384, 351, 301)
MAX_DIAGNOSTIC_FIELDS = 256
MAX_PROFILE_RECEIPT_BYTES = 512 * MIB
OTHER_LOGS_AND_STAGING_BYTES = 512 * MIB
PROJECTED_AUXILIARY_FILE_COUNT = 100


def _integer(name: str, value, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def estimate_storage_budget(*, source_shape, input_records, diagnostic_field_count,
                            current_bytes: int, cap_bytes: int,
                            filesystem_available_bytes: int,
                            filesystem_available_inodes: int,
                            quota_available_bytes: int,
                            quota_available_inodes: int) -> dict:
    """Estimate persisted peak and report its file-class itemization.

    The output inventory includes the endpoint arrays and six sink samples.
    The separate first-boundary checkpoint adds another E, POST, Qacc,
    Qacc_raman and diagnostics archive for each case. Diagnostic archives use
    a deliberately conservative bound derived from the frozen field count;
    PROFILE output is separately capped by the output inventory.
    """
    if not isinstance(source_shape, (list, tuple)) or len(source_shape) != 3:
        raise ValueError("source shape must contain nt, ny and nx")
    shape = tuple(_integer("source dimension", value, minimum=1)
                  for value in source_shape)
    if shape != EXPECTED_SOURCE_SHAPE:
        raise ValueError(f"S4 source shape mismatch: expected {EXPECTED_SOURCE_SHAPE}, got {shape}")
    if not isinstance(input_records, (list, tuple)) or len(input_records) != 14:
        raise ValueError("storage budget requires fourteen raw-hash-verified inputs")
    input_sizes = []
    for row in input_records:
        if not isinstance(row, dict) or row.get("identity_status") != "PASS":
            raise ValueError("storage budget requires fourteen raw-hash-verified inputs")
        input_sizes.append(_integer("input size_bytes", row.get("size_bytes")))
    diagnostic_field_count = _integer("diagnostic_field_count", diagnostic_field_count, minimum=1)
    if diagnostic_field_count > MAX_DIAGNOSTIC_FIELDS:
        raise ValueError("frozen diagnostic contract exceeds the bounded storage estimate")

    current_bytes = _integer("current_bytes", current_bytes)
    cap_bytes = _integer("cap_bytes", cap_bytes, minimum=1)
    filesystem_available_bytes = _integer("filesystem_available_bytes", filesystem_available_bytes)
    filesystem_available_inodes = _integer("filesystem_available_inodes", filesystem_available_inodes)
    quota_available_bytes = _integer("quota_available_bytes", quota_available_bytes)
    quota_available_inodes = _integer("quota_available_inodes", quota_available_inodes)
    if cap_bytes > FROZEN_CAP_MAX_BYTES:
        raise ValueError("storage cap exceeds frozen Q1 helper limit of 11 GiB")

    nt, ny, nx = shape
    pixels = ny * nx
    e_bytes = math.prod(shape) * 16
    endpoint_post_per_case = 3 * 8 * pixels * 8
    first_boundary_post_per_case = 3 * pixels * 8
    six_sinks_per_case = 6 * 8 * pixels * 8
    endpoint_qacc_per_case = pixels * 8
    boundary_accumulators_per_case = 2 * pixels * 8
    max_diag_elements = max(pixels, 8 * nt, 9, 3, 2)
    diagnostic_archive_upper_bound = diagnostic_field_count * max_diag_elements * 16
    masks_bytes = math.ceil(math.prod(shape) / 8) * 5
    itemization = {
        "two_endpoint_E_files_bytes": 2 * e_bytes,
        "two_first_boundary_E_files_bytes": 2 * e_bytes,
        "two_endpoint_POST_archives_bytes": 2 * endpoint_post_per_case,
        "two_first_boundary_POST_archives_bytes": 2 * first_boundary_post_per_case,
        "six_sink_arrays_per_case_bytes": 2 * six_sinks_per_case,
        "endpoint_Qacc_arrays_bytes": 2 * endpoint_qacc_per_case,
        "first_boundary_Qacc_and_Qacc_raman_arrays_bytes": 2 * boundary_accumulators_per_case,
        "endpoint_and_first_boundary_diagnostics_archives_upper_bound_bytes": 4 * diagnostic_archive_upper_bound,
        "persisted_comparison_masks_temporary_bytes": masks_bytes,
        "PROFILE_receipt_max_bytes": MAX_PROFILE_RECEIPT_BYTES,
        "other_logs_receipts_and_package_staging_bytes": OTHER_LOGS_AND_STAGING_BYTES,
    }
    projected_output_bytes = sum(itemization.values())
    reserved_growth_bytes = math.ceil(projected_output_bytes * (1.0 + RESERVE_FRACTION))
    required_cap_bytes = current_bytes + reserved_growth_bytes

    fixed_inventory_files = 2 * 21
    first_boundary_files = 2 * 9
    projected_files = fixed_inventory_files + first_boundary_files + PROJECTED_AUXILIARY_FILE_COUNT
    reserved_files = math.ceil(projected_files * (1.0 + RESERVE_FRACTION))
    checks = {
        "frozen_cap_limit": cap_bytes <= FROZEN_CAP_MAX_BYTES,
        "configured_cap_covers_peak": cap_bytes >= required_cap_bytes,
        "filesystem_bytes_cover_growth": filesystem_available_bytes >= reserved_growth_bytes,
        "quota_bytes_cover_growth": quota_available_bytes >= reserved_growth_bytes,
        "filesystem_inodes_cover_outputs": filesystem_available_inodes >= reserved_files,
        "quota_inodes_cover_outputs": quota_available_inodes >= reserved_files,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED_STORAGE_BUDGET"
    return {
        "schema": "filament.cpu_s4.storage_budget.v1",
        "status": status,
        "source_shape_nt_ny_nx": list(shape),
        "input_count": len(input_records),
        "verified_input_bytes": sum(input_sizes),
        "existing_storage_bytes": current_bytes,
        "projected_output_itemization_bytes": itemization,
        "projected_output_bytes": projected_output_bytes,
        "reserved_growth_bytes": reserved_growth_bytes,
        "reserve_fraction": RESERVE_FRACTION,
        "required_cap_bytes": required_cap_bytes,
        "configured_cap_bytes": cap_bytes,
        "frozen_helper_max_cap_bytes": FROZEN_CAP_MAX_BYTES,
        "diagnostic_field_count": diagnostic_field_count,
        "diagnostic_field_count_upper_bound": MAX_DIAGNOSTIC_FIELDS,
        "diagnostic_archive_upper_bound_bytes": diagnostic_archive_upper_bound,
        "profile_receipt_max_bytes": MAX_PROFILE_RECEIPT_BYTES,
        "projected_output_file_itemization": {
            "fixed_21_item_inventories": fixed_inventory_files,
            "first_boundary_snapshot_files": first_boundary_files,
            "comparison_logs_and_temporary_files_reserve": PROJECTED_AUXILIARY_FILE_COUNT,
        },
        "projected_output_files": projected_files,
        "reserved_inode_count": reserved_files,
        "filesystem_available_bytes": filesystem_available_bytes,
        "filesystem_available_inodes": filesystem_available_inodes,
        "quota_available_bytes": quota_available_bytes,
        "quota_available_inodes": quota_available_inodes,
        "checks": checks,
        "estimate_limitations": [
            "NPZ byte bounds use uncompressed array sizes and include every frozen diagnostic field at a conservative maximum shape and itemsize",
            "PROFILE JSON is rejected by output acceptance above its explicit 512 MiB bound",
            "logical file-size projection, not write-amplification or physical block-allocation measurement",
            "quota values must come from a current site quota query; input assets outside storage_root are reported but not added to required_cap_bytes",
            "comparison-mask estimate covers five full-field persisted masks; actual compressed mask sizes are data-dependent",
        ],
    }
