"""Fail-closed cgroup memory limit discovery for the S4 supervisor.

This module intentionally reads the target process' own cgroup membership and
mount namespace. It does not alter or depend on the frozen Q1 memory helper.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import posixpath
import re
import sys


class CgroupMemoryError(RuntimeError):
    """The effective memory limit could not be established safely."""


_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")
_V1_UNLIMITED = {sys.maxsize, (sys.maxsize // 4096) * 4096}



def _decode_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _safe_cgroup_path(value: str) -> str:
    if not value.startswith("/") or "\x00" in value:
        raise CgroupMemoryError("invalid_cgroup_membership_path")
    parts = value.split("/")
    if ".." in parts:
        raise CgroupMemoryError("unsafe_cgroup_membership_path")
    return posixpath.normpath(value)


def _read_membership(path: Path) -> tuple[str | None, str | None]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CgroupMemoryError(f"membership_unreadable:{type(exc).__name__}") from exc
    v2 = []
    v1 = []
    for line in lines:
        pieces = line.split(":", 2)
        if len(pieces) != 3:
            raise CgroupMemoryError("malformed_cgroup_membership")
        hierarchy, controllers, group = pieces
        if not hierarchy.isdigit():
            raise CgroupMemoryError("malformed_cgroup_hierarchy_id")
        group = _safe_cgroup_path(group)
        if hierarchy == "0" and not controllers:
            v2.append(group)
        elif controllers:
            v1.append((set(controllers.split(",")), group))
    if len(v2) > 1 or sum("memory" in names for names, _ in v1) > 1:
        raise CgroupMemoryError("ambiguous_cgroup_membership")
    v1_memory = next((group for names, group in v1 if "memory" in names), None)
    return (v2[0] if v2 else None), v1_memory


def _read_mounts(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CgroupMemoryError(f"mountinfo_unreadable:{type(exc).__name__}") from exc
    mounts = []
    for line in lines:
        try:
            before, after = line.split(" - ", 1)
            left = before.split()
            right = after.split()
            if len(left) < 6 or len(right) < 3:
                raise ValueError("short mountinfo row")
            mounts.append({
                "fs_type": right[0],
                "root": _safe_cgroup_path(_decode_mount_field(left[3])),
                "mountpoint": _safe_cgroup_path(_decode_mount_field(left[4])),
                "options": set((left[5] + "," + right[2] + "," + right[1]).split(",")),
            })
        except (ValueError, IndexError) as exc:
            raise CgroupMemoryError("malformed_mountinfo") from exc
    return mounts


def _virtual_path(value: str, virtual_fs_root: Path | None) -> Path:
    if virtual_fs_root is None:
        return Path(value)
    return virtual_fs_root.joinpath(*[part for part in value.split("/") if part])


def _covers(mount: dict, membership: str) -> bool:
    root = mount["root"].rstrip("/") or "/"
    return root == "/" or membership == root or membership.startswith(root + "/")


def _choose_mount(mounts: list[dict], membership: str, *, version: int) -> dict:
    wanted = "cgroup2" if version == 2 else "cgroup"
    candidates = [row for row in mounts
                  if row["fs_type"] == wanted and _covers(row, membership)
                  and (version == 2 or "memory" in row["options"])]
    if not candidates:
        raise CgroupMemoryError(f"cgroup_v{version}_mount_not_visible")
    most_specific = max(len(row["root"]) for row in candidates)
    selected = [row for row in candidates if len(row["root"]) == most_specific]
    if len(selected) != 1:
        raise CgroupMemoryError(f"ambiguous_cgroup_v{version}_mount")
    mount = selected[0]
    # The mount-root's own ancestors are hidden from this process. Their limits
    # may still constrain the allocation, so a partial hierarchy is not proof.
    if mount["root"] != "/":
        raise CgroupMemoryError("hidden_cgroup_ancestors_not_visible")
    return mount


def _group_paths(mount: dict, membership: str,
                 virtual_fs_root: Path | None) -> list[tuple[str, Path]]:
    if not _covers(mount, membership):
        raise CgroupMemoryError("membership_outside_cgroup_mount")
    root = mount["root"].rstrip("/") or "/"
    relative = membership.lstrip("/") if root == "/" else membership[len(root):].lstrip("/")
    leaf = posixpath.normpath(posixpath.join(mount["mountpoint"], relative))
    mountpoint = mount["mountpoint"].rstrip("/") or "/"
    if leaf != mountpoint and not leaf.startswith(mountpoint + "/"):
        raise CgroupMemoryError("resolved_cgroup_path_escaped_mount")
    result = []
    current = leaf
    while True:
        result.append((current, _virtual_path(current, virtual_fs_root)))
        if current == mountpoint:
            break
        parent = posixpath.dirname(current)
        if parent == current or not parent.startswith(mountpoint.rstrip("/") + "/"):
            if parent != mountpoint:
                raise CgroupMemoryError("cgroup_parent_chain_incomplete")
        current = parent
    return result


def _read_nonnegative(path: Path) -> int:
    try:
        raw = path.read_text(encoding="ascii").strip()
        if not raw or not raw.isdecimal():
            raise ValueError("not a nonnegative integer")
        return int(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CgroupMemoryError(f"invalid_or_unreadable_counter:{path.name}") from exc


def _v1_limit(path: Path) -> int | None:
    value = _read_nonnegative(path)
    if value <= 0: raise CgroupMemoryError("nonpositive_v1_limit")
    return None if value in _V1_UNLIMITED else value


def _read_v2(mount: dict, membership: str,
             virtual_fs_root: Path | None) -> tuple[list[dict], list[dict]]:
    paths = _group_paths(mount, membership, virtual_fs_root)
    root_files = paths[-1][1]
    try:
        controllers = set((root_files / "cgroup.controllers").read_text(encoding="ascii").split())
    except (OSError, UnicodeError) as exc:
        raise CgroupMemoryError("v2_controller_visibility_unknown") from exc
    if "memory" not in controllers:
        raise CgroupMemoryError("v2_memory_controller_not_available")
    levels = []
    limits = []
    for name, directory in paths:
        if directory == root_files:
            # A real v2 hierarchy root has no memory.max/high knobs. If they
            # exist here, this may be a namespaced subtree with hidden parents.
            if (directory / 'memory.max').exists() or (directory / 'memory.high').exists():
                raise CgroupMemoryError('v2_root_may_hide_ancestors')
            levels.append({'path': name, 'hard_limit_bytes': None,
                'memory_high_bytes': None, 'current_bytes': None,
                'hierarchical_accounting': True, 'limit_applies_to_target': False,
                'root_semantics': 'unconstrained_hierarchy_root'})
            continue
        try:
            raw_limit = (directory / "memory.max").read_text(encoding="ascii").strip()
            high = (directory / "memory.high").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise CgroupMemoryError(f"v2_memory_limit_unreadable:{name}") from exc
        limit = None if raw_limit == "max" else _parse_limit(raw_limit, "v2 memory.max")
        high_value = None if high == "max" else _parse_limit(high, "v2 memory.high")
        current = _read_nonnegative(directory / "memory.current")
        level = {"path": name, "hard_limit_bytes": limit, "memory_high_bytes": high_value,
                 "current_bytes": current, "hierarchical_accounting": True,
                 "limit_applies_to_target": limit is not None}
        levels.append(level)
        if limit is not None:
            limits.append({"path": name, "limit_bytes": limit, "current_bytes": current,
                           "version": 2})
    return levels, limits


def _parse_limit(raw: str, label: str) -> int:
    if not raw or not raw.isdecimal():
        raise CgroupMemoryError(f"invalid_{label.replace(' ', '_')}")
    value = int(raw)
    if value <= 0:
        raise CgroupMemoryError(f"nonpositive_{label.replace(' ', '_')}")
    return value


def _read_v1(mount: dict, membership: str,
             virtual_fs_root: Path | None) -> tuple[list[dict], list[dict], int | None]:
    paths = _group_paths(mount, membership, virtual_fs_root)
    levels = []
    limits = []
    for index, (name, directory) in enumerate(paths):
        limit = _v1_limit(directory / "memory.limit_in_bytes")
        current = _read_nonnegative(directory / "memory.usage_in_bytes")
        use_hierarchy = _read_nonnegative(directory / "memory.use_hierarchy")
        if use_hierarchy not in (0, 1):
            raise CgroupMemoryError(f"invalid_v1_hierarchy_flag:{name}")
        # The leaf limit is local. An ancestor limit constrains descendants
        # only when that ancestor charges usage hierarchically.
        applies = index == 0 or use_hierarchy == 1
        level = {"path": name, "hard_limit_bytes": limit,
                 "memory_high_bytes": None, "current_bytes": current,
                 "use_hierarchy": use_hierarchy,
                 "hierarchical_accounting": bool(use_hierarchy),
                 "limit_applies_to_target": bool(applies and limit is not None)}
        levels.append(level)
        if applies and limit is not None:
            limits.append({"path": name, "limit_bytes": limit,
                           "current_bytes": current, "version": 1})
    # v1's kernel-reported hierarchical limit is an independent check that
    # the visible mount walk found every tighter ancestor.
    leaf = paths[0][1]
    try:
        stat_lines = (leaf / "memory.stat").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CgroupMemoryError("v1_memory_stat_unreadable") from exc
    stat_values = {}
    for line in stat_lines:
        columns = line.split()
        if len(columns) == 2 and columns[0] == "hierarchical_memory_limit":
            stat_values[columns[0]] = columns[1]
    if "hierarchical_memory_limit" not in stat_values:
        raise CgroupMemoryError("v1_hierarchical_limit_missing")
    raw = stat_values["hierarchical_memory_limit"]
    if not raw.isdecimal():
        raise CgroupMemoryError("v1_hierarchical_limit_malformed")
    hierarchy_limit = None if int(raw) in _V1_UNLIMITED else int(raw)
    known = min((row["limit_bytes"] for row in limits), default=None)
    if hierarchy_limit is not None and (known is None or hierarchy_limit < known):
        raise CgroupMemoryError("unlocated_v1_hierarchical_limit")
    return levels, limits, hierarchy_limit


def snapshot_cgroup_memory(pid: int, *, allocation_bytes: int,
                           proc_root: str | os.PathLike = "/proc",
                           virtual_fs_root: str | os.PathLike | None = None) -> dict:
    """Read all visible effective cgroup constraints for a process.

    `virtual_fs_root` exists only to support isolated Linux-layout fixtures on
    non-Linux developer hosts. Production leaves it unset.
    """
    if not sys.platform.startswith("linux") and str(proc_root) == "/proc":
        return {"status": "NOT_LINUX", "reason": "linux_cgroup_qualification_not_available"}
    if not isinstance(allocation_bytes, int) or allocation_bytes <= 0:
        raise CgroupMemoryError("invalid_allocation_memory")
    proc = Path(proc_root) / str(int(pid))
    v2_group, v1_group = _read_membership(proc / "cgroup")
    mounts = _read_mounts(proc / "mountinfo")
    fs_root = Path(virtual_fs_root) if virtual_fs_root is not None else None
    failures = []
    if v2_group is not None:
        try:
            mount = _choose_mount(mounts, v2_group, version=2)
            levels, limits = _read_v2(mount, v2_group, fs_root)
            version = 2
            hierarchy_limit = None
        except CgroupMemoryError as exc:
            # Only an explicitly absent v2 memory controller permits the v1
            # hybrid route; never mask a malformed/hidden v2 hierarchy.
            if str(exc) != 'v2_memory_controller_not_available' or v1_group is None:
                raise
            failures.append(str(exc))
            mount = None
    else:
        mount = None
    # Hybrid kernels can expose a unified hierarchy without the memory
    # controller and a separate v1 memory mount. Use v1 only in that case.
    if mount is None and v1_group is not None:
        try:
            mount = _choose_mount(mounts, v1_group, version=1)
            levels, limits, hierarchy_limit = _read_v1(mount, v1_group, fs_root)
            version = 1
        except CgroupMemoryError as exc:
            failures.append(str(exc))
            mount = None
    if mount is None:
        raise CgroupMemoryError(";".join(failures) or "memory_cgroup_not_visible")
    cgroup_limit = min((row["limit_bytes"] for row in limits), default=None)
    candidates = [allocation_bytes]
    if cgroup_limit is not None:
        candidates.append(cgroup_limit)
    effective = min(candidates)
    return {"status": "READ", "version": version,
            "mount_root": mount["root"], "mountpoint": mount["mountpoint"],
            "membership_path": v2_group if version == 2 else v1_group,
            "allocation_limit_bytes": allocation_bytes,
            "cgroup_limit_bytes": cgroup_limit,
            "effective_limit_bytes": effective,
            "effective_limit_source": ("allocation" if effective == allocation_bytes
                and (cgroup_limit is None or allocation_bytes <= cgroup_limit)
                else min(limits, key=lambda row: row["limit_bytes"])["path"]),
            "hierarchical_memory_limit_bytes": hierarchy_limit,
            "levels_leaf_to_mount_root": levels,
            "enforced_cgroup_limits": limits,
            "memory_high_is_advisory_only": True}


def memory_fractions(*, rss_bytes: int, pss_bytes: int | None,
                     cgroup: dict, allocation_bytes: int) -> dict:
    if cgroup.get("status") != "READ":
        return {"tree_rss_bytes": rss_bytes, "tree_pss_bytes": pss_bytes,
                "max_enforced_fraction": None, "cgroup": cgroup,
                "status": "BLOCKED_CGROUP_MEMORY_UNKNOWN", "pss_is_report_only": True}
    effective = int(cgroup["effective_limit_bytes"])
    fractions = [{"source": "process_tree_rss/effective_limit",
                  "fraction": rss_bytes / effective}]
    fractions.extend({"source": row["path"], "fraction": row["current_bytes"] / row["limit_bytes"]}
                     for row in cgroup["enforced_cgroup_limits"])
    if any(not math.isfinite(item["fraction"]) or item["fraction"] < 0 for item in fractions):
        raise CgroupMemoryError("invalid_memory_fraction")
    return {"tree_rss_bytes": rss_bytes, "tree_pss_bytes": pss_bytes,
            "allocation_fraction_rss": rss_bytes / allocation_bytes,
            "effective_limit_bytes": effective,
            "fractions_by_enforcer": fractions,
            "max_enforced_fraction": max(item["fraction"] for item in fractions),
            "cgroup": cgroup, "status": "READ", "pss_is_report_only": True}
