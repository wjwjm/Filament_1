"""Read-only CLI compatibility for real system Git predating the -C option.

Management (clone/fetch/commit/push) always calls system Git directly. This
adapter only converts query invocation syntax; every byte of query output,
including HEAD and --version, is produced by the configured real executable.
"""
import argparse
import os
import shlex
import shutil
from pathlib import Path
import subprocess

def launcher_source(python, helper, system_git):
    return "#!/bin/sh\nexec "+shlex.quote(str(python))+" -s -B "+shlex.quote(str(helper))+" --system-git "+shlex.quote(str(system_git))+" -- \"$@\"\n"

def verify_runtime_query_tool(plan):
    """Bind all inherited bare-git queries to the exact real-Git launcher."""
    system=Path(plan["system_git"]).resolve(strict=True)
    launcher=Path(plan["query_git_launcher"]).resolve(strict=True)
    helper=Path(__file__).resolve(strict=True)
    found=shutil.which("git")
    if found is None or Path(found).resolve()!=launcher or system==launcher:
        raise RuntimeError("FIXED_RUNTIME_GIT_LAUNCHER_NOT_SELECTED")
    expected=launcher_source(plan["python"],helper,system).encode("utf-8")
    if launcher.read_bytes()!=expected or not os.access(launcher,os.X_OK):
        raise RuntimeError("FIXED_RUNTIME_GIT_LAUNCHER_BYTES")
    direct=subprocess.run([str(system),"--version"],capture_output=True,check=True)
    wrapped=subprocess.run([str(launcher),"--version"],capture_output=True,check=True)
    actual=subprocess.run([str(system),"rev-parse","HEAD"],cwd=plan["source_root"],capture_output=True,text=True,check=True)
    through=subprocess.run([str(launcher),"-C",plan["source_root"],"rev-parse","HEAD"],capture_output=True,text=True,check=True)
    if direct.stdout!=wrapped.stdout or actual.stdout!=through.stdout or actual.stdout.strip()!=plan["execution_commit"]:
        raise RuntimeError("SYSTEM_GIT_IDENTITY_NOT_PRESERVED")
    return dict(system_git=str(system),query_launcher=str(launcher),version=direct.stdout.decode().strip(),actual_head=actual.stdout.strip(),management_uses_system_git_directly=True)

def query(system_git, arguments):
    binary=Path(system_git).resolve(strict=True)
    args=list(arguments);cwd=None
    if args[:1]==["-C"]:
        if len(args)<3:raise RuntimeError("GIT_QUERY_CWD_MISSING")
        cwd=str(Path(args[1]).resolve(strict=True));args=args[2:]
    if args[:3]==["remote","get-url","origin"] and len(args)==3:
        args=["config","--get","remote.origin.url"]
    args=["--porcelain" if arg=="--porcelain=v1" else arg for arg in args]
    allowed={"--version","rev-parse","cat-file","merge-base","ls-files","ls-tree","diff","diff-index","status","show","rev-list","symbolic-ref","config"}
    if not args or args[0] not in allowed:raise RuntimeError("READONLY_GIT_QUERY_ONLY")
    if args[0]=="config" and args!=["config","--get","remote.origin.url"]:raise RuntimeError("READONLY_GIT_CONFIG_QUERY_ONLY")
    if args[0]=="symbolic-ref" and len([a for a in args[1:] if not a.startswith("-")])!=1:raise RuntimeError("READONLY_SYMBOLIC_REF_LOOKUP_ONLY")
    if any(a=="--output" or a.startswith("--output=") for a in args):raise RuntimeError("NO_GIT_QUERY_OUTPUT_FILE")
    env=os.environ.copy();env["GIT_OPTIONAL_LOCKS"]="0"
    return subprocess.run([str(binary),*args],cwd=cwd,env=env,check=False).returncode

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-git",required=True)
    parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    argv=args.arguments[1:] if args.arguments[:1]==["--"] else args.arguments
    return query(args.system_git,argv)

if __name__=="__main__":raise SystemExit(main())
