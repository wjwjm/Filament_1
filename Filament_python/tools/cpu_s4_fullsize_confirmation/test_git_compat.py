"""The compatibility query returns actual system Git identity and refuses writes."""
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from tools.cpu_s4_checkpoint_support.git_readonly_compat import query

def test_full_entry_git_compat_actual_head_and_version(capfd):
    system=shutil.which("git");assert system
    root=Path(__file__).resolve().parents[3]
    expected=subprocess.run([system,"-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout
    assert query(system,["-C",str(root),"rev-parse","HEAD"])==0
    assert capfd.readouterr().out==expected
    expected=subprocess.run([system,"--version"],capture_output=True,text=True,check=True).stdout
    assert query(system,["--version"])==0
    assert capfd.readouterr().out==expected

def test_full_entry_git_compat_refuses_management():
    system=shutil.which("git");assert system
    for args in (["commit"],["push"],["symbolic-ref","HEAD","refs/heads/other"]):
        with pytest.raises(RuntimeError,match="READONLY"):
            query(system,args)
