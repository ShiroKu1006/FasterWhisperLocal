from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_POWERSHELL = shutil.which("powershell.exe")


@pytest.mark.skipif(
    WINDOWS_POWERSHELL is None,
    reason="Windows PowerShell 5.1 is only available on Windows",
)
def test_verify_install_runs_under_windows_powershell() -> None:
    completed = subprocess.run(
        [
            WINDOWS_POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "verify_install.ps1"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "large-v3 model: present" in completed.stdout
