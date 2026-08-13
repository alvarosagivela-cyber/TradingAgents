"""Unit test for Phase 7 Windows Task Scheduler setup script.

This test verifies the content of scripts/setup_phase7_scheduler.ps1 without
executing it (PowerShell execution in CI is unreliable). The test reads the
script as text and verifies it contains the required PowerShell cmdlets and
documentation.

This proves (D-02) that the setup artifact exists, is documented as ONE-TIME
and human-run, and contains the correct Task Scheduler registration logic.
"""

from __future__ import annotations

import pathlib

import pytest


@pytest.mark.unit
def test_setup_script_exists():
    """Test that the setup script file exists."""
    script_path = pathlib.Path(__file__).parent.parent / "scripts" / "setup_phase7_scheduler.ps1"
    assert script_path.exists(), f"Setup script not found at {script_path}"


@pytest.mark.unit
def test_setup_script_contains_required_elements():
    """Test that setup script contains all required PowerShell elements and docs.

    Verifies (D-02 scope boundary):
    - ONE-TIME / one-time is documented (scope boundary)
    - Register-ScheduledTask is present (registration logic)
    - New-ScheduledTaskTrigger is present (daily trigger)
    - New-ScheduledTaskAction is present (action definition)
    - phase7_daily_runner.py is referenced (correct entry point)
    - Unregister-ScheduledTask is present (removal instructions, proving the
      script does not leave users with no way to clean up)
    """
    script_path = pathlib.Path(__file__).parent.parent / "scripts" / "setup_phase7_scheduler.ps1"
    content = script_path.read_text(encoding="utf-8")

    required_elements = {
        "ONE-TIME": "Documentation of one-time, manual execution",
        "Register-ScheduledTask": "PowerShell cmdlet to register the task",
        "New-ScheduledTaskTrigger": "PowerShell cmdlet to create daily trigger",
        "New-ScheduledTaskAction": "PowerShell cmdlet to define the action",
        "phase7_daily_runner.py": "Reference to the daily runner script",
        "Unregister-ScheduledTask": "Instructions for task removal/cleanup",
    }

    for element, description in required_elements.items():
        assert element in content, (
            f"Required element '{element}' ({description}) not found in setup script"
        )


@pytest.mark.unit
def test_setup_script_documents_human_run():
    """Test that the setup script explicitly documents it is human-run, not auto-invoked.

    Scope boundary verification: the script must be clear that it is a one-time
    manual setup step, never invoked automatically.
    """
    script_path = pathlib.Path(__file__).parent.parent / "scripts" / "setup_phase7_scheduler.ps1"
    content = script_path.read_text(encoding="utf-8")

    # Should document that this is one-time and human-run
    scope_keywords = [
        "ONE-TIME",
        "HUMAN-RUN",
        "manually",
    ]

    found_keywords = [kw for kw in scope_keywords if kw in content]
    assert len(found_keywords) >= 2, (
        f"Setup script should document one-time, human-run execution. "
        f"Found only: {found_keywords}"
    )


@pytest.mark.unit
def test_setup_script_survives_missed_trigger():
    """Test that the task settings let a missed trigger (PC off/asleep/on battery
    at 16:05) catch up instead of silently dropping the whole day.

    Without StartWhenAvailable, Task Scheduler's default is to skip a missed
    trigger entirely -- not even a log entry, since phase7_daily_runner.py never
    starts. D-08's retry-then-skip only guards failures inside a running process,
    it does nothing if the process never launches.
    """
    script_path = pathlib.Path(__file__).parent.parent / "scripts" / "setup_phase7_scheduler.ps1"
    content = script_path.read_text(encoding="utf-8")

    required_elements = {
        "New-ScheduledTaskSettingsSet": "settings object construction",
        "-StartWhenAvailable": "catch up a missed trigger once the PC is back on",
        "-AllowStartIfOnBatteries": "run on laptop battery power, not only plugged in",
    }

    for element, description in required_elements.items():
        assert element in content, (
            f"Required element '{element}' ({description}) not found in setup script"
        )
