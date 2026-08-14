import math
import re
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "attention_head_biography.md"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_attention_heads.py"


def test_biography_script_execution():
    """Verify that scripts/analyze_attention_heads.py executes successfully with exit code 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert REPORT_PATH.exists()
    assert REPORT_PATH.stat().st_size > 0


def test_biography_report_exists_and_non_empty():
    """Verify reports/attention_head_biography.md exists and is non-empty."""
    assert REPORT_PATH.exists(), f"Missing report at {REPORT_PATH}"
    assert REPORT_PATH.stat().st_size > 0, "Report is empty!"


def test_biography_report_required_sections():
    """Verify report contains all mandatory section headers."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    assert "# Attention Head Biography" in content
    assert "## 1. Methodology" in content
    assert "## 2. Model Configuration" in content
    assert "## 3. Head Statistics" in content
    assert "## 4. Head Biographies" in content
    assert "## 5. Cross-Head Comparison" in content
    assert "## 6. Limitations" in content


def test_biography_report_head_biography_subheadings():
    """Verify head biographies contain mandatory subheadings."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    assert "Observed pattern" in content
    assert "Evidence" in content
    assert "Interpretation" in content
    assert "Hypothesis" in content


def test_biography_report_contains_three_distinct_heads():
    """Verify report contains at least 3 distinct Layer X / Head Y biography sections."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    # Match headings like "### Layer 0 / Head 0"
    matches = re.findall(r"###\s+Layer\s+(\d+)\s+/\s+Head\s+(\d+)", content)
    assert len(matches) >= 3, f"Expected at least 3 head biographies, found {len(matches)}: {matches}"

    unique_heads = set(matches)
    assert len(unique_heads) >= 3, f"Head biographies must be distinct, got {unique_heads}"


def test_biography_report_head_statistics_table():
    """Verify head statistics table lists all 8 attention heads with finite numeric values."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    assert "| Layer | Head | Entropy | Concentration | Self Attention | Previous Token | Next Token | CLS Attention | Avg Distance |" in content

    lines = content.splitlines()
    table_rows = [l for l in lines if l.startswith("| 0 |") or l.startswith("| 1 |")]
    assert len(table_rows) == 8, f"Expected 8 table rows for 8 heads, found {len(table_rows)}"

    for row in table_rows:
        cols = [c.strip() for c in row.split("|")[1:-1]]
        assert len(cols) == 9
        for col in cols:
            val = float(col)
            assert math.isfinite(val)
