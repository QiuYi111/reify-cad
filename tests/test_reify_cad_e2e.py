"""End-to-end closed-loop acceptance test for Reify-CAD fork (CLI, Viewer HTTP, snapshot, certify, autosize, export)."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REIFY_BIN = str(pathlib.Path(sys.executable).with_name("reify"))
EXAMPLE = str(ROOT / "examples" / "drafted_enclosure_assembly.py")


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([REIFY_BIN, *args], cwd=str(ROOT), capture_output=True, text=True, check=check)


def test_reify_cad_closed_loop_e2e(tmp_path: pathlib.Path) -> None:
    # 1. Start warm server & verify secondary fallback `reify build123d --version`
    _run(["server", "start"])
    ver = _run(["build123d", "--version"])
    assert ver.returncode == 0

    # 2. `reify run` -> document id + live Web 3D Viewer HTTP 200
    run_res = _run(["run", EXAMPLE])
    m_id = re.search(r"id:\s+([0-9a-f-]+)", run_res.stdout)
    m_url = re.search(r"view:\s+(http://\S+)", run_res.stdout)
    assert m_id and m_url, f"Missing id/view in stdout:\n{run_res.stdout}"
    doc_id = m_id.group(1)
    view_url = m_url.group(1)

    with urllib.request.urlopen(view_url, timeout=10) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8", errors="replace")
        assert "html" in html.lower()

    # 3. `reify certify` at colliding draft_deg=1.0 -> FAIL + violation_overlay.png
    fail_dir = tmp_path / "certify_fail"
    cert_fail = _run(["certify", EXAMPLE, "--snapshot-dir", str(fail_dir)], check=False)
    assert cert_fail.returncode == 1
    fail_data = json.loads(cert_fail.stdout)
    assert fail_data["status"] == "FAIL"
    assert fail_data["certification_tier"] == "CORE_CERTIFIED"
    overlay_png = fail_dir / "violation_overlay.png"
    assert overlay_png.exists() and overlay_png.stat().st_size > 1000

    # 4. `reify autosize` -> FEASIBLE_FOUND, verify PASS, solved draft_deg > 4.0, multi-view PNGs
    solved_dir = tmp_path / "solved"
    auto_res = _run(["autosize", EXAMPLE, "--snapshot-dir", str(solved_dir), "--batch-size", "16", "--steps", "40"])
    auto_data = json.loads(auto_res.stdout)
    assert auto_data["status"] == "FEASIBLE_FOUND"
    assert auto_data["verify_status"] == "PASS"
    assert auto_data["params"]["draft_deg"] > 4.0
    solved_id = auto_data["document_id"]
    assert solved_id
    for name in ("iso.png", "top.png", "front.png", "section.png"):
        p = solved_dir / name
        assert p.exists() and p.stat().st_size > 1000

    # 5. `reify export` -> STEP + GLB + STL
    out_step = tmp_path / "enclosure.step"
    out_glb = tmp_path / "enclosure.glb"
    out_stl = tmp_path / "enclosure.stl"
    _run(["export", solved_id, str(out_step), str(out_glb), str(out_stl)])
    for out_f in (out_step, out_glb, out_stl):
        assert out_f.exists() and out_f.stat().st_size > 500

    # 6. `reify inspect`
    insp = _run(["inspect", doc_id])
    assert "drafted_enclosure" in insp.stdout


def test_reify_cad_pure_sdf_implicit_loop(tmp_path: pathlib.Path) -> None:
    """Verify SmoothUnion (non-B-Rep implicit) works end-to-end in run/snapshot/export .glb/.stl."""
    script = tmp_path / "smooth_part.py"
    script.write_text(
        "from reify import Box, SmoothUnion, Sphere, Transform, mm\n"
        "from reify.cad_bridge import reify_model\n\n"
        "@reify_model(name='smooth_part')\n"
        "def build():\n"
        "    return SmoothUnion(Box(mm(16), mm(16), mm(16)), Transform(Sphere(mm(9)), translation=(mm(8), mm(0), mm(0))), k=mm(3))\n\n"
        "if __name__ == '__main__':\n"
        "    build()\n",
        encoding="utf-8",
    )
    run_res = _run(["run", str(script)])
    m_id = re.search(r"id:\s+([0-9a-f-]+)", run_res.stdout)
    assert m_id, run_res.stdout
    doc_id = m_id.group(1)

    png_out = tmp_path / "smooth.png"
    _run(["snapshot", doc_id, str(png_out)])
    assert png_out.exists() and png_out.stat().st_size > 1000

    glb_out = tmp_path / "smooth.glb"
    stl_out = tmp_path / "smooth.stl"
    _run(["export", doc_id, str(glb_out), str(stl_out)])
    assert glb_out.exists() and glb_out.stat().st_size > 500
    assert stl_out.exists() and stl_out.stat().st_size > 500
