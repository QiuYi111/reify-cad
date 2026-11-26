"""Reify primary-path CLI wrapping text-to-cad server/renderer/exporter + Reify certify/autosize."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

from reify.ast import Node, Part
from reify.cad_bridge import (
    build_violation_overlay,
    get_certification_tier,
    to_build123d_shape,
)
from reify.finalize import BRepFinalized, verify_brep_clearance
from reify.solve import SolveStatus, solve
from reify.spec import Spec
from reify.types import CertificationTier, Status
from reify.verify import verify


def _cad_bin() -> str:
    candidate = pathlib.Path(sys.executable).with_name("cad")
    if candidate.exists():
        return str(candidate)
    return "cad"


def _forward_cad(args: list[str]) -> int:
    proc = subprocess.run([_cad_bin(), *args])
    return proc.returncode


def _run_cad_quiet(args: list[str]) -> int:
    proc = subprocess.run([_cad_bin(), *args], capture_output=True, text=True)
    return proc.returncode


def _load_script_module(script_path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_reify_user_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_target(mod: Any, target_name: str | None) -> Any:
    if target_name is not None:
        obj = getattr(mod, target_name)
    else:
        # Look for @reify_model first, then build_part / PART / first callable Model
        obj = None
        for attr in dir(mod):
            val = getattr(mod, attr)
            if hasattr(val, "_reify_builder"):
                obj = val
                break
        if obj is None:
            for cand in ("build_part", "part", "PART", "model"):
                if hasattr(mod, cand):
                    obj = getattr(mod, cand)
                    break
        if obj is None:
            raise RuntimeError("No target Reify builder/Part found; pass --target NAME")

    if hasattr(obj, "_reify_builder"):
        return obj._reify_builder()
    if callable(obj) and not isinstance(obj, (Node, Part, BRepFinalized)):
        return obj()
    return obj


def _extract_spec(mod: Any, spec_name: str | None) -> Spec:
    if spec_name is not None:
        obj = getattr(mod, spec_name)
        return obj() if callable(obj) else obj
    for cand in ("build_spec", "spec", "SPEC"):
        if hasattr(mod, cand):
            obj = getattr(mod, cand)
            return obj() if callable(obj) else obj
    return Spec()


def _extract_params_list(mod: Any, var_name: str | None) -> list[Any]:
    if var_name is not None:
        obj = getattr(mod, var_name)
        return list(obj() if callable(obj) else obj)
    for cand in ("build_params", "PARAMS", "params_list"):
        if hasattr(mod, cand):
            obj = getattr(mod, cand)
            return list(obj() if callable(obj) else obj)
    raise RuntimeError("No params_list found; define PARAMS / build_params() or pass --params-list NAME")


def _snapshot_views(doc_id: str, out_dir: pathlib.Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    views = [
        ("iso.png", []),
        ("top.png", ["--camera", "top"]),
        ("front.png", ["--camera", "front"]),
        ("section.png", ["--display", '{"clip":{"enabled":true,"axis":"z","at":0.0}}']),
    ]
    written = []
    for fname, extra in views:
        out_p = out_dir / fname
        rc = _run_cad_quiet(["model", "snapshot", doc_id, str(out_p), *extra])
        if rc == 0 and out_p.exists():
            written.append(str(out_p))
    return written


def _cmd_certify(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="reify certify")
    ap.add_argument("script", type=pathlib.Path)
    ap.add_argument("--target", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--params", default=None, help="JSON dict of parameter overrides")
    ap.add_argument("--snapshot-dir", type=pathlib.Path, default=None)
    ns = ap.parse_args(argv)

    mod = _load_script_module(ns.script.resolve())
    obj = _extract_target(mod, ns.target)
    spec_obj = _extract_spec(mod, ns.spec)
    params = json.loads(ns.params) if ns.params else None
    tier = get_certification_tier(obj)

    if tier == CertificationTier.BREP_ESCAPED:
        if not spec_obj.obstacles:
            rep_dict = {
                "status": Status.PASS.value,
                "certification_tier": tier.value,
                "message": "BREP_ESCAPED terminal object (no obstacles in Spec).",
                "witness_points": [],
            }
        else:
            worst = None
            for rule in spec_obj.obstacles:
                r = verify_brep_clearance(obj, rule.obstacle, rule.min_clearance_mm, params=params)
                worst = r
                if r.status != Status.PASS:
                    break
            assert worst is not None
            rep_dict = {
                "status": worst.status.value,
                "certification_tier": tier.value,
                "reason": worst.reason.value if worst.reason else None,
                "witness_points": worst.witness_points,
                "metrics": worst.metrics,
                "message": worst.message,
            }
        print(json.dumps(rep_dict, indent=2, default=str))
        return 0 if rep_dict["status"] == Status.PASS.value else 1

    rep = verify(obj, spec_obj, params=params)
    snapshots: list[str] = []
    if ns.snapshot_dir is not None:
        from cad.engine.model import model as cad_handover

        ns.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if rep.status != Status.PASS:
            handle = build_violation_overlay(obj, spec_obj, rep, params=params, name=f"{ns.script.stem}_violation")
            ov_path = ns.snapshot_dir / "violation_overlay.png"
            if _run_cad_quiet(["model", "snapshot", handle.id, str(ov_path)]) == 0 and ov_path.exists():
                snapshots.append(str(ov_path))
        else:
            shp = to_build123d_shape(obj, params=params, label=ns.script.stem)
            handle = cad_handover(shp, name=f"{ns.script.stem}_certified")
            snapshots.extend(_snapshot_views(handle.id, ns.snapshot_dir))

    out = {
        "status": rep.status.value,
        "certification_tier": tier.value,
        "reason": rep.reason.value if rep.reason else None,
        "witness_points": rep.witness_points,
        "metrics": rep.metrics,
        "message": rep.message,
        "snapshots": snapshots,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0 if rep.status == Status.PASS else 1


def _cmd_autosize(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="reify autosize")
    ap.add_argument("script", type=pathlib.Path)
    ap.add_argument("--target", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--params-list", default=None)
    ap.add_argument("--snapshot-dir", type=pathlib.Path, default=None)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ns = ap.parse_args(argv)

    mod = _load_script_module(ns.script.resolve())
    obj = _extract_target(mod, ns.target)
    spec_obj = _extract_spec(mod, ns.spec)
    params_list = _extract_params_list(mod, ns.params_list)

    res = solve(
        obj,
        params_list=params_list,
        spec=spec_obj,
        initial_batch_size=ns.batch_size,
        steps_per_start=ns.steps,
    )
    doc_id = None
    view_url = None
    snapshots: list[str] = []
    if res.status == SolveStatus.FEASIBLE_FOUND:
        from cad.engine.model import model as cad_handover

        shp = to_build123d_shape(obj, params=res.params, label=f"{ns.script.stem}_solved")
        handle = cad_handover(shp, name=f"{ns.script.stem}_solved")
        doc_id = handle.id
        view_url = handle.url
        if ns.snapshot_dir is not None:
            snapshots = _snapshot_views(doc_id, ns.snapshot_dir)

    out = {
        "status": res.status.value,
        "certification_tier": get_certification_tier(obj).value,
        "params": res.params,
        "verify_status": res.verify_report.status.value,
        "verify_message": res.verify_report.message,
        "document_id": doc_id,
        "view_url": view_url,
        "snapshots": snapshots,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0 if res.status == SolveStatus.FEASIBLE_FOUND else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(
            "Reify-CAD (Primary Certified Parametric CAD CLI)\n\n"
            "Primary commands:\n"
            "  reify server <start|status|stop>\n"
            "  reify run <script.py>\n"
            "  reify snapshot <id> <out.png> [flags...]\n"
            "  reify inspect <id> [ref]\n"
            "  reify export <id> <out1> [out2...]\n"
            "  reify certify <script.py> [--snapshot-dir DIR]\n"
            "  reify autosize <script.py> [--snapshot-dir DIR]\n\n"
            "Secondary compatibility fallback:\n"
            "  reify build123d <cad-subcommand...>\n"
        )
        return 0

    cmd, rest = args[0], args[1:]
    if cmd == "--version":
        from reify import __version__

        print(f"reify-cad {__version__}")
        return _forward_cad(["--version"])
    if cmd == "server":
        return _forward_cad([cmd, *rest])
    if cmd == "run":
        if not rest:
            return _forward_cad([cmd, *rest])
        proc = subprocess.run([sys.executable, *rest])
        return proc.returncode
    if cmd in ("inspect", "snapshot", "export"):
        return _forward_cad(["model", cmd, *rest])
    if cmd == "certify":
        return _cmd_certify(rest)
    if cmd == "autosize":
        return _cmd_autosize(rest)
    if cmd == "build123d":
        return _forward_cad(rest)

    # Fallback: forward unknown top-level commands to cad
    return _forward_cad([cmd, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
