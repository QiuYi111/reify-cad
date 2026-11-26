"""Reify primary-path CLI: 100% pure-SDF hot loop + hardened STEP export boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import trimesh

from reify.ast import Node, Part
from reify.brep_compiler import compile_brep_error, export_step
from reify.device import device_info
from reify.cad_bridge import (
    get_certification_tier,
    lookup_reify_document,
    register_sdf_document,
)
from reify.finalize import BRepFinalized, export_finalized_step, verify_brep_clearance
from reify.sdf_mesh import extract_sdf_mesh
from reify.sdf_renderer import render_sdf_snapshot, render_violation_overlay_png
from reify.sdf_webgl import export_sdf_webgl_html
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


def _snapshot_sdf_views(
    obj: Any,
    out_dir: pathlib.Path,
    *,
    params: dict[str, Any] | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    views = [
        ("iso.png", "iso", None),
        ("top.png", "top", None),
        ("front.png", "front", None),
        ("section.png", "iso", {"clip": {"enabled": True, "axis": "z", "at": 0.0}}),
    ]
    written = []
    for fname, cam, disp in views:
        out_p = out_dir / fname
        render_sdf_snapshot(obj, out_p, params=params, camera=cam, display=disp)
        if out_p.exists():
            written.append(str(out_p))
    return written


def _cmd_run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: reify run <script.py>")
        return 0
    script = pathlib.Path(argv[0]).resolve()
    proc = subprocess.run([sys.executable, str(script), *argv[1:]], capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode
    # If script had no __main__ calling @reify_model, register build_part() automatically
    if "id:" not in proc.stdout:
        mod = _load_script_module(script)
        obj = _extract_target(mod, None)
        doc = register_sdf_document(obj, name=script.stem, script_path=script)
        print(f"id: {doc.id}")
        print(f"view: {doc.url}")
    return 0


def _cmd_snapshot(argv: list[str]) -> int:
    if len(argv) >= 2:
        src = argv[0]
        if src.endswith(".py") and pathlib.Path(src).exists():
            ap = argparse.ArgumentParser(prog="reify snapshot", add_help=False)
            ap.add_argument("src", type=pathlib.Path)
            ap.add_argument("out", type=pathlib.Path)
            ap.add_argument("--target", default=None)
            ap.add_argument("--params", default=None)
            ap.add_argument("--camera", default="iso")
            ap.add_argument("--display", default=None)
            ap.add_argument("--width", type=int, default=800)
            ap.add_argument("--height", type=int, default=600)
            ap.add_argument("--device", default=None)
            ns, unknown = ap.parse_known_args(argv)
            if not unknown:
                mod = _load_script_module(ns.src.resolve())
                obj = _extract_target(mod, ns.target)
                params = json.loads(ns.params) if ns.params else None
                out_p = render_sdf_snapshot(
                    obj,
                    ns.out,
                    params=params,
                    camera=ns.camera,
                    display=ns.display,
                    width=ns.width,
                    height=ns.height,
                    device=ns.device,
                )
                print(f"wrote {out_p} ({ns.width}x{ns.height}, SDF sphere-trace)")
                return 0
        if src.endswith(".glb") and pathlib.Path(src).exists():
            return _forward_cad(["glb", "snapshot", *argv])
        entry = lookup_reify_document(src)
        if entry is not None:
            legacy_flags = ("--job", "--annotation", "--measurements", "--pose", "--preview")
            if entry.get("script") and pathlib.Path(entry["script"]).exists() and not any(
                a in legacy_flags for a in argv[2:]
            ):
                ap = argparse.ArgumentParser(prog="reify snapshot", add_help=False)
                ap.add_argument("src")
                ap.add_argument("out", type=pathlib.Path)
                ap.add_argument("--camera", default="iso")
                ap.add_argument("--display", default=None)
                ap.add_argument("--width", type=int, default=800)
                ap.add_argument("--height", type=int, default=600)
                ns, unknown = ap.parse_known_args(argv)
                if not unknown:
                    mod = _load_script_module(pathlib.Path(entry["script"]))
                    obj = _extract_target(mod, entry.get("target"))
                    try:
                        out_p = render_sdf_snapshot(
                            obj,
                            ns.out,
                            params=entry.get("params"),
                            camera=ns.camera,
                            display=ns.display,
                            width=ns.width,
                            height=ns.height,
                        )
                        print(f"wrote {out_p} ({ns.width}x{ns.height}, SDF sphere-trace)")
                        return 0
                    except NotImplementedError:
                        pass
            return _forward_cad(["glb", "snapshot", entry["glb"], *argv[1:]])
    return _forward_cad(["model", "snapshot", *argv])


def _cmd_inspect(argv: list[str]) -> int:
    if argv:
        src = argv[0]
        entry = lookup_reify_document(src)
        if entry is not None:
            print(json.dumps({"id": src, **entry}, indent=2))
            return _forward_cad(["glb", "inspect", entry["glb"]])
        if src.endswith(".glb") and pathlib.Path(src).exists():
            return _forward_cad(["glb", "inspect", *argv])
    return _forward_cad(["model", "inspect", *argv])


def _cmd_export(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: reify export <id> <out1> [out2 ...]", file=sys.stderr)
        return 2
    src, outs = argv[0], argv[1:]
    entry = lookup_reify_document(src)
    if entry is None:
        return _forward_cad(["model", "export", *argv])

    glb_path = pathlib.Path(entry["glb"])
    mesh_outs = [o for o in outs if pathlib.Path(o).suffix.lower() in (".glb", ".stl", ".3mf")]
    step_outs = [o for o in outs if pathlib.Path(o).suffix.lower() in (".step", ".stp")]
    other_outs = [o for o in outs if o not in mesh_outs and o not in step_outs]

    if other_outs:
        print(f"Unsupported Reify export format(s): {other_outs}", file=sys.stderr)
        return 2

    obj: Any = None
    params: dict[str, Any] | None = entry.get("params")
    if mesh_outs:
        for dst in mesh_outs:
            dst_p = pathlib.Path(dst)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            ext = dst_p.suffix.lower()
            if ext == ".glb":
                shutil.copy2(glb_path, dst_p)
                print(f"wrote {dst_p} ({dst_p.stat().st_size} bytes)")
            else:
                # Extract watertight unsplit mesh for STL/3MF when script is available, else cad glb export
                if entry.get("script") and pathlib.Path(entry["script"]).exists():
                    if obj is None:
                        mod = _load_script_module(pathlib.Path(entry["script"]))
                        obj = _extract_target(mod, entry.get("target"))
                    if isinstance(obj, dict):
                        meshes = [extract_sdf_mesh(v, params=params, split_creases=False) for v in obj.values()]
                        combined = trimesh.util.concatenate(meshes)
                        combined.export(str(dst_p))
                    else:
                        m = extract_sdf_mesh(obj, params=params, split_creases=False)
                        m.export(str(dst_p))
                    print(f"wrote {dst_p} ({dst_p.stat().st_size} bytes)")
                else:
                    rc = _forward_cad(["glb", "export", str(glb_path), str(dst_p)])
                    if rc != 0:
                        return rc

    if step_outs:
        if not entry.get("script") or not pathlib.Path(entry["script"]).exists():
            print("STEP export requires source Reify script recorded in document.", file=sys.stderr)
            return 2
        if obj is None:
            mod = _load_script_module(pathlib.Path(entry["script"]))
            obj = _extract_target(mod, entry.get("target"))
        for dst in step_outs:
            dst_p = pathlib.Path(dst)
            if isinstance(obj, BRepFinalized):
                rep = export_finalized_step(obj, dst_p)
            else:
                rep = export_step(obj, dst_p, params=params)
            if rep.status != Status.PASS:
                err_detail = compile_brep_error()
                msg = rep.message + (f" ({err_detail})" if err_detail else "")
                print(f"STEP export failed ({rep.status.value}): {msg}", file=sys.stderr)
                return 1
            print(f"wrote {dst_p} ({dst_p.stat().st_size} bytes, BRepCheck=PASS)")

    return 0


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
        ns.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if rep.status != Status.PASS:
            ov_png = ns.snapshot_dir / "violation_overlay.png"
            render_violation_overlay_png(obj, spec_obj, rep, ov_png, params=params)
            if ov_png.exists():
                snapshots.append(str(ov_png))
        else:
            snapshots.extend(_snapshot_sdf_views(obj, ns.snapshot_dir, params=params))

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

    script_resolved = ns.script.resolve()
    mod = _load_script_module(script_resolved)
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
        doc = register_sdf_document(
            obj,
            name=f"{ns.script.stem}_solved",
            script_path=script_resolved,
            target=ns.target,
            params=res.params,
        )
        doc_id = doc.id
        view_url = doc.url
        if ns.snapshot_dir is not None:
            snapshots = _snapshot_sdf_views(obj, ns.snapshot_dir, params=res.params)

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


def _cmd_viewer_sdf(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="reify viewer-sdf")
    ap.add_argument("script", type=pathlib.Path)
    ap.add_argument("--target", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ns = ap.parse_args(argv)

    script_resolved = ns.script.resolve()
    mod = _load_script_module(script_resolved)
    obj = _extract_target(mod, ns.target)
    params = json.loads(ns.params) if ns.params else None
    out_html = ns.out or pathlib.Path(f"{ns.script.stem}_sdf_viewer.html")
    export_sdf_webgl_html(obj, out_html, params=params, title=f"Reify SDF: {ns.script.stem}")
    print(f"wrote {out_html.resolve()} ({out_html.stat().st_size} bytes, WebGL2 Pure-SDF)")
    return 0


def _cmd_doctor(argv: list[str]) -> int:
    from reify import __version__

    info = {
        "reify_version": __version__,
        "compute_device": device_info(),
    }
    print(json.dumps(info, indent=2))
    cad_args = [a for a in argv if a != "--quiet"]
    return _run_cad_quiet(["doctor", *cad_args]) if "--quiet" in argv else _forward_cad(["doctor", *cad_args])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(
            "Reify-CAD (Pure-SDF Primary Certified Parametric CAD CLI)\n\n"
            "Primary commands (pure-SDF hot loop):\n"
            "  reify server <start|status|stop>\n"
            "  reify doctor\n"
            "  reify run <script.py>\n"
            "  reify snapshot <id|script.py> <out.png> [flags...]\n"
            "  reify inspect <id> [ref]\n"
            "  reify export <id> <out1> [out2...]   (.glb/.stl/.3mf via Manifold DC, .step via verified B-Rep)\n"
            "  reify certify <script.py> [--snapshot-dir DIR]\n"
            "  reify autosize <script.py> [--snapshot-dir DIR]\n"
            "  reify viewer-sdf <script.py> [--out viewer.html]\n\n"
            "Secondary compatibility fallback:\n"
            "  reify build123d <cad-subcommand...>\n"
        )
        return 0

    cmd, rest = args[0], args[1:]
    if cmd == "--version":
        from reify import __version__

        print(f"reify-cad {__version__}")
        return _forward_cad(["--version"])
    if cmd == "doctor":
        return _cmd_doctor(rest)
    if cmd == "server":
        return _forward_cad([cmd, *rest])
    if cmd == "run":
        return _cmd_run(rest)
    if cmd == "snapshot":
        return _cmd_snapshot(rest)
    if cmd == "inspect":
        return _cmd_inspect(rest)
    if cmd == "export":
        return _cmd_export(rest)
    if cmd == "certify":
        return _cmd_certify(rest)
    if cmd == "autosize":
        return _cmd_autosize(rest)
    if cmd == "viewer-sdf":
        return _cmd_viewer_sdf(rest)
    if cmd == "build123d":
        return _forward_cad(rest)

    # Fallback: forward unknown top-level commands to cad
    return _forward_cad([cmd, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
