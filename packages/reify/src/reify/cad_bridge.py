"""Bridge between Reify pure-SDF pipeline / BRepFinalized and text-to-cad Viewer/Snapshot/Export."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

from reify.ast import Node, Part
from reify.brep_compiler import compile_brep
from reify.finalize import BRepFinalized
from reify.sdf_mesh import export_sdf_glb, extract_sdf_mesh
from reify.types import CertificationTier, Status, VerificationReport

_PALETTE = ("#4a6fa5", "#5b8c5a", "#8e6c88", "#d4a373", "#3d5a80", "#e07a5f")
_CACHE_DIR = pathlib.Path.home() / ".cache" / "reify-cad"
_DOCS_DIR = _CACHE_DIR / "docs"
_REGISTRY_PATH = _CACHE_DIR / "registry.json"


@dataclass
class ReifyDocument:
    id: str
    url: str
    glb_path: pathlib.Path
    name: str
    tier: CertificationTier
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    verts: int
    faces: int
    watertight: bool
    obj: Any


def _cad_bin() -> str:
    candidate = pathlib.Path(sys.executable).with_name("cad")
    if candidate.exists():
        return str(candidate)
    return "cad"


def is_reify_object(obj: Any) -> bool:
    if isinstance(obj, (Node, Part, BRepFinalized)):
        return True
    if isinstance(obj, dict) and obj and all(isinstance(v, (Node, Part, BRepFinalized)) for v in obj.values()):
        return True
    return False


def get_certification_tier(obj: Any) -> CertificationTier:
    if isinstance(obj, BRepFinalized):
        return obj.tier
    if isinstance(obj, dict):
        if any(isinstance(v, BRepFinalized) for v in obj.values()):
            return CertificationTier.BREP_ESCAPED
    return CertificationTier.CORE_CERTIFIED


def _compute_bounds(
    obj: Node | Part | BRepFinalized | dict[str, Any],
    params: dict[str, Any] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if isinstance(obj, dict):
        bmins, bmaxs = [], []
        for v in obj.values():
            mn, mx = _compute_bounds(v, params=params)
            bmins.append(mn)
            bmaxs.append(mx)
        return (
            tuple(min(b[i] for b in bmins) for i in range(3)),  # type: ignore[return-value]
            tuple(max(b[i] for b in bmaxs) for i in range(3)),  # type: ignore[return-value]
        )
    if isinstance(obj, BRepFinalized):
        m = extract_sdf_mesh(obj, split_creases=False)
        if len(m.vertices) == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        b = m.bounds
        return tuple(float(x) for x in b[0]), tuple(float(x) for x in b[1])  # type: ignore[return-value]
    root = obj.root if isinstance(obj, Part) else obj
    return root.local_bounds(params)


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(reg: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def lookup_reify_document(doc_id: str) -> dict[str, Any] | None:
    return _load_registry().get(doc_id)


def _ensure_server_running() -> None:
    cad = _cad_bin()
    st = subprocess.run([cad, "server", "status"], capture_output=True, text=True)
    if st.returncode != 0:
        subprocess.run([cad, "server", "start"], capture_output=True, text=True, check=True)


def register_sdf_document(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    *,
    name: str,
    script_path: str | pathlib.Path | None = None,
    target: str | None = None,
    params: dict[str, Any] | None = None,
    color: str | None = None,
    resolution: int = 96,
) -> ReifyDocument:
    """Extract pure-SDF GLB mesh, register with Web 3D Viewer, and persist Reify metadata."""
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha1(f"{name}:{time.time_ns()}".encode("utf-8")).hexdigest()[:10]
    glb_path = _DOCS_DIR / f"{name}_{stamp}.glb"

    export_sdf_glb(obj, glb_path, params=params, color=color, resolution=resolution)

    # Measure raw mesh stats without crease splitting
    if isinstance(obj, dict):
        verts = 0
        faces = 0
        watertight = True
        for v in obj.values():
            mv = extract_sdf_mesh(v, params=params, resolution=resolution, split_creases=False)
            verts += len(mv.vertices)
            faces += len(mv.faces)
            watertight = watertight and bool(mv.is_watertight)
    else:
        mv = extract_sdf_mesh(obj, params=params, resolution=resolution, split_creases=False)
        verts = len(mv.vertices)
        faces = len(mv.faces)
        watertight = bool(mv.is_watertight)

    _ensure_server_running()
    proc = subprocess.run(
        [_cad_bin(), "viewer", str(glb_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout)
    docs = info.get("documents") or []
    if not docs:
        raise RuntimeError(f"cad viewer returned no document for {glb_path}: {proc.stdout}")
    doc_id = str(docs[0]["id"])
    view_url = str(docs[0].get("url") or info.get("url") or "")

    tier = get_certification_tier(obj)
    bounds = _compute_bounds(obj, params=params)
    reg = _load_registry()
    reg[doc_id] = {
        "name": name,
        "glb": str(glb_path),
        "script": str(pathlib.Path(script_path).resolve()) if script_path else None,
        "target": target,
        "params": params,
        "tier": tier.value,
        "bounds": [list(bounds[0]), list(bounds[1])],
        "verts": verts,
        "faces": faces,
        "watertight": watertight,
    }
    _save_registry(reg)
    return ReifyDocument(
        id=doc_id,
        url=view_url,
        glb_path=glb_path,
        name=name,
        tier=tier,
        bounds=bounds,
        verts=verts,
        faces=faces,
        watertight=watertight,
        obj=obj,
    )


def to_build123d_shape(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    *,
    params: dict[str, Any] | None = None,
    label: str | None = None,
    color: str | None = None,
) -> Any:
    """Compile a Reify Node/Part/BRepFinalized (or assembly dict) into a build123d Shape/Compound (STEP/legacy fallback)."""
    import build123d as bd

    if isinstance(obj, dict):
        children = []
        for idx, (part_name, item) in enumerate(obj.items()):
            c = color or _PALETTE[idx % len(_PALETTE)]
            shp = to_build123d_shape(item, params=params, label=part_name, color=c)
            children.append(shp)
        comp = bd.Compound(children=children)
        if label:
            comp.label = label
        return comp

    if isinstance(obj, BRepFinalized):
        warnings.warn(
            f"[BREP_ESCAPED / IRREVERSIBLE] Rendering terminal B-Rep ({obj.irreversible_reason}).",
            UserWarning,
            stacklevel=2,
        )
        shp = bd.Shape(obj.shape)
        shp.label = label or "finalized_brep"
        if color:
            shp.color = color
        return shp

    root = obj.root if isinstance(obj, Part) else obj
    default_label = obj.name if isinstance(obj, Part) else root.node_id
    st, occt_shp = compile_brep(root, params=params)
    if st != Status.PASS or occt_shp is None:
        raise RuntimeError(f"Failed to compile Reify AST {default_label!r} to OCCT B-Rep (status={st})")
    shp = bd.Shape(occt_shp)
    shp.label = label or default_label
    shp.color = color or _PALETTE[0]
    return shp


def reify_model(
    _fn: Callable[[], Any] | None = None,
    *,
    name: str | None = None,
    color: str | None = None,
    params: dict[str, Any] | None = None,
    resolution: int = 96,
) -> Any:
    """Decorator registering a Reify model via pure-SDF GLB + Web 3D Viewer (zero OCCT in hot path)."""

    def _decorate(fn: Callable[[], Any]) -> Any:
        model_name = name or fn.__name__

        def wrapper() -> ReifyDocument:
            t0 = time.perf_counter()
            obj = fn()
            script_file = getattr(fn, "__code__", None)
            script_path = script_file.co_filename if script_file is not None else None
            doc = register_sdf_document(
                obj,
                name=model_name,
                script_path=script_path,
                target=fn.__name__,
                params=params,
                color=color,
                resolution=resolution,
            )
            dt = time.perf_counter() - t0
            bmin, bmax = doc.bounds
            dx, dy, dz = bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]
            src_name = pathlib.Path(script_path).name if script_path else "<script>"
            print(
                f"built {src_name}::{model_name} (SDF mesh: {doc.verts} verts, {doc.faces} faces, "
                f"watertight={doc.watertight}) in {dt:.2f}s"
            )
            print(f"  bounds {dx:.1f} × {dy:.1f} × {dz:.1f} mm · tier {doc.tier.value}")
            print(f"  id: {doc.id}")
            print(f"  view: {doc.url}")
            return doc

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__doc__ = fn.__doc__
        wrapper._reify_builder = fn  # type: ignore[attr-defined]
        wrapper._reify_name = model_name  # type: ignore[attr-defined]
        return wrapper

    if _fn is not None:
        return _decorate(_fn)
    return _decorate
