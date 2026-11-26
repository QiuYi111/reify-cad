"""Bridge between Reify Class 1 / BRepFinalized geometry and text-to-cad build123d/OCCT engine."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

from reify.ast import Node, Part
from reify.brep_compiler import compile_brep
from reify.finalize import BRepFinalized
from reify.types import CertificationTier, Status, VerificationReport

_PALETTE = ("#4a6fa5", "#5b8c5a", "#8e6c88", "#d4a373", "#3d5a80", "#e07a5f")


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


def to_build123d_shape(
    obj: Node | Part | BRepFinalized | dict[str, Node | Part | BRepFinalized],
    *,
    params: dict[str, Any] | None = None,
    label: str | None = None,
    color: str | None = None,
) -> Any:
    """Compile a Reify Node/Part/BRepFinalized (or assembly dict) into a build123d Shape/Compound."""
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
) -> Any:
    """Decorator registering a Reify model function with text-to-cad's @cad.model warm server."""

    def _decorate(fn: Callable[[], Any]) -> Any:
        from cad.engine.model import model as cad_model_decorator

        decorated = cad_model_decorator(fn, name=name) if name is not None else cad_model_decorator(fn)
        decorated._reify_builder = fn
        return decorated

    if _fn is not None:
        return _decorate(_fn)
    return _decorate


def build_violation_overlay(
    part: Node | Part | BRepFinalized,
    spec: Any,
    report: VerificationReport,
    *,
    params: dict[str, Any] | None = None,
    name: str = "violation_overlay",
) -> Any:
    """Create a multi-color build123d Compound highlighting part, obstacles, and red witness spheres and register with server."""
    import build123d as bd
    from cad.engine.model import model as cad_handover

    children = [to_build123d_shape(part, params=params, label="part", color="#4a6fa5")]

    for idx, rule in enumerate(getattr(spec, "obstacles", []) or []):
        obs_shp = to_build123d_shape(rule.obstacle, params=params, label=f"obstacle_{idx}", color="#e67e22")
        children.append(obs_shp)

    for idx, pt in enumerate(getattr(report, "witness_points", []) or []):
        if len(pt) == 3:
            marker = bd.Pos(float(pt[0]), float(pt[1]), float(pt[2])) * bd.Sphere(1.25)
            marker.label = f"witness_{idx}"
            marker.color = "#e74c3c"
            children.append(marker)

    comp = bd.Compound(children=children)
    comp.label = name
    return cad_handover(comp, name=name)
