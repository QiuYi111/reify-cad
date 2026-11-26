"""Open-chain R/P kinematics, two-level conservative advancement CCD, and tool envelopes (M2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import fcl
import numpy as np
import torch
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from reify.ast import (
    Box,
    Cylinder,
    JointGraph,
    Node,
    PrismaticJoint,
    RevoluteJoint,
    Transform,
    Union,
)
from reify.brep_compiler import compile_brep
from reify.query import GeometryQuery
from reify.types import FailReason, Status, VerificationReport
from reify.units import deg, mm
from reify.verify import _RigidMatrixNode, check_clearance


class MotionPath(Protocol):
    """Explicitly parameterized joint trajectory q(s), s in [0, 1]."""

    def eval(self, s: float) -> dict[str, float]: ...

    def max_speed_on_interval(self, joint_name: str, s0: float, s1: float) -> float: ...


@dataclass(frozen=True)
class LinearMotionPath:
    """Linear joint trajectory over s in [0, 1]."""

    q_start: dict[str, float]
    q_end: dict[str, float]

    def eval(self, s: float) -> dict[str, float]:
        s_clamped = min(max(float(s), 0.0), 1.0)
        return {
            k: (1.0 - s_clamped) * self.q_start[k] + s_clamped * self.q_end[k]
            for k in self.q_start
        }

    def max_speed_on_interval(self, joint_name: str, s0: float, s1: float) -> float:
        return abs(self.q_end.get(joint_name, 0.0) - self.q_start.get(joint_name, 0.0))


@dataclass(frozen=True)
class PiecewiseLinearPath:
    """Piecewise-linear joint trajectory over s in [0, 1]."""

    waypoints: list[tuple[float, dict[str, float]]]

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("PiecewiseLinearPath requires at least 2 waypoints")

    def eval(self, s: float) -> dict[str, float]:
        s_c = min(max(float(s), 0.0), 1.0)
        for i in range(len(self.waypoints) - 1):
            s0, q0 = self.waypoints[i]
            s1, q1 = self.waypoints[i + 1]
            if s_c <= s1 or i == len(self.waypoints) - 2:
                t = 0.0 if abs(s1 - s0) < 1e-12 else (s_c - s0) / (s1 - s0)
                t = min(max(t, 0.0), 1.0)
                return {k: (1.0 - t) * q0[k] + t * q1[k] for k in q0}
        return dict(self.waypoints[-1][1])

    def max_speed_on_interval(self, joint_name: str, s0: float, s1: float) -> float:
        max_v = 0.0
        for i in range(len(self.waypoints) - 1):
            a, qa = self.waypoints[i]
            b, qb = self.waypoints[i + 1]
            if b < s0 or a > s1:
                continue
            ds = max(b - a, 1e-12)
            v = abs(qb.get(joint_name, 0.0) - qa.get(joint_name, 0.0)) / ds
            max_v = max(max_v, v)
        return max_v


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


class KinematicChain:
    """Open-chain forward kinematics and conservative R/P velocity bound."""

    def __init__(self, joint_graph: JointGraph, link_shapes: dict[str, Node]) -> None:
        self.joint_graph = joint_graph
        self.link_shapes = link_shapes
        self._parent_joint: dict[str, RevoluteJoint | PrismaticJoint] = {
            j.child_link: j for j in joint_graph.joints
        }

    def check_domain(self) -> Status:
        return self.joint_graph.check_domain()

    def ancestor_joints(self, link_name: str) -> list[RevoluteJoint | PrismaticJoint]:
        chain: list[RevoluteJoint | PrismaticJoint] = []
        cur = link_name
        seen: set[str] = set()
        while cur != self.joint_graph.root_link and cur in self._parent_joint:
            if cur in seen:
                break
            seen.add(cur)
            j = self._parent_joint[cur]
            chain.append(j)
            cur = j.parent_link
        chain.reverse()
        return chain

    def link_world_pose(
        self, link_name: str, q_dict: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        for j in self.ancestor_joints(link_name):
            q_val = float(q_dict.get(j.name, 0.0))
            if isinstance(j, RevoluteJoint):
                ax = np.array(j.axis_dir, dtype=np.float64)
                orig = np.array(j.axis_origin_mm, dtype=np.float64)
                R_j = _rodrigues(ax, q_val)
                # Rotation about orig in current frame: x_new = R @ (R_j @ (x - orig) + orig) + t
                t = t + R @ (orig - R_j @ orig)
                R = R @ R_j
            elif isinstance(j, PrismaticJoint):
                ax = np.array(j.axis_dir, dtype=np.float64)
                ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
                t = t + R @ (q_val * ax)
        return R, t

    def max_radius_from_joint(
        self,
        joint: RevoluteJoint,
        moving_link: str,
        path: MotionPath,
        s0: float,
        s1: float,
    ) -> float:
        """Conservative upper bound on perpendicular distance from joint axis to moving link."""
        shape = self.link_shapes[moving_link]
        bmin, bmax = shape.local_bounds()
        corners = np.array(
            [
                [x, y, z]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=np.float64,
        )
        # Bound downstream link extent relative to joint by summing link corner norm + downstream prismatic/origin offsets
        joints = self.ancestor_joints(moving_link)
        idx = joints.index(joint)
        downstream = joints[idx + 1 :]
        downstream_offset = 0.0
        for dj in downstream:
            if isinstance(dj, PrismaticJoint):
                downstream_offset += max(abs(dj.q_min_mm), abs(dj.q_max_mm))
            elif isinstance(dj, RevoluteJoint):
                downstream_offset += float(np.linalg.norm(dj.axis_origin_mm))

        orig = np.array(joint.axis_origin_mm, dtype=np.float64)
        ax = np.array(joint.axis_dir, dtype=np.float64)
        ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
        if not downstream:
            diffs = corners - orig
            perp = diffs - np.outer(diffs @ ax, ax)
            return float(np.max(np.linalg.norm(perp, axis=-1)))
        max_corner_norm = float(np.max(np.linalg.norm(corners, axis=-1)))
        return float(np.linalg.norm(orig)) + max_corner_norm + downstream_offset

    def interval_velocity_upper(
        self,
        moving_link: str,
        path: MotionPath,
        s0: float,
        s1: float,
    ) -> float:
        v_upper = 0.0
        for j in self.ancestor_joints(moving_link):
            speed = path.max_speed_on_interval(j.name, s0, s1)
            if isinstance(j, RevoluteJoint):
                r_max = self.max_radius_from_joint(j, moving_link, path, s0, s1)
                v_upper += speed * r_max
            elif isinstance(j, PrismaticJoint):
                v_upper += speed
        return v_upper


def _brep_to_fcl_bvh(shape: Any, deflection: float = 0.02) -> fcl.BVHModel | None:
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.2, True)
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    offset = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append([p.X(), p.Y(), p.Z()])
            n_tris = tri.NbTriangles()
            for t_i in range(1, n_tris + 1):
                i1, i2, i3 = tri.Triangle(t_i).Get()
                tris.append([offset + i1 - 1, offset + i2 - 1, offset + i3 - 1])
            offset += n_nodes
        exp.Next()
    if not verts or not tris:
        return None
    bvh = fcl.BVHModel()
    bvh.beginModel(len(verts), len(tris))
    bvh.addSubModel(np.array(verts, dtype=np.float64), np.array(tris, dtype=np.int32))
    bvh.endModel()
    return bvh


def check_kinematic_sweep(
    chain: KinematicChain,
    moving_link: str,
    obstacle_node: Node,
    path: MotionPath,
    min_clearance_mm: float = 0.0,
    eps_contact: float = 1e-3,
    max_iters: int = 2000,
    min_step: float = 1e-4,
    use_fcl: bool = True,
) -> VerificationReport:
    """Two-level conservative advancement + FCL CCD along explicit joint path q(s), s in [0, 1]."""
    if chain.check_domain() == Status.OUT_OF_DOMAIN:
        return VerificationReport(
            status=Status.OUT_OF_DOMAIN,
            message="Closed-loop kinematic graph rejected (OUT_OF_DOMAIN).",
        )

    link_node = chain.link_shapes[moving_link]
    q_obs = GeometryQuery(obstacle_node)

    def _pose_node(s_val: float) -> _RigidMatrixNode:
        q_d = path.eval(s_val)
        R_np, t_np = chain.link_world_pose(moving_link, q_d)
        return _RigidMatrixNode(
            link_node,
            torch.from_numpy(R_np).to(torch.float64),
            torch.from_numpy(t_np).to(torch.float64),
        )

    # 1. Pre-sweep non-penetration & non-containment check at s = 0
    rep0 = check_clearance(
        GeometryQuery(_pose_node(0.0)),
        q_obs,
        min_clearance_mm=min_clearance_mm,
        max_depth=3,
        initial_radius=0.5,
    )
    if rep0.status == Status.FAIL:
        return VerificationReport(
            status=Status.FAIL,
            reason=FailReason.KINEMATIC_COLLISION,
            witness_points=rep0.witness_points,
            metrics={"collision_s": 0.0, "detail": rep0.reason.value if rep0.reason else ""},
            message="Kinematic collision at initial pose s=0.0.",
        )

    # Optional FCL BVH acceleration when both geometries are B-Rep exportable
    fcl_link = None
    fcl_obs = None
    fcl_deflection = 0.02
    if use_fcl and link_node.is_brep_exportable() and obstacle_node.is_brep_exportable():
        s_l, shp_l = compile_brep(link_node)
        s_o, shp_o = compile_brep(obstacle_node)
        if s_l == Status.PASS and s_o == Status.PASS and shp_l is not None and shp_o is not None:
            bvh_l = _brep_to_fcl_bvh(shp_l, deflection=fcl_deflection)
            bvh_o = _brep_to_fcl_bvh(shp_o, deflection=fcl_deflection)
            if bvh_l is not None and bvh_o is not None:
                fcl_link = fcl.CollisionObject(bvh_l, fcl.Transform())
                fcl_obs = fcl.CollisionObject(bvh_o, fcl.Transform())

    def _eval_sep_and_collision(s_val: float) -> tuple[bool, float, list[tuple[float, float, float]]]:
        """Return (is_violation, conservative_sep_lower_mm, witness_points)."""
        if fcl_link is not None and fcl_obs is not None:
            q_d = path.eval(s_val)
            R_np, t_np = chain.link_world_pose(moving_link, q_d)
            fcl_link.setTransform(fcl.Transform(R_np, t_np))
            req_d = fcl.DistanceRequest(enable_nearest_points=True)
            res_d = fcl.DistanceResult()
            fcl_dist = float(fcl.distance(fcl_link, fcl_obs, req_d, res_d))
            # Safe chord-sagitta pad for tessellated mesh
            safe_lb = fcl_dist - 4.0 * fcl_deflection
            if safe_lb > min_clearance_mm + 0.08:
                return False, safe_lb, []

        # Near contact or FCL unavailable: use exact GeometryQuery clearance & signed check
        p_node = _pose_node(s_val)
        rep = check_clearance(
            GeometryQuery(p_node),
            q_obs,
            min_clearance_mm=min_clearance_mm,
            max_depth=3,
            initial_radius=0.5,
        )
        if rep.status == Status.FAIL:
            return True, 0.0, rep.witness_points
        if rep.status == Status.PASS:
            lb = float(rep.metrics.get("certified_min_lower_bound_mm", min_clearance_mm + eps_contact))
            return False, lb, []
        return False, float(rep.metrics.get("min_lower_bound_mm", 0.0)), []

    # 2. Level A conservative advancement + Level B near-contact bisection
    s = 0.0
    v_max_seen = 0.0
    min_sep_seen = float("inf")
    iters = 0

    while s < 1.0 and iters < max_iters:
        iters += 1
        violated, sep_lb, witnesses = _eval_sep_and_collision(s)
        min_sep_seen = min(min_sep_seen, sep_lb)
        if violated:
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.KINEMATIC_COLLISION,
                witness_points=witnesses,
                metrics={"collision_s": s, "steps": iters, "v_upper_max": v_max_seen},
                message=f"Kinematic collision detected at s={s:.6f}.",
            )

        v_upper = chain.interval_velocity_upper(moving_link, path, s, 1.0)
        v_max_seen = max(v_max_seen, v_upper)
        if v_upper < 1e-12:
            s = 1.0
            break

        margin_avail = sep_lb - min_clearance_mm - eps_contact
        if margin_avail > 0.0:
            ds = min(1.0 - s, max(0.5 * min_step, margin_avail / v_upper))
            # Refine interval velocity on [s, s + ds]
            v_loc = max(chain.interval_velocity_upper(moving_link, path, s, s + ds), 1e-12)
            ds = min(1.0 - s, margin_avail / v_loc)
            if ds < min_step * 0.25:
                ds = min(1.0 - s, min_step)
            s_next = s + ds
        else:
            # Near contact: step by min_step and check for collision
            s_next = min(1.0, s + min_step)

        # Check if s_next hits; if so, Level B bisect on [s, s_next] to pinpoint collision_s
        viol_next, _, w_next = _eval_sep_and_collision(s_next)
        if viol_next:
            lo, hi = s, s_next
            best_w = w_next
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                v_m, _, w_m = _eval_sep_and_collision(mid)
                if v_m:
                    hi = mid
                    best_w = w_m
                else:
                    lo = mid
            return VerificationReport(
                status=Status.FAIL,
                reason=FailReason.KINEMATIC_COLLISION,
                witness_points=best_w,
                metrics={"collision_s": hi, "steps": iters, "v_upper_max": v_max_seen},
                message=f"Kinematic collision pinpointed at s={hi:.6f}.",
            )
        s = s_next

    if s < 1.0 - 1e-9:
        return VerificationReport(
            status=Status.UNKNOWN,
            metrics={"reached_s": s, "steps": iters, "v_upper_max": v_max_seen},
            message=f"Conservative advancement reached max_iters={max_iters} at s={s:.4f}.",
        )

    return VerificationReport(
        status=Status.PASS,
        metrics={"steps": iters, "min_separation_mm": min_sep_seen, "v_upper_max": v_max_seen},
        message="Open-chain kinematic sweep certified collision-free.",
    )


@dataclass(frozen=True)
class ScrewdriverEnvelope:
    """Structured screwdriver geometry (tip, shaft, handle) + axial retraction sweep."""

    tip_radius_mm: float
    shaft_radius_mm: float
    shaft_len_mm: float
    handle_radius_mm: float
    handle_len_mm: float
    retract_mm: float = 20.0

    def build_node(self) -> Node:
        shaft = Transform(
            Cylinder(mm(self.shaft_radius_mm), mm(self.shaft_len_mm), node_id="sd_shaft"),
            translation=(mm(0.0), mm(0.0), mm(0.5 * self.shaft_len_mm)),
        )
        handle = Transform(
            Cylinder(mm(self.handle_radius_mm), mm(self.handle_len_mm), node_id="sd_handle"),
            translation=(mm(0.0), mm(0.0), mm(self.shaft_len_mm + 0.5 * self.handle_len_mm)),
        )
        return Union([shaft, handle], node_id="screwdriver")

    def check_access(
        self,
        obstacle_node: Node,
        fastener_origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        min_clearance_mm: float = 0.0,
    ) -> VerificationReport:
        jg = JointGraph(
            joints=[
                PrismaticJoint(
                    "retract",
                    "base",
                    "tool",
                    axis_dir=(0.0, 0.0, 1.0),
                    q_min_mm=0.0,
                    q_max_mm=self.retract_mm,
                )
            ]
        )
        tool = Transform(
            self.build_node(),
            translation=tuple(mm(v) for v in fastener_origin_mm),  # type: ignore[arg-type]
        )
        chain = KinematicChain(jg, {"tool": tool})
        path = LinearMotionPath({"retract": 0.0}, {"retract": self.retract_mm})
        return check_kinematic_sweep(chain, "tool", obstacle_node, path, min_clearance_mm=min_clearance_mm)


@dataclass(frozen=True)
class HexKeyEnvelope:
    """Structured L-shaped hex key geometry + swing sector sweep."""

    hex_across_flats_mm: float
    short_arm_mm: float
    long_arm_mm: float
    swing_deg: float = 60.0

    def build_node(self) -> Node:
        r_eff = 0.5 * self.hex_across_flats_mm / math.cos(math.pi / 6.0)
        short_arm = Transform(
            Cylinder(mm(r_eff), mm(self.short_arm_mm), node_id="hex_short"),
            translation=(mm(0.0), mm(0.0), mm(0.5 * self.short_arm_mm)),
        )
        long_arm = Transform(
            Transform(
                Cylinder(mm(r_eff), mm(self.long_arm_mm), node_id="hex_long"),
                rotation_deg=(deg(0.0), deg(90.0), deg(0.0)),
            ),
            translation=(mm(0.5 * self.long_arm_mm), mm(0.0), mm(self.short_arm_mm)),
        )
        return Union([short_arm, long_arm], node_id="hex_key")

    def check_access(
        self,
        obstacle_node: Node,
        fastener_origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        min_clearance_mm: float = 0.0,
    ) -> VerificationReport:
        swing_rad = math.radians(self.swing_deg)
        jg = JointGraph(
            joints=[
                RevoluteJoint(
                    "swing",
                    "base",
                    "tool",
                    axis_origin_mm=fastener_origin_mm,
                    axis_dir=(0.0, 0.0, 1.0),
                    q_min_rad=0.0,
                    q_max_rad=swing_rad,
                )
            ]
        )
        tool = Transform(
            self.build_node(),
            translation=tuple(mm(v) for v in fastener_origin_mm),  # type: ignore[arg-type]
        )
        chain = KinematicChain(jg, {"tool": tool})
        path = LinearMotionPath({"swing": 0.0}, {"swing": swing_rad})
        return check_kinematic_sweep(chain, "tool", obstacle_node, path, min_clearance_mm=min_clearance_mm)
