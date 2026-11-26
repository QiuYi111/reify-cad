"""M3 End-to-End Verification Suite (ACT-T-M3): FCM, WoS, MPM, Ray Occlusion, and Dual-Channel Mesh Bridge."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import meshio
import numpy as np

from reify import (
    Box,
    Cylinder,
    GeometryQuery,
    SmoothUnion,
    Sphere,
    Status,
    Subtract,
    Transform,
    check_ray_occlusion,
    compute_mac_matrix,
    export_calculix_inp,
    fcm_solve_elasticity_2d,
    fcm_solve_modal_2d,
    fcm_verify_3level_gci,
    mesh_brep_gmsh_channel_a,
    mesh_implicit_isosurface_stuffing_channel_b,
    mm,
    mpm_solve_quasistatic_2d,
    run_calculix_or_conformal_fem,
    wos_solve_point,
)


def _run_ngsolve_cantilever_ref(L: float, H: float, t: float, E: float, nu: float, P: float) -> float:
    """Run body-fitted conformal NGSolve reference in an isolated subprocess (avoids OCCT clash)."""
    code = f"""
import json, ngsolve as ngs
from netgen.occ import WorkPlane, OCCGeometry, X
rect = WorkPlane().Rectangle({L}, {H}).Face()
rect.edges.Min(X).name = "left"
rect.edges.Max(X).name = "right"
geo = OCCGeometry(rect, dim=2)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=1.5))
fes = ngs.VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()
E, nu = {E}, {nu}
mu = E / (2*(1+nu))
lam = E * nu / ((1+nu)*(1-2*nu))
lam_ps = 2*mu*lam / (lam + 2*mu)
def eps(w): return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)
a = ngs.BilinearForm(fes)
a += (2*mu*ngs.InnerProduct(eps(u), eps(v)) + lam_ps*ngs.div(u)*ngs.div(v)) * ngs.dx
f = ngs.LinearForm(fes)
# Apply vertical shear traction P/(H*t) on right edge (x = L)
f += (({P} / ({H} * {t})) * v[1]) * ngs.ds("right")
a.Assemble(); f.Assemble()
gfu = ngs.GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
tip = gfu(mesh({L}, {H}/2))[1]
print(json.dumps({{"tip_disp": float(tip)}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return float(json.loads(proc.stdout.strip().splitlines()[-1])["tip_disp"])


def _run_ngsolve_kirsch_plate_ref(
    W: float = 20.0, H: float = 20.0, r_hole: float = 4.0, E: float = 210000.0, nu: float = 0.3, sigma0: float = 100.0
) -> float:
    """Conformal NGSolve reference for 20x20 plate with central circular hole under uniaxial tension."""
    code = f"""
import json, ngsolve as ngs
from netgen.occ import WorkPlane, OCCGeometry, X
outer = WorkPlane().Rectangle({W}, {H}).Face()
hole = WorkPlane().MoveTo({W}/2, {H}/2).Circle({r_hole}).Face()
plate = outer - hole
plate.edges.Min(X).name = "left"
plate.edges.Max(X).name = "right"
geo = OCCGeometry(plate, dim=2)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=1.2))
fes = ngs.VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()
E, nu = {E}, {nu}
mu = E / (2*(1+nu))
lam = E * nu / ((1+nu)*(1-2*nu))
lam_ps = 2*mu*lam / (lam + 2*mu)
def eps(w): return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)
a = ngs.BilinearForm(fes)
a += (2*mu*ngs.InnerProduct(eps(u), eps(v)) + lam_ps*ngs.div(u)*ngs.div(v)) * ngs.dx
f = ngs.LinearForm(fes)
f += ({sigma0} * v[0]) * ngs.ds("right")
a.Assemble(); f.Assemble()
gfu = ngs.GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
comp = ngs.Integrate({sigma0} * gfu[0], mesh.Boundaries("right"))
print(json.dumps({{"compliance": float(comp)}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return float(json.loads(proc.stdout.strip().splitlines()[-1])["compliance"])


def test_m3_fcm_3level_gci_slanted_cantilever_kirsch_and_modal():
    """I-M3-1 & I-M3-2: 25-deg slanted Timoshenko cantilever 3-level GCI, NGSolve ref, tiny cut-cells, modal MAC."""
    L, H, t = 40.0, 8.0, 1.0
    E, nu, P = 210000.0, 0.3, 100.0
    theta = math.radians(25.0)
    ct, st = math.cos(theta), math.sin(theta)

    # Exact Timoshenko tip deflection
    I_z = t * H**3 / 12.0
    A = t * H
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    delta_exact = (P * L**3) / (3.0 * E * I_z) + (P * L) / (kappa * G * A)

    def _slanted_beam_sdf(pts: np.ndarray) -> np.ndarray:
        # Rotate world pts back to beam local frame [0, L] x [-H/2, H/2]
        lx = pts[:, 0] * ct + pts[:, 1] * st
        ly = -pts[:, 0] * st + pts[:, 1] * ct
        qx = np.abs(lx - 0.5 * L) - 0.5 * L
        qy = np.abs(ly) - 0.5 * H
        return np.linalg.norm(np.maximum(np.column_stack([qx, qy]), 0.0), axis=-1) + np.minimum(
            np.maximum(qx, qy), 0.0
        )

    # Slanted root segment (x_loc=0, y_loc in [-H/2, H/2]) & tip segment (x_loc=L)
    def _to_world(lx: float, ly: float) -> tuple[float, float]:
        return (lx * ct - ly * st, lx * st + ly * ct)

    root_p0 = _to_world(0.0, -0.5 * H)
    root_p1 = _to_world(0.0, +0.5 * H)
    tip_p0 = _to_world(L, -0.5 * H)
    tip_p1 = _to_world(L, +0.5 * H)
    tip_mid = np.array([_to_world(L, 0.0)], dtype=np.float64)

    # Bounding box of rotated beam + small margin (so grid edges do NOT align with beam!)
    corners = [_to_world(x, y) for x in (0.0, L) for y in (-0.5 * H, 0.5 * H)]
    bbox = (
        (min(c[0] for c in corners) - 0.7, min(c[1] for c in corners) - 0.7),
        (max(c[0] for c in corners) + 0.7, max(c[1] for c in corners) + 0.7),
    )

    # Load P perpendicular to beam axis: (-st, ct) * (P / (H * t))
    trac_vec = (-st * P / (H * t), ct * P / (H * t))

    def _solve_tip_perp(h_val: float) -> float:
        res = fcm_solve_elasticity_2d(
            _slanted_beam_sdf,
            bbox,
            E,
            nu,
            t,
            dirichlet_segments=[(root_p0, root_p1, (0.0, 0.0))],
            tractions=[(tip_p0, tip_p1, trac_vec)],
            h=h_val,
            alpha_k=1e-6,
            quad_depth=3,
            nitsche_gamma=40.0,
            plane_mode="plane_stress",
            probe_points=tip_mid,
        )
        assert res["status"] == Status.PASS
        u_tip = res["probe_disp"][0]
        return float(-u_tip[0] * st + u_tip[1] * ct)

    rep_gci = fcm_verify_3level_gci(_solve_tip_perp, h_coarse=4.0, gci_tol=0.05)
    assert rep_gci.status == Status.PASS
    assert rep_gci.metrics["p_obs"] > 0.5
    phi_fine = rep_gci.metrics["phi_h4"]
    assert abs(phi_fine - delta_exact) / delta_exact < 0.06

    # Cross-check against isolated subprocess NGSolve body-fitted FEM
    ng_tip = _run_ngsolve_cantilever_ref(L, H, t, E, nu, P)
    assert abs(phi_fine - ng_tip) / ng_tip < 0.05

    # Tiny cut-cell stability stress test (1e-5 cut volume fraction)
    res_tiny = fcm_solve_elasticity_2d(
        lambda pts: np.maximum(
            np.abs(pts[:, 0] - 5.0) - 5.0,
            np.abs(pts[:, 1] - 2.0005) - 2.0,  # grazes y=4.0005 vs y=4 grid line
        ),
        ((0.0, 0.0), (10.0, 6.0)),
        E,
        nu,
        t,
        dirichlet_segments=[((0.0, 0.0005), (0.0, 4.0005), (0.0, 0.0))],
        tractions=[((10.0, 0.0005), (10.0, 4.0005), (0.0, 10.0))],
        h=2.0,
        alpha_k=1e-6,
        quad_depth=3,
    )
    assert res_tiny["status"] == Status.PASS
    assert np.all(np.isfinite(res_tiny["u"]))

    # Modal analysis with physical-domain energy participation filter + MAC
    rho = 7.85e-9  # tonne/mm^3
    m_coarse = fcm_solve_modal_2d(
        _slanted_beam_sdf,
        bbox,
        E,
        nu,
        rho,
        t,
        dirichlet_segments=[(root_p0, root_p1, (0.0, 0.0))],
        h=2.5,
        n_modes=4,
    )
    assert m_coarse["status"] == Status.PASS
    assert len(m_coarse["frequencies_hz"]) >= 3
    assert all(eta >= 0.85 for eta in m_coarse["participations"])

    # Euler-Bernoulli 1st bending frequency check: f1 = (1.8751^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))
    f1_eb = (1.875104**2 / (2.0 * math.pi * L * L)) * math.sqrt((E * I_z) / (rho * A))
    assert abs(m_coarse["frequencies_hz"][0] - f1_eb) / f1_eb < 0.08

    # MAC self-consistency & mode orthogonality
    mac_self = compute_mac_matrix(m_coarse["modes"][:, :3], m_coarse["modes"][:, :3])
    assert np.all(np.diag(mac_self) > 0.99)
    assert mac_self[0, 1] < 0.15

    # Kirsch circular-hole plate benchmark vs isolated conformal NGSolve
    W_k, H_k, r_k, sigma0 = 20.0, 20.0, 4.0, 100.0

    def _kirsch_sdf(pts: np.ndarray) -> np.ndarray:
        qx = np.abs(pts[:, 0] - 0.5 * W_k) - 0.5 * W_k
        qy = np.abs(pts[:, 1] - 0.5 * H_k) - 0.5 * H_k
        box_d = np.linalg.norm(np.maximum(np.column_stack([qx, qy]), 0.0), axis=-1) + np.minimum(
            np.maximum(qx, qy), 0.0
        )
        hole_d = np.hypot(pts[:, 0] - 0.5 * W_k, pts[:, 1] - 0.5 * H_k) - r_k
        return np.maximum(box_d, -hole_d)

    res_k = fcm_solve_elasticity_2d(
        _kirsch_sdf,
        ((0.0, 0.0), (W_k, H_k)),
        E,
        nu,
        t,
        dirichlet_segments=[((0.0, 0.0), (0.0, H_k), (0.0, 0.0))],
        tractions=[((W_k, 0.0), (W_k, H_k), (sigma0, 0.0))],
        h=1.25,
        alpha_k=1e-6,
        quad_depth=3,
        nitsche_gamma=40.0,
        plane_mode="plane_stress",
    )
    assert res_k["status"] == Status.PASS
    ng_k_comp = _run_ngsolve_kirsch_plate_ref(W_k, H_k, r_k, E, nu, sigma0)
    solid_plate_comp = (sigma0 * sigma0 * W_k * H_k) / E
    assert res_k["compliance"] > solid_plate_comp
    assert abs(res_k["compliance"] - ng_k_comp) / ng_k_comp < 0.08


def test_m3_wos_convergence_ci_and_ray_occlusion():
    """I-M3-1: WoS SE O(N^-1/2), 95% CI coverage, OUT_OF_DOMAIN guards, and ray occlusion."""
    # Concentric spherical shell: R1 = 5 mm (T=100), R2 = 15 mm (T=0)
    shell = Subtract(Sphere(mm(15.0)), Sphere(mm(5.0)))
    gq = GeometryQuery(shell)
    r_q = 8.0
    u_exact = 100.0 * (5.0 * (15.0 - r_q)) / (r_q * (15.0 - 5.0))  # 43.75

    def _bc_fn(pts: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(pts, axis=-1)
        return np.where(r < 10.0, 100.0, 0.0)

    # 1. Domain guards
    assert wos_solve_point(gq, (r_q, 0.0, 0.0), _bc_fn, medium="heterogeneous").status == Status.OUT_OF_DOMAIN
    assert wos_solve_point(gq, (r_q, 0.0, 0.0), _bc_fn, bc_type="neumann").status == Status.OUT_OF_DOMAIN
    assert wos_solve_point(gq, (0.0, 0.0, 0.0), _bc_fn).status == Status.OUT_OF_DOMAIN

    # 2. SE convergence rate O(N^-1/2) across N = 100, 400, 1600
    ses = []
    ns = [100, 400, 1600]
    for n in ns:
        rep = wos_solve_point(gq, (r_q, 0.0, 0.0), _bc_fn, n_samples=n, seed=42)
        assert rep.status == Status.PASS
        ses.append(rep.metrics["standard_error"])
    slope, _ = np.polyfit(np.log(ns), np.log(ses), 1)
    assert -0.65 < slope < -0.35

    # 3. 95% normal-approximation CI empirical coverage >= 88/100
    covered = 0
    for trial in range(100):
        rep = wos_solve_point(gq, (r_q, 0.0, 0.0), _bc_fn, n_samples=200, seed=500 + trial)
        lo, hi = rep.metrics["ci_95"]
        if lo <= u_exact <= hi:
            covered += 1
    assert covered >= 88

    # 4. Threshold inside CI -> UNKNOWN
    rep_unk = wos_solve_point(gq, (r_q, 0.0, 0.0), _bc_fn, n_samples=200, limit_max=u_exact, seed=7)
    assert rep_unk.status == Status.UNKNOWN

    # 5. Ray occlusion check
    box = Transform(Box(mm(4.0), mm(4.0), mm(4.0)), translation=(mm(10.0), mm(0.0), mm(0.0)))
    rep_hit = check_ray_occlusion(
        box,
        np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        max_dist_mm=20.0,
    )
    assert rep_hit.status == Status.FAIL
    assert rep_hit.metrics["hit_ray_idx"] == 0

    rep_clr = check_ray_occlusion(
        box,
        np.array([[0.0, 5.0, 0.0], [0.0, -5.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        max_dist_mm=20.0,
    )
    assert rep_clr.status == Status.PASS


def test_m3_mpm_plane_stress_fzz_plane_strain_and_dual_channel_mesh_bridge(tmp_path: pathlib.Path):
    """I-M3-3: Plane Stress (sigma_zz=0) vs Plane Strain MPM, plus Gmsh Channel A & BCC Channel B."""
    # 1. Quasi-static MPM: 10 x 4 block clamped at x <= 0.1, axial pull along +X
    def _block_sdf(pts: np.ndarray) -> np.ndarray:
        qx = np.abs(pts[:, 0] - 5.0) - 5.0
        qy = np.abs(pts[:, 1] - 2.0) - 2.0
        return np.linalg.norm(np.maximum(np.column_stack([qx, qy]), 0.0), axis=-1) + np.minimum(
            np.maximum(qx, qy), 0.0
        )

    def _fixed_left(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0] <= 0.2

    # Uniform body force f_x = 0.05 N/mm^2 per particle volume
    def _body_pull(pts: np.ndarray) -> np.ndarray:
        f = np.zeros_like(pts)
        f[:, 0] = 0.02
        return f

    assert (
        mpm_solve_quasistatic_2d(
            _block_sdf, ((0.0, 0.0), (10.0, 4.0)), 100.0, 0.35, 1e-3, require_friction_grad=True
        ).status
        == Status.OUT_OF_DOMAIN
    )

    rep_ps = mpm_solve_quasistatic_2d(
        _block_sdf,
        ((0.0, 0.0), (10.0, 4.0)),
        E=100.0,
        nu=0.35,
        rho=1e-3,
        mode="plane_stress",
        fixed_region_fn=_fixed_left,
        particle_force_fn=_body_pull,
        dx=1.0,
        ppc_1d=2,
        max_steps=3000,
    )
    assert rep_ps.status == Status.PASS
    assert rep_ps.metrics["max_sigma_zz_rel"] < 1e-6

    rep_pe = mpm_solve_quasistatic_2d(
        _block_sdf,
        ((0.0, 0.0), (10.0, 4.0)),
        E=100.0,
        nu=0.35,
        rho=1e-3,
        mode="plane_strain",
        fixed_region_fn=_fixed_left,
        particle_force_fn=_body_pull,
        dx=1.0,
        ppc_1d=2,
        max_steps=3000,
    )
    assert rep_pe.status == Status.PASS
    # Plane stress is softer than plane strain under same axial pull
    assert rep_ps.metrics["mean_disp"][0] > rep_pe.metrics["mean_disp"][0]

    # 2. Dual-Channel Mesh Bridge: Channel A (Gmsh OCCT C3D4 & C3D10)
    bracket = Subtract(Box(mm(20.0), mm(8.0), mm(4.0)), Cylinder(mm(1.5), mm(8.0)))
    m_a1 = mesh_brep_gmsh_channel_a(bracket, mesh_size_mm=2.5, order=1)
    assert m_a1["status"] == Status.PASS
    assert m_a1["quality"]["min_volume"] > 0.0
    assert m_a1["quality"]["min_scaled_jacobian"] > 0.05
    # Verify sharp-edge bounding box preservation (< 1e-4 mm)
    n_a1 = m_a1["nodes"]
    assert np.max(np.abs(np.min(n_a1, axis=0) - np.array([-10.0, -4.0, -2.0]))) < 1e-4
    assert np.max(np.abs(np.max(n_a1, axis=0) - np.array([10.0, 4.0, 2.0]))) < 1e-4

    fixed_a = np.nonzero(n_a1[:, 0] <= -9.9)[0].tolist()
    load_a = np.nonzero(n_a1[:, 0] >= 9.9)[0].tolist()
    inp_a = export_calculix_inp(m_a1, tmp_path / "bracket_a.inp", 210000.0, 0.3, fixed_a, load_a, (0.0, 0.0, -100.0))
    assert len(meshio.read(str(inp_a)).points) == n_a1.shape[0]
    fem_a = run_calculix_or_conformal_fem(
        inp_a, m_a1, 210000.0, 0.3, fixed_a, load_a, (0.0, 0.0, -100.0), require_ccx=True
    )
    assert fem_a.status == Status.PASS
    assert fem_a.metrics["solver_backend"] == "ccx"
    assert fem_a.metrics["load_mean_disp_xyz"][2] < 0.0

    m_a2 = mesh_brep_gmsh_channel_a(bracket, mesh_size_mm=3.0, order=2)
    assert m_a2["status"] == Status.PASS and m_a2["elem_type"] == "C3D10"
    n_a2 = m_a2["nodes"]
    fixed_a2 = np.nonzero(n_a2[:, 0] <= -9.9)[0].tolist()
    load_a2 = np.nonzero(n_a2[:, 0] >= 9.9)[0].tolist()
    inp_a2 = export_calculix_inp(m_a2, tmp_path / "bracket_a_c3d10.inp", 210000.0, 0.3, fixed_a2, load_a2, (0.0, 0.0, -100.0))
    fem_a2 = run_calculix_or_conformal_fem(
        inp_a2, m_a2, 210000.0, 0.3, fixed_a2, load_a2, (0.0, 0.0, -100.0), require_ccx=True
    )
    assert fem_a2.status == Status.PASS
    assert fem_a2.metrics["solver_backend"] == "ccx"
    assert fem_a2.metrics["load_mean_disp_xyz"][2] < 0.0

    # 3. Channel B (BCC Isosurface Stuffing on non-BRep SmoothUnion)
    smooth_part = SmoothUnion(
        Box(mm(10.0), mm(10.0), mm(10.0)),
        Transform(Sphere(mm(5.0)), translation=(mm(5.0), mm(0.0), mm(0.0))),
        k=mm(2.0),
    )
    assert mesh_brep_gmsh_channel_a(smooth_part)["status"] == Status.OUT_OF_DOMAIN
    m_b = mesh_implicit_isosurface_stuffing_channel_b(smooth_part, cell_size_mm=2.0)
    assert m_b["status"] == Status.PASS
    q_b = m_b["quality"]
    assert q_b["min_volume"] > 0.0
    assert q_b["min_dihedral_deg"] > 10.0
    assert q_b["max_dihedral_deg"] < 166.0

    n_b = m_b["nodes"]
    fixed_b = np.nonzero(n_b[:, 0] <= np.min(n_b[:, 0]) + 1.0)[0].tolist()
    load_b = np.nonzero(n_b[:, 0] >= np.max(n_b[:, 0]) - 1.0)[0].tolist()
    inp_b = export_calculix_inp(m_b, tmp_path / "smooth_b.inp", 70000.0, 0.33, fixed_b, load_b, (0.0, 0.0, -50.0))
    fem_b = run_calculix_or_conformal_fem(
        inp_b, m_b, 70000.0, 0.33, fixed_b, load_b, (0.0, 0.0, -50.0), require_ccx=True
    )
    assert fem_b.status == Status.PASS
    assert fem_b.metrics["solver_backend"] == "ccx"
    assert fem_b.metrics["load_mean_disp_xyz"][2] < 0.0
