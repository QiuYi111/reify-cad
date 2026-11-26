"""M2 End-to-End Verification Suite (ACT-T-M2): R/P CCD, stock FFT vs brute-force, thickness, IID yield, MMC/LMC."""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage
import scipy.stats

from reify import (
    Box,
    Cylinder,
    FailReason,
    HexKeyEnvelope,
    JointGraph,
    KinematicChain,
    LinearMotionPath,
    MultivariateNormalDistribution,
    PrismaticJoint,
    RevoluteJoint,
    Status,
    Subtract,
    Transform,
    Union,
    brute_force_binary_morphology,
    check_directional_wall_thickness,
    check_kinematic_sweep,
    check_material_condition,
    check_mating_pin_hole_condition,
    check_stock_fft_accessibility,
    check_yield,
    compute_structural_thickness,
    deg,
    mm,
)


def test_m2_kinematics_rp_chain_hexkey_and_closed_loop():
    """I-M2-1: R+P open-chain thin blade sweep, HexKeyEnvelope 60-deg swing, and closed-loop OUT_OF_DOMAIN."""
    # 1. Closed-loop 4-bar linkage -> immediate OUT_OF_DOMAIN
    closed_jg = JointGraph(
        joints=[
            RevoluteJoint("j1", "base", "l1", (0, 0, 0), (0, 0, 1), -1.0, 1.0),
            RevoluteJoint("j2", "l1", "l2", (10, 0, 0), (0, 0, 1), -1.0, 1.0),
            RevoluteJoint("j3", "l2", "l3", (10, 10, 0), (0, 0, 1), -1.0, 1.0),
            RevoluteJoint("j4", "l3", "base", (0, 10, 0), (0, 0, 1), -1.0, 1.0),
        ]
    )
    closed_chain = KinematicChain(closed_jg, {"l3": Box(mm(4.0), mm(4.0), mm(4.0))})
    rep_closed = check_kinematic_sweep(
        closed_chain,
        "l3",
        Box(mm(2.0), mm(2.0), mm(2.0)),
        LinearMotionPath({"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0}, {"j1": 0.5, "j2": 0.0, "j3": 0.0, "j4": 0.0}),
    )
    assert rep_closed.status == Status.OUT_OF_DOMAIN

    # 2. Serial Prismatic + Revolute 0.3 mm thin blade sweeping into sharp obstacle corner
    jg_rp = JointGraph(
        joints=[
            PrismaticJoint("slide", "base", "carriage", axis_dir=(1.0, 0.0, 0.0), q_min_mm=0.0, q_max_mm=10.0),
            RevoluteJoint("pivot", "carriage", "blade", axis_origin_mm=(0.0, 0.0, 0.0), axis_dir=(0.0, 0.0, 1.0), q_min_rad=0.0, q_max_rad=math.pi / 2.0),
        ]
    )
    blade = Transform(Box(mm(20.0), mm(0.3), mm(4.0)), translation=(mm(10.0), mm(0.0), mm(0.0)))
    chain_rp = KinematicChain(jg_rp, {"blade": blade})
    path_rp = LinearMotionPath({"slide": 0.0, "pivot": 0.0}, {"slide": 5.0, "pivot": math.pi / 2.0})

    # Compute analytical first-contact parameter s* against obstacle [10, 14] x [8, 12] x [-5, 5]
    # Blade local x in [0, 20], y in [-0.15, +0.15]
    obs_hit = Transform(Box(mm(4.0), mm(4.0), mm(10.0)), translation=(mm(12.0), mm(10.0), mm(0.0)))
    s_grid = np.linspace(0.0, 1.0, 50000)
    # Closest blade corner/edge hitting y=8.0 or x=10.0
    s_star = None
    for sv in s_grid:
        tx = 5.0 * sv
        th = 0.5 * math.pi * sv
        ct, st = math.cos(th), math.sin(th)
        # Check sample points along blade upper/lower edges
        rs = np.linspace(0.0, 20.0, 200)
        for ey in (-0.15, 0.15):
            wx = tx + rs * ct - ey * st
            wy = rs * st + ey * ct
            if np.any((wx >= 10.0) & (wx <= 14.0) & (wy >= 8.0) & (wy <= 12.0)):
                s_star = float(sv)
                break
        if s_star is not None:
            break
    assert s_star is not None

    rep_hit = check_kinematic_sweep(chain_rp, "blade", obs_hit, path_rp, min_clearance_mm=0.0)
    assert rep_hit.status == Status.FAIL
    assert rep_hit.reason == FailReason.KINEMATIC_COLLISION
    assert abs(rep_hit.metrics["collision_s"] - s_star) < 5e-3

    # Clear obstacle well outside sweep
    obs_clear = Transform(Box(mm(4.0), mm(4.0), mm(10.0)), translation=(mm(30.0), mm(30.0), mm(0.0)))
    rep_clear = check_kinematic_sweep(chain_rp, "blade", obs_clear, path_rp, min_clearance_mm=0.2)
    assert rep_clear.status == Status.PASS

    # 3. L-shaped HexKeyEnvelope (60 deg swing) blocked by tall wall vs clear
    hex_key = HexKeyEnvelope(hex_across_flats_mm=4.0, short_arm_mm=12.0, long_arm_mm=30.0, swing_deg=60.0)
    wall_block = Transform(Box(mm(4.0), mm(4.0), mm(20.0)), translation=(mm(18.0), mm(10.0), mm(10.0)))
    rep_hex_fail = hex_key.check_access(wall_block)
    assert rep_hex_fail.status == Status.FAIL
    assert rep_hex_fail.reason == FailReason.KINEMATIC_COLLISION

    wall_far = Transform(Box(mm(4.0), mm(4.0), mm(20.0)), translation=(mm(-20.0), mm(-20.0), mm(10.0)))
    rep_hex_pass = hex_key.check_access(wall_far)
    assert rep_hex_pass.status == Status.PASS


def test_m2_stock_fft_vs_brute_force_and_directional_thickness():
    """I-M2-2: Stock-aware 3D FFT exact equivalence vs brute-force + directional vs structural thickness."""
    # 1. Deep pocket in stock P_stock where flute fits bottom, but wide shank hits unremoved stock wall
    # Part: 20x20x12 block with a narrow 4x4x8 deep slot in center (x,y in [-2,2], z in [-2,6])
    stock = Box(mm(20.0), mm(20.0), mm(12.0))
    slot = Transform(Box(mm(4.0), mm(4.0), mm(8.0)), translation=(mm(0.0), mm(0.0), mm(2.0)))
    # Also add a boundary-touching boss at x=+9 to stress-test zero-padding (no periodic wrap-around!)
    boss = Transform(Box(mm(2.0), mm(4.0), mm(4.0)), translation=(mm(9.0), mm(0.0), mm(8.0)))
    part = Union([Subtract(stock, slot), boss])

    # Tool: tip at (0,0,0); short cutting flute z in [0, 3], r=1.5; wider non-cutting shank z in [3, 9], r=3.5
    flute = Transform(Cylinder(mm(1.5), mm(3.0)), translation=(mm(0.0), mm(0.0), mm(1.5)))
    shank = Transform(Cylinder(mm(3.5), mm(6.0)), translation=(mm(0.0), mm(0.0), mm(6.0)))

    rep_fft = check_stock_fft_accessibility(
        part,
        stock,
        fixture_node=None,
        cutting_flute_node=flute,
        non_cutting_shank_node=shank,
        approach_axis="+z",
        voxel_mm=1.0,
    )
    assert rep_fft.status == Status.FAIL
    assert rep_fft.reason == FailReason.TOOL_COLLISION
    assert rep_fft.metrics["unremovable_voxels"] > 0

    # Brute-force voxel-for-voxel equivalence verification (I-M2-2)
    m = rep_fft.metrics
    v_part = m["v_part"]
    v_stock = m["v_stock"]
    v_fix = m["v_fix"]
    k_cut = m["k_cut"]
    k_shank = m["k_shank"]
    origin = m["kernel_origin"]

    bf_col_cut = brute_force_binary_morphology(v_part, k_cut, origin, correlate=True)
    bf_col_shank = brute_force_binary_morphology(v_stock | v_fix, k_shank, origin, correlate=True)
    bf_f_cons = ~(bf_col_cut | bf_col_shank)
    assert np.array_equal(bf_f_cons, m["f_cons_mask"]), "FFT f_cons disagrees with brute-force!"

    labels, _ = scipy.ndimage.label(bf_f_cons)
    nz = bf_f_cons.shape[2]
    seeds = np.unique(labels[:, :, nz - 1])
    seeds = seeds[seeds > 0]
    bf_f_reach = np.isin(labels, seeds)
    bf_swept = brute_force_binary_morphology(bf_f_reach, k_cut, origin, correlate=False)
    bf_m_rem = (v_stock & (~v_part)) & bf_swept
    assert np.array_equal(bf_m_rem, m["m_removable_mask"]), "FFT m_removable disagrees with brute-force!"

    # Memory byte-budget guard -> OUT_OF_DOMAIN
    rep_oom = check_stock_fft_accessibility(
        part, stock, None, flute, shank, voxel_mm=1.0, max_memory_bytes=1024
    )
    assert rep_oom.status == Status.OUT_OF_DOMAIN

    # 2. 30-deg slanted 1.0 mm thin wall -> directional wall thickness check
    slanted_wall = Transform(
        Box(mm(16.0), mm(16.0), mm(1.0)),
        rotation_deg=(deg(30.0), deg(0.0), deg(0.0)),
    )
    rep_w_fail = check_directional_wall_thickness(slanted_wall, min_wall_mm=1.4, sample_radius=1.0)
    assert rep_w_fail.status == Status.FAIL
    assert rep_w_fail.reason == FailReason.WALL_THICKNESS
    assert rep_w_fail.metrics["min_directional_thickness_mm"] < 1.1

    rep_w_pass = check_directional_wall_thickness(slanted_wall, min_wall_mm=0.6, sample_radius=1.0)
    assert rep_w_pass.status == Status.PASS

    # 3. T-junction rib (1.0 mm base + 1.0 mm rib): directional wall detects 1.0 mm; structural EDT shows thicker junction
    t_base = Box(mm(16.0), mm(12.0), mm(1.0))
    t_rib = Transform(Box(mm(1.0), mm(12.0), mm(6.0)), translation=(mm(0.0), mm(0.0), mm(3.5)))
    t_part = Union([t_base, t_rib])
    rep_t_dir = check_directional_wall_thickness(t_part, min_wall_mm=1.3, sample_radius=0.8)
    assert rep_t_dir.status == Status.FAIL
    struct_res = compute_structural_thickness(t_part, voxel_mm=0.25)
    assert struct_res["max_structural_thickness_mm"] > 1.2


def test_m2_iid_yield_clopper_pearson_and_mmc_lmc():
    """I-M2-3: IID Clopper-Pearson empirical coverage >= 95% nominal and deterministic MMC/LMC."""
    # Analytical correlated bivariate normal:
    # Gap G = X - Y, X ~ N(2.5, 0.2^2), Y ~ N(1.5, 0.2^2), corr = 0.7
    # Var(X - Y) = 0.04 + 0.04 - 2*0.7*0.2*0.2 = 0.024 -> std = sqrt(0.024)
    # E[X - Y] = 1.0, fail if X - Y < 0.7 -> Z = (0.7 - 1.0) / sqrt(0.024)
    cov = [[0.04, 0.7 * 0.04], [0.7 * 0.04, 0.04]]
    dist = MultivariateNormalDistribution(mean=[2.5, 1.5], cov=cov, names=["x", "y"])
    std_g = math.sqrt(0.04 + 0.04 - 2.0 * 0.7 * 0.04)
    p_true = float(1.0 - scipy.stats.norm.cdf((0.7 - 1.0) / std_g))

    def _pass_fn(s: dict[str, float]) -> bool:
        return (s["x"] - s["y"]) >= 0.7

    # 150 independent IID trials at confidence=0.95 -> empirical coverage >= 92%
    covered = 0
    n_trials = 150
    for trial in range(n_trials):
        rep = check_yield(_pass_fn, dist, n_samples=250, confidence=0.95, min_yield=0.90, seed=1000 + trial)
        assert rep.metrics["sampling_mode"] == "IID_PCG64"
        if rep.metrics["clopper_pearson_lower"] <= p_true:
            covered += 1
    assert covered / n_trials >= 0.92

    # Threshold inside [lower, upper] -> UNKNOWN; lower >= min_yield -> PASS; upper < min_yield -> FAIL
    rep_high = check_yield(_pass_fn, dist, n_samples=400, confidence=0.95, min_yield=0.50, seed=1)
    assert rep_high.status == Status.PASS
    rep_fail = check_yield(_pass_fn, dist, n_samples=400, confidence=0.95, min_yield=0.9999, seed=1)
    assert rep_fail.status == Status.FAIL
    rep_unk = check_yield(_pass_fn, dist, n_samples=200, confidence=0.95, min_yield=p_true, seed=1)
    assert rep_unk.status == Status.UNKNOWN

    # Deterministic MMC / LMC checks
    rep_pin_mmc_ok = check_mating_pin_hole_condition(10.0, 0.05, 10.2, 0.05, condition="MMC", min_clearance_mm=0.05)
    assert rep_pin_mmc_ok.status == Status.PASS
    assert abs(rep_pin_mmc_ok.metrics["radial_clearance_mm"] - 0.05) < 1e-9

    rep_pin_mmc_bad = check_mating_pin_hole_condition(10.0, 0.08, 10.1, 0.05, condition="MMC", min_clearance_mm=0.0)
    assert rep_pin_mmc_bad.status == Status.FAIL

    box_a = Box(mm(10.0), mm(10.0), mm(10.0))
    box_b = Transform(Box(mm(10.0), mm(10.0), mm(10.0)), translation=(mm(10.3), mm(0.0), mm(0.0)))
    # Under MMC (+0.2 expansion on each -> +0.4 total), boxes overlap by 0.1 mm -> FAIL
    rep_mmc_box = check_material_condition(box_a, box_b, tol_a_mm=0.2, tol_b_mm=0.2, condition="MMC", min_clearance_mm=0.0)
    assert rep_mmc_box.status == Status.FAIL
    # Under LMC (-0.2 contraction on each), gap becomes 0.7 mm -> PASS
    rep_lmc_box = check_material_condition(box_a, box_b, tol_a_mm=0.2, tol_b_mm=0.2, condition="LMC", min_clearance_mm=0.5)
    assert rep_lmc_box.status == Status.PASS
