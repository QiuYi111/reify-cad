"""Unfitted and meshfree physics plugins (FCM, WoS, MPM, ray occlusion, dual-channel mesh bridge)."""

from reify.physics.fcm import (
    compute_mac_matrix,
    eval_fcm_displacement_2d,
    fcm_solve_elasticity_2d,
    fcm_solve_modal_2d,
    fcm_verify_3level_gci,
)
from reify.physics.mesh_bridge import (
    compute_tet_quality,
    export_calculix_inp,
    mesh_brep_gmsh_channel_a,
    mesh_implicit_isosurface_stuffing_channel_b,
    run_calculix_or_conformal_fem,
)
from reify.physics.mpm import mpm_solve_quasistatic_2d
from reify.physics.optics import check_ray_occlusion
from reify.physics.wos import wos_solve_point

__all__ = [
    "check_ray_occlusion",
    "compute_mac_matrix",
    "compute_tet_quality",
    "eval_fcm_displacement_2d",
    "export_calculix_inp",
    "fcm_solve_elasticity_2d",
    "fcm_solve_modal_2d",
    "fcm_verify_3level_gci",
    "mesh_brep_gmsh_channel_a",
    "mesh_implicit_isosurface_stuffing_channel_b",
    "mpm_solve_quasistatic_2d",
    "run_calculix_or_conformal_fem",
    "wos_solve_point",
]
