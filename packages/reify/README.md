# Reify

**Reify** is a lightweight Python/PyTorch parametric geometry compilation, multi-start continuous sizing solver, bounded kinematics/DFM/unfitted-physics verification, and verified STEP export runtime for Coding & Engineering Agents (`v1.4`).

## Architecture Overview

- **Layer 0 (`reify.ast`, `reify.units`, `reify.spec`)**: Explicit physical units (`mm`, `deg`, `rad`), local-origin primitives (`Box`, `Cylinder`, `Sphere`, `ExtrudeRoundedRect`), differentiable `Anchor` frames, open-chain `R/P` `JointGraph`, and read-only `Spec` with `world_shell` / `datum_axis` freezes.
- **Layer 1 (`reify.query`, `reify.affine`, `reify.brep_compiler`)**: `GeometryQuery` with scale- and depth-aware numerical budget (`float32` screening, `float64` boundary certification), decoupled `classify_point` / `external_clearance_lower` / `unsigned_boundary_distance_lower`, first-order affine arithmetic, static `cover_surface` vs differentiable `sample_parametric_points`, `build123d`/OCCT B-Rep compilation, bidirectional Hausdorff checks, and round-trip STEP export.
- **Layer 2 (`reify.verify`, `reify.kinematics`, `reify.dfm`, `reify.tolerance`, `reify.physics`)**:
  - **M1**: Three-branch certified clearance (`PASS` via lower bound, `FAIL` via penetration or upper-bound witness, `UNKNOWN` otherwise) and bidirectional Hausdorff `world_shell` freeze.
  - **M2**: `FCL` + R/P interval velocity bound conservative advancement CCD & tool envelopes (`HexKeyEnvelope`, `ScrewdriverEnvelope`), stock-aware Local-ROI 3D FFT accessibility, directional wall thickness vs structural EDT thickness, and strict IID PCG64 Clopper-Pearson yield + deterministic `MMC/LMC`.
  - **M3**: Adaptive Nitsche FCM static/modal (with physical-domain energy participation filter & MAC), homogeneous Dirichlet WoS with normal-approx 95% CI, geometric ray occlusion, forward quasi-static ADR-MLS-MPM (with plane-stress $\sigma_{zz}=0$ $F_{zz}$ Newton solve), and dual-channel `CalculiX` `.inp` mesh bridge (`Gmsh` OCCT Channel A + BCC `Isosurface Stuffing` Channel B).
- **Layer 3 (`reify.solve`)**: Polynomial `SmoothBooleanProxy` ($C^1$ `quad_smin` / $C^2$ `cubic_smin`) + Sobol multi-start feasibility-first `solve()` with mandatory hard-geometry `verify()` final gate.

## Quickstart & E2E Verification

```bash
# Sync Python 3.12 environment (including optional NGSolve physics reference extra)
uv sync --python 3.12 --extra physics

# Run full M0-M3 end-to-end verification suite
uv run pytest tests/ -v
```

---

## EDAC Vault & Specification Graph

This repository is also an Obsidian Vault backed by a typed Markdown/YAML engineering graph (`specs -> needs -> requirements -> designs -> verification -> evidence`).

```bash
uv tool install git+https://github.com/Respine/RespineDev.git@v0.4.0

edac validate
edac doctor
edac coverage
```

Open the repository root as an Obsidian Vault starting with `docs/_indexes/Home.md` or `views/Engineering.base` (see `OBSIDIAN_SETUP.md`).
