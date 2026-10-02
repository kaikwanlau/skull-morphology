#!/usr/bin/env python3

import os as _os
import sys as _sys
_THIS_FILE = globals().get("__file__", _os.path.join(_os.getcwd(), "figure.py"))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(_THIS_FILE))))
import paths
_OUT = paths.output_dir(_THIS_FILE)

FINCH_MESH_DIR = str(paths.FINCHES)
HC_DIR = str(paths.HONEYCREEPERS)
CR_DIR = str(paths.CARDUELINES)
PERO_DIR = str(paths.PEROMYSCUS)
HUMAN_DIR = str(paths.HUMAN)
HUMAN_FIT = dict(ROI_START_PERCENT=0.10, ROI_END_PERCENT=0.50,
                 MIN_ORBIT_RADIUS=2.0, MAX_ORBIT_RADIUS=60.0,
                 CURVATURE_RADIUS=2.0, TARGET_POINT_COUNT=400,
                 MAX_SEED_ATTEMPTS=15)
OUT_DIR = str(_OUT / "figures")
CACHE_DIR = str(_OUT / "figure_cache")

PHOTO_FILE = str(paths.FIGURE_INPUTS / "photo" / "C.pallidus.jpg")
PHOTO_CROP = (0.184, 0.193, 0.865, 0.745)
WIRE_RESOLUTION = 30
WIRE_RESOLUTION_BY_FIGURE = {
    "FIG_overview_revised": 16,
    "FIG_orbit_fitting": 20,
    "FIG_sphere_examples": 16,
    "FIG_difference_curvature": 10,
    "FIG_SI_sphere_examples": 15,
}
RAW_MESH_FILE = str(paths.FIGURE_INPUTS / "raw_meshes" / "P2.InornataA.stl")
DISPLAY_MAX_FACES = 200000
S4_POINT_SCALE = 0
S4_SECTION_POINT_SCALE = 0
FIG12_LEGEND_FRAME = True
FIG12_MARKER_EDGE = False
BELOW_CRITERIA_STYLE = "filled"
PRIOR_REMESH_FILES = {"C.PallidusA": str(paths.FIGURE_INPUTS / "prior_remeshing" / "C.PallidusA.stl"),
                      "C.flaveolaA": str(paths.FIGURE_INPUTS / "prior_remeshing" / "C.flaveolaA.stl")}

SPECIMENS = {
    "overview": "P2.InornataA",
    "bounding": "G.SeptentrionalistA",
    "orbit_steps": "P2.InornataA",
    "sphere_examples": [("C.PallidusD", "C. pallidus"), ("G.FortisD", "G. fortis"),
                        ("E.campestrisB", "E. campestris"), ("M.nigraD", "M. nigra")],
    "curvature_contrast": ["T.canoraE", "G.ConirostrisD"],
    "ellipsoid_steps": "P2.InornataA",
    "ellipsoid_examples": [("C2.FuscaB", "C. fusca"), ("G.ConirostrisF", "G. conirostris"),
                           ("L.noctisE", "L. noctis"), ("T.canoraA", "T. canora")],
    "honeycreepers": [('L. caeruleirostrisA', "Loxops caeruleirostris"),
                      ('P. xanthophrysA', "Pseudonestor xanthophrys")],
    "remeshing": ["C.PallidusA", "C.flaveolaA"],
    "si_spheres": [("C.ParvulusB", "C. parvulus"), ("C.PsittaculaD", "C. psittacula"), ("G.ConirostrisE", "G. conirostris"),
                   ("G.DifficilisA", "G. difficilis"), ("G.FuliginosaD", "G. fuliginosa"), ("G.ScandensA", "G. scandens"),
                   ("G.SeptentrionalistB", "G. septentrionalis"), ("P.CrassirostrisD", "P. crassirostris"),
                   ("P2.InornataA", "P. inornata"), ("C.flaveolaD", "C. flaveola"), ("L.anoxanthusB", "L. anoxanthus"),
                   ("L.noctisB", "L. noctis"), ("L.violaceaD", "L. violacea"), ("T.canoraA", "T. canora"),
                   ("T.BicolorA", "T. bicolor")],
    "si_ellipsoids": [("C2.OlivaceaC", "C. olivacea"), ("C.PallidusA", "C. pallidus"), ("C.ParvulusA", "C. parvulus"),
                      ("C.PsittaculaA", "C. psittacula"), ("G.FortisB", "G. fortis"), ("G.FuliginosaB", "G. fuliginosa"),
                      ("G.ScandensB", "G. scandens"), ("G.SeptentrionalistA", "G. septentrionalis"),
                      ("P2.InornataA", "P. inornata"), ("E.campestrisE", "E. campestris"),
                      ("L.anoxanthusB", "L. anoxanthus"), ("L.portoricensisE", "L. portoricensis"),
                      ("M.nigraA", "M. nigra"), ("T.BicolorD", "T. bicolor"), ("T.olivaceaB", "T. olivacea")],
    "width": [("G.MagnirostrisA", "G. magnirostris (A)"), ("G.FuliginosaD", "G. fuliginosa (D)"),
              ("L.portoricensisA", "L. portoricensis (A)"), ("L.noctisA", "L. noctis (A)")],
    "definitions": "G.DifficilisA",
}

TRAINING_SET = set("""
C.PallidusA C.PallidusB C.ParvulusA C.ParvulusE C.PsittaculaA C.PsittaculaB C.PsittaculaC C.PsittaculaD
C2.FuscaA C2.FuscaB C2.OlivaceaC G.ConirostrisC G.DifficilisA G.FortisB G.FortisC G.FortisD G.FuliginosaA
G.FuliginosaB G.FuliginosaC G.FuliginosaD G.FuliginosaF G.MagnirostrisE G.ScandensB G.SeptentrionalistA
P.CrassirostrisB P.CrassirostrisC P.CrassirostrisD P2.InornataA P2.InornataB C.flaveolaA C.flaveolaC C.flaveolaD
C.flaveolaE E.campestrisB L.anoxanthusB L.anoxanthusD L.anoxanthusE L.noctisC L.portoricensisC L.portoricensisD
L.violaceaA L.violaceaB L.violaceaD L.violaceaE M.nigraA M.nigraC T.BicolorA T.BicolorC T.BicolorD T.olivaceaB
""".split())

EQ8 = (0.5653, -0.0013, -0.0004, -0.0153)

import os
import re
import sys
import glob
import argparse
import importlib.util
import itertools
import traceback
import warnings

_missing = [m for m in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "openpyxl", "trimesh", "rtree")
            if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit("Missing Python packages: " + ", ".join(_missing) +
             "\nInstall them with:\n    pip install " + " ".join(_missing))

import numpy as np
import pandas as pd
import trimesh
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

WRITTEN = []


def _p(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def _out(name):
    return os.path.join(_p(OUT_DIR), name)


def _folder_name(path):
    return os.path.basename(os.path.normpath(path))


class Skip(Exception):
    pass



def sphere_loss_function(params, points):
    center = params[:3]
    radius = params[3]
    if radius <= 0:
        return 1e9
    distances = np.linalg.norm(points - center, axis=1)
    return np.sum((distances - radius) ** 2)


def fit_sphere_iteratively(points, max_iterations=3, outlier_std_dev=2.0):
    current_points = points.copy()
    result = None
    for i in range(max_iterations):
        if len(current_points) < 4:
            return None, None, current_points
        initial_center = np.mean(current_points, axis=0)
        initial_radius = np.mean(np.linalg.norm(current_points - initial_center, axis=1))
        initial_guess = np.append(initial_center, initial_radius)
        result = minimize(sphere_loss_function, initial_guess, args=(current_points,),
                          method='L-BFGS-B',
                          bounds=[(None, None), (None, None), (None, None), (1e-6, None)])
        fit_center, fit_radius = result.x[:3], result.x[3]
        distances_to_center = np.linalg.norm(current_points - fit_center, axis=1)
        errors = np.abs(distances_to_center - fit_radius)
        mean_error, std_error = np.mean(errors), np.std(errors)
        inlier_mask = errors < (mean_error + outlier_std_dev * std_error)
        if np.all(inlier_mask):
            break
        current_points = current_points[inlier_mask]
    final_result = minimize(sphere_loss_function, result.x, args=(current_points,), method='L-BFGS-B',
                            bounds=[(None, None), (None, None), (None, None), (1e-6, None)])
    return final_result.x[:3], final_result.x[3], current_points


def run(file_path, ROI_START_PERCENT=0.30, ROI_END_PERCENT=0.70, MIN_ORBIT_RADIUS=2.0, MAX_ORBIT_RADIUS=6.0,
        CURVATURE_RADIUS=2.0, TARGET_POINT_COUNT=400, MAX_SEED_ATTEMPTS=15, MIN_COMPONENT=20,
        record_unclipped=False, half=None):
    original_mesh = trimesh.load_mesh(file_path)
    components = original_mesh.split(only_watertight=False)
    if len(components) > 1:
        processed_mesh = sorted(components, key=lambda c: len(c.vertices), reverse=True)[0]
    else:
        processed_mesh = original_mesh
    processed_mesh.process(validate=True)
    box_extents = processed_mesh.bounding_box.extents
    bounds = processed_mesh.bounds
    x_min = bounds[0, 0]
    x_range = box_extents[0]
    roi_x_min_threshold = x_min + (x_range * ROI_START_PERCENT)
    roi_x_max_threshold = x_min + (x_range * ROI_END_PERCENT)
    roi_mask = np.logical_and(processed_mesh.vertices[:, 0] > roi_x_min_threshold,
                              processed_mesh.vertices[:, 0] < roi_x_max_threshold)
    if half == 'y<0':
        roi_mask = np.logical_and(roi_mask, processed_mesh.vertices[:, 1] < 0)
    elif half == 'y>0':
        roi_mask = np.logical_and(roi_mask, processed_mesh.vertices[:, 1] > 0)
    if not np.any(roi_mask):
        roi_mask = np.ones(len(processed_mesh.vertices), dtype=bool)

    mean_curvatures = trimesh.curvature.discrete_mean_curvature_measure(
        processed_mesh, processed_mesh.vertices, radius=CURVATURE_RADIUS)
    curvatures_in_roi = mean_curvatures.copy()
    curvatures_in_roi[~roi_mask] = 1.0
    sorted_curvature_indices_local = np.argsort(curvatures_in_roi)
    top_N_seed_indices = np.arange(len(processed_mesh.vertices))[sorted_curvature_indices_local[:MAX_SEED_ATTEMPTS]]
    if curvatures_in_roi[top_N_seed_indices[0]] == 1.0:
        return dict(filename=os.path.basename(file_path), status='no_concave')

    sorted_indices = np.argsort(mean_curvatures)
    initial_mask = np.zeros(len(processed_mesh.vertices), dtype=bool)
    initial_mask[sorted_indices[:TARGET_POINT_COUNT]] = True
    valid_indices_mask = np.logical_and(initial_mask, roi_mask)

    rejected = []
    for attempt, seed_index in enumerate(top_N_seed_indices):
        if not valid_indices_mask[seed_index]:
            rejected.append((attempt + 1, 'seed_not_valid'))
            continue
        final_indices = np.array([])
        valid_edges_mask = valid_indices_mask[processed_mesh.edges].all(axis=1)
        selected_edges = processed_mesh.edges[valid_edges_mask]
        if len(selected_edges) > 0:
            comps = trimesh.graph.connected_components(selected_edges)
            if comps:
                for component in comps:
                    if seed_index in component:
                        final_indices = np.array(component)
                        break
                if len(final_indices) == 0:
                    final_indices = np.array(max(comps, key=len))
        if len(final_indices) < MIN_COMPONENT:
            rejected.append((attempt + 1, f'small_component_{len(final_indices)}'))
            continue
        picked_points = processed_mesh.vertices[final_indices]
        n_candidates = len(picked_points)
        fit_center, fit_radius, final_points = fit_sphere_iteratively(picked_points, outlier_std_dev=2.0)
        if fit_center is None:
            rejected.append((attempt + 1, 'fit_failed'))
            continue
        if not (MIN_ORBIT_RADIUS < fit_radius < MAX_ORBIT_RADIUS):
            rejected.append((attempt + 1, f'radius_{fit_radius:.2f}'))
            continue
        residuals = np.abs(np.linalg.norm(final_points - fit_center, axis=1) - fit_radius)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        out = dict(filename=os.path.basename(file_path), status='ok',
                   length_x=box_extents[0], width_y=box_extents[1], height_z=box_extents[2],
                   sphere_radius=fit_radius, cx=fit_center[0], cy=fit_center[1], cz=fit_center[2],
                   curvature=1 / fit_radius, seed_attempt=attempt + 1, n_candidates=n_candidates,
                   n_inliers=len(final_points), rms_mm=rms, fit_err_pct=100 * rms / fit_radius,
                   fit_side='y<0' if fit_center[1] < 0 else 'y>0',
                   centre_frac=(fit_center[0] - x_min) / x_range,
                   n_vertices=len(processed_mesh.vertices),
                   rejected=';'.join(f'{a}:{r}' for a, r in rejected))
        out['_inliers'] = final_points; out['_seed'] = processed_mesh.vertices[seed_index]
        if record_unclipped:
            valid_full = initial_mask.copy()
            ve = valid_full[processed_mesh.edges].all(axis=1)
            se = processed_mesh.edges[ve]
            comps = trimesh.graph.connected_components(se)
            n_full = 0
            for component in comps:
                if seed_index in component:
                    n_full = len(component)
                    break
            out['n_candidates_unclipped'] = n_full
        return out
    return dict(filename=os.path.basename(file_path), status='all_seeds_failed',
                length_x=box_extents[0], width_y=box_extents[1], height_z=box_extents[2],
                rejected=';'.join(f'{a}:{r}' for a, r in rejected))



def load_big(stl):
    m = trimesh.load_mesh(stl)
    comps = m.split(only_watertight=False)
    big = sorted(comps, key=lambda c: len(c.vertices), reverse=True)[0] if len(comps) > 1 else m
    big.process(validate=True)
    return big


def lateral(ax, big, c, R, inl, side, seed=None, alpha=0.55):
    V = big.vertices; F = big.faces
    sgn = 1.0 if side > 0 else -1.0
    sx = V[:, 0] * sgn
    sy = V[:, 2]
    depth = V[:, 1] * sgn
    tri_d = depth[F].mean(axis=1)
    order = np.argsort(tri_d)
    n = big.face_normals
    light = np.array([0.35 * sgn, 0.75 * sgn, 0.55]); light /= np.linalg.norm(light)
    sh = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
    polys = np.stack([sx[F], sy[F]], axis=2)[order]
    if EPS_SAFE:
        front = (n[:, 1] * sgn > 0)[order]
        grey = _over_white(sh[order][front], 0.75)
        cols = np.c_[np.repeat(grey[:, None], 3, axis=1), np.ones(front.sum())]
        ax.add_collection(PolyCollection(polys[front], facecolors=cols, edgecolors=cols, linewidths=0.3,
                                         antialiased=True))
    else:
        cols = np.c_[np.repeat(sh[order][:, None], 3, axis=1), np.full(len(order), alpha)]
        ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors='none', antialiased=True))
    B = np.array([[sgn, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, sgn, 0.0]])
    _gallery_sphere(ax, c, R, B, max(np.ptp(sx), np.ptp(sy)))
    if inl is not None and len(inl) and S4_POINT_SCALE:
        ax.scatter(inl[:, 0] * sgn, inl[:, 2], s=3.5 * S4_POINT_SCALE, c='#d40000', lw=0, zorder=5)
    if seed is not None:
        ax.scatter([seed[0] * sgn], [seed[2]], s=22, c='#00c000', edgecolors='k', lw=0.3, zorder=6)
    ax.set_aspect('equal'); ax.set_axis_off()
    pad = 0.8
    ax.set_xlim(sx.min() - pad, sx.max() + pad); ax.set_ylim(sy.min() - pad, sy.max() + pad)


def section(ax, big, c, R, inl, side, plane='z', halfwidth=0.6, crop=True):
    sgn = 1.0 if side > 0 else -1.0
    if plane == 'z':
        sec = big.section(plane_origin=c, plane_normal=[0, 0, 1])
        idx = (0, 1)
    else:
        sec = big.section(plane_origin=c, plane_normal=[1, 0, 0])
        idx = (1, 2)
    segs = []
    if sec is not None:
        for ent in sec.entities:
            pts = sec.vertices[ent.points]
            segs.append(np.c_[pts[:, idx[0]] * (sgn if idx[0] == 0 else 1), pts[:, idx[1]]])
    ax.add_collection(LineCollection(segs, colors='k', lw=0.8))
    t = np.linspace(0, 2 * np.pi, 200)
    if plane == 'z':
        ax.plot(c[0] * sgn + R * np.cos(t), c[1] + R * np.sin(t), color='#1f3fbf', lw=1.2)
        ax.plot([c[0] * sgn], [c[1]], '+', color='#1f3fbf', ms=6)
        if inl is not None and len(inl):
            near = inl[np.abs(inl[:, 2] - c[2]) < halfwidth]
            if S4_SECTION_POINT_SCALE:
                ax.scatter(near[:, 0] * sgn, near[:, 1], s=6 * S4_SECTION_POINT_SCALE, c='#d40000', lw=0, zorder=5)
        if crop:
            ax.set_xlim(c[0] * sgn - 2.4 * R, c[0] * sgn + 2.4 * R); ax.set_ylim(c[1] - 2.0 * R, c[1] + 2.0 * R)
    else:
        ax.plot(c[1] + R * np.cos(t), c[2] + R * np.sin(t), color='#1f3fbf', lw=1.2)
        if inl is not None and len(inl):
            near = inl[np.abs(inl[:, 0] - c[0]) < halfwidth]
            if S4_SECTION_POINT_SCALE:
                ax.scatter(near[:, 1], near[:, 2], s=6 * S4_SECTION_POINT_SCALE, c='#d40000', lw=0, zorder=5)
        if crop:
            ax.set_xlim(c[1] - 2.4 * R, c[1] + 2.4 * R); ax.set_ylim(c[2] - 2.0 * R, c[2] + 2.0 * R)
    ax.set_aspect('equal'); ax.set_axis_off()



def axis_aligned_ellipsoid_loss_function(params, points):
    center = params[:3]
    axes_lengths = params[3:6]
    if any(a <= 1e-6 for a in axes_lengths):
        return 1e9
    distances = np.sum(((points - center) / axes_lengths) ** 2, axis=1)
    return np.sum((distances - 1.0) ** 2)


def fit_axis_aligned_ellipsoid_iteratively(points, max_iterations=3, outlier_std_dev=1.5):
    current_points = points.copy()
    last_successful_result = None
    for i in range(max_iterations):
        if len(current_points) < 6:
            return None, None
        center_guess = np.mean(current_points, axis=0)
        axes_guess = np.std(current_points - center_guess, axis=0) * 2.0
        axes_guess[axes_guess < 1e-6] = 1e-6
        initial_guess = np.concatenate([center_guess, axes_guess])
        min_b, max_b = np.min(current_points, axis=0), np.max(current_points, axis=0)
        buffer = (max_b - min_b) * 0.5
        optimizer_bounds = list(zip(min_b - buffer, max_b + buffer)) + [(1e-6, None)] * 3
        result = minimize(axis_aligned_ellipsoid_loss_function, initial_guess, args=(current_points,),
                          method='L-BFGS-B', bounds=optimizer_bounds)
        if result.success:
            last_successful_result = result
        fit_center, fit_axes = result.x[:3], result.x[3:6]
        errors = np.abs(np.sum(((current_points - fit_center) / fit_axes) ** 2, axis=1) - 1.0)
        mean_error, std_error = np.mean(errors), np.std(errors)
        if std_error < 1e-6:
            break
        inlier_mask = errors < (mean_error + outlier_std_dev * std_error)
        if np.all(inlier_mask):
            break
        current_points = current_points[inlier_mask]
    if last_successful_result is None:
        return None, None
    final_result = minimize(axis_aligned_ellipsoid_loss_function, last_successful_result.x, args=(current_points,),
                            method='L-BFGS-B', bounds=optimizer_bounds)
    if not final_result.success:
        return None, None
    return final_result.x, current_points


def fit_ellipsoid(file_path, POSTERIOR_PERCENTILE=0.40, CURVATURE_RADIUS=3, TARGET_POINT_COUNT=1200):
    mesh = load_big(file_path)
    lo, hi = mesh.bounds[:, 0]
    posterior_mask = mesh.vertices[:, 0] > hi - (hi - lo) * POSTERIOR_PERCENTILE
    H = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius=CURVATURE_RADIUS)
    H[~posterior_mask] = -np.inf
    valid = np.zeros(len(mesh.vertices), dtype=bool)
    valid[np.argsort(H)[::-1][:TARGET_POINT_COUNT]] = True
    valid &= posterior_mask
    seed = int(np.argmax(H))
    comps = trimesh.graph.connected_components(mesh.edges[valid[mesh.edges].all(axis=1)])
    comp = next((c for c in comps if seed in c), None)
    idx = np.array(list(comp)) if comp is not None else np.array(list(max(comps, key=len)))
    if len(idx) < 30:
        return dict(status='too_few_points')
    params, pts = fit_axis_aligned_ellipsoid_iteratively(mesh.vertices[idx])
    if params is None:
        return dict(status='fit_failed')
    return dict(status='ok', center=params[:3], axes=params[3:6], _points=pts, _seed=mesh.vertices[seed])



def orient_principal_axes(mesh):
    V = mesh.vertices - mesh.vertices.mean(axis=0)
    _, E = np.linalg.eigh(V.T @ V)
    Rm = E[:, [2, 1, 0]].T
    for k in range(3):
        if np.sum((V @ Rm[k]) ** 3) < 0:
            Rm[k] *= -1
    if np.linalg.det(Rm) < 0:
        Rm[2] *= -1
    return trimesh.Trimesh(V @ Rm.T, mesh.faces, process=False)


def oriented_peromyscus_dir():
    out = os.path.join(_p(CACHE_DIR), "Peromyscus_oriented")
    os.makedirs(out, exist_ok=True)
    for fp in sorted(glob.glob(os.path.join(_p(PERO_DIR), "*.stl"))):
        target = os.path.join(out, os.path.basename(fp))
        if not os.path.isfile(target):
            orient_principal_axes(trimesh.load_mesh(fp)).export(target)
    return out


def _topology(file_path):
    m = trimesh.load_mesh(file_path)
    comps = m.split(only_watertight=False)
    big = sorted(comps, key=lambda c: len(c.vertices), reverse=True)[0] if len(comps) > 1 else m
    genus = (2 - (len(big.vertices) - len(big.edges_unique) + len(big.faces))) / 2
    return len(comps), genus


def measure_folder(folder, tag, ellipsoid=False):
    rows = []
    files = sorted(glob.glob(os.path.join(folder, "*.stl")))
    for k, fp in enumerate(files, 1):
        n_comp, genus = _topology(fp)
        a = run(fp)
        row = dict(filename=os.path.basename(fp), folder=tag, n_components=n_comp, genus=genus,
                   **{key: v for key, v in a.items() if not key.startswith('_') and key != 'filename'})
        if a.get('status') == 'ok':
            b = run(fp, half='y>0' if a['cy'] < 0 else 'y<0')
            row.update(status_second=b.get('status'), r_second=b.get('sphere_radius'), n_inliers_second=b.get('n_inliers'),
                       fit_err_pct_second=b.get('fit_err_pct'), cx2=b.get('cx'), cy2=b.get('cy'), cz2=b.get('cz'))
        if ellipsoid:
            e = fit_ellipsoid(fp)
            if e['status'] == 'ok':
                row.update(ellipsoid_center_x=e['center'][0], ellipsoid_center_y=e['center'][1],
                           ellipsoid_center_z=e['center'][2], ellipsoid_axis_a=e['axes'][0],
                           ellipsoid_axis_b=e['axes'][1], ellipsoid_axis_c=e['axes'][2])
        rows.append(row)
        print(f"    [{k}/{len(files)}] {row['filename']}: {row.get('status')}", flush=True)
    return pd.DataFrame(rows)


def get_measurements(refit=False):
    # Internal diagnostic cache: retain unsuccessful orbit-identification fits
    # for the limitation figures. These are not exported orbital measurements.
    os.makedirs(_p(CACHE_DIR), exist_ok=True)
    finch_csv = os.path.join(_p(CACHE_DIR), "measurements_finches.csv")
    other_csv = os.path.join(_p(CACHE_DIR), "measurements_other_taxa.csv")
    if refit or not os.path.isfile(finch_csv):
        if not glob.glob(os.path.join(_p(FINCH_MESH_DIR), "*.stl")):
            raise Skip(f"no .stl files in {_p(FINCH_MESH_DIR)} (see FINCH_MESH_DIR under SETTINGS)")
        print("  measuring the finch meshes (done once) ...")
        measure_folder(_p(FINCH_MESH_DIR), "finch", ellipsoid=True).to_csv(finch_csv, index=False)
    if refit or not os.path.isfile(other_csv):
        parts = []
        for folder, tag in ((_p(HC_DIR), _folder_name(HC_DIR)), (_p(CR_DIR), _folder_name(CR_DIR))):
            if glob.glob(os.path.join(folder, "*.stl")):
                print(f"  measuring {folder} (done once) ...")
                parts.append(measure_folder(folder, tag))
        if glob.glob(os.path.join(_p(PERO_DIR), "*.stl")):
            print("  measuring the Peromyscus meshes (done once) ...")
            parts.append(measure_folder(oriented_peromyscus_dir(), "Peromyscus"))
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["filename", "folder", "status", "n_inliers", "fit_err_pct"])).to_csv(other_csv, index=False)
    return pd.read_csv(finch_csv), pd.read_csv(other_csv)


def meets_criteria(n_inliers, fit_err_pct):
    return (n_inliers >= 40) & (fit_err_pct <= 10)


def dataset_table(finch):
    d = finch.copy()
    d["stem"] = d.filename.str.replace(".stl", "", regex=False)
    d["species"] = d.stem.str.replace(r"[A-G]$", "", regex=True)
    d["genus"] = d.species.map(SPECIES_TO_GENUS)
    missing = d.loc[d.genus.isna(), "species"].unique()
    if len(missing):
        raise ValueError(f"species without a genus in SPECIES_TO_GENUS: {missing}")
    d["Type"] = np.where(d.genus.isin(DF_GENERA), "Darwins Finches", "DF Relatives")
    d["Model"] = np.where(d.stem.isin(TRAINING_SET), "Training", "Test")
    d["curvature"] = 1 / d.sphere_radius
    d["predict_curvature"] = EQ8[0] + EQ8[1] * d.length_x + EQ8[2] * d.width_y + EQ8[3] * d.height_z
    d["L"] = (d.length_x * d.width_y * d.height_z) ** (1 / 3)
    d["kappa_tilde"] = d.L / d.sphere_radius
    d["cb"] = d.ellipsoid_axis_c / d.ellipsoid_axis_b
    d["xy"] = d.length_x / d.width_y
    d["_order"] = (d.Type != "Darwins Finches").astype(int)
    return d.sort_values(["_order", "filename"]).drop(columns="_order").reset_index(drop=True)


def check_against_paper(finch, other, d):
    rows = []

    def add(what, value, paper, fmt="{:.3f}"):
        v, p = fmt.format(value), fmt.format(paper)
        rows.append((what, v, p, "ok" if v == p else "DIFFERS"))

    sp = lambda a, b: stats.spearmanr(a, b)[0]
    for name, s, vals in [("DF", d[d.Type == "Darwins Finches"], (0.740, 0.802, 0.775)),
                          ("relatives", d[d.Type == "DF Relatives"], (0.712, 0.863, 0.902)),
                          ("all", d, (0.759, 0.838, 0.848))]:
        for col, v in zip(["length_x", "width_y", "height_z"], vals):
            add(f"Spearman(r, {col[0]}) {name}", sp(s.sphere_radius, s[col]), v)
    te = d[d.Model == "Test"]
    add("Eq. 8 R2 on test set", np.corrcoef(te.predict_curvature, te.curvature)[0, 1] ** 2, 0.8636, "{:.4f}")
    add("Eq. 8 mean relative error (%)", 100 * np.mean(abs(te.curvature - te.predict_curvature) / te.curvature), 6.14, "{:.2f}")
    add("Spearman(kappa~, L)", sp(d.kappa_tilde, d.L), -0.05, "{:.2f}")
    groups = lambda v: [v[d.genus == g] for g in sorted(d.genus.unique())]
    add("ANOVA F, kappa~", stats.f_oneway(*groups(d.kappa_tilde))[0], 3.62, "{:.2f}")
    add("ANOVA F, c/b", stats.f_oneway(*groups(d.cb))[0], 14.016)
    add("ANOVA F, x/y", stats.f_oneway(*groups(d.xy))[0], 14.021)
    add("finch inliers, median", finch.n_inliers.median(), 112, "{:.0f}")
    add("finch fit error, median (%)", finch.fit_err_pct.median(), 5.4, "{:.1f}")
    add("finch topology: intact", ((finch.n_components == 1) & (finch.genus == 0)).sum(), 37, "{:.0f}")
    add("finch topology: fragments", (finch.n_components > 1).sum(), 27, "{:.0f}")
    add("finch second orbits meeting criteria",
        meets_criteria(finch.n_inliers_second, finch.fit_err_pct_second).sum(), 83, "{:.0f}")
    new = other[other.folder.isin([_folder_name(HC_DIR), _folder_name(CR_DIR)])]
    if len(new):
        add("new taxa: inliers, median", new.n_inliers.median(), 76, "{:.0f}")
        add("new taxa: fits meeting criteria", meets_criteria(new.n_inliers, new.fit_err_pct).sum(), 38, "{:.0f}")
    w = max(len(r[0]) for r in rows)
    print(f"  {'quantity':{w}s}  {'computed':>9s}  {'paper':>7s}")
    for what, v, p, flag in rows:
        print(f"  {what:{w}s}  {v:>9s}  {p:>7s}  {flag}")
    n_diff = sum(r[3] != "ok" for r in rows)
    print(f"  -> {len(rows) - n_diff} of {len(rows)} values reproduce the paper" +
          ("" if n_diff == 0 else "  (check the input meshes)"))


LAYOUT = {
 "FIG_overview_revised": {
  "page": (
   2512.5,
   900.0
  ),
  "panels": {
   "a": (
    84.5,
    38.0,
    797.5,
    423.5
   ),
   "b": (
    838.7,
    44.5,
    1609.2,
    399.0
   ),
   "c": (
    1708.7,
    27.5,
    2455.2,
    401.0
   ),
   "d": (
    108.6,
    471.5,
    766.6,
    881.0
   ),
   "e": (
    854.5,
    473.0,
    1598.0,
    864.0
   ),
   "f": (
    1685.0,
    464.0,
    2495.5,
    867.0
   )
  },
  "labels": (
   (
    "(a)",
    14.4,
    62.0
   ),
   (
    "(b)",
    859.7,
    62.0
   ),
   (
    "(c)",
    1698.0,
    62.0
   ),
   (
    "(d)",
    14.4,
    486.8
   ),
   (
    "(e)",
    859.7,
    486.8
   ),
   (
    "(f)",
    1698.0,
    486.8
   )
  ),
  "label_size": 62.0
 },
 "FIG_bounding": {
  "page": (
   2692.9,
   1615.7
  ),
  "panels": (
   (
    195.2,
    58.3,
    1300.5,
    777.6
   ),
   (
    1403.7,
    75.2,
    2625.6,
    750.6
   ),
   (
    209.8,
    869.6,
    1312.8,
    1556.3
   ),
   (
    1408.2,
    887.5,
    2620.0,
    1549.5
   )
  ),
  "labels": (
   (
    "(a)",
    24.3,
    96.3
   ),
   (
    "(b)",
    24.3,
    888.2
   )
  ),
  "label_size": 97.87
 },
 "FIG_orbit_fitting": {
  "page": (
   2976.4,
   1474.0
  ),
  "panels": (
   (
    67.0,
    29.8,
    1327.0,
    658.5
   ),
   (
    1664.3,
    43.4,
    2940.4,
    643.6
   ),
   (
    76.9,
    813.5,
    1338.1,
    1422.5
   ),
   (
    1676.7,
    792.5,
    2911.9,
    1443.5
   )
  ),
  "labels": (
   (
    "(a)",
    26.5,
    113.1
   ),
   (
    "(b)",
    1655.5,
    113.1
   ),
   (
    "(c)",
    29.5,
    871.0
   ),
   (
    "(d)",
    1652.5,
    871.0
   )
  ),
  "label_size": 106.52
 },
 "FIG_difference_curvature": {
  "page": (
   1425.0,
   444.0
  ),
  "panels": (
   (
    52.2,
    14.2,
    703.0,
    404.9
   ),
   (
    724.4,
    18.4,
    1356.1,
    401.4
   )
  ),
  "labels": (
   (
    "(a)",
    16.8,
    70.0
   ),
   (
    "(b)",
    720.8,
    70.0
   )
  ),
  "label_size": 60.0
 },
 "FIG_ellipsoid_fitting": {
  "page": (
   2834.6,
   907.1
  ),
  "panels": (
   (
    72.0,
    93.3,
    1357.1,
    845.7
   ),
   (
    1511.8,
    81.5,
    2802.8,
    858.7
   )
  ),
  "labels": (
   (
    "(a)",
    16.8,
    96.3
   ),
   (
    "(b)",
    1454.0,
    96.3
   )
  ),
  "label_size": 97.87
 },
 "FIG_SI_remeshing": {
  "page": (
   1417.3,
   850.4
  ),
  "panels": (
   (
    80.3,
    18.3,
    681.5,
    433.5
   ),
   (
    744.7,
    28.3,
    1376.6,
    422.8
   ),
   (
    28.3,
    467.7,
    686.8,
    836.8
   ),
   (
    703.9,
    467.1,
    1363.6,
    836.2
   )
  )
 },
 "FIG_SI_more_mammal_examples_rowscale": {
  "page": (
   518.4,
   453.6
  ),
  "panels": (
   (
    34.8,
    66.8,
    149.5,
    174.2
   ),
   (
    183.0,
    80.0,
    333.0,
    161.2
   ),
   (
    366.5,
    45.5,
    463.2,
    195.5
   ),
   (
    33.0,
    295.8,
    150.5,
    399.0
   ),
   (
    180.2,
    303.0,
    334.0,
    392.0
   ),
   (
    363.8,
    270.5,
    454.5,
    424.2
   )
  )
 },
 "FIG_sphere_examples": {
  "page": (
   3344.9,
   2352.8
  ),
  "title_size": 120.0,
  "cells": {
   "C. pallidus": {
    "title": (
     859.4,
     111.7
    ),
    "box": (
     90.6,
     186.8,
     1626.4,
     1105.2
    )
   },
   "G. fortis": {
    "title": (
     2543.0,
     111.7
    ),
    "box": (
     1839.7,
     174.2,
     3276.6,
     1095.4
    )
   },
   "E. campestris": {
    "title": (
     880.1,
     1271.4
    ),
    "box": (
     33.4,
     1397.9,
     1725.4,
     2178.4
    )
   },
   "M. nigra": {
    "title": (
     2574.3,
     1271.4
    ),
    "box": (
     1843.9,
     1329.6,
     3291.9,
     2316.3
    )
   }
  }
 },
 "FIG_ellipsoid_examples": {
  "page": (
   3344.9,
   2267.7
  ),
  "title_size": 120.0,
  "cells": {
   "C. fusca": {
    "title": (
     863.9,
     111.7
    ),
    "box": (
     55.7,
     202.1,
     1651.5,
     1011.8
    )
   },
   "G. conirostris": {
    "title": (
     2568.5,
     111.7
    ),
    "box": (
     1719.8,
     186.8,
     3276.6,
     1029.9
    )
   },
   "L. noctis": {
    "title": (
     883.1,
     1194.8
    ),
    "box": (
     71.1,
     1283.6,
     1639.0,
     2220.2
    )
   },
   "T. canora": {
    "title": (
     2539.8,
     1194.9
    ),
    "box": (
     1810.4,
     1273.8,
     3252.9,
     2214.6
    )
   }
  }
 },
 "FIG_SI_sphere_examples": {
  "page": (
   4195.3,
   4960.6
  ),
  "title_size": 120.0,
  "cells": {
   "C. parvulus": {
    "title": (
     717.2,
     111.7
    ),
    "box": (
     136.4,
     184.0,
     1347.6,
     985.9
    )
   },
   "C. psittacula": {
    "title": (
     2189.9,
     111.7
    ),
    "box": (
     1465.5,
     181.9,
     2850.3,
     940.5
    )
   },
   "G. conirostris": {
    "title": (
     3553.2,
     109.6
    ),
    "box": (
     2964.0,
     167.4,
     4142.1,
     983.9
    )
   },
   "G. difficilis": {
    "title": (
     731.7,
     1104.8
    ),
    "box": (
     37.2,
     1184.4,
     1432.4,
     1928.4
    )
   },
   "G. fuliginosa": {
    "title": (
     2181.3,
     1104.8
    ),
    "box": (
     1577.1,
     1149.2,
     2771.8,
     1959.4
    )
   },
   "G. scandens": {
    "title": (
     3529.0,
     1104.8
    ),
    "box": (
     2866.8,
     1196.8,
     4162.8,
     1918.1
    )
   },
   "G. septentrionalis": {
    "title": (
     735.2,
     2114.0
    ),
    "box": (
     37.2,
     2166.1,
     1420.0,
     2928.8
    )
   },
   "P. crassirostris": {
    "title": (
     2155.7,
     2110.2
    ),
    "box": (
     1537.8,
     2172.3,
     2782.1,
     2918.5
    )
   },
   "P. inornata": {
    "title": (
     3525.0,
     2110.2
    ),
    "box": (
     2835.8,
     2182.7,
     4152.5,
     2897.8
    )
   },
   "C. flaveola": {
    "title": (
     729.7,
     3093.7
    ),
    "box": (
     47.5,
     3154.1,
     1411.7,
     3865.2
    )
   },
   "L. anoxanthus": {
    "title": (
     2158.8,
     3093.7
    ),
    "box": (
     1589.5,
     3129.3,
     2736.6,
     3914.8
    )
   },
   "L. noctis": {
    "title": (
     3545.1,
     3089.8
    ),
    "box": (
     2943.3,
     3154.1,
     4123.5,
     3852.8
    )
   },
   "L. violacea": {
    "title": (
     752.6,
     4049.7
    ),
    "box": (
     74.4,
     4144.2,
     1384.8,
     4906.9
    )
   },
   "T. canora": {
    "title": (
     2155.1,
     4045.8
    ),
    "box": (
     1544.0,
     4111.1,
     2715.9,
     4913.1
    )
   },
   "T. bicolor": {
    "title": (
     3577.8,
     4049.7
    ),
    "box": (
     2889.6,
     4121.5,
     4127.7,
     4923.4
    )
   }
  }
 },
 "FIG_SI_ellipsoid_examples": {
  "page": (
   4195.3,
   4960.6
  ),
  "title_size": 120.0,
  "cells": {
   "C. olivacea": {
    "title": (
     717.3,
     111.7
    ),
    "box": (
     55.8,
     181.9,
     1415.8,
     835.0
    )
   },
   "C. pallidus": {
    "title": (
     2189.8,
     111.7
    ),
    "box": (
     1515.1,
     177.8,
     2825.5,
     876.4
    )
   },
   "C. parvulus": {
    "title": (
     3527.4,
     111.7
    ),
    "box": (
     2922.6,
     157.1,
     4123.5,
     909.4
    )
   },
   "C. psittacula": {
    "title": (
     731.7,
     1061.3
    ),
    "box": (
     121.9,
     1126.5,
     1358.0,
     1901.6
    )
   },
   "G. fortis": {
    "title": (
     2181.3,
     1061.3
    ),
    "box": (
     1498.5,
     1130.6,
     2790.4,
     1872.6
    )
   },
   "G. fuliginosa": {
    "title": (
     3529.0,
     1061.3
    ),
    "box": (
     2840.0,
     1130.6,
     4156.6,
     1885.0
    )
   },
   "G. scandens": {
    "title": (
     735.1,
     2093.0
    ),
    "box": (
     64.1,
     2188.9,
     1420.0,
     2837.9
    )
   },
   "G. septentrionalis": {
    "title": (
     2188.7,
     2089.2
    ),
    "box": (
     1492.3,
     2143.4,
     2823.4,
     2879.2
    )
   },
   "P. inornata": {
    "title": (
     3525.1,
     2089.2
    ),
    "box": (
     2881.3,
     2172.3,
     4164.9,
     2842.0
    )
   },
   "E. campestris": {
    "title": (
     729.7,
     3047.2
    ),
    "box": (
     64.1,
     3152.1,
     1380.7,
     3838.3
    )
   },
   "L. anoxanthus": {
    "title": (
     2167.8,
     3047.2
    ),
    "box": (
     1500.6,
     3114.9,
     2794.5,
     3881.7
    )
   },
   "L. portoricensis": {
    "title": (
     3539.1,
     3043.3
    ),
    "box": (
     2866.8,
     3104.5,
     4156.6,
     3871.4
    )
   },
   "M. nigra": {
    "title": (
     733.1,
     4029.3
    ),
    "box": (
     111.6,
     4127.7,
     1343.5,
     4909.0
    )
   },
   "T. bicolor": {
    "title": (
     2155.9,
     4033.2
    ),
    "box": (
     1486.1,
     4096.7,
     2751.1,
     4892.4
    )
   },
   "T. olivacea": {
    "title": (
     3529.3,
     4033.2
    ),
    "box": (
     2902.0,
     4107.0,
     4144.2,
     4913.1
    )
   }
  }
 }
}


BOUNDING_PERSPECTIVE = (60.0, None, 60.0, None)
OVERVIEW_PERSPECTIVE = 60.0

CAMERAS = {
 "FIG_overview_revised": {"b": (251.5, -55.0, -6.5), "c": (273.5, 14.0, 8), "d": (243.8, 20.2, 6.8), "e": (256.0, 21.0, 8), "f": (264.5, 8.5,
     11.5)},
 "FIG_bounding": ((242.2, 17.4, 5.2), (286.1, 80.5, 1.9), (242.2, 17.4, 5.2), (180.0, 85.0, 0.0)),
 "FIG_orbit_fitting": ((274.5, 13.5, 8), (261.5, -2.5, 11.0), (274.5, 12.5, 9.0), (252.5, 18.5, 9.0)),
 "FIG_difference_curvature": ((248.5, 12.5, -8), (248.5, 16.5, 0)),
 "FIG_ellipsoid_fitting": ((105.5, -42.5, -15.0), (107.5, -40.5, -14.5)),
 "FIG_SI_remeshing": ((281.5, -5.0, -9.0), (288.5, -19.0, -13.0)),
 "FIG_sphere_examples": {"C. pallidus": (281.0, -3.5, -10.0), "G. fortis": (260.5, 11.5, 0), "E. campestris": (280.5,
     -2.5, -1.5), "M. nigra": (255.0, 9.5, -12.0)},
 "FIG_ellipsoid_examples": {"C. fusca": (268.0, -14.0, 0), "G. conirostris": (274.5, -6.5, 6.0), "L. noctis": (273.5, 19.5, -6.0), "T. canora": (282.5, -5.5, -7.5)},
 "FIG_SI_sphere_examples": {"C. parvulus": (259.0, -2.5, -7.0), "C. psittacula": (258.0, -3.0, 0), "G. conirostris": (281.5,
     -5.5, -8), "G. difficilis": (281.0, -6.0, 8), "G. fuliginosa": (281.5, -12.5, -8), "G. scandens": (283.0, -13.5,
     -8), "G. septentrionalis": (295.0, -14.0, -2.0), "P. crassirostris": (253.5, 13.0, 2.0), "P. inornata": (259.0,
     2.0, 0), "C. flaveola": (256.0, 15.5, -8), "L. anoxanthus": (291.5, -9.5, -12.0), "L. noctis": (282.0, -9.0, -6.0),
     "L. violacea": (256.5, 10.5, -3.0), "T. canora": (284.5, -11.0, -10.0), "T. bicolor": (264.5, 18.0, 8.5)},
 "FIG_SI_ellipsoid_examples": {"C. olivacea": (275.0, -2.0, 0), "C. pallidus": (274.5, -17.5, 8), "C. parvulus": (264.5,
     16.5, 7.5), "C. psittacula": (272.0, -26.0, 0), "G. fortis": (270.5, -11.0, 2.0), "G. fuliginosa": (264.5, 11.5,
     8), "G. scandens": (255.5, 19.0, 8), "G. septentrionalis": (289.5, -5.5, 5.5), "P. inornata": (281.0, -16.5, 8),
     "E. campestris": (256.5, -11.0, -6.5), "L. anoxanthus": (272.0, -2.0, 0), "L. portoricensis": (257.0, -1.0, 2.0),
     "M. nigra": (261.5, 0.5, 5.0), "T. bicolor": (254.0, 0.0, 0), "T. olivacea": (285.5, 0.0, -11.0)},
}


from matplotlib.patches import Rectangle

MESH_RGB = np.array([0.80, 0.80, 0.80])
SPHERE_BLUE = "#0000ff"
INLIER_RED = "#dc143c"
SEED_GREEN = "#00ff00"
AXIS_COLORS = ("#ff0000", "#008000", "#0000ff")
RENDER_RC = {"font.family": "serif",
             "font.serif": ["Liberation Serif", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}


def eye_from(azim_deg, elev_deg):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


FINCH_EYE = eye_from(-75, 18)


def view(angles):
    a, e, r = np.radians(angles)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0.0, 1.0, 0.0]) if abs(eye[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, eye); x /= np.linalg.norm(x)
    x = x * np.cos(r) + np.cross(eye, x) * np.sin(r)
    return np.vstack([x, np.cross(eye, x), eye])


def camera(eye, up=(0.0, 0.0, 1.0)):
    zc = np.asarray(eye, float); zc = zc / np.linalg.norm(zc)
    xc = np.cross(up, zc); xc = xc / np.linalg.norm(xc)
    return np.vstack([xc, np.cross(zc, xc), zc])


PERSPECTIVE = None


def set_perspective(distance=None, centre=None):
    global PERSPECTIVE
    PERSPECTIVE = None if distance is None else (float(distance), np.zeros(3) if centre is None else np.asarray(centre, float))


def to_screen(P, B):
    P = np.atleast_2d(np.asarray(P, float))
    if PERSPECTIVE is None:
        Q = P @ B.T
        return Q[:, :2], Q[:, 2]
    d, c = PERSPECTIVE
    Q = (P - c) @ B.T
    return Q[:, :2] * (d / np.maximum(d - Q[:, 2], 1e-6))[:, None], Q[:, 2]


def _over_white(rgb, alpha):
    return 1.0 - alpha * (1.0 - np.asarray(rgb, float))


def draw_mesh(ax, mesh, B, alpha=0.40, rgb=MESH_RGB, ambient=0.42, edge=None, edge_lw=0.1, cull=False,
              zorder=1, light=(-0.35, 0.45, 1.0), seam_lw=0.3, depth_grey=None):
    translucent = alpha < 1
    if EPS_SAFE and translucent:
        cull, rgb, alpha = True, _over_white(rgb, 0.75), 1.0
    V2, depth = to_screen(mesh.vertices, B)
    F, N = mesh.faces, mesh.face_normals @ B.T
    if cull:
        keep = N[:, 2] > 0
        F, N = F[keep], N[keep]
    L = np.asarray(light, float); L = L / np.linalg.norm(L)
    lam = np.abs(N @ L) if not cull else np.clip(N @ L, 0, 1)
    shade = ambient + (1 - ambient) * lam
    order = np.argsort(depth[F].mean(axis=1))
    face = np.c_[np.clip(shade[order, None] * np.asarray(rgb)[None, :], 0, 1), np.full(len(order), alpha)]
    if depth_grey is not None:
        dz = depth[F].mean(axis=1)[order]
        g = depth_grey[1] - (depth_grey[1] - depth_grey[0]) * (dz - dz.min()) / max(np.ptp(dz), 1e-12)
        face = np.c_[g, g, g, np.ones(len(g))]
    if edge is not None:
        ec, lw = edge, edge_lw
    elif alpha >= 1:
        ec, lw = face, seam_lw
    else:
        ec, lw = "none", 0
    ax.add_collection(PolyCollection(V2[F][order], facecolors=face, edgecolors=ec, linewidths=lw, zorder=zorder))
    return V2


def _uv_sphere(n_theta=30, n_phi=30):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ph = np.linspace(0, np.pi, n_phi)[1:-1]
    V = np.array([[np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)] for p in ph for t in th])
    V = np.vstack([V, [0, 0, 1], [0, 0, -1]])
    top, bottom = len(V) - 2, len(V) - 1
    idx = lambda i, j: i * n_theta + (j % n_theta)
    E = []
    for i in range(len(ph)):
        for j in range(n_theta):
            E.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < len(ph):
                E += [(idx(i, j), idx(i + 1, j)), (idx(i, j), idx(i + 1, j + 1))]
    for j in range(n_theta):
        E += [(top, idx(0, j)), (bottom, idx(len(ph) - 1, j))]
    return V, np.array(E)


_UV_CACHE = {}


def uv_sphere(n=None):
    n = int(n or WIRE_RESOLUTION)
    if n not in _UV_CACHE:
        _UV_CACHE[n] = _uv_sphere(n, n)
    return _UV_CACHE[n]


def draw_wire_ellipsoid(ax, center, axes, B, color=SPHERE_BLUE, lw=0.3, back_alpha=0.45, zorder=5, back_color=None,
                        back_lw=None, wire_n=None):
    V, E = uv_sphere(wire_n)
    S, depth = to_screen(np.asarray(center) + V * np.asarray(axes, float), B)
    front = depth[E].mean(axis=1) >= to_screen(center, B)[1][0]
    for mask, a, z, width in ((~front, back_alpha, zorder, lw if back_lw is None else back_lw), (front, 1.0, zorder + 0.1, lw)):
        col = matplotlib.colors.to_rgb(color)
        if mask is not front and back_color is not None:
            col, a = matplotlib.colors.to_rgb(back_color), 1.0
        if EPS_SAFE and a < 1:
            col, a = _over_white(col, a), 1.0
        ax.add_collection(LineCollection(S[E[mask]], colors=[col], linewidths=width, alpha=a, zorder=z))
    return S


def draw_points(ax, P, B, color=INLIER_RED, s=4, zorder=7, edge=None, lw=0.0):
    S, _ = to_screen(P, B)
    ax.scatter(S[:, 0], S[:, 1], s=s, color=color, edgecolors=edge if edge else "none", linewidths=lw, zorder=zorder)


def draw_segments(ax, segments, B, color, lw=1.0, zorder=6):
    ax.add_collection(LineCollection([to_screen(s, B)[0] for s in segments], colors=color, linewidths=lw, zorder=zorder))


def box_edges(corners):
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def aabb_corners(mesh):
    lo, hi = mesh.bounds
    return np.array([[(lo, hi)[a][0], (lo, hi)[b][1], (lo, hi)[c][2]] for a, b, c in itertools.product((0, 1), repeat=3)])


def frame(ax, pts, pad=0.03, aspect=None):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (lo + hi) / 2
    w, h = (hi - lo) * (1 + 2 * pad)
    if aspect is None:
        bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
        aspect = (bb.width * fw) / (bb.height * fh)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - h / 2, cy + h / 2)
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()


def placeholder(ax, text, size=6.5, lw=0.6):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(Rectangle((0.04, 0.06), 0.92, 0.88, facecolor="#f3f3f3", edgecolor="#9a9a9a", lw=lw, ls="--"))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=size, color="#555555")


def panel_label(ax, text, x=0.0, y=1.0, ha="left", size=10):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top", fontsize=size)


_MEMO = {}


def _memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


def finch_path(stem):
    path = os.path.join(_p(FINCH_MESH_DIR), stem + ".stl")
    if not os.path.isfile(path):
        raise Skip(f"mesh not found: {path} (see SPECIMENS / FINCH_MESH_DIR)")
    return path


def mesh_of(path):
    return _memo(("mesh", path), lambda: load_big(path))


def sphere_of(path):
    r = _memo(("sphere", path), lambda: run(path))
    if r.get("status") != "ok":
        raise RuntimeError(f"no accepted sphere fit for {path}")
    return r


def ellipsoid_of(path):
    e = _memo(("ellipsoid", path), lambda: fit_ellipsoid(path))
    if e.get("status") != "ok":
        raise RuntimeError(f"no ellipsoid fit for {path}")
    return e


def any_mesh(path):
    m = trimesh.load_mesh(path)
    return m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m


def display_mesh(path, max_faces=None):
    m = any_mesh(path)
    limit = DISPLAY_MAX_FACES if max_faces is None else max_faces
    if not limit or len(m.faces) <= limit:
        return m
    pitch = np.cbrt(m.area / len(m.faces)) * np.sqrt(len(m.faces) / limit)
    for _ in range(8):
        key = np.floor(m.vertices / pitch).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        pos = np.zeros((inv.max() + 1, 3))
        np.add.at(pos, inv, m.vertices)
        pos /= np.maximum(np.bincount(inv, minlength=len(pos))[:, None], 1)
        f = inv[m.faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        out = trimesh.Trimesh(pos, f, process=False)
        if len(out.faces) <= limit:
            print(f"    {os.path.basename(path)}: {len(m.faces)} faces simplified to {len(out.faces)} for display")
            return out
        pitch *= 1.3
    return out


def render_sphere_fit(ax, path, eye=None, up=(0, 0, 1), sphere=True, inliers=True, seed=True, alpha=0.40,
                      wire_lw=0.3, point_s=3.0, seed_s=16, back_alpha=0.45, seed_edge="k", mesh_style=None,
                      B=None, point_z=6, wire_n=None):
    m, r = mesh_of(path), sphere_of(path)
    B = B if B is not None else camera(FINCH_EYE if eye is None else eye, up)
    V2 = draw_mesh(ax, m, B, alpha=alpha, **(mesh_style or {}))
    c, R = np.array([r["cx"], r["cy"], r["cz"]]), r["sphere_radius"]
    S = draw_wire_ellipsoid(ax, c, [R] * 3, B, lw=wire_lw, back_alpha=back_alpha, wire_n=wire_n) if sphere \
        else to_screen(c, B)[0]
    if inliers:
        draw_points(ax, r["_inliers"], B, s=point_s, zorder=point_z)
    if seed:
        draw_points(ax, r["_seed"], B, color=SEED_GREEN, s=seed_s, edge=seed_edge, lw=0.3, zorder=8)
    return m, r, B, V2, S


def render_ellipsoid_fit(ax, path, eye=None, up=(0, 0, 1), ellipsoid=True, points=True, seed=True, axes_lines=True,
                         alpha=0.35, wire_lw=0.25, point_s=2.5, seed_s=16, axis_lw=1.4,
                         back_alpha=0.45, seed_edge="k", mesh_style=None, B=None, wire_n=None):
    m, e = mesh_of(path), ellipsoid_of(path)
    B = B if B is not None else camera(FINCH_EYE if eye is None else eye, up)
    V2 = draw_mesh(ax, m, B, alpha=alpha, **(mesh_style or {}))
    S = draw_wire_ellipsoid(ax, e["center"], e["axes"], B, lw=wire_lw, back_alpha=back_alpha, wire_n=wire_n) if ellipsoid \
        else to_screen(e["center"], B)[0]
    if points:
        draw_points(ax, e["_points"], B, s=point_s)
    if seed:
        draw_points(ax, e["_seed"], B, color=SEED_GREEN, s=seed_s, edge=seed_edge, lw=0.3, zorder=8)
    if axes_lines:
        for k, col in enumerate(AXIS_COLORS):
            d = np.zeros(3); d[k] = e["axes"][k]
            draw_segments(ax, [[e["center"] - d, e["center"] + d]], B, col, lw=axis_lw, zorder=9)
    return m, e, B, V2, S


def posterior_points(m, B, V2, frac):
    lo, hi = m.bounds[:, 0]
    return V2[m.vertices[:, 0] > lo + frac * (hi - lo)]


COMPOSITE_RC = {"font.family": "serif",
                "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "Times",
                               "TeX Gyre Termes", "DejaVu Serif"],
                "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}


PX = {"FIG_overview_revised": 0.47, "FIG_bounding": 0.88, "FIG_orbit_fitting": 0.75,
      "FIG_sphere_examples": 0.89, "FIG_difference_curvature": 1.21, "FIG_ellipsoid_fitting": 0.85,
      "FIG_ellipsoid_examples": 1.08, "FIG_SI_remeshing": 0.42, "FIG_SI_sphere_examples": 0.86,
      "FIG_SI_ellipsoid_examples": 0.90}
OPAQUE = dict(cull=True, rgb=np.array([0.83] * 3), ambient=0.30)
CAL = {"FIG_overview_revised": (1.00, 1.00), "FIG_bounding": (1.44, 1.00), "FIG_orbit_fitting": (0.93, 0.87),
       "FIG_sphere_examples": (1.70, 2.27), "FIG_difference_curvature": (1.00, 0.50),
       "FIG_ellipsoid_fitting": (1.07, 0.66), "FIG_ellipsoid_examples": (1.98, 2.18),
       "FIG_SI_remeshing": (0.35, 1.00), "FIG_SI_sphere_examples": (1.57, 0.78),
       "FIG_SI_ellipsoid_examples": (1.62, 1.79)}


def pv_sizes(name, kind="sphere"):
    p = PX[name]; f_line, f_area = CAL[name]
    style = dict(wire_lw=2 * p * f_line, point_s=(6 * p) ** 2 * f_area, seed_s=(15 * p) ** 2 * f_area,
                 back_alpha=1.0, seed_edge=None, wire_n=WIRE_RESOLUTION_BY_FIGURE.get(name))
    if kind == "ellipsoid":
        style.update(axis_lw=5 * p, alpha=0.30)
    return style


def page_figure(page):
    fig = plt.figure(figsize=(page[0] / 72.0, page[1] / 72.0))
    fig.page = tuple(page)
    return fig


def page_axes(fig, box, pad=0.0):
    W, H = fig.page
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])


def page_text(fig, x, y, text, size, ha="left", **kw):
    W, H = fig.page
    return fig.text(x / W, 1 - y / H, text, fontsize=size, ha=ha, va="baseline", **kw)


def page_overlay(fig):
    W, H = fig.page
    ax = fig.add_axes([0, 0, 1, 1], zorder=20)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    return ax


def fit_box(ax, pts):
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    bw, bh = bb.width * fw, bb.height * fh
    s = max((hi[0] - lo[0]) / bw, (hi[1] - lo[1]) / bh)
    cx, cy = (lo + hi) / 2
    ax.set_xlim(cx - s * bw / 2, cx + s * bw / 2); ax.set_ylim(cy - s * bh / 2, cy + s * bh / 2)
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()


def pin(ax, data_pt, page_pt, scale):
    bb = ax.get_position(); W, H = ax.figure.page
    X0, X1, Y0, Y1 = bb.x0 * W, bb.x1 * W, (1 - bb.y1) * H, (1 - bb.y0) * H
    ax.set_xlim(data_pt[0] + (X0 - page_pt[0]) / scale, data_pt[0] + (X1 - page_pt[0]) / scale)
    ax.set_ylim(data_pt[1] - (Y1 - page_pt[1]) / scale, data_pt[1] - (Y0 - page_pt[1]) / scale)
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()


def page_arrow(ov, start, tip, lw, head_length, head_width, color="k"):
    start, tip = np.asarray(start, float), np.asarray(tip, float)
    d = (tip - start) / np.linalg.norm(tip - start); n = np.array([-d[1], d[0]])
    base = tip - head_length * d
    ov.plot([start[0], base[0]], [start[1], base[1]], color=color, lw=lw, solid_capstyle="butt")
    ov.add_patch(Polygon([tip, base + n * head_width / 2, base - n * head_width / 2], closed=True,
                         facecolor=color, edgecolor="none"))


def draw_labels(fig, lay):
    for text, x, y in lay.get("labels", []):
        page_text(fig, x, y, text, lay["label_size"])


def fig01_overview():
    lay = LAYOUT["FIG_overview_revised"]; k = lay["page"][0] / 532.8; p = PX["FIG_overview_revised"]
    path = finch_path(SPECIMENS["overview"]); m = mesh_of(path)
    fig = page_figure(lay["page"])
    ax = {key: page_axes(fig, box) for key, box in lay["panels"].items()}
    photo = _p(PHOTO_FILE)
    if os.path.isfile(photo):
        img = plt.imread(photo)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = PHOTO_CROP
        ax["a"].imshow(img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], aspect="auto")
        ax["a"].set_axis_off()
        print("    note: Fig 1(a) embeds the photograph, the only raster element in any figure")
    else:
        placeholder(ax["a"], "photograph of $\\it{Camarhynchus\\ pallidus}$\n(Wikimedia Commons, CC BY 4.0)\n"
                             "file not found: set PHOTO_FILE", size=6.5 * k, lw=0.6 * k)
    cams = CAMERAS["FIG_overview_revised"]
    raw = _p(RAW_MESH_FILE)
    if os.path.isfile(raw):
        fit_box(ax["b"], draw_mesh(ax["b"], display_mesh(raw), view(cams["b"]), alpha=1.0,
                                   seam_lw=0.3 * k, **OPAQUE))
    else:
        placeholder(ax["b"], "raw scan mesh of $\\it{Pinaroloxias\\ inornata}$\nfile not found: set RAW_MESH_FILE",
                    size=6.5 * k, lw=0.6 * k)
    fit_box(ax["c"], draw_mesh(ax["c"], m, view(cams["c"]), alpha=1.0, seam_lw=0.3 * k, **OPAQUE))
    set_perspective(OVERVIEW_PERSPECTIVE, m.vertices.mean(axis=0))
    Bd = view(cams["d"]); corners = aabb_corners(m)
    V2 = draw_mesh(ax["d"], m, Bd, alpha=0.40)
    draw_segments(ax["d"], box_edges(corners), Bd, "#ff0000", lw=2 * p)
    fit_box(ax["d"], np.vstack([V2, to_screen(corners, Bd)[0]]))
    set_perspective(None)
    _, _, _, V2, S = render_sphere_fit(ax["e"], path, inliers=False, seed=False, B=view(cams["e"]),
                                       **pv_sizes("FIG_overview_revised"))
    fit_box(ax["e"], np.vstack([V2, S]))
    _, _, _, V2, S = render_ellipsoid_fit(ax["f"], path, points=False, seed=False, B=view(cams["f"]),
                                          **pv_sizes("FIG_overview_revised", "ellipsoid"))
    fit_box(ax["f"], np.vstack([V2, S]))
    draw_labels(fig, lay)
    save(fig, "FIG_overview_revised")


def fig02_bounding():
    lay = LAYOUT["FIG_bounding"]; p = PX["FIG_bounding"]
    path = finch_path(SPECIMENS["bounding"]); m = mesh_of(path)
    obb = m.bounding_box_oriented
    T, ext = np.asarray(obb.primitive.transform), np.asarray(obb.primitive.extents)
    obb_corners = (np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * ext) @ T[:3, :3].T + T[:3, 3]
    fig = page_figure(lay["page"])
    up_axis = int(np.argmax(np.abs(T[2, :3])))
    side_axis = int(np.argmax([abs(T[1, j]) if j != up_axis else -1 for j in range(3)]))
    tilt = np.radians(4.5)
    obb_B = camera(T[:3, up_axis] * np.cos(tilt) + T[:3, 3 - up_axis - side_axis] * np.sin(tilt),
                   T[:3, side_axis])
    for i, box in enumerate(lay["panels"]):
        corners, color = (obb_corners, "#008000") if i < 2 else (aabb_corners(m), "#ff0000")
        ax = page_axes(fig, box)
        set_perspective(BOUNDING_PERSPECTIVE[i], m.vertices.mean(axis=0))
        B = obb_B if i == 1 else view(CAMERAS["FIG_bounding"][i])
        V2 = draw_mesh(ax, m, B, alpha=0.6 if i < 2 else 0.4)
        draw_segments(ax, box_edges(corners), B, color, lw=2 * p * (1.44 if i < 2 else 1.98))
        fit_box(ax, np.vstack([V2, to_screen(corners, B)[0]]))
        set_perspective(None)
    draw_labels(fig, lay)
    save(fig, "FIG_bounding")


def fig03_orbit_fitting():
    lay = LAYOUT["FIG_orbit_fitting"]; k = lay["page"][0] / 532.8
    path = finch_path(SPECIMENS["orbit_steps"]); style = pv_sizes("FIG_orbit_fitting")
    fig = page_figure(lay["page"])
    steps = [dict(alpha=1.0, mesh_style=dict(seam_lw=0.3 * k, **OPAQUE), sphere=False, inliers=False, seed=False),
             dict(sphere=False, inliers=False, seed=False),
             dict(sphere=False, inliers=True, seed=False, point_s=style["point_s"]),
             dict(sphere=True, inliers=False, seed=False, wire_lw=style["wire_lw"], back_alpha=1.0,
                  wire_n=style["wire_n"])]
    for i, (box, opts) in enumerate(zip(lay["panels"], steps)):
        ax = page_axes(fig, box)
        _, r, B, V2, S = render_sphere_fit(ax, path, B=view(CAMERAS["FIG_orbit_fitting"][i]), **opts)
        fit_box(ax, np.vstack([V2, S]))
        if i in (1, 2):
            draw_points(ax, r["_seed"], B, color="#006e00", s=64.6 ** 2, zorder=9)
    ov = page_overlay(fig)
    for start, tip in (((1380.6, 353.5), (1680.3, 353.5)), ((1672.6, 630.8), (1337.4, 840.4)),
                       ((1387.5, 1116.5), (1687.2, 1116.5))):
        page_arrow(ov, start, tip, lw=12.76, head_length=98.3, head_width=71.4)
    draw_labels(fig, lay)
    save(fig, "FIG_orbit_fitting")


def _cells_figure(key, kind, stem_out, crop=None):
    lay = LAYOUT[stem_out]
    fig = page_figure(lay["page"]); cells = list(lay["cells"].values())
    opts = pv_sizes(stem_out, kind)
    render = render_sphere_fit if kind == "sphere" else render_ellipsoid_fit
    for i, (stem, label) in enumerate(SPECIMENS[key]):
        cell = lay["cells"].get(label, cells[i])
        ax = page_axes(fig, cell["box"])
        angles = CAMERAS.get(stem_out, {}).get(label)
        m, _, B, V2, S = render(ax, finch_path(stem), B=view(angles) if angles else None, **opts)
        fit_box(ax, np.vstack([posterior_points(m, B, V2, crop) if crop else V2, S]))
        page_text(fig, cell["title"][0], cell["title"][1], label, lay["title_size"], ha="center", style="italic")
    save(fig, stem_out)


def fig04_sphere_examples():
    _cells_figure("sphere_examples", "sphere", "FIG_sphere_examples")


def fig05_difference_curvature():
    lay = LAYOUT["FIG_difference_curvature"]; style = pv_sizes("FIG_difference_curvature")
    fig = page_figure(lay["page"])
    for i, (box, stem) in enumerate(zip(lay["panels"], SPECIMENS["curvature_contrast"])):
        ax = page_axes(fig, box)
        _, _, _, V2, S = render_sphere_fit(ax, finch_path(stem), seed=False,
                                           B=view(CAMERAS["FIG_difference_curvature"][i]), **style)
        fit_box(ax, np.vstack([V2, S]))
    draw_labels(fig, lay)
    save(fig, "FIG_difference_curvature")


def fig06_ellipsoid_fitting():
    lay = LAYOUT["FIG_ellipsoid_fitting"]; style = pv_sizes("FIG_ellipsoid_fitting", "ellipsoid")
    path = finch_path(SPECIMENS["ellipsoid_steps"])
    fig = page_figure(lay["page"])
    for i, (box, opts) in enumerate(zip(lay["panels"], (dict(ellipsoid=False), dict(points=False, seed=False)))):
        ax = page_axes(fig, box)
        _, _, _, V2, S = render_ellipsoid_fit(ax, path, B=view(CAMERAS["FIG_ellipsoid_fitting"][i]),
                                              **style, **opts)
        fit_box(ax, np.vstack([V2, S]))
    draw_labels(fig, lay)
    save(fig, "FIG_ellipsoid_fitting")


def fig07_ellipsoid_examples():
    _cells_figure("ellipsoid_examples", "ellipsoid", "FIG_ellipsoid_examples")


def fig13_honeycreeper_examples():
    lay = LAYOUT["FIG_SI_more_mammal_examples_rowscale"]
    shift = (24.0, 48.5)
    fig = page_figure((lay["page"][0], lay["page"][1] - shift[1]))
    ov = page_overlay(fig)
    scales = (32.1 / 5.0, 26.0 / 5.0)
    views = [((0.75, 0.62, 0.55), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((0, 0, 1), (-1, 0, 0))]
    view_labels = [("oblique (behind, right, above)", (92.05, 193.2), (91.65, 415.8)),
                   ("right lateral (from $+y$)", (257.75, 179.3), (257.0, 408.3)),
                   ("dorsal (from $+z$, anterior at top)", (414.65, 215.7), (408.95, 442.5))]
    headers, bars = ((20.7, 14.0), (20.7, 206.0)), (((31.6, 177.5), (47.65, 174.3)), ((30.4, 401.6), (43.4, 399.0)))
    for row, (stem, species) in enumerate(SPECIMENS["honeycreepers"]):
        path = os.path.join(_p(HC_DIR), stem + ".stl")
        if not os.path.isfile(path):
            raise Skip(f"mesh not found: {path} (see HC_DIR)")
        m, r = mesh_of(path), sphere_of(path)
        c, R = np.array([r["cx"], r["cy"], r["cz"]]), r["sphere_radius"]
        dy = shift[row]
        page_text(fig, headers[row][0], headers[row][1], f"({'ab'[row]})", 9, fontweight="bold",
                  family="sans-serif")
        for v, ((eye, up), (label, *label_xy)) in enumerate(zip(views, view_labels)):
            box = lay["panels"][3 * row + v]
            box = (box[0], box[1] - dy, box[2], box[3] - dy)
            ax = page_axes(fig, box, pad=15)
            B = camera(eye, up)
            V2 = draw_mesh(ax, m, B, alpha=0.40)
            S = draw_wire_ellipsoid(ax, c, [R] * 3, B, color=(0.08, 0.14, 0.78), lw=0.45,
                                    back_color=(0.23, 0.32, 0.85), back_lw=0.35)
            draw_points(ax, r["_inliers"], B, color=(0.78, 0.12, 0.12), s=3.2)
            draw_points(ax, r["_seed"], B, color=(0.09, 0.64, 0.09), s=16, zorder=8)
            pts = np.vstack([V2, S])
            pin(ax, (pts.min(0) + pts.max(0)) / 2, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), scales[row])
            page_text(fig, label_xy[row][0], label_xy[row][1] - dy, label, 7.5, ha="center")
        (x0, y), (tx, ty) = bars[row]
        ov.plot([x0, x0 + 5.0 * scales[row]], [y - dy, y - dy], color="k", lw=1.1, solid_capstyle="butt")
        page_text(fig, tx, ty - dy, "5 mm", 6.5, ha="center")
        if row == 0:
            page_text(fig, 38.0, 68.5 - dy - 18, "missing dorsal vault", 6.5)
            page_text(fig, 38.0, 75.7 - dy - 18, "(surface closed by hole filling)", 6.5)
            ov.annotate("", xy=(111.3, 122.6 - dy), xytext=(84.6, 80.8 - dy - 18),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=6.5, shrinkA=0, shrinkB=0))
    save(fig, "FIG_SI_more_mammal_examples_rowscale")


def figS1_remeshing():
    lay = LAYOUT["FIG_SI_remeshing"]; k = lay["page"][0] / 532.8
    fig = page_figure(lay["page"])
    style = dict(alpha=1.0, cull=True, rgb=np.array([0.86] * 3), edge=(0.18, 0.18, 0.18),
                 edge_lw=PX["FIG_SI_remeshing"] * CAL["FIG_SI_remeshing"][0])
    for row, stem in enumerate(SPECIMENS["remeshing"]):
        B = view(CAMERAS["FIG_SI_remeshing"][row])
        left, right = lay["panels"][2 * row], lay["panels"][2 * row + 1]
        ax = page_axes(fig, left); prior = PRIOR_REMESH_FILES.get(stem)
        if prior and os.path.isfile(_p(prior)):
            fit_box(ax, draw_mesh(ax, display_mesh(_p(prior)), B, **style))
        else:
            placeholder(ax, f"earlier remeshing of {stem}\n(prior studies)\nfile not found: set PRIOR_REMESH_FILES",
                        size=6.5 * k, lw=0.6 * k)
        ax = page_axes(fig, right)
        fit_box(ax, draw_mesh(ax, mesh_of(finch_path(stem)), B, **style))
    save(fig, "FIG_SI_remeshing")


def figS2_sphere_examples():
    _cells_figure("si_spheres", "sphere", "FIG_SI_sphere_examples")


def figS3_ellipsoid_examples():
    _cells_figure("si_ellipsoids", "ellipsoid", "FIG_SI_ellipsoid_examples")


def figS5_width_location():
    W, H = 1036.8, 460.8
    fig = plt.figure(figsize=(W / 72, H / 72))
    gs = gridspec.GridSpec(2, 4, figure=fig, left=10.8 / W, right=1026.0 / W, top=1 - 34.8 / H, bottom=1 - 422.9 / H,
                           wspace=10.8 / 245.7, hspace=21.2 / ((210.6 + 156.3) / 2), height_ratios=[210.6, 156.3])
    for col, (stem, label) in enumerate(SPECIMENS["width"]):
        m = mesh_of(finch_path(stem))
        ends = m.vertices[[np.argmin(m.vertices[:, 1]), np.argmax(m.vertices[:, 1])]]
        titles = (f"{label}, posterior view\nwidth {np.ptp(m.vertices[:, 1]):.1f} mm", "dorsal view")
        for row, (eye, up) in enumerate((((1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)))):
            ax = fig.add_subplot(gs[row, col]); B = camera(eye, up)
            V2 = draw_mesh(ax, m, B, alpha=1.0, depth_grey=(0.32, 0.86), seam_lw=0)
            draw_points(ax, ends, B, color="#dc143c", s=36, edge="k", lw=1.0, zorder=9)
            lo, hi = V2.min(axis=0), V2.max(axis=0); margin = 0.05 * (hi - lo)
            ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0]); ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
            ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(titles[row], fontsize=9)
    save(fig, "FIG_SI_width_location")


def figS8_definitions():
    path = finch_path(SPECIMENS["definitions"]); m, r = mesh_of(path), sphere_of(path)
    c, R, inl = np.array([r["cx"], r["cy"], r["cz"]]), r["sphere_radius"], r["_inliers"]
    b = run(path, half="y>0" if c[1] < 0 else "y<0")
    if b.get("status") != "ok":
        raise RuntimeError("no second-orbit fit for the S8 specimen")
    c2, R2 = np.array([b["cx"], b["cy"], b["cz"]]), b["sphere_radius"]
    blue, red, orange, grey = "#1f3fbf", "#d40000", "#ff7f0e", (0.5, 0.5, 0.5)
    fig = page_figure((532.8, 266.4)); ov = page_overlay(fig)
    t = np.linspace(0, 2 * np.pi, 400)
    sec = m.section(plane_origin=c, plane_normal=[0, 0, 1])
    segs = [sec.vertices[e.points][:, :2] for e in sec.entities] if sec is not None else []
    near = inl[np.abs(inl[:, 2] - c[2]) < 0.6]

    def draw_section(ax, gap_lw, circle_lw):
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.9))
        ax.plot(c[0] + R * np.cos(t), c[1] + R * np.sin(t), color=blue, lw=circle_lw)
        for p in near:
            q = c + (p - c) / np.linalg.norm(p - c) * R
            ax.plot([p[0], q[0]], [p[1], q[1]], color=orange, lw=gap_lw, solid_capstyle="butt", zorder=4)
        ax.scatter(near[:, 0], near[:, 1], s=9, c=red, lw=0, zorder=5)

    scale_a, centre_a = 15.34, (133.2, 109.2)
    axa = page_axes(fig, (24.2, 10.5, 242.2, 178.0))
    draw_section(axa, 1.6, 1.4)
    axa.plot([c[0]], [c[1]], "+", color=blue, ms=8, mew=1.2)
    ang = np.radians(-28.7)
    axa.plot([c[0], c[0] + R * np.cos(ang)], [c[1], c[1] + R * np.sin(ang)], "--", color=blue, lw=0.8)
    pin(axa, c[:2], centre_a, scale_a)
    page_text(fig, 161.9, 126.8, "$\\hat{r}$", 9, color=blue)
    zoom, inset = (114.0, 54.5, 127.8, 68.3), (159.3, 14.7, 234.2, 89.6)
    ins = page_axes(fig, inset)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    for spine in ins.spines.values():
        spine.set_visible(False)
    draw_section(ins, 2.6, 1.2)
    to_data = lambda X, Y: (c[0] + (X - centre_a[0]) / scale_a, c[1] - (Y - centre_a[1]) / scale_a)
    (xa, ya), (xb, yb) = to_data(zoom[0], zoom[3]), to_data(zoom[2], zoom[1])
    ins.set_xlim(xa, xb); ins.set_ylim(ya, yb)
    for (x0, y0, x1, y1), lw in ((zoom, 0.6), (inset, 0.7)):
        ov.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=grey, lw=lw))
    ov.plot([zoom[0], inset[0]], [zoom[1], inset[1]], color=grey, lw=0.6)
    ov.plot([zoom[0], inset[0]], [zoom[3], inset[3]], color=grey, lw=0.6)
    page_text(fig, 236.5, 34.0, "$e_i$", 7, color=(0.78, 0.35, 0.0))
    page_text(fig, 133.1, 234.4, "orange: the gap $e_i$ of a bone vertex to the sphere (inset: magnified)", 7, ha="center")
    page_text(fig, 132.8, 252.7, "fit error $=\\sqrt{\\mathrm{mean}(e_i^2)}\\,/\\,\\hat{r}$" +
              f" = {r['fit_err_pct']:.1f}% on this skull ({r['n_inliers']} inliers)", 7, ha="center")
    page_text(fig, 24.3, 17.5, "(a)", 9, fontweight="bold")
    scale_b, centre_b = 7.77, (374.35, 140.95)
    axb = page_axes(fig, (262.0, 30.0, 532.8, 225.0))
    B = camera((0, 0, 1), (0, 1, 0))
    draw_mesh(axb, m, B, alpha=1.0, cull=True, rgb=np.array([0.99] * 3), ambient=0.62, seam_lw=0)
    for cc, rr, col in ((c, R, blue), (c2, R2, red)):
        axb.plot(cc[0] + rr * np.cos(t), cc[1] + rr * np.sin(t), color=col, lw=1.4, zorder=6)
        axb.plot([cc[0]], [cc[1]], "+", color=col, ms=7, mew=1.1, zorder=6)
    pin(axb, c[:2], centre_b, scale_b)
    y_mid = centre_b[1] + c[1] * scale_b
    ov.plot([266.4, 527.5], [y_mid, y_mid], "--", color="k", lw=0.7)
    for x, y, text in ((488.7, y_mid - 2.4, "midline"), (488.7, y_mid + 4.3, "$y$ = 0"),
                       (274.6, 163.1, "half searched by"), (274.6, 169.9, "the pipeline (y < 0)"),
                       (274.6, 62.5, "other half, used for"), (274.6, 69.3, "the second fit (y > 0)")):
        page_text(fig, x, y, text, 6.3)
    page_text(fig, 410.0, 48.3, f"the two radii differ by {100 * abs(R2 - R) / R:.1f}%", 7, ha="center")
    ov.plot([291.9, 302.0], [197.0, 197.0], color=blue, lw=1.4)
    ov.plot([291.9, 302.0], [207.9, 207.9], color=red, lw=1.4)
    page_text(fig, 307.0, 198.8, f"orbit fitted by the pipeline, $\\hat{{r}}$ = {R:.2f} mm", 6.3)
    page_text(fig, 307.0, 209.7, f"second orbit, search restricted to the other half, $\\hat{{r}}'$ = {R2:.2f} mm", 6.3)
    page_text(fig, 266.4, 49.9, "(b)", 9, fontweight="bold")
    save(fig, "FIG_SI_definitions")



EPS_SAFE = False


def blend(c, a):
    from matplotlib.colors import to_rgb
    return tuple(1.0 - (1.0 - v) * a for v in to_rgb(c))


def A(c, a):
    return {"color": blend(c, a), "alpha": 1.0} if EPS_SAFE else {"color": c, "alpha": a}


FIG_RC = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "savefig.bbox": None,
}

SPEC = {
    "fig08": dict(figsize=(13.28,  6.50), tick=21.5, label=28.0,               legend=21.5, genus=24.0),
    "fig09": dict(figsize=(51.97, 38.03), tick=80.0, label=85.0, title=93.5,                 marker=400),
    "fig10": dict(figsize=(44.79, 36.46), tick=72.0, label=83.6, title=96.0, tag=113.2, cbar=70.2, marker=425),
    "fig11": dict(figsize=( 8.44,  3.49), tick=15.8, label=17.3,               legend=12.8, marker=32, legend_ms=6.5),
    "fig12": dict(figsize=(22.83, 12.60), tick=37.8, label=50.8, tag=52.5,     legend=44.3, marker=72, legend_ms=20.6),
}


SPECIES_TO_GENUS = {
    "C.Pallidus": "Camarhynchus", "C.Parvulus": "Camarhynchus",
    "C.Psittacula": "Camarhynchus",
    "C2.Fusca": "Certhidea", "C2.Olivacea": "Certhidea",
    "G.Conirostris": "Geospiza", "G.Difficilis": "Geospiza",
    "G.Fortis": "Geospiza", "G.Fuliginosa": "Geospiza",
    "G.Magnirostris": "Geospiza", "G.Scandens": "Geospiza",
    "G.Septentrionalist": "Geospiza",
    "P.Crassirostris": "Platyspiza",
    "P2.Inornata": "Pinaroloxias",
    "C.flaveola": "Coereba",
    "E.campestris": "Euneornis",
    "L.noctis": "Loxigilla", "L.portoricensis": "Loxigilla",
    "L.violacea": "Loxigilla",
    "L.anoxanthus": "Loxipasser",
    "M.nigra": "Melopyrrha",
    "T.Bicolor": "Tiaris", "T.canora": "Tiaris", "T.olivacea": "Tiaris",
}

DF_GENERA = ["Camarhynchus", "Certhidea", "Geospiza", "Pinaroloxias", "Platyspiza"]
RE_GENERA = ["Coereba", "Euneornis", "Loxigilla", "Loxipasser", "Melopyrrha", "Tiaris"]



FINCH_TREE = (("Camarhynchus", "Geospiza"),
              ("Platyspiza", ("Certhidea", "Pinaroloxias")))
RELATIVE_TREE = ("Coereba", (("Tiaris", "Melopyrrha"),
                             ("Euneornis", ("Loxipasser", "Loxigilla"))))
TREE = (FINCH_TREE, RELATIVE_TREE)

ORDER = ["Camarhynchus", "Geospiza", "Platyspiza", "Certhidea", "Pinaroloxias",
         "Coereba", "Tiaris", "Melopyrrha", "Euneornis", "Loxipasser", "Loxigilla"]

BLUE, ORANGE = "#3B8DBE", "#E0761F"


def _draw(node, xpos, ax, depth_of):
    if isinstance(node, str):
        return xpos[node], 0.0
    kids = [_draw(c, xpos, ax, depth_of) for c in node]
    y = min(k[1] for k in kids) - 1.0
    for kx, ky in kids:
        ax.plot([kx, kx], [ky, y], color="0.25", lw=1.1,
                solid_capstyle="butt", clip_on=False)
    ax.plot([kids[0][0], kids[-1][0]], [y, y], color="0.25", lw=1.1,
            solid_capstyle="butt", clip_on=False)
    return (kids[0][0] + kids[-1][0]) / 2, y


def fig08(d):
    S = SPEC["fig08"]
    L, Wd = 0.083, 0.910
    fig = plt.figure(figsize=S["figsize"])
    ax = fig.add_axes([L, 1 - 0.606, Wd, 0.606 - 0.012])
    lab_ax = fig.add_axes([L, 1 - 0.833, Wd, 0.833 - 0.550]); lab_ax.axis("off")
    tree_ax = fig.add_axes([L, 1 - 0.979, Wd, 0.979 - 0.854]); tree_ax.axis("off")

    data = [d.loc[d.genus == g, "curvature"].values for g in ORDER]
    colors = [BLUE if g in DF_GENERA else ORANGE for g in ORDER]

    bp = ax.boxplot(data, positions=range(len(ORDER)), widths=0.62,
                    patch_artist=True, whis=1.5, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, c in zip(bp["boxes"], colors):
        fc = blend(c, 0.85) if EPS_SAFE else c
        patch.set_facecolor(fc); patch.set_edgecolor(fc)
        patch.set_alpha(1.0 if EPS_SAFE else 0.85)
    for key in ("whiskers", "caps"):
        for i, art in enumerate(bp[key]):
            art.set_color(colors[i // 2]); art.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=13,
                   **A("0.35", 0.75), zorder=3, linewidths=0)

    XLIM = (-0.7, len(ORDER) - 0.3)
    ax.set_xlim(*XLIM)
    ax.set_ylabel("Orbit Curvature", fontsize=S["label"], labelpad=10)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=S["tick"])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, **{"fc": blend(BLUE, 0.85), "ec": blend(BLUE, 0.85)}
                      if EPS_SAFE else {"fc": BLUE, "ec": BLUE, "alpha": 0.85}),
        plt.Rectangle((0, 0), 1, 1, **{"fc": blend(ORANGE, 0.85), "ec": blend(ORANGE, 0.85)}
                      if EPS_SAFE else {"fc": ORANGE, "ec": ORANGE, "alpha": 0.85})],
        labels=["Darwin's Finches", "Darwin's Finch Relatives"],
        loc="upper right", frameon=False, fontsize=S["legend"])

    lab_ax.set_xlim(*XLIM); lab_ax.set_ylim(0, 1)
    for i, g in enumerate(ORDER):
        lab_ax.text(i, 0.0, g, rotation=90, ha="center", va="bottom",
                    fontsize=S["genus"],
                    color=BLUE if g in DF_GENERA else ORANGE)

    tree_ax.set_xlim(*XLIM)
    _draw(TREE, {g: i for i, g in enumerate(ORDER)}, tree_ax, None)
    tree_ax.set_ylim(-5.4, 0.25)

    save(fig, "FIG_plots_with_data_and_tree_revised3")
    plt.close(fig)



def fig09(d):
    subsets = [("Darwin's finches", d[d.Type == "Darwins Finches"]),
               ("relatives", d[d.Type == "DF Relatives"]),
               ("all", d)]
    cols = [("length_x", "Length (x)"), ("width_y", "Width (y)"),
            ("height_z", "Height (z)")]
    S = SPEC["fig09"]
    fig, axes = plt.subplots(3, 3, figsize=S["figsize"])
    for r, (_, sub) in enumerate(subsets):
        ymax = sub.curvature.max() * 1.5
        for c, (col, lab) in enumerate(cols):
            ax = axes[r, c]
            sns.regplot(ax=ax, data=sub, x=col, y="curvature",
                        ci=None if EPS_SAFE else 95,
                        seed=0,
                        scatter_kws=dict(s=S["marker"], alpha=1.0 if EPS_SAFE else 0.9),
                        line_kws=dict(lw=6))
            ax.set_ylim(0, ymax)
            ax.set_title(f"Curvature vs. {lab.split()[0]}", fontsize=S["title"])
            ax.set_xlabel(lab, fontsize=S["label"])
            ax.set_ylabel("Curvature (1/Radius)", fontsize=S["label"])
            ax.tick_params(labelsize=S["tick"])
    fig.tight_layout(pad=1.2)
    save(fig, "FIG_correlation_revised_v4")
    plt.close(fig)



def fig10(d):
    train = d[d.Model == "Training"]
    TICKS = {0: dict(x=[22, 26, 30, 34, 38], y=[12, 16, 20], z=[12, 16, 20]),
             1: dict(x=[20, 25, 30, 35], y=[12, 16, 20, 24], z=[12, 16, 20])}
    rows = [(train, "(a)", "(b)"), (d, "(c)", "(d)")]
    S = SPEC["fig10"]
    fig = plt.figure(figsize=S["figsize"])
    for r, (sub, la, lb) in enumerate(rows):
        both = np.concatenate([sub.curvature.values,
                               sub.predict_curvature.values])
        vmin, vmax = np.nanquantile(both, 0.02), np.nanquantile(both, 0.98)
        for c, (vals, title, lab) in enumerate(
                [(sub.curvature, "Actual Curvature", la),
                 (sub.predict_curvature, "Predicted Curvature", lb)]):
            ax = fig.add_subplot(2, 2, 2 * r + c + 1, projection="3d")
            if EPS_SAFE:
                for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                    pane.set_pane_color((0.96, 0.96, 0.96, 1.0))
            s = ax.scatter(sub.length_x, sub.width_y, sub.height_z, c=vals,
                           cmap="viridis", s=S["marker"],
                           alpha=1.0 if EPS_SAFE else 0.85,
                           vmin=vmin, vmax=vmax, depthshade=False)
            ax.set_xlabel("Length (x)", fontsize=S["label"], labelpad=45)
            ax.set_ylabel("Width (y)", fontsize=S["label"], labelpad=45)
            ax.set_zlabel("Height (z)", fontsize=S["label"], labelpad=55)
            ax.set_title(title, fontsize=S["title"], pad=2)
            ax.xaxis.set_major_locator(FixedLocator(TICKS[r]["x"]))
            ax.yaxis.set_major_locator(FixedLocator(TICKS[r]["y"]))
            ax.zaxis.set_major_locator(FixedLocator(TICKS[r]["z"]))
            ax.tick_params(labelsize=S["tick"], pad=12)
            ax.text2D(-0.06, 1.00, lab, transform=ax.transAxes,
                      fontsize=S["tag"])
            cb = fig.colorbar(s, ax=ax, shrink=0.55, pad=0.10)
            cb.locator = MultipleLocator(0.025)
            cb.formatter = FormatStrFormatter("%.3f")
            cb.update_ticks()
            cb.solids.set_rasterized(False)
            cb.ax.tick_params(labelsize=S["cbar"])
    fig.tight_layout(pad=1.0)
    save(fig, "FIG_Actual_VS_Model_Curvature_revised3")
    plt.close(fig)



LOXIPASSER = "#17806d"

GENUS_COLOURS = {
    "Camarhynchus": "#1f77b4", "Certhidea": "#d62728", "Geospiza": "#ff7f0e",
    "Pinaroloxias": "#9467bd", "Platyspiza": "#2ca02c",
    "Coereba": "#17becf", "Euneornis": "#7f7f7f", "Loxigilla": "#bcbd22",
    "Loxipasser": LOXIPASSER, "Melopyrrha": "#e377c2", "Tiaris": "#8c564b",
}
PALETTE_MORPHO = PALETTE_NORM = GENUS_COLOURS

GRID = "#d6d6d6"
EDGE = "black"


def _key(g, marker, ms, pal, edge=EDGE):
    return Line2D([], [], marker=marker, ls="", ms=ms, color=pal[g],
                  markeredgecolor=edge, markeredgewidth=ms * 0.09, label=g)


def _genus_legend(ms, pal):
    return ([_key(g, "o", ms, pal) for g in DF_GENERA],
            [_key(g, "^", ms, pal) for g in RE_GENERA])



def fig11(d):
    S = SPEC["fig11"]
    fig, ax = plt.subplots(figsize=S["figsize"])
    for g in DF_GENERA + RE_GENERA:
        s = d[d.genus == g]
        ax.scatter(s.L, s.kappa_tilde, s=S["marker"],
                   marker="o" if g in DF_GENERA else "^",
                   edgecolors=EDGE, linewidths=S["marker"] ** 0.5 * 0.09,
                   label=g, **A(PALETTE_NORM[g], 0.95))
    ax.set_xlabel(r"Characteristic skull size  $L$", fontsize=S["label"])
    ax.set_ylabel(r"Normalized curvature $\tilde{\kappa}$", fontsize=S["label"])
    ax.grid(True, lw=0.8, color=GRID)
    ax.tick_params(labelsize=S["tick"])

    dfh, reh = _genus_legend(S["legend_ms"], PALETTE_NORM)
    handles = ([Line2D([], [], ls="", label="Darwin's Finches")] + dfh +
               [Line2D([], [], ls="", label="DF Relatives")] + reh)
    fig.subplots_adjust(left=0.085, right=0.745, top=0.965, bottom=0.155)
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=S["legend"],
                    frameon=True, handletextpad=0.5, labelspacing=0.30,
                    borderpad=0.45)
    for t in leg.get_texts():
        if t.get_text() in ("Darwin's Finches", "DF Relatives"):
            t.set_fontweight("bold")
    save(fig, "FIG_normalized_by_genus_v3")
    plt.close(fig)



def fig12(d):
    S = SPEC["fig12"]
    fig = plt.figure(figsize=S["figsize"])
    AX_Y, AX_H, AX_W = 1 - 0.647, 0.647 - 0.034, 0.387
    axes = [fig.add_axes([0.091, AX_Y, AX_W, AX_H]),
            fig.add_axes([0.595, AX_Y, AX_W, AX_H])]

    lw_mk = S["marker"] ** 0.5 * 0.09
    for ax, (xc, xlab, tag, tx) in zip(axes, [
            ("kappa_tilde", r"$\tilde{\kappa}$", "(a)", 0.007),
            ("xy", r"$x/y$", "(b)", 0.498)]):
        for g in DF_GENERA + RE_GENERA:
            s = d[d.genus == g]
            c = PALETTE_MORPHO[g]
            if len(s) > 2:
                pts = s[[xc, "cb"]].values
                try:
                    h = ConvexHull(pts)
                    kw = (dict(facecolor=blend(c, 0.15), edgecolor=c, lw=1.6)
                          if EPS_SAFE else
                          dict(facecolor=c, alpha=0.15, edgecolor=c, lw=1.6))
                    ax.add_patch(Polygon(pts[h.vertices], closed=True,
                                         zorder=2, **kw))
                except Exception:
                    pass
            ax.scatter(s[xc], s.cb, s=S["marker"],
                       marker="o" if g in DF_GENERA else "^",
                       edgecolors=EDGE if FIG12_MARKER_EDGE else "none", linewidths=lw_mk, zorder=3,
                       **A(c, 0.95))
        ax.set_xlabel(xlab, fontsize=S["label"])
        ax.set_ylabel(r"$c/b$", fontsize=S["label"])
        ax.grid(True, lw=0.9, color=GRID)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("0.45"); sp.set_linewidth(1.1)
        ax.tick_params(labelsize=S["tick"], length=0)
        fig.text(tx, 0.996, tag, fontsize=S["tag"], va="top")

    ms = S["legend_ms"]
    Wpt = S["figsize"][0] * 72.0
    HL, HTP = 0.49, 0.38
    GAP_COL, GAP_GRP, PAD = 15.0 / Wpt, 40.0 / Wpt, 6.0 / Wpt
    Y_TITLE = 1 - 717.0 / 907.1
    Y_ROWS = 1 - 770.0 / 907.1

    def columns(gens, n=3):
        rows = -(-len(gens) // n)
        return [[gens[r * n + c] for r in range(rows) if r * n + c < len(gens)]
                for c in range(n)]

    groups = [(columns(DF_GENERA), "o", "Darwin's Finches"),
              (columns(RE_GENERA), "^", "Darwin's Finch Relatives")]

    legs, widths = [], []
    for cols, marker, _ in groups:
        gw = []
        for col in cols:
            lg = fig.legend(handles=[_key(g, marker, ms, PALETTE_MORPHO, edge=EDGE if FIG12_MARKER_EDGE else "none")
                                     for g in col],
                            ncol=1, fontsize=S["legend"], frameon=False,
                            handlelength=HL, handletextpad=HTP,
                            labelspacing=0.42, borderpad=0.0,
                            borderaxespad=0.0,
                            loc="upper left", bbox_to_anchor=(0.0, Y_ROWS),
                            bbox_transform=fig.transFigure)
            fig.canvas.draw()
            w = lg.get_window_extent().width / (fig.get_size_inches()[0] * fig.dpi)
            legs.append(lg); gw.append(w)
        widths.append(gw)

    total = [sum(g) + GAP_COL * (len(g) - 1) for g in widths]
    starts = [0.048 + PAD, 0.978 - PAD - total[1]]
    if starts[1] - (starts[0] + total[0]) < 12.0 / Wpt:
        span = total[0] + GAP_GRP + total[1]
        starts = [0.5 - span / 2, 0.5 - span / 2 + total[0] + GAP_GRP]
    k = 0
    for gi, (cols, _, title) in enumerate(groups):
        x = starts[gi]
        for ci in range(len(cols)):
            legs[k].set_bbox_to_anchor((x, Y_ROWS), transform=fig.transFigure)
            x += widths[gi][ci] + GAP_COL
            k += 1
        fig.text(starts[gi] + total[gi] / 2, Y_TITLE, title, ha="center",
                 va="top", fontsize=S["legend"], fontweight="bold")
    print(f"      Fig 12 legend at {S['legend']:.1f} pt; groups span "
          f"{total[0]:.3f} + {total[1]:.3f} of page width")

    if FIG12_LEGEND_FRAME:
        fig.patches.append(plt.Rectangle(
            (0.048, 1 - 0.985), 0.978 - 0.048, 0.985 - 0.781,
            transform=fig.transFigure, facecolor="none", edgecolor="0.15",
            linewidth=1.4, zorder=0))

    save(fig, "FIG_morphospace_2panel_revised_v3")
    plt.close(fig)


HC_RED, CR_BLUE = "#d62728", "#1f77b4"


def _axes_pt(fig, x0, y0, x1, y1):
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])


def _split(tbl, ok):
    if BELOW_CRITERIA_STYLE == "open":
        return tbl[ok], tbl[~ok]
    if BELOW_CRITERIA_STYLE == "filled":
        return tbl, tbl.iloc[0:0]
    return tbl[ok], tbl.iloc[0:0]


def _new_taxa(other):
    new = other[other.folder.isin([_folder_name(HC_DIR), _folder_name(CR_DIR)]) & (other.status == "ok")].copy()
    if new.empty:
        raise Skip("no honeycreeper or relative measurements (see HC_DIR and CR_DIR)")
    new["L"] = (new.length_x * new.width_y * new.height_z) ** (1 / 3)
    new["reliable"] = meets_criteria(new.n_inliers, new.fit_err_pct)
    return new


def figS6_other_taxa(other, d):
    new = _new_taxa(other)
    groups = ((_folder_name(HC_DIR), HC_RED, "Hawaiian honeycreepers", "honeycreepers"),
              (_folder_name(CR_DIR), CR_BLUE, "cardueline relatives", "relatives"))
    fig = plt.figure(figsize=(792.0 / 72, 331.2 / 72))
    a = _axes_pt(fig, 51.5, 24.7, 381.1, 283.8)
    b = _axes_pt(fig, 440.6, 24.7, 770.2, 283.8)
    a.scatter(d.L, d.sphere_radius, s=14.4, c="#a6a6a6", label=f"Darwin's finches and relatives (n={len(d)})", zorder=2)
    for folder, col, name, _ in groups:
        g = new[new.folder == folder]
        full, hollow = _split(g, g.reliable)
        a.scatter(full.L, full.sphere_radius, s=26, c=col, zorder=3, label=f"{name} (n={len(full)})")
        if BELOW_CRITERIA_STYLE == "open":
            a.scatter(hollow.L, hollow.sphere_radius, s=26, facecolors="none", edgecolors=col, lw=1.0,
                      zorder=3, label=f"{name}, below quality criteria (n={len(hollow)})")
    a.set_xlabel("skull size $L = (xyz)^{1/3}$ (mm)"); a.set_ylabel("fitted orbit radius $\\hat{r}$ (mm)")
    a.set_title("(a) orbit radius against skull size", fontsize=10)
    a.legend(fontsize=7, loc="upper left")
    te = d[d.Model == "Test"]
    b.scatter(te.curvature, te.predict_curvature, s=14.4, c="#a6a6a6", label=f"finch test set (n={len(te)})", zorder=2)
    for folder, col, name, _ in groups:
        g = new[new.folder == folder]
        full, hollow = _split(g, g.reliable)
        for part, solid in ((full, True), (hollow, False)):
            if len(part) == 0:
                continue
            meas = 1 / part.sphere_radius
            pred = EQ8[0] + EQ8[1] * part.length_x + EQ8[2] * part.width_y + EQ8[3] * part.height_z
            if solid:
                b.scatter(meas, pred, s=26, c=col, zorder=3, label=name)
            else:
                b.scatter(meas, pred, s=26, facecolors="none", edgecolors=col, lw=1.0, zorder=3)
    b.plot([0.15, 0.5], [0.15, 0.5], "k--", lw=1)
    b.set_xlim(0.15, 0.5); b.set_ylim(0.15, 0.5)
    b.set_xlabel("measured curvature $1/\\hat{r}$ (mm$^{-1}$)"); b.set_ylabel("curvature predicted by Eq. (8) (mm$^{-1}$)")
    b.set_title("(b) finch-trained Eq. (8) applied to the new skulls", fontsize=10)
    b.legend(fontsize=7, loc="upper left")
    save(fig, "FIG_SI_other_taxa")





GALLERY_ROWS = [
    [('HC', 'C. flavaA', 'C. flava A'), ('HC', 'H.wilsoniB', 'H. wilsoni B'),
     ('HC', 'T. cantansC', 'T. cantans C'), ('HC', 'M. phaeosomaA', 'M. phaeosoma A'),
     ('HC', 'L. caeruleirostrisA', 'L. caeruleirostris A'), ('HC', 'P.doleiA', 'P. dolei A')],
    [('CR', 'L. arctoaA', 'L. arctoa A'), ('CR', 'P.pyrrhulaA', 'P. pyrrhula A'),
     ('HC', 'V. coccineaB', 'V. coccinea B'), ('HC', 'V. coccineaD', 'V. coccinea D'),
     ('HC', 'C. stejnegeriD', 'C. stejnegeri D'), ('HC', 'C. virensB', 'C. virens B')],
]
GALLERY_PERO = [('Peromyscus_Gossypinua_watertight', 'P. gossypinus'),
                ('Peromyscus_Simulus_Watertight', 'P. simulus')]
GALLERY_TITLE_SIZE = 8.5
GALLERY_TEXT_SIZE = 8.0
GALLERY_WIRE_LW = 0.3
GALLERY_OUTLINE_LW = 0.6
GALLERY_WIRE_SPACING = 5.0
GALLERY_CELL_PT = 83.0


def _gallery_sphere(ax, c, R, B, extent):
    d_pt = 2 * R / extent * GALLERY_CELL_PT
    n = int(np.clip(round(np.pi * d_pt / GALLERY_WIRE_SPACING), 8, 20))
    draw_wire_ellipsoid(ax, c, [R] * 3, B, lw=GALLERY_WIRE_LW, back_alpha=0.45, wire_n=n)
    cs = to_screen(c, B)[0][0]
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(cs[0] + R * np.cos(t), cs[1] + R * np.sin(t), color=SPHERE_BLUE, lw=GALLERY_OUTLINE_LW, zorder=6)


def _gallery_fit(stl):
    r = run(stl)
    if r.get('status') != 'ok':
        raise RuntimeError(f"no accepted sphere fit for {stl} (status: {r.get('status')})")
    return r


def _beak_to_the_right(axes, V, sgn):
    x = V[:, 0]; xl, xh = np.percentile(x, [2, 98])
    height = lambda sel: np.ptp(V[sel, 2]) if sel.any() else 0.0
    back_high = height(x > xh - 0.1 * (xh - xl)) > height(x < xl + 0.1 * (xh - xl))
    tip = (V[np.argmin(x)] if back_high else V[np.argmax(x)])[0] * sgn
    if tip < (x.min() * sgn + x.max() * sgn) / 2:
        for ax in axes:
            ax.invert_xaxis()


def _gallery_cell(fig, g, folder, stem, label):
    ax = fig.add_subplot(g)
    stl = os.path.join(folder, stem + '.stl')
    if not os.path.isfile(stl):
        raise Skip(f'mesh not found: {stl}')
    r = _gallery_fit(stl); big = load_big(stl)
    c = np.array([r['cx'], r['cy'], r['cz']]); R = r['sphere_radius']; side = np.sign(c[1])
    lateral(ax, big, c, R, r['_inliers'], side, seed=r['_seed'])
    _beak_to_the_right((ax,), big.vertices, 1.0 if side > 0 else -1.0)
    ax.set_title(label, fontsize=GALLERY_TITLE_SIZE, style='italic', pad=1.0)
    ax.text(0.5, -0.03, f"fit error {r['fit_err_pct']:.1f}%", transform=ax.transAxes, ha='center', va='top',
             fontsize=GALLERY_TEXT_SIZE)


HUMAN_PURPLE = "#6a3d9a"


def human_results():
    out = []
    for path in sorted(glob.glob(os.path.join(_p(HUMAN_DIR), "*.stl"))):
        r = _memo(("human", path), lambda path=path: run(path, **HUMAN_FIT))
        if r.get("status") != "ok":
            print(f"    no orbit fit on {os.path.basename(path)}")
            continue
        ext = mesh_of(path).bounding_box.extents
        name = re.sub(r"^human_cranium_|_raw.*$", "", os.path.basename(path)[:-4]).replace("_", " ")
        out.append(dict(r, path=path, name=name, L=float(np.prod(ext) ** (1 / 3))))
    return out


def _gallery_human_cell(fig, spec, h):
    ax = fig.add_subplot(spec)
    m = mesh_of(h["path"]); c = np.array([h["cx"], h["cy"], h["cz"]]); R = h["sphere_radius"]
    side = 1.0 if c[1] >= m.vertices[:, 1].mean() else -1.0
    a = np.radians(35)
    B = camera((-np.cos(a), side * np.sin(a), 0.3), (0, 0, 1))
    V2 = draw_mesh(ax, m, B, alpha=0.55)
    _gallery_sphere(ax, c, R, B, np.ptp(V2, axis=0).max())
    if S4_POINT_SCALE:
        draw_points(ax, h["_inliers"], B, color='#d40000', s=3.5 * S4_POINT_SCALE, zorder=7)
    draw_points(ax, h["_seed"], B, color='#00c000', s=22, edge='k', lw=0.3, zorder=8)
    lo, hi = V2.min(axis=0), V2.max(axis=0); pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect('equal'); ax.set_axis_off()
    if side > 0:
        ax.invert_xaxis()
    ax.set_title("$\\it{H.\\ sapiens}$\n(" + h["name"] + ")", fontsize=GALLERY_TITLE_SIZE, pad=1.0)
    ax.text(0.5, -0.03, f"fit error {h['fit_err_pct']:.1f}%",
            transform=ax.transAxes, ha='center', va='top', fontsize=GALLERY_TEXT_SIZE)


def _gallery_pero_cell(ax, stem, label):
    stl = os.path.join(oriented_peromyscus_dir(), stem + '.stl'); r = _gallery_fit(stl); m = mesh_of(stl)
    c = np.array([r['cx'], r['cy'], r['cz']]); R = r['sphere_radius']
    B = camera((0, 0, 1), (0, -1, 0))
    V2 = draw_mesh(ax, m, B, alpha=0.55)
    _gallery_sphere(ax, c, R, B, np.ptp(V2, axis=0).max())
    if S4_POINT_SCALE:
        draw_points(ax, r['_inliers'], B, color='#d40000', s=3.5 * S4_POINT_SCALE, zorder=7)
    draw_points(ax, r['_seed'], B, color='#00c000', s=22, edge='k', lw=0.3, zorder=8)
    lo, hi = V2.min(axis=0), V2.max(axis=0); pad = 0.04 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect('equal'); ax.set_axis_off()
    tip = to_screen(m.vertices[np.argmax(m.vertices[:, 0])], B)[0][0]
    if tip[0] < (lo[0] + hi[0]) / 2:
        ax.invert_xaxis()
    ax.set_title(label, fontsize=GALLERY_TITLE_SIZE, style='italic', pad=1.0)
    ax.text(0.5, -0.03, f"fit error {r['fit_err_pct']:.1f}%",
            transform=ax.transAxes, ha='center', va='top', fontsize=GALLERY_TEXT_SIZE)


def make_gallery_figure(finch, other):
    folders = {'HC': _p(HC_DIR), 'CR': _p(CR_DIR)}
    humans = human_results()
    fig = plt.figure(figsize=(7.4, 5.0))
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.36, wspace=0.08)

    for ri, row in enumerate(GALLERY_ROWS):
        for ci, (key, stem, label) in enumerate(row):
            print(f"  {label}", flush=True)
            _gallery_cell(fig, gs[ri, ci], folders[key], stem, label)

    for ci, (stem, label) in enumerate(GALLERY_PERO):
        print(f"  {label}", flush=True)
        _gallery_pero_cell(fig.add_subplot(gs[2, ci]), stem, label)
    for ci, h in enumerate(humans[:4]):
        print(f"  H. sapiens ({h['name']})", flush=True)
        _gallery_human_cell(fig, gs[2, 2 + ci], h)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.07)
    save(fig, 'FIG_other_taxa_fits')


def human_pairs():
    out = []
    for h in human_results():
        half = "y>0" if h["cy"] < 0 else "y<0"
        r2 = _memo(("human2", h["path"]), lambda h=h, half=half: run(h["path"], half=half, **HUMAN_FIT))
        if r2.get("status") == "ok":
            out.append((h["sphere_radius"], r2["sphere_radius"]))
    return out


FIG13_GREY = "#8c8c8c"
FIG13_PERO_BROWN = "#8c564b"
FIG13_EXAMPLES = [("Darwin's finch", FIG13_GREY, "FINCH", "G.DifficilisA", "Geospiza difficilis"),
                  ("Hawaiian honeycreeper", HC_RED, "HC", 'L. caeruleirostrisA', "Loxops caeruleirostris"),
                  ("rodent", FIG13_PERO_BROWN, "PERO", "Peromyscus_Simulus_Watertight", "Peromyscus simulus"),
                  ("human cranium", HUMAN_PURPLE, "HUMAN", "BodyParts3D", "Homo sapiens")]
FIG13_DAMAGED = 'L. caeruleirostrisA'
FIG13_WIRE_N = {"FINCH": 16, "HC": 16, "PERO": 12, "HUMAN": 8}
FIG13_WIRE_LW = 0.3


def _f13_title(fig, cell, group, colour, species):
    x = (cell.x0 + cell.x1) / 2
    fig.text(x, cell.y1 + 0.052, group, ha="center", va="bottom", fontsize=9, color=colour, fontweight="bold")
    fig.text(x, cell.y1 + 0.018, species, ha="center", va="bottom", fontsize=8.5, style="italic")


def _f13_caption(fig, cell, r):
    fig.text((cell.x0 + cell.x1) / 2, cell.y0 - 0.012, f"fit error {r['fit_err_pct']:.1f}%",
             ha="center", va="top", fontsize=8)


def _f13_sphere_and_seed(ax, c, R, seed, B, wire_n):
    draw_wire_ellipsoid(ax, c, [R] * 3, B, lw=FIG13_WIRE_LW, back_alpha=0.45, wire_n=wire_n)
    draw_points(ax, seed, B, color="#00c000", s=22, edge="k", lw=0.3, zorder=8)


def _f13_frame(ax, V2, pad_frac=0.04):
    lo, hi = V2.min(axis=0), V2.max(axis=0)
    pad = pad_frac * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal"); ax.set_axis_off()
    return lo, hi


def _f13_bird_cell(ax, folder, stem, wire_n):
    stl = os.path.join(folder, stem + ".stl")
    r = _gallery_fit(stl)
    m = mesh_of(stl)
    V = m.vertices
    c = np.array([r["cx"], r["cy"], r["cz"]])
    B = camera((0, 1.0 if c[1] >= 0 else -1.0, 0), (0, 0, 1))
    V2 = draw_mesh(ax, m, B, alpha=0.55)
    _f13_sphere_and_seed(ax, c, r["sphere_radius"], r["_seed"], B, wire_n)
    lo, hi = _f13_frame(ax, V2)
    x = V[:, 0]; xl, xh = np.percentile(x, [2, 98])
    height = lambda sel: np.ptp(V[sel, 2]) if sel.any() else 0.0
    back_high = height(x > xh - 0.1 * (xh - xl)) > height(x < xl + 0.1 * (xh - xl))
    tip = to_screen(V[np.argmin(x)] if back_high else V[np.argmax(x)], B)[0][0]
    if tip[0] < (lo[0] + hi[0]) / 2:
        ax.invert_xaxis()
    if stem == FIG13_DAMAGED:
        xt = xh - 0.2 * (xh - xl) if back_high else xl + 0.2 * (xh - xl)
        win = np.abs(x - xt) < 0.05 * (xh - xl)
        roof = to_screen(V[np.flatnonzero(win)[np.argmax(V[win, 2])]], B)[0][0]
        ax.annotate("damaged", xy=tuple(roof), xycoords="data", xytext=(0.55, 1.02), textcoords="axes fraction",
                    fontsize=7.5, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k", mutation_scale=6, shrinkA=1, shrinkB=1))
    return r


def _f13_pero_cell(ax, stem, wire_n):
    stl = os.path.join(oriented_peromyscus_dir(), stem + ".stl")
    r = _gallery_fit(stl)
    m = mesh_of(stl)
    c = np.array([r["cx"], r["cy"], r["cz"]])
    B = camera((0, 0, 1), (0, -1, 0))
    V2 = draw_mesh(ax, m, B, alpha=0.55)
    _f13_sphere_and_seed(ax, c, r["sphere_radius"], r["_seed"], B, wire_n)
    lo, hi = _f13_frame(ax, V2)
    tip = to_screen(m.vertices[np.argmax(m.vertices[:, 0])], B)[0][0]
    if tip[0] < (lo[0] + hi[0]) / 2:
        ax.invert_xaxis()
    return r


def _f13_human_cell(ax, h, wire_n):
    m = mesh_of(h["path"])
    c = np.array([h["cx"], h["cy"], h["cz"]])
    side = 1.0 if c[1] >= m.vertices[:, 1].mean() else -1.0
    a = np.radians(35)
    B = camera((-np.cos(a), side * np.sin(a), 0.3), (0, 0, 1))
    V2 = draw_mesh(ax, m, B, alpha=0.55)
    _f13_sphere_and_seed(ax, c, h["sphere_radius"], h["_seed"], B, wire_n)
    _f13_frame(ax, V2, pad_frac=0.05)


def fig13_generalization(finch, other):
    humans = human_results()
    folders = {"FINCH": _p(FINCH_MESH_DIR), "HC": _p(HC_DIR), "CR": _p(CR_DIR)}
    fig = plt.figure(figsize=(7.4, 5.1))
    top = gridspec.GridSpec(1, 4, figure=fig, left=0.02, right=0.99, top=0.84, bottom=0.595, wspace=0.08)
    bot = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.455, bottom=0.105, wspace=0.34,
                            width_ratios=[2.05, 1])

    for i, (group, colour, key, stem, species) in enumerate(FIG13_EXAMPLES):
        ax = fig.add_subplot(top[0, i])
        if key == "HUMAN":
            r = next(h for h in humans if h["name"] == stem)
            _f13_human_cell(ax, r, FIG13_WIRE_N[key])
        elif key == "PERO":
            r = _f13_pero_cell(ax, stem, FIG13_WIRE_N[key])
        else:
            r = _f13_bird_cell(ax, folders[key], stem, FIG13_WIRE_N[key])
        cell = top[0, i].get_position(fig)
        _f13_title(fig, cell, group, colour, species)
        _f13_caption(fig, cell, r)

    axb = fig.add_subplot(bot[0, 0])
    ok = other[other.status == "ok"]
    fin = finch[finch.status == "ok"]
    hcr = ok[ok.folder.isin([_folder_name(HC_DIR), _folder_name(CR_DIR)])]
    pero = ok[ok.filename.str.contains("Peromyscus")]
    hum = pd.DataFrame([{"n_inliers": h["n_inliers"], "fit_err_pct": h["fit_err_pct"]} for h in humans])
    for d, col, lab, z in ((fin, FIG13_GREY, "Darwin's finches and relatives", 1),
                           (hcr, HC_RED, "Hawaiian honeycreepers and relatives", 3),
                           (pero, FIG13_PERO_BROWN, "$\\it{Peromyscus}$ rodents", 5),
                           (hum, HUMAN_PURPLE, "human crania", 4)):
        axb.scatter(d.n_inliers, d.fit_err_pct, s=12, c=col, lw=0, zorder=z, label=f"{lab} ($n$ = {len(d)})")
    axb.axhline(10, ls="--", lw=0.7, c="k", zorder=0)
    axb.set_xlim(0, 175); axb.set_ylim(0, 23)
    axb.set_xlabel("number of inliers", fontsize=9); axb.set_ylabel("fit error (% of radius)", fontsize=9)
    axb.tick_params(labelsize=8)
    axb.legend(fontsize=7.5, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    axb.spines[["top", "right"]].set_visible(False)

    axc = fig.add_subplot(bot[0, 1])
    nt = other[other.folder.isin([_folder_name(HC_DIR), _folder_name(CR_DIR)])]
    nt = nt[~nt.filename.str.contains("caeruleirostris|coccineus")]
    counts = []
    for tbl, col in ((finch, FIG13_GREY), (nt, HC_RED)):
        s = tbl[(tbl.status == "ok") & (tbl.status_second == "ok")]
        s = s[meets_criteria(s.n_inliers_second, s.fit_err_pct_second)]
        counts.append((len(s), col))
        axc.scatter(s.sphere_radius, s.r_second, s=12, c=col, lw=0, zorder=3)
    axc.plot([2, 6], [2, 6], "k--", lw=0.7, zorder=1)
    axc.set_xlim(2, 6.1); axc.set_ylim(2, 6.1)
    axc.set_xticks([2, 3, 4, 5, 6]); axc.set_yticks(np.arange(2, 6.01, 1.0))
    axc.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axc.tick_params(labelsize=8)
    axc.set_xlabel("orbit fitted by the pipeline, $\\hat{r}$ (mm)", fontsize=9)
    axc.set_ylabel("second orbit, fitted\nindependently (mm)", fontsize=9)
    for j, (n, col) in enumerate(counts):
        axc.text(0.97, 0.15 - 0.085 * j, f"$n$ = {n}", transform=axc.transAxes, fontsize=8, color=col,
                 va="bottom", ha="right")
    axc.spines[["top", "right"]].set_visible(False)
    axc.set_aspect("equal", adjustable="box")

    fig.canvas.draw()
    fig.text(0.008, 0.965, "(a)", fontsize=18, fontfamily="serif", va="top")
    for ax, letter in ((axb, "(b)"), (axc, "(c)")):
        p = ax.get_position()
        fig.text(p.x0 - 0.075, p.y1 + 0.045, letter, fontsize=18, fontfamily="serif", va="top")
    print(f"  (b) n = {len(fin)}, {len(hcr)}, {len(pero)}, {len(hum)};  (c) n = {counts[0][0]}, {counts[1][0]}")
    save(fig, "FIG_generalization")


EPS_NAMES = {"FIG_overview_revised": "Fig1", "FIG_bounding": "Fig2", "FIG_orbit_fitting": "Fig3",
             "FIG_sphere_examples": "Fig4", "FIG_difference_curvature": "Fig5", "FIG_ellipsoid_fitting": "Fig6",
             "FIG_ellipsoid_examples": "Fig7", "FIG_plots_with_data_and_tree_revised3": "Fig8",
             "FIG_correlation_revised_v4": "Fig9", "FIG_Actual_VS_Model_Curvature_revised3": "Fig10",
             "FIG_normalized_by_genus_v3": "Fig11", "FIG_morphospace_2panel_revised_v3": "Fig12",
             "FIG_generalization": "Fig13", "FIG_other_taxa_fits": "Fig14"}


def save(fig, stem):
    os.makedirs(_p(OUT_DIR), exist_ok=True)
    if EPS_SAFE:
        path = _out(EPS_NAMES.get(stem, stem) + ".eps")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not support transparency.*")
            fig.savefig(path, format="eps")
    else:
        path = _out(stem + ".pdf")
        fig.savefig(path, format="pdf")
    plt.close(fig)
    WRITTEN.append(path)


def figure_list(finch, other, d):
    R, P, S = COMPOSITE_RC, FIG_RC, {}
    return [("1", "Fig 1", fig01_overview, R), ("2", "Fig 2", fig02_bounding, R),
            ("3", "Fig 3", fig03_orbit_fitting, R), ("4", "Fig 4", fig04_sphere_examples, R),
            ("5", "Fig 5", fig05_difference_curvature, R), ("6", "Fig 6", fig06_ellipsoid_fitting, R),
            ("7", "Fig 7", fig07_ellipsoid_examples, R),
            ("8", "Fig 8", lambda: fig08(d), P), ("9", "Fig 9", lambda: fig09(d), P),
            ("10", "Fig 10", lambda: fig10(d), P), ("11", "Fig 11", lambda: fig11(d), P),
            ("12", "Fig 12", lambda: fig12(d), P), ("13", "Fig 13", lambda: fig13_generalization(finch, other), R),
            ("14", "Fig 14", lambda: make_gallery_figure(finch, other), R),
            ("S1", "Fig S1", figS1_remeshing, R),
            ("S2", "Fig S2", figS8_definitions, S),
            ("S3", "Fig S3", figS5_width_location, S),
            ("S4", "Fig S4", figS2_sphere_examples, R),
            ("S5", "Fig S5", figS3_ellipsoid_examples, R),
            ("S6", "Fig S6", lambda: figS6_other_taxa(other, d), S)]


def main():
    global EPS_SAFE
    ap = argparse.ArgumentParser(description="Make every figure of the paper (vector PDF, optional EPS).")
    ap.add_argument("--only", help="comma-separated figures, e.g. 1,4,9,S2 (default: all)")
    ap.add_argument("--eps", action="store_true", help="also write Fig1.eps ... Fig14.eps (main text, for PLOS)")
    ap.add_argument("--refit", action="store_true", help="measure the meshes again instead of using CACHE_DIR")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("[1/3] measurements from the meshes")
    try:
        finch, other = get_measurements(args.refit)
    except Skip as e:
        sys.exit(f"cannot continue: {e}")
    d = dataset_table(finch)
    print(f"  {len(d)} finch specimens in {d.genus.nunique()} genera, {len(other)} other skulls")
    print("[2/3] the measurements against the numbers printed in the paper")
    check_against_paper(finch, other, d)

    print("[3/3] figures")
    wanted = {k.strip().upper() for k in args.only.split(",")} if args.only else None
    failed, skipped = [], []
    for eps in ([False, True] if args.eps else [False]):
        EPS_SAFE = eps
        for key, name, make, rc in figure_list(finch, other, d):
            if (wanted and key not in wanted) or (eps and key.startswith("S")):
                continue
            print(f"  {name}{' (EPS)' if eps else ''}", flush=True)
            try:
                with plt.rc_context(rc):
                    make()
            except Skip as e:
                skipped.append(f"{name}: {e}")
                print(f"    skipped: {e}")
            except Exception:
                failed.append(name)
                traceback.print_exc()
            finally:
                plt.close("all")
    EPS_SAFE = False

    print("\nWritten:")
    for f in WRITTEN:
        print("  " + os.path.relpath(f, HERE))
    for s in skipped:
        print("SKIPPED " + s)
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
