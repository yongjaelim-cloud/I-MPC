#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline redirection-ratio analysis from a rosbag.

Re-computes, step by step, the same quantities plotter_exp2.py draws:
  * ground-truth k=0 impact normal  (_exact_current_impact_normal_world)
  * minimum distance Phi_GT         (exact signed distance in the ego unit-disk
                                     frame minus 1, nearest convex group)
from /predicted_impact_normal_candidateN (frozen x0 + h=0 geometry), and the
designated sector_id / ego size from /setup (setter.py).

One step = one message on the candidate topic inside [t_start, t_end].

  cand step       : min_distance (Phi_GT) <= d_cmp
  delta condition : dominance-gap condition of the proposed field
                    (utils_linearized / setup_config, Theorem bounds)
                      obstacle gap : phi_(2) - phi_(1) >= delta_o
                      feature gap  : s_(1)  - s_(2)  >= delta_f
                    phi_o = max_f s_f   (exact per-obstacle field value)
                    s_f   = -(n_f . p_f) - 1 - d_safe*||S^-1 n_f||   (q frame,
                            edge + N_n pseudo-normal features, as in
                            collision_lin_cpp build_polygon_features_seq)
                    The feature gap is taken inside the dominant obstacle
                    argmin_o phi_o.  With one obstacle the obstacle gap is inf.
  redirection     : cand AND GT impact normal lies outside the designated sector

Printed (all ratios are over #total steps in the window):
  redirection ratio = #redir / #cand
  delta ratio       = #delta / #total   (delta counted on every step, independent of cand)

Usage:
  python3 redirection_ratio.py BAG T_START T_END D_CMP [options]
  T_START/T_END are seconds relative to bag start (use --abs for epoch time).
"""

import argparse
import math
import os
import sys

import numpy as np
import rosbag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_config import setup_msg_to_raw  # noqa: E402


# --------------------------------------------------------------------------
# Geometry (copied from plotter_exp2.py so the result is identical)
# --------------------------------------------------------------------------
def convex_hull_2d(points):
    pts = sorted(set((float(p[0]), float(p[1])) for p in points
                     if np.isfinite(p[0]) and np.isfinite(p[1])))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 1.0e-12:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 1.0e-12:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def origin_inside_convex_polygon(V):
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    if V.shape[0] < 3:
        return False
    signs = []
    for i in range(V.shape[0]):
        a = V[i]
        b = V[(i + 1) % V.shape[0]]
        e = b - a
        cr = float(e[0] * (-a[1]) - e[1] * (-a[0]))
        if abs(cr) > 1.0e-10:
            signs.append(1 if cr > 0.0 else -1)
    return (not signs) or all(v == signs[0] for v in signs)


def closest_point_from_origin_to_polygon_boundary(V):
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    n = int(V.shape[0])
    if n <= 0:
        return None
    if n == 1:
        return V[0].copy(), float(np.linalg.norm(V[0]))
    best_q, best_d2 = None, float("inf")
    edge_count = n if n >= 3 else 1
    for i in range(edge_count):
        j = (i + 1) % n if n >= 3 else 1
        a, b = V[i], V[j]
        d = b - a
        den = float(np.dot(d, d))
        if den <= 1.0e-24:
            q = a.copy()
        else:
            t = min(max(-float(np.dot(a, d)) / den, 0.0), 1.0)
            q = a + t * d
        d2 = float(np.dot(q, q))
        if d2 < best_d2:
            best_d2, best_q = d2, q.copy()
    if best_q is None:
        return None
    return best_q, math.sqrt(max(best_d2, 0.0))


def seg_seg_dist(p1, p2, q1, q2):
    """Euclidean distance between segments p1p2 and q1q2."""
    def pt_seg(p, a, b):
        d = b - a
        den = float(np.dot(d, d))
        t = 0.0 if den <= 1e-24 else min(max(float(np.dot(p - a, d)) / den, 0.0), 1.0)
        return float(np.linalg.norm(p - (a + t * d)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    if d1 * d2 < 0.0 and d3 * d4 < 0.0:
        return 0.0
    return min(pt_seg(p1, q1, q2), pt_seg(p2, q1, q2),
               pt_seg(q1, p1, p2), pt_seg(q2, p1, p2))


def point_in_convex(p, V):
    return origin_inside_convex_polygon(np.asarray(V, dtype=float) - p)


def rect_polygon_distance(x0, L, W, hull_world):
    """Exact Euclidean distance between the ego rectangle and a convex hull."""
    c, s = math.cos(x0[3]), math.sin(x0[3])
    R = np.array([[c, -s], [s, c]])
    body = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
    rect = body.dot(R.T) + np.asarray(x0[:2]).reshape(1, 2)
    H = np.asarray(hull_world, dtype=float).reshape(-1, 2)
    if H.shape[0] >= 3 and (point_in_convex(rect[0], H) or point_in_convex(H[0], rect)):
        return 0.0
    edges_r = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
    if H.shape[0] == 1:
        edges_h = [(H[0], H[0])]
    elif H.shape[0] == 2:
        edges_h = [(H[0], H[1])]
    else:
        edges_h = [(H[i], H[(i + 1) % H.shape[0]]) for i in range(H.shape[0])]
    return min(seg_seg_dist(a, b, p, q) for a, b in edges_r for p, q in edges_h)


def exact_impact_normal(x0, points, gids, L, W):
    """plotter_exp2._exact_current_impact_normal_world, also returning n in q-frame."""
    pbar = np.asarray(x0[:2], dtype=float)
    yaw = float(x0[3])
    a = max(abs(L) / math.sqrt(2.0), 1.0e-12)
    b = max(abs(W) / math.sqrt(2.0), 1.0e-12)
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c / a, s / a], [-s / b, c / b]], dtype=float)

    best = None
    for gid in np.unique(gids):
        hull_world = convex_hull_2d(points[gids == gid])
        if len(hull_world) < 2:
            continue
        Vw = np.asarray(hull_world, dtype=float).reshape(-1, 2)
        Vl = (Vw - pbar.reshape(1, 2)).dot(T.T)
        hull_local = convex_hull_2d(Vl)
        if len(hull_local) < 2:
            continue
        closest = closest_point_from_origin_to_polygon_boundary(hull_local)
        if closest is None:
            continue
        q_local, radius = closest
        if not math.isfinite(radius) or radius <= 1.0e-10:
            continue
        inside = origin_inside_convex_polygon(hull_local) if len(hull_local) >= 3 else False
        sign = -1.0 if inside else 1.0
        phi_gt = sign * float(radius) - 1.0
        n_local = sign * np.asarray(q_local, dtype=float).reshape(2) / float(radius)
        n_world = T.T @ n_local
        nrm = float(np.linalg.norm(n_world))
        if not math.isfinite(nrm) or nrm <= 1.0e-12:
            continue
        if best is None or phi_gt < best["phi_gt"]:
            best = {
                "group_id": int(gid),
                "phi_gt": float(phi_gt),
                "normal_world": n_world / nrm,
                "normal_q": n_local,  # unit, ego unit-disk frame
                "hull_world": Vw,
            }
    return best


# --------------------------------------------------------------------------
# Proposed-field features and dominance gaps (delta_o / delta_f condition)
# --------------------------------------------------------------------------
def proposed_feature_scores(hull_q, n_pseudo, inward, inv_s, d_safe):
    """Feature scores s_f of one convex obstacle in the q frame.

    Mirrors collision_lin_cpp.build_polygon_features_seq (h=0) followed by the
    feature score in the scaled branch of the proposed-field kernel:
        edges   : point = edge midpoint, normal = edge normal
        corners : N_n pseudo normals interpolated between adjacent edge normals
        s_f = -(n_f . p_f) - 1 - d_safe * ||S^-1 n_f||
    (with outward normals this is the support-plane signed distance minus 1).
    """
    V = np.asarray(hull_q, dtype=float).reshape(-1, 2)  # CCW
    n = V.shape[0]
    if n < 3:
        return np.zeros(0)
    edge_n = []
    for i in range(n):
        e = V[(i + 1) % n] - V[i]
        en = float(np.linalg.norm(e))
        if en <= 1e-12:
            edge_n.append(None)
            continue
        nn = np.array([e[1], -e[0]]) / en  # outward for CCW
        edge_n.append(-nn if inward else nn)
    pts, nrms = [], []
    for i in range(n):
        if edge_n[i] is not None:
            pts.append(0.5 * (V[i] + V[(i + 1) % n]))
            nrms.append(edge_n[i])
    for i in range(n):
        npv, nnv = edge_n[(i - 1) % n], edge_n[i]
        if npv is None or nnv is None:
            continue
        delta = math.atan2(npv[0] * nnv[1] - npv[1] * nnv[0], float(npv @ nnv))
        if delta <= 0.0:
            delta += 2.0 * math.pi
        if not (1e-10 < delta < math.pi - 1e-10):
            continue
        th0 = math.atan2(npv[1], npv[0])
        for j in range(1, n_pseudo + 1):
            th = th0 + j * delta / (n_pseudo + 1)
            pts.append(V[i])
            nrms.append(np.array([math.cos(th), math.sin(th)]))
    P = np.asarray(pts)
    N = np.asarray(nrms)
    # C++: s_f = -(n_f . p_f) - 1 - d_safe/raw_norm with the configured
    # (default inward) normal; raw_norm = 1/||S^-1 n_f||.  Written with the
    # outward normal n_out = -n_in this is the support distance n_out . p - 1.
    n_out = -N if inward else N
    infl = d_safe * np.linalg.norm(N * inv_s.reshape(1, 2), axis=1)
    return np.einsum("ij,ij->i", n_out, P) - 1.0 - infl


def dominance_gaps(x0, points, gids, L, W, n_pseudo, inward, d_safe):
    """Return (obstacle_gap, feature_gap, n_obstacles) at the h=0 state."""
    pbar = np.asarray(x0[:2], dtype=float)
    yaw = float(x0[3])
    a = max(abs(L) / math.sqrt(2.0), 1.0e-12)
    b = max(abs(W) / math.sqrt(2.0), 1.0e-12)
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c / a, s / a], [-s / b, c / b]], dtype=float)
    inv_s = np.array([1.0 / a, 1.0 / b])

    phis, scores_list = [], []
    for gid in np.unique(gids):
        hw = convex_hull_2d(points[gids == gid])
        if len(hw) < 3:
            continue
        hq = convex_hull_2d((np.asarray(hw) - pbar.reshape(1, 2)).dot(T.T))
        sc = proposed_feature_scores(hq, n_pseudo, inward, inv_s, d_safe)
        if sc.size == 0:
            continue
        phis.append(float(np.max(sc)))
        scores_list.append(sc)
    if not phis:
        return float("nan"), float("nan"), 0
    order = np.argsort(phis)
    obs_gap = (phis[order[1]] - phis[order[0]]) if len(phis) >= 2 else float("inf")
    sc = np.sort(scores_list[order[0]])[::-1]
    feat_gap = float(sc[0] - sc[1]) if sc.size >= 2 else float("inf")
    return float(obs_gap), feat_gap, len(phis)


# --------------------------------------------------------------------------
# Sector (utils_linearized.sector_rays_body_from_cfg + q-frame scaling)
# --------------------------------------------------------------------------
def sector_rays_q(L, W, sector_id, sector_dir):
    fl = np.array([L / 2.0, W / 2.0])
    fr = np.array([L / 2.0, -W / 2.0])
    rr = np.array([-L / 2.0, -W / 2.0])
    rl = np.array([-L / 2.0, W / 2.0])
    corners = {1: (fl, rl), 2: (rl, rr), 3: (rr, fr), 4: (fr, fl)}
    la, lb = corners.get(int(sector_id), corners[1])
    if str(sector_dir).strip().lower() == "inward":
        la, lb = -la, -lb
    S = np.array([abs(L) / math.sqrt(2.0), abs(W) / math.sqrt(2.0)])
    la = la / S
    lb = lb / S
    la = la / np.linalg.norm(la)
    lb = lb / np.linalg.norm(lb)
    lc = la + lb
    lc = lc / np.linalg.norm(lc)
    alpha = 0.5 * math.acos(min(max(float(la @ lb), -1.0), 1.0))
    return la, lb, lc, alpha


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def inside_sector(v, la, lb):
    """True if v lies in the (convex, < pi) cone spanned by la -> lb."""
    sgn = 1.0 if cross2(la, lb) >= 0.0 else -1.0
    return sgn * cross2(la, v) >= 0.0 and sgn * cross2(v, lb) >= 0.0


# --------------------------------------------------------------------------
# Bag decoding
# --------------------------------------------------------------------------
def decode_candidate(data):
    d = np.asarray(data, dtype=float).reshape(-1)
    if d.size < 11 or not np.isfinite(d[10]):
        return None
    state_count = int(round(d[10]))
    if state_count < 0:
        return None
    i = 11 + 4 * state_count
    if d.size < i + 5 or not np.isfinite(d[i]):
        return None
    geom_count = int(round(d[i]))
    if geom_count < 0:
        return None
    x0 = d[i + 1:i + 5].copy()
    j = i + 5
    if d.size < j + 3 * geom_count:
        return None
    geom = d[j:j + 3 * geom_count].reshape(-1, 3)
    return {
        "seq_id": int(round(d[1])) if np.isfinite(d[1]) else -1,
        "x0": x0,
        "points": geom[:, :2].copy(),
        "group_ids": np.rint(geom[:, 2]).astype(np.int64),
    }


def setup_params(msg):
    raw = setup_msg_to_raw(msg)
    cfg = raw.get("config", {}) if isinstance(raw.get("config"), dict) else {}
    ego = raw.get("ego", {}) if isinstance(raw.get("ego"), dict) else {}
    mpc = raw.get("mpc", {}) if isinstance(raw.get("mpc"), dict) else {}
    ncfg = raw.get("normal_cfg", {}) if isinstance(raw.get("normal_cfg"), dict) else {}
    sid = int(cfg.get("sector_id", 1))
    if sid not in (1, 2, 3, 4):
        sid = 1
    return {
        "sector_id": sid,
        "sector_dir": str(cfg.get("sector_dir", "outward")),
        "ego_L": float(ego.get("ego_L", 0.0)),
        "ego_W": float(ego.get("ego_W", 0.0)),
        "delta_o": float(mpc.get("delta_o", 0.008)),
        "delta_f": float(mpc.get("delta_f", 0.005)),
        "N_n": int(ncfg.get("N_n", mpc.get("N_n", 5))),
        "d_safe": float(mpc.get("d_safe", 0.3)),
        "inward": str(mpc.get("normal_orientation", "inward")).strip().lower() == "inward",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("t_start", type=float, help="start time [s] (relative to bag start unless --abs)")
    ap.add_argument("t_end", type=float, help="end time [s] (relative to bag start unless --abs)")
    ap.add_argument("d_cmp", type=float, help="min-distance threshold defining cand steps")
    ap.add_argument("--abs", action="store_true", help="t_start/t_end are absolute epoch seconds")
    ap.add_argument("--topic", default="/predicted_impact_normal_candidate0",
                    help="candidate topic defining the steps (default: %(default)s)")
    ap.add_argument("--metric", choices=("phi", "euclid"), default="phi",
                    help="phi: Phi_GT in ego unit-disk frame (plotter value, default); "
                         "euclid: Euclidean distance ego rectangle <-> nearest obstacle hull [m]")
    ap.add_argument("--sector-id", type=int, choices=(1, 2, 3, 4), default=None,
                    help="evaluate against this sector instead of /setup sector_id")
    ap.add_argument("--delta-o", type=float, default=None, help="override delta_o from /setup")
    ap.add_argument("--delta-f", type=float, default=None, help="override delta_f from /setup")
    ap.add_argument("--csv", default=None, help="write per-step results to this CSV")
    args = ap.parse_args()

    bag = rosbag.Bag(args.bag, "r")
    t0_bag = bag.get_start_time()
    if args.abs:
        ts, te = args.t_start, args.t_end
    else:
        ts, te = t0_bag + args.t_start, t0_bag + args.t_end
    if te < ts:
        sys.exit("t_end must be >= t_start")

    # /setup is latched and has only a few messages: load all once.
    setups = [(t.to_sec(), setup_params(m)) for _, m, t in bag.read_messages(topics=["/setup"])]
    setups.sort(key=lambda x: x[0])
    if not setups:
        sys.exit("no /setup message in bag")

    def setup_at(t):
        cur = setups[0][1]
        for st, p in setups:
            if st <= t:
                cur = p
            else:
                break
        return cur

    rows = []
    n_total = n_cand = n_delta = n_redir = n_nogeom = 0
    sectors_seen = set()
    import rospy  # only for Time
    for _, msg, t in bag.read_messages(topics=[args.topic],
                                       start_time=rospy.Time.from_sec(ts),
                                       end_time=rospy.Time.from_sec(te)):
        tt = t.to_sec()
        n_total += 1
        p = dict(setup_at(tt))
        if args.sector_id is not None:
            p["sector_id"] = args.sector_id
        sectors_seen.add(p["sector_id"])
        L, W = p["ego_L"], p["ego_W"]
        la, lb, lc, alpha = sector_rays_q(L, W, p["sector_id"], p["sector_dir"])

        dec = decode_candidate(msg.data)
        gt = None
        if (dec is not None and dec["x0"].size >= 4 and np.all(np.isfinite(dec["x0"]))
                and dec["points"].shape[0] >= 2):
            gt = exact_impact_normal(dec["x0"], dec["points"], dec["group_ids"], L, W)

        if gt is None:
            n_nogeom += 1
            rows.append((tt - t0_bag, p["sector_id"], float("nan"), float("nan"),
                         float("nan"), float("nan"), 0, float("nan"), float("nan"), 0, 0, 0))
            continue

        if args.metric == "phi":
            dist = gt["phi_gt"]
        else:
            dist = rect_polygon_distance(dec["x0"], L, W, gt["hull_world"])
            # nearest group in Euclidean sense
            for gid in np.unique(dec["group_ids"]):
                hull = convex_hull_2d(dec["points"][dec["group_ids"] == gid])
                if len(hull) >= 1:
                    dist = min(dist, rect_polygon_distance(dec["x0"], L, W, hull))

        nq = gt["normal_q"]
        in_sec = inside_sector(nq, la, lb)
        ang = math.degrees(math.acos(min(max(float(nq @ lc), -1.0), 1.0)))
        d_o = args.delta_o if args.delta_o is not None else p["delta_o"]
        d_f = args.delta_f if args.delta_f is not None else p["delta_f"]
        obs_gap, feat_gap, _ = dominance_gaps(dec["x0"], dec["points"], dec["group_ids"],
                                              L, W, p["N_n"], p["inward"], p["d_safe"])
        delta_ok = (obs_gap >= d_o) and (feat_gap >= d_f)
        cand = dist <= args.d_cmp
        redir = cand and (not in_sec)
        n_cand += int(cand)
        n_delta += int(delta_ok)
        n_redir += int(redir)
        rows.append((tt - t0_bag, p["sector_id"], dist, gt["phi_gt"], ang,
                     math.degrees(alpha), int(in_sec), obs_gap, feat_gap,
                     int(cand), int(delta_ok), int(redir)))
    bag.close()

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("t_rel,sector_id,min_dist,phi_gt,angle_to_lc_deg,alpha_deg,"
                    "normal_in_sector,obstacle_gap,feature_gap,cand,delta_condition,redirected\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

    def ratio(a, b):
        return (float(a) / float(b)) if b > 0 else float("nan")

    p0 = setup_at(ts)
    d_o = args.delta_o if args.delta_o is not None else p0["delta_o"]
    d_f = args.delta_f if args.delta_f is not None else p0["delta_f"]
    print("bag            : %s" % args.bag)
    print("topic          : %s" % args.topic)
    print("interval       : [%.3f, %.3f] s (rel. to bag start)" % (ts - t0_bag, te - t0_bag))
    print("sector_id(s)   : %s" % sorted(sectors_seen))
    print("metric / d_cmp : %s / %.4f" % (args.metric, args.d_cmp))
    print("delta_o/delta_f: %.4g / %.4g   (N_n=%d, d_safe=%.3f)"
          % (d_o, d_f, p0["N_n"], p0["d_safe"]))
    print("total steps    : %d   (no GT normal / no geometry: %d)" % (n_total, n_nogeom))
    print("cand steps     : %d   (min_dist <= d_cmp)" % n_cand)
    print("delta steps    : %d   (obstacle_gap>=delta_o AND feature_gap>=delta_f)" % n_delta)
    print("redirect steps : %d   (cand AND GT normal outside sector)" % n_redir)
    print("-" * 60)
    print("redirection ratio (redirect / cand)  : %.4f" % ratio(n_redir, n_cand))
    print("delta ratio       (delta / total)    : %.4f" % ratio(n_delta, n_total))

if __name__ == "__main__":
    main()
