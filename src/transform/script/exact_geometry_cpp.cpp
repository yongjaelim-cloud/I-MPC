#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kTol = 1.0e-10;
constexpr double kNormalEps = 1.0e-10;

struct Vec2 {
    double x{0.0};
    double y{0.0};
};

inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(double s, const Vec2& a) { return {s * a.x, s * a.y}; }
inline double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline double cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }
inline double norm2(const Vec2& a) { return dot(a, a); }
inline double norm(const Vec2& a) { return std::sqrt(std::max(0.0, norm2(a))); }
inline bool finite(const Vec2& a) { return std::isfinite(a.x) && std::isfinite(a.y); }

struct ClosestResult {
    double distance{std::numeric_limits<double>::infinity()};
    Vec2 ellipse_point{};
    Vec2 obstacle_point{};
    bool overlap{false};
};

inline double point_segment_distance(const Vec2& p, const Vec2& a, const Vec2& b, Vec2* q_out = nullptr) {
    const Vec2 d = b - a;
    const double den = norm2(d);
    double t = 0.0;
    if (den > 1.0e-24) {
        t = dot(p - a, d) / den;
        t = std::max(0.0, std::min(1.0, t));
    }
    const Vec2 q = a + t * d;
    if (q_out) *q_out = q;
    return norm(p - q);
}

std::vector<Vec2> convex_hull(std::vector<Vec2> pts) {
    pts.erase(std::remove_if(pts.begin(), pts.end(), [](const Vec2& p) { return !finite(p); }), pts.end());
    if (pts.size() <= 1) return pts;
    std::sort(pts.begin(), pts.end(), [](const Vec2& a, const Vec2& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    std::vector<Vec2> uniq;
    uniq.reserve(pts.size());
    for (const auto& p : pts) {
        if (uniq.empty() || std::abs(p.x - uniq.back().x) > 1.0e-12 || std::abs(p.y - uniq.back().y) > 1.0e-12) {
            uniq.push_back(p);
        }
    }
    if (uniq.size() <= 2) return uniq;

    std::vector<Vec2> lower, upper;
    for (const auto& p : uniq) {
        while (lower.size() >= 2) {
            const Vec2 a = lower[lower.size() - 2];
            const Vec2 b = lower[lower.size() - 1];
            if (cross(b - a, p - a) > 1.0e-12) break;
            lower.pop_back();
        }
        lower.push_back(p);
    }
    for (auto it = uniq.rbegin(); it != uniq.rend(); ++it) {
        const Vec2 p = *it;
        while (upper.size() >= 2) {
            const Vec2 a = upper[upper.size() - 2];
            const Vec2 b = upper[upper.size() - 1];
            if (cross(b - a, p - a) > 1.0e-12) break;
            upper.pop_back();
        }
        upper.push_back(p);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

bool point_inside_convex_polygon(const Vec2& p, const std::vector<Vec2>& poly) {
    if (poly.size() < 3) return false;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Vec2 a = poly[i];
        const Vec2 b = poly[(i + 1) % poly.size()];
        if (cross(b - a, p - a) < -kTol) return false;
    }
    return true;
}

inline bool point_inside_ellipse(const Vec2& p, double a, double b) {
    const double v = (p.x * p.x) / (a * a) + (p.y * p.y) / (b * b);
    return v <= 1.0 + kTol;
}

bool segment_intersects_ellipse(const Vec2& A, const Vec2& B, double a, double b) {
    const Vec2 D = B - A;
    const double aa = a * a;
    const double bb = b * b;
    const double q2 = (D.x * D.x) / aa + (D.y * D.y) / bb;
    const double q1 = 2.0 * (A.x * D.x / aa + A.y * D.y / bb);
    const double q0 = (A.x * A.x) / aa + (A.y * A.y) / bb - 1.0;
    if (std::abs(q2) <= 1.0e-24) {
        if (std::abs(q1) <= 1.0e-24) return std::abs(q0) <= kTol;
        const double t = -q0 / q1;
        return t >= -kTol && t <= 1.0 + kTol;
    }
    double disc = q1 * q1 - 4.0 * q2 * q0;
    if (disc < -kTol) return false;
    disc = std::max(0.0, disc);
    const double sd = std::sqrt(disc);
    const double t1 = (-q1 - sd) / (2.0 * q2);
    const double t2 = (-q1 + sd) / (2.0 * q2);
    return (t1 >= -kTol && t1 <= 1.0 + kTol) || (t2 >= -kTol && t2 <= 1.0 + kTol);
}

// Closest point on an axis-aligned ellipse to a point known to be outside it.
Vec2 closest_point_on_ellipse_external(const Vec2& p, double a, double b) {
    const double aa = a * a;
    const double bb = b * b;
    auto f = [&](double t) {
        const double tx = a * p.x / (t + aa);
        const double ty = b * p.y / (t + bb);
        return tx * tx + ty * ty - 1.0;
    };

    double lo = 0.0;
    double hi = std::max({aa, bb, std::abs(a * p.x), std::abs(b * p.y), 1.0});
    while (f(hi) > 0.0 && hi < 1.0e30) hi *= 2.0;
    for (int i = 0; i < 72; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) lo = mid;
        else hi = mid;
    }
    const double t = 0.5 * (lo + hi);
    return {aa * p.x / (t + aa), bb * p.y / (t + bb)};
}

void consider_pair(const Vec2& e, const Vec2& q, ClosestResult& best) {
    const double d = norm(e - q);
    if (d < best.distance) {
        best.distance = d;
        best.ellipse_point = e;
        best.obstacle_point = q;
    }
}

ClosestResult ellipse_polygon_closest(const std::vector<Vec2>& polygon_raw, double a, double b) {
    ClosestResult best;
    const std::vector<Vec2> V = convex_hull(polygon_raw);
    const size_t n = V.size();
    if (n == 0) return best;

    // Set-overlap tests: these make the occupancy clearance exactly zero.
    if (n >= 3 && point_inside_convex_polygon({0.0, 0.0}, V)) {
        best.distance = 0.0;
        best.overlap = true;
        return best;
    }
    for (const auto& p : V) {
        if (point_inside_ellipse(p, a, b)) {
            best.distance = 0.0;
            best.overlap = true;
            return best;
        }
    }
    const size_t edge_count = (n >= 3 ? n : (n == 2 ? 1 : 0));
    for (size_t i = 0; i < edge_count; ++i) {
        const Vec2 A = V[i];
        const Vec2 B = V[(i + 1) % n];
        if (segment_intersects_ellipse(A, B, a, b)) {
            best.distance = 0.0;
            best.overlap = true;
            return best;
        }
    }

    // Polygon vertices against the full continuous ellipse.
    for (const auto& q : V) {
        const Vec2 e = closest_point_on_ellipse_external(q, a, b);
        consider_pair(e, q, best);
    }

    // Interior edge-projection KKT candidates. At a closest pair with the
    // obstacle point inside an edge, the ellipse normal is perpendicular to
    // that edge. The support point for +/- edge normal gives both candidates.
    for (size_t i = 0; i < edge_count; ++i) {
        const Vec2 A = V[i];
        const Vec2 B = V[(i + 1) % n];
        const Vec2 D = B - A;
        const double len2 = norm2(D);
        if (len2 <= 1.0e-24) continue;
        const double len = std::sqrt(len2);
        const Vec2 nrm{-D.y / len, D.x / len};
        const double den = std::sqrt((a * nrm.x) * (a * nrm.x) + (b * nrm.y) * (b * nrm.y));
        if (den <= 1.0e-24) continue;
        const Vec2 e_plus{a * a * nrm.x / den, b * b * nrm.y / den};
        for (int sign : {1, -1}) {
            const Vec2 e = (sign > 0 ? e_plus : (-1.0) * e_plus);
            const double t = dot(e - A, D) / len2;
            if (t < -kTol || t > 1.0 + kTol) continue;
            const double tc = std::max(0.0, std::min(1.0, t));
            const Vec2 q = A + tc * D;
            consider_pair(e, q, best);
        }
    }

    return best;
}

struct GroupTemplate {
    int id{0};
    std::vector<int> indices;
    bool dynamic{false};
};

}  // namespace

extern "C" int exact_geometry_cpp_api() {
    return 1;
}

extern "C" void exact_geometry_set_omp_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// Output per horizon stage (6 doubles):
// [0] exact clearance to perceived obstacle after dynamic uncertainty inflation
// [1] impact normal x in ego body frame
// [2] impact normal y in ego body frame
// [3] normal-valid flag (1.0 valid, 0.0 undefined at overlap/no obstacle)
// [4] geometric (uninflated) ellipse-polygon clearance
// [5] selected obstacle/group id, -1 when none
extern "C" int evaluate_exact_clearance_impact_seq(
    const double* X, int H,
    const double* vertices, int K, int vertices_stagewise,
    const int32_t* vertex_types, int has_vertex_types,
    const int32_t* obstacle_ids, int has_obstacle_ids,
    const double* h_steps,
    const double* dt_vals,
    double ellipse_a,
    double ellipse_b,
    double sigma_vx,
    double sigma_vy,
    double probability_collision,
    int use_probability_margin,
    double far_value,
    double* out6
) {
    if (!X || !vertices || !h_steps || !dt_vals || !out6 || H < 0 || K < 0) return -1;
    if (!(ellipse_a > 0.0) || !(ellipse_b > 0.0)) return -2;

    std::map<int, GroupTemplate> group_map;
    for (int i = 0; i < K; ++i) {
        const int gid = has_obstacle_ids ? static_cast<int>(obstacle_ids[i]) : 0;
        if (gid < 0) continue;
        auto& g = group_map[gid];
        g.id = gid;
        g.indices.push_back(i);
        if (has_vertex_types && vertex_types[i] == 1) g.dynamic = true;
    }
    std::vector<GroupTemplate> groups;
    groups.reserve(group_map.size());
    for (auto& kv : group_map) groups.push_back(std::move(kv.second));

    const double sigma_iso = std::max(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));
    const double pc = std::max(0.0, std::min(probability_collision, 1.0 - 1.0e-12));
    const double chance_scale = std::sqrt(std::max(-2.0 * std::log(1.0 - pc), 0.0));
    const double far = (std::isfinite(far_value) && far_value > 0.0) ? far_value : 1.0e15;

    #pragma omp parallel for schedule(static) if(H > 2)
    for (int h = 0; h < H; ++h) {
        const double px = X[4 * h + 0];
        const double py = X[4 * h + 1];
        const double yaw = X[4 * h + 3];
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);

        double best_clearance = far;
        double best_geom = far;
        Vec2 best_normal{0.0, 0.0};
        bool best_valid = false;
        int best_gid = -1;

        for (const auto& g : groups) {
            std::vector<Vec2> body_pts;
            body_pts.reserve(g.indices.size());
            for (int idx : g.indices) {
                const int base = vertices_stagewise ? (2 * (h * K + idx)) : (2 * idx);
                const double wx = vertices[base + 0];
                const double wy = vertices[base + 1];
                if (!std::isfinite(wx) || !std::isfinite(wy)) continue;
                const double dx = wx - px;
                const double dy = wy - py;
                // R(yaw)^T * (p_world - ego_pos)
                body_pts.push_back({c * dx + s * dy, -s * dx + c * dy});
            }
            if (body_pts.empty()) continue;

            const ClosestResult cr = ellipse_polygon_closest(body_pts, ellipse_a, ellipse_b);
            if (!std::isfinite(cr.distance)) continue;

            double margin = 0.0;
            if (use_probability_margin && g.dynamic) {
                const double hstep = std::max(h_steps[h], 0.0);
                const double dt = std::max(dt_vals[h], 0.0);
                margin = hstep * dt * sigma_iso * chance_scale;
            }
            const double clearance = cr.distance - margin;
            if (clearance >= best_clearance) continue;

            best_clearance = clearance;
            best_geom = cr.distance;
            best_gid = g.id;
            best_valid = false;
            best_normal = {0.0, 0.0};
            if (!cr.overlap && cr.distance > kNormalEps) {
                // Exact ego impact normal: from ego boundary toward obstacle.
                const Vec2 d = cr.obstacle_point - cr.ellipse_point;
                const double dn = norm(d);
                if (std::isfinite(dn) && dn > kNormalEps) {
                    best_normal = (1.0 / dn) * d;
                    best_valid = true;
                }
            }
        }

        out6[6 * h + 0] = best_clearance;
        out6[6 * h + 1] = best_normal.x;
        out6[6 * h + 2] = best_normal.y;
        out6[6 * h + 3] = best_valid ? 1.0 : 0.0;
        out6[6 * h + 4] = best_geom;
        out6[6 * h + 5] = static_cast<double>(best_gid);
    }
    return 0;
}
