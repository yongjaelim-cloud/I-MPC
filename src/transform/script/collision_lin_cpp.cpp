#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "collision_lin_cuda.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {
constexpr int D = 6;
constexpr double EPS = 1.0e-12;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr long long COLLISION_OMP_WORK_THRESHOLD = 8192LL;
constexpr ssize_t RK4_OMP_HORIZON_THRESHOLD = 256;

// Native CUDA policy for proposed field_flag=4.  The Python layer only sets
// these lightweight knobs; all GPU allocation, kernels, reductions and the
// final D2H copy live inside collision_lin_cpp/native CUDA.
std::atomic<bool> PROPOSED_CUDA_ENABLED{true};
std::atomic<long long> PROPOSED_CUDA_MIN_HF{1024LL};
std::atomic<int> PROPOSED_CUDA_THREADS{128};
std::atomic<bool> PROPOSED_CUDA_FALLBACK_TO_CPU{true};
std::atomic<bool> PROPOSED_CUDA_PROFILE{false};
std::atomic<bool> PROPOSED_CUDA_COMPARE_CPU{false};
std::atomic<int> PROPOSED_CUDA_PROFILE_EVERY{1};
std::atomic<unsigned long long> PROPOSED_CUDA_CALL_COUNT{0ULL};

// Native CUDA policy for BP-SDF field_flag=2. Fitting remains CPU/cached;
// repeated metric-SDF/gradient queries and collision linearization run on GPU.
std::atomic<bool> BP_CUDA_ENABLED{true};
std::atomic<long long> BP_CUDA_MIN_HO{16LL};
std::atomic<int> BP_CUDA_THREADS{32};
std::atomic<bool> BP_CUDA_FALLBACK_TO_CPU{true};
std::atomic<bool> BP_CUDA_PROFILE{false};
std::atomic<bool> BP_CUDA_COMPARE_CPU{false};
std::atomic<int> BP_CUDA_PROFILE_EVERY{1};
std::atomic<unsigned long long> BP_CUDA_CALL_COUNT{0ULL};

inline double clampd(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

inline double elapsed_ms(std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline double max_abs_array_diff(const py::array_t<double>& a, const py::array_t<double>& b) {
    const auto ai = a.request();
    const auto bi = b.request();
    if (ai.size != bi.size) return std::numeric_limits<double>::infinity();
    const double* ap = static_cast<const double*>(ai.ptr);
    const double* bp = static_cast<const double*>(bi.ptr);
    double m = 0.0;
    for (ssize_t i = 0; i < ai.size; ++i) m = std::max(m, std::abs(ap[i] - bp[i]));
    return m;
}

inline double wrap_to_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

struct MatView2D {
    const double* p = nullptr;
    ssize_t r = 0;
    ssize_t c = 0;
    ssize_t s0 = 0;
    ssize_t s1 = 0;
    inline double operator()(ssize_t i, ssize_t j) const {
        return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(p) + i * s0 + j * s1);
    }
};

// A feature tensor may be shared by all horizon stages (K x 2) or supplied
// stage-wise (H x K x 2).  The latter is used for predicted/inflated obstacle
// geometry published by vertex_selector.py.
struct FeatureAccessor {
    const double* p = nullptr;
    ssize_t H = 0;
    ssize_t K = 0;
    ssize_t s0 = 0;
    ssize_t s1 = 0;
    ssize_t s2 = 0;
    bool stagewise = false;

    inline double get(ssize_t h, ssize_t k, ssize_t j) const {
        const char* base = reinterpret_cast<const char*>(p);
        if (stagewise) {
            const ssize_t hh = (H == 1) ? 0 : h;
            return *reinterpret_cast<const double*>(base + hh * s0 + k * s1 + j * s2);
        }
        return *reinterpret_cast<const double*>(base + k * s0 + j * s1);
    }
};

FeatureAccessor make_feature_accessor(
    const py::array_t<double, py::array::forcecast>& a,
    ssize_t expected_H)
{
    auto b = a.request();
    FeatureAccessor out;
    out.p = static_cast<const double*>(b.ptr);
    if (b.ndim == 2) {
        if (b.shape[1] < 2) throw std::runtime_error("feature array must have shape Kx2 or HxKx2");
        out.H = expected_H;
        out.K = b.shape[0];
        out.s0 = b.strides[0];
        out.s1 = b.strides[1];
        out.stagewise = false;
        return out;
    }
    if (b.ndim == 3) {
        if (b.shape[2] < 2) throw std::runtime_error("stage-wise feature array must have shape HxKx2");
        if (b.shape[0] != 1 && expected_H > 0 && b.shape[0] != expected_H) {
            throw std::runtime_error("stage-wise feature horizon does not match X horizon");
        }
        out.H = b.shape[0];
        out.K = b.shape[1];
        out.s0 = b.strides[0];
        out.s1 = b.strides[1];
        out.s2 = b.strides[2];
        out.stagewise = true;
        return out;
    }
    throw std::runtime_error("feature array must be 2-D Kx2 or 3-D HxKx2");
}

MatView2D view2d(const py::array_t<double, py::array::forcecast>& a) {
    auto b = a.request();
    if (b.ndim != 2) throw std::runtime_error("expected a 2-D float64 array");
    return {static_cast<const double*>(b.ptr), b.shape[0], b.shape[1], b.strides[0], b.strides[1]};
}

struct VecViewD {
    const double* p = nullptr;
    ssize_t n = 0;
    ssize_t stride = 0;
    inline double operator()(ssize_t i) const {
        return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(p) + i * stride);
    }
};

VecViewD view1d_double(const py::array_t<double, py::array::forcecast>& a) {
    auto b = a.request();
    if (b.ndim != 1) throw std::runtime_error("expected a 1-D float64 array");
    return {static_cast<const double*>(b.ptr), b.shape[0], b.strides[0]};
}

struct VecViewI {
    const int32_t* p = nullptr;
    ssize_t n = 0;
    ssize_t stride = 0;
    inline int32_t operator()(ssize_t i) const {
        return *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(p) + i * stride);
    }
};

VecViewI view1d_int(const py::array_t<int32_t, py::array::forcecast>& a) {
    auto b = a.request();
    if (b.ndim != 1) throw std::runtime_error("expected a 1-D int32 array");
    return {static_cast<const int32_t*>(b.ptr), b.shape[0], b.strides[0]};
}

struct StateAccessor {
    MatView2D X;
    bool state_by_stage = true;
    ssize_t H = 0;
    inline double get(ssize_t h, ssize_t j) const {
        return state_by_stage ? X(h, j) : X(j, h);
    }
};

StateAccessor make_state_accessor(const py::array_t<double, py::array::forcecast>& Xarr) {
    MatView2D X = view2d(Xarr);
    StateAccessor out;
    out.X = X;
    if (X.c == 4) {
        out.state_by_stage = true;
        out.H = X.r;
    } else if (X.c == 6 && X.r != 4 && X.r != 6) {
        out.state_by_stage = true;
        out.H = X.r;
    } else if (X.r == 4 || X.r == 6) {
        out.state_by_stage = false;
        out.H = X.c;
    } else if (X.c >= 4) {
        out.state_by_stage = true;
        out.H = X.r;
    } else {
        throw std::runtime_error("X must be Hx4/Hx6 or 4xH/6xH");
    }
    return out;
}

inline void normalize2(double& x, double& y) {
    const double n = std::hypot(x, y);
    if (n > EPS) {
        x /= n;
        y /= n;
    } else {
        x = 1.0;
        y = 0.0;
    }
}

struct FeatureValue {
    double score = 0.0;
    std::array<double, D> dscore{};
};

struct GroupLayout {
    std::vector<int> offsets;
    std::vector<int> indices;
    bool singleton_groups = true;

    int group_count() const {
        return offsets.empty() ? 0 : static_cast<int>(offsets.size()) - 1;
    }
};

GroupLayout make_group_layout(
    const std::vector<int32_t>& obs_ids,
    bool has_obs_ids,
    int K)
{
    GroupLayout layout;
    if (!has_obs_ids) {
        layout.offsets.resize(static_cast<size_t>(K) + 1U);
        layout.indices.resize(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) {
            layout.offsets[static_cast<size_t>(k)] = k;
            layout.indices[static_cast<size_t>(k)] = k;
        }
        layout.offsets[static_cast<size_t>(K)] = K;
        layout.singleton_groups = true;
        return layout;
    }

    std::unordered_map<int32_t, int> id_to_group;
    id_to_group.reserve(static_cast<size_t>(K) * 2U);
    std::vector<std::vector<int>> groups;
    groups.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const int32_t id = obs_ids[static_cast<size_t>(k)];
        auto it = id_to_group.find(id);
        int group = 0;
        if (it == id_to_group.end()) {
            group = static_cast<int>(groups.size());
            id_to_group.emplace(id, group);
            groups.emplace_back();
        } else {
            group = it->second;
        }
        groups[static_cast<size_t>(group)].push_back(k);
    }

    layout.offsets.reserve(groups.size() + 1U);
    layout.indices.reserve(static_cast<size_t>(K));
    layout.offsets.push_back(0);
    layout.singleton_groups = true;
    for (const auto& group : groups) {
        if (group.size() != 1U) layout.singleton_groups = false;
        layout.indices.insert(layout.indices.end(), group.begin(), group.end());
        layout.offsets.push_back(static_cast<int>(layout.indices.size()));
    }
    return layout;
}


struct RawPolygonGroup {
    int32_t id = 0;
    int32_t type = 0;
    std::vector<int> indices;
    std::vector<unsigned char> valid_corner;
};

inline double cross2(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

// Convert ordered convex-polygon vertices into the paper's finite feature set.
//
// IMPORTANT: pseudo-vertex normals are sampled uniformly in the ego body-fixed
// normalized frame q = S^{-1} R(yaw)^T (P-p), not in the world frame.  This is
// required because the anisotropic scale S^{-1} does not preserve angles.
//
// The collision kernels downstream still accept world-frame feature normals and
// internally map them with S R^T.  Therefore, after constructing a desired
// local normal n_q, this routine returns the world-equivalent covector
//
//      n_w \propto R(yaw) S^{-1} n_q,
//
// so that the existing downstream transform recovers exactly n_q (up to
// normalization).  Feature points remain in world coordinates because affine
// transforms preserve edge midpoints and polygon vertices.
py::tuple build_polygon_features_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types,
    bool has_feature_obs_indices,
    int vertex_capacity,
    int pseudo_normal_num,
    const std::string& normal_orientation,
    int expected_H)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = std::max<ssize_t>(expected_H, 0);
    if (X.H != H && X.H != 1) {
        throw std::runtime_error("X horizon does not match expected_H in build_polygon_features_seq");
    }
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const VecViewI types = view1d_int(vertex_types);
    const VecViewI obs = view1d_int(feature_obs_indices);
    const VecViewD scales = view1d_double(axis_scale);
    if (scales.n < 2) throw std::runtime_error("axis_scale must contain [sx, sy]");

    const double sx = std::max(std::abs(scales(0)), EPS);
    const double sy = std::max(std::abs(scales(1)), EPS);
    const double isx = 1.0 / sx;
    const double isy = 1.0 / sy;
    const int K = std::max(0, std::min(vertex_capacity, static_cast<int>(V.K)));
    const int Nn = std::max(pseudo_normal_num, 0);
    const bool inward = normal_orientation == "inward";

    auto state_yaw = [&](ssize_t h) {
        const ssize_t hh = X.H == 1 ? 0 : h;
        return X.get(hh, 3);
    };

    // Transform a world edge vector into the body-fixed normalized frame and
    // return its configured (inward/outward) unit normal there.
    auto local_edge_normal = [&](double exw, double eyw, double yaw) {
        const double c = std::cos(yaw), s = std::sin(yaw);
        const double exq = isx * (c * exw + s * eyw);
        const double eyq = isy * (-s * exw + c * eyw);
        const double en = std::hypot(exq, eyq);
        std::array<double,2> n{{0.0, 0.0}};
        if (en <= EPS) return n;
        // CCW polygon in q-frame: [ey,-ex] is outward.
        n[0] = eyq / en;
        n[1] = -exq / en;
        if (inward) { n[0] = -n[0]; n[1] = -n[1]; }
        return n;
    };

    // Convert a configured q-frame unit normal to a world-frame covector that
    // the existing collision kernel maps back to the same q-frame direction.
    auto local_normal_to_world = [&](double nqx, double nqy, double yaw) {
        const double c = std::cos(yaw), s = std::sin(yaw);
        const double tx = isx * nqx;
        const double ty = isy * nqy;
        double nwx = c * tx - s * ty;
        double nwy = s * tx + c * ty;
        normalize2(nwx, nwy);
        return std::array<double,2>{{nwx, nwy}};
    };

    std::vector<RawPolygonGroup> groups;
    std::unordered_map<int32_t, size_t> lookup;
    lookup.reserve(static_cast<size_t>(std::max(K, 1)) * 2U);
    for (int k = 0; k < K; ++k) {
        const int32_t oid = has_feature_obs_indices && k < obs.n ? obs(k) : 0;
        auto it = lookup.find(oid);
        size_t gi;
        if (it == lookup.end()) {
            gi = groups.size();
            lookup.emplace(oid, gi);
            RawPolygonGroup g;
            g.id = oid;
            g.type = has_vertex_types && k < types.n ? types(k) : 0;
            groups.push_back(std::move(g));
        } else {
            gi = it->second;
            if (has_vertex_types && k < types.n && types(k) == 1) groups[gi].type = 1;
        }
        groups[gi].indices.push_back(k);
    }

    // Normalize polygon orientation using stage 0.  det(S^{-1}R^T)>0, so the
    // CCW/CW orientation is identical in world and q frames.
    size_t F = 0U;
    for (auto& g : groups) {
        auto& idx = g.indices;
        if (idx.size() >= 2U) {
            const int ia = idx.front(), ib = idx.back();
            const double dx = V.get(0, ia, 0) - V.get(0, ib, 0);
            const double dy = V.get(0, ia, 1) - V.get(0, ib, 1);
            if (std::hypot(dx, dy) <= 1.0e-10) idx.pop_back();
        }
        if (idx.size() < 3U) {
            g.valid_corner.clear();
            continue;
        }

        double area2 = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) {
            const int ia = idx[i], ib = idx[(i + 1U) % idx.size()];
            area2 += V.get(0, ia, 0) * V.get(0, ib, 1) - V.get(0, ia, 1) * V.get(0, ib, 0);
        }
        if (area2 < 0.0) std::reverse(idx.begin(), idx.end());

        const size_t n = idx.size();
        g.valid_corner.assign(n, static_cast<unsigned char>(0));
        std::vector<std::array<double,2>> edge_n_q(n);
        const double yaw0 = state_yaw(0);
        for (size_t i = 0; i < n; ++i) {
            const int ia = idx[i], ib = idx[(i + 1U) % n];
            const double exw = V.get(0, ib, 0) - V.get(0, ia, 0);
            const double eyw = V.get(0, ib, 1) - V.get(0, ia, 1);
            edge_n_q[i] = local_edge_normal(exw, eyw, yaw0);
        }

        size_t valid_count = 0U;
        for (size_t i = 0; i < n; ++i) {
            const auto& np = edge_n_q[(i + n - 1U) % n];
            const auto& nn = edge_n_q[i];
            if (std::hypot(np[0], np[1]) <= EPS || std::hypot(nn[0], nn[1]) <= EPS) continue;
            double delta = std::atan2(
                cross2(np[0], np[1], nn[0], nn[1]),
                np[0] * nn[0] + np[1] * nn[1]);
            if (delta <= 0.0) delta += 2.0 * PI;
            if (delta > 1.0e-10 && delta < PI - 1.0e-10) {
                g.valid_corner[i] = static_cast<unsigned char>(1);
                ++valid_count;
            }
        }
        F += n + valid_count * static_cast<size_t>(Nn);
    }

    py::array_t<double> feature_points({H, static_cast<ssize_t>(F), static_cast<ssize_t>(2)});
    py::array_t<double> feature_normals({H, static_cast<ssize_t>(F), static_cast<ssize_t>(2)});
    py::array_t<int32_t> feature_types(static_cast<ssize_t>(F));
    py::array_t<int32_t> feature_groups(static_cast<ssize_t>(F));
    auto Pout = feature_points.mutable_unchecked<3>();
    auto Nout = feature_normals.mutable_unchecked<3>();
    auto Tout = feature_types.mutable_unchecked<1>();
    auto Oout = feature_groups.mutable_unchecked<1>();

    size_t base = 0U;
    for (const auto& g : groups) {
        const size_t n = g.indices.size();
        if (n < 3U) continue;
        size_t valid_count = 0U;
        for (unsigned char q : g.valid_corner) if (q) ++valid_count;
        const size_t group_F = n + valid_count * static_cast<size_t>(Nn);
        for (size_t f = 0; f < group_F; ++f) {
            Tout(static_cast<ssize_t>(base + f)) = g.type;
            Oout(static_cast<ssize_t>(base + f)) = g.id;
        }

        for (ssize_t h = 0; h < H; ++h) {
            const double yaw = state_yaw(h);
            std::vector<std::array<double,2>> edge_n_q(n);
            for (size_t i = 0; i < n; ++i) {
                const int ia = g.indices[i], ib = g.indices[(i + 1U) % n];
                const double ax = V.get(h, ia, 0), ay = V.get(h, ia, 1);
                const double bx = V.get(h, ib, 0), by = V.get(h, ib, 1);
                const double exw = bx - ax, eyw = by - ay;
                edge_n_q[i] = local_edge_normal(exw, eyw, yaw);
                const auto nw = local_normal_to_world(edge_n_q[i][0], edge_n_q[i][1], yaw);
                Pout(h, static_cast<ssize_t>(base + i), 0) = 0.5 * (ax + bx);
                Pout(h, static_cast<ssize_t>(base + i), 1) = 0.5 * (ay + by);
                Nout(h, static_cast<ssize_t>(base + i), 0) = nw[0];
                Nout(h, static_cast<ssize_t>(base + i), 1) = nw[1];
            }

            size_t cursor = base + n;
            for (size_t i = 0; i < n; ++i) {
                if (!g.valid_corner[i]) continue;
                const auto& np = edge_n_q[(i + n - 1U) % n];
                const auto& nn = edge_n_q[i];
                double delta = std::atan2(
                    cross2(np[0], np[1], nn[0], nn[1]),
                    np[0] * nn[0] + np[1] * nn[1]);
                if (delta <= 0.0) delta += 2.0 * PI;
                const double theta0 = std::atan2(np[1], np[0]);
                const int iv = g.indices[i];
                const double vx = V.get(h, iv, 0), vy = V.get(h, iv, 1);
                for (int j = 1; j <= Nn; ++j) {
                    const double theta = theta0 + static_cast<double>(j) * delta / static_cast<double>(Nn + 1);
                    const double nqx = std::cos(theta);
                    const double nqy = std::sin(theta);
                    const auto nw = local_normal_to_world(nqx, nqy, yaw);
                    Pout(h, static_cast<ssize_t>(cursor), 0) = vx;
                    Pout(h, static_cast<ssize_t>(cursor), 1) = vy;
                    Nout(h, static_cast<ssize_t>(cursor), 0) = nw[0];
                    Nout(h, static_cast<ssize_t>(cursor), 1) = nw[1];
                    ++cursor;
                }
            }
        }
        base += group_F;
    }

    return py::make_tuple(feature_points, feature_normals, feature_types, feature_groups);
}

struct CollisionWorkspace {
    std::vector<FeatureValue> feature;
    std::vector<double> phi_group;
    std::vector<std::array<double, D>> dphi_group;

    CollisionWorkspace(int K, int O)
        : feature(static_cast<size_t>(K)),
          phi_group(static_cast<size_t>(O), 0.0),
          dphi_group(static_cast<size_t>(O)) {}
};

std::pair<double, std::array<double, D>> smooth_field_value_grad_precomputed(
    CollisionWorkspace& ws,
    const GroupLayout& groups,
    double tau_f,
    double tau_u)
{
    const int K = static_cast<int>(ws.feature.size());
    const int O = groups.group_count();
    std::array<double, D> dphi{};
    dphi.fill(0.0);

    // Common fast path: every feature is its own obstacle. This avoids the
    // feature-level softmax and all per-obstacle temporary work.
    if (groups.singleton_groups) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            mx = std::max(mx, -ws.feature[static_cast<size_t>(k)].score / tau_u);
        }
        double z = 0.0;
        for (int k = 0; k < K; ++k) {
            z += std::exp(-ws.feature[static_cast<size_t>(k)].score / tau_u - mx);
        }
        z = std::max(z, 1.0e-300);
        for (int k = 0; k < K; ++k) {
            const double w = std::exp(-ws.feature[static_cast<size_t>(k)].score / tau_u - mx) / z;
            const auto& ds = ws.feature[static_cast<size_t>(k)].dscore;
            for (int d = 0; d < D; ++d) dphi[d] += w * ds[d];
        }
        return {-tau_u * (mx + std::log(z)), dphi};
    }

    for (int o = 0; o < O; ++o) {
        const int begin = groups.offsets[static_cast<size_t>(o)];
        const int end = groups.offsets[static_cast<size_t>(o + 1)];
        double mx = -std::numeric_limits<double>::infinity();
        for (int p = begin; p < end; ++p) {
            const int k = groups.indices[static_cast<size_t>(p)];
            mx = std::max(mx, ws.feature[static_cast<size_t>(k)].score / tau_f);
        }
        double z = 0.0;
        for (int p = begin; p < end; ++p) {
            const int k = groups.indices[static_cast<size_t>(p)];
            z += std::exp(ws.feature[static_cast<size_t>(k)].score / tau_f - mx);
        }
        z = std::max(z, 1.0e-300);
        ws.phi_group[static_cast<size_t>(o)] = tau_f * (
            mx + std::log(z) - std::log(static_cast<double>(std::max(end - begin, 1)))
        );
        auto& dg = ws.dphi_group[static_cast<size_t>(o)];
        dg.fill(0.0);
        for (int p = begin; p < end; ++p) {
            const int k = groups.indices[static_cast<size_t>(p)];
            const double w = std::exp(ws.feature[static_cast<size_t>(k)].score / tau_f - mx) / z;
            const auto& ds = ws.feature[static_cast<size_t>(k)].dscore;
            for (int d = 0; d < D; ++d) dg[d] += w * ds[d];
        }
    }

    if (O == 1) return {ws.phi_group[0], ws.dphi_group[0]};

    double mx = -std::numeric_limits<double>::infinity();
    for (int o = 0; o < O; ++o) {
        mx = std::max(mx, -ws.phi_group[static_cast<size_t>(o)] / tau_u);
    }
    double z = 0.0;
    for (int o = 0; o < O; ++o) {
        z += std::exp(-ws.phi_group[static_cast<size_t>(o)] / tau_u - mx);
    }
    z = std::max(z, 1.0e-300);
    for (int o = 0; o < O; ++o) {
        const double w = std::exp(-ws.phi_group[static_cast<size_t>(o)] / tau_u - mx) / z;
        const auto& dg = ws.dphi_group[static_cast<size_t>(o)];
        for (int d = 0; d < D; ++d) dphi[d] += w * dg[d];
    }
    return {-tau_u * (mx + std::log(z)), dphi};
}

py::array_t<double> build_collision_dual_lin_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& sector_rays_body,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types, bool has_feature_obs_indices, int vertex_capacity,
    double dt_init, double dt_min, double dt_max, double h_upper_bound, double d_safe,
    double Ns, double regulation_lambda, double ego_L, double ego_W, double halfspace_buffer_m,
    bool use_scaled, const std::string& normal_orientation, double boundary_buffer_m,
    double smooth_extra_buffer_m, double tau_f, double tau_u, double normal_eps,
    double sigma_vx, double sigma_vy, double p_col, bool use_probability_margin, double pred_rate);


inline bool dense_feature_buffer(const py::buffer_info& b) {
    const ssize_t es = static_cast<ssize_t>(sizeof(double));
    if (b.ndim == 2) {
        return b.shape[1] >= 2 && b.strides[1] == es && b.strides[0] == b.shape[1] * es;
    }
    if (b.ndim == 3) {
        return b.shape[2] >= 2 && b.strides[2] == es &&
               b.strides[1] == b.shape[2] * es &&
               b.strides[0] == b.shape[1] * b.shape[2] * es;
    }
    return false;
}

py::array_t<double> build_collision_dual_lin_cuda(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& sector_rays_body,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types, bool has_feature_obs_indices, int vertex_capacity,
    double dt_init, double dt_min, double dt_max, double h_upper_bound, double d_safe,
    double Ns, double regulation_lambda, double ego_L, double ego_W, double halfspace_buffer_m,
    bool use_scaled, const std::string& normal_orientation, double boundary_buffer_m,
    double smooth_extra_buffer_m, double tau_f, double tau_u, double normal_eps,
    double sigma_vx, double sigma_vy, double p_col, bool use_probability_margin, double pred_rate)
{
    (void)tau_u;
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t Hs = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, Hs);
    const FeatureAccessor N = make_feature_accessor(vertex_normals, Hs);
    const VecViewI types = view1d_int(vertex_types);
    const VecViewI obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps);
    const VecViewD dts = view1d_double(dt_vals);
    const MatView2D rays = view2d(sector_rays_body);
    const VecViewD scales = view1d_double(axis_scale);

    const int H = static_cast<int>(Hs);
    const int K = std::max(0, std::min({vertex_capacity, static_cast<int>(V.K), static_cast<int>(N.K)}));
    // API-32: 3 x 7 blocks per stage: avoidance, regulation, and
    // direct impact-aligned ranking value. Only row[14] is populated in the
    // third block; its derivative columns are intentionally zero.
    py::array_t<double> out({Hs, static_cast<ssize_t>(21)});
    auto out_info = out.request();
    double* out_data = static_cast<double*>(out_info.ptr);
    const double disabled = -h_upper_bound;
    std::fill(out_data, out_data + static_cast<size_t>(std::max<ssize_t>(Hs, 0)) * 21U, 0.0);
    for (ssize_t h = 0; h < Hs; ++h) {
        out_data[static_cast<size_t>(h) * 21U + 0U] = disabled;
        out_data[static_cast<size_t>(h) * 21U + 7U] = disabled;
    }
    if (H <= 0 || K <= 0) return out;

    const auto vb = vertex_set.request();
    const auto nb = vertex_normals.request();
    if (!dense_feature_buffer(vb) || !dense_feature_buffer(nb)) {
        throw std::runtime_error("native CUDA proposed field requires C-contiguous feature arrays");
    }
    const int point_stage_count = vb.ndim == 3 ? static_cast<int>(vb.shape[0]) : 1;
    const int normal_stage_count = nb.ndim == 3 ? static_cast<int>(nb.shape[0]) : 1;
    if ((point_stage_count != 1 && point_stage_count != H) ||
        (normal_stage_count != 1 && normal_stage_count != H)) {
        throw std::runtime_error("native CUDA proposed feature horizon mismatch");
    }

    std::vector<double> x4(static_cast<size_t>(H) * 4U, 0.0);
    std::vector<double> hvec(static_cast<size_t>(H), 0.0);
    std::vector<double> dtvec(static_cast<size_t>(H), dt_init);
    for (int h = 0; h < H; ++h) {
        for (int j = 0; j < 4; ++j) x4[static_cast<size_t>(h) * 4U + static_cast<size_t>(j)] = X.get(h, j);
        hvec[static_cast<size_t>(h)] = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : static_cast<double>(h);
        dtvec[static_cast<size_t>(h)] = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : dt_init;
    }

    std::vector<double> dyn(static_cast<size_t>(K), 1.0);
    std::vector<int32_t> obs_ids(static_cast<size_t>(K), 0);
    for (int k = 0; k < K; ++k) {
        if (has_vertex_types) dyn[static_cast<size_t>(k)] = (k < types.n && types(k) == 1) ? 1.0 : 0.0;
        obs_ids[static_cast<size_t>(k)] = has_feature_obs_indices && k < obs.n ? obs(k) : k;
    }
    const GroupLayout group_layout = make_group_layout(obs_ids, has_feature_obs_indices, K);
    const int O = group_layout.group_count();
    if (O <= 0) return out;
    std::vector<int32_t> offsets(static_cast<size_t>(O) + 1U, 0);
    std::vector<int32_t> indices(static_cast<size_t>(K), 0);
    for (int o = 0; o <= O; ++o) offsets[static_cast<size_t>(o)] = static_cast<int32_t>(group_layout.offsets[static_cast<size_t>(o)]);
    for (int k = 0; k < K; ++k) indices[static_cast<size_t>(k)] = static_cast<int32_t>(group_layout.indices[static_cast<size_t>(k)]);

    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : std::max(0.5 * std::abs(ego_L), EPS);
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : std::max(0.5 * std::abs(ego_W), EPS);
    const double geom_buffer = std::max(halfspace_buffer_m, 0.0) +
        std::max(boundary_buffer_m, 0.0) + std::max(smooth_extra_buffer_m, 0.0);
    const double lambda = std::max(std::abs(tau_f), EPS);
    regulation_lambda = std::max(std::abs(regulation_lambda), EPS);
    normal_eps = std::max(std::abs(normal_eps), EPS);
    p_col = clampd(p_col, 1.0e-12, 1.0 - 1.0e-12);
    const double confidence = std::sqrt(std::max(-2.0 * std::log(1.0 - p_col), 0.0));
    const double sigma_iso = std::hypot(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));

    double lcbx = 1.0, lcby = 0.0;
    if (rays.r >= 3 && rays.c >= 2) { lcbx = rays(2, 0); lcby = rays(2, 1); }
    normalize2(lcbx, lcby);
    double lc_local_x = lcbx / sx;
    double lc_local_y = lcby / sy;
    normalize2(lc_local_x, lc_local_y);

    const bool flip = normal_orientation == "inward";
    const int threads = PROPOSED_CUDA_THREADS.load();
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = collision_proposed_cuda_run(
            x4.data(), static_cast<const double*>(vb.ptr), point_stage_count,
            static_cast<const double*>(nb.ptr), normal_stage_count,
            dyn.data(), offsets.data(), indices.data(), hvec.data(), dtvec.data(),
            H, K, O, sx, sy, geom_buffer, lambda, regulation_lambda, sigma_iso, confidence,
            use_probability_margin, pred_rate, dt_init, dt_min, dt_max, h_upper_bound, flip,
            use_scaled, d_safe, Ns, normal_eps, lcbx, lcby, lc_local_x,
            lc_local_y, disabled, threads, out_data);
    }
    if (!ok) {
        throw std::runtime_error(std::string("native CUDA proposed linearization failed: ") +
                                 collision_proposed_cuda_last_error());
    }
    return out;
}

py::array_t<double> build_phi_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types, bool has_feature_obs_indices, int vertex_capacity,
    double dt_init, double dt_min, double dt_max, double h_upper_bound, double d_safe, double ego_L, double ego_W,
    double halfspace_buffer_m, bool use_scaled, const std::string& normal_orientation,
    double boundary_buffer_m, double smooth_extra_buffer_m, double tau_f, double tau_u,
    double sigma_vx, double sigma_vy, double p_col, bool use_probability_margin, double pred_rate);

// -----------------------------------------------------------------------------
// API-25 selectable collision fields (field_flag = 0..4).
//
// field 0: body-fixed normalized-frame circle over-estimation
// field 1: DCSAT
// field 2: WORLD-frame Geometry-Aware Bernstein-polynomial SDF; one polynomial
//          is fitted per obstacle on the fixed world-map domain and reused
// field 3: Wu et al. smooth polygon field
// field 4: proposed field (delegates to the existing analytic HxK kernel)
//
// Field 0 retains the proposed-style body-fixed ego normalization.  Field 2 is
// intentionally a world-frame baseline: its coefficients are independent of ego
// pose/yaw and do not include the unit-disk ego-shape offset.
// -----------------------------------------------------------------------------

struct MFPoint {
    double x = 0.0;
    double y = 0.0;
};

struct MFGroup {
    int id = 0;
    bool dynamic = false;
    std::vector<MFPoint> V;
};

struct MFFieldEval {
    double phi = std::numeric_limits<double>::infinity();
    double gx = 0.0;
    double gy = 0.0;
};

struct MFBPModel {
    bool empty = true;
    int Q = 2;
    std::vector<double> W;
    std::array<double, 4> Tref{{1.0, 0.0, 0.0, 1.0}};
    std::array<double, 2> pref{{0.0, 0.0}};
    std::array<double, 2> xr{{-1.0, 1.0}};
    std::array<double, 2> yr{{-1.0, 1.0}};
    double safety_margin = 0.13;
    // Piecewise-linear approximation of the fitted Bernstein level set
    //     B(q) = safety_margin.
    // The final BP-SDF is recomputed as signed Euclidean distance to this
    // fitted boundary, rather than using B(q)-safety_margin as a distance.
    std::vector<std::array<double,4>> boundary_segments;
};

inline double mf_cross(const MFPoint& o, const MFPoint& a, const MFPoint& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

inline long long mf_round12(double x) {
    return static_cast<long long>(std::llround(x * 1.0e12));
}

std::vector<MFPoint> mf_unique_ordered_points(std::vector<MFPoint> pts) {
    std::vector<MFPoint> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        bool duplicate = false;
        for (const auto& q : out) {
            if (mf_round12(p.x) == mf_round12(q.x) && mf_round12(p.y) == mf_round12(q.y)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) out.push_back(p);
    }
    return out;
}

std::vector<MFPoint> mf_convex_hull(std::vector<MFPoint> pts) {
    pts.erase(std::remove_if(pts.begin(), pts.end(), [](const MFPoint& p) {
        return !std::isfinite(p.x) || !std::isfinite(p.y);
    }), pts.end());
    if (pts.size() <= 1U) return pts;
    std::sort(pts.begin(), pts.end(), [](const MFPoint& a, const MFPoint& b) {
        const long long ax = mf_round12(a.x), bx = mf_round12(b.x);
        if (ax != bx) return ax < bx;
        const long long ay = mf_round12(a.y), by = mf_round12(b.y);
        if (ay != by) return ay < by;
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    std::vector<MFPoint> uniq;
    uniq.reserve(pts.size());
    for (const auto& p : pts) {
        if (uniq.empty() || mf_round12(p.x) != mf_round12(uniq.back().x) ||
            mf_round12(p.y) != mf_round12(uniq.back().y)) {
            uniq.push_back(p);
        }
    }
    if (uniq.size() <= 2U) return uniq;
    std::vector<MFPoint> lower, upper;
    lower.reserve(uniq.size()); upper.reserve(uniq.size());
    for (const auto& p : uniq) {
        while (lower.size() >= 2U && mf_cross(lower[lower.size()-2], lower.back(), p) <= 1.0e-12) lower.pop_back();
        lower.push_back(p);
    }
    for (auto it = uniq.rbegin(); it != uniq.rend(); ++it) {
        while (upper.size() >= 2U && mf_cross(upper[upper.size()-2], upper.back(), *it) <= 1.0e-12) upper.pop_back();
        upper.push_back(*it);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    // Monotone-chain is CCW; retain a defensive orientation check.
    double area2 = 0.0;
    for (size_t i = 0; i < lower.size(); ++i) {
        const auto& a = lower[i];
        const auto& b = lower[(i + 1U) % lower.size()];
        area2 += a.x * b.y - a.y * b.x;
    }
    if (area2 < 0.0) std::reverse(lower.begin(), lower.end());
    return lower;
}

std::vector<MFGroup> mf_build_world_groups(
    const FeatureAccessor& V,
    ssize_t h,
    int K,
    const VecViewI& types,
    const VecViewI& obs,
    bool has_types,
    bool has_obs_ids,
    bool convexify)
{
    std::unordered_map<int, size_t> lookup;
    std::vector<MFGroup> groups;
    groups.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const int oid = has_obs_ids && k < obs.n ? static_cast<int>(obs(k)) : 0;
        if (oid < 0) continue;
        auto it = lookup.find(oid);
        size_t gi;
        if (it == lookup.end()) {
            gi = groups.size();
            lookup.emplace(oid, gi);
            MFGroup g; g.id = oid; groups.push_back(std::move(g));
        } else {
            gi = it->second;
        }
        groups[gi].V.push_back({V.get(h, k, 0), V.get(h, k, 1)});
        if (has_types && k < types.n && types(k) == 1) groups[gi].dynamic = true;
    }
    std::vector<MFGroup> out;
    out.reserve(groups.size());
    for (auto& g : groups) {
        g.V = convexify ? mf_convex_hull(std::move(g.V))
                        : mf_unique_ordered_points(std::move(g.V));
        if (!g.V.empty()) out.push_back(std::move(g));
    }
    return out;
}


// -----------------------------------------------------------------------------
// API-25 exact continuous ego-surface distance kernels (mpc_flag = 0..3).
//
// These functions port the Python ellipse-arc <-> convex-polygon closest-pair
// calculation to C++.  No angular surface sampling and no finite-difference
// geometry evaluations are used.  The Jacobian follows from the closest pair
// via the same envelope-theorem expression used by utils_linearized.py.
// -----------------------------------------------------------------------------

inline double surf_dot(const MFPoint& a, const MFPoint& b) {
    return a.x * b.x + a.y * b.y;
}
inline MFPoint surf_add(const MFPoint& a, const MFPoint& b) { return {a.x + b.x, a.y + b.y}; }
inline MFPoint surf_sub(const MFPoint& a, const MFPoint& b) { return {a.x - b.x, a.y - b.y}; }
inline MFPoint surf_mul(const MFPoint& a, double s) { return {a.x * s, a.y * s}; }
inline double surf_norm2(const MFPoint& a) { return surf_dot(a, a); }
inline double surf_norm(const MFPoint& a) { return std::hypot(a.x, a.y); }

inline double surf_poly_eval(const std::vector<double>& c, double x) {
    double y = 0.0;
    for (double a : c) y = y * x + a;
    return y;
}

std::vector<double> surf_unique_sorted_roots(std::vector<double> roots) {
    std::sort(roots.begin(), roots.end());
    std::vector<double> out;
    out.reserve(roots.size());
    for (double r : roots) {
        if (!std::isfinite(r)) continue;
        if (out.empty() || std::abs(r - out.back()) > 1.0e-8 * (1.0 + std::abs(r))) out.push_back(r);
    }
    return out;
}

// All real roots of a polynomial of degree <= 4.  Derivative roots partition
// the line into monotone intervals, so simple roots are bracketed exactly;
// repeated roots are detected at derivative roots.  This avoids an Eigen/LAPACK
// dependency while matching numpy.roots for the quartic used by the ellipse
// closest-point condition to numerical precision.
std::vector<double> surf_real_poly_roots(std::vector<double> coeff) {
    double scale = 1.0;
    for (double v : coeff) scale = std::max(scale, std::abs(v));
    while (coeff.size() > 1U && std::abs(coeff.front()) <= 1.0e-14 * scale) coeff.erase(coeff.begin());
    const int n = static_cast<int>(coeff.size()) - 1;
    if (n <= 0) return {};
    if (n == 1) {
        if (std::abs(coeff[0]) <= 1.0e-30) return {};
        return {-coeff[1] / coeff[0]};
    }

    std::vector<double> deriv(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) deriv[static_cast<size_t>(i)] = coeff[static_cast<size_t>(i)] * static_cast<double>(n - i);
    std::vector<double> crit = surf_real_poly_roots(deriv);

    const double lead = std::abs(coeff[0]);
    double R = 1.0;
    if (lead > 0.0) {
        for (size_t i = 1; i < coeff.size(); ++i) R = std::max(R, 1.0 + std::abs(coeff[i]) / lead);
    }
    std::vector<double> cuts;
    cuts.reserve(crit.size() + 2U);
    cuts.push_back(-R);
    for (double r : crit) if (r > -R && r < R) cuts.push_back(r);
    cuts.push_back(R);
    std::sort(cuts.begin(), cuts.end());

    std::vector<double> roots;
    auto root_tol = [&](double x) {
        return 1.0e-13 * scale * std::pow(std::max(1.0, std::abs(x)), n);
    };
    for (double r : crit) {
        if (r >= -R && r <= R && std::abs(surf_poly_eval(coeff, r)) <= root_tol(r)) roots.push_back(r);
    }

    for (size_t i = 0; i + 1U < cuts.size(); ++i) {
        double lo = cuts[i], hi = cuts[i + 1U];
        double flo = surf_poly_eval(coeff, lo), fhi = surf_poly_eval(coeff, hi);
        if (std::abs(flo) <= root_tol(lo)) roots.push_back(lo);
        if (std::abs(fhi) <= root_tol(hi)) roots.push_back(hi);
        if (!std::isfinite(flo) || !std::isfinite(fhi) || flo == 0.0 || fhi == 0.0 || ((flo > 0.0) == (fhi > 0.0))) continue;
        for (int it = 0; it < 100; ++it) {
            const double mid = 0.5 * (lo + hi);
            const double fm = surf_poly_eval(coeff, mid);
            if (std::abs(fm) <= root_tol(mid) || (hi - lo) <= 1.0e-14 * (1.0 + std::abs(mid))) {
                lo = hi = mid;
                break;
            }
            if ((flo > 0.0) == (fm > 0.0)) { lo = mid; flo = fm; }
            else { hi = mid; fhi = fm; }
        }
        roots.push_back(0.5 * (lo + hi));
    }
    return surf_unique_sorted_roots(std::move(roots));
}

std::vector<double> surf_angle_lifts(double theta, double lo, double hi, double tol = 1.0e-10) {
    const double two_pi = 2.0 * PI;
    const int k0 = static_cast<int>(std::floor((lo - theta) / two_pi));
    std::vector<double> out;
    out.reserve(2);
    for (int k = k0 - 1; k <= k0 + 3; ++k) {
        double v = theta + two_pi * static_cast<double>(k);
        if (v >= lo - tol && v <= hi + tol) out.push_back(clampd(v, lo, hi));
    }
    return out;
}

inline MFPoint surf_arc_point(double theta, double a, double b) {
    return {a * std::cos(theta), b * std::sin(theta)};
}

inline MFPoint surf_segment_closest_point(const MFPoint& p, const MFPoint& A, const MFPoint& B) {
    const MFPoint D = surf_sub(B, A);
    const double den = surf_norm2(D);
    if (den <= 1.0e-24) return A;
    const double t = clampd(surf_dot(surf_sub(p, A), D) / den, 0.0, 1.0);
    return surf_add(A, surf_mul(D, t));
}

bool surf_point_inside_convex(const MFPoint& p, const std::vector<MFPoint>& V, double tol = 1.0e-10) {
    if (V.size() < 3U) return false;
    for (size_t i = 0; i < V.size(); ++i) {
        const MFPoint& A = V[i];
        const MFPoint& B = V[(i + 1U) % V.size()];
        if (cross2(B.x - A.x, B.y - A.y, p.x - A.x, p.y - A.y) < -tol) return false;
    }
    return true;
}

bool surf_arc_segment_intersection_point(
    const MFPoint& A, const MFPoint& B, double a, double b,
    double theta_lo, double theta_hi, MFPoint& hit, double tol = 1.0e-10)
{
    const MFPoint D = surf_sub(B, A);
    const double aa = std::max(a, 1.0e-12), bb = std::max(b, 1.0e-12);
    const double q2 = D.x * D.x / (aa * aa) + D.y * D.y / (bb * bb);
    const double q1 = 2.0 * (A.x * D.x / (aa * aa) + A.y * D.y / (bb * bb));
    const double q0 = A.x * A.x / (aa * aa) + A.y * A.y / (bb * bb) - 1.0;
    std::array<double, 2> roots{{0.0, 0.0}};
    int nr = 0;
    if (std::abs(q2) <= 1.0e-24) {
        if (std::abs(q1) <= 1.0e-24) {
            if (std::abs(q0) <= tol) roots[nr++] = 0.0;
        } else roots[nr++] = -q0 / q1;
    } else {
        double disc = q1 * q1 - 4.0 * q2 * q0;
        if (disc >= -tol) {
            disc = std::max(disc, 0.0);
            const double sq = std::sqrt(disc);
            roots[nr++] = (-q1 - sq) / (2.0 * q2);
            roots[nr++] = (-q1 + sq) / (2.0 * q2);
        }
    }
    for (int i = 0; i < nr; ++i) {
        const double t = roots[static_cast<size_t>(i)];
        if (t < -tol || t > 1.0 + tol) continue;
        const MFPoint P = surf_add(A, surf_mul(D, clampd(t, 0.0, 1.0)));
        const double theta = std::atan2(P.y / bb, P.x / aa);
        if (!surf_angle_lifts(theta, theta_lo, theta_hi, tol).empty()) { hit = P; return true; }
    }
    return false;
}

std::pair<double, MFPoint> surf_arc_closest_to_point(
    const MFPoint& P, double ellipse_a, double ellipse_b, double theta_lo, double theta_hi)
{
    const double a = std::max(ellipse_a, 1.0e-12), b = std::max(ellipse_b, 1.0e-12);
    const double D = b * b - a * a;
    std::vector<double> cand{theta_lo, theta_hi};
    for (double base : {0.0, 0.5 * PI, PI, -0.5 * PI}) {
        const auto lifts = surf_angle_lifts(base, theta_lo, theta_hi);
        cand.insert(cand.end(), lifts.begin(), lifts.end());
    }
    std::vector<double> coeff{
        b * P.y,
        2.0 * (a * P.x - D),
        0.0,
        2.0 * (D + a * P.x),
        -b * P.y
    };
    for (double root : surf_real_poly_roots(coeff)) {
        const double theta = 2.0 * std::atan(root);
        const auto lifts = surf_angle_lifts(theta, theta_lo, theta_hi);
        cand.insert(cand.end(), lifts.begin(), lifts.end());
    }
    double best = std::numeric_limits<double>::infinity();
    MFPoint best_e = surf_arc_point(theta_lo, a, b);
    for (double theta : cand) {
        const MFPoint E = surf_arc_point(theta, a, b);
        const double d = surf_norm(surf_sub(E, P));
        if (d < best) { best = d; best_e = E; }
    }
    return {best, best_e};
}

struct SurfClosestPair {
    double d = std::numeric_limits<double>::infinity();
    MFPoint E{};
    MFPoint Q{};
};

SurfClosestPair surf_arc_polygon_closest(
    const std::vector<MFPoint>& poly, double ellipse_a, double ellipse_b,
    double theta_lo, double theta_hi)
{
    // Input is already a CCW convex hull from mf_build_world_groups().
    // A rigid world->body transform preserves convexity/orientation, so doing
    // another convex hull for every surface is pure duplicate work.
    SurfClosestPair out;
    const size_t n = poly.size();
    if (n == 0U) return out;
    if (n == 1U) {
        const auto cp = surf_arc_closest_to_point(poly[0], ellipse_a, ellipse_b, theta_lo, theta_hi);
        out.d = cp.first; out.E = cp.second; out.Q = poly[0]; return out;
    }
    const double a = std::max(ellipse_a, 1.0e-12), b = std::max(ellipse_b, 1.0e-12);
    const MFPoint E0 = surf_arc_point(theta_lo, a, b), E1 = surf_arc_point(theta_hi, a, b);
    if (n >= 3U && surf_point_inside_convex(E0, poly)) return {0.0, E0, E0};
    if (n >= 3U && surf_point_inside_convex(E1, poly)) return {0.0, E1, E1};
    const size_t edge_count = n >= 3U ? n : 1U;

    for (size_t i = 0; i < edge_count; ++i) {
        const size_t j = n >= 3U ? (i + 1U) % n : 1U;
        MFPoint hit{};
        if (surf_arc_segment_intersection_point(poly[i], poly[j], a, b, theta_lo, theta_hi, hit)) return {0.0, hit, hit};
    }

    out.E = E0; out.Q = poly[0];
    for (const MFPoint& E : {E0, E1}) {
        for (size_t i = 0; i < edge_count; ++i) {
            const size_t j = n >= 3U ? (i + 1U) % n : 1U;
            const MFPoint Q = surf_segment_closest_point(E, poly[i], poly[j]);
            const double d = surf_norm(surf_sub(E, Q));
            if (d < out.d) { out.d = d; out.E = E; out.Q = Q; }
        }
    }
    for (const MFPoint& P : poly) {
        const auto cp = surf_arc_closest_to_point(P, a, b, theta_lo, theta_hi);
        if (cp.first < out.d) { out.d = cp.first; out.E = cp.second; out.Q = P; }
    }
    for (size_t i = 0; i < edge_count; ++i) {
        const size_t j = n >= 3U ? (i + 1U) % n : 1U;
        const MFPoint A = poly[i], B = poly[j], Dseg = surf_sub(B, A);
        const double len2 = surf_norm2(Dseg);
        if (len2 <= 1.0e-24) continue;
        const double len = std::sqrt(len2);
        const MFPoint normal{-Dseg.y / len, Dseg.x / len};
        const double theta0 = std::atan2(b * normal.y, a * normal.x);
        for (double base : {theta0, theta0 + PI}) {
            for (double theta : surf_angle_lifts(base, theta_lo, theta_hi)) {
                const MFPoint E = surf_arc_point(theta, a, b);
                const double t = surf_dot(surf_sub(E, A), Dseg) / len2;
                if (t < -1.0e-10 || t > 1.0 + 1.0e-10) continue;
                const MFPoint Q = surf_add(A, surf_mul(Dseg, clampd(t, 0.0, 1.0)));
                const double d = surf_norm(surf_sub(E, Q));
                if (d < out.d) { out.d = d; out.E = E; out.Q = Q; }
            }
        }
    }
    return out;
}

std::array<std::array<double, 2>, 4> surf_intervals(double ego_L, double ego_W, double a, double b) {
    const double x_f = ego_L / 2.0 - 0.02, x_r = -ego_L / 2.0 + 0.02;
    const double y_l = ego_W / 2.0, y_r = -ego_W / 2.0;
    const MFPoint fl{x_f, y_l}, fr{x_f, y_r}, rr{x_r, y_r}, rl{x_r, y_l};
    const std::array<std::array<MFPoint, 2>, 4> rays{{{{fl, rl}}, {{rl, rr}}, {{rr, fr}}, {{fr, fl}}}};
    std::array<std::array<double, 2>, 4> out{};
    for (int j = 0; j < 4; ++j) {
        double lo = std::atan2(rays[static_cast<size_t>(j)][0].y / b, rays[static_cast<size_t>(j)][0].x / a);
        double hi = std::atan2(rays[static_cast<size_t>(j)][1].y / b, rays[static_cast<size_t>(j)][1].x / a);
        while (hi <= lo) hi += 2.0 * PI;
        out[static_cast<size_t>(j)] = {{lo, hi}};
    }
    return out;
}

struct SurfAABB {
    double minx = std::numeric_limits<double>::infinity();
    double maxx = -std::numeric_limits<double>::infinity();
    double miny = std::numeric_limits<double>::infinity();
    double maxy = -std::numeric_limits<double>::infinity();
};

inline void surf_aabb_include(SurfAABB& b, const MFPoint& p) {
    b.minx = std::min(b.minx, p.x); b.maxx = std::max(b.maxx, p.x);
    b.miny = std::min(b.miny, p.y); b.maxy = std::max(b.maxy, p.y);
}

inline double surf_interval_gap(double amin, double amax, double bmin, double bmax) {
    if (amax < bmin) return bmin - amax;
    if (bmax < amin) return amin - bmax;
    return 0.0;
}

inline double surf_aabb_distance(const SurfAABB& a, const SurfAABB& b) {
    const double dx = surf_interval_gap(a.minx, a.maxx, b.minx, b.maxx);
    const double dy = surf_interval_gap(a.miny, a.maxy, b.miny, b.maxy);
    return std::hypot(dx, dy);
}

SurfAABB surf_arc_aabb(double a, double b, double theta_lo, double theta_hi) {
    SurfAABB box;
    const MFPoint e0 = surf_arc_point(theta_lo, a, b);
    const MFPoint e1 = surf_arc_point(theta_hi, a, b);
    surf_aabb_include(box, e0); surf_aabb_include(box, e1);
    for (double base : {0.0, 0.5 * PI, PI, -0.5 * PI}) {
        for (double theta : surf_angle_lifts(base, theta_lo, theta_hi)) {
            surf_aabb_include(box, surf_arc_point(theta, a, b));
        }
    }
    return box;
}

struct SurfIndexedPoint {
    MFPoint p{};
    int idx = -1;
};

std::vector<int> surf_convex_hull_indices(
    const FeatureAccessor& V, ssize_t h, const std::vector<int>& raw_indices)
{
    std::vector<SurfIndexedPoint> pts;
    pts.reserve(raw_indices.size());
    for (int idx : raw_indices) {
        const MFPoint p{V.get(h, idx, 0), V.get(h, idx, 1)};
        if (std::isfinite(p.x) && std::isfinite(p.y)) pts.push_back({p, idx});
    }
    if (pts.size() <= 1U) {
        std::vector<int> out;
        if (!pts.empty()) out.push_back(pts[0].idx);
        return out;
    }
    std::sort(pts.begin(), pts.end(), [](const SurfIndexedPoint& a, const SurfIndexedPoint& b) {
        const long long ax = mf_round12(a.p.x), bx = mf_round12(b.p.x);
        if (ax != bx) return ax < bx;
        const long long ay = mf_round12(a.p.y), by = mf_round12(b.p.y);
        if (ay != by) return ay < by;
        if (a.p.x != b.p.x) return a.p.x < b.p.x;
        return a.p.y < b.p.y;
    });
    std::vector<SurfIndexedPoint> uniq;
    uniq.reserve(pts.size());
    for (const auto& item : pts) {
        if (uniq.empty() || mf_round12(item.p.x) != mf_round12(uniq.back().p.x) ||
            mf_round12(item.p.y) != mf_round12(uniq.back().p.y)) {
            uniq.push_back(item);
        }
    }
    if (uniq.size() <= 2U) {
        std::vector<int> out;
        out.reserve(uniq.size());
        for (const auto& item : uniq) out.push_back(item.idx);
        return out;
    }
    std::vector<SurfIndexedPoint> lower, upper;
    lower.reserve(uniq.size()); upper.reserve(uniq.size());
    for (const auto& item : uniq) {
        while (lower.size() >= 2U && mf_cross(lower[lower.size()-2].p, lower.back().p, item.p) <= 1.0e-12) lower.pop_back();
        lower.push_back(item);
    }
    for (auto it = uniq.rbegin(); it != uniq.rend(); ++it) {
        while (upper.size() >= 2U && mf_cross(upper[upper.size()-2].p, upper.back().p, it->p) <= 1.0e-12) upper.pop_back();
        upper.push_back(*it);
    }
    lower.pop_back(); upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    double area2 = 0.0;
    for (size_t i = 0; i < lower.size(); ++i) {
        const auto& a0 = lower[i].p;
        const auto& b0 = lower[(i + 1U) % lower.size()].p;
        area2 += a0.x * b0.y - a0.y * b0.x;
    }
    if (area2 < 0.0) std::reverse(lower.begin(), lower.end());
    std::vector<int> out;
    out.reserve(lower.size());
    for (const auto& item : lower) out.push_back(item.idx);
    return out;
}

struct SurfGroupLayout {
    int id = 0;
    bool dynamic = false;
    bool rigid_translation = true;
    std::vector<int> raw_indices;
    std::vector<int> hull_indices;
};

std::vector<SurfGroupLayout> surf_build_group_layouts(
    const FeatureAccessor& V, ssize_t H, int K,
    const VecViewI& types, const VecViewI& obs,
    bool has_types, bool has_obs_ids)
{
    std::unordered_map<int, size_t> lookup;
    std::vector<SurfGroupLayout> groups;
    groups.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const int oid = has_obs_ids && k < obs.n ? static_cast<int>(obs(k)) : 0;
        if (oid < 0) continue;
        auto it = lookup.find(oid);
        size_t gi;
        if (it == lookup.end()) {
            gi = groups.size(); lookup.emplace(oid, gi);
            SurfGroupLayout g; g.id = oid; groups.push_back(std::move(g));
        } else gi = it->second;
        groups[gi].raw_indices.push_back(k);
        if (has_types && k < types.n && types(k) == 1) groups[gi].dynamic = true;
    }

    // vertex_selector4 publishes a stage-wise tensor where every convex part
    // keeps its shape and only translates under constant-velocity prediction.
    // Detect that property rather than assuming it, then reuse the stage-0 hull
    // indices across the whole horizon.  Non-rigid inputs fall back to a
    // per-stage hull below, preserving the old general behavior.
    for (auto& g : groups) {
        if (g.raw_indices.empty()) continue;
        g.hull_indices = surf_convex_hull_indices(V, 0, g.raw_indices);
        if (!V.stagewise || H <= 1 || g.raw_indices.size() <= 1U) {
            g.rigid_translation = true;
            continue;
        }
        const int anchor = g.raw_indices.front();
        const double ax0 = V.get(0, anchor, 0), ay0 = V.get(0, anchor, 1);
        for (ssize_t h = 1; h < H && g.rigid_translation; ++h) {
            const double axh = V.get(h, anchor, 0), ayh = V.get(h, anchor, 1);
            for (int idx : g.raw_indices) {
                const double rx0 = V.get(0, idx, 0) - ax0;
                const double ry0 = V.get(0, idx, 1) - ay0;
                const double rxh = V.get(h, idx, 0) - axh;
                const double ryh = V.get(h, idx, 1) - ayh;
                const double tolx = 5.0e-11 * (1.0 + std::max(std::abs(rx0), std::abs(rxh)));
                const double toly = 5.0e-11 * (1.0 + std::max(std::abs(ry0), std::abs(ryh)));
                if (std::abs(rxh - rx0) > tolx || std::abs(ryh - ry0) > toly) {
                    g.rigid_translation = false;
                    break;
                }
            }
        }
    }
    groups.erase(std::remove_if(groups.begin(), groups.end(), [](const SurfGroupLayout& g) {
        return g.raw_indices.empty();
    }), groups.end());
    return groups;
}

struct SurfPreparedGroup {
    bool dynamic = false;
    std::vector<MFPoint> Vb;
    SurfAABB box{};
    std::array<double, 4> lower_bound{{0.0, 0.0, 0.0, 0.0}};
    double sort_key = 0.0;
};

struct SurfStageEval {
    std::array<double, 4> d{};
    std::array<std::array<double, 4>, 4> ddx{};
    std::array<double, 4> ddt{};
};

SurfStageEval surf_evaluate_stage_fast(
    const std::array<double, 4>& x,
    const FeatureAccessor& V, ssize_t h, const std::vector<SurfGroupLayout>& layouts,
    double h_step, double dt_val,
    double ellipse_a, double ellipse_b, double ego_L, double ego_W,
    double h_upper_bound, double sigma_vx, double sigma_vy, double p_col,
    bool use_probability_margin)
{
    SurfStageEval out;
    const double far = std::max(h_upper_bound, 1.0e6);
    out.d.fill(far); out.ddt.fill(0.0);
    for (auto& row : out.ddx) row.fill(0.0);
    if (layouts.empty()) return out;

    const double a = std::max(ellipse_a, 1.0e-12), b = std::max(ellipse_b, 1.0e-12);
    const auto intervals = surf_intervals(ego_L, ego_W, a, b);
    std::array<SurfAABB, 4> arc_boxes{};
    for (int j = 0; j < 4; ++j) {
        arc_boxes[static_cast<size_t>(j)] = surf_arc_aabb(
            a, b, intervals[static_cast<size_t>(j)][0], intervals[static_cast<size_t>(j)][1]);
    }
    const double c = std::cos(x[3]), s = std::sin(x[3]);
    double uncertainty = 0.0, margin_rate = 0.0;
    if (use_probability_margin) {
        const double pc = clampd(p_col, 0.0, 1.0 - 1.0e-12);
        const double conf = std::sqrt(std::max(-2.0 * std::log(1.0 - pc), 0.0));
        const double sigma_iso = std::hypot(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));
        uncertainty = sigma_iso * std::max(h_step, 0.0) * std::max(dt_val, 0.0) * conf;
        if (dt_val >= 0.0) margin_rate = sigma_iso * std::max(h_step, 0.0) * conf;
    }

    // Build each obstacle only once per stage.  For the normal selector path
    // this is a direct indexed transform of the cached convex hull, avoiding
    // the former unordered-map + sort + convex-hull rebuild at every stage.
    std::vector<SurfPreparedGroup> prepared;
    prepared.reserve(layouts.size());
    for (const auto& layout : layouts) {
        SurfPreparedGroup pg;
        pg.dynamic = layout.dynamic;
        if (layout.rigid_translation) {
            pg.Vb.reserve(layout.hull_indices.size());
            for (int idx : layout.hull_indices) {
                const double dx = V.get(h, idx, 0) - x[0];
                const double dy = V.get(h, idx, 1) - x[1];
                const MFPoint p{c * dx + s * dy, -s * dx + c * dy};
                pg.Vb.push_back(p); surf_aabb_include(pg.box, p);
            }
        } else {
            std::vector<MFPoint> world;
            world.reserve(layout.raw_indices.size());
            for (int idx : layout.raw_indices) {
                world.push_back({V.get(h, idx, 0), V.get(h, idx, 1)});
            }
            world = mf_convex_hull(std::move(world));
            pg.Vb.reserve(world.size());
            for (const auto& pw : world) {
                const double dx = pw.x - x[0], dy = pw.y - x[1];
                const MFPoint p{c * dx + s * dy, -s * dx + c * dy};
                pg.Vb.push_back(p); surf_aabb_include(pg.box, p);
            }
        }
        if (pg.Vb.empty()) continue;
        const double margin = pg.dynamic ? uncertainty : 0.0;
        pg.sort_key = std::numeric_limits<double>::infinity();
        for (int j = 0; j < 4; ++j) {
            // Distance between AABBs is a rigorous lower bound on the exact
            // arc-to-polygon distance.  Subtract the same dynamic uncertainty
            // margin used by the exact result, so pruning cannot change minima.
            pg.lower_bound[static_cast<size_t>(j)] =
                surf_aabb_distance(pg.box, arc_boxes[static_cast<size_t>(j)]) - margin;
            pg.sort_key = std::min(pg.sort_key, pg.lower_bound[static_cast<size_t>(j)]);
        }
        prepared.push_back(std::move(pg));
    }
    std::sort(prepared.begin(), prepared.end(), [](const SurfPreparedGroup& a0, const SurfPreparedGroup& b0) {
        return a0.sort_key < b0.sort_key;
    });

    for (const auto& pg : prepared) {
        const double margin = pg.dynamic ? uncertainty : 0.0;
        const double margin_dt = pg.dynamic ? margin_rate : 0.0;
        for (int j = 0; j < 4; ++j) {
            const size_t js = static_cast<size_t>(j);
            if (pg.lower_bound[js] >= out.d[js] + 1.0e-12 * (1.0 + std::abs(out.d[js]))) continue;
            const auto cp = surf_arc_polygon_closest(
                pg.Vb, a, b, intervals[js][0], intervals[js][1]);
            const double dval = cp.d - margin;
            if (!(dval < out.d[js])) continue;
            out.d[js] = dval;
            out.ddx[js].fill(0.0);
            out.ddt[js] = -margin_dt;
            if (cp.d > 1.0e-10 && std::isfinite(cp.d)) {
                const MFPoint u = surf_mul(surf_sub(cp.E, cp.Q), 1.0 / cp.d);
                out.ddx[js][0] = c * u.x - s * u.y;
                out.ddx[js][1] = s * u.x + c * u.y;
                out.ddx[js][2] = 0.0;
                out.ddx[js][3] = u.x * (-cp.E.y) + u.y * cp.E.x;
            }
        }
    }
    for (int j = 0; j < 4; ++j) {
        if (!std::isfinite(out.d[static_cast<size_t>(j)])) out.d[static_cast<size_t>(j)] = far;
        if (!std::isfinite(out.ddt[static_cast<size_t>(j)])) out.ddt[static_cast<size_t>(j)] = 0.0;
        for (double& v : out.ddx[static_cast<size_t>(j)]) if (!std::isfinite(v)) v = 0.0;
    }
    return out;
}

py::array_t<double> build_surface_distance_lin_hk(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types, bool has_feature_obs_indices, int vertex_capacity,
    double h_upper_bound, double d_safe, double ego_L, double ego_W,
    double sigma_vx, double sigma_vy, double p_col, bool use_probability_margin)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const VecViewI types = view1d_int(vertex_types), obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps), dts = view1d_double(dt_vals), scales = view1d_double(axis_scale);
    if (scales.n < 2) throw std::runtime_error("axis_scale must have two ellipse radii");
    const int K = std::min<int>(std::max(vertex_capacity, 0), static_cast<int>(V.K));
    py::array_t<double> out({H, static_cast<ssize_t>(28)});
    auto bo = out.mutable_unchecked<2>();

    const auto layouts = K > 0 ? surf_build_group_layouts(
        V, H, K, types, obs, has_vertex_types, has_feature_obs_indices)
        : std::vector<SurfGroupLayout>{};
    auto process_stage = [&](ssize_t h) {
        const std::array<double, 4> x{{X.get(h,0), X.get(h,1), X.get(h,2), X.get(h,3)}};
        const double hh = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : 1.0;
        const double dt = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : 0.0;
        const SurfStageEval ev = surf_evaluate_stage_fast(
            x, V, h, layouts, hh, dt, scales(0), scales(1), ego_L, ego_W,
            h_upper_bound, sigma_vx, sigma_vy, p_col, use_probability_margin);
        for (int j = 0; j < 4; ++j) {
            const int off = 7 * j;
            bo(h, off) = d_safe - ev.d[static_cast<size_t>(j)];
            bo(h, off + 1) = -ev.ddx[static_cast<size_t>(j)][0];
            bo(h, off + 2) = -ev.ddx[static_cast<size_t>(j)][1];
            bo(h, off + 3) = -ev.ddx[static_cast<size_t>(j)][2];
            bo(h, off + 4) = -ev.ddx[static_cast<size_t>(j)][3];
            bo(h, off + 5) = -ev.ddt[static_cast<size_t>(j)];
            bo(h, off + 6) = 0.0;
        }
    };
#ifdef _OPENMP
    if (H >= 4) {
        #pragma omp parallel for schedule(static)
        for (ssize_t h = 0; h < H; ++h) process_stage(h);
    } else
#endif
    {
        for (ssize_t h = 0; h < H; ++h) process_stage(h);
    }
    return out;
}

py::array_t<double> evaluate_surface_distance_seq_cpp(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types, bool has_feature_obs_indices, int vertex_capacity,
    double h_upper_bound, double ego_L, double ego_W,
    double sigma_vx, double sigma_vy, double p_col, bool use_probability_margin)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const VecViewI types = view1d_int(vertex_types), obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps), dts = view1d_double(dt_vals), scales = view1d_double(axis_scale);
    if (scales.n < 2) throw std::runtime_error("axis_scale must have two ellipse radii");
    const int K = std::min<int>(std::max(vertex_capacity, 0), static_cast<int>(V.K));
    py::array_t<double> out({H, static_cast<ssize_t>(4)});
    auto bo = out.mutable_unchecked<2>();
    const auto layouts = K > 0 ? surf_build_group_layouts(
        V, H, K, types, obs, has_vertex_types, has_feature_obs_indices)
        : std::vector<SurfGroupLayout>{};
    auto process_stage = [&](ssize_t h) {
        const std::array<double, 4> x{{X.get(h,0), X.get(h,1), X.get(h,2), X.get(h,3)}};
        const double hh = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : 1.0;
        const double dt = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : 0.0;
        const SurfStageEval ev = surf_evaluate_stage_fast(
            x, V, h, layouts, hh, dt, scales(0), scales(1), ego_L, ego_W,
            h_upper_bound, sigma_vx, sigma_vy, p_col, use_probability_margin);
        for (int j = 0; j < 4; ++j) bo(h, j) = ev.d[static_cast<size_t>(j)];
    };
#ifdef _OPENMP
    if (H >= 4) {
        #pragma omp parallel for schedule(static)
        for (ssize_t h = 0; h < H; ++h) process_stage(h);
    } else
#endif
    {
        for (ssize_t h = 0; h < H; ++h) process_stage(h);
    }
    return out;
}

std::vector<MFGroup> mf_to_local_groups(
    const std::vector<MFGroup>& world,
    const std::array<double,4>& x,
    double sx,
    double sy)
{
    const double c = std::cos(x[3]), s = std::sin(x[3]);
    const double isx = 1.0 / std::max(std::abs(sx), EPS);
    const double isy = 1.0 / std::max(std::abs(sy), EPS);
    std::vector<MFGroup> out;
    out.reserve(world.size());
    for (const auto& gw : world) {
        MFGroup g; g.id = gw.id; g.dynamic = gw.dynamic; g.V.reserve(gw.V.size());
        for (const auto& p : gw.V) {
            const double dx = p.x - x[0], dy = p.y - x[1];
            g.V.push_back({isx * (c * dx + s * dy), isy * (-s * dx + c * dy)});
        }
        out.push_back(std::move(g));
    }
    return out;
}

inline MFPoint mf_group_centroid(const MFGroup& g) {
    MFPoint c{0.0, 0.0};
    if (g.V.empty()) return c;
    for (const auto& p : g.V) { c.x += p.x; c.y += p.y; }
    const double inv = 1.0 / static_cast<double>(g.V.size());
    c.x *= inv; c.y *= inv;
    return c;
}

MFGroup mf_to_centered_local_group(
    const MFGroup& world,
    const MFPoint& center_world,
    double yaw,
    double sx,
    double sy)
{
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double isx = 1.0 / std::max(std::abs(sx), EPS);
    const double isy = 1.0 / std::max(std::abs(sy), EPS);
    MFGroup out; out.id = world.id; out.dynamic = world.dynamic; out.V.reserve(world.V.size());
    for (const auto& p : world.V) {
        const double dx = p.x - center_world.x, dy = p.y - center_world.y;
        out.V.push_back({isx * (c * dx + s * dy), isy * (-s * dx + c * dy)});
    }
    return out;
}

MFFieldEval mf_smooth_min(
    const std::vector<double>& vals,
    const std::vector<std::array<double,2>>& grads,
    double tau)
{
    MFFieldEval out;
    if (vals.empty()) return out;
    if (!(tau > 0.0)) {
        size_t idx = 0;
        for (size_t i = 1; i < vals.size(); ++i) if (vals[i] < vals[idx]) idx = i;
        out.phi = vals[idx]; out.gx = grads[idx][0]; out.gy = grads[idx][1];
        return out;
    }
    tau = std::max(tau, EPS);
    double m = -std::numeric_limits<double>::infinity();
    for (double v : vals) m = std::max(m, -v / tau);
    double z = 0.0;
    for (double v : vals) z += std::exp(-v / tau - m);
    z = std::max(z, 1.0e-300);
    out.phi = -tau * (m + std::log(z));
    for (size_t i = 0; i < vals.size(); ++i) {
        const double w = std::exp(-vals[i] / tau - m) / z;
        out.gx += w * grads[i][0]; out.gy += w * grads[i][1];
    }
    return out;
}

std::pair<double,std::array<double,2>> mf_normalized_smooth_max(
    const std::vector<double>& vals,
    const std::vector<std::array<double,2>>& grads,
    double alpha)
{
    if (vals.empty()) return {-std::numeric_limits<double>::infinity(), {{0.0,0.0}}};
    alpha = std::max(alpha, EPS);
    double m = -std::numeric_limits<double>::infinity();
    for (double v : vals) m = std::max(m, alpha * v);
    double z = 0.0;
    for (double v : vals) z += std::exp(alpha * v - m);
    z = std::max(z, 1.0e-300);
    std::array<double,2> g{{0.0,0.0}};
    for (size_t i = 0; i < vals.size(); ++i) {
        const double w = std::exp(alpha * vals[i] - m) / z;
        g[0] += w * grads[i][0]; g[1] += w * grads[i][1];
    }
    return {(m + std::log(z) - std::log(static_cast<double>(std::max<size_t>(vals.size(),1U)))) / alpha, g};
}

std::vector<MFPoint> mf_polygon_vertex_edge_points(const std::vector<MFPoint>& V) {
    std::vector<MFPoint> samples;
    if (V.empty()) return samples;
    samples.reserve(V.size() * 2U);
    for (const auto& p : V) samples.push_back(p);
    if (V.size() >= 2U) {
        for (size_t i = 0; i < V.size(); ++i) {
            const auto& a = V[i];
            const auto& b = V[(i + 1U) % V.size()];
            samples.push_back({0.5 * (a.x + b.x), 0.5 * (a.y + b.y)});
        }
    }
    return samples;
}

MFFieldEval mf_circle_field_local(
    const std::vector<MFGroup>& groups_local,
    double qx,
    double qy,
    double union_lambda)
{
    // C-SDF in the same ego body-fixed normalized frame as the proposed field:
    //     q = S^{-1} R(yaw)^T (P-p).
    // The ego circumscribed ellipse is the unit disk in this frame.  Hence an
    // obstacle enclosing circle (c_m,r_m) gives the ego-aware clearance
    //     ||q-c_m|| - r_m - 1.
    // The exact minimum across obstacle circles is retained as in the baseline.
    std::vector<double> vals;
    std::vector<std::array<double,2>> grads;
    vals.reserve(groups_local.size());
    grads.reserve(groups_local.size());
    for (const auto& g : groups_local) {
        if (g.V.empty()) continue;
        double cx = 0.0, cy = 0.0;
        for (const auto& p : g.V) { cx += p.x; cy += p.y; }
        cx /= static_cast<double>(g.V.size());
        cy /= static_cast<double>(g.V.size());
        double radius = 0.0;
        for (const auto& p : g.V) radius = std::max(radius, std::hypot(p.x - cx, p.y - cy));
        const double dx = qx - cx, dy = qy - cy, dist = std::hypot(dx, dy);
        vals.push_back(dist - radius - 1.0);
        grads.push_back(dist > EPS ? std::array<double,2>{{dx/dist,dy/dist}}
                                   : std::array<double,2>{{0.0,0.0}});
    }
    (void)union_lambda;
    return mf_smooth_min(vals, grads, 0.0);
}

MFFieldEval mf_dcsat_field(
    const std::vector<MFGroup>& groups,
    double margin_local,
    double alpha,
    double union_lambda)
{
    std::vector<double> vals;
    std::vector<std::array<double,2>> grads;
    for (const auto& g : groups) {
        const size_t n = g.V.size(); if (n < 3U) continue;
        std::vector<double> terms; terms.reserve(2U*n);
        std::vector<std::array<double,2>> deriv; deriv.reserve(2U*n);
        std::vector<double> maxproj(n, -std::numeric_limits<double>::infinity());
        std::vector<double> minproj(n, std::numeric_limits<double>::infinity());
        std::vector<std::array<double,2>> axes(n);
        for (size_t i=0;i<n;++i) {
            const auto& a=g.V[i]; const auto& b=g.V[(i+1U)%n];
            double ax=b.y-a.y, ay=-(b.x-a.x); const double nn=std::max(std::hypot(ax,ay),EPS); ax/=nn; ay/=nn;
            axes[i]={{ax,ay}};
            for (const auto& p:g.V) { const double pr=p.x*ax+p.y*ay; maxproj[i]=std::max(maxproj[i],pr); minproj[i]=std::min(minproj[i],pr); }
        }
        for(size_t i=0;i<n;++i){ terms.push_back(-maxproj[i]); deriv.push_back(axes[i]); }
        for(size_t i=0;i<n;++i){ terms.push_back(minproj[i]); deriv.push_back({{-axes[i][0],-axes[i][1]}}); }
        auto sm=mf_normalized_smooth_max(terms,deriv,alpha);
        vals.push_back(sm.first - 1.0 - (g.dynamic?margin_local:0.0)); grads.push_back(sm.second);
    }
    return mf_smooth_min(vals,grads,union_lambda);
}

MFFieldEval mf_wu_field(
    const std::vector<MFGroup>& groups,
    double margin_local,
    double kappa,
    double union_lambda,
    double buffer_offset)
{
    std::vector<double> vals;
    std::vector<std::array<double,2>> grads;
    kappa=std::max(kappa,EPS);
    for(const auto& g:groups){
        const size_t n=g.V.size(); if(n<3U) continue;
        std::vector<double> atomic(n); std::vector<std::array<double,2>> A(n);
        double m=-std::numeric_limits<double>::infinity();
        for(size_t i=0;i<n;++i){
            const auto&a=g.V[i]; const auto&b=g.V[(i+1U)%n];
            double ax=b.y-a.y, ay=-(b.x-a.x); const double nn=std::max(std::hypot(ax,ay),EPS); ax/=nn; ay/=nn;
            A[i]={{ax,ay}}; const double bi=ax*a.x+ay*a.y; atomic[i]=-bi; m=std::max(m,kappa*atomic[i]);
        }
        double z=0.0; for(double v:atomic) z+=std::exp(kappa*v-m); z=std::max(z,1.0e-300);
        std::array<double,2> gg{{0.0,0.0}};
        for(size_t i=0;i<n;++i){ const double w=std::exp(kappa*atomic[i]-m)/z; gg[0]+=w*A[i][0]; gg[1]+=w*A[i][1]; }
        const double buffer=std::log(static_cast<double>(std::max<size_t>(n,1U)))+buffer_offset;
        vals.push_back((m+std::log(z)-buffer)/kappa - 1.0 - (g.dynamic?margin_local:0.0)); grads.push_back(gg);
    }
    return mf_smooth_min(vals,grads,union_lambda);
}

inline double mf_binom(int n,int k){
    if (k < 0 || k > n) return 0.0;
    k = std::min(k, n - k);
    double r = 1.0;
    for(int i=1;i<=k;++i) r*=static_cast<double>(n-k+i)/static_cast<double>(i);
    return r;
}

void mf_bernstein_basis(double x,int Q,std::vector<double>&B,std::vector<double>&dB){
    x=clampd(x,0.0,1.0); const int degree=std::max(Q-1,0); B.assign(static_cast<size_t>(Q),0.0); dB.assign(static_cast<size_t>(Q),0.0);
    for(int q=0;q<Q;++q) B[static_cast<size_t>(q)]=mf_binom(degree,q)*std::pow(x,q)*std::pow(1.0-x,degree-q);
    if(degree<=0) return;
    std::vector<double> low(static_cast<size_t>(degree),0.0);
    for(int q=0;q<degree;++q) low[static_cast<size_t>(q)]=mf_binom(degree-1,q)*std::pow(x,q)*std::pow(1.0-x,degree-1-q);
    for(int q=0;q<Q;++q){ const double left=q>=1?low[static_cast<size_t>(q-1)]:0.0; const double right=q<=degree-1?low[static_cast<size_t>(q)]:0.0; dB[static_cast<size_t>(q)]=degree*(left-right); }
}

// Evaluate the fitted Bernstein scalar field in its q chart.  This helper is
// used only to define/extract the fitted obstacle boundary B(q)=l.
inline bool mf_bp_raw_q(
    const MFBPModel& m, double qx, double qy,
    double& raw, double& gx, double& gy)
{
    raw = 0.0; gx = 0.0; gy = 0.0;
    if (m.empty || m.Q < 2 || static_cast<int>(m.W.size()) < m.Q * m.Q) return false;
    const double dxr = std::max(m.xr[1] - m.xr[0], EPS);
    const double dyr = std::max(m.yr[1] - m.yr[0], EPS);
    const double xi = (qx - m.xr[0]) / dxr;
    const double eta = (qy - m.yr[0]) / dyr;
    std::vector<double> Bx, dBx, By, dBy;
    mf_bernstein_basis(xi, m.Q, Bx, dBx);
    mf_bernstein_basis(eta, m.Q, By, dBy);
    for (int ix = 0; ix < m.Q; ++ix) {
        for (int iy = 0; iy < m.Q; ++iy) {
            const double w = m.W[static_cast<size_t>(ix * m.Q + iy)];
            raw += Bx[static_cast<size_t>(ix)] * w * By[static_cast<size_t>(iy)];
            gx += dBx[static_cast<size_t>(ix)] * w * By[static_cast<size_t>(iy)];
            gy += Bx[static_cast<size_t>(ix)] * w * dBy[static_cast<size_t>(iy)];
        }
    }
    gx /= dxr; gy /= dyr;
    return std::isfinite(raw) && std::isfinite(gx) && std::isfinite(gy);
}

inline std::array<double,2> mf_bp_level_crossing(
    double x0, double y0, double f0,
    double x1, double y1, double f1)
{
    double t = 0.5;
    const double den = f0 - f1;
    if (std::abs(den) > 1.0e-14) t = clampd(f0 / den, 0.0, 1.0);
    return {{x0 + t * (x1 - x0), y0 + t * (y1 - y0)}};
}

void mf_build_bp_boundary_segments(MFBPModel& m)
{
    m.boundary_segments.clear();
    if (m.empty || m.Q < 2) return;
    const double dxr = m.xr[1] - m.xr[0];
    const double dyr = m.yr[1] - m.yr[0];
    if (!(dxr > EPS) || !(dyr > EPS)) return;

    // This is only a geometric extraction of the already-fitted level set.
    // A moderately dense grid is enough because distance queries are then made
    // to the resulting line segments, not to the grid samples themselves.
    const int n = std::max(64, 6 * m.Q);
    std::vector<double> F(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    for (int iy = 0; iy < n; ++iy) {
        const double y = m.yr[0] + static_cast<double>(iy) / static_cast<double>(n - 1) * dyr;
        for (int ix = 0; ix < n; ++ix) {
            const double x = m.xr[0] + static_cast<double>(ix) / static_cast<double>(n - 1) * dxr;
            double raw = 0.0, gx = 0.0, gy = 0.0;
            if (!mf_bp_raw_q(m, x, y, raw, gx, gy)) raw = m.safety_margin + 1.0;
            F[static_cast<size_t>(iy) * n + ix] = raw - m.safety_margin;
        }
    }

    for (int iy = 0; iy < n - 1; ++iy) {
        const double y0 = m.yr[0] + static_cast<double>(iy) / static_cast<double>(n - 1) * dyr;
        const double y1 = m.yr[0] + static_cast<double>(iy + 1) / static_cast<double>(n - 1) * dyr;
        for (int ix = 0; ix < n - 1; ++ix) {
            const double x0 = m.xr[0] + static_cast<double>(ix) / static_cast<double>(n - 1) * dxr;
            const double x1 = m.xr[0] + static_cast<double>(ix + 1) / static_cast<double>(n - 1) * dxr;
            const double f00 = F[static_cast<size_t>(iy) * n + ix];
            const double f10 = F[static_cast<size_t>(iy) * n + (ix + 1)];
            const double f11 = F[static_cast<size_t>(iy + 1) * n + (ix + 1)];
            const double f01 = F[static_cast<size_t>(iy + 1) * n + ix];
            std::vector<std::array<double,2>> c;
            c.reserve(4);
            auto add_edge = [&](double ax, double ay, double fa, double bx, double by, double fb) {
                if (!std::isfinite(fa) || !std::isfinite(fb)) return;
                if (fa == 0.0 && fb == 0.0) return;
                if (fa == 0.0 || fb == 0.0 || ((fa < 0.0) != (fb < 0.0)))
                    c.push_back(mf_bp_level_crossing(ax, ay, fa, bx, by, fb));
            };
            add_edge(x0, y0, f00, x1, y0, f10);
            add_edge(x1, y0, f10, x1, y1, f11);
            add_edge(x1, y1, f11, x0, y1, f01);
            add_edge(x0, y1, f01, x0, y0, f00);
            if (c.size() >= 2U) {
                const size_t pairs = c.size() / 2U;
                for (size_t k = 0; k < pairs; ++k) {
                    const auto& a = c[2U * k];
                    const auto& b = c[2U * k + 1U];
                    m.boundary_segments.push_back({{a[0], a[1], b[0], b[1]}});
                }
            }
        }
    }
}


// BP fitting evaluates the same local convex polygons at every training sample.
// Precompute edge vectors/inverse squared lengths once per model so the hot
// sampling loop does not rebuild them for each of the nx*ny CDF/SDF queries.
struct MFBPPreparedPolygon {
    bool dynamic = false;
    std::vector<double> ax, ay, bx, by, ex, ey, inv_ee;
};

std::vector<MFBPPreparedPolygon> mf_prepare_bp_polygons(const std::vector<MFGroup>& groups_local) {
    std::vector<MFBPPreparedPolygon> out;
    out.reserve(groups_local.size());
    for (const auto& g : groups_local) {
        const size_t n = g.V.size();
        if (n < 3U) continue;
        MFBPPreparedPolygon p;
        p.dynamic = g.dynamic;
        p.ax.resize(n); p.ay.resize(n); p.bx.resize(n); p.by.resize(n);
        p.ex.resize(n); p.ey.resize(n); p.inv_ee.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const MFPoint& a = g.V[i];
            const MFPoint& b = g.V[(i + 1U) % n];
            const double ex = b.x - a.x, ey = b.y - a.y;
            const double ee = ex * ex + ey * ey;
            p.ax[i] = a.x; p.ay[i] = a.y; p.bx[i] = b.x; p.by[i] = b.y;
            p.ex[i] = ex; p.ey[i] = ey;
            p.inv_ee[i] = ee > 1.0e-14 ? 1.0 / ee : 0.0;
        }
        out.push_back(std::move(p));
    }
    return out;
}

inline double mf_signed_distance_prepared(
    double px, double py, const MFBPPreparedPolygon& p)
{
    const size_t n = p.ax.size();
    if (n < 3U) return std::numeric_limits<double>::infinity();
    double min_sq = std::numeric_limits<double>::infinity();
    bool inside = false;
    for (size_t i = 0; i < n; ++i) {
        double tau = 0.0;
        if (p.inv_ee[i] > 0.0) {
            tau = ((px - p.ax[i]) * p.ex[i] + (py - p.ay[i]) * p.ey[i]) * p.inv_ee[i];
            tau = clampd(tau, 0.0, 1.0);
        }
        const double cx = p.ax[i] + tau * p.ex[i];
        const double cy = p.ay[i] + tau * p.ey[i];
        const double dx = px - cx, dy = py - cy;
        min_sq = std::min(min_sq, dx * dx + dy * dy);

        const bool ycross = ((p.ay[i] > py) != (p.by[i] > py));
        if (ycross) {
            const double den = p.by[i] - p.ay[i];
            const double xcross = p.ex[i] * (py - p.ay[i]) / (den + 1.0e-300) + p.ax[i];
            if (px < xcross) inside = !inside;
        }
    }
    double d = std::sqrt(std::max(min_sq, 0.0));
    return inside ? -d : d;
}

inline double mf_exact_union_sdf_point_prepared(
    double px, double py,
    const std::vector<MFBPPreparedPolygon>& groups_world)
{
    double best = std::numeric_limits<double>::infinity();
    for (const auto& g : groups_world) {
        const double sd = mf_signed_distance_prepared(px, py, g);
        if (!std::isfinite(sd)) continue;
        best = std::min(best, sd);
    }
    return best;
}

bool mf_cholesky_factor_spd(const std::vector<double>& A, int n, std::vector<double>& L) {
    if (n <= 0 || static_cast<int>(A.size()) != n * n) return false;
    L.assign(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[static_cast<size_t>(i) * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[static_cast<size_t>(i) * n + k] * L[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                L[static_cast<size_t>(i) * n + j] = std::sqrt(sum);
            } else {
                const double den = L[static_cast<size_t>(j) * n + j];
                if (!(den > 0.0) || !std::isfinite(den)) return false;
                L[static_cast<size_t>(i) * n + j] = sum / den;
            }
        }
    }
    return true;
}

bool mf_cholesky_solve_factor(const std::vector<double>& L, const std::vector<double>& b, int n, std::vector<double>& x) {
    if (n <= 0 || static_cast<int>(b.size()) != n || static_cast<int>(L.size()) != n * n) return false;
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = b[static_cast<size_t>(i)];
        for (int k = 0; k < i; ++k) sum -= L[static_cast<size_t>(i) * n + k] * y[static_cast<size_t>(k)];
        const double den = L[static_cast<size_t>(i) * n + i];
        if (!(den > 0.0) || !std::isfinite(den)) return false;
        y[static_cast<size_t>(i)] = sum / den;
    }
    x.assign(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[static_cast<size_t>(i)];
        for (int k = i + 1; k < n; ++k) sum -= L[static_cast<size_t>(k) * n + i] * x[static_cast<size_t>(k)];
        const double den = L[static_cast<size_t>(i) * n + i];
        if (!(den > 0.0) || !std::isfinite(den)) return false;
        x[static_cast<size_t>(i)] = sum / den;
    }
    return true;
}

struct MFBPThinQR {
    int rows = 0;
    int cols = 0;
    // Householder QR in LAPACK-like compact form: diagonal/upper triangle is R,
    // entries below the diagonal store v[1:], and tau stores each reflector.
    std::vector<double> qr;
    std::vector<double> tau;
};

bool mf_build_thin_qr(const std::vector<std::vector<double>>& B, MFBPThinQR& out) {
    const int m = static_cast<int>(B.size());
    const int n = m > 0 ? static_cast<int>(B[0].size()) : 0;
    if (m <= 0 || n <= 0 || m < n) return false;
    out.rows = m;
    out.cols = n;
    out.qr.assign(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0);
    out.tau.assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < m; ++i) {
        if (static_cast<int>(B[static_cast<size_t>(i)].size()) != n) return false;
        for (int j = 0; j < n; ++j) out.qr[static_cast<size_t>(i) * n + j] = B[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
    for (int k = 0; k < n; ++k) {
        double norm = 0.0;
        for (int i = k; i < m; ++i) norm = std::hypot(norm, out.qr[static_cast<size_t>(i) * n + k]);
        if (!(norm > 0.0) || !std::isfinite(norm)) return false;
        const double alpha = out.qr[static_cast<size_t>(k) * n + k];
        const double beta = -std::copysign(norm, alpha == 0.0 ? 1.0 : alpha);
        const double v0 = alpha - beta;
        if (std::abs(v0) <= 1.0e-300) return false;
        const double tau = (beta - alpha) / beta;
        out.tau[static_cast<size_t>(k)] = tau;
        out.qr[static_cast<size_t>(k) * n + k] = beta;
        for (int i = k + 1; i < m; ++i) out.qr[static_cast<size_t>(i) * n + k] /= v0;
        for (int j = k + 1; j < n; ++j) {
            double dot = out.qr[static_cast<size_t>(k) * n + j];
            for (int i = k + 1; i < m; ++i) {
                dot += out.qr[static_cast<size_t>(i) * n + k] * out.qr[static_cast<size_t>(i) * n + j];
            }
            dot *= tau;
            out.qr[static_cast<size_t>(k) * n + j] -= dot;
            for (int i = k + 1; i < m; ++i) {
                out.qr[static_cast<size_t>(i) * n + j] -= out.qr[static_cast<size_t>(i) * n + k] * dot;
            }
        }
    }
    return true;
}

bool mf_qr_least_squares(const MFBPThinQR& qr, const std::vector<double>& b, std::vector<double>& x) {
    const int m = qr.rows, n = qr.cols;
    if (m <= 0 || n <= 0 || m < n || static_cast<int>(b.size()) != m) return false;
    std::vector<double> y = b;
    // Apply Q^T to b in the same reflector order used during factorization.
    for (int k = 0; k < n; ++k) {
        double dot = y[static_cast<size_t>(k)];
        for (int i = k + 1; i < m; ++i) dot += qr.qr[static_cast<size_t>(i) * n + k] * y[static_cast<size_t>(i)];
        dot *= qr.tau[static_cast<size_t>(k)];
        y[static_cast<size_t>(k)] -= dot;
        for (int i = k + 1; i < m; ++i) y[static_cast<size_t>(i)] -= qr.qr[static_cast<size_t>(i) * n + k] * dot;
    }
    x.assign(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double rhs = y[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) rhs -= qr.qr[static_cast<size_t>(i) * n + j] * x[static_cast<size_t>(j)];
        const double den = qr.qr[static_cast<size_t>(i) * n + i];
        if (std::abs(den) <= 1.0e-300 || !std::isfinite(den)) return false;
        x[static_cast<size_t>(i)] = rhs / den;
    }
    return true;
}

struct MFBPFitCache {
    int Q = 2;
    int nx = 4;
    int ny = 4;
    std::vector<std::vector<double>> Bx;
    std::vector<std::vector<double>> By;
    MFBPThinQR qrx;
    MFBPThinQR qry;
};

MFBPFitCache mf_build_bp_fit_cache(int bp_order, int bp_training_n_x, int bp_training_n_y) {
    MFBPFitCache cache;
    cache.Q = std::max(bp_order, 2);
    cache.nx = std::max(bp_training_n_x, cache.Q + 2);
    cache.ny = std::max(bp_training_n_y, cache.Q + 2);
    cache.Bx.resize(static_cast<size_t>(cache.nx));
    cache.By.resize(static_cast<size_t>(cache.ny));
    std::vector<double> dummy;
    for (int ix = 0; ix < cache.nx; ++ix) {
        const double xi = cache.nx <= 1 ? 0.0 : static_cast<double>(ix) / static_cast<double>(cache.nx - 1);
        mf_bernstein_basis(xi, cache.Q, cache.Bx[static_cast<size_t>(ix)], dummy);
    }
    for (int iy = 0; iy < cache.ny; ++iy) {
        const double eta = cache.ny <= 1 ? 0.0 : static_cast<double>(iy) / static_cast<double>(cache.ny - 1);
        mf_bernstein_basis(eta, cache.Q, cache.By[static_cast<size_t>(iy)], dummy);
    }
    if (!mf_build_thin_qr(cache.Bx, cache.qrx) || !mf_build_thin_qr(cache.By, cache.qry)) {
        throw std::runtime_error("Bernstein ordinary-LS QR factorization failed");
    }
    return cache;
}

const MFBPFitCache& mf_get_bp_fit_cache(int bp_order, int bp_training_n_x, int bp_training_n_y) {
    const int Q = std::max(bp_order, 2);
    const int nx = std::max(bp_training_n_x, Q + 2);
    const int ny = std::max(bp_training_n_y, Q + 2);
    // The normalized Bernstein design depends only on (Q,nx,ny).  Reuse its
    // thin QR factors across fits; obstacle geometry changes only the target Y.
    thread_local std::vector<MFBPFitCache> caches;
    for (const auto& cache : caches) {
        if (cache.Q == Q && cache.nx == nx && cache.ny == ny) return cache;
    }
    caches.push_back(mf_build_bp_fit_cache(Q, nx, ny));
    return caches.back();
}

MFBPModel mf_fit_bp_model_cpp(
    const std::vector<MFGroup>& groups_world,
    int bp_order,
    int bp_training_n_x,
    int bp_training_n_y,
    double bp_safety_margin_l,
    double bp_domain_padding,
    double world_x_min,
    double world_x_max,
    double world_y_min,
    double world_y_max,
    const MFBPFitCache* fit_cache)
{
    MFBPModel model;
    model.Q = std::max(bp_order, 2);
    model.safety_margin = bp_safety_margin_l;
    // MATLAB-aligned world-frame chart: q_ref == p_world.
    model.pref = {{0.0, 0.0}};
    model.Tref = {{1.0, 0.0, 0.0, 1.0}};
    model.W.assign(static_cast<size_t>(model.Q) * static_cast<size_t>(model.Q), 0.0);
    model.empty = groups_world.empty();
    if (model.empty) return model;

    // Fit each obstacle only in a local world-frame box around its geometry.
    // This preserves enough inside/outside samples for the fitted l-level set
    // to remain a meaningful closed boundary even for small obstacles.
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (const auto& g : groups_world) {
        for (const auto& v : g.V) {
            if (!std::isfinite(v.x) || !std::isfinite(v.y)) continue;
            xmin = std::min(xmin, v.x); xmax = std::max(xmax, v.x);
            ymin = std::min(ymin, v.y); ymax = std::max(ymax, v.y);
        }
    }
    if (!std::isfinite(xmin) || !std::isfinite(xmax) ||
        !std::isfinite(ymin) || !std::isfinite(ymax)) {
        xmin = world_x_min; xmax = world_x_max;
        ymin = world_y_min; ymax = world_y_max;
    }
    if (!(xmax - xmin > 1.0e-9) || !(ymax - ymin > 1.0e-9)) {
        throw std::runtime_error("BP-SDF obstacle fitting domain is degenerate");
    }
    const double pad = std::max(std::abs(bp_domain_padding), std::abs(bp_safety_margin_l) + 1.0e-3);
    model.xr = {{xmin - pad, xmax + pad}};
    model.yr = {{ymin - pad, ymax + pad}};

    MFBPFitCache local_cache;
    const MFBPFitCache* cache = fit_cache;
    if (cache == nullptr || cache->Q != model.Q) {
        local_cache = mf_build_bp_fit_cache(model.Q, bp_training_n_x, bp_training_n_y);
        cache = &local_cache;
    }
    const int nx = cache->nx;
    const int ny = cache->ny;
    const double dxr = model.xr[1] - model.xr[0];
    const double dyr = model.yr[1] - model.yr[0];
    const auto prepared_groups = mf_prepare_bp_polygons(groups_world);

    // Sample the exact WORLD-frame polygon-union SDF on the same uniform
    // Cartesian grid used by the MATLAB implementation.  robot_radius=0, so
    // the target is S_O(p); the user level l is subtracted only at evaluation.
    std::vector<double> target(static_cast<size_t>(ny) * static_cast<size_t>(nx), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((static_cast<long long>(nx) * ny >= 512LL) && !omp_in_parallel())
#endif
    for (int iy = 0; iy < ny; ++iy) {
        const double py = model.yr[0] + (ny <= 1 ? 0.0 : static_cast<double>(iy) / static_cast<double>(ny - 1)) * dyr;
        for (int ix = 0; ix < nx; ++ix) {
            const double px = model.xr[0] + (nx <= 1 ? 0.0 : static_cast<double>(ix) / static_cast<double>(nx - 1)) * dxr;
            target[static_cast<size_t>(iy) * nx + ix] = mf_exact_union_sdf_point_prepared(px, py, prepared_groups);
        }
    }

    // Ordinary least squares on the tensor-product Bernstein design.
    // MATLAB solves vec(Y) ~= (Bx kron By) w with backslash.  Because the
    // design is separable, the same unweighted OLS solution is obtained by
    // two cached thin-QR solves: first along x, then along y.  No ridge term.
    std::vector<double> Z(static_cast<size_t>(ny) * static_cast<size_t>(model.Q), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(ny >= 16 && !omp_in_parallel())
#endif
    for (int iy = 0; iy < ny; ++iy) {
        std::vector<double> rhs(static_cast<size_t>(nx));
        for (int ix = 0; ix < nx; ++ix) rhs[static_cast<size_t>(ix)] = target[static_cast<size_t>(iy) * nx + ix];
        std::vector<double> sol;
        if (!mf_qr_least_squares(cache->qrx, rhs, sol)) continue;
        for (int qx = 0; qx < model.Q; ++qx) Z[static_cast<size_t>(iy) * model.Q + qx] = sol[static_cast<size_t>(qx)];
    }

    for (int qx = 0; qx < model.Q; ++qx) {
        std::vector<double> rhs(static_cast<size_t>(ny));
        for (int iy = 0; iy < ny; ++iy) rhs[static_cast<size_t>(iy)] = Z[static_cast<size_t>(iy) * model.Q + qx];
        std::vector<double> sol;
        if (!mf_qr_least_squares(cache->qry, rhs, sol)) throw std::runtime_error("Bernstein ordinary-LS solve failed");
        for (int qy = 0; qy < model.Q; ++qy) {
            // Existing ABI/evaluator layout: W[qx,qy] multiplies Bx[qx]*By[qy].
            model.W[static_cast<size_t>(qx) * model.Q + qy] = sol[static_cast<size_t>(qy)];
        }
    }
    model.empty = false;
    return model;
}

py::tuple fit_bp_models_seq(
    const py::array_t<double,py::array::forcecast>& X_bar,
    const py::array_t<double,py::array::forcecast>& vertex_set,
    const py::array_t<int32_t,py::array::forcecast>& vertex_types,
    const py::array_t<int32_t,py::array::forcecast>& feature_obs_indices,
    const py::array_t<double,py::array::forcecast>& h_steps,
    const py::array_t<double,py::array::forcecast>& dt_vals,
    const py::array_t<double,py::array::forcecast>& axis_scale,
    bool has_vertex_types,
    bool has_feature_obs_indices,
    int vertex_capacity,
    double dt_init,
    double ego_L,
    double ego_W,
    double sigma_vx,
    double sigma_vy,
    double p_col,
    bool use_probability_margin,
    int bp_order,
    int bp_training_n_x,
    int bp_training_n_y,
    double bp_safety_margin_l,
    double bp_domain_padding,
    int horizon_N,
    double v_max,
    double a_min,
    double a_max,
    double world_x_min,
    double world_x_max,
    double world_y_min,
    double world_y_max)
{
    // World-frame fitted-boundary BP-SDF baseline.
    //
    // IMPORTANT: each obstacle is fitted exactly once from its k=0 WORLD-frame
    // polygon.  For later horizon stages the fitted field is transported only
    // by translation:
    //
    //     phi_{m,k}(p) = phi_{m,0}(p - Delta p_{m,k}).
    //
    // The evaluator already uses q = T_ref * (p - p_ref), so the Bernstein
    // coefficients W, T_ref, and the fitting domain remain unchanged; only the
    // per-stage/per-obstacle p_ref is shifted by the obstacle translation.
    // This removes the previous H-times repeated BP fitting for stage-wise
    // predicted polygons.
    //
    // Legacy arguments are retained in the pybind ABI.  The fit itself does
    // not depend on ego scale/state, horizon reach, or probability inflation.
    // bp_domain_padding defines the obstacle-local fitting chart.
    (void)dt_init; (void)ego_L; (void)ego_W; (void)sigma_vx; (void)sigma_vy;
    (void)p_col; (void)use_probability_margin;
    (void)horizon_N; (void)v_max; (void)a_min; (void)a_max;
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const VecViewI types = view1d_int(vertex_types), obs = view1d_int(feature_obs_indices);
    (void)h_steps; (void)dt_vals; (void)axis_scale;
    const int K = std::max(0, std::min(vertex_capacity, static_cast<int>(V.K)));
    const int Q = std::max(bp_order, 2);

    // Shared Kx2 geometry needs one model set.  Stage-wise HxKx2 geometry still
    // returns H model sets because p_ref changes with stage, but W is fitted only
    // from stage 0 and copied to all later stages.
    const ssize_t Ho = (H > 0 && V.stagewise) ? H : 1;
    std::vector<std::vector<MFGroup>> stage_groups(static_cast<size_t>(Ho));
    if (H > 0 && K > 0) {
        stage_groups[0] = mf_build_world_groups(
            V, 0, K, types, obs, has_vertex_types, has_feature_obs_indices, true);
        if (V.stagewise) {
            for (ssize_t h = 1; h < H; ++h) {
                stage_groups[static_cast<size_t>(h)] = mf_build_world_groups(
                    V, h, K, types, obs, has_vertex_types, has_feature_obs_indices, true);
            }
        }
    }

    const auto& base_groups = stage_groups[0];
    const ssize_t Omax = std::max<ssize_t>(1, static_cast<ssize_t>(base_groups.size()));

    py::array_t<double> W({Ho, Omax, static_cast<ssize_t>(Q), static_cast<ssize_t>(Q)});
    py::array_t<double> T_ref({Ho, Omax, static_cast<ssize_t>(2), static_cast<ssize_t>(2)});
    py::array_t<double> p_ref({Ho, Omax, static_cast<ssize_t>(2)});
    py::array_t<double> x_range({Ho, Omax, static_cast<ssize_t>(2)});
    py::array_t<double> y_range({Ho, Omax, static_cast<ssize_t>(2)});
    py::array_t<int32_t> empty({Ho, Omax});
    auto Wb = W.request(); auto Tb = T_ref.request(); auto pb = p_ref.request();
    auto xrb = x_range.request(); auto yrb = y_range.request(); auto eb = empty.request();
    double* Wp = static_cast<double*>(Wb.ptr); double* Tp = static_cast<double*>(Tb.ptr);
    double* pp = static_cast<double*>(pb.ptr); double* xrp = static_cast<double*>(xrb.ptr);
    double* yrp = static_cast<double*>(yrb.ptr); int32_t* ep = static_cast<int32_t*>(eb.ptr);
    std::fill(Wp, Wp + static_cast<size_t>(Ho) * static_cast<size_t>(Omax) * Q * Q, 0.0);
    std::fill(Tp, Tp + static_cast<size_t>(Ho) * static_cast<size_t>(Omax) * 4U, 0.0);
    std::fill(pp, pp + static_cast<size_t>(Ho) * static_cast<size_t>(Omax) * 2U, 0.0);
    std::fill(xrp, xrp + static_cast<size_t>(Ho) * static_cast<size_t>(Omax) * 2U, 0.0);
    std::fill(yrp, yrp + static_cast<size_t>(Ho) * static_cast<size_t>(Omax) * 2U, 0.0);
    std::fill(ep, ep + static_cast<size_t>(Ho) * static_cast<size_t>(Omax), static_cast<int32_t>(1));

    const MFBPFitCache& fit_cache = mf_get_bp_fit_cache(Q, bp_training_n_x, bp_training_n_y);

    auto store_model = [&](ssize_t h, ssize_t o, const MFBPModel& m) {
        const size_t so = static_cast<size_t>(h) * static_cast<size_t>(Omax) + static_cast<size_t>(o);
        ep[so] = m.empty ? 1 : 0;
        for (int iy = 0; iy < Q; ++iy) for (int ix = 0; ix < Q; ++ix) {
            Wp[((so * static_cast<size_t>(Q) + static_cast<size_t>(iy)) * static_cast<size_t>(Q)) + static_cast<size_t>(ix)] =
                m.W[static_cast<size_t>(iy * Q + ix)];
        }
        for (int j = 0; j < 4; ++j) Tp[so * 4U + static_cast<size_t>(j)] = m.Tref[static_cast<size_t>(j)];
        pp[so * 2U + 0U] = m.pref[0]; pp[so * 2U + 1U] = m.pref[1];
        xrp[so * 2U + 0U] = m.xr[0]; xrp[so * 2U + 1U] = m.xr[1];
        yrp[so * 2U + 0U] = m.yr[0]; yrp[so * 2U + 1U] = m.yr[1];
    };

    if (H <= 0) {
        MFBPModel m; m.Q = Q; m.W.assign(static_cast<size_t>(Q * Q), 0.0);
        store_model(0, 0, m);
        return py::make_tuple(W, T_ref, p_ref, x_range, y_range, empty);
    }

    // Fit k=0 only, one independent polynomial per obstacle.
    std::vector<MFBPModel> base_models(base_groups.size());
    {
        py::gil_scoped_release release;
        int failed = 0;
        std::string failure_message;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(base_groups.size() >= 2U)
#endif
        for (ssize_t o = 0; o < static_cast<ssize_t>(base_groups.size()); ++o) {
            try {
                std::vector<MFGroup> one_group{base_groups[static_cast<size_t>(o)]};
                base_models[static_cast<size_t>(o)] = mf_fit_bp_model_cpp(
                    one_group, Q, bp_training_n_x, bp_training_n_y, bp_safety_margin_l,
                    bp_domain_padding, world_x_min, world_x_max, world_y_min, world_y_max, &fit_cache);
            } catch (const std::exception& exc) {
#ifdef _OPENMP
#pragma omp critical(mf_bp_fit_error)
#endif
                {
                    failed = 1;
                    if (failure_message.empty()) failure_message = exc.what();
                }
            }
        }
        if (failed) throw std::runtime_error("Bernstein obstacle-wise k=0 fit failed: " + failure_message);
    }

    auto centroid = [](const MFGroup& g) {
        MFPoint c{0.0, 0.0};
        if (g.V.empty()) return c;
        for (const auto& p : g.V) { c.x += p.x; c.y += p.y; }
        const double inv = 1.0 / static_cast<double>(g.V.size());
        c.x *= inv; c.y *= inv;
        return c;
    };

    std::vector<MFPoint> base_centroids(base_groups.size());
    for (size_t o = 0; o < base_groups.size(); ++o) base_centroids[o] = centroid(base_groups[o]);

    // Stage 0 stores the fitted models directly.  Later stages reuse the exact
    // same coefficients and shift only p_ref by the obstacle centroid motion.
    for (ssize_t h = 0; h < Ho; ++h) {
        std::unordered_map<int, const MFGroup*> current_by_id;
        if (h > 0) {
            const auto& current = stage_groups[static_cast<size_t>(h)];
            current_by_id.reserve(current.size() * 2U + 1U);
            for (const auto& g : current) current_by_id.emplace(g.id, &g);
        }

        for (size_t o = 0; o < base_models.size(); ++o) {
            MFBPModel model = base_models[o];
            if (h > 0) {
                const auto it = current_by_id.find(base_groups[o].id);
                if (it == current_by_id.end()) {
                    throw std::runtime_error(
                        "BP-SDF translation reuse requires the same obstacle IDs at every horizon stage");
                }
                const MFPoint ck = centroid(*it->second);
                const double tx = ck.x - base_centroids[o].x;
                const double ty = ck.y - base_centroids[o].y;
                model.pref[0] += tx;
                model.pref[1] += ty;
            }
            store_model(h, static_cast<ssize_t>(o), model);
        }
    }

    return py::make_tuple(W, T_ref, p_ref, x_range, y_range, empty);
}

MFBPModel mf_load_bp_model(
    ssize_t h,
    ssize_t o,
    const py::buffer_info& Wb,
    const py::buffer_info& Tb,
    const py::buffer_info& pb,
    const py::buffer_info& xrb,
    const py::buffer_info& yrb,
    const py::buffer_info& eb,
    int Q,
    double safety_margin);

inline ssize_t mf_bp_obstacle_count(const py::buffer_info& Wb) {
    if (Wb.ndim == 4 && Wb.shape[1] > 0) return Wb.shape[1];
    if (Wb.ndim == 3 && Wb.shape[0] > 0) return 1;  // legacy BP_FIT_API <= 3
    return 0;
}

std::vector<MFBPModel> mf_load_bp_model_set(
    ssize_t h,
    const py::buffer_info& Wb,
    const py::buffer_info& Tb,
    const py::buffer_info& pb,
    const py::buffer_info& xrb,
    const py::buffer_info& yrb,
    const py::buffer_info& eb,
    int Q,
    double safety_margin)
{
    const ssize_t O = mf_bp_obstacle_count(Wb);
    std::vector<MFBPModel> models;
    models.reserve(static_cast<size_t>(std::max<ssize_t>(O, 0)));
    for (ssize_t o = 0; o < O; ++o) {
        MFBPModel m = mf_load_bp_model(h, o, Wb, Tb, pb, xrb, yrb, eb, Q, safety_margin);
        if (!m.empty) models.push_back(std::move(m));
    }
    return models;
}

MFFieldEval mf_bp_field(const std::array<double,4>&x,const MFBPModel&m,double sx,double sy){
    (void)sx; (void)sy;
    MFFieldEval out;
    if(m.empty){ out.phi=std::numeric_limits<double>::infinity(); return out; }
    const double dx=x[0]-m.pref[0], dy=x[1]-m.pref[1];
    const double qx=m.Tref[0]*dx+m.Tref[1]*dy, qy=m.Tref[2]*dx+m.Tref[3]*dy;

    // Sign comes from the fitted implicit boundary B(q)=l.  Outside the local
    // fitting box is always exterior because the extracted fitted boundary is
    // contained in that box.
    double raw=0.0, rgx=0.0, rgy=0.0;
    const bool in_domain = qx >= m.xr[0] && qx <= m.xr[1] && qy >= m.yr[0] && qy <= m.yr[1];
    const bool raw_ok = mf_bp_raw_q(m, qx, qy, raw, rgx, rgy);
    const double level = raw_ok ? raw - m.safety_margin : 1.0;
    const double sign = (in_domain && level < 0.0) ? -1.0 : 1.0;

    // Recompute a true metric SDF to the fitted l-level boundary.
    double best_sq = std::numeric_limits<double>::infinity();
    double best_x = qx, best_y = qy;
    for (const auto& seg : m.boundary_segments) {
        const double ax=seg[0], ay=seg[1], bx=seg[2], by=seg[3];
        const double ex=bx-ax, ey=by-ay;
        const double ee=ex*ex+ey*ey;
        double t=0.0;
        if(ee>1.0e-18) t=clampd(((qx-ax)*ex+(qy-ay)*ey)/ee,0.0,1.0);
        const double cx=ax+t*ex, cy=ay+t*ey;
        const double ddx=qx-cx, ddy=qy-cy, dsq=ddx*ddx+ddy*ddy;
        if(dsq<best_sq){ best_sq=dsq; best_x=cx; best_y=cy; }
    }

    if(std::isfinite(best_sq)){
        const double dist=std::sqrt(std::max(best_sq,0.0));
        out.phi=sign*dist;
        if(dist>1.0e-12){
            const double gqx=sign*(qx-best_x)/dist;
            const double gqy=sign*(qy-best_y)/dist;
            out.gx=m.Tref[0]*gqx+m.Tref[2]*gqy;
            out.gy=m.Tref[1]*gqx+m.Tref[3]*gqy;
        } else {
            const double gn=std::hypot(rgx,rgy);
            if(gn>1.0e-12){
                const double gqx=rgx/gn, gqy=rgy/gn;
                out.gx=m.Tref[0]*gqx+m.Tref[2]*gqy;
                out.gy=m.Tref[1]*gqx+m.Tref[3]*gqy;
            }
        }
        return out;
    }

    // Small MPC batches avoid contour rasterization: project directly onto the
    // implicit fitted boundary B(q)=l with Newton iterations, then measure the
    // Euclidean distance to the converged boundary point.  Because the fitted
    // field is trained from an SDF, its gradient is a strong projection direction.
    double zx=clampd(qx,m.xr[0],m.xr[1]);
    double zy=clampd(qy,m.yr[0],m.yr[1]);
    bool projected=false;
    for(int it=0;it<12;++it){
        double zr=0.0,zgx=0.0,zgy=0.0;
        if(!mf_bp_raw_q(m,zx,zy,zr,zgx,zgy)) break;
        const double f=zr-m.safety_margin;
        if(std::abs(f)<=1.0e-8){ projected=true; break; }
        const double g2=zgx*zgx+zgy*zgy;
        if(!(g2>1.0e-16)) break;
        double step=f/g2;
        // Cap displacement to keep Newton stable near polynomial artifacts.
        const double step_norm=std::abs(step)*std::sqrt(g2);
        const double max_step=0.35*std::max(m.xr[1]-m.xr[0],m.yr[1]-m.yr[0]);
        if(step_norm>max_step && step_norm>1.0e-16) step*=max_step/step_norm;
        zx=clampd(zx-step*zgx,m.xr[0],m.xr[1]);
        zy=clampd(zy-step*zgy,m.yr[0],m.yr[1]);
        if(it==11){
            double rr=0.0,ggx=0.0,ggy=0.0;
            projected=mf_bp_raw_q(m,zx,zy,rr,ggx,ggy) && std::abs(rr-m.safety_margin)<=1.0e-5;
        }
    }
    if(projected){
        const double ddx=qx-zx, ddy=qy-zy;
        const double dist=std::hypot(ddx,ddy);
        out.phi=sign*dist;
        if(dist>1.0e-12){
            const double gqx=sign*ddx/dist, gqy=sign*ddy/dist;
            out.gx=m.Tref[0]*gqx+m.Tref[2]*gqy;
            out.gy=m.Tref[1]*gqx+m.Tref[3]*gqy;
        } else {
            const double gn=std::hypot(rgx,rgy);
            if(gn>1.0e-12){
                const double gqx=rgx/gn, gqy=rgy/gn;
                out.gx=m.Tref[0]*gqx+m.Tref[2]*gqy;
                out.gy=m.Tref[1]*gqx+m.Tref[3]*gqy;
            }
        }
        return out;
    }

    // Pathological fallback only if the l-level set cannot be projected.
    out.phi=level;
    out.gx=m.Tref[0]*rgx+m.Tref[2]*rgy;
    out.gy=m.Tref[1]*rgx+m.Tref[3]*rgy;
    return out;
}

MFFieldEval mf_bp_field_set(
    const std::array<double,4>& x,
    const std::vector<MFBPModel>& models,
    double sx,
    double sy)
{
    MFFieldEval best;
    best.phi = std::numeric_limits<double>::infinity();
    // BP-SDF of an obstacle union is the pointwise minimum of the individually
    // fitted obstacle fields.  The gradient/normal comes from the active model.
    for (const auto& model : models) {
        const MFFieldEval cur = mf_bp_field(x, model, sx, sy);
        if (cur.phi < best.phi) best = cur;
    }
    return best;
}

// Evaluate already-fitted WORLD-frame per-obstacle Bernstein SDF models at
// arbitrary world query states. No obstacle reconstruction or fitting occurs.
py::array_t<double> evaluate_bp_model_phi_seq(
    const py::array_t<double,py::array::forcecast>& X_bar,
    const py::array_t<double,py::array::forcecast>& axis_scale,
    double h_upper_bound,
    const py::array_t<double,py::array::forcecast>& bp_W,
    const py::array_t<double,py::array::forcecast>& bp_T_ref,
    const py::array_t<double,py::array::forcecast>& bp_p_ref,
    const py::array_t<double,py::array::forcecast>& bp_x_range,
    const py::array_t<double,py::array::forcecast>& bp_y_range,
    const py::array_t<int32_t,py::array::forcecast>& bp_empty,
    int bp_order,
    double bp_safety_margin_l)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const VecViewD scales = view1d_double(axis_scale);
    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : 1.0;
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : 1.0;
    py::array_t<double> out({H});
    auto ob = out.request();
    double* od = static_cast<double*>(ob.ptr);
    std::fill(od, od + static_cast<size_t>(std::max<ssize_t>(H, 0)), h_upper_bound);
    if (H <= 0) return out;

    const auto Wb = bp_W.request(), Tb = bp_T_ref.request(), pb = bp_p_ref.request();
    const auto xrb = bp_x_range.request(), yrb = bp_y_range.request(), eb = bp_empty.request();
    const bool shared = (Wb.ndim == 4 || Wb.ndim == 3) && Wb.shape[0] == 1;
    std::vector<MFBPModel> shared_models;
    if (shared) {
        shared_models = mf_load_bp_model_set(0, Wb, Tb, pb, xrb, yrb, eb, bp_order, bp_safety_margin_l);
        // Dense visualization/field-evaluation batches amortize one contour
        // extraction over thousands of queries. Small MPC batches instead use
        // the implicit Newton projection inside mf_bp_field().
        if (H >= 512)
            for (auto& model : shared_models) mf_build_bp_boundary_segments(model);
    }

    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(shared && H >= 2048)
#endif
    for (ssize_t h = 0; h < H; ++h) {
        std::vector<MFBPModel> stage_models;
        const std::vector<MFBPModel>* bp = nullptr;
        if (shared) {
            bp = &shared_models;
        } else {
            stage_models = mf_load_bp_model_set(h, Wb, Tb, pb, xrb, yrb, eb, bp_order, bp_safety_margin_l);
            bp = &stage_models;
        }
        const std::array<double,4> x{{X.get(h,0), X.get(h,1), X.get(h,2), X.get(h,3)}};
        const MFFieldEval f = mf_bp_field_set(x, *bp, sx, sy);
        od[h] = std::isfinite(f.phi) ? f.phi : h_upper_bound;
    }
    return out;
}

MFBPModel mf_load_bp_model(
    ssize_t h,
    ssize_t o,
    const py::buffer_info& Wb,
    const py::buffer_info& Tb,
    const py::buffer_info& pb,
    const py::buffer_info& xrb,
    const py::buffer_info& yrb,
    const py::buffer_info& eb,
    int Q,
    double safety_margin)
{
    MFBPModel m; m.Q=std::max(Q,2); m.safety_margin=safety_margin;
    auto atD = [](const py::buffer_info& b, std::initializer_list<ssize_t> idx)->double {
        if (b.ndim <= 0 || b.ptr == nullptr) return 0.0;
        const char* base = static_cast<const char*>(b.ptr);
        ssize_t off = 0, d = 0;
        for (ssize_t v : idx) {
            if (d >= b.ndim) break;
            const ssize_t ii = std::min<ssize_t>(std::max<ssize_t>(v, 0), b.shape[d] - 1);
            off += ii * b.strides[d];
            ++d;
        }
        return *reinterpret_cast<const double*>(base + off);
    };
    auto atI = [](const py::buffer_info& b, std::initializer_list<ssize_t> idx)->int32_t {
        if (b.ndim <= 0 || b.ptr == nullptr) return 1;
        const char* base = static_cast<const char*>(b.ptr);
        ssize_t off = 0, d = 0;
        for (ssize_t v : idx) {
            if (d >= b.ndim) break;
            const ssize_t ii = std::min<ssize_t>(std::max<ssize_t>(v, 0), b.shape[d] - 1);
            off += ii * b.strides[d];
            ++d;
        }
        return *reinterpret_cast<const int32_t*>(base + off);
    };

    const bool v4 = Wb.ndim == 4;
    if (v4) {
        m.empty = eb.ndim < 2 || eb.shape[0] <= 0 || eb.shape[1] <= 0 ? true : (atI(eb,{h,o}) != 0);
        m.W.assign(static_cast<size_t>(m.Q*m.Q),0.0);
        if(Wb.shape[2]>=m.Q && Wb.shape[3]>=m.Q)
            for(int iy=0;iy<m.Q;++iy) for(int ix=0;ix<m.Q;++ix)
                m.W[static_cast<size_t>(iy*m.Q+ix)] = atD(Wb,{h,o,iy,ix});
        if(Tb.ndim==4&&Tb.shape[2]>=2&&Tb.shape[3]>=2)
            m.Tref={{atD(Tb,{h,o,0,0}),atD(Tb,{h,o,0,1}),atD(Tb,{h,o,1,0}),atD(Tb,{h,o,1,1})}};
        if(pb.ndim==3&&pb.shape[2]>=2) m.pref={{atD(pb,{h,o,0}),atD(pb,{h,o,1})}};
        if(xrb.ndim==3&&xrb.shape[2]>=2) m.xr={{atD(xrb,{h,o,0}),atD(xrb,{h,o,1})}};
        if(yrb.ndim==3&&yrb.shape[2]>=2) m.yr={{atD(yrb,{h,o,0}),atD(yrb,{h,o,1})}};
    } else {
        // Backward-compatible reader for the previous one-model-per-stage ABI.
        m.empty = eb.ndim<1 || eb.shape[0]<=0 ? true : (atI(eb,{h})!=0);
        m.W.assign(static_cast<size_t>(m.Q*m.Q),0.0);
        if(Wb.ndim==3 && Wb.shape[1]>=m.Q && Wb.shape[2]>=m.Q)
            for(int iy=0;iy<m.Q;++iy) for(int ix=0;ix<m.Q;++ix)
                m.W[static_cast<size_t>(iy*m.Q+ix)] = atD(Wb,{h,iy,ix});
        if(Tb.ndim==3&&Tb.shape[1]>=2&&Tb.shape[2]>=2)
            m.Tref={{atD(Tb,{h,0,0}),atD(Tb,{h,0,1}),atD(Tb,{h,1,0}),atD(Tb,{h,1,1})}};
        if(pb.ndim==2&&pb.shape[1]>=2) m.pref={{atD(pb,{h,0}),atD(pb,{h,1})}};
        if(xrb.ndim==2&&xrb.shape[1]>=2) m.xr={{atD(xrb,{h,0}),atD(xrb,{h,1})}};
        if(yrb.ndim==2&&yrb.shape[1]>=2) m.yr={{atD(yrb,{h,0}),atD(yrb,{h,1})}};
    }
    return m;
}

MFFieldEval mf_eval_nonproposed_field(
    int field_flag,
    const std::array<double,4>&x,
    const std::vector<MFGroup>&world_groups,
    const std::vector<MFBPModel>*bp,
    double sx,double sy,double margin_local,
    double circle_union_lambda,double dcsat_alpha,double dcsat_union_lambda,
    double wu_kappa,double wu_union_lambda,double wu_buffer_offset)
{
    if(field_flag==2) return mf_bp_field_set(x,*bp,sx,sy);
    const auto local=mf_to_local_groups(world_groups,x,sx,sy);
    if(local.empty()) return MFFieldEval{};
    if(field_flag==0) return mf_circle_field_local(local,0.0,0.0,circle_union_lambda);
    if(field_flag==1) return mf_dcsat_field(local,margin_local,dcsat_alpha,dcsat_union_lambda);
    if(field_flag==3) return mf_wu_field(local,margin_local,wu_kappa,wu_union_lambda,wu_buffer_offset);
    throw std::runtime_error("multifield evaluator received invalid non-proposed field_flag");
}

struct MFResidualPair {
    double ga=0.0, gr=0.0, phi=0.0;
    double align=0.0;  // N^T n(phi_c), with N=-grad(Phi)/||grad(Phi)||.
    // Gradient of phi with respect to the local query point at the ego origin.
    // For non-proposed fields this lets the avoidance x/y Jacobian be mapped
    // analytically back to world coordinates instead of re-evaluating the
    // whole obstacle field with central finite differences.
    double phi_gx_local=0.0, phi_gy_local=0.0;
};

MFResidualPair mf_residual_pair(
    int field_flag,
    const std::array<double,4>&x,
    double dt_val,double h_step,
    const std::vector<MFGroup>&world_groups,const std::vector<MFBPModel>*bp,
    double sx,double sy,double ego_L,double ego_W,
    double h_upper_bound,double d_safe,double Ns,double regulation_lambda,double normal_eps,
    double sigma_vx,double sigma_vy,double p_col,bool use_probability_margin,
    const std::array<double,2>&lc_body,
    const std::array<double,2>&lc_local,
    bool impact_enabled,
    double circle_union_lambda,double dcsat_alpha,double dcsat_union_lambda,
    double wu_kappa,double wu_union_lambda,double wu_buffer_offset)
{
    const double disabled=-h_upper_bound;
    double margin_local=0.0;
    if(field_flag!=0 && field_flag!=2 && use_probability_margin){
        p_col=clampd(p_col,0.0,1.0-1.0e-12);
        const double confidence=std::sqrt(std::max(-2.0*std::log(1.0-p_col),0.0));
        const double tpred=std::max(h_step,0.0)*std::max(dt_val,0.0);
        const double sigma_iso=std::hypot(std::max(sigma_vx,0.0),std::max(sigma_vy,0.0))*tpred;
        const double scale=std::max(2.0/std::max(std::abs(ego_L),EPS),2.0/std::max(std::abs(ego_W),EPS));
        margin_local=sigma_iso*confidence*scale;
    }
    const auto f=mf_eval_nonproposed_field(field_flag,x,world_groups,bp,sx,sy,margin_local,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
    if(!std::isfinite(f.phi)) return {d_safe-h_upper_bound,disabled,h_upper_bound,0.0,0.0,0.0};
    const double ga=d_safe-f.phi;
    const double gn=std::hypot(f.gx,f.gy);
    const double epsn=std::max(normal_eps,1.0e-12);
    if(!std::isfinite(gn)||gn<=epsn) return {ga,disabled,f.phi,0.0,f.gx,f.gy};
    const double nfx=-f.gx/gn,nfy=-f.gy/gn;
    double align = 0.0;
    if(field_flag==2){
        // BP-SDF is fitted/evaluated in the world frame. Rotate the body-fixed
        // vulnerable-sector center into world coordinates before alignment.
        const double c=std::cos(x[3]), s=std::sin(x[3]);
        const double lcwx=c*lc_body[0]-s*lc_body[1];
        const double lcwy=s*lc_body[0]+c*lc_body[1];
        align=nfx*lcwx+nfy*lcwy;
    } else {
        align=nfx*lc_local[0]+nfy*lc_local[1];
    }
    if(!impact_enabled) return {ga,disabled,f.phi,align,f.gx,f.gy};
    const double lam_r=std::max(std::abs(regulation_lambda),1.0e-9);
    const double za=(f.phi-d_safe)/lam_r;
    const double zb=(Ns-align)/lam_r;
    const double zmax=std::max(za,zb);
    const double lse=zmax+std::log(std::exp(za-zmax)+std::exp(zb-zmax));
    // Solver residual for
    //   lambda_r*LSE((Phi-d_safe)/lambda_r,
    //                (Ns-N^T n_c)/lambda_r) >= lambda_r*log(2).
    const double gr=lam_r*(std::log(2.0)-lse);
    if(!std::isfinite(gr)) return {ga,disabled,f.phi,align,f.gx,f.gy};
    return {ga,gr,f.phi,align,f.gx,f.gy};
}


inline bool dense_c_buffer_any(const py::buffer_info& b, ssize_t itemsize) {
    if (b.ndim <= 0 || b.ptr == nullptr) return false;
    ssize_t expected = itemsize;
    for (ssize_t d = b.ndim - 1; d >= 0; --d) {
        if (b.shape[d] <= 0) return false;
        if (b.strides[d] != expected) return false;
        expected *= b.shape[d];
    }
    return true;
}

py::array_t<double> build_bp_collision_dual_lin_cuda(
    const py::array_t<double,py::array::forcecast>& X_bar,
    const py::array_t<double,py::array::forcecast>& sector_rays_body,
    double h_upper_bound,double d_safe,double Ns,double regulation_lambda,double normal_eps,
    const py::array_t<double,py::array::forcecast>& bp_W,
    const py::array_t<double,py::array::forcecast>& bp_T_ref,
    const py::array_t<double,py::array::forcecast>& bp_p_ref,
    const py::array_t<double,py::array::forcecast>& bp_x_range,
    const py::array_t<double,py::array::forcecast>& bp_y_range,
    const py::array_t<int32_t,py::array::forcecast>& bp_empty,
    int bp_order,double bp_safety_margin_l,double fd_position_eps_m,double fd_yaw_eps_rad,
    bool impact_normal_enabled)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const int H = static_cast<int>(X.H);
    py::array_t<double> out({X.H, static_cast<ssize_t>(21)});
    auto oi = out.request();
    double* od = static_cast<double*>(oi.ptr);
    std::fill(od, od + static_cast<size_t>(std::max(H, 0)) * 21U, 0.0);
    if (H <= 0) return out;

    const auto Wb = bp_W.request(), Tb = bp_T_ref.request(), pb = bp_p_ref.request();
    const auto xrb = bp_x_range.request(), yrb = bp_y_range.request(), eb = bp_empty.request();
    const int Q = std::max(bp_order, 2);
    if (Q > 16) throw std::runtime_error("native CUDA BP-SDF supports bp_order <= 16");
    if (Wb.ndim != 4 || Wb.shape[0] <= 0 || Wb.shape[1] <= 0 || Wb.shape[2] < Q || Wb.shape[3] < Q)
        throw std::runtime_error("native CUDA BP-SDF requires W[Hm,O,Q,Q]");
    const int Hm = static_cast<int>(Wb.shape[0]);
    const int O = static_cast<int>(Wb.shape[1]);
    if (Hm != 1 && Hm != H) throw std::runtime_error("native CUDA BP-SDF model horizon mismatch");
    if (Tb.ndim != 4 || Tb.shape[0] != Hm || Tb.shape[1] != O || Tb.shape[2] < 2 || Tb.shape[3] < 2 ||
        pb.ndim != 3 || pb.shape[0] != Hm || pb.shape[1] != O || pb.shape[2] < 2 ||
        xrb.ndim != 3 || xrb.shape[0] != Hm || xrb.shape[1] != O || xrb.shape[2] < 2 ||
        yrb.ndim != 3 || yrb.shape[0] != Hm || yrb.shape[1] != O || yrb.shape[2] < 2 ||
        eb.ndim != 2 || eb.shape[0] != Hm || eb.shape[1] != O)
        throw std::runtime_error("native CUDA BP-SDF model-array shape mismatch");
    if (!dense_c_buffer_any(Wb, static_cast<ssize_t>(sizeof(double))) ||
        !dense_c_buffer_any(Tb, static_cast<ssize_t>(sizeof(double))) ||
        !dense_c_buffer_any(pb, static_cast<ssize_t>(sizeof(double))) ||
        !dense_c_buffer_any(xrb, static_cast<ssize_t>(sizeof(double))) ||
        !dense_c_buffer_any(yrb, static_cast<ssize_t>(sizeof(double))) ||
        !dense_c_buffer_any(eb, static_cast<ssize_t>(sizeof(int32_t))))
        throw std::runtime_error("native CUDA BP-SDF requires C-contiguous model arrays");

    std::vector<double> x4(static_cast<size_t>(H) * 4U, 0.0);
    for (int h = 0; h < H; ++h)
        for (int j = 0; j < 4; ++j) x4[static_cast<size_t>(h) * 4U + static_cast<size_t>(j)] = X.get(h, j);
    const MatView2D sector = view2d(sector_rays_body);
    double lcbx = 1.0, lcby = 0.0;
    if (sector.r >= 3 && sector.c >= 2) { lcbx = sector(2,0); lcby = sector(2,1); }
    normalize2(lcbx, lcby);

    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = collision_bp_cuda_run(
            x4.data(), static_cast<const double*>(Wb.ptr), static_cast<const double*>(Tb.ptr),
            static_cast<const double*>(pb.ptr), static_cast<const double*>(xrb.ptr),
            static_cast<const double*>(yrb.ptr), static_cast<const int32_t*>(eb.ptr),
            H, Hm, O, Q, bp_safety_margin_l, h_upper_bound, d_safe, Ns, regulation_lambda,
            normal_eps, fd_position_eps_m, fd_yaw_eps_rad, lcbx, lcby, impact_normal_enabled,
            BP_CUDA_THREADS.load(), od);
    }
    if (!ok) throw std::runtime_error(std::string("native CUDA BP-SDF linearization failed: ") + collision_proposed_cuda_last_error());
    return out;
}

py::array_t<double> build_collision_multifield_dual_lin_hk(
    const py::array_t<double,py::array::forcecast>&X_bar,
    const py::array_t<double,py::array::forcecast>&vertex_set,
    const py::array_t<double,py::array::forcecast>&vertex_normals,
    const py::array_t<int32_t,py::array::forcecast>&vertex_types,
    const py::array_t<int32_t,py::array::forcecast>&feature_obs_indices,
    const py::array_t<double,py::array::forcecast>&h_steps,
    const py::array_t<double,py::array::forcecast>&dt_vals,
    const py::array_t<double,py::array::forcecast>&sector_rays_body,
    const py::array_t<double,py::array::forcecast>&axis_scale,
    bool has_vertex_types,bool has_feature_obs_indices,int vertex_capacity,
    double dt_init,double dt_min,double dt_max,double h_upper_bound,double d_safe,double Ns,double regulation_lambda,
    double ego_L,double ego_W,double halfspace_buffer_m,bool use_scaled,const std::string&normal_orientation,
    double boundary_buffer_m,double smooth_extra_buffer_m,double tau_f,double tau_u,double normal_eps,
    double sigma_vx,double sigma_vy,double p_col,bool use_probability_margin,double pred_rate,
    int field_flag,double circle_union_lambda,double dcsat_alpha,double dcsat_union_lambda,
    double wu_kappa,double wu_union_lambda,double wu_buffer_offset,
    const py::array_t<double,py::array::forcecast>&bp_W,
    const py::array_t<double,py::array::forcecast>&bp_T_ref,
    const py::array_t<double,py::array::forcecast>&bp_p_ref,
    const py::array_t<double,py::array::forcecast>&bp_x_range,
    const py::array_t<double,py::array::forcecast>&bp_y_range,
    const py::array_t<int32_t,py::array::forcecast>&bp_empty,
    int bp_order,double bp_safety_margin_l,double fd_position_eps_m,double fd_yaw_eps_rad,
    bool impact_normal_enabled)
{
    py::array_t<double> bp_cuda_compare_out;
    bool bp_cuda_compare_pending = false;
    bool bp_wants_cuda = false;
    bool bp_profile = false;
    unsigned long long bp_call_id = 0ULL;
    long long bp_work = 0LL, bp_threshold = 0LL;
    double bp_cuda_wall_ms = 0.0;
    CollisionBPCudaTiming bp_gt{};
    std::chrono::steady_clock::time_point bp_cpu_t0;
    bool bp_cpu_timing = false;

    if(field_flag==4){
        const StateAccessor X_cuda_probe = make_state_accessor(X_bar);
        const FeatureAccessor V_cuda_probe = make_feature_accessor(vertex_set, X_cuda_probe.H);
        const FeatureAccessor N_cuda_probe = make_feature_accessor(vertex_normals, X_cuda_probe.H);
        const int K_cuda_probe = std::max(0, std::min({vertex_capacity, static_cast<int>(V_cuda_probe.K), static_cast<int>(N_cuda_probe.K)}));
        const long long hf = static_cast<long long>(X_cuda_probe.H) * static_cast<long long>(K_cuda_probe);
        const long long threshold = PROPOSED_CUDA_MIN_HF.load();
        const bool enabled = PROPOSED_CUDA_ENABLED.load();
        const bool wants_cuda = enabled && hf >= threshold;
        const unsigned long long call_id = PROPOSED_CUDA_CALL_COUNT.fetch_add(1ULL) + 1ULL;
        const int every = std::max(1, PROPOSED_CUDA_PROFILE_EVERY.load());
        const bool profile = PROPOSED_CUDA_PROFILE.load() && ((call_id - 1ULL) % static_cast<unsigned long long>(every) == 0ULL);
        const bool compare_cpu = profile && PROPOSED_CUDA_COMPARE_CPU.load();
        const bool cuda_ready = collision_proposed_cuda_compiled() && collision_proposed_cuda_available();
        // In compare mode, evaluate CUDA even below the policy threshold, but
        // still return the backend selected by the normal policy.
        const bool run_cuda = cuda_ready && enabled && (wants_cuda || compare_cpu);

        if (run_cuda) {
            try {
                const auto t_cuda0 = std::chrono::steady_clock::now();
                py::array_t<double> cuda_out = build_collision_dual_lin_cuda(X_bar,vertex_set,vertex_normals,vertex_types,feature_obs_indices,h_steps,dt_vals,sector_rays_body,axis_scale,
                    has_vertex_types,has_feature_obs_indices,vertex_capacity,dt_init,dt_min,dt_max,h_upper_bound,d_safe,Ns,regulation_lambda,ego_L,ego_W,
                    halfspace_buffer_m,use_scaled,normal_orientation,boundary_buffer_m,smooth_extra_buffer_m,tau_f,tau_u,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,pred_rate);
                const auto t_cuda1 = std::chrono::steady_clock::now();
                const double cuda_wall_ms = elapsed_ms(t_cuda0, t_cuda1);
                CollisionProposedCudaTiming gt{};
                collision_proposed_cuda_get_last_timing(&gt);

                if (compare_cpu) {
                    const auto t_cpu0 = std::chrono::steady_clock::now();
                    py::array_t<double> cpu_out = build_collision_dual_lin_seq(X_bar,vertex_set,vertex_normals,vertex_types,feature_obs_indices,h_steps,dt_vals,sector_rays_body,axis_scale,
                        has_vertex_types,has_feature_obs_indices,vertex_capacity,dt_init,dt_min,dt_max,h_upper_bound,d_safe,Ns,regulation_lambda,ego_L,ego_W,
                        halfspace_buffer_m,use_scaled,normal_orientation,boundary_buffer_m,smooth_extra_buffer_m,tau_f,tau_u,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,pred_rate);
                    const auto t_cpu1 = std::chrono::steady_clock::now();
                    const double cpu_ms = elapsed_ms(t_cpu0, t_cpu1);
                    const double max_diff = max_abs_array_diff(cuda_out, cpu_out);
                    std::printf("[COLLISION-BENCH] call=%llu policy=%s H=%lld F=%d HF=%lld threshold=%lld cpu_ms=%.4f cuda_wall_ms=%.4f cuda_device_ms=%.4f h2d_ms=%.4f kernel_group_ms=%.4f kernel_stage_ms=%.4f d2h_ms=%.4f h2d_kb=%.2f d2h_kb=%.2f feature_copy=%d max_abs_diff=%.3e\n",
                        call_id, wants_cuda ? "CUDA" : "CPU", static_cast<long long>(X_cuda_probe.H), K_cuda_probe, hf, threshold,
                        cpu_ms, cuda_wall_ms, gt.total_device_ms, gt.h2d_ms, gt.kernel_group_ms, gt.kernel_stage_ms, gt.d2h_ms,
                        static_cast<double>(gt.h2d_bytes) / 1024.0, static_cast<double>(gt.d2h_bytes) / 1024.0,
                        (gt.feature_points_copied || gt.feature_normals_copied) ? 1 : 0, max_diff);
                    std::fflush(stdout);
                    return wants_cuda ? cuda_out : cpu_out;
                }

                if (profile) {
                    std::printf("[COLLISION-BENCH] call=%llu policy=CUDA H=%lld F=%d HF=%lld threshold=%lld cuda_wall_ms=%.4f cuda_device_ms=%.4f h2d_ms=%.4f kernel_group_ms=%.4f kernel_stage_ms=%.4f d2h_ms=%.4f h2d_kb=%.2f d2h_kb=%.2f feature_copy=%d\n",
                        call_id, static_cast<long long>(X_cuda_probe.H), K_cuda_probe, hf, threshold, cuda_wall_ms, gt.total_device_ms,
                        gt.h2d_ms, gt.kernel_group_ms, gt.kernel_stage_ms, gt.d2h_ms, static_cast<double>(gt.h2d_bytes) / 1024.0,
                        static_cast<double>(gt.d2h_bytes) / 1024.0, (gt.feature_points_copied || gt.feature_normals_copied) ? 1 : 0);
                    std::fflush(stdout);
                }
                if (wants_cuda) return cuda_out;
                // compare_cpu is the only reason to run CUDA below threshold.
            } catch (const std::exception& exc) {
                if (profile) {
                    std::printf("[COLLISION-BENCH] call=%llu CUDA_FAIL H=%lld F=%d HF=%lld threshold=%lld error=%s\n",
                        call_id, static_cast<long long>(X_cuda_probe.H), K_cuda_probe, hf, threshold, exc.what());
                    std::fflush(stdout);
                }
                if (wants_cuda && !PROPOSED_CUDA_FALLBACK_TO_CPU.load()) throw;
            }
        } else if (wants_cuda && !PROPOSED_CUDA_FALLBACK_TO_CPU.load()) {
            throw std::runtime_error(std::string("native CUDA requested but unavailable: ") + collision_proposed_cuda_last_error());
        }

        const auto t_cpu0 = std::chrono::steady_clock::now();
        py::array_t<double> cpu_out = build_collision_dual_lin_seq(X_bar,vertex_set,vertex_normals,vertex_types,feature_obs_indices,h_steps,dt_vals,sector_rays_body,axis_scale,
            has_vertex_types,has_feature_obs_indices,vertex_capacity,dt_init,dt_min,dt_max,h_upper_bound,d_safe,Ns,regulation_lambda,ego_L,ego_W,
            halfspace_buffer_m,use_scaled,normal_orientation,boundary_buffer_m,smooth_extra_buffer_m,tau_f,tau_u,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,pred_rate);
        const auto t_cpu1 = std::chrono::steady_clock::now();
        if (profile) {
            std::printf("[COLLISION-BENCH] call=%llu policy=CPU H=%lld F=%d HF=%lld threshold=%lld cpu_ms=%.4f cuda_ready=%d\n",
                call_id, static_cast<long long>(X_cuda_probe.H), K_cuda_probe, hf, threshold, elapsed_ms(t_cpu0, t_cpu1), cuda_ready ? 1 : 0);
            std::fflush(stdout);
        }
        return cpu_out;
    }
    if(field_flag==2){
        const StateAccessor X_bp_probe = make_state_accessor(X_bar);
        const auto Wb_probe = bp_W.request();
        const int O_bp_probe = (Wb_probe.ndim == 4 && Wb_probe.shape[1] > 0) ? static_cast<int>(Wb_probe.shape[1]) : 0;
        bp_work = static_cast<long long>(X_bp_probe.H) * static_cast<long long>(O_bp_probe);
        bp_threshold = BP_CUDA_MIN_HO.load();
        const bool enabled = BP_CUDA_ENABLED.load();
        bp_wants_cuda = enabled && bp_work >= bp_threshold;
        bp_call_id = BP_CUDA_CALL_COUNT.fetch_add(1ULL) + 1ULL;
        const int every = std::max(1, BP_CUDA_PROFILE_EVERY.load());
        bp_profile = BP_CUDA_PROFILE.load() && ((bp_call_id - 1ULL) % static_cast<unsigned long long>(every) == 0ULL);
        const bool compare_cpu = bp_profile && BP_CUDA_COMPARE_CPU.load();
        const bool cuda_ready = collision_proposed_cuda_compiled() && collision_proposed_cuda_available();
        const bool run_cuda = cuda_ready && enabled && (bp_wants_cuda || compare_cpu);
        if (run_cuda) {
            try {
                const auto t0 = std::chrono::steady_clock::now();
                py::array_t<double> cuda_out = build_bp_collision_dual_lin_cuda(
                    X_bar, sector_rays_body, h_upper_bound, d_safe, Ns, regulation_lambda, normal_eps,
                    bp_W, bp_T_ref, bp_p_ref, bp_x_range, bp_y_range, bp_empty,
                    bp_order, bp_safety_margin_l, fd_position_eps_m, fd_yaw_eps_rad,
                    impact_normal_enabled);
                const auto t1 = std::chrono::steady_clock::now();
                bp_cuda_wall_ms = elapsed_ms(t0, t1);
                collision_bp_cuda_get_last_timing(&bp_gt);
                if (compare_cpu) {
                    bp_cuda_compare_out = cuda_out;
                    bp_cuda_compare_pending = true;
                } else {
                    if (bp_profile) {
                        std::printf("[BP-CUDA-BENCH] call=%llu policy=CUDA H=%lld O=%d HO=%lld threshold=%lld cuda_wall_ms=%.4f cuda_device_ms=%.4f h2d_ms=%.4f kernel_eval_ms=%.4f kernel_assemble_ms=%.4f d2h_ms=%.4f h2d_kb=%.2f d2h_kb=%.2f model_copy=%d\n",
                            bp_call_id, static_cast<long long>(X_bp_probe.H), O_bp_probe, bp_work, bp_threshold,
                            bp_cuda_wall_ms, bp_gt.total_device_ms, bp_gt.h2d_ms, bp_gt.kernel_eval_ms,
                            bp_gt.kernel_assemble_ms, bp_gt.d2h_ms, static_cast<double>(bp_gt.h2d_bytes)/1024.0,
                            static_cast<double>(bp_gt.d2h_bytes)/1024.0, bp_gt.model_copied ? 1 : 0);
                        std::fflush(stdout);
                    }
                    if (bp_wants_cuda) return cuda_out;
                }
            } catch (const std::exception& exc) {
                if (bp_profile) {
                    std::printf("[BP-CUDA-BENCH] call=%llu CUDA_FAIL H=%lld O=%d HO=%lld threshold=%lld error=%s\n",
                        bp_call_id, static_cast<long long>(X_bp_probe.H), O_bp_probe, bp_work, bp_threshold, exc.what());
                    std::fflush(stdout);
                }
                if (bp_wants_cuda && !BP_CUDA_FALLBACK_TO_CPU.load()) throw;
            }
        } else if (bp_wants_cuda && !BP_CUDA_FALLBACK_TO_CPU.load()) {
            throw std::runtime_error(std::string("native CUDA BP-SDF requested but unavailable: ") + collision_proposed_cuda_last_error());
        }
        if (bp_profile || bp_cuda_compare_pending) { bp_cpu_t0 = std::chrono::steady_clock::now(); bp_cpu_timing = true; }
    }
    if(field_flag<0||field_flag>3) throw std::runtime_error("field_flag must be 0..4");
    const StateAccessor X=make_state_accessor(X_bar); const ssize_t H=X.H;
    const FeatureAccessor V=make_feature_accessor(vertex_set,H); const VecViewI types=view1d_int(vertex_types); const VecViewI obs=view1d_int(feature_obs_indices);
    const VecViewD hs=view1d_double(h_steps); const VecViewD dts=view1d_double(dt_vals); const VecViewD scales=view1d_double(axis_scale); const MatView2D sector=view2d(sector_rays_body);
    const int K=std::max(0,std::min(vertex_capacity,static_cast<int>(V.K)));
    // API-32 output: avoidance block, regulation block, impact-aligned value block.
    py::array_t<double> out({H,static_cast<ssize_t>(21)}); auto oi=out.request(); double*od=static_cast<double*>(oi.ptr); std::fill(od,od+static_cast<size_t>(std::max<ssize_t>(H,0))*21U,0.0);
    if(H<=0) return out;
    const double sx=scales.n>=1?std::max(std::abs(scales(0)),EPS):std::max(std::abs(ego_L)/std::sqrt(2.0),EPS);
    const double sy=scales.n>=2?std::max(std::abs(scales(1)),EPS):std::max(std::abs(ego_W)/std::sqrt(2.0),EPS);
    double lcbx=sector.r>=3&&sector.c>=2?sector(2,0):1.0, lcby=sector.r>=3&&sector.c>=2?sector(2,1):0.0;
    normalize2(lcbx,lcby);
    const std::array<double,2>lc_body{{lcbx,lcby}};
    double lcx=lcbx/sx, lcy=lcby/sy; normalize2(lcx,lcy);
    const std::array<double,2>lc_local{{lcx,lcy}};
    const auto Wb=bp_W.request(),Tb=bp_T_ref.request(),pb=bp_p_ref.request(),xrb=bp_x_range.request(),yrb=bp_y_range.request(),eb=bp_empty.request();
    const bool shared_bp_model = field_flag==2 && (Wb.ndim==4 || Wb.ndim==3) && Wb.shape[0]==1;
    std::vector<MFBPModel> shared_bpms;
    if(shared_bp_model) shared_bpms=mf_load_bp_model_set(0,Wb,Tb,pb,xrb,yrb,eb,bp_order,bp_safety_margin_l);
    const double eps_pos=std::max(fd_position_eps_m,1.0e-7),eps_yaw=std::max(fd_yaw_eps_rad,1.0e-7);
    const double eps_dt=std::max(1.0e-6,std::min(1.0e-4,0.1*std::max(dt_init,1.0e-6))); const double disabled=-h_upper_bound;
    std::atomic<int> nonfinite_found{0};
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(H >= 2)
#endif
    for(ssize_t h=0;h<H;++h){
        std::array<double,4>x{{X.get(h,0),X.get(h,1),X.get(h,2),X.get(h,3)}};
        const double hv=hs.n>0?hs(std::min<ssize_t>(h,hs.n-1)):static_cast<double>(h+1);
        const double dt0=dts.n>0?dts(std::min<ssize_t>(h,dts.n-1)):dt_init;
        const auto groups=(field_flag==2||K<=0)?std::vector<MFGroup>{}:mf_build_world_groups(V,h,K,types,obs,has_vertex_types,has_feature_obs_indices,field_flag!=0);
        std::vector<MFBPModel> bpms; const std::vector<MFBPModel>*bp=nullptr;
        if(field_flag==2){
            if(shared_bp_model) bp=&shared_bpms;
            else { bpms=mf_load_bp_model_set(h,Wb,Tb,pb,xrb,yrb,eb,bp_order,bp_safety_margin_l); bp=&bpms; }
        }
        const auto r0=mf_residual_pair(field_flag,x,dt0,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,impact_normal_enabled,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
        const auto neggrad_cost = [&](const MFResidualPair& r)->double {
            if(field_flag==2){
                const double c=std::cos(x[3]), s=std::sin(x[3]);
                const double lcwx=c*lc_body[0]-s*lc_body[1];
                const double lcwy=s*lc_body[0]+c*lc_body[1];
                return -(r.phi_gx_local*lcwx + r.phi_gy_local*lcwy);
            }
            return -(r.phi_gx_local*lc_local[0] + r.phi_gy_local*lc_local[1]);
        };
        double*row=od+static_cast<size_t>(h)*21U;
        row[0]=r0.ga; row[7]=r0.gr; row[14]=neggrad_cost(r0);

        if(!impact_normal_enabled){
            // Translation derivatives are analytical. BP-SDF returns a world-frame
            // gradient directly; C-SDF/DCSAT/Wu return the body-fixed normalized
            // q-frame gradient and are mapped back with T^T.
            double dphi_dx=0.0, dphi_dy=0.0;
            if(field_flag==2){
                dphi_dx=r0.phi_gx_local;
                dphi_dy=r0.phi_gy_local;
            } else {
                const double c=std::cos(x[3]), s=std::sin(x[3]);
                dphi_dx=(c/sx)*r0.phi_gx_local-(s/sy)*r0.phi_gy_local;
                dphi_dy=(s/sx)*r0.phi_gx_local+(c/sy)*r0.phi_gy_local;
            }
            row[1] = -dphi_dx;
            row[2] = -dphi_dy;

            // BP-SDF is world-frame and its scalar field is yaw-independent.
            // The other fields are rebuilt from the yaw-dependent local geometry.
            if(field_flag!=2){
                auto xp=x,xm=x; xp[3]+=eps_yaw; xm[3]-=eps_yaw;
                const auto rp=mf_residual_pair(field_flag,xp,dt0,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,false,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
                const auto rm=mf_residual_pair(field_flag,xm,dt0,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,false,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
                row[4]=(rp.ga-rm.ga)/(2.0*eps_yaw);
            }
        } else {
            // Regulation depends on the normalized field gradient, so retain
            // central differences for x/y/yaw when that constraint is active.
            const int dims[3]={0,1,3}; const double epss[3]={eps_pos,eps_pos,eps_yaw};
            for(int jj=0;jj<3;++jj){ const int d=dims[jj]; const double ep=epss[jj]; auto xp=x,xm=x; xp[d]+=ep; xm[d]-=ep;
                const auto rp=mf_residual_pair(field_flag,xp,dt0,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,true,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
                const auto rm=mf_residual_pair(field_flag,xm,dt0,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,true,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
                row[1+d]=(rp.ga-rm.ga)/(2.0*ep);
                if(r0.gr>disabled*0.5&&rp.gr>disabled*0.5&&rm.gr>disabled*0.5) row[8+d]=(rp.gr-rm.gr)/(2.0*ep);
            }
        }
        if(field_flag!=0&&field_flag!=2&&use_probability_margin){
            const double dtp=std::max(dt0+eps_dt,1.0e-9),dtm=std::max(dt0-eps_dt,1.0e-9);
            const auto rp=mf_residual_pair(field_flag,x,dtp,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,impact_normal_enabled,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
            const auto rm=mf_residual_pair(field_flag,x,dtm,hv,groups,bp,sx,sy,ego_L,ego_W,h_upper_bound,d_safe,Ns,regulation_lambda,normal_eps,sigma_vx,sigma_vy,p_col,use_probability_margin,lc_body,lc_local,impact_normal_enabled,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
            const double den=std::max(dtp-dtm,1.0e-12); row[5]=(rp.ga-rm.ga)/den;
            if(r0.gr>disabled*0.5&&rp.gr>disabled*0.5&&rm.gr>disabled*0.5) row[12]=(rp.gr-rm.gr)/den;
        }
        for(int j=0;j<21;++j){
            if(!std::isfinite(row[j])) { nonfinite_found.store(1, std::memory_order_relaxed); break; }
        }
    }
    if(nonfinite_found.load(std::memory_order_relaxed))
        throw std::runtime_error("non-finite multifield collision linearization");
    if (field_flag == 2 && bp_cpu_timing) {
        const double cpu_ms = elapsed_ms(bp_cpu_t0, std::chrono::steady_clock::now());
        if (bp_cuda_compare_pending) {
            const double max_diff = max_abs_array_diff(bp_cuda_compare_out, out);
            std::printf("[BP-CUDA-BENCH] call=%llu policy=%s H=%lld HO=%lld threshold=%lld cpu_ms=%.4f cuda_wall_ms=%.4f cuda_device_ms=%.4f h2d_ms=%.4f kernel_eval_ms=%.4f kernel_assemble_ms=%.4f d2h_ms=%.4f model_copy=%d max_abs_diff=%.3e\n",
                bp_call_id, bp_wants_cuda ? "CUDA" : "CPU", static_cast<long long>(H), bp_work, bp_threshold, cpu_ms,
                bp_cuda_wall_ms, bp_gt.total_device_ms, bp_gt.h2d_ms, bp_gt.kernel_eval_ms, bp_gt.kernel_assemble_ms,
                bp_gt.d2h_ms, bp_gt.model_copied ? 1 : 0, max_diff);
            std::fflush(stdout);
            return bp_wants_cuda ? bp_cuda_compare_out : out;
        }
        if (bp_profile) {
            std::printf("[BP-CUDA-BENCH] call=%llu policy=CPU H=%lld HO=%lld threshold=%lld cpu_ms=%.4f\n",
                bp_call_id, static_cast<long long>(H), bp_work, bp_threshold, cpu_ms);
            std::fflush(stdout);
        }
    }
    return out;
}


py::array_t<double> build_collision_multifield_phi_seq(
    const py::array_t<double,py::array::forcecast>&X_bar,
    const py::array_t<double,py::array::forcecast>&vertex_set,
    const py::array_t<double,py::array::forcecast>&vertex_normals,
    const py::array_t<int32_t,py::array::forcecast>&vertex_types,
    const py::array_t<int32_t,py::array::forcecast>&feature_obs_indices,
    const py::array_t<double,py::array::forcecast>&h_steps,
    const py::array_t<double,py::array::forcecast>&dt_vals,
    const py::array_t<double,py::array::forcecast>&axis_scale,
    bool has_vertex_types,bool has_feature_obs_indices,int vertex_capacity,
    double dt_init,double dt_min,double dt_max,double h_upper_bound,double d_safe,double ego_L,double ego_W,
    double halfspace_buffer_m,bool use_scaled,const std::string&normal_orientation,
    double boundary_buffer_m,double smooth_extra_buffer_m,double tau_f,double tau_u,
    double sigma_vx,double sigma_vy,double p_col,bool use_probability_margin,double pred_rate,
    int field_flag,double circle_union_lambda,double dcsat_alpha,double dcsat_union_lambda,
    double wu_kappa,double wu_union_lambda,double wu_buffer_offset,
    const py::array_t<double,py::array::forcecast>&bp_W,
    const py::array_t<double,py::array::forcecast>&bp_T_ref,
    const py::array_t<double,py::array::forcecast>&bp_p_ref,
    const py::array_t<double,py::array::forcecast>&bp_x_range,
    const py::array_t<double,py::array::forcecast>&bp_y_range,
    const py::array_t<int32_t,py::array::forcecast>&bp_empty,
    int bp_order,double bp_safety_margin_l)
{
    if(field_flag==4){
        return build_phi_seq(X_bar,vertex_set,vertex_normals,vertex_types,feature_obs_indices,h_steps,dt_vals,axis_scale,
            has_vertex_types,has_feature_obs_indices,vertex_capacity,dt_init,dt_min,dt_max,h_upper_bound,d_safe,ego_L,ego_W,
            halfspace_buffer_m,use_scaled,normal_orientation,boundary_buffer_m,smooth_extra_buffer_m,tau_f,tau_u,
            sigma_vx,sigma_vy,p_col,use_probability_margin,pred_rate);
    }
    if(field_flag<0||field_flag>3) throw std::runtime_error("field_flag must be 0..4");
    const StateAccessor X=make_state_accessor(X_bar); const ssize_t H=X.H; const FeatureAccessor V=make_feature_accessor(vertex_set,H);
    const VecViewI types=view1d_int(vertex_types),obs=view1d_int(feature_obs_indices); const VecViewD hs=view1d_double(h_steps),dts=view1d_double(dt_vals),scales=view1d_double(axis_scale);
    const int K=std::max(0,std::min(vertex_capacity,static_cast<int>(V.K))); py::array_t<double> out({H}); auto oi=out.request(); double*od=static_cast<double*>(oi.ptr);
    std::fill(od,od+static_cast<size_t>(std::max<ssize_t>(H,0)),h_upper_bound); if(H<=0) return out;
    const double sx=scales.n>=1?std::max(std::abs(scales(0)),EPS):std::max(std::abs(ego_L)/std::sqrt(2.0),EPS);
    const double sy=scales.n>=2?std::max(std::abs(scales(1)),EPS):std::max(std::abs(ego_W)/std::sqrt(2.0),EPS);
    const auto Wb=bp_W.request(),Tb=bp_T_ref.request(),pb=bp_p_ref.request(),xrb=bp_x_range.request(),yrb=bp_y_range.request(),eb=bp_empty.request();
    const bool shared_bp_model = field_flag==2 && (Wb.ndim==4 || Wb.ndim==3) && Wb.shape[0]==1;
    std::vector<MFBPModel> shared_bpms;
    if(shared_bp_model) shared_bpms=mf_load_bp_model_set(0,Wb,Tb,pb,xrb,yrb,eb,bp_order,bp_safety_margin_l);
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(H >= 2)
#endif
    for(ssize_t h=0;h<H;++h){
        std::array<double,4>x{{X.get(h,0),X.get(h,1),X.get(h,2),X.get(h,3)}}; const double hv=hs.n>0?hs(std::min<ssize_t>(h,hs.n-1)):static_cast<double>(h+1); const double dt=dts.n>0?dts(std::min<ssize_t>(h,dts.n-1)):dt_init;
        const auto groups=(field_flag==2||K<=0)?std::vector<MFGroup>{}:mf_build_world_groups(V,h,K,types,obs,has_vertex_types,has_feature_obs_indices,field_flag!=0);
        std::vector<MFBPModel> bpms; const std::vector<MFBPModel>*bp=nullptr;
        if(field_flag==2){
            if(shared_bp_model) bp=&shared_bpms;
            else { bpms=mf_load_bp_model_set(h,Wb,Tb,pb,xrb,yrb,eb,bp_order,bp_safety_margin_l); bp=&bpms; }
        }
        double margin=0.0; if(field_flag!=0&&field_flag!=2&&use_probability_margin){ const double pc=clampd(p_col,0.0,1.0-1.0e-12),conf=std::sqrt(std::max(-2.0*std::log(1.0-pc),0.0)); const double t=std::max(hv,0.0)*std::max(dt,0.0); const double sig=std::hypot(std::max(sigma_vx,0.0),std::max(sigma_vy,0.0))*t; const double sc=std::max(2.0/std::max(std::abs(ego_L),EPS),2.0/std::max(std::abs(ego_W),EPS)); margin=sig*conf*sc; }
        const auto f=mf_eval_nonproposed_field(field_flag,x,groups,bp,sx,sy,margin,circle_union_lambda,dcsat_alpha,dcsat_union_lambda,wu_kappa,wu_union_lambda,wu_buffer_offset);
        od[h]=std::isfinite(f.phi)?f.phi:h_upper_bound;
    }
    return out;
}

py::array_t<double> build_collision_lin_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& /*sector_rays_body*/,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types,
    bool has_feature_obs_indices,
    int vertex_capacity,
    double dt_init,
    double dt_min,
    double dt_max,
    double h_upper_bound,
    double d_safe,
    double ego_L,
    double ego_W,
    double halfspace_buffer_m,
    double /*normal_mag_margin*/,
    double /*sector_relax_beta*/,
    bool use_scaled,
    const std::string& normal_orientation,
    double boundary_buffer_m,
    double smooth_extra_buffer_m,
    double /*initial_safe_margin_m*/,
    double tau_f,
    double tau_u,
    double /*sigma_m*/,
    double /*penetration_sigma_m*/,
    double /*normal_eps*/,
    double sigma_vx,
    double sigma_vy,
    double p_col,
    bool use_probability_margin,
    double pred_rate)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const FeatureAccessor N = make_feature_accessor(vertex_normals, H);
    const VecViewI types = view1d_int(vertex_types);
    const VecViewI obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps);
    const VecViewD dts = view1d_double(dt_vals);
    const VecViewD scales = view1d_double(axis_scale);

    const int K = std::max(0, std::min({vertex_capacity, static_cast<int>(V.K), static_cast<int>(N.K)}));
    py::array_t<double> out({H, static_cast<ssize_t>(7)});
    auto out_info = out.request();
    double* out_data = static_cast<double*>(out_info.ptr);
    const double disabled = -h_upper_bound;
    std::fill(out_data, out_data + static_cast<size_t>(std::max<ssize_t>(H, 0)) * 7U, 0.0);
    for (ssize_t h = 0; h < H; ++h) out_data[static_cast<size_t>(h) * 7U] = disabled;
    if (H <= 0 || K <= 0) return out;

    const bool flip = normal_orientation == "inward";
    std::vector<double> dyn(static_cast<size_t>(K), 1.0);
    std::vector<int32_t> obs_ids(static_cast<size_t>(K), 0);
    for (int k = 0; k < K; ++k) {
        if (has_vertex_types) dyn[static_cast<size_t>(k)] = (k < types.n && types(k) == 1) ? 1.0 : 0.0;
        obs_ids[static_cast<size_t>(k)] = has_feature_obs_indices && k < obs.n ? obs(k) : k;
    }

    // Obstacle membership is invariant over the planning horizon. Build it
    // exactly once, not once per stage.
    const GroupLayout group_layout = make_group_layout(obs_ids, has_feature_obs_indices, K);
    const int O = group_layout.group_count();

    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : std::max(0.5 * std::abs(ego_L), EPS);
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : std::max(0.5 * std::abs(ego_W), EPS);
    const double isx = 1.0 / sx;
    const double isy = 1.0 / sy;
    const double d_safe_world = std::max(std::abs(d_safe), 0.0);
    const double geom_buffer = std::max(halfspace_buffer_m, 0.0) + std::max(boundary_buffer_m, 0.0) +
        std::max(smooth_extra_buffer_m, 0.0) + d_safe_world;
    tau_f = std::max(std::abs(tau_f), EPS);
    tau_u = tau_f;  // common lambda: lambda_o == lambda_f
    p_col = clampd(p_col, 1.0e-12, 1.0 - 1.0e-12);
    const double confidence = std::sqrt(std::max(-2.0 * std::log(1.0 - p_col), 0.0));

    auto process_stage = [&](ssize_t h, CollisionWorkspace& ws) {
        const double px = X.get(h, 0);
        const double py = X.get(h, 1);
        const double yaw = X.get(h, 3);
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        const double hstep = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : static_cast<double>(h);
        const double dt_raw = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : dt_init;
        const double dt = clampd(std::isfinite(dt_raw) ? dt_raw : dt_init, dt_min, dt_max);
        const double t_scale = std::max(hstep, 0.0);
        const double fallback_drdt = t_scale * std::max(pred_rate, 0.0);

        for (int k = 0; k < K; ++k) {
            auto& f = ws.feature[static_cast<size_t>(k)];
            f.dscore.fill(0.0);
            const double vx = V.get(h, k, 0);
            const double vy = V.get(h, k, 1);
            double nwx = N.get(h, k, 0);
            double nwy = N.get(h, k, 1);
            if (flip) { nwx = -nwx; nwy = -nwy; }
            normalize2(nwx, nwy);
            // Isotropic inflation disk used in the paper:
            // r_h = sqrt(-2 log(1-p_c)) * h*dt * sqrt(sigma_vx^2 + sigma_vy^2).
            // The geometry is not shifted upstream; r_h is subtracted directly
            // from the feature score below.
            const double sigma_iso = std::hypot(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));
            const double drdt = use_probability_margin
                ? t_scale * confidence * sigma_iso
                : fallback_drdt;
            const double rbuf = std::max(drdt * dt, 0.0);
            if (use_scaled) {
                const double rx = vx - px;
                const double ry = vy - py;
                const double plx = isx * (c * rx + s * ry);
                const double ply = isy * (-s * rx + c * ry);

                const double rawx = sx * (c * nwx + s * nwy);
                const double rawy = sy * (-s * nwx + c * nwy);
                const double raw_norm = std::max(std::hypot(rawx, rawy), EPS);
                const double nlx = rawx / raw_norm;
                const double nly = rawy / raw_norm;

                const double dplx = isx * (-s * rx + c * ry);
                const double dply = isy * (-c * rx - s * ry);
                const double drawx = sx * (-s * nwx + c * nwy);
                const double drawy = sy * (-c * nwx - s * nwy);
                const double ndraw = nlx * drawx + nly * drawy;
                const double dnlx = (drawx - nlx * ndraw) / raw_norm;
                const double dnly = (drawy - nly * ndraw) / raw_norm;

                const double delta_world = geom_buffer + rbuf * dyn[static_cast<size_t>(k)];
                const double support_scale = 1.0 / raw_norm;
                f.score = -(nlx * plx + nly * ply) - 1.0 - delta_world * support_scale;
                f.dscore[0] = isx * c * nlx - isy * s * nly;
                f.dscore[1] = isx * s * nlx + isy * c * nly;
                f.dscore[3] = -(dnlx * plx + dnly * ply + nlx * dplx + nly * dply)
                    + delta_world * ndraw / (raw_norm * raw_norm);
                f.dscore[4] = -drdt * dyn[static_cast<size_t>(k)] * support_scale;
            } else {
                f.score = px * nwx + py * nwy - (vx * nwx + vy * nwy) - geom_buffer - rbuf * dyn[static_cast<size_t>(k)] - 1.0;
                f.dscore[0] = nwx;
                f.dscore[1] = nwy;
                f.dscore[4] = -drdt * dyn[static_cast<size_t>(k)];
            }
        }

        const auto result = smooth_field_value_grad_precomputed(ws, group_layout, tau_f, tau_u);
        if (!std::isfinite(result.first)) return;
        double* row = out_data + static_cast<size_t>(h) * 7U;
        row[0] = -result.first;
        for (int d = 0; d < D; ++d) {
            const double value = -result.second[d];
            row[d + 1] = std::isfinite(value) ? value : 0.0;
        }
    };

    {
        py::gil_scoped_release release;
        // Keep tiny batches serial. For a normal MPC horizon, create one OpenMP
        // team and one reusable workspace per thread; never allocate/group per stage.
        const long long work = static_cast<long long>(H) * static_cast<long long>(K);
        if (work < COLLISION_OMP_WORK_THRESHOLD) {
            CollisionWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
        } else {
#ifdef _OPENMP
#pragma omp parallel
            {
                CollisionWorkspace ws(K, O);
#pragma omp for schedule(static)
                for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
            }
#else
            CollisionWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
#endif
        }
    }
    return out;
}


struct PhiWorkspace {
    std::vector<double> score;
    std::vector<double> phi_group;
    std::vector<double> exp_weight;

    PhiWorkspace(int K, int O)
        : score(static_cast<size_t>(K), 0.0),
          phi_group(static_cast<size_t>(O), 0.0),
          exp_weight(static_cast<size_t>(std::max(K, O)), 0.0) {}
};

inline double smooth_phi_value_precomputed(
    PhiWorkspace& ws,
    const GroupLayout& groups,
    double tau_f,
    double tau_u)
{
    const int K = static_cast<int>(ws.score.size());
    const int O = groups.group_count();
    if (K <= 0 || O <= 0) return std::numeric_limits<double>::infinity();

    if (groups.singleton_groups) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) mx = std::max(mx, -ws.score[static_cast<size_t>(k)] / tau_u);
        double z = 0.0;
        for (int k = 0; k < K; ++k) z += std::exp(-ws.score[static_cast<size_t>(k)] / tau_u - mx);
        z = std::max(z, 1.0e-300);
        return -tau_u * (mx + std::log(z));
    }

    for (int o = 0; o < O; ++o) {
        const int begin = groups.offsets[static_cast<size_t>(o)];
        const int end = groups.offsets[static_cast<size_t>(o + 1)];
        double mx = -std::numeric_limits<double>::infinity();
        for (int q = begin; q < end; ++q) {
            const int k = groups.indices[static_cast<size_t>(q)];
            mx = std::max(mx, ws.score[static_cast<size_t>(k)] / tau_f);
        }
        double z = 0.0;
        for (int q = begin; q < end; ++q) {
            const int k = groups.indices[static_cast<size_t>(q)];
            z += std::exp(ws.score[static_cast<size_t>(k)] / tau_f - mx);
        }
        z = std::max(z, 1.0e-300);
        ws.phi_group[static_cast<size_t>(o)] = tau_f * (
            mx + std::log(z) - std::log(static_cast<double>(std::max(end - begin, 1)))
        );
    }

    if (O == 1) return ws.phi_group[0];
    double mx = -std::numeric_limits<double>::infinity();
    for (int o = 0; o < O; ++o) mx = std::max(mx, -ws.phi_group[static_cast<size_t>(o)] / tau_u);
    double z = 0.0;
    for (int o = 0; o < O; ++o) z += std::exp(-ws.phi_group[static_cast<size_t>(o)] / tau_u - mx);
    z = std::max(z, 1.0e-300);
    return -tau_u * (mx + std::log(z));
}

py::array_t<double> build_phi_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types,
    bool has_feature_obs_indices,
    int vertex_capacity,
    double dt_init,
    double dt_min,
    double dt_max,
    double h_upper_bound,
    double d_safe,
    double ego_L,
    double ego_W,
    double halfspace_buffer_m,
    bool use_scaled,
    const std::string& normal_orientation,
    double boundary_buffer_m,
    double smooth_extra_buffer_m,
    double tau_f,
    double tau_u,
    double sigma_vx,
    double sigma_vy,
    double p_col,
    bool use_probability_margin,
    double pred_rate)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const FeatureAccessor N = make_feature_accessor(vertex_normals, H);
    const VecViewI types = view1d_int(vertex_types);
    const VecViewI obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps);
    const VecViewD dts = view1d_double(dt_vals);
    const VecViewD scales = view1d_double(axis_scale);

    const int K = std::max(0, std::min({vertex_capacity, static_cast<int>(V.K), static_cast<int>(N.K)}));
    py::array_t<double> out({H});
    auto out_info = out.request();
    double* out_data = static_cast<double*>(out_info.ptr);
    std::fill(out_data, out_data + static_cast<size_t>(std::max<ssize_t>(H, 0)), h_upper_bound);
    if (H <= 0 || K <= 0) return out;

    const bool flip = normal_orientation == "inward";
    std::vector<double> dyn(static_cast<size_t>(K), 1.0);
    std::vector<int32_t> obs_ids(static_cast<size_t>(K), 0);
    for (int k = 0; k < K; ++k) {
        if (has_vertex_types) dyn[static_cast<size_t>(k)] = (k < types.n && types(k) == 1) ? 1.0 : 0.0;
        obs_ids[static_cast<size_t>(k)] = has_feature_obs_indices && k < obs.n ? obs(k) : k;
    }

    const GroupLayout group_layout = make_group_layout(obs_ids, has_feature_obs_indices, K);
    const int O = group_layout.group_count();
    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : std::max(0.5 * std::abs(ego_L), EPS);
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : std::max(0.5 * std::abs(ego_W), EPS);
    const double isx = 1.0 / sx;
    const double isy = 1.0 / sy;
    const double d_safe_world = std::max(std::abs(d_safe), 0.0);
    const double geom_buffer = std::max(halfspace_buffer_m, 0.0) + std::max(boundary_buffer_m, 0.0) +
        std::max(smooth_extra_buffer_m, 0.0) + d_safe_world;
    tau_f = std::max(std::abs(tau_f), EPS);
    tau_u = tau_f;  // common lambda: lambda_o == lambda_f
    p_col = clampd(p_col, 1.0e-12, 1.0 - 1.0e-12);
    const double confidence = std::sqrt(std::max(-2.0 * std::log(1.0 - p_col), 0.0));

    auto process_stage = [&](ssize_t h, PhiWorkspace& ws) {
        const double px = X.get(h, 0);
        const double py = X.get(h, 1);
        const double yaw = X.get(h, 3);
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        const double hstep = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : static_cast<double>(h);
        const double dt_raw = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : dt_init;
        const double dt = clampd(std::isfinite(dt_raw) ? dt_raw : dt_init, dt_min, dt_max);
        const double t_scale = std::max(hstep, 0.0);
        const double fallback_drdt = t_scale * std::max(pred_rate, 0.0);

        for (int k = 0; k < K; ++k) {
            const double vx = V.get(h, k, 0);
            const double vy = V.get(h, k, 1);
            double nwx = N.get(h, k, 0);
            double nwy = N.get(h, k, 1);
            if (flip) { nwx = -nwx; nwy = -nwy; }
            normalize2(nwx, nwy);
            // Isotropic inflation disk used in the paper:
            // r_h = sqrt(-2 log(1-p_c)) * h*dt * sqrt(sigma_vx^2 + sigma_vy^2).
            // The geometry is not shifted upstream; r_h is subtracted directly
            // from the feature score below.
            const double sigma_iso = std::hypot(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));
            const double drdt = use_probability_margin
                ? t_scale * confidence * sigma_iso
                : fallback_drdt;
            const double rbuf = std::max(drdt * dt, 0.0);
            if (use_scaled) {
                const double rx = vx - px;
                const double ry = vy - py;
                const double plx = isx * (c * rx + s * ry);
                const double ply = isy * (-s * rx + c * ry);
                const double rawx = sx * (c * nwx + s * nwy);
                const double rawy = sy * (-s * nwx + c * nwy);
                const double raw_norm = std::max(std::hypot(rawx, rawy), EPS);
                const double nlx = rawx / raw_norm;
                const double nly = rawy / raw_norm;
                const double delta_world = geom_buffer + rbuf * dyn[static_cast<size_t>(k)];
                const double support_scale = 1.0 / raw_norm;
                ws.score[static_cast<size_t>(k)] = -(nlx * plx + nly * ply) - 1.0 - delta_world * support_scale;
            } else {
                ws.score[static_cast<size_t>(k)] = px * nwx + py * nwy
                    - (vx * nwx + vy * nwy) - geom_buffer - rbuf * dyn[static_cast<size_t>(k)] - 1.0;
            }
        }
        const double phi = smooth_phi_value_precomputed(ws, group_layout, tau_f, tau_u);
        if (std::isfinite(phi)) out_data[static_cast<size_t>(h)] = phi;
    };

    {
        py::gil_scoped_release release;
        const long long work = static_cast<long long>(H) * static_cast<long long>(K);
        if (work < COLLISION_OMP_WORK_THRESHOLD) {
            PhiWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
        } else {
#ifdef _OPENMP
#pragma omp parallel
            {
                PhiWorkspace ws(K, O);
#pragma omp for schedule(static)
                for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
            }
#else
            PhiWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
#endif
        }
    }
    return out;
}


struct FullFeatureValue {
    double score = 0.0;
    std::array<double, 2> normal{};
    std::array<double, D> dscore{};
    std::array<double, 2 * D> dnormal{};
};

struct FullFieldResult {
    double phi = 0.0;
    // Common-lambda substitution:
    //   Phi = -lambda * L,
    //   L = log sum_o exp(L_o),
    //   L_o = log(n_o) - log sum_{f in o} exp(score_f/lambda).
    double log_partition = 0.0;

    // Direct numerator for the impact normal, before the outer softmax
    // normalization.  Let q_o = exp(L_o - L_max), Z = sum_o q_o and
    // g_o = grad_q(phi_o). Then
    //   grad_q L = -(1/lambda) * (sum_o q_o g_o) / Z,
    // so the positive scale 1/(lambda Z) cancels in N.  We therefore use
    //   r = -sum_o q_o g_o,
    //   N = r / max(||r||, Z*normal_eps),
    // without ever forming grad(Phi) or normalized obstacle weights for N.
    double normal_weight_sum = 0.0;  // Z
    std::array<double, 2> normal_num{};  // r
    std::array<double, D> dnormal_weight_sum{};  // dZ (common max-shift derivative omitted)
    std::array<double, 2 * D> dnormal_num{};  // dr, same harmless common shift omitted

    // dPhi/dx_aug is still needed by the avoidance row and by dL=-dPhi/lambda
    // in the regulation residual derivative.
    std::array<double, D> dphi{};
};

struct FullCollisionWorkspace {
    std::vector<FullFeatureValue> feature;
    std::vector<double> phi_group;
    std::vector<double> log_outer_group;
    std::vector<std::array<double, 2>> grad_group;
    std::vector<std::array<double, D>> dphi_group;
    std::vector<std::array<double, 2 * D>> dgrad_group;

    FullCollisionWorkspace(int K, int O)
        : feature(static_cast<size_t>(K)),
          phi_group(static_cast<size_t>(O), 0.0),
          log_outer_group(static_cast<size_t>(O), 0.0),
          grad_group(static_cast<size_t>(O)),
          dphi_group(static_cast<size_t>(O)),
          dgrad_group(static_cast<size_t>(O)) {}
};

FullFieldResult smooth_field_full_precomputed(
    FullCollisionWorkspace& ws,
    const GroupLayout& groups,
    double tau_f,
    double tau_u)
{
    // Proposed-field linearization is derived for one common field smoothing
    // parameter. Enforce lambda_o=lambda_f locally before applying the exact
    // log-exp cancellation below.
    const double common_lambda = std::max(std::abs(tau_f), EPS);
    tau_f = common_lambda;
    tau_u = common_lambda;

    const int K = static_cast<int>(ws.feature.size());
    const int O = groups.group_count();
    FullFieldResult out;
    out.normal_num.fill(0.0);
    out.dnormal_weight_sum.fill(0.0);
    out.dnormal_num.fill(0.0);
    out.dphi.fill(0.0);

    if (K <= 0 || O <= 0) {
        out.phi = std::numeric_limits<double>::infinity();
        out.log_partition = -std::numeric_limits<double>::infinity();
        return out;
    }

    // Two-pass common-sum formulation.
    // After a max pass for numerical stability, every exponential contribution
    // needed by Phi, grad(Phi), and d grad(Phi) is accumulated in one pass.
    // This avoids materializing feature softmax weights and avoids re-scanning
    // every feature for dgrad after grad/dphi have already been computed.

    // Fast path: one feature per obstacle. Here L_k=-score_k/tau_u.
    if (groups.singleton_groups) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            mx = std::max(mx, -ws.feature[static_cast<size_t>(k)].score / tau_u);
        }

        double z = 0.0;
        std::array<double, D> dphi_num{};
        dphi_num.fill(0.0);
        for (int k = 0; k < K; ++k) {
            const auto& f = ws.feature[static_cast<size_t>(k)];
            const double q = std::exp(-f.score / tau_u - mx);
            z += q;
            out.normal_num[0] -= q * f.normal[0];
            out.normal_num[1] -= q * f.normal[1];
            for (int d = 0; d < D; ++d) {
                dphi_num[d] += q * f.dscore[d];
                const double dLk = -f.dscore[d] / tau_u;
                out.dnormal_weight_sum[d] += q * dLk;
                out.dnormal_num[0 * D + d] -= q * (
                    f.dnormal[0 * D + d] + f.normal[0] * dLk);
                out.dnormal_num[1 * D + d] -= q * (
                    f.dnormal[1 * D + d] + f.normal[1] * dLk);
            }
        }
        z = std::max(z, 1.0e-300);
        out.normal_weight_sum = z;
        out.log_partition = mx + std::log(z);
        out.phi = -tau_u * out.log_partition;
        for (int d = 0; d < D; ++d) out.dphi[d] = dphi_num[d] / z;
        return out;
    }

    // Per obstacle define, with a stable max shift,
    //   S = sum e_f,
    //   A = sum e_f n_f,
    //   B = sum e_f da_f,
    //   C = sum e_f (dn_f + n_f da_f/tau_f).
    // Then
    //   grad(phi_o)  = A/S,
    //   dphi_o       = B/S,
    //   dgrad(phi_o) = C/S - (A/S) (B/S)/tau_f.
    // Thus only max-pass + one aggregate-pass are required per obstacle.
    for (int o = 0; o < O; ++o) {
        const int begin = groups.offsets[static_cast<size_t>(o)];
        const int end = groups.offsets[static_cast<size_t>(o + 1)];
        double mx = -std::numeric_limits<double>::infinity();
        for (int p = begin; p < end; ++p) {
            const int k = groups.indices[static_cast<size_t>(p)];
            mx = std::max(mx, ws.feature[static_cast<size_t>(k)].score / tau_f);
        }

        double S = 0.0;
        std::array<double, 2> A{};
        std::array<double, D> B{};
        std::array<double, 2 * D> C{};
        A.fill(0.0);
        B.fill(0.0);
        C.fill(0.0);

        for (int p = begin; p < end; ++p) {
            const int k = groups.indices[static_cast<size_t>(p)];
            const auto& f = ws.feature[static_cast<size_t>(k)];
            const double e = std::exp(f.score / tau_f - mx);
            S += e;
            A[0] += e * f.normal[0];
            A[1] += e * f.normal[1];
            for (int d = 0; d < D; ++d) {
                const double ds = f.dscore[d];
                B[d] += e * ds;
                C[0 * D + d] += e * (
                    f.dnormal[0 * D + d] + f.normal[0] * ds / tau_f);
                C[1 * D + d] += e * (
                    f.dnormal[1 * D + d] + f.normal[1] * ds / tau_f);
            }
        }
        S = std::max(S, 1.0e-300);
        const double invS = 1.0 / S;

        auto& go = ws.grad_group[static_cast<size_t>(o)];
        auto& dpo = ws.dphi_group[static_cast<size_t>(o)];
        auto& dgo = ws.dgrad_group[static_cast<size_t>(o)];
        go[0] = A[0] * invS;
        go[1] = A[1] * invS;
        for (int d = 0; d < D; ++d) {
            dpo[d] = B[d] * invS;
            dgo[0 * D + d] = C[0 * D + d] * invS - go[0] * dpo[d] / tau_f;
            dgo[1 * D + d] = C[1 * D + d] * invS - go[1] * dpo[d] / tau_f;
        }

        // Common-lambda cancellation (tau_f=tau_u=lambda):
        //   phi_o/lambda = mx_o + log(S_o) - log(F_o)
        //   exp(-phi_o/lambda) = exp(-mx_o) * F_o/S_o.
        // Store the stable exponential shift and algebraic prefactor directly;
        // do not form phi_o and exponentiate it again in the outer smooth-min.
        const double feature_count = static_cast<double>(std::max(end - begin, 1));
        ws.log_outer_group[static_cast<size_t>(o)] = -mx;       // exponential shift
        ws.phi_group[static_cast<size_t>(o)] = feature_count * invS;  // F_o/S_o
    }

    // Outer obstacle smooth-min after the exact log-exp cancellation above.
    // Only the final log required to recover Phi remains:
    //   C = sum_o exp(-mx_o) F_o/S_o,  Phi = -lambda log C.
    // dPhi is accumulated unnormalized and divided by Z only once at the end.
    double mx = -std::numeric_limits<double>::infinity();
    for (int o = 0; o < O; ++o) {
        mx = std::max(mx, ws.log_outer_group[static_cast<size_t>(o)]);
    }

    double z = 0.0;
    std::array<double, D> dphi_num{};
    dphi_num.fill(0.0);
    for (int o = 0; o < O; ++o) {
        const double prefactor = ws.phi_group[static_cast<size_t>(o)];
        const double q = std::exp(ws.log_outer_group[static_cast<size_t>(o)] - mx) * prefactor;
        z += q;
        const auto& go = ws.grad_group[static_cast<size_t>(o)];
        const auto& dpo = ws.dphi_group[static_cast<size_t>(o)];
        const auto& dgo = ws.dgrad_group[static_cast<size_t>(o)];
        out.normal_num[0] -= q * go[0];
        out.normal_num[1] -= q * go[1];
        for (int d = 0; d < D; ++d) {
            dphi_num[d] += q * dpo[d];
            const double dLo = -dpo[d] / tau_u;
            out.dnormal_weight_sum[d] += q * dLo;
            out.dnormal_num[0 * D + d] -= q * (dgo[0 * D + d] + go[0] * dLo);
            out.dnormal_num[1 * D + d] -= q * (dgo[1 * D + d] + go[1] * dLo);
        }
    }
    z = std::max(z, 1.0e-300);
    out.normal_weight_sum = z;
    out.log_partition = mx + std::log(z);
    out.phi = -tau_u * out.log_partition;
    for (int d = 0; d < D; ++d) out.dphi[d] = dphi_num[d] / z;
    return out;
}

py::array_t<double> build_collision_dual_lin_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& vertex_set,
    const py::array_t<double, py::array::forcecast>& vertex_normals,
    const py::array_t<int32_t, py::array::forcecast>& vertex_types,
    const py::array_t<int32_t, py::array::forcecast>& feature_obs_indices,
    const py::array_t<double, py::array::forcecast>& h_steps,
    const py::array_t<double, py::array::forcecast>& dt_vals,
    const py::array_t<double, py::array::forcecast>& sector_rays_body,
    const py::array_t<double, py::array::forcecast>& axis_scale,
    bool has_vertex_types,
    bool has_feature_obs_indices,
    int vertex_capacity,
    double dt_init,
    double dt_min,
    double dt_max,
    double h_upper_bound,
    double d_safe,
    double Ns,
    double regulation_lambda,
    double ego_L,
    double ego_W,
    double halfspace_buffer_m,
    bool use_scaled,
    const std::string& normal_orientation,
    double boundary_buffer_m,
    double smooth_extra_buffer_m,
    double tau_f,
    double tau_u,
    double normal_eps,
    double sigma_vx,
    double sigma_vy,
    double p_col,
    bool use_probability_margin,
    double pred_rate)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor V = make_feature_accessor(vertex_set, H);
    const FeatureAccessor N = make_feature_accessor(vertex_normals, H);
    const VecViewI types = view1d_int(vertex_types);
    const VecViewI obs = view1d_int(feature_obs_indices);
    const VecViewD hs = view1d_double(h_steps);
    const VecViewD dts = view1d_double(dt_vals);
    const MatView2D rays = view2d(sector_rays_body);
    const VecViewD scales = view1d_double(axis_scale);

    const int K = std::max(0, std::min({vertex_capacity, static_cast<int>(V.K), static_cast<int>(N.K)}));
    // API-32: [avoidance 7 | regulation 7 | impact-aligned value 7].
    py::array_t<double> out({H, static_cast<ssize_t>(21)});
    auto out_info = out.request();
    double* out_data = static_cast<double*>(out_info.ptr);
    const double disabled = -h_upper_bound;
    // IMPORTANT: the acados nonlinear constraint is two-sided,
    //   -h_upper_bound <= h(x,u) <= 0.
    // Therefore using -h_upper_bound as an "inactive" regulation residual
    // actually puts the row exactly on its LOWER bound.  Since h=g_r-s_r,
    // any positive s_r would then violate the lower bound and the huge
    // magnitude can also poison HPIPM scaling.  Use a modest strictly
    // interior constant with zero Jacobian instead.
    const double inactive_regulation = -1.0;
    std::fill(out_data, out_data + static_cast<size_t>(std::max<ssize_t>(H, 0)) * 21U, 0.0);
    for (ssize_t h = 0; h < H; ++h) {
        double* row = out_data + static_cast<size_t>(h) * 21U;
        row[0] = disabled;
        row[7] = inactive_regulation;
    }
    if (H <= 0 || K <= 0) return out;

    const bool flip = normal_orientation == "inward";
    std::vector<double> dyn(static_cast<size_t>(K), 1.0);
    std::vector<int32_t> obs_ids(static_cast<size_t>(K), 0);
    for (int k = 0; k < K; ++k) {
        if (has_vertex_types) dyn[static_cast<size_t>(k)] = (k < types.n && types(k) == 1) ? 1.0 : 0.0;
        obs_ids[static_cast<size_t>(k)] = has_feature_obs_indices && k < obs.n ? obs(k) : k;
    }

    const GroupLayout group_layout = make_group_layout(obs_ids, has_feature_obs_indices, K);
    const int O = group_layout.group_count();

    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : std::max(0.5 * std::abs(ego_L), EPS);
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : std::max(0.5 * std::abs(ego_W), EPS);
    const double isx = 1.0 / sx;
    const double isy = 1.0 / sy;
    const double d_safe_world = std::max(std::abs(d_safe), 0.0);
    const double geom_buffer = std::max(halfspace_buffer_m, 0.0) + std::max(boundary_buffer_m, 0.0) +
        std::max(smooth_extra_buffer_m, 0.0) + d_safe_world;
    // The proposed field uses one common smoothing parameter
    // lambda_f=lambda_o.  The regulation smoothing lambda_r is independent.
    const double common_lambda = std::max(std::abs(tau_f), EPS);
    tau_f = common_lambda;
    tau_u = common_lambda;
    regulation_lambda = std::max(std::abs(regulation_lambda), EPS);
    normal_eps = std::max(std::abs(normal_eps), EPS);
    d_safe = std::max(std::abs(d_safe), EPS);
    p_col = clampd(p_col, 1.0e-12, 1.0 - 1.0e-12);
    const double confidence = std::sqrt(std::max(-2.0 * std::log(1.0 - p_col), 0.0));

    double lcbx = 1.0, lcby = 0.0;
    if (rays.r >= 3 && rays.c >= 2) {
        lcbx = rays(2, 0);
        lcby = rays(2, 1);
    }
    normalize2(lcbx, lcby);

    double lc_local_x = isx * lcbx;
    double lc_local_y = isy * lcby;
    normalize2(lc_local_x, lc_local_y);

    auto process_stage = [&](ssize_t h, FullCollisionWorkspace& ws) {
        const double px = X.get(h, 0);
        const double py = X.get(h, 1);
        const double yaw = X.get(h, 3);
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        const double hstep = hs.n > 0 ? hs(std::min<ssize_t>(h, hs.n - 1)) : static_cast<double>(h);
        const double dt_raw = dts.n > 0 ? dts(std::min<ssize_t>(h, dts.n - 1)) : dt_init;
        const double dt = clampd(std::isfinite(dt_raw) ? dt_raw : dt_init, dt_min, dt_max);
        const double t_scale = std::max(hstep, 0.0);
        const double fallback_drdt = t_scale * std::max(pred_rate, 0.0);

        for (int k = 0; k < K; ++k) {
            auto& f = ws.feature[static_cast<size_t>(k)];
            f.dscore.fill(0.0);
            f.dnormal.fill(0.0);
            const double vx = V.get(h, k, 0);
            const double vy = V.get(h, k, 1);
            double nwx = N.get(h, k, 0);
            double nwy = N.get(h, k, 1);
            if (flip) { nwx = -nwx; nwy = -nwy; }
            normalize2(nwx, nwy);
            // Isotropic inflation disk used in the paper:
            // r_h = sqrt(-2 log(1-p_c)) * h*dt * sqrt(sigma_vx^2 + sigma_vy^2).
            // The geometry is not shifted upstream; r_h is subtracted directly
            // from the feature score below.
            const double sigma_iso = std::hypot(std::max(sigma_vx, 0.0), std::max(sigma_vy, 0.0));
            const double drdt = use_probability_margin
                ? t_scale * confidence * sigma_iso
                : fallback_drdt;
            const double rbuf = std::max(drdt * dt, 0.0);
            if (use_scaled) {
                const double rx = vx - px;
                const double ry = vy - py;
                const double plx = isx * (c * rx + s * ry);
                const double ply = isy * (-s * rx + c * ry);

                const double rawx = sx * (c * nwx + s * nwy);
                const double rawy = sy * (-s * nwx + c * nwy);
                const double raw_norm = std::max(std::hypot(rawx, rawy), EPS);
                const double nlx = rawx / raw_norm;
                const double nly = rawy / raw_norm;

                const double dplx = isx * (-s * rx + c * ry);
                const double dply = isy * (-c * rx - s * ry);
                const double drawx = sx * (-s * nwx + c * nwy);
                const double drawy = sy * (-c * nwx - s * nwy);
                const double ndraw = nlx * drawx + nly * drawy;
                const double dnlx = (drawx - nlx * ndraw) / raw_norm;
                const double dnly = (drawy - nly * ndraw) / raw_norm;

                const double delta_world = geom_buffer + rbuf * dyn[static_cast<size_t>(k)];
                const double support_scale = 1.0 / raw_norm;
                f.score = -(nlx * plx + nly * ply) - 1.0 - delta_world * support_scale;
                f.normal = {nlx, nly};
                f.dscore[0] = isx * c * nlx - isy * s * nly;
                f.dscore[1] = isx * s * nlx + isy * c * nly;
                f.dscore[3] = -(dnlx * plx + dnly * ply + nlx * dplx + nly * dply)
                    + delta_world * ndraw / (raw_norm * raw_norm);
                f.dscore[4] = -drdt * dyn[static_cast<size_t>(k)] * support_scale;
                f.dnormal[0 * D + 3] = dnlx;
                f.dnormal[1 * D + 3] = dnly;
            } else {
                f.score = px * nwx + py * nwy - (vx * nwx + vy * nwy) - geom_buffer - rbuf * dyn[static_cast<size_t>(k)] - 1.0;
                f.normal = {nwx, nwy};
                f.dscore[0] = nwx;
                f.dscore[1] = nwy;
                f.dscore[4] = -drdt * dyn[static_cast<size_t>(k)];
            }
        }

        const FullFieldResult field = smooth_field_full_precomputed(ws, group_layout, tau_f, tau_u);
        if (!std::isfinite(field.phi) || !std::isfinite(field.log_partition) ||
            !std::isfinite(field.normal_num[0]) || !std::isfinite(field.normal_num[1]) ||
            !(field.normal_weight_sum > 0.0)) return;

        double lcx = lc_local_x;
        double lcy = lc_local_y;
        double dlcx_dyaw = 0.0;
        double dlcy_dyaw = 0.0;
        if (!use_scaled) {
            lcx = c * lcbx - s * lcby;
            lcy = s * lcbx + c * lcby;
            dlcx_dyaw = -s * lcbx - c * lcby;
            dlcy_dyaw = c * lcbx - s * lcby;
        }

        double* row = out_data + static_cast<size_t>(h) * 21U;
        // d_safe [m] is already applied as a world-frame support-plane inflation
        // inside every proposed-field feature score.  Hence the safe field uses zero threshold:
        //   g_a = -Phi_safe <= s^a.
        row[0] = -field.phi;
        for (int d = 0; d < D; ++d) {
            const double value = -field.dphi[d];
            row[1 + d] = std::isfinite(value) ? value : 0.0;
        }

        // Fully substituted impact normal.  With
        //   q_o = exp(L_o-Lmax), Z=sum q_o, r=-sum q_o grad(phi_o),
        // the original grad(Phi) is -(1/Z)sum q_o grad(phi_o), while
        // N=-grad(Phi)/max(||grad(Phi)||,eps). Therefore
        //   N = r / max(||r||, Z*eps),
        // and neither Phi nor the normalized outer softmax weights are needed.
        const double rx = field.normal_num[0];
        const double ry = field.normal_num[1];
        const double ZN = field.normal_weight_sum;
        const double raw_r_norm = std::hypot(rx, ry);
        const double clamp_norm = ZN * normal_eps;
        const double r_norm = std::max(raw_r_norm, clamp_norm);
        if (!(r_norm > 0.0) || !std::isfinite(r_norm)) return;
        const double Nx = rx / r_norm;
        const double Ny = ry / r_norm;
        const double align = Nx * lcx + Ny * lcy;

        // J_global ranks candidates with the unnormalized negative field gradient.
        // Since r = Z*(-grad_q Phi), recover -grad_q Phi without normalization.
        const double Gx = rx / ZN;
        const double Gy = ry / ZN;
        const double grad_cost = Gx * lcx + Gy * lcy;
        row[14] = std::isfinite(grad_cost) ? grad_cost : 0.0;

        // The field-level nested log/exp expressions have already been
        // collapsed using lambda_o=lambda_f before reaching this point.
        // The regulation smoothing lambda_r is independent:
        //   F = lambda_r*LSE(Phi_safe/lambda_r,
        //                    (Ns-N^T n_c)/lambda_r) - lambda_r*log(2) >= 0.
        // The native row stores gbar_r=-F. The final OCP residual is g_r=gbar_r-s^r <= 0.
        const double za = field.phi / regulation_lambda;
        const double zb = (Ns - align) / regulation_lambda;
        const double zmax = std::max(za, zb);
        const double ea = std::exp(za - zmax);
        const double eb = std::exp(zb - zmax);
        const double esum = ea + eb;
        if (!(esum > 0.0) || !std::isfinite(esum)) {
            row[7] = inactive_regulation;
            return;
        }
        const double lse = zmax + std::log(esum);
        row[7] = regulation_lambda * (std::log(2.0) - lse);
        const double wa = ea / esum;
        const double wb = eb / esum;

        // Differentiate the complete normalization denominator of N.
        // With denom=max(||r||, Z*eps),
        //   dN = dr/denom - r*d(denom)/denom^2.
        // On the regular branch d(denom)=(r^T dr)/||r||, yielding the familiar
        //   dN=(I-NN^T)dr/||r||.
        // If the numerical floor is active, denom=Z*eps and its dZ term is
        // differentiated as well.  The omitted common max-shift derivative in
        // (dr,dZ) is a common scale term and cancels in either normalized ratio.
        const bool regular_normal = raw_r_norm > clamp_norm;
        const double r_norm2 = r_norm * r_norm;
        for (int d = 0; d < D; ++d) {
            const double drx = field.dnormal_num[0 * D + d];
            const double dry = field.dnormal_num[1 * D + d];
            const double ddenom = regular_normal
                ? (rx * drx + ry * dry) / raw_r_norm
                : normal_eps * field.dnormal_weight_sum[d];
            const double dNx = drx / r_norm - rx * ddenom / r_norm2;
            const double dNy = dry / r_norm - ry * ddenom / r_norm2;
            double dalign = dNx * lcx + dNy * lcy;
            if (!use_scaled && d == 3) {
                dalign += Nx * dlcx_dyaw + Ny * dlcy_dyaw;
            }

            // Regulation still uses normalized N: d(gbar_r) = -w_a*dPhi + w_b*d(N^T n_c).
            const double value = -wa * field.dphi[d] + wb * dalign;
            row[8 + d] = std::isfinite(value) ? value : 0.0;
        }
        if (!std::isfinite(row[0])) row[0] = disabled;
        if (!std::isfinite(row[7])) row[7] = inactive_regulation;
    };

    {
        py::gil_scoped_release release;
        // The API receives the complete H x K feature batch in one call.
        // Internally, keep each stage's K-feature workspace cache-local; this
        // benchmarks faster than materializing a global H*K feature tensor.
        const long long work = static_cast<long long>(H) * static_cast<long long>(K);
        if (work < COLLISION_OMP_WORK_THRESHOLD) {
            FullCollisionWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
        } else {
#ifdef _OPENMP
#pragma omp parallel
            {
                FullCollisionWorkspace ws(K, O);
#pragma omp for schedule(static)
                for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
            }
#else
            FullCollisionWorkspace ws(K, O);
            for (ssize_t h = 0; h < H; ++h) process_stage(h, ws);
#endif
        }
    }
    return out;
}


py::tuple transform_points_normals_seq(
    const py::array_t<double, py::array::forcecast>& X_bar,
    const py::array_t<double, py::array::forcecast>& points_world,
    const py::array_t<double, py::array::forcecast>& normals_world,
    const py::array_t<double, py::array::forcecast>& axis_scale)
{
    const StateAccessor X = make_state_accessor(X_bar);
    const ssize_t H = X.H;
    const FeatureAccessor P = make_feature_accessor(points_world, H);
    const FeatureAccessor N = make_feature_accessor(normals_world, H);
    const VecViewD scales = view1d_double(axis_scale);
    const ssize_t K = std::min(P.K, N.K);
    const double sx = scales.n >= 1 ? std::max(std::abs(scales(0)), EPS) : 1.0;
    const double sy = scales.n >= 2 ? std::max(std::abs(scales(1)), EPS) : 1.0;
    const double isx = 1.0 / sx, isy = 1.0 / sy;

    py::array_t<double> Po({H, K, static_cast<ssize_t>(2)});
    py::array_t<double> No({H, K, static_cast<ssize_t>(2)});
    py::array_t<double> So({H, K});
    auto pi = Po.request(); auto ni = No.request(); auto si = So.request();
    double* pd = static_cast<double*>(pi.ptr);
    double* nd = static_cast<double*>(ni.ptr);
    double* sd = static_cast<double*>(si.ptr);
    {
        py::gil_scoped_release release;
        const long long work = static_cast<long long>(H) * static_cast<long long>(K);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= COLLISION_OMP_WORK_THRESHOLD)
#endif
        for (ssize_t h = 0; h < H; ++h) {
            const double px = X.get(h, 0), py = X.get(h, 1), yaw = X.get(h, 3);
            const double c = std::cos(yaw), s = std::sin(yaw);
            for (ssize_t k = 0; k < K; ++k) {
                const double rx = P.get(h, k, 0) - px, ry = P.get(h, k, 1) - py;
                const double plx = isx * (c * rx + s * ry);
                const double ply = isy * (-s * rx + c * ry);
                const double rawx = sx * (c * N.get(h, k, 0) + s * N.get(h, k, 1));
                const double rawy = sy * (-s * N.get(h, k, 0) + c * N.get(h, k, 1));
                const double raw_norm = std::max(std::hypot(rawx, rawy), EPS);
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(K) + static_cast<size_t>(k));
                pd[2U * idx + 0U] = plx;
                pd[2U * idx + 1U] = ply;
                nd[2U * idx + 0U] = rawx / raw_norm;
                nd[2U * idx + 1U] = rawy / raw_norm;
                sd[idx] = 1.0 / raw_norm;
            }
        }
    }
    return py::make_tuple(Po, No, So);
}


struct DynEval {
    std::array<double, 4> f{};
    std::array<double, 16> A{};
    std::array<double, 8> B{};
};

DynEval dyn_eval(const std::array<double, 4>& x, const std::array<double, 2>& u) {
    DynEval o;
    const double v = x[2], yaw = x[3], c = std::cos(yaw), s = std::sin(yaw);
    o.f = {v * c, v * s, u[1], u[0]};
    o.A.fill(0.0);
    o.A[0 * 4 + 2] = c;
    o.A[0 * 4 + 3] = -v * s;
    o.A[1 * 4 + 2] = s;
    o.A[1 * 4 + 3] = v * c;
    o.B.fill(0.0);
    o.B[2 * 2 + 1] = 1.0;
    o.B[3 * 2 + 0] = 1.0;
    return o;
}

inline std::array<double, 16> eye4() {
    std::array<double, 16> I{};
    I.fill(0.0);
    for (int i = 0; i < 4; ++i) I[i * 4 + i] = 1.0;
    return I;
}

std::array<double, 16> mm44(const std::array<double, 16>& A, const std::array<double, 16>& B) {
    std::array<double, 16> C{};
    C.fill(0.0);
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) {
        const double a = A[i * 4 + k];
        for (int j = 0; j < 4; ++j) C[i * 4 + j] += a * B[k * 4 + j];
    }
    return C;
}

std::array<double, 8> mm42(const std::array<double, 16>& A, const std::array<double, 8>& B) {
    std::array<double, 8> C{};
    C.fill(0.0);
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) {
        const double a = A[i * 4 + k];
        for (int j = 0; j < 2; ++j) C[i * 2 + j] += a * B[k * 2 + j];
    }
    return C;
}

std::array<double, 4> mv4(const std::array<double, 16>& A, const std::array<double, 4>& b) {
    std::array<double, 4> c{};
    c.fill(0.0);
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) c[i] += A[i * 4 + k] * b[k];
    return c;
}

template <size_t N>
std::array<double, N> add_scaled(const std::array<double, N>& a, const std::array<double, N>& b, double scale) {
    std::array<double, N> c{};
    for (size_t i = 0; i < N; ++i) c[i] = a[i] + scale * b[i];
    return c;
}



inline std::array<double, 4> dyn_f_clamped(
    const std::array<double, 4>& x,
    const std::array<double, 2>& u,
    double v_min,
    double v_max)
{
    const double v = clampd(x[2], v_min, v_max);
    const double yaw = x[3];
    return {v * std::cos(yaw), v * std::sin(yaw), u[1], u[0]};
}

py::array_t<double> rollout_ego_seq(
    const py::array_t<double, py::array::forcecast>& x0arr,
    const py::array_t<double, py::array::forcecast>& Uarr,
    const py::array_t<double, py::array::forcecast>& dtarr,
    double yawrate_min,
    double yawrate_max,
    double a_min,
    double a_max,
    double v_min,
    double v_max)
{
    const VecViewD x0 = view1d_double(x0arr);
    const MatView2D U = view2d(Uarr);
    const VecViewD dt = view1d_double(dtarr);
    if (x0.n < 4) throw std::runtime_error("rollout_ego_seq expects x0 with at least 4 values");
    if (U.c < 2) throw std::runtime_error("rollout_ego_seq expects U=Nx2");

    const ssize_t N = U.r;
    py::array_t<double> out({N + 1, static_cast<ssize_t>(4)});
    auto oi = out.mutable_unchecked<2>();

    std::array<double, 4> x = {x0(0), x0(1), x0(2), wrap_to_pi(x0(3))};
    for (int j = 0; j < 4; ++j) oi(0, j) = x[j];

    {
        py::gil_scoped_release release;
        for (ssize_t k = 0; k < N; ++k) {
            std::array<double, 2> u = {
                clampd(U(k, 0), yawrate_min, yawrate_max),
                clampd(U(k, 1), a_min, a_max)
            };
            double h = dt.n > 0 ? dt(std::min<ssize_t>(k, dt.n - 1)) : 0.05;
            if (!std::isfinite(h)) h = 0.05;

            const auto k1 = dyn_f_clamped(x, u, v_min, v_max);
            const auto x2 = add_scaled(x, k1, 0.5 * h);
            const auto k2 = dyn_f_clamped(x2, u, v_min, v_max);
            const auto x3 = add_scaled(x, k2, 0.5 * h);
            const auto k3 = dyn_f_clamped(x3, u, v_min, v_max);
            const auto x4 = add_scaled(x, k3, h);
            const auto k4 = dyn_f_clamped(x4, u, v_min, v_max);

            for (int j = 0; j < 4; ++j) {
                x[j] += (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
            x[2] = clampd(x[2], v_min, v_max);
            x[3] = wrap_to_pi(x[3]);
            for (int j = 0; j < 4; ++j) oi(k + 1, j) = x[j];
        }
    }
    return out;
}

py::tuple rk4_lin_seq(
    const py::array_t<double, py::array::forcecast>& Xarr,
    const py::array_t<double, py::array::forcecast>& Uarr,
    const py::array_t<double, py::array::forcecast>& dtarr,
    double v_min,
    double v_max)
{
    const MatView2D X = view2d(Xarr);
    const MatView2D U = view2d(Uarr);
    const VecViewD dt = view1d_double(dtarr);
    if (X.c < 4 || U.c < 2 || U.r != X.r) throw std::runtime_error("rk4_lin_seq expects X=Hx4, U=Hx2");
    const ssize_t H = X.r;
    py::array_t<double> Xn({H, static_cast<ssize_t>(4)});
    py::array_t<double> Ao({H, static_cast<ssize_t>(4), static_cast<ssize_t>(4)});
    py::array_t<double> Bo({H, static_cast<ssize_t>(4), static_cast<ssize_t>(2)});
    py::array_t<double> Fo({H, static_cast<ssize_t>(4)});
    auto xn = Xn.mutable_unchecked<2>();
    auto ao = Ao.mutable_unchecked<3>();
    auto bo = Bo.mutable_unchecked<3>();
    auto fo = Fo.mutable_unchecked<2>();

    {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(H >= RK4_OMP_HORIZON_THRESHOLD)
#endif
    for (ssize_t h = 0; h < H; ++h) {
        std::array<double, 4> x = {X(h,0), X(h,1), X(h,2), X(h,3)};
        std::array<double, 2> u = {U(h,0), U(h,1)};
        const double d = dt.n > 0 ? dt(std::min<ssize_t>(h, dt.n - 1)) : 0.05;
        const auto I = eye4();
        std::array<double, 4> z4{}; z4.fill(0.0);

        auto e1 = dyn_eval(x, u);
        const auto K1x = e1.A; const auto K1u = e1.B; const auto K1dt = z4;

        auto x2 = add_scaled(x, e1.f, 0.5*d);
        auto S2x = I; for (int i=0;i<16;++i) S2x[i] += 0.5*d*K1x[i];
        auto S2u = K1u; for (double& q:S2u) q *= 0.5*d;
        auto S2dt = e1.f; for (int i=0;i<4;++i) S2dt[i] = 0.5*e1.f[i] + 0.5*d*K1dt[i];
        auto e2 = dyn_eval(x2,u);
        auto K2x = mm44(e2.A,S2x);
        auto K2u = mm42(e2.A,S2u); for(int i=0;i<8;++i) K2u[i]+=e2.B[i];
        auto K2dt = mv4(e2.A,S2dt);

        auto x3 = add_scaled(x,e2.f,0.5*d);
        auto S3x=I; for(int i=0;i<16;++i) S3x[i]+=0.5*d*K2x[i];
        auto S3u=K2u; for(double& q:S3u) q*=0.5*d;
        std::array<double,4> S3dt{}; for(int i=0;i<4;++i) S3dt[i]=0.5*e2.f[i]+0.5*d*K2dt[i];
        auto e3=dyn_eval(x3,u);
        auto K3x=mm44(e3.A,S3x);
        auto K3u=mm42(e3.A,S3u); for(int i=0;i<8;++i) K3u[i]+=e3.B[i];
        auto K3dt=mv4(e3.A,S3dt);

        auto x4=add_scaled(x,e3.f,d);
        auto S4x=I; for(int i=0;i<16;++i) S4x[i]+=d*K3x[i];
        auto S4u=K3u; for(double& q:S4u) q*=d;
        std::array<double,4> S4dt{}; for(int i=0;i<4;++i) S4dt[i]=e3.f[i]+d*K3dt[i];
        auto e4=dyn_eval(x4,u);
        auto K4x=mm44(e4.A,S4x);
        auto K4u=mm42(e4.A,S4u); for(int i=0;i<8;++i) K4u[i]+=e4.B[i];
        auto K4dt=mv4(e4.A,S4dt);

        std::array<double,4> rsum{};
        for(int i=0;i<4;++i) rsum[i]=e1.f[i]+2.0*e2.f[i]+2.0*e3.f[i]+e4.f[i];
        std::array<double,4> xnext{};
        for(int i=0;i<4;++i) xnext[i]=x[i]+d*rsum[i]/6.0;
        xnext[2]=clampd(xnext[2],v_min,v_max); xnext[3]=wrap_to_pi(xnext[3]);

        for(int i=0;i<4;++i) xn(h,i)=xnext[i];
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
            const int z=i*4+j;
            ao(h,i,j)=I[z]+d*(K1x[z]+2.0*K2x[z]+2.0*K3x[z]+K4x[z])/6.0;
        }
        for(int i=0;i<4;++i) for(int j=0;j<2;++j) {
            const int z=i*2+j;
            bo(h,i,j)=d*(K1u[z]+2.0*K2u[z]+2.0*K3u[z]+K4u[z])/6.0;
        }
        for(int i=0;i<4;++i) fo(h,i)=rsum[i]/6.0+d*(K1dt[i]+2.0*K2dt[i]+2.0*K3dt[i]+K4dt[i])/6.0;
    }
    }
    return py::make_tuple(Xn,Ao,Bo,Fo);
}

}  // namespace

PYBIND11_MODULE(collision_lin_cpp, m) {
    m.attr("COLLISION_RESIDUAL_API") = 39;
    m.attr("SURFACE_DISTANCE_API") = 4;
    m.attr("MULTIFIELD_COLLISION_API") = 1;
    m.attr("BP_FIT_API") = 9;
    m.attr("BP_TENSOR_GRAM_ENABLED") = false;
    m.attr("BP_TENSOR_QR_ENABLED") = true;
    m.attr("BP_PARALLEL_SAMPLING_ENABLED") = true;
    m.attr("PHI_SEQUENCE_API") = 1;
    m.attr("HORIZON_ROLLOUT_API") = 1;
    m.attr("HORIZON_BATCH_COLLISION") = true;
    m.attr("HORIZON_FEATURE_BATCH_COLLISION") = true;
    m.attr("DYNAMICS_LINEARIZATION_API") = 1;
    m.attr("STAGEWISE_FEATURE_API") = 1;
    m.attr("UTILS_LINEARIZED_MIN_API") = 22;
    m.attr("COLLISION_OMP_WORK_THRESHOLD") = COLLISION_OMP_WORK_THRESHOLD;
    m.attr("RK4_OMP_HORIZON_THRESHOLD") = RK4_OMP_HORIZON_THRESHOLD;
    m.attr("PROPOSED_CUDA_API") = 2;
    m.attr("BP_CUDA_API") = 1;
    m.attr("NATIVE_CUDA_COMPILED") = collision_proposed_cuda_compiled();
    m.def("cuda_available", []() { return collision_proposed_cuda_available(); });
    m.def("cuda_last_error", []() { return std::string(collision_proposed_cuda_last_error()); });
    m.def("cuda_reset_cache", []() { collision_proposed_cuda_reset_cache(); collision_bp_cuda_reset_cache(); });
    m.def("set_cuda_enabled", [](bool enabled) { PROPOSED_CUDA_ENABLED.store(enabled); }, py::arg("enabled"));
    m.def("set_cuda_min_hf", [](long long v) { PROPOSED_CUDA_MIN_HF.store(std::max(0LL, v)); }, py::arg("min_hf"));
    m.def("set_cuda_threads", [](int v) {
        int t = std::max(32, std::min(v, 256));
        int p = 32; while ((p << 1) <= t) p <<= 1;
        PROPOSED_CUDA_THREADS.store(p);
    }, py::arg("threads"));
    m.def("set_cuda_fallback_to_cpu", [](bool enabled) { PROPOSED_CUDA_FALLBACK_TO_CPU.store(enabled); }, py::arg("enabled"));
    m.def("set_cuda_profile", [](bool enabled, bool compare_cpu, int every) {
        PROPOSED_CUDA_PROFILE.store(enabled);
        PROPOSED_CUDA_COMPARE_CPU.store(compare_cpu);
        PROPOSED_CUDA_PROFILE_EVERY.store(std::max(1, every));
        collision_proposed_cuda_set_profiling(enabled);
    }, py::arg("enabled"), py::arg("compare_cpu") = false, py::arg("every") = 1);
    m.def("set_bp_cuda_enabled", [](bool enabled) { BP_CUDA_ENABLED.store(enabled); }, py::arg("enabled"));
    m.def("set_bp_cuda_min_ho", [](long long v) { BP_CUDA_MIN_HO.store(std::max(0LL, v)); }, py::arg("min_ho"));
    m.def("set_bp_cuda_threads", [](int v) {
        int t = std::max(32, std::min(v, 256));
        int p = 32; while ((p << 1) <= t) p <<= 1;
        BP_CUDA_THREADS.store(p);
    }, py::arg("threads"));
    m.def("set_bp_cuda_fallback_to_cpu", [](bool enabled) { BP_CUDA_FALLBACK_TO_CPU.store(enabled); }, py::arg("enabled"));
    m.def("set_bp_cuda_profile", [](bool enabled, bool compare_cpu, int every) {
        BP_CUDA_PROFILE.store(enabled);
        BP_CUDA_COMPARE_CPU.store(compare_cpu);
        BP_CUDA_PROFILE_EVERY.store(std::max(1, every));
        collision_bp_cuda_set_profiling(enabled);
    }, py::arg("enabled"), py::arg("compare_cpu") = false, py::arg("every") = 1);
    m.def("get_cuda_config", []() {
        py::dict d;
        d["enabled"] = PROPOSED_CUDA_ENABLED.load();
        d["min_hf"] = PROPOSED_CUDA_MIN_HF.load();
        d["threads"] = PROPOSED_CUDA_THREADS.load();
        d["fallback_to_cpu"] = PROPOSED_CUDA_FALLBACK_TO_CPU.load();
        d["profile"] = PROPOSED_CUDA_PROFILE.load();
        d["compare_cpu"] = PROPOSED_CUDA_COMPARE_CPU.load();
        d["profile_every"] = PROPOSED_CUDA_PROFILE_EVERY.load();
        d["compiled"] = collision_proposed_cuda_compiled();
        d["available"] = collision_proposed_cuda_available();
        d["bp_api"] = 1;
        d["bp_enabled"] = BP_CUDA_ENABLED.load();
        d["bp_min_ho"] = BP_CUDA_MIN_HO.load();
        d["bp_threads"] = BP_CUDA_THREADS.load();
        d["bp_fallback_to_cpu"] = BP_CUDA_FALLBACK_TO_CPU.load();
        d["bp_profile"] = BP_CUDA_PROFILE.load();
        d["bp_compare_cpu"] = BP_CUDA_COMPARE_CPU.load();
        d["bp_profile_every"] = BP_CUDA_PROFILE_EVERY.load();
        return d;
    });
#ifdef _OPENMP
    m.attr("OPENMP_ENABLED") = true;
    m.def("set_omp_threads", [](int n) { omp_set_dynamic(0); omp_set_num_threads(std::max(1, n)); }, py::arg("n"));
    m.def("get_omp_max_threads", []() { return omp_get_max_threads(); });
#else
    m.attr("OPENMP_ENABLED") = false;
    m.def("set_omp_threads", [](int) {}, py::arg("n"));
    m.def("get_omp_max_threads", []() { return 1; });
#endif
    m.doc() = "C++ API-33 collision kernels with parallel/tensor-product Bernstein SDF fitting, fast non-proposed-field linearization, and exact continuous surface-distance batch acceleration (surface API 4)";

    m.def("build_surface_distance_lin_hk", &build_surface_distance_lin_hk,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_types"),
          py::arg("feature_obs_indices"), py::arg("h_steps"), py::arg("dt_vals"),
          py::arg("axis_scale"), py::arg("has_vertex_types"),
          py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
          py::arg("h_upper_bound"), py::arg("d_safe"), py::arg("ego_L"), py::arg("ego_W"),
          py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"), py::arg("use_probability_margin"));
    m.def("evaluate_surface_distance_seq", &evaluate_surface_distance_seq_cpp,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_types"),
          py::arg("feature_obs_indices"), py::arg("h_steps"), py::arg("dt_vals"),
          py::arg("axis_scale"), py::arg("has_vertex_types"),
          py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
          py::arg("h_upper_bound"), py::arg("ego_L"), py::arg("ego_W"),
          py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"), py::arg("use_probability_margin"));
    m.def("build_polygon_features_seq", &build_polygon_features_seq,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_types"),
          py::arg("feature_obs_indices"), py::arg("axis_scale"),
          py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"),
          py::arg("vertex_capacity"), py::arg("pseudo_normal_num"),
          py::arg("normal_orientation"), py::arg("expected_H"));

    auto bind_collision_builder = [&](const char* name) {
        m.def(name, &build_collision_lin_seq, py::arg("X_bar"), py::arg("vertex_set"),
              py::arg("vertex_normals"), py::arg("vertex_types"), py::arg("feature_obs_indices"),
              py::arg("h_steps"), py::arg("dt_vals"), py::arg("sector_rays_body"), py::arg("axis_scale"),
              py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
              py::arg("dt_init"), py::arg("dt_min"), py::arg("dt_max"), py::arg("h_upper_bound"),
              py::arg("d_safe"), py::arg("ego_L"), py::arg("ego_W"), py::arg("halfspace_buffer_m"),
              py::arg("normal_mag_margin"), py::arg("sector_relax_beta"), py::arg("use_scaled"),
              py::arg("normal_orientation"), py::arg("boundary_buffer_m"), py::arg("smooth_extra_buffer_m"),
              py::arg("initial_safe_margin_m"), py::arg("tau_f"), py::arg("tau_u"), py::arg("sigma_m"),
              py::arg("penetration_sigma_m"), py::arg("normal_eps"), py::arg("sigma_vx"), py::arg("sigma_vy"),
              py::arg("p_col"), py::arg("use_probability_margin"), py::arg("pred_rate"));
    };
    bind_collision_builder("build_collision_lin_horizon");
    bind_collision_builder("build_collision_lin_seq");  // backward-compatible alias
    auto bind_dual_builder = [&](const char* name) {
        m.def(name, &build_collision_dual_lin_seq,
              py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_normals"),
              py::arg("vertex_types"), py::arg("feature_obs_indices"), py::arg("h_steps"),
              py::arg("dt_vals"), py::arg("sector_rays_body"), py::arg("axis_scale"),
              py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"),
              py::arg("vertex_capacity"), py::arg("dt_init"), py::arg("dt_min"),
              py::arg("dt_max"), py::arg("h_upper_bound"), py::arg("d_safe"),
              py::arg("Ns"), py::arg("regulation_lambda"), py::arg("ego_L"), py::arg("ego_W"),
              py::arg("halfspace_buffer_m"), py::arg("use_scaled"),
              py::arg("normal_orientation"), py::arg("boundary_buffer_m"),
              py::arg("smooth_extra_buffer_m"), py::arg("tau_f"), py::arg("tau_u"),
              py::arg("normal_eps"), py::arg("sigma_vx"), py::arg("sigma_vy"),
              py::arg("p_col"), py::arg("use_probability_margin"), py::arg("pred_rate"));
    };
    m.def("evaluate_bp_model_phi_seq", &evaluate_bp_model_phi_seq,
          py::arg("X_bar"), py::arg("axis_scale"), py::arg("h_upper_bound"),
          py::arg("bp_W"), py::arg("bp_T_ref"), py::arg("bp_p_ref"),
          py::arg("bp_x_range"), py::arg("bp_y_range"), py::arg("bp_empty"),
          py::arg("bp_order"), py::arg("bp_safety_margin_l"));
    m.def("fit_bp_models_seq", &fit_bp_models_seq,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_types"),
          py::arg("feature_obs_indices"), py::arg("h_steps"), py::arg("dt_vals"),
          py::arg("axis_scale"), py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"),
          py::arg("vertex_capacity"), py::arg("dt_init"), py::arg("ego_L"), py::arg("ego_W"),
          py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"),
          py::arg("use_probability_margin"), py::arg("bp_order"), py::arg("bp_training_n_x"),
          py::arg("bp_training_n_y"), py::arg("bp_safety_margin_l"), py::arg("bp_domain_padding"),
          py::arg("horizon_N"), py::arg("v_max"), py::arg("a_min"), py::arg("a_max"),
          py::arg("world_x_min"), py::arg("world_x_max"), py::arg("world_y_min"), py::arg("world_y_max"));
    m.def("build_collision_multifield_dual_lin_hk", &build_collision_multifield_dual_lin_hk,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_normals"),
          py::arg("vertex_types"), py::arg("feature_obs_indices"), py::arg("h_steps"),
          py::arg("dt_vals"), py::arg("sector_rays_body"), py::arg("axis_scale"),
          py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
          py::arg("dt_init"), py::arg("dt_min"), py::arg("dt_max"), py::arg("h_upper_bound"),
          py::arg("d_safe"), py::arg("Ns"), py::arg("regulation_lambda"), py::arg("ego_L"), py::arg("ego_W"),
          py::arg("halfspace_buffer_m"), py::arg("use_scaled"), py::arg("normal_orientation"),
          py::arg("boundary_buffer_m"), py::arg("smooth_extra_buffer_m"), py::arg("tau_f"), py::arg("tau_u"),
          py::arg("normal_eps"), py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"),
          py::arg("use_probability_margin"), py::arg("pred_rate"), py::arg("field_flag"),
          py::arg("circle_union_lambda"), py::arg("dcsat_alpha"), py::arg("dcsat_union_lambda"),
          py::arg("wu_kappa"), py::arg("wu_union_lambda"), py::arg("wu_buffer_offset"),
          py::arg("bp_W"), py::arg("bp_T_ref"), py::arg("bp_p_ref"), py::arg("bp_x_range"),
          py::arg("bp_y_range"), py::arg("bp_empty"), py::arg("bp_order"), py::arg("bp_safety_margin_l"),
          py::arg("fd_position_eps_m"), py::arg("fd_yaw_eps_rad"), py::arg("impact_normal_enabled"));
    // Preferred API: one Python->C++ call receives all H states and K features.
    // Both avoidance and regulation residual/Jacobian blocks are produced together.
    bind_dual_builder("build_collision_dual_lin_hk");
    bind_dual_builder("build_collision_dual_lin_seq");  // backward-compatible alias
    m.def("build_collision_multifield_phi_seq", &build_collision_multifield_phi_seq,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_normals"), py::arg("vertex_types"),
          py::arg("feature_obs_indices"), py::arg("h_steps"), py::arg("dt_vals"), py::arg("axis_scale"),
          py::arg("has_vertex_types"), py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
          py::arg("dt_init"), py::arg("dt_min"), py::arg("dt_max"), py::arg("h_upper_bound"),
          py::arg("d_safe"), py::arg("ego_L"), py::arg("ego_W"), py::arg("halfspace_buffer_m"), py::arg("use_scaled"),
          py::arg("normal_orientation"), py::arg("boundary_buffer_m"), py::arg("smooth_extra_buffer_m"),
          py::arg("tau_f"), py::arg("tau_u"), py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"),
          py::arg("use_probability_margin"), py::arg("pred_rate"), py::arg("field_flag"),
          py::arg("circle_union_lambda"), py::arg("dcsat_alpha"), py::arg("dcsat_union_lambda"),
          py::arg("wu_kappa"), py::arg("wu_union_lambda"), py::arg("wu_buffer_offset"),
          py::arg("bp_W"), py::arg("bp_T_ref"), py::arg("bp_p_ref"), py::arg("bp_x_range"),
          py::arg("bp_y_range"), py::arg("bp_empty"), py::arg("bp_order"), py::arg("bp_safety_margin_l"));
    m.def("build_phi_seq", &build_phi_seq,
          py::arg("X_bar"), py::arg("vertex_set"), py::arg("vertex_normals"),
          py::arg("vertex_types"), py::arg("feature_obs_indices"), py::arg("h_steps"),
          py::arg("dt_vals"), py::arg("axis_scale"), py::arg("has_vertex_types"),
          py::arg("has_feature_obs_indices"), py::arg("vertex_capacity"),
          py::arg("dt_init"), py::arg("dt_min"), py::arg("dt_max"),
          py::arg("h_upper_bound"), py::arg("d_safe"), py::arg("ego_L"), py::arg("ego_W"),
          py::arg("halfspace_buffer_m"), py::arg("use_scaled"),
          py::arg("normal_orientation"), py::arg("boundary_buffer_m"),
          py::arg("smooth_extra_buffer_m"), py::arg("tau_f"), py::arg("tau_u"),
          py::arg("sigma_vx"), py::arg("sigma_vy"), py::arg("p_col"),
          py::arg("use_probability_margin"), py::arg("pred_rate"));
    m.def("transform_points_normals_seq", &transform_points_normals_seq,
          py::arg("X_bar"), py::arg("points_world"), py::arg("normals_world"), py::arg("axis_scale"));
    m.def("rollout_ego_seq", &rollout_ego_seq,
          py::arg("x0"), py::arg("U"), py::arg("dt"),
          py::arg("yawrate_min"), py::arg("yawrate_max"),
          py::arg("a_min"), py::arg("a_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("build_dynamics_lin_seq", &rk4_lin_seq, py::arg("X"), py::arg("U"), py::arg("dt"), py::arg("v_min"), py::arg("v_max"));
    m.def("rk4_lin_seq", &rk4_lin_seq, py::arg("X"), py::arg("U"), py::arg("dt"), py::arg("v_min"), py::arg("v_max"));  // legacy alias
}
