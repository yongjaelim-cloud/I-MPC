#include "collision_lin_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace {

constexpr int GROUP_COMPONENTS = 16;
constexpr int STAGE_COMPONENTS = 20;

std::mutex g_mutex;
std::string g_last_error;
bool g_profile_enabled = false;
CollisionProposedCudaTiming g_last_timing{};
bool g_bp_profile_enabled = false;
CollisionBPCudaTiming g_bp_last_timing{};

inline void set_error(const char* where, cudaError_t err) {
    g_last_error = std::string(where) + ": " + cudaGetErrorString(err);
}

inline int sanitize_threads(int requested) {
    int t = std::max(32, std::min(requested, 256));
    int p = 32;
    while ((p << 1) <= t) p <<= 1;
    return p;
}

template <typename T>
bool ensure_device_capacity(T*& ptr, std::size_t& capacity, std::size_t needed) {
    if (needed <= capacity && ptr != nullptr) return true;
    if (ptr != nullptr) {
        cudaError_t e = cudaFree(ptr);
        if (e != cudaSuccess) {
            set_error("cudaFree", e);
            return false;
        }
        ptr = nullptr;
        capacity = 0;
    }
    if (needed == 0) return true;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&ptr), needed * sizeof(T));
    if (e != cudaSuccess) {
        set_error("cudaMalloc", e);
        return false;
    }
    capacity = needed;
    return true;
}

struct Workspace {
    cudaStream_t stream = nullptr;
    bool stream_ready = false;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_after_h2d = nullptr;
    cudaEvent_t ev_after_group = nullptr;
    cudaEvent_t ev_after_stage = nullptr;
    cudaEvent_t ev_after_d2h = nullptr;
    bool events_ready = false;

    double* x = nullptr; std::size_t x_cap = 0;
    double* p = nullptr; std::size_t p_cap = 0;
    double* n = nullptr; std::size_t n_cap = 0;
    double* dyn = nullptr; std::size_t dyn_cap = 0;
    int32_t* offsets = nullptr; std::size_t offsets_cap = 0;
    int32_t* indices = nullptr; std::size_t indices_cap = 0;
    double* hsteps = nullptr; std::size_t hsteps_cap = 0;
    double* dtvals = nullptr; std::size_t dtvals_cap = 0;

    double* log_outer = nullptr; std::size_t log_outer_cap = 0;
    double* grad_group = nullptr; std::size_t grad_group_cap = 0;
    double* dphi_group = nullptr; std::size_t dphi_group_cap = 0;
    double* dgrad_group = nullptr; std::size_t dgrad_group_cap = 0;
    double* out = nullptr; std::size_t out_cap = 0;

    // Feature geometry is deterministic and cached by utils_linearized during
    // one MPC solve.  Reuse the device copy while the host buffer identity and
    // shape remain unchanged, so outer SQP re-linearizations update only X.
    const void* last_p_host = nullptr;
    const void* last_n_host = nullptr;
    std::size_t last_p_count = 0;
    std::size_t last_n_count = 0;

    ~Workspace() { release(); }

    bool ensure_stream() {
        if (stream_ready) return true;
        cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            set_error("cudaStreamCreateWithFlags", e);
            return false;
        }
        stream_ready = true;
        return true;
    }

    bool ensure_events() {
        if (events_ready) return true;
        cudaEvent_t* events[] = {&ev_start, &ev_after_h2d, &ev_after_group, &ev_after_stage, &ev_after_d2h};
        for (cudaEvent_t* ev : events) {
            cudaError_t e = cudaEventCreateWithFlags(ev, cudaEventDefault);
            if (e != cudaSuccess) {
                set_error("cudaEventCreateWithFlags", e);
                return false;
            }
        }
        events_ready = true;
        return true;
    }

    void release() {
#define FREE_PTR(name) do { if (name) cudaFree(name); name = nullptr; } while (0)
        FREE_PTR(x); FREE_PTR(p); FREE_PTR(n); FREE_PTR(dyn);
        FREE_PTR(offsets); FREE_PTR(indices); FREE_PTR(hsteps); FREE_PTR(dtvals);
        FREE_PTR(log_outer); FREE_PTR(grad_group); FREE_PTR(dphi_group);
        FREE_PTR(dgrad_group); FREE_PTR(out);
#undef FREE_PTR
        x_cap = p_cap = n_cap = dyn_cap = offsets_cap = indices_cap = 0;
        hsteps_cap = dtvals_cap = log_outer_cap = grad_group_cap = 0;
        dphi_group_cap = dgrad_group_cap = out_cap = 0;
        last_p_host = last_n_host = nullptr;
        last_p_count = last_n_count = 0;
        if (events_ready) {
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_after_h2d) cudaEventDestroy(ev_after_h2d);
            if (ev_after_group) cudaEventDestroy(ev_after_group);
            if (ev_after_stage) cudaEventDestroy(ev_after_stage);
            if (ev_after_d2h) cudaEventDestroy(ev_after_d2h);
        }
        ev_start = ev_after_h2d = ev_after_group = ev_after_stage = ev_after_d2h = nullptr;
        events_ready = false;
        if (stream_ready && stream) cudaStreamDestroy(stream);
        stream = nullptr;
        stream_ready = false;
    }

    bool ensure(int H, int K, int O, std::size_t p_count, std::size_t n_count) {
        if (!ensure_stream()) return false;
        return ensure_device_capacity(x, x_cap, static_cast<std::size_t>(H) * 4U)
            && ensure_device_capacity(p, p_cap, p_count)
            && ensure_device_capacity(n, n_cap, n_count)
            && ensure_device_capacity(dyn, dyn_cap, static_cast<std::size_t>(K))
            && ensure_device_capacity(offsets, offsets_cap, static_cast<std::size_t>(O) + 1U)
            && ensure_device_capacity(indices, indices_cap, static_cast<std::size_t>(K))
            && ensure_device_capacity(hsteps, hsteps_cap, static_cast<std::size_t>(H))
            && ensure_device_capacity(dtvals, dtvals_cap, static_cast<std::size_t>(H))
            && ensure_device_capacity(log_outer, log_outer_cap, static_cast<std::size_t>(H) * O)
            && ensure_device_capacity(grad_group, grad_group_cap, static_cast<std::size_t>(H) * O * 2U)
            && ensure_device_capacity(dphi_group, dphi_group_cap, static_cast<std::size_t>(H) * O * 4U)
            && ensure_device_capacity(dgrad_group, dgrad_group_cap, static_cast<std::size_t>(H) * O * 8U)
            && ensure_device_capacity(out, out_cap, static_cast<std::size_t>(H) * 21U);
    }
};

Workspace g_ws;

constexpr int BP_MAX_Q = 16;

// CUDA 11.4 compatibility: do not depend on CUDART_INF from
// <math_constants.h>. Construct IEEE-754 +infinity with a CUDA intrinsic.
__device__ __forceinline__ double bp_pos_inf() {
    return __longlong_as_double(0x7ff0000000000000LL);
}

struct BPWorkspace {
    cudaStream_t stream = nullptr;
    bool stream_ready = false;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_after_h2d = nullptr;
    cudaEvent_t ev_after_eval = nullptr;
    cudaEvent_t ev_after_assemble = nullptr;
    cudaEvent_t ev_after_d2h = nullptr;
    bool events_ready = false;

    double* x = nullptr; std::size_t x_cap = 0;
    double* W = nullptr; std::size_t W_cap = 0;
    double* T = nullptr; std::size_t T_cap = 0;
    double* pref = nullptr; std::size_t pref_cap = 0;
    double* xr = nullptr; std::size_t xr_cap = 0;
    double* yr = nullptr; std::size_t yr_cap = 0;
    int32_t* empty = nullptr; std::size_t empty_cap = 0;
    double* best = nullptr; std::size_t best_cap = 0;
    double* out = nullptr; std::size_t out_cap = 0;

    const void* last_W_host = nullptr; std::size_t last_W_count = 0;
    const void* last_T_host = nullptr; std::size_t last_T_count = 0;
    const void* last_pref_host = nullptr; std::size_t last_pref_count = 0;
    const void* last_xr_host = nullptr; std::size_t last_xr_count = 0;
    const void* last_yr_host = nullptr; std::size_t last_yr_count = 0;
    const void* last_empty_host = nullptr; std::size_t last_empty_count = 0;

    ~BPWorkspace() { release(); }

    bool ensure_stream() {
        if (stream_ready) return true;
        cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) { set_error("cudaStreamCreateWithFlags(BP)", e); return false; }
        stream_ready = true;
        return true;
    }

    bool ensure_events() {
        if (events_ready) return true;
        cudaEvent_t* events[] = {&ev_start, &ev_after_h2d, &ev_after_eval, &ev_after_assemble, &ev_after_d2h};
        for (cudaEvent_t* ev : events) {
            cudaError_t e = cudaEventCreateWithFlags(ev, cudaEventDefault);
            if (e != cudaSuccess) { set_error("cudaEventCreateWithFlags(BP)", e); return false; }
        }
        events_ready = true;
        return true;
    }

    void reset_host_cache() {
        last_W_host = last_T_host = last_pref_host = last_xr_host = last_yr_host = last_empty_host = nullptr;
        last_W_count = last_T_count = last_pref_count = last_xr_count = last_yr_count = last_empty_count = 0;
    }

    void release() {
#define BP_FREE_PTR(name) do { if (name) cudaFree(name); name = nullptr; } while (0)
        BP_FREE_PTR(x); BP_FREE_PTR(W); BP_FREE_PTR(T); BP_FREE_PTR(pref);
        BP_FREE_PTR(xr); BP_FREE_PTR(yr); BP_FREE_PTR(empty); BP_FREE_PTR(best); BP_FREE_PTR(out);
#undef BP_FREE_PTR
        x_cap = W_cap = T_cap = pref_cap = xr_cap = yr_cap = empty_cap = best_cap = out_cap = 0;
        reset_host_cache();
        if (events_ready) {
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_after_h2d) cudaEventDestroy(ev_after_h2d);
            if (ev_after_eval) cudaEventDestroy(ev_after_eval);
            if (ev_after_assemble) cudaEventDestroy(ev_after_assemble);
            if (ev_after_d2h) cudaEventDestroy(ev_after_d2h);
        }
        ev_start = ev_after_h2d = ev_after_eval = ev_after_assemble = ev_after_d2h = nullptr;
        events_ready = false;
        if (stream_ready && stream) cudaStreamDestroy(stream);
        stream = nullptr;
        stream_ready = false;
    }

    bool ensure(int H, int model_stage_count, int O, int Q, int variants) {
        if (!ensure_stream()) return false;
        const std::size_t models = static_cast<std::size_t>(model_stage_count) * static_cast<std::size_t>(O);
        return ensure_device_capacity(x, x_cap, static_cast<std::size_t>(H) * 4U)
            && ensure_device_capacity(W, W_cap, models * static_cast<std::size_t>(Q) * static_cast<std::size_t>(Q))
            && ensure_device_capacity(T, T_cap, models * 4U)
            && ensure_device_capacity(pref, pref_cap, models * 2U)
            && ensure_device_capacity(xr, xr_cap, models * 2U)
            && ensure_device_capacity(yr, yr_cap, models * 2U)
            && ensure_device_capacity(empty, empty_cap, models)
            && ensure_device_capacity(best, best_cap, static_cast<std::size_t>(H) * static_cast<std::size_t>(variants) * 3U)
            && ensure_device_capacity(out, out_cap, static_cast<std::size_t>(H) * 21U);
    }
};

BPWorkspace g_bp_ws;

__device__ __forceinline__ double clampd_gpu(double v, double lo, double hi) {
    return fmin(fmax(v, lo), hi);
}

__device__ __forceinline__ void normalize2_gpu(double &x, double &y) {
    const double n = fmax(sqrt(x*x + y*y), 1.0e-12);
    x /= n; y /= n;
}

template<int NC>
__device__ __forceinline__ void merge_states(double* a, const double* b) {
    const double neg = -1.0e300;
    const double ma = a[0], mb = b[0];
    if (mb <= neg * 0.5) return;
    if (ma <= neg * 0.5) {
#pragma unroll
        for (int c = 0; c < NC; ++c) a[c] = b[c];
        return;
    }
    const double m = fmax(ma, mb);
    const double sa = exp(ma - m);
    const double sb = exp(mb - m);
    a[0] = m;
#pragma unroll
    for (int c = 1; c < NC; ++c) a[c] = sa * a[c] + sb * b[c];
}

template<int NC>
__device__ __forceinline__ void add_lse_state(double* st, double a, const double* v) {
    const double neg = -1.0e300;
    if (st[0] <= neg * 0.5) {
        st[0] = a;
#pragma unroll
        for (int c = 1; c < NC; ++c) st[c] = v[c];
        return;
    }
    if (a > st[0]) {
        const double scale = exp(st[0] - a);
        st[0] = a;
#pragma unroll
        for (int c = 1; c < NC; ++c) st[c] = scale * st[c] + v[c];
    } else {
        const double w = exp(a - st[0]);
#pragma unroll
        for (int c = 1; c < NC; ++c) st[c] += w * v[c];
    }
}

__global__ void proposed_group_reduce(
    const double* __restrict__ X,
    const double* __restrict__ P,
    const double* __restrict__ N,
    const double* __restrict__ dyn,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ hsteps,
    const double* __restrict__ dtvals,
    int H, int K, int O, int point_stage_count, int normal_stage_count,
    double sx, double sy,
    double geom_buffer, double lambda,
    double sigma_iso, double confidence,
    int use_probability_margin, double pred_rate,
    double dt_init, double dt_min, double dt_max,
    int flip_normal, int use_scaled,
    double* __restrict__ log_outer,
    double* __restrict__ grad_group,
    double* __restrict__ dphi_group,
    double* __restrict__ dgrad_group)
{
    const int bo = static_cast<int>(blockIdx.x);
    const int h = bo / O;
    const int o = bo - h * O;
    if (h >= H || o >= O) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int begin = offsets[o];
    const int end = offsets[o + 1];
    const int count = end - begin;
    if (count <= 0) return;

    const double px = X[h*4 + 0];
    const double py = X[h*4 + 1];
    const double yaw = X[h*4 + 3];
    const double cc = cos(yaw), ss = sin(yaw);
    const double isx = 1.0 / sx, isy = 1.0 / sy;
    const double hstep = hsteps[h];
    double dt = dtvals[h];
    if (!isfinite(dt)) dt = dt_init;
    dt = clampd_gpu(dt, dt_min, dt_max);
    const double t_scale = fmax(hstep, 0.0);
    // sigma_iso is the paper RSS velocity uncertainty sqrt(sigma_vx^2 + sigma_vy^2),
    // precomputed by the C++ host wrapper.
    const double drdt = use_probability_margin
        ? t_scale * confidence * sigma_iso
        : t_scale * fmax(pred_rate, 0.0);
    const double rbuf = fmax(drdt * dt, 0.0);
    const double inv_lambda = 1.0 / lambda;

    // [max, S, Ax, Ay, B0, B1, B3, B4,
    //  Cx0,Cx1,Cx3,Cx4,Cy0,Cy1,Cy3,Cy4]
    double local[GROUP_COMPONENTS];
    local[0] = -1.0e300;
#pragma unroll
    for (int q = 1; q < GROUP_COMPONENTS; ++q) local[q] = 0.0;

    for (int q = begin + tid; q < end; q += static_cast<int>(blockDim.x)) {
        const int k = indices[q];
        if (k < 0 || k >= K) continue;
        const int ph = point_stage_count == 1 ? 0 : h;
        const int nh = normal_stage_count == 1 ? 0 : h;
        const long long pbase = ((long long)ph * K + k) * 2LL;
        const long long nbase = ((long long)nh * K + k) * 2LL;
        const double vx = P[pbase + 0];
        const double vy = P[pbase + 1];
        double nwx = N[nbase + 0];
        double nwy = N[nbase + 1];
        if (flip_normal) { nwx = -nwx; nwy = -nwy; }
        normalize2_gpu(nwx, nwy);

        double score, nx, ny;
        double ds0, ds1, ds3, ds4;
        double dnx3 = 0.0, dny3 = 0.0;
        const double dynamic = dyn[k];

        if (use_scaled) {
            const double rxw = vx - px;
            const double ryw = vy - py;
            const double plx = isx * (cc * rxw + ss * ryw);
            const double ply = isy * (-ss * rxw + cc * ryw);

            const double rawx = sx * (cc * nwx + ss * nwy);
            const double rawy = sy * (-ss * nwx + cc * nwy);
            const double raw_norm = fmax(sqrt(rawx*rawx + rawy*rawy), 1.0e-12);
            nx = rawx / raw_norm;
            ny = rawy / raw_norm;

            const double dplx = isx * (-ss * rxw + cc * ryw);
            const double dply = isy * (-cc * rxw - ss * ryw);
            const double drawx = sx * (-ss * nwx + cc * nwy);
            const double drawy = sy * (-cc * nwx - ss * nwy);
            const double ndraw = nx * drawx + ny * drawy;
            dnx3 = (drawx - nx * ndraw) / raw_norm;
            dny3 = (drawy - ny * ndraw) / raw_norm;

            const double delta_world = geom_buffer + rbuf * dynamic;
            const double support_scale = 1.0 / raw_norm;
            score = -(nx * plx + ny * ply) - 1.0 - delta_world * support_scale;
            ds0 = isx * cc * nx - isy * ss * ny;
            ds1 = isx * ss * nx + isy * cc * ny;
            ds3 = -(dnx3 * plx + dny3 * ply + nx * dplx + ny * dply)
                  + delta_world * ndraw / (raw_norm * raw_norm);
            ds4 = -drdt * dynamic * support_scale;
        } else {
            nx = nwx; ny = nwy;
            score = px*nwx + py*nwy - (vx*nwx + vy*nwy)
                    - geom_buffer - rbuf * dynamic - 1.0;
            ds0 = nwx;
            ds1 = nwy;
            ds3 = 0.0;
            ds4 = -drdt * dynamic;
        }

        double v[GROUP_COMPONENTS];
        v[0] = 0.0;
        v[1] = 1.0;
        v[2] = nx; v[3] = ny;
        v[4] = ds0; v[5] = ds1; v[6] = ds3; v[7] = ds4;
        v[8]  = nx * ds0 * inv_lambda;
        v[9]  = nx * ds1 * inv_lambda;
        v[10] = dnx3 + nx * ds3 * inv_lambda;
        v[11] = nx * ds4 * inv_lambda;
        v[12] = ny * ds0 * inv_lambda;
        v[13] = ny * ds1 * inv_lambda;
        v[14] = dny3 + ny * ds3 * inv_lambda;
        v[15] = ny * ds4 * inv_lambda;
        add_lse_state<GROUP_COMPONENTS>(local, score * inv_lambda, v);
    }

    extern __shared__ double shared[];
    double* mine = shared + tid * GROUP_COMPONENTS;
#pragma unroll
    for (int q = 0; q < GROUP_COMPONENTS; ++q) mine[q] = local[q];
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double* other = shared + (tid + stride) * GROUP_COMPONENTS;
            merge_states<GROUP_COMPONENTS>(mine, other);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double S = fmax(mine[1], 1.0e-300);
        const double invS = 1.0 / S;
        const double gox = mine[2] * invS;
        const double goy = mine[3] * invS;
        const long long goff = (long long)h * O + o;
        const double feature_logmean = mine[0] + log(S) - log((double)count);
        log_outer[goff] = -feature_logmean;
        grad_group[goff*2 + 0] = gox;
        grad_group[goff*2 + 1] = goy;

        const int Bidx[4]  = {4,5,6,7};
        const int Cxidx[4] = {8,9,10,11};
        const int Cyidx[4] = {12,13,14,15};
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const double dpo = mine[Bidx[a]] * invS;
            dphi_group[goff*4 + a] = dpo;
            dgrad_group[goff*8 + 0*4 + a] = mine[Cxidx[a]] * invS - gox * dpo * inv_lambda;
            dgrad_group[goff*8 + 1*4 + a] = mine[Cyidx[a]] * invS - goy * dpo * inv_lambda;
        }
    }
}

__global__ void proposed_stage_reduce(
    const double* __restrict__ X,
    const double* __restrict__ log_outer,
    const double* __restrict__ grad_group,
    const double* __restrict__ dphi_group,
    const double* __restrict__ dgrad_group,
    int H, int O,
    double lambda, double regulation_lambda, double d_safe, double Ns, double normal_eps,
    double lcbx, double lcby, double lc_local_x, double lc_local_y,
    int use_scaled, double disabled,
    double* __restrict__ out)
{
    // d_safe is intentionally unused here: it is already applied during
    // proposed_group_reduce as world-frame support-plane inflation.
    (void)d_safe;
    const int h = static_cast<int>(blockIdx.x);
    if (h >= H) return;
    const int tid = static_cast<int>(threadIdx.x);
    const double inv_lambda = 1.0 / lambda;

    // [max, Z, rx, ry, dPhiNum(4), dZ(4), drx(4), dry(4)]
    double local[STAGE_COMPONENTS];
    local[0] = -1.0e300;
#pragma unroll
    for (int q = 1; q < STAGE_COMPONENTS; ++q) local[q] = 0.0;

    for (int o = tid; o < O; o += static_cast<int>(blockDim.x)) {
        const long long goff = (long long)h * O + o;
        const double L = log_outer[goff];
        const double gox = grad_group[goff*2 + 0];
        const double goy = grad_group[goff*2 + 1];
        double v[STAGE_COMPONENTS];
        v[0] = 0.0;
        v[1] = 1.0;
        v[2] = -gox;
        v[3] = -goy;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const double dpo = dphi_group[goff*4 + a];
            const double dLo = -dpo * inv_lambda;
            const double dgx = dgrad_group[goff*8 + 0*4 + a];
            const double dgy = dgrad_group[goff*8 + 1*4 + a];
            v[4 + a] = dpo;
            v[8 + a] = dLo;
            v[12 + a] = -(dgx + gox * dLo);
            v[16 + a] = -(dgy + goy * dLo);
        }
        add_lse_state<STAGE_COMPONENTS>(local, L, v);
    }

    extern __shared__ double shared[];
    double* mine = shared + tid * STAGE_COMPONENTS;
#pragma unroll
    for (int q = 0; q < STAGE_COMPONENTS; ++q) mine[q] = local[q];
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double* other = shared + (tid + stride) * STAGE_COMPONENTS;
            merge_states<STAGE_COMPONENTS>(mine, other);
        }
        __syncthreads();
    }

    if (tid == 0) {
        double* row = out + (long long)h * 21LL;
#pragma unroll
        for (int j = 0; j < 21; ++j) row[j] = 0.0;
        row[0] = disabled;
        // Do not use -h_upper_bound for an inactive regulation row: acados
        // imposes -h_upper_bound <= h <= 0, so that value is exactly on the
        // lower bound and the final h=g_r=gbar_r-s_r can violate it for s_r>0.
        row[7] = -1.0;

        const double Z = fmax(mine[1], 1.0e-300);
        if (!(mine[0] > -5.0e299) || !isfinite(mine[0]) || !(Z > 0.0)) return;
        const double log_partition = mine[0] + log(Z);
        const double phi = -lambda * log_partition;
        if (!isfinite(phi)) return;

        const int dims[4] = {0,1,3,4};
        double dphi[4];
        row[0] = -phi;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            dphi[a] = mine[4 + a] / Z;
            row[1 + dims[a]] = -dphi[a];
        }

        const double rx = mine[2], ry = mine[3];
        const double raw_r_norm = sqrt(rx*rx + ry*ry);
        const double clamp_norm = Z * normal_eps;
        const double r_norm = fmax(raw_r_norm, clamp_norm);
        if (!(r_norm > 0.0) || !isfinite(r_norm)) return;
        const double Nx = rx / r_norm;
        const double Ny = ry / r_norm;

        double lcx, lcy, dlcx = 0.0, dlcy = 0.0;
        if (use_scaled) {
            lcx = lc_local_x; lcy = lc_local_y;
        } else {
            const double yaw = X[h*4 + 3];
            const double c = cos(yaw), s = sin(yaw);
            lcx = c*lcbx - s*lcby;
            lcy = s*lcbx + c*lcby;
            dlcx = -s*lcbx - c*lcby;
            dlcy =  c*lcbx - s*lcby;
        }
        const double align = Nx*lcx + Ny*lcy;

        // Third API-32 block carries only the J_global impact-aligned value
        //   (-grad_q Phi)^T n_c. Derivative columns remain zero.
        // Here r = Z*(-grad_q Phi), so divide by Z but do not normalize.
        const double Gx = rx / Z;
        const double Gy = ry / Z;
        const double grad_cost = Gx*lcx + Gy*lcy;
        row[14] = isfinite(grad_cost) ? grad_cost : 0.0;

        // d_safe [m] was already applied in proposed_group_reduce as a world-frame
        // support-plane inflation. The safe field therefore uses zero threshold.
        // lambda_r is independent of the field smoothing.
        const double za = phi / regulation_lambda;
        const double zb = (Ns - align) / regulation_lambda;
        const double zmax = fmax(za, zb);
        const double ea = exp(za - zmax);
        const double eb = exp(zb - zmax);
        const double esum = ea + eb;
        if (!(esum > 0.0) || !isfinite(esum)) {
            row[7] = -1.0;
            return;
        }
        const double lse = zmax + log(esum);
        row[7] = regulation_lambda * (log(2.0) - lse);
        const double wa = ea / esum;
        const double wb = eb / esum;

        // Differentiate denom=max(||r||, Z*eps) instead of freezing it.
        // The regular branch gives dN=(I-NN^T)dr/||r||; if the numerical
        // floor is active, include d(Z*eps)=eps*dZ as well.
        const int regular_normal = raw_r_norm > clamp_norm;
        const double r_norm2 = r_norm * r_norm;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int d = dims[a];
            const double dZ = mine[8 + a];
            const double drx = mine[12 + a];
            const double dry = mine[16 + a];
            const double ddenom = regular_normal
                ? (rx*drx + ry*dry) / raw_r_norm
                : normal_eps * dZ;
            const double dNx = drx / r_norm - rx * ddenom / r_norm2;
            const double dNy = dry / r_norm - ry * ddenom / r_norm2;
            double dalign = dNx*lcx + dNy*lcy;
            if (!use_scaled && d == 3) dalign += Nx*dlcx + Ny*dlcy;

            // Regulation still uses normalized N.
            const double value = -wa*dphi[a] + wb*dalign;
            row[8 + d] = isfinite(value) ? value : 0.0;
        }
        if (!isfinite(row[0])) row[0] = disabled;
        if (!isfinite(row[7])) row[7] = -1.0;
    }
}


__device__ __forceinline__ void bp_basis_gpu(double x, int Q, double* B, double* dB) {
    x = clampd_gpu(x, 0.0, 1.0);
    const int n = Q - 1;
#pragma unroll
    for (int i = 0; i < BP_MAX_Q; ++i) { B[i] = 0.0; dB[i] = 0.0; }
    B[0] = 1.0;
    for (int j = 1; j <= n; ++j) {
        double saved = 0.0;
        for (int k = 0; k < j; ++k) {
            const double temp = B[k];
            B[k] = saved + (1.0 - x) * temp;
            saved = x * temp;
        }
        B[j] = saved;
    }
    if (n <= 0) return;
    double low[BP_MAX_Q];
#pragma unroll
    for (int i = 0; i < BP_MAX_Q; ++i) low[i] = 0.0;
    low[0] = 1.0;
    for (int j = 1; j <= n - 1; ++j) {
        double saved = 0.0;
        for (int k = 0; k < j; ++k) {
            const double temp = low[k];
            low[k] = saved + (1.0 - x) * temp;
            saved = x * temp;
        }
        low[j] = saved;
    }
    for (int q = 0; q < Q; ++q) {
        const double left = q >= 1 ? low[q - 1] : 0.0;
        const double right = q <= n - 1 ? low[q] : 0.0;
        dB[q] = static_cast<double>(n) * (left - right);
    }
}

__device__ __forceinline__ bool bp_raw_gpu(
    const double* Wm, int Q,
    double xr0, double xr1, double yr0, double yr1,
    double qx, double qy,
    double& raw, double& gx, double& gy)
{
    const double dxr = fmax(xr1 - xr0, 1.0e-12);
    const double dyr = fmax(yr1 - yr0, 1.0e-12);
    const double xi = (qx - xr0) / dxr;
    const double eta = (qy - yr0) / dyr;
    double Bx[BP_MAX_Q], dBx[BP_MAX_Q], By[BP_MAX_Q], dBy[BP_MAX_Q];
    bp_basis_gpu(xi, Q, Bx, dBx);
    bp_basis_gpu(eta, Q, By, dBy);
    raw = 0.0; gx = 0.0; gy = 0.0;
    for (int ix = 0; ix < Q; ++ix) {
        double row_v = 0.0, row_dy = 0.0;
        const double* row = Wm + static_cast<std::size_t>(ix) * static_cast<std::size_t>(Q);
        for (int iy = 0; iy < Q; ++iy) {
            const double w = row[iy];
            row_v += w * By[iy];
            row_dy += w * dBy[iy];
        }
        raw += Bx[ix] * row_v;
        gx += dBx[ix] * row_v;
        gy += Bx[ix] * row_dy;
    }
    gx /= dxr;
    gy /= dyr;
    return isfinite(raw) && isfinite(gx) && isfinite(gy);
}

__device__ __forceinline__ bool bp_metric_model_gpu(
    const double* Wm, const double* Tm, const double* pm,
    const double* xrm, const double* yrm,
    int Q, double safety_margin,
    double px, double py,
    double& phi, double& gxw, double& gyw)
{
    const double dx = px - pm[0], dy = py - pm[1];
    const double qx = Tm[0] * dx + Tm[1] * dy;
    const double qy = Tm[2] * dx + Tm[3] * dy;
    double raw = 0.0, rgx = 0.0, rgy = 0.0;
    const bool raw_ok = bp_raw_gpu(Wm, Q, xrm[0], xrm[1], yrm[0], yrm[1], qx, qy, raw, rgx, rgy);
    const double level = raw_ok ? raw - safety_margin : 1.0;
    const bool in_domain = qx >= xrm[0] && qx <= xrm[1] && qy >= yrm[0] && qy <= yrm[1];
    const double sign = (in_domain && level < 0.0) ? -1.0 : 1.0;

    double zx = clampd_gpu(qx, xrm[0], xrm[1]);
    double zy = clampd_gpu(qy, yrm[0], yrm[1]);
    bool projected = false;
    for (int it = 0; it < 12; ++it) {
        double zr = 0.0, zgx = 0.0, zgy = 0.0;
        if (!bp_raw_gpu(Wm, Q, xrm[0], xrm[1], yrm[0], yrm[1], zx, zy, zr, zgx, zgy)) break;
        const double f = zr - safety_margin;
        if (fabs(f) <= 1.0e-8) { projected = true; break; }
        const double g2 = zgx * zgx + zgy * zgy;
        if (!(g2 > 1.0e-16)) break;
        double step = f / g2;
        const double step_norm = fabs(step) * sqrt(g2);
        const double max_step = 0.35 * fmax(xrm[1] - xrm[0], yrm[1] - yrm[0]);
        if (step_norm > max_step && step_norm > 1.0e-16) step *= max_step / step_norm;
        zx = clampd_gpu(zx - step * zgx, xrm[0], xrm[1]);
        zy = clampd_gpu(zy - step * zgy, yrm[0], yrm[1]);
        if (it == 11) {
            double rr = 0.0, ggx = 0.0, ggy = 0.0;
            projected = bp_raw_gpu(Wm, Q, xrm[0], xrm[1], yrm[0], yrm[1], zx, zy, rr, ggx, ggy)
                && fabs(rr - safety_margin) <= 1.0e-5;
        }
    }

    if (projected) {
        const double ddx = qx - zx, ddy = qy - zy;
        const double dist = sqrt(ddx * ddx + ddy * ddy);
        phi = sign * dist;
        double gqx = 0.0, gqy = 0.0;
        if (dist > 1.0e-12) {
            gqx = sign * ddx / dist;
            gqy = sign * ddy / dist;
        } else {
            const double gn = sqrt(rgx * rgx + rgy * rgy);
            if (gn > 1.0e-12) { gqx = rgx / gn; gqy = rgy / gn; }
        }
        gxw = Tm[0] * gqx + Tm[2] * gqy;
        gyw = Tm[1] * gqx + Tm[3] * gqy;
        return isfinite(phi) && isfinite(gxw) && isfinite(gyw);
    }

    // Same pathological fallback as the CPU evaluator.
    phi = level;
    gxw = Tm[0] * rgx + Tm[2] * rgy;
    gyw = Tm[1] * rgx + Tm[3] * rgy;
    return isfinite(phi) && isfinite(gxw) && isfinite(gyw);
}

__global__ void bp_query_reduce_kernel(
    const double* __restrict__ X,
    const double* __restrict__ W,
    const double* __restrict__ T,
    const double* __restrict__ pref,
    const double* __restrict__ xr,
    const double* __restrict__ yr,
    const int32_t* __restrict__ empty,
    int H, int model_stage_count, int O, int Q, int variants,
    double safety_margin, double eps_pos,
    double* __restrict__ best)
{
    const int hv = static_cast<int>(blockIdx.x);
    const int h = hv / variants;
    const int v = hv - h * variants;
    if (h >= H) return;
    double px = X[static_cast<std::size_t>(h) * 4U + 0U];
    double py = X[static_cast<std::size_t>(h) * 4U + 1U];
    if (v == 1) px += eps_pos;
    else if (v == 2) px -= eps_pos;
    else if (v == 3) py += eps_pos;
    else if (v == 4) py -= eps_pos;
    const int mh = model_stage_count == 1 ? 0 : h;

    double local_phi = bp_pos_inf();
    double local_gx = 0.0, local_gy = 0.0;
    int local_idx = 0x7fffffff;
    for (int o = static_cast<int>(threadIdx.x); o < O; o += static_cast<int>(blockDim.x)) {
        const std::size_t mo = static_cast<std::size_t>(mh) * static_cast<std::size_t>(O) + static_cast<std::size_t>(o);
        if (empty[mo] != 0) continue;
        const double* Wm = W + mo * static_cast<std::size_t>(Q) * static_cast<std::size_t>(Q);
        const double* Tm = T + mo * 4U;
        const double* pm = pref + mo * 2U;
        const double* xrm = xr + mo * 2U;
        const double* yrm = yr + mo * 2U;
        double phi = bp_pos_inf(), gx = 0.0, gy = 0.0;
        if (!bp_metric_model_gpu(Wm, Tm, pm, xrm, yrm, Q, safety_margin, px, py, phi, gx, gy)) continue;
        if (phi < local_phi || (phi == local_phi && o < local_idx)) {
            local_phi = phi; local_gx = gx; local_gy = gy; local_idx = o;
        }
    }

    extern __shared__ unsigned char smem_raw[];
    double* sphi = reinterpret_cast<double*>(smem_raw);
    double* sgx = sphi + blockDim.x;
    double* sgy = sgx + blockDim.x;
    int* sidx = reinterpret_cast<int*>(sgy + blockDim.x);
    const int t = static_cast<int>(threadIdx.x);
    sphi[t] = local_phi; sgx[t] = local_gx; sgy[t] = local_gy; sidx[t] = local_idx;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            const double bp = sphi[t + stride];
            const int bi = sidx[t + stride];
            if (bp < sphi[t] || (bp == sphi[t] && bi < sidx[t])) {
                sphi[t] = bp; sgx[t] = sgx[t + stride]; sgy[t] = sgy[t + stride]; sidx[t] = bi;
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        const std::size_t base = static_cast<std::size_t>(hv) * 3U;
        best[base + 0U] = sphi[0];
        best[base + 1U] = sgx[0];
        best[base + 2U] = sgy[0];
    }
}

__device__ __forceinline__ void bp_residual_gpu(
    double phi, double gx, double gy, double yaw,
    double d_safe, double Ns, double regulation_lambda, double normal_eps,
    double lcbx, double lcby, double disabled, int impact_enabled,
    double& ga, double& gr)
{
    if (!isfinite(phi)) { ga = d_safe - (-disabled); gr = disabled; return; }
    ga = d_safe - phi;
    if (!impact_enabled) { gr = disabled; return; }
    const double gn = sqrt(gx * gx + gy * gy);
    if (!isfinite(gn) || gn <= normal_eps) { gr = disabled; return; }
    const double nfx = -gx / gn, nfy = -gy / gn;
    const double c = cos(yaw), s = sin(yaw);
    const double lcwx = c * lcbx - s * lcby;
    const double lcwy = s * lcbx + c * lcby;
    const double align = nfx * lcwx + nfy * lcwy;
    const double lam = fmax(fabs(regulation_lambda), 1.0e-9);
    const double za = (phi - d_safe) / lam;
    const double zb = (Ns - align) / lam;
    const double zm = fmax(za, zb);
    const double lse = zm + log(exp(za - zm) + exp(zb - zm));
    gr = lam * (log(2.0) - lse);
    if (!isfinite(gr)) gr = disabled;
}

__global__ void bp_assemble_kernel(
    const double* __restrict__ X,
    const double* __restrict__ best,
    int H, int variants,
    double h_upper_bound, double d_safe, double Ns, double regulation_lambda,
    double normal_eps, double eps_pos, double eps_yaw,
    double lcbx, double lcby, int impact_enabled,
    double* __restrict__ out)
{
    const int h = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (h >= H) return;
    double* row = out + static_cast<std::size_t>(h) * 21U;
#pragma unroll
    for (int j = 0; j < 21; ++j) row[j] = 0.0;
    const double disabled = -h_upper_bound;
    const std::size_t b0 = static_cast<std::size_t>(h * variants) * 3U;
    double phi0 = best[b0 + 0U], gx0 = best[b0 + 1U], gy0 = best[b0 + 2U];
    if (!isfinite(phi0)) { phi0 = h_upper_bound; gx0 = 0.0; gy0 = 0.0; }
    const double yaw = X[static_cast<std::size_t>(h) * 4U + 3U];
    double ga0 = 0.0, gr0 = disabled;
    bp_residual_gpu(phi0, gx0, gy0, yaw, d_safe, Ns, regulation_lambda, normal_eps,
                    lcbx, lcby, disabled, impact_enabled, ga0, gr0);
    row[0] = ga0;
    row[7] = gr0;
    const double c = cos(yaw), s = sin(yaw);
    const double lcwx = c * lcbx - s * lcby;
    const double lcwy = s * lcbx + c * lcby;
    row[14] = -(gx0 * lcwx + gy0 * lcwy);

    if (!impact_enabled) {
        row[1] = -gx0;
        row[2] = -gy0;
        return;
    }

    // BP scalar field/gradient are yaw-independent. x/y retain the CPU central
    // differences, while yaw +/-eps reuses the nominal field exactly and only
    // changes the body-fixed sector direction in world coordinates.
    for (int axis = 0; axis < 2; ++axis) {
        const int vp = axis == 0 ? 1 : 3;
        const int vm = axis == 0 ? 2 : 4;
        const std::size_t pp = static_cast<std::size_t>(h * variants + vp) * 3U;
        const std::size_t pm = static_cast<std::size_t>(h * variants + vm) * 3U;
        double phip = best[pp + 0U], gxp = best[pp + 1U], gyp = best[pp + 2U];
        double phim = best[pm + 0U], gxm = best[pm + 1U], gym = best[pm + 2U];
        if (!isfinite(phip)) { phip = h_upper_bound; gxp = 0.0; gyp = 0.0; }
        if (!isfinite(phim)) { phim = h_upper_bound; gxm = 0.0; gym = 0.0; }
        double gap = 0.0, grp = disabled, gam = 0.0, grm = disabled;
        bp_residual_gpu(phip, gxp, gyp, yaw, d_safe, Ns, regulation_lambda, normal_eps,
                        lcbx, lcby, disabled, 1, gap, grp);
        bp_residual_gpu(phim, gxm, gym, yaw, d_safe, Ns, regulation_lambda, normal_eps,
                        lcbx, lcby, disabled, 1, gam, grm);
        row[1 + axis] = (gap - gam) / (2.0 * eps_pos);
        if (gr0 > disabled * 0.5 && grp > disabled * 0.5 && grm > disabled * 0.5)
            row[8 + axis] = (grp - grm) / (2.0 * eps_pos);
    }

    double gayp = 0.0, gryp = disabled, gaym = 0.0, grym = disabled;
    bp_residual_gpu(phi0, gx0, gy0, yaw + eps_yaw, d_safe, Ns, regulation_lambda, normal_eps,
                    lcbx, lcby, disabled, 1, gayp, gryp);
    bp_residual_gpu(phi0, gx0, gy0, yaw - eps_yaw, d_safe, Ns, regulation_lambda, normal_eps,
                    lcbx, lcby, disabled, 1, gaym, grym);
    row[4] = 0.0;
    if (gr0 > disabled * 0.5 && gryp > disabled * 0.5 && grym > disabled * 0.5)
        row[11] = (gryp - grym) / (2.0 * eps_yaw);
}

bool bp_copy_async(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind, const char* where) {
    if (bytes == 0) return true;
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, g_bp_ws.stream);
    if (e != cudaSuccess) { set_error(where, e); return false; }
    return true;
}

bool copy_async(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind, const char* where) {
    if (bytes == 0) return true;
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, g_ws.stream);
    if (e != cudaSuccess) {
        set_error(where, e);
        return false;
    }
    return true;
}

}  // namespace

bool collision_proposed_cuda_compiled() { return true; }

bool collision_proposed_cuda_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_mutex);
        set_error("cudaGetDeviceCount", e);
        return false;
    }
    if (count <= 0) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = "no CUDA device available";
        return false;
    }
    return true;
}

const char* collision_proposed_cuda_last_error() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

void collision_proposed_cuda_reset_cache() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_ws.last_p_host = nullptr;
    g_ws.last_n_host = nullptr;
    g_ws.last_p_count = 0;
    g_ws.last_n_count = 0;
}

void collision_proposed_cuda_set_profiling(bool enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_profile_enabled = enabled;
    if (!enabled) g_last_timing = CollisionProposedCudaTiming{};
}

bool collision_proposed_cuda_get_last_timing(CollisionProposedCudaTiming* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    *out = g_last_timing;
    return g_profile_enabled;
}


void collision_bp_cuda_reset_cache() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bp_ws.reset_host_cache();
}

void collision_bp_cuda_set_profiling(bool enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bp_profile_enabled = enabled;
    if (!enabled) g_bp_last_timing = CollisionBPCudaTiming{};
}

bool collision_bp_cuda_get_last_timing(CollisionBPCudaTiming* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    *out = g_bp_last_timing;
    return g_bp_profile_enabled;
}

bool collision_bp_cuda_run(
    const double* x_h4,
    const double* W,
    const double* T_ref,
    const double* p_ref,
    const double* x_range,
    const double* y_range,
    const int32_t* empty,
    int H,
    int model_stage_count,
    int O,
    int Q,
    double safety_margin,
    double h_upper_bound,
    double d_safe,
    double Ns,
    double regulation_lambda,
    double normal_eps,
    double fd_position_eps_m,
    double fd_yaw_eps_rad,
    double lcbx,
    double lcby,
    bool impact_normal_enabled,
    int threads,
    double* out_h21)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error.clear();
    g_bp_last_timing = CollisionBPCudaTiming{};
    if (!x_h4 || !W || !T_ref || !p_ref || !x_range || !y_range || !empty || !out_h21 ||
        H <= 0 || O <= 0 || Q < 2 || Q > BP_MAX_Q ||
        (model_stage_count != 1 && model_stage_count != H)) {
        g_last_error = "invalid native CUDA BP-SDF arguments";
        return false;
    }
    int device_count = 0;
    cudaError_t e = cudaGetDeviceCount(&device_count);
    if (e != cudaSuccess || device_count <= 0) {
        if (e != cudaSuccess) set_error("cudaGetDeviceCount(BP)", e);
        else g_last_error = "no CUDA device available";
        return false;
    }

    const int variants = impact_normal_enabled ? 5 : 1;
    threads = sanitize_threads(threads);
    normal_eps = fmax(fabs(normal_eps), 1.0e-12);
    fd_position_eps_m = fmax(fabs(fd_position_eps_m), 1.0e-7);
    fd_yaw_eps_rad = fmax(fabs(fd_yaw_eps_rad), 1.0e-7);
    regulation_lambda = fmax(fabs(regulation_lambda), 1.0e-9);
    const double lcn = fmax(sqrt(lcbx * lcbx + lcby * lcby), 1.0e-12);
    lcbx /= lcn; lcby /= lcn;

    if (!g_bp_ws.ensure(H, model_stage_count, O, Q, variants)) return false;
    if (g_bp_profile_enabled && !g_bp_ws.ensure_events()) return false;

    const std::size_t models = static_cast<std::size_t>(model_stage_count) * static_cast<std::size_t>(O);
    const std::size_t count_W = models * static_cast<std::size_t>(Q) * static_cast<std::size_t>(Q);
    const std::size_t count_T = models * 4U;
    const std::size_t count_2 = models * 2U;
    const std::size_t count_empty = models;
    const bool copy_W = g_bp_ws.last_W_host != W || g_bp_ws.last_W_count != count_W;
    const bool copy_T = g_bp_ws.last_T_host != T_ref || g_bp_ws.last_T_count != count_T;
    const bool copy_pref = g_bp_ws.last_pref_host != p_ref || g_bp_ws.last_pref_count != count_2;
    const bool copy_xr = g_bp_ws.last_xr_host != x_range || g_bp_ws.last_xr_count != count_2;
    const bool copy_yr = g_bp_ws.last_yr_host != y_range || g_bp_ws.last_yr_count != count_2;
    const bool copy_empty = g_bp_ws.last_empty_host != empty || g_bp_ws.last_empty_count != count_empty;
    const bool model_copied = copy_W || copy_T || copy_pref || copy_xr || copy_yr || copy_empty;

    const std::size_t bytes_x = static_cast<std::size_t>(H) * 4U * sizeof(double);
    const std::size_t bytes_W = copy_W ? count_W * sizeof(double) : 0U;
    const std::size_t bytes_T = copy_T ? count_T * sizeof(double) : 0U;
    const std::size_t bytes_pref = copy_pref ? count_2 * sizeof(double) : 0U;
    const std::size_t bytes_xr = copy_xr ? count_2 * sizeof(double) : 0U;
    const std::size_t bytes_yr = copy_yr ? count_2 * sizeof(double) : 0U;
    const std::size_t bytes_empty = copy_empty ? count_empty * sizeof(int32_t) : 0U;
    const std::size_t bytes_out = static_cast<std::size_t>(H) * 21U * sizeof(double);
    if (g_bp_profile_enabled) {
        g_bp_last_timing.h2d_bytes = bytes_x + bytes_W + bytes_T + bytes_pref + bytes_xr + bytes_yr + bytes_empty;
        g_bp_last_timing.d2h_bytes = bytes_out;
        g_bp_last_timing.model_copied = model_copied;
        e = cudaEventRecord(g_bp_ws.ev_start, g_bp_ws.stream);
        if (e != cudaSuccess) { set_error("cudaEventRecord(BP start)", e); return false; }
    }

    if (!bp_copy_async(g_bp_ws.x, x_h4, bytes_x, cudaMemcpyHostToDevice, "copy BP X")) return false;
    if (copy_W) {
        if (!bp_copy_async(g_bp_ws.W, W, bytes_W, cudaMemcpyHostToDevice, "copy BP W")) return false;
        g_bp_ws.last_W_host = W; g_bp_ws.last_W_count = count_W;
    }
    if (copy_T) {
        if (!bp_copy_async(g_bp_ws.T, T_ref, bytes_T, cudaMemcpyHostToDevice, "copy BP T")) return false;
        g_bp_ws.last_T_host = T_ref; g_bp_ws.last_T_count = count_T;
    }
    if (copy_pref) {
        if (!bp_copy_async(g_bp_ws.pref, p_ref, bytes_pref, cudaMemcpyHostToDevice, "copy BP pref")) return false;
        g_bp_ws.last_pref_host = p_ref; g_bp_ws.last_pref_count = count_2;
    }
    if (copy_xr) {
        if (!bp_copy_async(g_bp_ws.xr, x_range, bytes_xr, cudaMemcpyHostToDevice, "copy BP x_range")) return false;
        g_bp_ws.last_xr_host = x_range; g_bp_ws.last_xr_count = count_2;
    }
    if (copy_yr) {
        if (!bp_copy_async(g_bp_ws.yr, y_range, bytes_yr, cudaMemcpyHostToDevice, "copy BP y_range")) return false;
        g_bp_ws.last_yr_host = y_range; g_bp_ws.last_yr_count = count_2;
    }
    if (copy_empty) {
        if (!bp_copy_async(g_bp_ws.empty, empty, bytes_empty, cudaMemcpyHostToDevice, "copy BP empty")) return false;
        g_bp_ws.last_empty_host = empty; g_bp_ws.last_empty_count = count_empty;
    }
    if (g_bp_profile_enabled) {
        e = cudaEventRecord(g_bp_ws.ev_after_h2d, g_bp_ws.stream);
        if (e != cudaSuccess) { set_error("cudaEventRecord(BP after_h2d)", e); return false; }
    }

    const dim3 eval_grid(static_cast<unsigned int>(H * variants));
    const dim3 eval_block(static_cast<unsigned int>(threads));
    const std::size_t eval_smem = static_cast<std::size_t>(threads) * (3U * sizeof(double) + sizeof(int));
    bp_query_reduce_kernel<<<eval_grid, eval_block, eval_smem, g_bp_ws.stream>>>(
        g_bp_ws.x, g_bp_ws.W, g_bp_ws.T, g_bp_ws.pref, g_bp_ws.xr, g_bp_ws.yr, g_bp_ws.empty,
        H, model_stage_count, O, Q, variants, safety_margin, fd_position_eps_m, g_bp_ws.best);
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_error("bp_query_reduce_kernel launch", e); return false; }
    if (g_bp_profile_enabled) {
        e = cudaEventRecord(g_bp_ws.ev_after_eval, g_bp_ws.stream);
        if (e != cudaSuccess) { set_error("cudaEventRecord(BP after_eval)", e); return false; }
    }

    const int assemble_threads = 128;
    const int assemble_blocks = (H + assemble_threads - 1) / assemble_threads;
    bp_assemble_kernel<<<dim3(static_cast<unsigned int>(assemble_blocks)), dim3(static_cast<unsigned int>(assemble_threads)), 0, g_bp_ws.stream>>>(
        g_bp_ws.x, g_bp_ws.best, H, variants, h_upper_bound, d_safe, Ns, regulation_lambda,
        normal_eps, fd_position_eps_m, fd_yaw_eps_rad, lcbx, lcby,
        impact_normal_enabled ? 1 : 0, g_bp_ws.out);
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_error("bp_assemble_kernel launch", e); return false; }
    if (g_bp_profile_enabled) {
        e = cudaEventRecord(g_bp_ws.ev_after_assemble, g_bp_ws.stream);
        if (e != cudaSuccess) { set_error("cudaEventRecord(BP after_assemble)", e); return false; }
    }

    if (!bp_copy_async(out_h21, g_bp_ws.out, bytes_out, cudaMemcpyDeviceToHost, "copy BP output")) return false;
    if (g_bp_profile_enabled) {
        e = cudaEventRecord(g_bp_ws.ev_after_d2h, g_bp_ws.stream);
        if (e != cudaSuccess) { set_error("cudaEventRecord(BP after_d2h)", e); return false; }
    }
    e = cudaStreamSynchronize(g_bp_ws.stream);
    if (e != cudaSuccess) { set_error("cudaStreamSynchronize(BP)", e); return false; }

    if (g_bp_profile_enabled) {
        float h2d = 0.0f, ke = 0.0f, ka = 0.0f, d2h = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&h2d, g_bp_ws.ev_start, g_bp_ws.ev_after_h2d);
        cudaEventElapsedTime(&ke, g_bp_ws.ev_after_h2d, g_bp_ws.ev_after_eval);
        cudaEventElapsedTime(&ka, g_bp_ws.ev_after_eval, g_bp_ws.ev_after_assemble);
        cudaEventElapsedTime(&d2h, g_bp_ws.ev_after_assemble, g_bp_ws.ev_after_d2h);
        cudaEventElapsedTime(&total, g_bp_ws.ev_start, g_bp_ws.ev_after_d2h);
        g_bp_last_timing.h2d_ms = static_cast<double>(h2d);
        g_bp_last_timing.kernel_eval_ms = static_cast<double>(ke);
        g_bp_last_timing.kernel_assemble_ms = static_cast<double>(ka);
        g_bp_last_timing.d2h_ms = static_cast<double>(d2h);
        g_bp_last_timing.total_device_ms = static_cast<double>(total);
    }
    return true;
}

bool collision_proposed_cuda_run(
    const double* x_h4,
    const double* points,
    int point_stage_count,
    const double* normals,
    int normal_stage_count,
    const double* dynamic_mask,
    const int32_t* group_offsets,
    const int32_t* group_indices,
    const double* h_steps,
    const double* dt_vals,
    int H,
    int K,
    int O,
    double sx,
    double sy,
    double geom_buffer,
    double lambda,
    double regulation_lambda,
    double sigma_iso,
    double confidence,
    bool use_probability_margin,
    double pred_rate,
    double dt_init,
    double dt_min,
    double dt_max,
    double h_upper_bound,
    bool flip_normal,
    bool use_scaled,
    double d_safe,
    double Ns,
    double normal_eps,
    double lcbx,
    double lcby,
    double lc_local_x,
    double lc_local_y,
    double disabled,
    int threads,
    double* out_h21)
{
    (void)h_upper_bound;  // disabled=-h_upper_bound is supplied explicitly by the caller.
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error.clear();
    g_last_timing = CollisionProposedCudaTiming{};
    if (!x_h4 || !points || !normals || !dynamic_mask || !group_offsets ||
        !group_indices || !h_steps || !dt_vals || !out_h21 || H <= 0 || K <= 0 || O <= 0) {
        g_last_error = "invalid native CUDA proposed-field arguments";
        return false;
    }
    int device_count = 0;
    cudaError_t e = cudaGetDeviceCount(&device_count);
    if (e != cudaSuccess || device_count <= 0) {
        if (e != cudaSuccess) set_error("cudaGetDeviceCount", e);
        else g_last_error = "no CUDA device available";
        return false;
    }

    point_stage_count = point_stage_count == 1 ? 1 : H;
    normal_stage_count = normal_stage_count == 1 ? 1 : H;
    regulation_lambda = fmax(fabs(regulation_lambda), 1.0e-12);
    const std::size_t p_count = static_cast<std::size_t>(point_stage_count) * K * 2U;
    const std::size_t n_count = static_cast<std::size_t>(normal_stage_count) * K * 2U;
    if (!g_ws.ensure(H, K, O, p_count, n_count)) return false;
    if (g_profile_enabled && !g_ws.ensure_events()) return false;

    const bool copy_points = (g_ws.last_p_host != points || g_ws.last_p_count != p_count);
    const bool copy_normals = (g_ws.last_n_host != normals || g_ws.last_n_count != n_count);
    const std::size_t bytes_x = static_cast<std::size_t>(H) * 4U * sizeof(double);
    const std::size_t bytes_dyn = static_cast<std::size_t>(K) * sizeof(double);
    const std::size_t bytes_offsets = (static_cast<std::size_t>(O) + 1U) * sizeof(int32_t);
    const std::size_t bytes_indices = static_cast<std::size_t>(K) * sizeof(int32_t);
    const std::size_t bytes_h = static_cast<std::size_t>(H) * sizeof(double);
    const std::size_t bytes_dt = static_cast<std::size_t>(H) * sizeof(double);
    const std::size_t bytes_p = copy_points ? p_count * sizeof(double) : 0U;
    const std::size_t bytes_n = copy_normals ? n_count * sizeof(double) : 0U;
    const std::size_t bytes_out = static_cast<std::size_t>(H) * 21U * sizeof(double);
    if (g_profile_enabled) {
        g_last_timing.h2d_bytes = bytes_x + bytes_dyn + bytes_offsets + bytes_indices + bytes_h + bytes_dt + bytes_p + bytes_n;
        g_last_timing.d2h_bytes = bytes_out;
        g_last_timing.feature_points_copied = copy_points;
        g_last_timing.feature_normals_copied = copy_normals;
        cudaError_t pe = cudaEventRecord(g_ws.ev_start, g_ws.stream);
        if (pe != cudaSuccess) { set_error("cudaEventRecord(start)", pe); return false; }
    }

    if (!copy_async(g_ws.x, x_h4, bytes_x, cudaMemcpyHostToDevice, "copy X")) return false;
    if (!copy_async(g_ws.dyn, dynamic_mask, bytes_dyn, cudaMemcpyHostToDevice, "copy dynamic mask")) return false;
    if (!copy_async(g_ws.offsets, group_offsets, bytes_offsets, cudaMemcpyHostToDevice, "copy group offsets")) return false;
    if (!copy_async(g_ws.indices, group_indices, bytes_indices, cudaMemcpyHostToDevice, "copy group indices")) return false;
    if (!copy_async(g_ws.hsteps, h_steps, bytes_h, cudaMemcpyHostToDevice, "copy h_steps")) return false;
    if (!copy_async(g_ws.dtvals, dt_vals, bytes_dt, cudaMemcpyHostToDevice, "copy dt_vals")) return false;

    if (copy_points) {
        if (!copy_async(g_ws.p, points, bytes_p, cudaMemcpyHostToDevice, "copy feature points")) return false;
        g_ws.last_p_host = points;
        g_ws.last_p_count = p_count;
    }
    if (copy_normals) {
        if (!copy_async(g_ws.n, normals, bytes_n, cudaMemcpyHostToDevice, "copy feature normals")) return false;
        g_ws.last_n_host = normals;
        g_ws.last_n_count = n_count;
    }
    if (g_profile_enabled) {
        cudaError_t pe = cudaEventRecord(g_ws.ev_after_h2d, g_ws.stream);
        if (pe != cudaSuccess) { set_error("cudaEventRecord(after_h2d)", pe); return false; }
    }

    threads = sanitize_threads(threads);
    const dim3 group_grid(static_cast<unsigned int>(H * O));
    const dim3 block(static_cast<unsigned int>(threads));
    const std::size_t group_smem = static_cast<std::size_t>(threads) * GROUP_COMPONENTS * sizeof(double);
    proposed_group_reduce<<<group_grid, block, group_smem, g_ws.stream>>>(
        g_ws.x, g_ws.p, g_ws.n, g_ws.dyn, g_ws.offsets, g_ws.indices,
        g_ws.hsteps, g_ws.dtvals, H, K, O, point_stage_count, normal_stage_count,
        sx, sy, geom_buffer + fmax(fabs(d_safe), 0.0), lambda, sigma_iso, confidence,
        use_probability_margin ? 1 : 0, pred_rate, dt_init, dt_min, dt_max,
        flip_normal ? 1 : 0, use_scaled ? 1 : 0,
        g_ws.log_outer, g_ws.grad_group, g_ws.dphi_group, g_ws.dgrad_group);
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_error("proposed_group_reduce launch", e); return false; }
    if (g_profile_enabled) {
        cudaError_t pe = cudaEventRecord(g_ws.ev_after_group, g_ws.stream);
        if (pe != cudaSuccess) { set_error("cudaEventRecord(after_group)", pe); return false; }
    }

    const std::size_t stage_smem = static_cast<std::size_t>(threads) * STAGE_COMPONENTS * sizeof(double);
    proposed_stage_reduce<<<dim3(static_cast<unsigned int>(H)), block, stage_smem, g_ws.stream>>>(
        g_ws.x, g_ws.log_outer, g_ws.grad_group, g_ws.dphi_group, g_ws.dgrad_group,
        H, O, lambda, regulation_lambda, d_safe, Ns, normal_eps,
        lcbx, lcby, lc_local_x, lc_local_y, use_scaled ? 1 : 0, disabled, g_ws.out);
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_error("proposed_stage_reduce launch", e); return false; }
    if (g_profile_enabled) {
        cudaError_t pe = cudaEventRecord(g_ws.ev_after_stage, g_ws.stream);
        if (pe != cudaSuccess) { set_error("cudaEventRecord(after_stage)", pe); return false; }
    }

    if (!copy_async(out_h21, g_ws.out, bytes_out, cudaMemcpyDeviceToHost, "copy output")) return false;
    if (g_profile_enabled) {
        cudaError_t pe = cudaEventRecord(g_ws.ev_after_d2h, g_ws.stream);
        if (pe != cudaSuccess) { set_error("cudaEventRecord(after_d2h)", pe); return false; }
    }
    e = cudaStreamSynchronize(g_ws.stream);
    if (e != cudaSuccess) { set_error("cudaStreamSynchronize", e); return false; }

    if (g_profile_enabled) {
        float h2d = 0.0f, kg = 0.0f, ks = 0.0f, d2h = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&h2d, g_ws.ev_start, g_ws.ev_after_h2d);
        cudaEventElapsedTime(&kg, g_ws.ev_after_h2d, g_ws.ev_after_group);
        cudaEventElapsedTime(&ks, g_ws.ev_after_group, g_ws.ev_after_stage);
        cudaEventElapsedTime(&d2h, g_ws.ev_after_stage, g_ws.ev_after_d2h);
        cudaEventElapsedTime(&total, g_ws.ev_start, g_ws.ev_after_d2h);
        g_last_timing.h2d_ms = static_cast<double>(h2d);
        g_last_timing.kernel_group_ms = static_cast<double>(kg);
        g_last_timing.kernel_stage_ms = static_cast<double>(ks);
        g_last_timing.d2h_ms = static_cast<double>(d2h);
        g_last_timing.total_device_ms = static_cast<double>(total);
    }
    return true;
}
