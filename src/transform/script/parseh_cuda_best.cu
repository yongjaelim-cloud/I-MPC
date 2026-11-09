#include "parseh_core.hpp"
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace {
struct Workspace {
 float *initial=nullptr,*controls=nullptr,*vertices=nullptr,*velocities=nullptr,*scores=nullptr,*trajs=nullptr;
 int32_t *counts=nullptr;
 int *best=nullptr;
 size_t ccap=0, vcap=0, ncap=0, velcap=0, rcap=0, tcap=0;
 int countcap=0;
 ~Workspace(){cudaFree(initial);cudaFree(controls);cudaFree(vertices);cudaFree(velocities);cudaFree(scores);cudaFree(trajs);cudaFree(counts);cudaFree(best);}
};
std::mutex gpu_lock;
Workspace ws;
// Keep capacity and reuse buffers across planning cycles.
template<class T> cudaError_t reserve(T*& p,size_t& cap,size_t n){
 if(cap>=n)return cudaSuccess;
 if(p){auto e=cudaFree(p);if(e!=cudaSuccess)return e;p=nullptr;}
 auto e=cudaMalloc((void**)&p,n);if(e==cudaSuccess)cap=n;return e;
}
__global__ void evaluate(PHConfig p,const float* initial,const float* controls,
 const float* vertices,const int32_t* counts,const float* velocities,float* scores,float* trajectories){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<p.C)ph_evaluate_one(p,i,initial,controls,vertices,counts,velocities,
                    scores+(size_t)i*8,trajectories+(size_t)i*(p.H+1)*4);
}
// One thread scans 5000 values (<100 KB). This is a small, deterministic
// reduction and avoids copying the full score table to the host.
__global__ void select_best(const float* scores,int C,int* idx){
 if(blockIdx.x||threadIdx.x)return;
 int best=0;
 for(int i=1;i<C;++i){
  const float *a=scores+(size_t)i*8,*b=scores+(size_t)best*8;
  bool af=isfinite(a[0]),bf=isfinite(b[0]);
  if((af&&!bf)||(af&&bf&&(a[1]<b[1]||(a[1]==b[1]&&a[0]<b[0]))))best=i;
 }
 *idx=best;
}
}
extern "C" int ph_cuda_best(const PHConfig* p,const float* initial,const float* controls,
 const float* vertices,const int32_t* counts,const float* velocities,
 float* bestscore,float* besttraj,int* bestindex){
 if(!p||!initial||!controls||!vertices||!counts||!velocities||!bestscore||!besttraj||!bestindex)return -1;
 std::lock_guard<std::mutex> guard(gpu_lock);
 int devices=0;
 if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return -2;
 size_t sc=(size_t)p->C*p->H*2*sizeof(float), sv=(size_t)p->H*p->O*p->V*2*sizeof(float);
 size_t sn=(size_t)p->O*sizeof(int32_t), vel=(size_t)p->H*p->O*2*sizeof(float);
 size_t sr=(size_t)p->C*8*sizeof(float), st=(size_t)p->C*(p->H+1)*4*sizeof(float);
 cudaError_t e=cudaSuccess;
 #define CK(f) do {e=(f);if(e!=cudaSuccess)return -3;}while(0)
 CK(reserve(ws.initial,ws.ncap,sizeof(float)*4));
 CK(reserve(ws.controls,ws.ccap,sc));
 CK(reserve(ws.vertices,ws.vcap,sv));
 CK(reserve(ws.velocities,ws.velcap,vel));
 CK(reserve(ws.scores,ws.rcap,sr));
 CK(reserve(ws.trajs,ws.tcap,st));
 if(!ws.best) CK(cudaMalloc((void**)&ws.best,sizeof(int)));
 if(!ws.counts||ws.countcap<p->O){if(ws.counts)CK(cudaFree(ws.counts));CK(cudaMalloc((void**)&ws.counts,sn));ws.countcap=p->O;}
 CK(cudaMemcpy(ws.initial,initial,sizeof(float)*4,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(ws.controls,controls,sc,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(ws.vertices,vertices,sv,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(ws.counts,counts,sn,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(ws.velocities,velocities,vel,cudaMemcpyHostToDevice));
 evaluate<<<(p->C+127)/128,128>>>(*p,ws.initial,ws.controls,ws.vertices,ws.counts,
                                  ws.velocities,ws.scores,ws.trajs);
 CK(cudaGetLastError());
 select_best<<<1,1>>>(ws.scores,p->C,ws.best);
 CK(cudaGetLastError());
 CK(cudaMemcpy(bestindex,ws.best,sizeof(int),cudaMemcpyDeviceToHost));
 if(*bestindex<0||*bestindex>=p->C)return -4;
 CK(cudaMemcpy(bestscore,ws.scores+(size_t)*bestindex*8,8*sizeof(float),cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(besttraj,ws.trajs+(size_t)*bestindex*(p->H+1)*4,
               (size_t)(p->H+1)*4*sizeof(float),cudaMemcpyDeviceToHost));
 return 0;
}
