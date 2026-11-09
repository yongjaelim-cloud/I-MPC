#include "parseh_core.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

__global__ void ph_kernel(PHConfig cfg,const float* initial,const float* controls,
 const float* vertices,const int32_t* counts,const float* velocities,float* results,float* trajectories){
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<cfg.C)ph_evaluate_one(cfg,c,initial,controls,vertices,counts,velocities,
                               results+c*8,trajectories+(size_t)c*(cfg.H+1)*4);
}
extern "C" int ph_cuda_evaluate(const PHConfig* p,const float* initial,const float* controls,
 const float* vertices,const int32_t* counts,const float* velocities,float* results,float* trajectories){
    if(!p)return -1;
    int devcount=0;if(cudaGetDeviceCount(&devcount)!=cudaSuccess || devcount==0)return -2;
    const size_t sc=sizeof(float)*(size_t)p->C*p->H*2;
    const size_t sv=sizeof(float)*(size_t)p->H*p->O*p->V*2;
    const size_t sn=sizeof(int32_t)*(size_t)p->O;
    const size_t svel=sizeof(float)*(size_t)p->H*p->O*2;
    const size_t sr=sizeof(float)*(size_t)p->C*8;
    const size_t st=sizeof(float)*(size_t)p->C*(p->H+1)*4;
    float *di=nullptr,*dc=nullptr,*dv=nullptr,*dvel=nullptr,*dr=nullptr,*dt=nullptr;
    int32_t*dn=nullptr;
    cudaError_t err=cudaSuccess;
    #define PH_CUDA(call) do{err=(call);if(err!=cudaSuccess)goto done;}while(0)
    PH_CUDA(cudaMalloc((void**)&di,sizeof(float)*4));
    PH_CUDA(cudaMalloc((void**)&dc,sc));
    PH_CUDA(cudaMalloc((void**)&dv,sv));
    PH_CUDA(cudaMalloc((void**)&dn,sn));
    PH_CUDA(cudaMalloc((void**)&dvel,svel));
    PH_CUDA(cudaMalloc((void**)&dr,sr));
    PH_CUDA(cudaMalloc((void**)&dt,st));
    PH_CUDA(cudaMemcpy(di,initial,sizeof(float)*4,cudaMemcpyHostToDevice));
    PH_CUDA(cudaMemcpy(dc,controls,sc,cudaMemcpyHostToDevice));
    PH_CUDA(cudaMemcpy(dv,vertices,sv,cudaMemcpyHostToDevice));
    PH_CUDA(cudaMemcpy(dn,counts,sn,cudaMemcpyHostToDevice));
    PH_CUDA(cudaMemcpy(dvel,velocities,svel,cudaMemcpyHostToDevice));
    ph_kernel<<<(p->C+127)/128,128>>>(*p,di,dc,dv,dn,dvel,dr,dt);
    PH_CUDA(cudaGetLastError());
    PH_CUDA(cudaMemcpy(results,dr,sr,cudaMemcpyDeviceToHost));
    PH_CUDA(cudaMemcpy(trajectories,dt,st,cudaMemcpyDeviceToHost));
    done:
    cudaFree(di);cudaFree(dc);cudaFree(dv);cudaFree(dn);cudaFree(dvel);cudaFree(dr);cudaFree(dt);
    return err==cudaSuccess?0:-3;
}
