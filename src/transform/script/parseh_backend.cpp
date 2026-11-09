#include "parseh_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef PARSEH_WITH_CUDA
extern "C" int ph_cuda_evaluate(const PHConfig*,const float*,const float*,const float*,const int32_t*,const float*,float*,float*);
#endif
struct PHAxis {float x,y,lo,hi;};
struct PHObstacle {
 const float *poly=nullptr;
 int n=0;
 float x=0,y=0,r2=0;
 std::vector<PHAxis> axes;
};
static std::vector<PHObstacle> prepare(const PHConfig& p,const float* points,const int32_t* counts){
 std::vector<PHObstacle> ret((size_t)p.H*p.O);
 for(int h=0;h<p.H;++h)for(int o=0;o<p.O;++o){
   auto &r=ret[(size_t)h*p.O+o];r.n=std::min(p.V,std::max(0,(int)counts[o]));
   r.poly=points+((size_t)h*p.O+o)*p.V*2;
   if(!r.n)continue;
   for(int i=0;i<r.n;++i){r.x+=r.poly[2*i];r.y+=r.poly[2*i+1];}
   r.x/=r.n;r.y/=r.n;
   for(int i=0;i<r.n;++i){float dx=r.poly[2*i]-r.x,dy=r.poly[2*i+1]-r.y;r.r2=std::max(r.r2,dx*dx+dy*dy);}
   if(r.n<3)continue;
   r.axes.reserve((size_t)r.n);
   for(int j=0;j<r.n;++j){int k=(j+1)%r.n;float ax=-(r.poly[2*k+1]-r.poly[2*j]),ay=r.poly[2*k]-r.poly[2*j];
      if(ax*ax+ay*ay<1.e-15f)continue;
      float lo=1.e30f,hi=-1.e30f;
      for(int i=0;i<r.n;++i){float q=r.poly[2*i]*ax+r.poly[2*i+1]*ay;lo=std::min(lo,q);hi=std::max(hi,q);}
      r.axes.push_back({ax,ay,lo,hi});
   }
 }
 return ret;
}
static inline bool hit_fast(const PHObstacle& r,float x,float y,float yaw,float halfL,float halfW,float margin,
                            float c,float s){
 if(r.n==0)return false;
 float dx=x-r.x,dy=y-r.y;
 // Circular broadphase: the polygon cannot intersect the footprint if its
 // enclosing disk and the footprint enclosing disk are disjoint.
 float br=std::sqrt(r.r2)+std::hypot(halfL+margin,halfW+margin);
 if(dx*dx+dy*dy>br*br)return false;
 if(!ph_overlap_axis(r.poly,r.n,x,y,c,s,-s,c,halfL,halfW,margin,c,s))return false;
 if(!ph_overlap_axis(r.poly,r.n,x,y,c,s,-s,c,halfL,halfW,margin,-s,c))return false;
 if(r.n<3)return ph_poly_collision(r.poly,r.n,x,y,yaw,halfL,halfW,margin);
 for(const auto& axis:r.axes){
    float ec=x*axis.x+y*axis.y;
    float rad=(halfL+margin)*std::fabs(c*axis.x+s*axis.y)
             +(halfW+margin)*std::fabs(-s*axis.x+c*axis.y);
    if(axis.hi<ec-rad || axis.lo>ec+rad)return false;
 }
 return true;
}
static void evaluate_fast(const PHConfig& p,int candidate,const float* initial,const float* controls,
  const std::vector<PHObstacle>& geometry,const float* velocities,float* result,float* trajectory){
 float x=initial[0],y=initial[1],v=ph_max(p.v_min,ph_min(p.v_max,initial[2])),yaw=initial[3];
 const float halfL=.5f*p.length,halfW=.5f*p.width;
 int hit=0,side=-1,obstacle=-1;
 float contact_time=-1.f,severity=0.f,closing_speed=0.f;
 trajectory[0]=x;trajectory[1]=y;trajectory[2]=v;trajectory[3]=yaw;
 const int subs=p.substeps>0?p.substeps:1;const float ds=p.dt/subs;
 for(int h=0;h<p.H;++h){
  float a=controls[((size_t)candidate*p.H+h)*2],omega=controls[((size_t)candidate*p.H+h)*2+1];
  for(int t=0;t<subs;++t){
   float midyaw=yaw+.5f*omega*ds;float travel=v*ds+.5f*a*ds*ds;
   x+=travel*std::cos(midyaw);y+=travel*std::sin(midyaw);
   v=ph_max(p.v_min,ph_min(p.v_max,v+a*ds));yaw+=omega*ds;
   if(!hit){
    float c=std::cos(yaw),s=std::sin(yaw);
    for(int o=0;o<p.O;++o){
      const auto& poly=geometry[(size_t)h*p.O+o];
      if(hit_fast(poly,x,y,yaw,halfL,halfW,p.margin,c,s)){
        hit=1;side=ph_surface(poly.poly,poly.n,x,y,yaw,halfL,halfW);
        obstacle=o;contact_time=h*p.dt+(t+1)*ds;
        float nx=c,ny=s;
        if(side==1){nx=-s;ny=c;}else if(side==2){nx=-c;ny=-s;}else if(side==3){nx=s;ny=-c;}
        float vx=velocities[((size_t)h*p.O+o)*2],vy=velocities[((size_t)h*p.O+o)*2+1];
        closing_speed=ph_max(0.f,(v*c-vx)*nx+(v*s-vy)*ny);
        closing_speed=std::hypot(v*c-vx,v*s-vy);
        severity=(p.vulnerable_mask&(1u<<side))?12.f:1.f;break;
      }
    }
   }
  }
  float* st=trajectory+(h+1)*4;st[0]=x;st[1]=y;st[2]=v;st[3]=yaw;
 }
 float dx=x-p.goal_x,dy=y-p.goal_y;
 float goal_cost=p.goal_weight*(dx*dx+dy*dy),impact_cost=hit?severity:0.f;
 if(!hit){
   float closest=1.e30f,ovx=0.f,ovy=0.f;
   const int h=p.H-1;
   for(int o=0;o<p.O;++o){
      const auto& obj=geometry[(size_t)h*p.O+o];if(obj.n<1)continue;
      float dd=(x-obj.x)*(x-obj.x)+(y-obj.y)*(y-obj.y);
      if(dd<closest){closest=dd;ovx=velocities[((size_t)h*p.O+o)*2];ovy=velocities[((size_t)h*p.O+o)*2+1];}
   }
   closing_speed=std::hypot(v*std::cos(yaw)-ovx,v*std::sin(yaw)-ovy);
 }
 result[0]=goal_cost+impact_cost+p.impact_weight*closing_speed;result[1]=(float)hit;result[2]=contact_time;
 result[3]=(float)side;result[4]=severity;result[5]=closing_speed;
 result[6]=(float)obstacle;result[7]=goal_cost;
}
extern "C" int ph_cuda_available(){
#ifdef PARSEH_WITH_CUDA
 return 1;
#else
 return 0;
#endif
}
extern "C" int ph_run(PHConfig cfg,const float* initial,const float* controls,const float* vertices,
  const int32_t* counts,const float* velocities,int backend,float* results,float* trajectories,int* best_idx){
 if(!initial||!controls||!vertices||!counts||!velocities||!results||!trajectories||!best_idx||
 cfg.C<1||cfg.H<1||cfg.O<1||cfg.V<1||cfg.dt<=0||cfg.length<=0||cfg.width<=0||cfg.substeps<1)return -1;
 int used=0;
#ifdef PARSEH_WITH_CUDA
 if(backend==1||(backend==2&&(long long)cfg.C*cfg.H*cfg.O*cfg.V>=65536)){
    int code=ph_cuda_evaluate(&cfg,initial,controls,vertices,counts,velocities,results,trajectories);
    if(code==0)used=1;else if(backend==1)return -2;
 }
#else
 if(backend==1)return -2;
#endif
 if(!used){
   const auto geom=prepare(cfg,vertices,counts);
   // Avoid OpenMP overhead for small workloads, including the original 9x5 library.
   const long long work=(long long)cfg.C*cfg.H*cfg.O*cfg.substeps;
   const bool parallel=(work>=30000 && cfg.C>=64);
   #pragma omp parallel for schedule(static) if(parallel)
   for(int c=0;c<cfg.C;++c)
      evaluate_fast(cfg,c,initial,controls,geom,velocities,results+(size_t)c*8,
                    trajectories+(size_t)c*(cfg.H+1)*4);
 }
 int best=0;
 for(int c=1;c<cfg.C;++c){const float* a=results+(size_t)c*8,*b=results+(size_t)best*8;
    if(a[1]<b[1]||(a[1]==b[1]&&a[0]<b[0]))best=c;}
 *best_idx=best;return used;
}

#ifdef PARSEH_WITH_CUDA
extern "C" int ph_cuda_best(const PHConfig*,const float*,const float*,const float*,
 const int32_t*,const float*,float*,float*,int*);
#endif
// Selected-candidate API: no full CxHx4 result transfer to Python.
extern "C" int ph_run_best(PHConfig cfg,const float* initial,const float* controls,
 const float* vertices,const int32_t* counts,const float* velocities,int backend,
 float* best_score,float* best_traj,int* best_idx){
 if(cfg.C<1||cfg.H<1||cfg.O<1||cfg.V<1||cfg.dt<=0||cfg.substeps<1||
    !initial||!controls||!vertices||!counts||!velocities||!best_score||!best_traj||!best_idx)return -1;
#ifdef PARSEH_WITH_CUDA
 if(backend!=0){
  int rc=ph_cuda_best(&cfg,initial,controls,vertices,counts,velocities,best_score,best_traj,best_idx);
  if(rc==0)return 1;
  if(backend==1)return -2;
 }
#else
 if(backend==1)return -2;
#endif
 const auto geom=prepare(cfg,vertices,counts);
 std::vector<float> scores((size_t)cfg.C*8);
 std::vector<float> states((size_t)cfg.C*(cfg.H+1)*4);
 const long long work=(long long)cfg.C*cfg.H*cfg.O*cfg.substeps;
 #pragma omp parallel for schedule(static) if(work>=30000 && cfg.C>=64)
 for(int c=0;c<cfg.C;++c)
  evaluate_fast(cfg,c,initial,controls,geom,velocities,scores.data()+(size_t)c*8,
                states.data()+(size_t)c*(cfg.H+1)*4);
 int best=0;
 for(int i=1;i<cfg.C;++i){
  const float* a=scores.data()+(size_t)i*8,*b=scores.data()+(size_t)best*8;
  if((std::isfinite(a[0])&&!std::isfinite(b[0]))||
     (std::isfinite(a[0])&&std::isfinite(b[0])&&
      (a[1]<b[1]||(a[1]==b[1]&&a[0]<b[0]))))best=i;
 }
 *best_idx=best;
 std::copy_n(scores.data()+(size_t)best*8,8,best_score);
 std::copy_n(states.data()+(size_t)best*(cfg.H+1)*4,(size_t)(cfg.H+1)*4,best_traj);
 return 0;
}
