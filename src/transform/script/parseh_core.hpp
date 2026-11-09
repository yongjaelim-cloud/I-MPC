#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define PH_HD __host__ __device__
#else
#define PH_HD
#endif

// Packed collision geometry: vertices[(h*O+o)*V*2 + 2*v + xy].
// Only convex polygons in CCW or CW order are supported.
struct PHConfig {
    int C, H, O, V, substeps;
    float dt, length, width, margin, goal_x, goal_y, goal_weight;
    float vulnerable_penalty, impact_weight;
    unsigned int vulnerable_mask; // bit 0 front, 1 left, 2 rear, 3 right
    float v_min, v_max;
};

PH_HD inline float ph_min(float a,float b){return a<b?a:b;}
PH_HD inline float ph_max(float a,float b){return a>b?a:b;}
PH_HD inline float ph_abs(float a){return a<0?-a:a;}
PH_HD inline void ph_axes(float yaw,float& cx,float& cy,float& lx,float& ly){
    cx=cosf(yaw);cy=sinf(yaw);lx=-cy;ly=cx;
}
PH_HD inline bool ph_overlap_axis(const float* poly,int n,float x,float y,float cx,float cy,float lx,float ly,
                                   float halfL,float halfW,float margin,float ax,float ay){
    float low=1.e30f,high=-1.e30f;
    for(int i=0;i<n;++i){float p=poly[2*i]*ax+poly[2*i+1]*ay;low=ph_min(low,p);high=ph_max(high,p);}
    float ec=x*ax+y*ay;
    float r=(halfL+margin)*ph_abs(cx*ax+cy*ay)+(halfW+margin)*ph_abs(lx*ax+ly*ay);
    return !(high<ec-r || low>ec+r);
}
PH_HD inline bool ph_poly_collision(const float* poly,int n,float x,float y,float yaw,
                                     float halfL,float halfW,float margin){
    if(n<=0)return false;
    float cx,cy,lx,ly;ph_axes(yaw,cx,cy,lx,ly);
    if(!ph_overlap_axis(poly,n,x,y,cx,cy,lx,ly,halfL,halfW,margin,cx,cy))return false;
    if(!ph_overlap_axis(poly,n,x,y,cx,cy,lx,ly,halfL,halfW,margin,lx,ly))return false;
    if(n==1)return true;
    for(int j=0;j<(n==2?1:n);++j){
        int k=(j+1)%n;
        float ex=poly[2*k]-poly[2*j],ey=poly[2*k+1]-poly[2*j];
        float ax=-ey,ay=ex;
        if(ax*ax+ay*ay<1.e-15f)continue;
        if(!ph_overlap_axis(poly,n,x,y,cx,cy,lx,ly,halfL,halfW,margin,ax,ay))return false;
    }
    return true;
}
// Contact-side heuristic: closest polygon vertex to the ego center, then nearest ego face.
// This is NOT the original Parseh contact model; requires validation against it.
// Original Parseh contact-side surrogate: obstacle CENTROID direction,
// not the nearest vertex and not the physical contact normal.
PH_HD inline int ph_surface(const float* poly,int n,float x,float y,float yaw,float halfL,float halfW){
    float cx=0.f,cy=0.f;
    for(int i=0;i<n;++i){cx+=poly[2*i];cy+=poly[2*i+1];}
    cx/=n;cy/=n;
    float dx=cx-x,dy=cy-y,c=cosf(yaw),s=sinf(yaw);
    float bx=c*dx+s*dy,by=-s*dx+c*dy;
    float theta=atan2f(by,bx);
    // 1-left, 2-rear, 3-right, 4-front -> 0-front,1-left,2-rear,3-right
    float edge=atan2f(halfW,halfL);
    float a=theta<0.f?theta+6.2831853071795864769f:theta;
    if(a>=edge && a<3.141592653589793f-edge)return 1;
    if(a>=3.141592653589793f-edge && a<3.141592653589793f+edge)return 2;
    if(a>=3.141592653589793f+edge && a<6.283185307179586f-edge)return 3;
    return 0;
}

PH_HD inline void ph_evaluate_one(const PHConfig& p,int candidate,const float* initial,
    const float* controls,const float* vertices,const int32_t* counts,const float* velocities,
    float* result,float* trajectory){
    float x=initial[0],y=initial[1],v=ph_max(p.v_min,ph_min(p.v_max,initial[2])),yaw=initial[3];
    const float halfL=.5f*p.length,halfW=.5f*p.width;
    int hit=0,side=-1,obstacle=-1;
    float contact_time=-1.f,severity=0.f,closing_speed=0.f;
    trajectory[0]=x;trajectory[1]=y;trajectory[2]=v;trajectory[3]=yaw;
    const int subs=p.substeps>0?p.substeps:1;
    const float ds=p.dt/subs;
    for(int h=0;h<p.H;++h){
        float a=controls[(candidate*p.H+h)*2],omega=controls[(candidate*p.H+h)*2+1];
        for(int t=0;t<subs;++t){
            // Constant acceleration / yaw-rate integration, shared CPU/GPU.
            float midyaw=yaw+.5f*omega*ds;
            float travel=v*ds+.5f*a*ds*ds;
            x+=travel*cosf(midyaw);y+=travel*sinf(midyaw);
            v=ph_max(p.v_min,ph_min(p.v_max,v+a*ds));yaw+=omega*ds;
            // Sample stage h geometry; no interpolation between predicted obstacle stages.
            if(!hit){
                for(int o=0;o<p.O;++o){
                    int n=counts[o];if(n<1)continue;if(n>p.V)n=p.V;
                    const float* poly=vertices+((h*p.O+o)*p.V*2);
                    if(ph_poly_collision(poly,n,x,y,yaw,halfL,halfW,p.margin)){
                        hit=1;side=ph_surface(poly,n,x,y,yaw,halfL,halfW);
                        obstacle=o;contact_time=h*p.dt+(t+1)*ds;
                        float nx=cosf(yaw),ny=sinf(yaw);
                        if(side==1){nx=-sinf(yaw);ny=cosf(yaw);}
                        else if(side==2){nx=-cosf(yaw);ny=-sinf(yaw);}
                        else if(side==3){nx=sinf(yaw);ny=-cosf(yaw);}
                        float vx=velocities[(h*p.O+o)*2],vy=velocities[(h*p.O+o)*2+1];
                        closing_speed=ph_max(0.f, (v*cosf(yaw)-vx)*nx+(v*sinf(yaw)-vy)*ny);
                        // Relative-speed magnitude, matching original Parseh's W1*Vrel.
                        closing_speed=hypotf(v*cosf(yaw)-vx,v*sinf(yaw)-vy);
                        severity=(p.vulnerable_mask&(1u<<side))?12.f:1.f;
                        break;
                    }
                }
            }
        }
        float* st=trajectory+(h+1)*4;
        st[0]=x;st[1]=y;st[2]=v;st[3]=yaw;
    }
    float dx=x-p.goal_x,dy=y-p.goal_y;
    float goal_cost=p.goal_weight*(dx*dx+dy*dy);
    float impact_cost=hit?severity:0.f;
    if(!hit){
        // Original noncollision score uses Vrel at terminal time.
        float closest=1.e30f,ovx=0.f,ovy=0.f;
        const int last=p.H-1;
        for(int o=0;o<p.O;++o){
            int n=counts[o];if(n<1)continue;
            const float* poly=vertices+((last*p.O+o)*p.V*2);
            float cx=0.f,cy=0.f;
            for(int i=0;i<n;++i){cx+=poly[2*i];cy+=poly[2*i+1];}
            cx/=n;cy/=n;
            float dd=(x-cx)*(x-cx)+(y-cy)*(y-cy);
            if(dd<closest){closest=dd;ovx=velocities[(last*p.O+o)*2];ovy=velocities[(last*p.O+o)*2+1];}
        }
        closing_speed=hypotf(v*cosf(yaw)-ovx,v*sinf(yaw)-ovy);
    }
    // Outputs: [rank cost, collided, time, surface, severity, closing speed, obstacle, goal cost]
    result[0]=goal_cost+impact_cost+p.impact_weight*closing_speed;result[1]=(float)hit;result[2]=contact_time;
    result[3]=(float)side;result[4]=severity;result[5]=closing_speed;
    result[6]=(float)obstacle;result[7]=goal_cost;
}
