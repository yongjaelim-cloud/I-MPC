#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr double EPS = 1e-12;

struct Vec2 {
    double x{0.0}, y{0.0};
};
inline Vec2 operator+(Vec2 a, Vec2 b){ return {a.x+b.x,a.y+b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b){ return {a.x-b.x,a.y-b.y}; }
inline Vec2 operator*(Vec2 a, double s){ return {a.x*s,a.y*s}; }
inline Vec2 operator/(Vec2 a, double s){ return {a.x/s,a.y/s}; }
inline double dot(Vec2 a, Vec2 b){ return a.x*b.x+a.y*b.y; }
inline double cross(Vec2 a, Vec2 b){ return a.x*b.y-a.y*b.x; }
inline double norm2(Vec2 a){ return dot(a,a); }
inline double norm(Vec2 a){ return std::sqrt(norm2(a)); }
inline Vec2 normalized(Vec2 a){ double n=norm(a); return n>EPS ? a/n : Vec2{}; }

struct Edge {
    Vec2 a,b,s,a_rel;
    double num_t{0.0}; // cross(a-ego, s), constant for all rays in this cycle
    int type{0};
    Vec2 vel;
};
struct Ray {
    double rel_deg{0.0}; Vec2 dir, endpoint; double distance{0.0};
    bool hit{false}; int hit_type{0}; Vec2 hit_vel; int edge_idx{-1};
};
struct Segment {
    Vec2 center, direction, normal, normal_out, start, end;
    double max_error{0.0}, rms_error{0.0};
};
struct TerminalEdge {
    int side{0}; // -1 start, +1 end
    Vec2 start,end,direction,normal_out;
};
struct SelectedRow {
    double dist{0.0}; Vec2 xy; int obs_type{0}, obs_idx{0}, group_id{0}, local_idx{0}, partition_idx{0};
};
struct Group {
    int obs_idx{0}; int obs_type{0}; Vec2 velocity; double nearest{0.0};
    std::vector<int> ray_indices;
    std::vector<Vec2> hit_points;
    std::vector<Segment> segments;
    std::vector<Vec2> vertices;
    std::vector<TerminalEdge> terminal_edges;
};

inline double polygon_area(const std::vector<Vec2>& p){
    if(p.size()<3) return 0.0;
    double s=0.0; for(size_t i=0;i<p.size();++i){const auto&a=p[i]; const auto&b=p[(i+1)%p.size()]; s+=cross(a,b);} return 0.5*s;
}

std::vector<Vec2> dedupe(const std::vector<Vec2>& in, double tol){
    std::vector<Vec2> out; out.reserve(in.size());
    const double t2=tol*tol;
    for(const auto& v: in){ if(out.empty() || norm2(v-out.back())>t2) out.push_back(v); }
    if(out.size()>1 && norm2(out.front()-out.back())<=t2) out.pop_back();
    return out;
}

bool is_convex(const std::vector<Vec2>& p,double tol=1e-10){
    if(p.size()<3) return false; int sign=0;
    for(size_t i=0;i<p.size();++i){
        Vec2 a=p[(i+1)%p.size()]-p[i], b=p[(i+2)%p.size()]-p[(i+1)%p.size()];
        double z=cross(a,b); if(std::abs(z)<=tol) continue; int s=z>0?1:-1;
        if(sign==0) sign=s; else if(sign!=s) return false;
    }
    return sign!=0;
}

bool point_in_tri(Vec2 p,Vec2 a,Vec2 b,Vec2 c,double tol=1e-12){
    double c1=cross(b-a,p-a), c2=cross(c-b,p-b), c3=cross(a-c,p-c);
    return c1>=-tol && c2>=-tol && c3>=-tol;
}

// Convexify instead of partitioning: every obstacle polygon (convex or not)
// is replaced by its convex hull and returned as ONE convex part, so a
// nonconvex obstacle is never split into multiple groups.  The name is kept
// for call-site compatibility.
bool ear_clip(std::vector<Vec2> p, std::vector<std::vector<Vec2>>& parts){
    parts.clear(); p=dedupe(p,1e-10); if(p.size()<3 || std::abs(polygon_area(p))<1e-10) return false;
    if(polygon_area(p)<0) std::reverse(p.begin(),p.end());
    if(is_convex(p)){parts.push_back(p);return true;}
    std::vector<Vec2> pts=p;
    std::sort(pts.begin(),pts.end(),[](const Vec2&a,const Vec2&b){return a.x<b.x||(a.x==b.x&&a.y<b.y);});
    std::vector<Vec2> h(2*pts.size()); size_t k=0;
    for(size_t i=0;i<pts.size();++i){ while(k>=2 && cross(h[k-1]-h[k-2],pts[i]-h[k-2])<=1e-12) --k; h[k++]=pts[i]; }
    for(size_t i=pts.size()-1,t=k+1;i-->0;){ while(k>=t && cross(h[k-1]-h[k-2],pts[i]-h[k-2])<=1e-12) --k; h[k++]=pts[i]; }
    h.resize(k>0?k-1:0);
    if(h.size()<3 || std::abs(polygon_area(h))<1e-10) return false;
    parts.push_back(h);  // CCW
    return true;
}

inline double distance_point_segment(Vec2 p,Vec2 a,Vec2 b){
    Vec2 e=b-a; double d=norm2(e); if(d<=EPS) return norm(p-a);
    double t=std::clamp(dot(p-a,e)/d,0.0,1.0); return norm(p-(a+e*t));
}

bool line_intersection(Vec2 p1,Vec2 d1,Vec2 p2,Vec2 d2,Vec2& out){
    double den=cross(d1,d2); if(std::abs(den)<=1e-10) return false;
    double t=cross(p2-p1,d2)/den; out=p1+d1*t; return std::isfinite(out.x)&&std::isfinite(out.y);
}

Segment fit_tls(const std::vector<Vec2>& pts, bool& ok, Vec2 ego){
    ok=false; Segment f; if(pts.size()<2) return f;
    Vec2 c{}; for(auto p:pts)c=c+p; c=c/(double)pts.size();
    double xx=0,xy=0,yy=0; for(auto p:pts){Vec2 q=p-c;xx+=q.x*q.x;xy+=q.x*q.y;yy+=q.y*q.y;}
    double theta=0.5*std::atan2(2.0*xy,xx-yy); Vec2 d{std::cos(theta),std::sin(theta)};
    if(norm2(d)<=EPS) return f;
    Vec2 traversal=pts.back()-pts.front(); if(dot(d,traversal)<0)d=d*-1.0;
    Vec2 n{-d.y,d.x};
    double minp=std::numeric_limits<double>::infinity(),maxp=-minp,maxerr=0,sse=0;
    for(auto p:pts){Vec2 q=p-c; double pr=dot(q,d); minp=std::min(minp,pr);maxp=std::max(maxp,pr);double e=std::abs(dot(q,n));maxerr=std::max(maxerr,e);sse+=e*e;}
    f.center=c;f.direction=d;f.normal=n;f.start=c+d*minp;f.end=c+d*maxp;f.max_error=maxerr;f.rms_error=std::sqrt(sse/pts.size());
    Vec2 no=n; if(dot(no,ego-c)<0)no=no*-1.0; f.normal_out=no; ok=true; return f;
}

void split_fit_rec(const std::vector<Vec2>& P,int depth,int min_pts,double max_error,double min_len,Vec2 ego,std::vector<Segment>& out){
    if(P.size()<2)return;
    std::vector<double> residual(P.size(),0.0);
    if(P.size()>=3){
        Vec2 chord=P.back()-P.front(); double L=norm(chord);
        if(L<=EPS){ Vec2 c{};for(auto p:P)c=c+p;c=c/(double)P.size(); for(size_t i=0;i<P.size();++i)residual[i]=norm(P[i]-c); }
        else {Vec2 n{-chord.y/L,chord.x/L};for(size_t i=0;i<P.size();++i)residual[i]=std::abs(dot(P[i]-P.front(),n));}
    }
    size_t split_idx=std::distance(residual.begin(),std::max_element(residual.begin(),residual.end()));
    bool can=depth<12 && (int)P.size()>=2*min_pts-1 && residual[split_idx]>max_error && (int)split_idx>=min_pts-1 && ((int)P.size()-(int)split_idx)>=min_pts;
    if(can){
        std::vector<Vec2> L(P.begin(),P.begin()+split_idx+1),R(P.begin()+split_idx,P.end());
        split_fit_rec(L,depth+1,min_pts,max_error,min_len,ego,out); split_fit_rec(R,depth+1,min_pts,max_error,min_len,ego,out); return;
    }
    bool ok=false; Segment f=fit_tls(P,ok,ego); if(ok && norm(f.end-f.start)>=min_len) out.push_back(f);
}

// Split-and-fit followed by a merge of adjacent nearly collinear pieces.
// Split-and-merge on the max residual alone over-segments a straight but
// noisy surface (a few 2-3 sigma range outliers trigger splits), producing
// short segments with noisy directions and fake vertices that jitter from
// frame to frame.  Merge adjacent pieces when the joint TLS fit is consistent
// with the range noise (rms test, not max test) and the directions agree.
void fit_segments(const std::vector<Vec2>& P,int min_pts,double max_error,double min_len,Vec2 ego,std::vector<Segment>& out){
    std::vector<Segment> raw;
    split_fit_rec(P,0,min_pts,max_error,min_len,ego,raw);
    if(raw.size()<2){ out.insert(out.end(),raw.begin(),raw.end()); return; }
    const double cos_merge=std::cos(12.0*M_PI/180.0);
    std::vector<Segment> merged; merged.push_back(raw.front());
    for(size_t i=1;i<raw.size();++i){
        Segment& a=merged.back(); const Segment& b=raw[i];
        bool tried=false;
        // Direction of a short piece (few noisy points) is unreliable, so the
        // angle test is applied only when both pieces are long; the joint-fit
        // residual test below decides in every case.
        const double la=norm(a.end-a.start), lb=norm(b.end-b.start);
        const bool short_piece=std::min(la,lb)<0.3;
        if(short_piece || std::abs(dot(a.direction,b.direction))>=cos_merge){
            Vec2 chord=b.end-a.start; double L=norm(chord);
            if(L>EPS){
                Vec2 d=chord/L, n{-d.y,d.x};
                std::vector<Vec2> pts;
                for(const auto&p:P){
                    double u=dot(p-a.start,d);
                    if(u<-max_error||u>L+max_error) continue;
                    if(std::abs(dot(p-a.start,n))>4.0*max_error) continue;
                    pts.push_back(p);
                }
                if(pts.size()>=2){
                    bool ok=false; Segment f=fit_tls(pts,ok,ego);
                    // Noise-adaptive: merging must not make the fit clearly worse
                    // than the pieces themselves (a real corner fails this by far).
                    double piece_rms=std::max(a.rms_error,b.rms_error);
                    double rms_ok=std::max(0.6*max_error,1.3*piece_rms);
                    double max_ok=std::max(2.5*max_error,4.0*f.rms_error);
                    if(ok && f.rms_error<=rms_ok && f.max_error<=max_ok){ a=f; tried=true; }
                }
            }
        }
        if(!tried) merged.push_back(b);
    }
    out.insert(out.end(),merged.begin(),merged.end());
}

Vec2 terminal_vertex(const Segment& seg,const Ray* ray,Vec2 fallback,Vec2 origin,double max_range,double max_ext,double fit_err){
    if(!ray)return fallback; Vec2 d=normalized(ray->dir); if(norm2(d)<=EPS)return fallback;
    Vec2 q; if(!line_intersection(seg.start,seg.direction,origin,d,q))return fallback;
    double rt=dot(q-origin,d); if(rt<-1e-8 || rt>max_range+max_ext)return fallback;
    double ext=std::max({max_ext,2.0*fit_err,0.05}); if(norm(q-ray->endpoint)>ext)return fallback; return q;
}

std::vector<Vec2> build_vertices(const std::vector<Segment>& segs,const Ray* sr,const Ray* er,Vec2 origin,double range,double ext,double fiterr){
    std::vector<Vec2> out; if(segs.empty())return out;
    Vec2 a=terminal_vertex(segs.front(),sr,segs.front().start,origin,range,ext,fiterr);
    Vec2 b=terminal_vertex(segs.back(),er,segs.back().end,origin,range,ext,fiterr);
    out.push_back(a);
    for(size_t i=0;i+1<segs.size();++i){
        Vec2 q; bool ok=line_intersection(segs[i].start,segs[i].direction,segs[i+1].start,segs[i+1].direction,q);
        Vec2 join=(segs[i].end+segs[i+1].start)*0.5;
        if(ok && std::max(norm(q-segs[i].end),norm(q-segs[i+1].start))<=ext)join=q;
        out.push_back(join);
    }
    out.push_back(b); return dedupe(out,1e-6);
}

std::vector<TerminalEdge> terminal_edges(const std::vector<Vec2>& V,const Ray* sr,const Ray* er,Vec2 origin,double range){
    std::vector<TerminalEdge> out; if(V.size()<2||!sr||!er)return out;
    const Ray* rays[2]={sr,er}; int sides[2]={-1,1}; Vec2 corners[2]={V.front(),V.back()};
    for(int k=0;k<2;++k){Vec2 d=normalized(rays[k]->dir);if(norm2(d)<=EPS)continue;Vec2 far=origin+d*range;if(norm(far-corners[k])<=1e-8)continue;Vec2 no=sides[k]<0?Vec2{d.y,-d.x}:Vec2{-d.y,d.x};out.push_back({sides[k],corners[k],far,d,no});}
    return out;
}

double median(std::vector<double> v){
    if(v.empty())return 0.0; size_t n=v.size(),m=n/2; std::nth_element(v.begin(),v.begin()+m,v.end()); double hi=v[m]; if(n%2)return hi; std::nth_element(v.begin(),v.begin()+m-1,v.begin()+m); return 0.5*(hi+v[m-1]);
}



Vec2 median_vec2(const std::vector<Vec2>& v){
    if(v.empty()) return {};
    std::vector<double> xs; xs.reserve(v.size());
    std::vector<double> ys; ys.reserve(v.size());
    for(const auto&p:v){ xs.push_back(p.x); ys.push_back(p.y); }
    return {median(xs), median(ys)};
}


// Observation of the most recent fit, per obs_idx.  The motion tracker is
// called right after the fit with the same group indices; keeping this inside
// the library leaves the ctypes ABI of vertex_selector_motion_update unchanged.
//  - hits: raw hit points (registered point-to-line against the previous lines)
//  - ends: the two visible ends of the obstacle.  An end is "real" (a body-fixed
//    corner/edge of the object) when the neighbouring ray beyond it misses or
//    hits something clearly farther.  When the neighbour is nearer, the end is
//    an occlusion boundary and slides with the occluder; near the sensing range
//    it is a range cut-off.  Only real ends give tangential motion information.
struct ObsEnd { bool known{false}; bool real{false}; Vec2 p{}, tangent{}; double sigma{0.0}; };
struct GroupObs { std::vector<Vec2> hits; ObsEnd ends[2]; };
std::vector<GroupObs> g_last_obs;

void stash_hits(const std::vector<Group>& groups, const std::vector<Ray>& rays, Vec2 origin,
                double sensing_range, double ray_step_deg, double cluster_gap){
    g_last_obs.assign(groups.size(),{});
    const int nr=(int)rays.size();
    const bool full_circle=nr>0 && rays.front().rel_deg<=-179.0 && rays.back().rel_deg>=179.0;
    const double max_angle_gap=std::max(ray_step_deg*1.6,0.1);
    const double step_rad=std::max(ray_step_deg,0.05)*M_PI/180.0;
    for(const auto&g:groups){
        if(g.obs_idx<0 || g.obs_idx>=(int)groups.size()) continue;
        GroupObs& o=g_last_obs[(size_t)g.obs_idx];
        o.hits=g.hit_points;
        if(g.ray_indices.empty() || g.vertices.size()<2 || g.segments.empty()) continue;
        for(int side=0; side<2; ++side){
            int ri=side==0?g.ray_indices.front():g.ray_indices.back();
            int ni=side==0?ri-1:ri+1;
            if(full_circle) ni=(ni+nr)%nr;
            ObsEnd e;
            e.p=side==0?g.vertices.front():g.vertices.back();
            e.tangent=normalized(side==0?g.segments.front().direction:g.segments.back().direction);
            const Ray& r=rays[(size_t)ri];
            bool real=false;
            if(ni>=0 && ni<nr && r.distance<sensing_range-0.3){
                const Ray& n=rays[(size_t)ni];
                double ag=std::abs(n.rel_deg-r.rel_deg); if(ag>180.0) ag=360.0-ag;
                if(ag<=max_angle_gap)
                    real=!n.hit || n.distance>r.distance+std::max(cluster_gap,0.1);
            }
            e.known=true; e.real=real;
            // Along-face uncertainty of the end: angular resolution projected
            // on the face (grows at grazing incidence).
            Vec2 rd=normalized(e.p-origin);
            double s_inc=std::max(0.25,std::abs(cross(rd,e.tangent)));
            double rng=norm(e.p-origin);
            e.sigma=std::max(0.03,0.5*rng*step_rad/s_inc);
            o.ends[side]=e;
        }
    }
}

} // namespace

extern "C" int vertex_selector_process(
    const double* poly_x, const double* poly_y, int poly_len,
    const int* vertex_counts, const int* obs_types,
    const double* obs_vx, const double* obs_vy,
    int obs_cnt, int stride,
    double ego_x, double ego_y, double ego_yaw,
    double sensing_range, double fov_half_deg, double ray_step_deg,
    double cluster_gap, double line_fit_max_error, int line_fit_min_points,
    double line_fit_min_length, double line_intersection_max_extension,
    int max_obs, int field_flag,
    int emit_rays,
    double* out_rays, int cap_rays,
    double* out_segments, int cap_segments,
    double* out_vertices, int cap_vertices,
    double* out_terminal_edges, int cap_terminal_edges,
    double* out_hit_points, int cap_hit_points,
    double* out_selected, int cap_selected,
    double* out_meta, int cap_meta,
    int* out_counts, int out_counts_len)
{
    // row widths: rays=10, segments=11, vertices=3, terminal=10, hits=3, selected=9, meta=4
    if(!out_counts || out_counts_len<10 || obs_cnt<0 || stride<=0) return -1;
    std::fill(out_counts,out_counts+out_counts_len,0);
    Vec2 ego{ego_x,ego_y};
    sensing_range=std::max(0.0,sensing_range); fov_half_deg=std::clamp(fov_half_deg,0.0,180.0); ray_step_deg=std::max(ray_step_deg,0.05);
    max_obs=std::max(0,max_obs);

    std::vector<Edge> edges;
    edges.reserve((size_t)obs_cnt*(size_t)stride);
    for(int oi=0;oi<obs_cnt;++oi){
        int nv=vertex_counts?std::clamp(vertex_counts[oi],0,stride):0; if(nv<3)continue;
        int base=oi*stride; if(base<0 || base+nv>poly_len)continue;
        std::vector<Vec2> P; P.reserve(nv); bool finite=true;
        for(int j=0;j<nv;++j){Vec2 p{poly_x[base+j],poly_y[base+j]}; if(!std::isfinite(p.x)||!std::isfinite(p.y)){finite=false;break;}P.push_back(p);} if(!finite)continue;
        Vec2 c{}; for(auto p:P)c=c+p;c=c/(double)P.size(); double md=norm(c-ego);
        for(int j=0;j<nv;++j) md=std::min(md,distance_point_segment(ego,P[j],P[(j+1)%nv]));
        if(md>sensing_range+1e-10)continue;
        int type=obs_types? (obs_types[oi]==1?1:0):0; Vec2 vel{obs_vx?obs_vx[oi]:0.0,obs_vy?obs_vy[oi]:0.0}; if(!std::isfinite(vel.x)||!std::isfinite(vel.y))vel={};
        for(int j=0;j<nv;++j){
            Vec2 a=P[j],b=P[(j+1)%nv],s=b-a;
            if(norm2(s)>1e-20){
                Vec2 ar=a-ego;
                edges.push_back({a,b,s,ar,cross(ar,s),type,vel});
            }
        }
    }
    if(edges.empty()) return 0;

    std::vector<double> rels;
    if(fov_half_deg>=180.0-1e-12){
        int k0=(int)std::ceil(-180.0/ray_step_deg-1e-12), k1=(int)std::floor((180.0-1e-10)/ray_step_deg);
        for(int k=k0;k<=k1;++k)rels.push_back(k*ray_step_deg);
    } else {
        int k0=(int)std::ceil(-fov_half_deg/ray_step_deg-1e-12),k1=(int)std::floor(fov_half_deg/ray_step_deg+1e-12);
        for(int k=k0;k<=k1;++k)rels.push_back(k*ray_step_deg);
    }
    if(rels.empty()) return 0;
    std::vector<Ray> rays; rays.resize(rels.size());
    const int n_rays=(int)rels.size();
    const long long ray_edge_work=(long long)n_rays*(long long)edges.size();
    // Ray tests are independent.  Keep small scans serial to avoid OpenMP launch
    // overhead, and parallelize only when the ray x edge workload is substantial.
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(ray_edge_work > 200000)
    #endif
    for(int ri=0;ri<n_rays;++ri){
        double th=ego_yaw+rels[(size_t)ri]*M_PI/180.0;
        Vec2 d{std::cos(th),std::sin(th)};
        double best=std::numeric_limits<double>::infinity();
        int bei=-1;
        for(size_t ei=0;ei<edges.size();++ei){
            const auto&e=edges[ei];
            double den=cross(d,e.s);
            if(std::abs(den)<=1e-12)continue;
            double t=e.num_t/den;
            if(t<-1e-10 || t>sensing_range+1e-10 || t>=best)continue;
            double u=cross(e.a_rel,d)/den;
            if(u>=-1e-10&&u<=1.0+1e-10){best=t;bei=(int)ei;}
        }
        Ray r; r.rel_deg=rels[(size_t)ri];r.dir=d;r.edge_idx=bei;r.hit=bei>=0;
        r.distance=r.hit?std::max(0.0,best):sensing_range;
        r.endpoint=ego+d*r.distance;
        if(r.hit){r.hit_type=edges[(size_t)bei].type;r.hit_vel=edges[(size_t)bei].vel;}
        rays[(size_t)ri]=r;
    }

    std::vector<std::vector<int>> clusters;
    const double max_angle_gap=std::max(ray_step_deg*1.6,0.1);
    for(size_t ri=0;ri<rays.size();++ri){ if(!rays[ri].hit)continue; if(clusters.empty()){clusters.push_back({(int)ri});continue;} int pidx=clusters.back().back(); double ag=std::abs(rays[ri].rel_deg-rays[pidx].rel_deg); double sg2=norm2(rays[ri].endpoint-rays[pidx].endpoint); if(ag>max_angle_gap||sg2>cluster_gap*cluster_gap)clusters.push_back({(int)ri}); else clusters.back().push_back((int)ri); }
    if(fov_half_deg>=180.0-1e-12 && clusters.size()>=2){ auto &first=clusters.front(); auto &last=clusters.back(); double wg=(rays[first.front()].rel_deg+360.0)-rays[last.back()].rel_deg; double sg2=norm2(rays[first.front()].endpoint-rays[last.back()].endpoint); if(wg<=max_angle_gap&&sg2<=cluster_gap*cluster_gap){std::vector<int> merged=last;merged.insert(merged.end(),first.begin(),first.end());clusters.front()=std::move(merged);clusters.pop_back();} }
    clusters.erase(std::remove_if(clusters.begin(),clusters.end(),[](const auto& c){return c.size()<2;}),clusters.end());

    struct Cand{double nearest; std::vector<int> idx;}; std::vector<Cand> cands;cands.reserve(clusters.size());
    for(auto& c:clusters){double nd=std::numeric_limits<double>::infinity();for(int ri:c)nd=std::min(nd,rays[ri].distance);cands.push_back({nd,c});}
    std::sort(cands.begin(),cands.end(),[](const Cand&a,const Cand&b){return a.nearest<b.nearest;});

    std::vector<Group> groups;groups.reserve(std::min((int)cands.size(),max_obs));
    std::vector<SelectedRow> selected;
    int next_group_id=0, failed=0;
    for(auto &cand:cands){ if((int)groups.size()>=max_obs)break; Group g;g.obs_idx=(int)groups.size();g.nearest=cand.nearest;g.ray_indices=cand.idx;g.hit_points.reserve(cand.idx.size());for(int ri:cand.idx)g.hit_points.push_back(rays[ri].endpoint);
        fit_segments(g.hit_points,std::max(2,line_fit_min_points),line_fit_max_error,line_fit_min_length,ego,g.segments); if(g.segments.empty()){++failed;continue;}
        const Ray* sr=&rays[cand.idx.front()]; const Ray* er=&rays[cand.idx.back()]; g.vertices=build_vertices(g.segments,sr,er,ego,sensing_range,line_intersection_max_extension,line_fit_max_error); g.terminal_edges=terminal_edges(g.vertices,sr,er,ego,sensing_range);
        std::vector<Vec2> poly; const TerminalEdge *se=nullptr,*ee=nullptr; for(auto& te:g.terminal_edges){if(te.side<0)se=&te;else ee=&te;} if(se)poly.push_back(se->end);for(auto v:g.vertices)poly.push_back(v);if(ee)poly.push_back(ee->end);if(poly.size()<3)poly=g.hit_points;poly=dedupe(poly,1e-6);if(poly.size()<3){++failed;continue;}
        if(polygon_area(poly)<0)std::reverse(poly.begin(),poly.end()); std::vector<std::vector<Vec2>> parts; if(field_flag==0)parts={poly}; else if(!ear_clip(poly,parts)){++failed;continue;}
        int dyn=0;std::vector<double> vxs,vys;for(int ri:cand.idx){if(rays[ri].hit_type==1){++dyn;vxs.push_back(rays[ri].hit_vel.x);vys.push_back(rays[ri].hit_vel.y);}}g.obs_type=(dyn*2>=(int)cand.idx.size() && dyn>0)?1:0;g.velocity=g.obs_type?Vec2{median(vxs),median(vys)}:Vec2{};
        int part_idx=0;for(auto part:parts){part=dedupe(part,1e-10);if(part.size()<3){++part_idx;continue;}if(polygon_area(part)<0)std::reverse(part.begin(),part.end());int gid=next_group_id++;for(size_t li=0;li<part.size();++li)selected.push_back({norm(part[li]-ego),part[li],g.obs_type,g.obs_idx,gid,(int)li,part_idx});++part_idx;}
        groups.push_back(std::move(g));
    }

    int nr=emit_rays?(int)rays.size():0, ns=0,nv=0,nte=0,nh=0,nsel=(int)selected.size(),nm=(int)groups.size();
    for(auto&g:groups){ns+=(int)g.segments.size();nv+=(int)g.vertices.size();nte+=(int)g.terminal_edges.size();nh+=(int)g.hit_points.size();}
    stash_hits(groups,rays,ego,sensing_range,ray_step_deg,cluster_gap);
    if(nr>cap_rays||ns>cap_segments||nv>cap_vertices||nte>cap_terminal_edges||nh>cap_hit_points||nsel>cap_selected||nm>cap_meta){out_counts[0]=nr;out_counts[1]=nm;out_counts[2]=ns;out_counts[3]=nv;out_counts[4]=nte;out_counts[5]=nh;out_counts[6]=nsel;out_counts[7]=nm;out_counts[8]=failed;out_counts[9]=next_group_id;return -2;}

    if(emit_rays&&out_rays){for(int i=0;i<nr;++i){const auto&r=rays[i];double*o=out_rays+10*i;o[0]=r.rel_deg;o[1]=r.dir.x;o[2]=r.dir.y;o[3]=r.endpoint.x;o[4]=r.endpoint.y;o[5]=r.distance;o[6]=r.hit?1.0:0.0;o[7]=(double)r.hit_type;o[8]=r.hit_vel.x;o[9]=r.hit_vel.y;}}
    int si=0,vi=0,tei=0,hi=0;for(const auto&g:groups){for(const auto&s:g.segments){double*o=out_segments+11*si++;o[0]=g.obs_idx;o[1]=s.start.x;o[2]=s.start.y;o[3]=s.end.x;o[4]=s.end.y;o[5]=s.direction.x;o[6]=s.direction.y;o[7]=s.normal_out.x;o[8]=s.normal_out.y;o[9]=s.max_error;o[10]=s.rms_error;}for(auto v:g.vertices){double*o=out_vertices+3*vi++;o[0]=g.obs_idx;o[1]=v.x;o[2]=v.y;}for(const auto&e:g.terminal_edges){double*o=out_terminal_edges+10*tei++;o[0]=g.obs_idx;o[1]=e.side;o[2]=e.start.x;o[3]=e.start.y;o[4]=e.end.x;o[5]=e.end.y;o[6]=e.direction.x;o[7]=e.direction.y;o[8]=e.normal_out.x;o[9]=e.normal_out.y;}for(auto p:g.hit_points){double*o=out_hit_points+3*hi++;o[0]=g.obs_idx;o[1]=p.x;o[2]=p.y;}double* m=out_meta+4*g.obs_idx;m[0]=g.obs_idx;m[1]=g.obs_type;m[2]=g.velocity.x;m[3]=g.velocity.y;}
    for(int i=0;i<nsel;++i){const auto&s=selected[i];double*o=out_selected+9*i;o[0]=s.dist;o[1]=s.xy.x;o[2]=s.xy.y;o[3]=s.obs_type;o[4]=s.obs_idx;o[5]=s.group_id;o[6]=s.local_idx;o[7]=0.0;o[8]=s.partition_idx;}
    out_counts[0]=nr;out_counts[1]=(int)groups.size();out_counts[2]=ns;out_counts[3]=nv;out_counts[4]=nte;out_counts[5]=nh;out_counts[6]=nsel;out_counts[7]=nm;out_counts[8]=failed;out_counts[9]=next_group_id;
    return 0;
}


extern "C" int vertex_selector_process_rays(
    const double* in_rays, int n_rays,
    double ego_x, double ego_y,
    double sensing_range, double ray_step_deg,
    double cluster_gap, double line_fit_max_error, int line_fit_min_points,
    double line_fit_min_length, double line_intersection_max_extension,
    int max_obs, int field_flag,
    double* out_segments, int cap_segments,
    double* out_vertices, int cap_vertices,
    double* out_terminal_edges, int cap_terminal_edges,
    double* out_hit_points, int cap_hit_points,
    double* out_selected, int cap_selected,
    double* out_meta, int cap_meta,
    int* out_counts, int out_counts_len)
{
    // Input ray row width 10, identical to vertex_selector_process out_rays:
    // [rel_deg, dir_x, dir_y, endpoint_x, endpoint_y, distance,
    //  hit, hit_type, hit_vx, hit_vy]
    // counts keep the legacy layout; counts[0] is zero because rays are external.
    if(!out_counts || out_counts_len<10 || n_rays<0) return -1;
    std::fill(out_counts,out_counts+out_counts_len,0);
    Vec2 ego{ego_x,ego_y};
    sensing_range=std::max(0.0,sensing_range);
    ray_step_deg=std::max(ray_step_deg,0.05);
    max_obs=std::max(0,max_obs);
    if(n_rays==0 || !in_rays) return 0;

    std::vector<Ray> rays;
    rays.reserve((size_t)n_rays);
    for(int i=0;i<n_rays;++i){
        const double* r=in_rays+10*i;
        Ray q;
        q.rel_deg=r[0];
        q.dir={r[1],r[2]};
        q.endpoint={r[3],r[4]};
        q.distance=r[5];
        q.hit=(r[6]>0.5);
        q.hit_type=((int)std::llround(r[7])==1)?1:0;
        q.hit_vel={r[8],r[9]};
        if(!std::isfinite(q.rel_deg) || !std::isfinite(q.dir.x) || !std::isfinite(q.dir.y) ||
           !std::isfinite(q.endpoint.x) || !std::isfinite(q.endpoint.y) || !std::isfinite(q.distance))
            continue;
        if(!std::isfinite(q.hit_vel.x)||!std::isfinite(q.hit_vel.y)) q.hit_vel={};
        rays.push_back(q);
    }
    if(rays.empty()) return 0;
    std::sort(rays.begin(),rays.end(),[](const Ray&a,const Ray&b){return a.rel_deg<b.rel_deg;});

    std::vector<std::vector<int>> clusters;
    const double max_angle_gap=std::max(ray_step_deg*1.6,0.1);
    for(size_t ri=0;ri<rays.size();++ri){
        if(!rays[ri].hit) continue;
        if(clusters.empty()){clusters.push_back({(int)ri});continue;}
        int pidx=clusters.back().back();
        double ag=std::abs(rays[ri].rel_deg-rays[(size_t)pidx].rel_deg);
        double sg2=norm2(rays[ri].endpoint-rays[(size_t)pidx].endpoint);
        if(ag>max_angle_gap||sg2>cluster_gap*cluster_gap) clusters.push_back({(int)ri});
        else clusters.back().push_back((int)ri);
    }
    // Preserve the full-360 wrap merge when the external cloud spans both ends.
    if(clusters.size()>=2 && rays.front().rel_deg<=-179.0 && rays.back().rel_deg>=179.0){
        auto &first=clusters.front(); auto &last=clusters.back();
        double wg=(rays[(size_t)first.front()].rel_deg+360.0)-rays[(size_t)last.back()].rel_deg;
        double sg2=norm2(rays[(size_t)first.front()].endpoint-rays[(size_t)last.back()].endpoint);
        if(wg<=max_angle_gap&&sg2<=cluster_gap*cluster_gap){
            std::vector<int> merged=last; merged.insert(merged.end(),first.begin(),first.end());
            clusters.front()=std::move(merged); clusters.pop_back();
        }
    }
    clusters.erase(std::remove_if(clusters.begin(),clusters.end(),[](const auto& c){return c.size()<2;}),clusters.end());

    struct Cand{double nearest; std::vector<int> idx;};
    std::vector<Cand> cands; cands.reserve(clusters.size());
    for(auto& c:clusters){
        double nd=std::numeric_limits<double>::infinity();
        for(int ri:c) nd=std::min(nd,rays[(size_t)ri].distance);
        cands.push_back({nd,c});
    }
    std::sort(cands.begin(),cands.end(),[](const Cand&a,const Cand&b){return a.nearest<b.nearest;});

    std::vector<Group> groups; groups.reserve(std::min((int)cands.size(),max_obs));
    std::vector<SelectedRow> selected;
    int next_group_id=0, failed=0;
    for(auto &cand:cands){
        if((int)groups.size()>=max_obs) break;
        Group g; g.obs_idx=(int)groups.size(); g.nearest=cand.nearest; g.ray_indices=cand.idx;
        g.hit_points.reserve(cand.idx.size());
        for(int ri:cand.idx) g.hit_points.push_back(rays[(size_t)ri].endpoint);
        fit_segments(g.hit_points,std::max(2,line_fit_min_points),line_fit_max_error,line_fit_min_length,ego,g.segments);
        if(g.segments.empty()){++failed;continue;}
        const Ray* sr=&rays[(size_t)cand.idx.front()];
        const Ray* er=&rays[(size_t)cand.idx.back()];
        g.vertices=build_vertices(g.segments,sr,er,ego,sensing_range,line_intersection_max_extension,line_fit_max_error);
        g.terminal_edges=terminal_edges(g.vertices,sr,er,ego,sensing_range);
        std::vector<Vec2> poly;
        const TerminalEdge *se=nullptr,*ee=nullptr;
        for(auto& te:g.terminal_edges){if(te.side<0)se=&te;else ee=&te;}
        if(se) poly.push_back(se->end);
        for(auto v:g.vertices) poly.push_back(v);
        if(ee) poly.push_back(ee->end);
        if(poly.size()<3) poly=g.hit_points;
        poly=dedupe(poly,1e-6);
        if(poly.size()<3){++failed;continue;}
        if(polygon_area(poly)<0) std::reverse(poly.begin(),poly.end());
        std::vector<std::vector<Vec2>> parts;
        if(field_flag==0) parts={poly};
        else if(!ear_clip(poly,parts)){++failed;continue;}
        int dyn=0; std::vector<double> vxs,vys;
        for(int ri:cand.idx){
            const auto& rr=rays[(size_t)ri];
            if(rr.hit_type==1){++dyn;vxs.push_back(rr.hit_vel.x);vys.push_back(rr.hit_vel.y);}
        }
        g.obs_type=(dyn*2>=(int)cand.idx.size() && dyn>0)?1:0;
        g.velocity=g.obs_type?Vec2{median(vxs),median(vys)}:Vec2{};
        int part_idx=0;
        for(auto part:parts){
            part=dedupe(part,1e-10);
            if(part.size()<3){++part_idx;continue;}
            if(polygon_area(part)<0) std::reverse(part.begin(),part.end());
            int gid=next_group_id++;
            for(size_t li=0;li<part.size();++li)
                selected.push_back({norm(part[li]-ego),part[li],g.obs_type,g.obs_idx,gid,(int)li,part_idx});
            ++part_idx;
        }
        groups.push_back(std::move(g));
    }

    int ns=0,nv=0,nte=0,nh=0,nsel=(int)selected.size(),nm=(int)groups.size();
    for(auto&g:groups){ns+=(int)g.segments.size();nv+=(int)g.vertices.size();nte+=(int)g.terminal_edges.size();nh+=(int)g.hit_points.size();}
    stash_hits(groups,rays,ego,sensing_range,ray_step_deg,cluster_gap);
    if(ns>cap_segments||nv>cap_vertices||nte>cap_terminal_edges||nh>cap_hit_points||nsel>cap_selected||nm>cap_meta){
        out_counts[0]=0;out_counts[1]=nm;out_counts[2]=ns;out_counts[3]=nv;out_counts[4]=nte;out_counts[5]=nh;out_counts[6]=nsel;out_counts[7]=nm;out_counts[8]=failed;out_counts[9]=next_group_id;return -2;
    }
    int si=0,vi=0,tei=0,hi=0;
    for(const auto&g:groups){
        for(const auto&s:g.segments){double*o=out_segments+11*si++;o[0]=g.obs_idx;o[1]=s.start.x;o[2]=s.start.y;o[3]=s.end.x;o[4]=s.end.y;o[5]=s.direction.x;o[6]=s.direction.y;o[7]=s.normal_out.x;o[8]=s.normal_out.y;o[9]=s.max_error;o[10]=s.rms_error;}
        for(auto v:g.vertices){double*o=out_vertices+3*vi++;o[0]=g.obs_idx;o[1]=v.x;o[2]=v.y;}
        for(const auto&e:g.terminal_edges){double*o=out_terminal_edges+10*tei++;o[0]=g.obs_idx;o[1]=e.side;o[2]=e.start.x;o[3]=e.start.y;o[4]=e.end.x;o[5]=e.end.y;o[6]=e.direction.x;o[7]=e.direction.y;o[8]=e.normal_out.x;o[9]=e.normal_out.y;}
        for(auto p:g.hit_points){double*o=out_hit_points+3*hi++;o[0]=g.obs_idx;o[1]=p.x;o[2]=p.y;}
        double* m=out_meta+4*g.obs_idx;m[0]=g.obs_idx;m[1]=g.obs_type;m[2]=g.velocity.x;m[3]=g.velocity.y;
    }
    for(int i=0;i<nsel;++i){const auto&s=selected[(size_t)i];double*o=out_selected+9*i;o[0]=s.dist;o[1]=s.xy.x;o[2]=s.xy.y;o[3]=s.obs_type;o[4]=s.obs_idx;o[5]=s.group_id;o[6]=s.local_idx;o[7]=0.0;o[8]=s.partition_idx;}
    out_counts[0]=0;out_counts[1]=(int)groups.size();out_counts[2]=ns;out_counts[3]=nv;out_counts[4]=nte;out_counts[5]=nh;out_counts[6]=nsel;out_counts[7]=nm;out_counts[8]=failed;out_counts[9]=next_group_id;
    return 0;
}


extern "C" int vertex_selector_finalize_selected(
    const double* selected, int n_selected,
    const double* motion_meta, int n_groups,
    double* out_packed, int cap_packed)
{
    // packed row width 8: [x, y, vx, vy, type, group_id, local_idx, obstacle_ref]
    if(n_selected < 0 || n_groups < 0 || cap_packed < n_selected) return -1;
    if(n_selected == 0) return 0;
    if(!selected || !out_packed) return -1;
    for(int i=0;i<n_selected;++i){
        const double* s = selected + 9*i;
        double* o = out_packed + 8*i;
        const int obs = (int)std::llround(s[4]);
        int type = (int)std::llround(s[3]);
        double vx = 0.0, vy = 0.0;
        if(motion_meta && obs >= 0 && obs < n_groups){
            const double* m = motion_meta + 7*obs;
            type = (int)std::llround(m[1]);
            if(type == 1 && std::isfinite(m[2]) && std::isfinite(m[3])){
                vx = m[2];
                vy = m[3];
            }
        }
        o[0]=s[1]; o[1]=s[2];
        o[2]=vx;   o[3]=vy;
        o[4]=(double)type;
        o[5]=s[5];
        o[6]=s[6];
        o[7]=s[4];
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Obstacle motion tracker: point-to-line registration + Kalman filter.
//
// Why not vertex/centroid differencing (the previous implementation): the end
// vertices of a fitted obstacle are wherever the scan starts/stops seeing it
// (occlusion and FOV boundaries).  They slide along the surface whenever the
// obstacle or the robot moves, so nearest-vertex and centroid differences mix
// fake tangential motion into the velocity, and a two-frame finite difference
// amplifies the range noise by 1/dt.
//
// What is used instead:
//  1. Association: the previous track's lines are predicted forward by v*dt
//     and the current group is matched by the median point-to-line distance.
//  2. Measurement: every current raw hit point is registered against the
//     previous frame's fitted lines (2-D translation ICP, point-to-line).
//     A point's residual is measured only along the line normal, so which
//     part of a face is visible does not matter.  All points of the obstacle
//     contribute, so range noise averages out.  Points beyond the ends of the
//     previous lines (newly visible surface) are not used.
//  3. Observability: one face only constrains its normal direction.  The
//     registration information matrix is eigen-decomposed and only well
//     constrained directions are used as measurements.
//  4. Filter: constant-velocity Kalman filter on the 2-D velocity.  The
//     unobserved direction keeps its prior (with growing uncertainty).
// ---------------------------------------------------------------------------
namespace {

struct Mat2 { double a{0}, b{0}, c{0}, d{0}; }; // [[a b][c d]] (kept symmetric)

struct MotionTrack {
    Vec2 center{};
    Vec2 velocity{};
    Mat2 P{};                       // velocity covariance
    double last_stamp{0.0};
    int updates{0};                 // frames with an accepted measurement
    int rejects{0};                 // consecutive gated-out frames
    std::deque<bool> moving_history;
    std::vector<Vec2> vertices;
    std::vector<Segment> segments;
    std::vector<Vec2> hits;         // raw hit points of the last observation
    ObsEnd ends[2];
};

struct TrackerState {
    bool have_stamp{false};
    double stamp{0.0};
    std::vector<MotionTrack> tracks;
};

TrackerState g_tracker;
double g_meas_sigma = 0.03;    // [m] LiDAR range / line-fit noise (1 sigma)
double g_accel_sigma = 2.0;    // [m/s^2] velocity random walk
double g_init_vel_sigma = 1.0; // [m/s] prior of a new track
constexpr int kEffectivePoints = 15; // cap: hit-point noise is not independent of line-fit noise
constexpr double kOcclusionMaxGap = 1.5; // [m] largest shadow an obstacle may re-emerge across

inline Segment shifted(const Segment& s, Vec2 off){
    Segment o=s; o.start=s.start+off; o.end=s.end+off; o.center=s.center+off; return o;
}

// Median distance of current points to the (predicted) previous polyline.
double polyline_cost(const std::vector<Vec2>& C, const std::vector<Segment>& P, Vec2 off){
    if(C.empty()||P.empty()) return std::numeric_limits<double>::infinity();
    std::vector<double> d; d.reserve(C.size());
    for(const auto&v:C){
        double best=std::numeric_limits<double>::infinity();
        for(const auto&s:P) best=std::min(best,distance_point_segment(v,s.start+off,s.end+off));
        d.push_back(best);
    }
    return median(d);
}

// Same, but from previous points (shifted by off) to the current polyline.
// The visible part of one obstacle changes between frames (ego motion,
// occlusion, a new face coming into view), so the current points are often
// only partly covered by the previous lines, and vice versa.  Association uses
// min(forward, reverse): it accepts "current is a subset of previous" and
// "previous is a subset of current", while two different obstacles fail both.
double polyline_cost_rev(const std::vector<Vec2>& Pp, const std::vector<Segment>& Cs, Vec2 off){
    if(Pp.empty()||Cs.empty()) return std::numeric_limits<double>::infinity();
    std::vector<double> d; d.reserve(Pp.size());
    for(const auto&v:Pp){
        double best=std::numeric_limits<double>::infinity();
        for(const auto&s:Cs) best=std::min(best,distance_point_segment(v+off,s.start,s.end));
        d.push_back(best);
    }
    return median(d);
}

// Symmetric 2x2 eigen-decomposition: A = l1 e1 e1^T + l2 e2 e2^T, l1>=l2.
void eig_sym2(double a,double b,double d,double& l1,double& l2,Vec2& e1,Vec2& e2){
    double tr=a+d, det=a*d-b*b;
    double disc=std::sqrt(std::max(0.0,0.25*tr*tr-det));
    l1=0.5*tr+disc; l2=0.5*tr-disc;
    if(std::abs(b)>1e-12) e1=normalized(Vec2{l1-d,b});
    else e1=(a>=d)?Vec2{1.0,0.0}:Vec2{0.0,1.0};
    e2={-e1.y,e1.x};
}

struct Registration {
    bool ok{false};
    Vec2 delta{};          // estimated translation previous -> current
    double l1{0}, l2{0};   // information eigenvalues (already / sigma^2)
    Vec2 e1{}, e2{};
    int n_used{0};
    double rms{0.0};
};

// Point-to-line ICP for a pure 2-D translation.
Registration register_points(const std::vector<Vec2>& pts, const std::vector<Segment>& prev,
                             Vec2 init, double gate){
    Registration R;
    if(pts.size()<3 || prev.empty()) return R;
    Vec2 off=init;
    const double gates[4]={gate, std::max(0.25*gate,4.0*g_meas_sigma),
                           std::max(0.12*gate,3.0*g_meas_sigma), std::max(0.12*gate,3.0*g_meas_sigma)};
    double A_a=0,A_b=0,A_d=0; int n_used=0; double sse=0;
    for(int it=0;it<4;++it){
        A_a=A_b=A_d=0; double b_x=0,b_y=0; n_used=0; sse=0;
        for(const auto&p:pts){
            double best=std::numeric_limits<double>::infinity(); double best_r=0; Vec2 best_n{};
            for(const auto&s0:prev){
                Vec2 a=s0.start+off, e=s0.end-s0.start; double L2=norm2(e);
                if(L2<=EPS) continue;
                double u=dot(p-a,e)/L2;
                // Only points that lie alongside the previous face; beyond its
                // ends the correspondence is unknown (newly visible surface).
                double L=std::sqrt(L2), margin=std::min(0.05,0.1*L)/L;
                if(u<-margin || u>1.0+margin) continue;
                Vec2 n{-e.y/L,e.x/L};
                double r=dot(p-a,n);
                if(std::abs(r)<best){best=std::abs(r);best_r=r;best_n=n;}
            }
            if(best>gates[it]) continue;
            A_a+=best_n.x*best_n.x; A_b+=best_n.x*best_n.y; A_d+=best_n.y*best_n.y;
            b_x+=best_n.x*best_r; b_y+=best_n.y*best_r;
            sse+=best_r*best_r; ++n_used;
        }
        if(n_used<3) return R;
        double l1,l2; Vec2 e1,e2; eig_sym2(A_a,A_b,A_d,l1,l2,e1,e2);
        // Solve only in well constrained directions (pseudo-inverse).
        Vec2 b{b_x,b_y}; Vec2 step{};
        if(l1>0.5) step=step+e1*(dot(e1,b)/l1);
        if(l2>0.08*n_used && l2>0.5) step=step+e2*(dot(e2,b)/l2);
        off=off+step;
        if(norm(step)<1e-4 && it>=1) break;
    }
    double l1,l2; Vec2 e1,e2; eig_sym2(A_a,A_b,A_d,l1,l2,e1,e2);
    double rms=std::sqrt(sse/std::max(1,n_used));
    double sig2=std::max(g_meas_sigma*g_meas_sigma, rms*rms);
    double scale=std::min(1.0,(double)kEffectivePoints/(double)n_used)/sig2;
    R.ok=true; R.delta=off; R.e1=e1; R.e2=e2; R.n_used=n_used; R.rms=rms;
    R.l1=l1*scale;
    R.l2=(l2>0.08*n_used)?l2*scale:0.0;   // direction not observable -> no info
    return R;
}

// Scalar Kalman update of the velocity component along unit direction h,
// decoupled from the orthogonal component.  A full 2-D update would let a
// slightly tilted line normal (a 1-2 deg fit error) leak normal-direction
// noise into the unobserved tangential direction through the large prior
// variance there; the decoupled form cannot do that.
void kf_axis_update(MotionTrack& t, Vec2 h, double z, double r){
    h=normalized(h); if(norm2(h)<=EPS || !(r>0.0)) return;
    Vec2 hp{-h.y,h.x};
    double Phh=t.P.a*h.x*h.x+2.0*t.P.b*h.x*h.y+t.P.d*h.y*h.y;
    double Ppp=t.P.a*hp.x*hp.x+2.0*t.P.b*hp.x*hp.y+t.P.d*hp.y*hp.y;
    double vh=dot(h,t.velocity);
    double K=Phh/(Phh+r);
    t.velocity=t.velocity+h*(K*(z-vh));
    Phh=(1.0-K)*Phh;
    // P = Phh h h^T + Ppp hp hp^T (cross-covariance dropped on purpose)
    t.P.a=Phh*h.x*h.x+Ppp*hp.x*hp.x;
    t.P.b=t.P.c=Phh*h.x*h.y+Ppp*hp.x*hp.y;
    t.P.d=Phh*h.y*h.y+Ppp*hp.y*hp.y;
}

double axis_nis(const MotionTrack& t, Vec2 h, double z, double r){
    h=normalized(h);
    double Phh=t.P.a*h.x*h.x+2.0*t.P.b*h.x*h.y+t.P.d*h.y*h.y;
    double innov=z-dot(h,t.velocity);
    return innov*innov/std::max(Phh+r,EPS);
}

} // namespace

extern "C" void vertex_selector_motion_reset(){
    g_tracker = TrackerState{};
}

// Optional tuning hook. Values <= 0 leave the current setting unchanged.
extern "C" void vertex_selector_motion_set_noise(double meas_sigma, double accel_sigma, double init_vel_sigma){
    if(meas_sigma>0.0 && std::isfinite(meas_sigma)) g_meas_sigma=meas_sigma;
    if(accel_sigma>0.0 && std::isfinite(accel_sigma)) g_accel_sigma=accel_sigma;
    if(init_vel_sigma>0.0 && std::isfinite(init_vel_sigma)) g_init_vel_sigma=init_vel_sigma;
}

extern "C" int vertex_selector_motion_update(
    const double* vertices, int n_vertices,
    const double* segments, int n_segments,
    int n_groups,
    double observation_stamp,
    double /*ego_x*/, double /*ego_y*/, double /*ego_yaw*/,
    double /*sensing_range*/,
    double /*line_extension*/,
    int enable_velocity,
    double match_gate,
    double /*min_displacement*/,
    double dynamic_speed_threshold,
    double velocity_max,
    double /*velocity_alpha: superseded by the Kalman gain*/,
    double track_max_dt,
    int emit_records,
    double* out_motion_meta, int cap_motion_meta,
    double* out_motion_records, int cap_motion_records,
    int* out_motion_counts, int out_motion_counts_len)
{
    // motion_meta row width 7:
    // [group_idx, obs_type, vx, vy, speed, valid, displacement]
    // motion_record row width 16, matching Python debug payload.
    if(!out_motion_counts || out_motion_counts_len<2 || n_groups<0 || n_vertices<0 || n_segments<0) return -1;
    out_motion_counts[0]=n_groups;
    out_motion_counts[1]=emit_records ? n_vertices : 0;
    if(n_groups>cap_motion_meta || (emit_records && n_vertices>cap_motion_records)) return -2;

    // ---- current observation ------------------------------------------------
    std::vector<MotionTrack> current((size_t)n_groups);
    if(vertices){
        for(int i=0;i<n_vertices;++i){
            const double* r=vertices+3*i;
            int gi=(int)std::llround(r[0]);
            if(gi>=0 && gi<n_groups && std::isfinite(r[1]) && std::isfinite(r[2]))
                current[(size_t)gi].vertices.push_back({r[1],r[2]});
        }
    }
    if(segments){
        for(int i=0;i<n_segments;++i){
            const double* r=segments+11*i;
            int gi=(int)std::llround(r[0]);
            if(gi<0 || gi>=n_groups) continue;
            Segment s;
            s.start={r[1],r[2]}; s.end={r[3],r[4]};
            s.direction=normalized(s.end-s.start);
            if(norm2(s.direction)<=EPS) continue;
            s.normal={-s.direction.y,s.direction.x};
            s.center=(s.start+s.end)*0.5;
            s.max_error=r[9]; s.rms_error=r[10];
            current[(size_t)gi].segments.push_back(s);
        }
    }
    // Raw hit points / ends of this fit (same group indexing); fall back to vertices.
    const bool have_obs=(int)g_last_obs.size()==n_groups;
    std::vector<std::vector<Vec2>> pts((size_t)n_groups);
    for(int gi=0;gi<n_groups;++gi){
        pts[(size_t)gi]=have_obs && !g_last_obs[(size_t)gi].hits.empty()
                        ? g_last_obs[(size_t)gi].hits : current[(size_t)gi].vertices;
        if(have_obs){
            current[(size_t)gi].ends[0]=g_last_obs[(size_t)gi].ends[0];
            current[(size_t)gi].ends[1]=g_last_obs[(size_t)gi].ends[1];
        }
        current[(size_t)gi].center=median_vec2(pts[(size_t)gi]);
    }
    g_last_obs.clear();

    const bool same_observation=g_tracker.have_stamp && std::isfinite(observation_stamp)
                                && std::abs(observation_stamp-g_tracker.stamp)<=1e-6;
    auto& prev=g_tracker.tracks;

    if(!same_observation){
        prev.erase(std::remove_if(prev.begin(),prev.end(),[&](const MotionTrack&t){
            double age=observation_stamp-t.last_stamp;
            return !std::isfinite(age) || age<=1e-6 || age>track_max_dt;
        }),prev.end());
    }

    // ---- association (greedy on predicted geometry) -------------------------
    std::vector<int> match((size_t)n_groups,-1);
    if(enable_velocity && !current.empty() && !prev.empty()){
        struct Cand{double fwd; double rev; int ci, pi;};
        std::vector<Cand> cand;
        for(int ci=0;ci<n_groups;++ci){
            const auto&C=current[(size_t)ci];
            if(pts[(size_t)ci].empty()) continue;
            for(int pi=0;pi<(int)prev.size();++pi){
                const auto&T=prev[(size_t)pi];
                double dt=same_observation?0.0:observation_stamp-T.last_stamp;
                Vec2 off=T.velocity*dt;
                double extent=0.0;
                for(const auto&s:T.segments) extent=std::max(extent,norm(s.end-s.start));
                for(const auto&s:C.segments) extent=std::max(extent,norm(s.end-s.start));
                if(norm(C.center-(T.center+off))>extent+2.0*match_gate+0.5) continue;
                double fwd=polyline_cost(pts[(size_t)ci],T.segments,off);
                double rev=polyline_cost_rev(T.hits.empty()?T.vertices:T.hits,C.segments,off);
                if(std::min(fwd,rev)<=match_gate) cand.push_back({fwd,rev,ci,pi});
            }
        }
        std::vector<char> uc((size_t)n_groups,0), up(prev.size(),0);
        // Pass 1: current points lie on the predicted previous lines.  This is
        // the primary test; when two obstacles merge into one group the larger
        // (dominant) one wins here, so the small one cannot steal the group.
        std::sort(cand.begin(),cand.end(),[](const Cand&a,const Cand&b){return a.fwd<b.fwd;});
        for(const auto&c:cand){
            if(c.fwd>match_gate || uc[(size_t)c.ci]||up[(size_t)c.pi]) continue;
            uc[(size_t)c.ci]=1; up[(size_t)c.pi]=1; match[(size_t)c.ci]=c.pi;
        }
        // Pass 2 (leftovers only): the previous observation lies on the current
        // lines, i.e. more of the same obstacle came into view (ego moved, an
        // occluder left, a new face appeared).  Without this the grown
        // observation fails pass 1 and is reborn as a new, "static" obstacle.
        std::sort(cand.begin(),cand.end(),[](const Cand&a,const Cand&b){return a.rev<b.rev;});
        for(const auto&c:cand){
            if(c.rev>match_gate || uc[(size_t)c.ci]||up[(size_t)c.pi]) continue;
            uc[(size_t)c.ci]=1; up[(size_t)c.pi]=1; match[(size_t)c.ci]=c.pi;
        }
        // Pass 3 (split): an occluder or a scan gap can cut one obstacle into
        // several groups.  A piece that lies tightly on the previous lines is
        // the same physical object and inherits that track (velocity, moving
        // history) instead of starting a fresh "static" one.
        std::sort(cand.begin(),cand.end(),[](const Cand&a,const Cand&b){return a.fwd<b.fwd;});
        for(const auto&c:cand){
            if(uc[(size_t)c.ci] || c.fwd>0.5*match_gate) continue;
            uc[(size_t)c.ci]=1; match[(size_t)c.ci]=c.pi;
        }
        // Pass 4 (occlusion continuation): when an occluder passes in front of
        // an obstacle, the part that re-emerges on the far side of the shadow
        // does not touch the previous lines at all, but it lies on the
        // extension of the previous face beyond an occlusion end (an end that
        // is not a real object corner).  Inherit that track; the piece is then
        // filtered with the track's velocity instead of restarting at 0.
        const double cos_par=std::cos(15.0*M_PI/180.0);
        for(int ci=0;ci<n_groups;++ci){
            if(uc[(size_t)ci]) continue;
            const auto&C=current[(size_t)ci];
            const auto&P=pts[(size_t)ci];
            if(P.size()<3 || C.segments.empty()) continue;
            int best_pi=-1; double best_gap=std::numeric_limits<double>::infinity();
            for(int pi=0;pi<(int)prev.size();++pi){
                const auto&T=prev[(size_t)pi];
                if(T.segments.empty()) continue;
                double dt=same_observation?0.0:observation_stamp-T.last_stamp;
                Vec2 off=T.velocity*dt;
                for(int k=0;k<2;++k){
                    const ObsEnd& e=T.ends[k];
                    if(e.real || norm2(e.tangent)<=EPS) continue;
                    Vec2 t=normalized(e.tangent), n{-t.y,t.x};
                    bool parallel=false;
                    for(const auto&s:C.segments)
                        if(std::abs(dot(s.direction,t))>=cos_par){ parallel=true; break; }
                    if(!parallel) continue;
                    Vec2 ep=e.p+off;
                    Vec2 out=dot(ep-(T.center+off),t)>=0.0?t:t*-1.0;
                    std::vector<double> perp; perp.reserve(P.size());
                    double gap=std::numeric_limits<double>::infinity();
                    for(const auto&p:P){
                        perp.push_back(std::abs(dot(p-ep,n)));
                        gap=std::min(gap,dot(p-ep,out));
                    }
                    if(median(perp)>0.5*match_gate) continue;
                    if(gap<-0.5*match_gate || gap>kOcclusionMaxGap) continue;
                    if(gap<best_gap){ best_gap=gap; best_pi=pi; }
                }
            }
            if(best_pi>=0){ uc[(size_t)ci]=1; match[(size_t)ci]=best_pi; }
        }
    }

    // ---- per-group filtering ------------------------------------------------
    int record_row=0;
    std::vector<char> prev_used(prev.size(),0);
    for(int ci=0;ci<n_groups;++ci){
        auto&cg=current[(size_t)ci];
        const int pi=match[(size_t)ci];
        double displacement=0.0;
        double dt_rec=0.0;

        if(pi>=0){
            const MotionTrack&T=prev[(size_t)pi];
            prev_used[(size_t)pi]=1;
            cg.velocity=T.velocity; cg.P=T.P; cg.updates=T.updates; cg.rejects=T.rejects;
            cg.moving_history=T.moving_history;
            cg.last_stamp=T.last_stamp;
            const double dt=observation_stamp-T.last_stamp;
            dt_rec=std::isfinite(dt)?dt:0.0;

            if(!same_observation && dt>1e-6 && dt<=track_max_dt){
                double q=g_accel_sigma*g_accel_sigma*dt*dt;
                cg.P.a+=q; cg.P.d+=q;
                Vec2 off=cg.velocity*dt;
                bool measured=false;

                Registration reg=register_points(pts[(size_t)ci],T.segments,off,match_gate);
                if(reg.ok && reg.l1>0.0){
                    // Measurements: e_i^T v = e_i^T delta / dt, var = 1/(l_i dt^2)
                    struct M{Vec2 h; double z, r;};
                    std::vector<M> ms;
                    ms.push_back({reg.e1,dot(reg.e1,reg.delta)/dt,1.0/(reg.l1*dt*dt)});
                    if(reg.l2>0.0) ms.push_back({reg.e2,dot(reg.e2,reg.delta)/dt,1.0/(reg.l2*dt*dt)});
                    bool plausible=norm(reg.delta)/dt<=velocity_max;
                    double d2=0.0; for(const auto&m:ms) d2+=axis_nis(cg,m.h,m.z,m.r);
                    bool gated=cg.updates>=3 && d2>(double)ms.size()*9.0;
                    if(plausible && gated && cg.rejects>=1){
                        // Two inconsistent frames in a row: the motion changed
                        // (start/stop/turn).  Re-open the filter instead of locking.
                        double p0=g_init_vel_sigma*g_init_vel_sigma;
                        cg.P.a=std::max(cg.P.a,p0); cg.P.d=std::max(cg.P.d,p0); cg.P.b=cg.P.c=0.0;
                        gated=false;
                    }
                    if(plausible && !gated){
                        for(const auto&m:ms) kf_axis_update(cg,m.h,m.z,m.r);
                        displacement=norm(reg.delta);
                        measured=true; cg.rejects=0;
                    } else if(plausible){
                        cg.rejects+=1;
                    }
                }
                // Tangential motion from real object ends (body-fixed corners).
                // Registration only sees motion normal to the faces; a real end
                // pins the motion along its face.  Occlusion/range-cut ends are
                // never used because they slide with the occluder/viewpoint.
                // Skipped for small objects: the visible ends of a round object
                // (pole, person) are silhouette points that slide with the
                // viewpoint, and the curved surface already makes both velocity
                // components observable through registration.
                double group_extent=0.0;
                for(const auto&p:pts[(size_t)ci]) group_extent=std::max(group_extent,norm(p-cg.center));
                if(reg.ok && group_extent>0.35){
                    bool used_prev_end[2]={false,false};
                    for(int k=0;k<2;++k){
                        const ObsEnd& ce=cg.ends[k];
                        if(!ce.real) continue;
                        int best=-1; double best_d=std::numeric_limits<double>::infinity();
                        for(int j=0;j<2;++j){
                            const ObsEnd& pe=T.ends[j];
                            if(!pe.real || used_prev_end[j]) continue;
                            if(std::abs(dot(normalized(ce.tangent),normalized(pe.tangent)))<std::cos(25.0*M_PI/180.0)) continue;
                            double d=norm(ce.p-(pe.p+reg.delta));
                            if(d<best_d){best_d=d;best=j;}
                        }
                        if(best<0) continue;
                        const ObsEnd& pe=T.ends[best];
                        double s2=ce.sigma*ce.sigma+pe.sigma*pe.sigma;
                        if(best_d>std::max(3.0*std::sqrt(s2),0.15)) continue;
                        used_prev_end[best]=true;
                        Vec2 t=normalized(ce.tangent);
                        double z=dot(ce.p-pe.p,t)/dt, r=s2/(dt*dt);
                        if(std::abs(z)>velocity_max) continue;
                        if(cg.updates>=3 && axis_nis(cg,t,z,r)>9.0) continue;
                        kf_axis_update(cg,t,z,r);
                        measured=true;
                    }
                }
                // Small round objects (people, poles) may produce no usable face.
                if(!measured && reg.n_used<3){
                    double ext=0.0;
                    for(const auto&p:pts[(size_t)ci]) ext=std::max(ext,norm(p-cg.center));
                    Vec2 dc=cg.center-T.center;
                    // The centroid of a piece bounded by an occlusion edge slides
                    // with the occluder's shadow, not with the object itself.
                    const bool occluded_end=(cg.ends[0].known && !cg.ends[0].real) ||
                                            (cg.ends[1].known && !cg.ends[1].real);
                    if(!occluded_end && ext<=0.5 && norm(dc)/dt<=velocity_max && norm(dc-off)<=match_gate){
                        double sc=std::max(g_meas_sigma,0.5*ext);
                        double r=2.0*sc*sc/(dt*dt);
                        kf_axis_update(cg,Vec2{1.0,0.0},dc.x/dt,r);
                        kf_axis_update(cg,Vec2{0.0,1.0},dc.y/dt,r);
                        displacement=norm(dc); measured=true;
                    }
                }
#ifdef VS_MOTION_DEBUG
                std::fprintf(stderr,"ci=%d pi=%d n=%zu used=%d l1=%.1f l2=%.1f e1=(%.2f,%.2f) d=(%.3f,%.3f) v=(%.2f,%.2f) meas=%d rej=%d\n",
                    ci,pi,pts[(size_t)ci].size(),reg.n_used,reg.l1,reg.l2,reg.e1.x,reg.e1.y,reg.delta.x,reg.delta.y,cg.velocity.x,cg.velocity.y,(int)measured,cg.rejects);
#endif
                if(measured) cg.updates+=1;
            }
        } else {
            cg.velocity={}; cg.updates=0; cg.rejects=0;
            double p0=g_init_vel_sigma*g_init_vel_sigma;
            cg.P=Mat2{p0,0.0,0.0,p0};
        }
        if(!same_observation) cg.last_stamp=observation_stamp;
        cg.hits=pts[(size_t)ci];

        double speed=norm(cg.velocity);
        if(!std::isfinite(speed)){ cg.velocity={}; speed=0.0; cg.P=Mat2{1,0,0,1}; }
        if(speed>velocity_max){ cg.velocity=cg.velocity*(velocity_max/speed); speed=velocity_max; }
        const bool velocity_valid=enable_velocity && cg.updates>=2;

        // Moving flag: 3 consecutive frames above the threshold to switch on;
        // once moving, stays on until the speed falls below 60% of it.
        if(!same_observation){
            bool was_moving=cg.moving_history.size()==3 &&
                std::all_of(cg.moving_history.begin(),cg.moving_history.end(),[](bool x){return x;});
            double thr=was_moving?0.6*dynamic_speed_threshold:dynamic_speed_threshold;
            cg.moving_history.push_back(velocity_valid && speed>=thr);
            if(cg.moving_history.size()>3) cg.moving_history.pop_front();
        }
        bool moving=cg.moving_history.size()==3 &&
            std::all_of(cg.moving_history.begin(),cg.moving_history.end(),[](bool x){return x;});

        if(out_motion_meta){
            double*m=out_motion_meta+7*ci;
            m[0]=ci;m[1]=moving?1:0;m[2]=cg.velocity.x;m[3]=cg.velocity.y;
            m[4]=speed;m[5]=velocity_valid?1.0:0.0;m[6]=displacement;
        }
        if(emit_records && out_motion_records){
            for(size_t vi=0;vi<cg.vertices.size();++vi){
                double*o=out_motion_records+16*record_row++;
                const auto&cur=cg.vertices[vi];
                Vec2 prv=cur-cg.velocity*dt_rec;
                o[0]=ci;o[1]=(double)vi;o[2]=cur.x;o[3]=cur.y;o[4]=prv.x;o[5]=prv.y;
                o[6]=cur.x-prv.x;o[7]=cur.y-prv.y;o[8]=displacement;o[9]=dt_rec;
                o[10]=cg.velocity.x;o[11]=cg.velocity.y;o[12]=speed;
                o[13]=(pi>=0)?1.0:0.0;o[14]=velocity_valid?1.0:0.0;o[15]=moving?1.0:0.0;
            }
        }
    }
    out_motion_counts[1]=emit_records?record_row:0;

    // Unmatched tracks coast for up to track_max_dt, so one dropped or merged
    // frame does not reset the velocity estimate.
    for(size_t pi=0;pi<prev.size();++pi)
        if(!prev_used[pi]) current.push_back(prev[pi]);
    g_tracker.tracks=std::move(current);
    if(!same_observation){ g_tracker.have_stamp=true; g_tracker.stamp=observation_stamp; }
    return 0;
}
