#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

bool ear_clip(std::vector<Vec2> p, std::vector<std::vector<Vec2>>& parts){
    parts.clear(); p=dedupe(p,1e-10); if(p.size()<3 || std::abs(polygon_area(p))<1e-10) return false;
    if(polygon_area(p)<0) std::reverse(p.begin(),p.end());
    if(is_convex(p)){parts.push_back(p);return true;}
    std::vector<int> active(p.size()); std::iota(active.begin(),active.end(),0);
    const int max_iter=10*(int)p.size()*(int)p.size(); int it=0;
    while(active.size()>3 && it++<max_iter){
        bool found=false;
        for(size_t pos=0;pos<active.size();++pos){
            int ip=active[(pos+active.size()-1)%active.size()], ic=active[pos], in=active[(pos+1)%active.size()];
            Vec2 a=p[ip],b=p[ic],c=p[in]; if(cross(b-a,c-b)<=1e-12) continue;
            bool contains=false;
            for(int idx:active){ if(idx==ip||idx==ic||idx==in) continue; if(point_in_tri(p[idx],a,b,c)){contains=true;break;} }
            if(contains) continue;
            parts.push_back({a,b,c}); active.erase(active.begin()+pos); found=true; break;
        }
        if(!found) return false;
    }
    if(active.size()==3) parts.push_back({p[active[0]],p[active[1]],p[active[2]]});
    return !parts.empty();
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



struct MotionPrevGroup {
    int obs_idx{0};
    Vec2 center{};
    Vec2 velocity{};
    int streak{0};
    std::vector<Vec2> vertices;
    std::vector<Segment> segments;
};

struct MotionState {
    bool have_stamp{false};
    double stamp{0.0};
    bool have_ego{false};
    double ego_x{0.0}, ego_y{0.0}, ego_yaw{0.0};
    std::vector<MotionPrevGroup> groups;
};

MotionState g_motion_state;

inline double wrap_pi_cpp(double a){ return std::atan2(std::sin(a), std::cos(a)); }

Vec2 median_vec2(const std::vector<Vec2>& v){
    if(v.empty()) return {};
    std::vector<double> xs; xs.reserve(v.size());
    std::vector<double> ys; ys.reserve(v.size());
    for(const auto&p:v){ xs.push_back(p.x); ys.push_back(p.y); }
    return {median(xs), median(ys)};
}

Vec2 robust_center_cpp(const std::vector<Vec2>& v){ return median_vec2(v); }

bool stationary_terminal_reference_cpp(
    Vec2 current_xy,
    const std::vector<Segment>& previous_segments,
    Vec2 origin,
    double sensing_range,
    double line_extension,
    double match_gate,
    Vec2& best_point)
{
    Vec2 ray=current_xy-origin;
    double rn=norm(ray);
    if(rn<=1e-10) return false;
    Vec2 ray_dir=ray/rn;
    double best_error=std::numeric_limits<double>::infinity();
    double max_extension=std::max(line_extension,match_gate);
    bool found=false;
    for(const auto&seg: previous_segments){
        Vec2 d=normalized(seg.direction);
        if(norm2(d)<=EPS) continue;
        Vec2 q;
        if(!line_intersection(origin,ray_dir,seg.start,d,q)) continue;
        double ray_t=dot(q-origin,ray_dir);
        if(ray_t < -1e-8 || ray_t > sensing_range+max_extension) continue;
        double seg_len=norm(seg.end-seg.start);
        double along=dot(q-seg.start,d);
        if(along < -max_extension || along > seg_len+max_extension) continue;
        double err=norm(current_xy-q);
        if(err<best_error){best_error=err;best_point=q;found=true;}
    }
    return found && best_error<=match_gate;
}

struct MotionCandidate { double cost{0.0}, min_dist{0.0}; int ci{-1}, pi{-1}; };
struct VertexPair { double dist{0.0}; int ci{-1}, pi{-1}; };

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
        split_fit_rec(g.hit_points,0,std::max(2,line_fit_min_points),line_fit_max_error,line_fit_min_length,ego,g.segments); if(g.segments.empty()){++failed;continue;}
        const Ray* sr=&rays[cand.idx.front()]; const Ray* er=&rays[cand.idx.back()]; g.vertices=build_vertices(g.segments,sr,er,ego,sensing_range,line_intersection_max_extension,line_fit_max_error); g.terminal_edges=terminal_edges(g.vertices,sr,er,ego,sensing_range);
        std::vector<Vec2> poly; const TerminalEdge *se=nullptr,*ee=nullptr; for(auto& te:g.terminal_edges){if(te.side<0)se=&te;else ee=&te;} if(se)poly.push_back(se->end);for(auto v:g.vertices)poly.push_back(v);if(ee)poly.push_back(ee->end);if(poly.size()<3)poly=g.hit_points;poly=dedupe(poly,1e-6);if(poly.size()<3){++failed;continue;}
        if(polygon_area(poly)<0)std::reverse(poly.begin(),poly.end()); std::vector<std::vector<Vec2>> parts; if(field_flag==0)parts={poly}; else if(!ear_clip(poly,parts)){++failed;continue;}
        int dyn=0;std::vector<double> vxs,vys;for(int ri:cand.idx){if(rays[ri].hit_type==1){++dyn;vxs.push_back(rays[ri].hit_vel.x);vys.push_back(rays[ri].hit_vel.y);}}g.obs_type=(dyn*2>=(int)cand.idx.size() && dyn>0)?1:0;g.velocity=g.obs_type?Vec2{median(vxs),median(vys)}:Vec2{};
        int part_idx=0;for(auto part:parts){part=dedupe(part,1e-10);if(part.size()<3){++part_idx;continue;}if(polygon_area(part)<0)std::reverse(part.begin(),part.end());int gid=next_group_id++;for(size_t li=0;li<part.size();++li)selected.push_back({norm(part[li]-ego),part[li],g.obs_type,g.obs_idx,gid,(int)li,part_idx});++part_idx;}
        groups.push_back(std::move(g));
    }

    int nr=emit_rays?(int)rays.size():0, ns=0,nv=0,nte=0,nh=0,nsel=(int)selected.size(),nm=(int)groups.size();
    for(auto&g:groups){ns+=(int)g.segments.size();nv+=(int)g.vertices.size();nte+=(int)g.terminal_edges.size();nh+=(int)g.hit_points.size();}
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

    // Raw ray-hit clusters only: no TLS fit, segment construction, or ear clipping.
    std::vector<Group> groups; groups.reserve(std::min((int)cands.size(),max_obs));
    std::vector<SelectedRow> selected;
    int next_group_id=0, failed=0;
    for(auto& cand:cands){
        if((int)groups.size()>=max_obs) break;
        Group g; g.obs_idx=(int)groups.size(); g.nearest=cand.nearest;
        g.ray_indices=cand.idx;
        for(int ri:cand.idx) g.hit_points.push_back(rays[(size_t)ri].endpoint);
        if(g.hit_points.size()<2){++failed;continue;}
        // The motion tracker receives raw sampled hits in the "vertices" buffer.
        // Downsample to cap matching cost, independently of polygon output.
        const size_t K=std::min(size_t(64),g.hit_points.size());
        for(size_t j=0;j<K;++j){
            size_t at=(K==1)?0:j*(g.hit_points.size()-1)/(K-1);
            g.vertices.push_back(g.hit_points[at]);
        }
        // Three dummy vertices reserve one group in the legacy selected ABI.
        // Python replaces these BEFORE publishing with the enclosing circle,
        // and drops groups that have not yet been classified as moving.
        Vec2 c=robust_center_cpp(g.hit_points);
        const double e=0.01;
        std::vector<Vec2> placeholder={{c.x-e,c.y-e},{c.x+e,c.y-e},{c.x,c.y+e}};
        int gid=next_group_id++;
        for(size_t j=0;j<placeholder.size();++j)
            selected.push_back({norm(placeholder[j]-ego),placeholder[j],0,g.obs_idx,gid,(int)j,0});
        groups.push_back(std::move(g));
    }

    int ns=0,nv=0,nte=0,nh=0,nsel=(int)selected.size(),nm=(int)groups.size();
    for(auto&g:groups){ns+=(int)g.segments.size();nv+=(int)g.vertices.size();nte+=(int)g.terminal_edges.size();nh+=(int)g.hit_points.size();}
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


extern "C" void vertex_selector_motion_reset(){
    g_motion_state = MotionState{};
}

extern "C" int vertex_selector_motion_update(
    const double* vertices, int n_vertices,
    const double* segments, int n_segments,
    int n_groups,
    double observation_stamp,
    double ego_x, double ego_y, double ego_yaw,
    double sensing_range,
    double line_extension,
    int enable_velocity,
    double match_gate,
    double min_displacement,
    double dynamic_speed_threshold,
    double velocity_max,
    double velocity_alpha,
    double track_max_dt,
    int emit_records,
    double* out_motion_meta, int cap_motion_meta,
    double* out_motion_records, int cap_motion_records,
    int* out_motion_counts, int out_motion_counts_len)
{
    (void)segments;(void)n_segments;(void)ego_x;(void)ego_y;(void)ego_yaw;
    (void)sensing_range;(void)line_extension;
    if(!out_motion_counts || out_motion_counts_len<2 || n_groups<0 || n_vertices<0) return -1;
    out_motion_counts[0]=n_groups;
    out_motion_counts[1]=0;
    if(n_groups>cap_motion_meta || (emit_records && n_vertices>cap_motion_records)) return -2;
    std::vector<MotionPrevGroup> current((size_t)n_groups);
    for(int gi=0;gi<n_groups;++gi) current[(size_t)gi].obs_idx=gi;
    for(int i=0;vertices && i<n_vertices;++i){
        const double* v=vertices+3*i;
        int gi=(int)std::llround(v[0]);
        if(gi>=0 && gi<n_groups && std::isfinite(v[1]) && std::isfinite(v[2]))
            current[(size_t)gi].vertices.push_back({v[1],v[2]});
    }
    for(auto&g:current) g.center=robust_center_cpp(g.vertices);
    const bool had_stamp=g_motion_state.have_stamp;
    const double dt=had_stamp?observation_stamp-g_motion_state.stamp:0.0;
    const bool valid_dt=std::isfinite(dt)&&dt>1e-6&&dt<=track_max_dt;
    const bool duplicate=had_stamp && std::abs(dt)<=1e-6;
    const auto& prev=g_motion_state.groups;
    std::vector<int> match((size_t)n_groups,-1);
    if(enable_velocity && (valid_dt||duplicate)){
        struct Link{int c,p;double cost;};
        std::vector<Link> links;
        // Conservative nearest-center gating, in the fixed odom frame.
        const double gate=std::max(0.15,2.0*match_gate);
        for(int ci=0;ci<n_groups;++ci) for(int pi=0;pi<(int)prev.size();++pi){
            double d=norm(current[(size_t)ci].center-prev[(size_t)pi].center);
            if(d<=gate) links.push_back({ci,pi,d});
        }
        std::sort(links.begin(),links.end(),[](const Link&a,const Link&b){return a.cost<b.cost;});
        std::vector<char> usedc((size_t)n_groups,0),usedp(prev.size(),0);
        for(auto l:links) if(!usedc[(size_t)l.c]&&!usedp[(size_t)l.p]){
            usedc[(size_t)l.c]=1;usedp[(size_t)l.p]=1;match[(size_t)l.c]=l.p;
        }
    }
    int record_row=0;
    for(int ci=0;ci<n_groups;++ci){
        auto& g=current[(size_t)ci];
        Vec2 velocity{};double displacement=0.0;
        bool measured=false;
        int pi=match[(size_t)ci];
        if(pi>=0){
            const auto& p=prev[(size_t)pi];
            g.streak=p.streak;
            g.velocity=p.velocity;
            if(duplicate){velocity=p.velocity; measured=true;}
            else if(valid_dt&&!g.vertices.empty()&&!p.vertices.empty()){
                // Correspondence-free nearest-hit residuals: static surfaces
                // retain near-zero residual even when ego rotates in odom.
                std::vector<Vec2> deltas;
                std::vector<double> residuals;
                for(const auto& q:g.vertices){
                    double best=std::numeric_limits<double>::infinity();Vec2 closest{};
                    for(const auto& old:p.vertices){
                        double d=norm2(q-old);
                        if(d<best){best=d;closest=old;}
                    }
                    if(std::sqrt(best)<=match_gate){
                        deltas.push_back(q-closest);residuals.push_back(std::sqrt(best));
                    }
                }
                if(deltas.size()>=std::max(size_t(2),g.vertices.size()/3)){
                    Vec2 measured_delta=median_vec2(deltas);
                    displacement=norm(measured_delta);
                    Vec2 measured_v=measured_delta/dt;
                    if(norm(measured_v)<=velocity_max){
                        velocity=measured_v*velocity_alpha+p.velocity*(1.0-velocity_alpha);
                        measured=true;
                    }
                    // Require coherent directional displacement, not just noisy hit shifts.
                    size_t coherent=0;
                    for(auto d:deltas) if(dot(d,measured_delta)>0) ++coherent;
                    bool motion_evidence=displacement>=min_displacement &&
                        norm(measured_v)>=dynamic_speed_threshold &&
                        coherent*10>=deltas.size()*7;
                    g.streak=motion_evidence?std::min(p.streak+1,100):0;
                } else g.streak=0;
            }
        }
        if(!duplicate && (!valid_dt || pi<0)) g.streak=0;
        if(!measured) velocity={};
        g.velocity=velocity;
        // Five DISTINCT consecutive LiDAR observations with motion evidence.
        const bool moving=enable_velocity && measured && g.streak>=5 &&
            norm(velocity)>=dynamic_speed_threshold;
        if(out_motion_meta){
            double* m=out_motion_meta+7*ci;
            m[0]=ci;m[1]=moving?1.0:0.0;
            m[2]=moving?velocity.x:0.0;m[3]=moving?velocity.y:0.0;
            m[4]=norm(velocity);m[5]=measured?1.0:0.0;m[6]=displacement;
        }
        if(emit_records&&out_motion_records){
            for(size_t vi=0;vi<g.vertices.size();++vi){
                double*o=out_motion_records+16*record_row++;
                for(int j=0;j<16;++j)o[j]=0.0;
                o[0]=ci;o[1]=vi;o[2]=g.vertices[vi].x;o[3]=g.vertices[vi].y;
                o[4]=NAN;o[5]=NAN;o[8]=displacement;o[9]=dt;
                o[10]=velocity.x;o[11]=velocity.y;o[12]=norm(velocity);
                o[13]=(pi>=0)?1.0:0.0;o[14]=measured?1.0:0.0;o[15]=moving?1.0:0.0;
            }
        }
    }
    out_motion_counts[1]=emit_records?record_row:0;
    if(!duplicate){
        g_motion_state.groups=std::move(current);
        g_motion_state.have_stamp=true;g_motion_state.stamp=observation_stamp;
    }
    return 0;
}
