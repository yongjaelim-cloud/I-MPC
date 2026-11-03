Header header

int32 obs_cnt

float32[200] obs_xs
float32[200] obs_ys
float32[200] obs_zs

float32[200] obs_vxs
float32[200] obs_vys
float32[200] obs_vzs

float32[200] obs_rs

# 0: static, 1: dynamic
int32[200] obs_types

# polygon vertex count for each obstacle
int32[200] obs_vertex_counts

# flattened polygon vertices
# index = obstacle_index * 10 + vertex_index
float32[2000] obs_poly_xs
float32[2000] obs_poly_ys
float32[2000] obs_poly_zs
