// ball_query.cl
// OpenCL kernel for ball query operation
// Based on the C++ implementation in ball_query.cpp
#define MAX_N 2048
#define DEBUG_FLAG false
__kernel void ov_ball_query(
    __global const INPUT0_TYPE* new_xyz,   // (B, npoint, 3)
    __global const INPUT1_TYPE* xyz,       // (B, N, 3)
    __global OUTPUT0_TYPE* idx)               // (B, npoint, nsample)
    {
    uint batch_index = get_global_id(0);      // batch index
    uint point_index = get_global_id(1);      // point index (npoint)
    uint sample_index = get_global_id(2);      // sample index (nsample)
    
    if (DEBUG_FLAG){
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
            printf("======== [GPU ov_ball_query] ======== \n");
            printf("batch_index: %d, point_index: %d, sample_index:%d \n",batch_index, point_index, sample_index);
        } 
    }
    
    int b = INPUT0_DIMS[0];
    int n = INPUT1_DIMS[1];
    int npoint = INPUT0_DIMS[1];

    float radius2 = radius * radius;
    int cnt = 0;

    uint inp_1_offset = batch_index * npoint * 3;
    uint inp_2_offset = batch_index * n * 3;
    uint output_offset = batch_index * npoint * nsample;

    INPUT0_TYPE new_x = new_xyz[inp_1_offset + point_index * 3 + 0];
    INPUT0_TYPE new_y = new_xyz[inp_1_offset + point_index * 3 + 1];
    INPUT0_TYPE new_z = new_xyz[inp_1_offset + point_index * 3 + 2];

    __local INPUT0_TYPE local_xyz[1024 * 3]; 
    int local_size = get_local_size(1);
    int local_id = get_local_id(1);

    for (int l = local_id; l < n; l += local_size) {
        local_xyz[l * 3 + 0] = xyz[inp_2_offset + l * 3 + 0];
        local_xyz[l * 3 + 1] = xyz[inp_2_offset + l * 3 + 1];
        local_xyz[l * 3 + 2] = xyz[inp_2_offset + l * 3 + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k=0; k < n && cnt < nsample; ++k){
        INPUT1_TYPE x = local_xyz[k * 3 + 0];
        INPUT1_TYPE y = local_xyz[k * 3 + 1];
        INPUT1_TYPE z = local_xyz[k * 3 + 2];

        float dist = (new_x - x) * (new_x - x) +
                    (new_y - y) * (new_y - y) +
                    (new_z - z) * (new_z - z);


        if (dist < radius2) {
            if (cnt == 0) {
                for (int l = 0; l < nsample; ++l) {
                    idx[output_offset + point_index * nsample + l] = k;
                    }
                }
            idx[output_offset+ point_index * nsample + cnt] = k;
            ++cnt;
        }
    }
        
    
} 