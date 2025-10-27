#define DEBUG_FLAG false

__kernel void grouping_operation(
    __global const INPUT0_TYPE* points,    // (B, C, N)
    __global const INPUT1_TYPE* idx,         // (B, NPOINT, NSAMPLE)
    __global OUTPUT0_TYPE* out              // (B, C, NPOINT, NSAMPLE)
) {
    // 获取当前工作组（Work-group）的 ID，对应 CUDA 的 blockIdx.x
    int global_id = get_global_id(0);

    // 获取当前工作项（Work-item）在工作组内的 ID，对应 threadIdx.x
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    // 获取工作组大小，对应 blockDim.x
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);

    // 计算当前工作组处理的 total_threads（可选，用于循环处理更多数据）
    int total_threads = get_num_groups(0) * get_local_size(0);
    if (DEBUG_FLAG){
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
            printf("======== [GPU grouping_operation] ======== \n");
        } 
    }

    int B = INPUT0_DIMS[0];
    int C = INPUT0_DIMS[1];
    int N = INPUT0_DIMS[2];
    int NPOINT = INPUT1_DIMS[1];
    int NSAMPLE = INPUT1_DIMS[2];

    int total = B * C * NPOINT * NSAMPLE;

    if (global_id >= total) return;

    // printf("global_id: %d \n",global_id);

    int batch = global_id / (C * NPOINT * NSAMPLE);
    int channel = (global_id / (NSAMPLE * NPOINT)) % C;
    int point = (global_id / NSAMPLE) % NPOINT;
    int sample = global_id % NSAMPLE;

    
    int idx_offset = batch * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    int a = idx[idx_offset];

    float val = 0.0f;

    if (a >= 0 && a < N) {
        int input_offset = batch * (C * N) + channel * N + a;
        val = points[input_offset];

        if (isnan(val) || isinf(val)) {
            val = 0.0f;
        }
    }

    int out_offset = batch * (C * NPOINT * NSAMPLE) + channel * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    out[out_offset] = val;

    // if (global_id < 5) {
    //     printf("global_id=%d, B=%d, C=%d, N=%d, NPOINT=%d, NSAMPLE=%d\n",global_id, B, C, N, NPOINT, NSAMPLE);
    //     printf("b=%d, c=%d, point=%d, sample=%d, idx=%d, val=%f\n", batch, channel, point, sample, a, val);
    // }

}