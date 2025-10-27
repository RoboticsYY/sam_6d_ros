// ov_debug_node.cl
// Pass-through + printf first K elements in linear order.
// Works for FP32/FP16/INT by casting to float for printing.

#define MAX_PRINT_ELEMS 200
#define DEBUG_FLAG false
__kernel void custom_debug_node(
    __global const INPUT0_TYPE* in,   // Any shape
    __global OUTPUT0_TYPE* out        // Same shape as input
) {
    // Total elements = product of BFYX
    const int B = INPUT0_DIMS[0];
    const int F = INPUT0_DIMS[1];
    const int Y = INPUT0_DIMS[2];
    const int X = INPUT0_DIMS[3];
    const int total = B * F * Y * X;

    if (DEBUG_FLAG){
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
            printf("======== [GPU custom_debug_node] ======== \n");
        } 
    }

    // 1D global range over linearized elements
    const int gid = get_global_id(0);
    if (gid < total) {
        // Pass-through
        out[gid] = in[gid];
    }

    // Single work-item prints shape + first K elements in linear order
    if (gid == 0) {
        printf("[OV DebugNode] shape=[%d,%d,%d,%d], total=%d\n", B, F, Y, X, total);
        const int k = (total < MAX_PRINT_ELEMS) ? total : MAX_PRINT_ELEMS;

        // Print first k values in the same linear order as memory layout
        for (int i = 0; i < k; ++i) {
            // Cast to float for unified printing (int/bool will show as float)
            float v = (float)in[i];
            printf("%0.2f ", v);
        }
        if (total > MAX_PRINT_ELEMS) {
            printf("\n... (truncated, total %d elements)\n", total);
        } else {
            printf("\n");
        }
    }
}