// furthest_point_sampling_xkd.cl
// OpenCL kernel for furthest point sampling operation
// Matches C++ implementation (sequential per batch, min-distance to selected set)
#define MAX_ELEMS 2048
#define DEBUG_FLAG false

__kernel void ov_furthest_point_sampling(
    __global const float* pts,           // (B, N, 3) input points
    __global const INPUT1_TYPE* npoint_i,
    __global int* output                 // (B, npoint) output indices
) {
    // Parallelize over batches only
    uint batch_index = get_global_id(0);

    int B = INPUT0_DIMS[0];              // batch size
    int N = INPUT0_DIMS[1];              // number of points
    int npoint = INPUT1_DIMS[0];
    
    if (batch_index >= B) {
        return;
    }
    if (DEBUG_FLAG){
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
            printf("======== [GPU ov_furthest_point_sampling 2 input] ======== \n");
        } 
    }

    uint pts_offset = batch_index * N * 3;
    uint output_offset = batch_index * npoint;

    // Pointers to current batch data
    __global const float* current_dataset = pts + pts_offset;
    __global int* current_idxs = output + output_offset;

    // Local/private array to store minimum distance from each point to the selected set
    // Using private memory (__private/local array) - size must be known at compile time or use __local with dynamic allocation if supported and handled by host
    // For simplicity and assuming N is not too large, we use private array here.
    // If N is very large, __local memory with reduction techniques would be needed.
    float temp[MAX_ELEMS]; // This might cause issues if N is too large for private memory

    // Initialize temp array to maximum value
    for (int i = 0; i < N; ++i) {
        temp[i] = FLT_MAX;
    }

    // Initialize the first point
    current_idxs[0] = 0;

    // For each subsequent sample, pick the point with maximum of the minimum distance to the selected set
    // Note: Loop bound changed to use npoint
    for (int j = 1; j < npoint; ++j) {
        int last_selected_idx = current_idxs[j - 1];

        // Update minimum distances based on the last selected point
        // Vectorize this update if possible
        float last_x = current_dataset[last_selected_idx * 3 + 0];
        float last_y = current_dataset[last_selected_idx * 3 + 1];
        float last_z = current_dataset[last_selected_idx * 3 + 2];

        for (int k = 0; k < N; ++k) {
             // Optional: Skip very small magnitude points if needed, similar to C++ (commented out in C++)
             // float mag = current_dataset[k * 3 + 0] * current_dataset[k * 3 + 0] +
             //             current_dataset[k * 3 + 1] * current_dataset[k * 3 + 1] +
             //             current_dataset[k * 3 + 2] * current_dataset[k * 3 + 2];
             // if (mag <= 1e-3f) continue;

            float dx = current_dataset[k * 3 + 0] - last_x;
            float dy = current_dataset[k * 3 + 1] - last_y;
            float dz = current_dataset[k * 3 + 2] - last_z;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            // Update temp[k] to be the minimum distance to any selected point so far
            temp[k] = fmin(temp[k], dist_sq);
        }

        // Find the point with the maximum minimum distance
        int best_idx = 0;
        float best_val = -FLT_MAX; // Or 0.0f, since distances are non-negative

        for (int k = 0; k < N; ++k) {
            // The value in temp[k] now represents the minimum distance from point k to the set {current_idxs[0], ..., current_idxs[j-1]}
            if (temp[k] > best_val) {
                best_val = temp[k];
                best_idx = k;
            }
        }

        // Store next selected point
        current_idxs[j] = best_idx;
        // Optional: Set the distance of the selected point to 0 or a very small value
        // to prevent it from being selected again, though the logic should naturally avoid it
        // if distances are strictly positive and fmin is used correctly.
        // temp[best_idx] = -1.0f; // Or 0.0f
    }
}