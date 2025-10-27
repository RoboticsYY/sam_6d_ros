#define DEBUG_FLAG false

__kernel void gather_operation(
	__global const float* features,  // (B, C, N)
	__global const int* idx,         // (B, NPOINT)
	__global float* output           // (B, C, NPOINT)
) {
	int gid = get_global_id(0);

	int B = INPUT0_DIMS[0];
	int C = INPUT0_DIMS[1];
	int N = INPUT0_DIMS[2];
	int NPOINT = INPUT1_DIMS[1];

	int total = B * C * NPOINT;
	if (gid >= total) return;
	if (DEBUG_FLAG){
		if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
			printf("======== [GPU gather_operation] ======== \n");
		} 
	}
	int j = gid % NPOINT;                 // point index within npoints
	int l = (gid / NPOINT) % C;           // channel index
	int i = gid / (C * NPOINT);           // batch index

	int a = idx[i * NPOINT + j];          // gather index from input idx

	float out_val = 0.0f;
	if (a >= 0 && a < N) {
		float in_val = features[i * (C * N) + l * N + a];
		// Match C++ behavior: sanitize NaN/Inf to 0.0
		if (isnan(in_val) || isinf(in_val)) {
			out_val = 0.0f;
		} else {
			out_val = in_val;
		}
	}

	output[i * (C * NPOINT) + l * NPOINT + j] = out_val;
}
