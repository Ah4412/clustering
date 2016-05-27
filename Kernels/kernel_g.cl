#ifdef cl_khr_fp64
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
	#error "Double precision floating point not supported by opencl imple"
#endif

__kernel void mstep(__global const int *KKruns, __global const int *maxDK, __global const int *sparseKKs, __global const int *sparseLCs, __global const int *start, __global const int *end, __global const int *spikes, __global const int *s_offset, __global float *bern)
{

	int  g_clust = get_global_id(0);
	int  KKrun   = get_global_id(1);
	int  l_clust = get_global_id(2);

	int  off1 = s_offset[g_clust];
	int  off2 = s_offset[g_clust+1];
	int  offset = g_clust*(*KKruns)*(*maxDK)+KKrun*(*maxDK)+ l_clust;

	int x=0;
	int s;

	for(int i=off1; i<off2; i++){
			s = spikes[i];
			for(int j=start[s]; j<end[s]; j++){
				if( (KKrun == sparseKKs[j]) & (l_clust == sparseLCs[j])){
					x += 1.0;
				}
			}	
	}
	bern[offset] = (float)x;
}

__kernel void prelog(__global const int *KKruns, __global const int *maxDK,__global const int *n_spikes, __global const int *sparseKKs, __global const int *sparseLCs, __global const int *start, __global const int *end, __global const float *bern, __global float *prelog, __global const float *filler, __global float *debug1)
{

	int g_clust = get_global_id(0);
	int s = get_global_id(1);

	int offset = g_clust*(*KKruns)*(*maxDK);
	float x = filler[g_clust];

	debug1[s + g_clust*(*n_spikes)] = bern[7];

	int id0;
	int id1;
	float y;

	for(int i=start[s]; i<end[s]; i++){
		id0 = sparseKKs[i];
        id1 = sparseLCs[i];
        x += bern[offset + id0*(*maxDK) + id1];
        y = bern[offset + id0*(*maxDK)];
        if ( isfinite(y) ){
        	x -= y;
        }
	}
	prelog[s + g_clust*(*n_spikes)] = x;
}


__kernel void assign(__global const int *KKruns, __global const int *maxDK,__global const int *n_spikes, __global const int *sparseKKs, __global const int *sparseLCs, __global const int *start, __global const int *end, __global const float *bern, __global float *prelog, __global const float *filler, __global float *debug1)
{

}