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


__kernel void assign(__global const int *n_spikes, __global const float *prelog, __global float *log_p_best,__global float *log_p_second_best, __global int *clusters, __global int *clusters_second_best, __global const int *ng_clust, __global const int *ooec, __global float *debug1)
{
	int s = get_global_id(0);
	int sortbest;
	int secondsortbest;
	float best;
	float secondbest;
	float temp;
	int ns = *n_spikes;

	if(*ng_clust<=1){ 
        sortbest = 0;
    }else{    
        sortbest = 0;
        secondsortbest = 1;
        best = prelog[s];
        secondbest = prelog[ns+s];
        if(secondbest > best){
        	sortbest = 1;
        	secondsortbest = 0;
        	temp = best;
        	best = secondbest;
        	secondbest = temp;
        }
        for(int i = 2; i<*ng_clust; i++){
        	temp = prelog[i*ns + s];
            if(temp>secondbest){
                if(temp>best){
                    secondsortbest = sortbest;  
                    sortbest = i;
                    secondbest = best;
                    best = temp;
                }else{
                    secondsortbest = i;
                    secondbest = temp;
                }
            }
        }
    }
    debug1[s] = best;
    debug1[ns + s] = secondbest;
    debug1[2*ns + s] = sortbest;
    debug1[3*ns + s] = secondsortbest;


    float cur_log_p_best = log_p_best[s];
    if(ooec){
        float cur_log_p_second_best = log_p_second_best[s];

        if(cur_log_p_best > best){
            if(cur_log_p_second_best > best){
                log_p_second_best[s] = cur_log_p_second_best;
            }else{    
                log_p_second_best[s] = best;
            }  
        }else{    
            log_p_best[s] = best;
  
            if((*ng_clust>1) & (isfinite(secondbest))){
                log_p_second_best[s] = secondbest;
            }else{ 
                log_p_second_best[s] = best;
            }
            clusters[s] = sortbest;
            if(*ng_clust>1){
                clusters_second_best[s] = secondsortbest;
	        }
        }
    }else{
        log_p_best[s] = best;
    }
}