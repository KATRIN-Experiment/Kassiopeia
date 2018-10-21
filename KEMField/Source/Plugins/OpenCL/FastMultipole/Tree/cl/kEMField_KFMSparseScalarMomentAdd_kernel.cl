#include "kEMField_opencl_defines.h"

__kernel void
SparseScalarMomentAdd(const unsigned int n_moment_sets, //total moment sets we need to add
                      const unsigned int spatial_stride, //size of spatial block
                      const unsigned int n_terms, //number of terms in series
                      __global unsigned int* block_set_local_ids,
                      __global unsigned int* primary_node_ids,
                      __global CL_TYPE2* block_set_moments, //the moments of the current block set
                      __global CL_TYPE2* primary_moments) //the moments of the entire set of primary nodes
{
    unsigned int i = get_global_id(0); //sums the i'th moment
    if(i < n_moment_sets)
    {
        unsigned int block_id = block_set_local_ids[i];
        unsigned int primary_id = primary_node_ids[i];

        for(unsigned int n=0; n<n_terms; n++)
        {
            CL_TYPE2 block_moment = block_set_moments[n*spatial_stride + block_id];
            CL_TYPE2 primary_moment = primary_moments[primary_id*n_terms + n];
            primary_moment += block_moment;
            primary_moments[primary_id*n_terms + n] = primary_moment;
        }
    }
}                                                            
