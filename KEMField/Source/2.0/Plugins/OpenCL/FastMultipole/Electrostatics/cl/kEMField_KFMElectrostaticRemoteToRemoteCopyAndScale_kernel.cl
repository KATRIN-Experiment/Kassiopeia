#include "kEMField_defines.h"

__kernel void
ElectrostaticRemoteToRemoteCopyAndScale(const unsigned int n_moment_sets,
                                        const unsigned int term_stride,
                                        const unsigned int tree_level,
                                        __constant const CL_TYPE* scale_factor_array,
                                        __global unsigned int* node_ids,
                                        __global unsigned int* block_set_ids,
                                        __global CL_TYPE2* node_moments,
                                        __global CL_TYPE2* block_moments)
{
    unsigned int i = get_global_id(0);
    if(i < n_moment_sets)
    {
        unsigned int block_id = block_set_ids[i];
        unsigned int node_id = node_ids[i];

        for(unsigned int n=0; n<term_stride; n++)
        {
            CL_TYPE scale_factor = scale_factor_array[tree_level*term_stride + n];
            block_moments[block_id*term_stride + n] = scale_factor*node_moments[node_id*term_stride + n];
        }
    }
}
