#include "kEMField_opencl_defines.h"

__kernel void
ElectrostaticRemoteToLocalCopyAndScale(const unsigned int n_moment_sets,
                                       const unsigned int term_stride,
                                       const unsigned int spatial_stride,
                                       const unsigned int tree_level,
                                       const unsigned int parent_node_start_index,
                                       __constant const CL_TYPE* scale_factor_array,
                                       __global unsigned int* node_ids,
                                       __global unsigned int* block_set_ids,
                                       __global CL_TYPE2* node_moments,
                                       __global CL_TYPE2* block_moments)
{
    unsigned int i_global = get_global_id(0);
    if(i_global < n_moment_sets*term_stride)
    {
        unsigned int block_id = block_set_ids[parent_node_start_index + i_global/term_stride];
        unsigned int node_id = node_ids[parent_node_start_index + i_global/term_stride];
        unsigned int term_index = i_global%term_stride;

        CL_TYPE scale_factor = scale_factor_array[tree_level*term_stride + term_index];
        block_moments[term_index*spatial_stride + block_id] =  scale_factor*node_moments[node_id*term_stride + term_index];
    }
}
