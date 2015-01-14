#include "kEMField_defines.h"

__kernel void
ElectrostaticRemoteToRemoteReduceAndScale(const unsigned int n_moment_sets,
                                          const unsigned int term_stride,
                                          const unsigned int tree_level,
                                          const unsigned int parent_node_id,
                                          __constant const CL_TYPE* scale_factor_array,
                                          __global unsigned int* block_set_ids,
                                          __global CL_TYPE2* node_moments,
                                          __global CL_TYPE2* transformed_child_moments)
{
    unsigned int i_global = get_global_id(0);
    if(i_global < term_stride)
    {
        unsigned int term_index = i_global;

        CL_TYPE2 sum = 0.0;
        for(unsigned int i=0; i<n_moment_sets; i++)
        {
            unsigned int block_id = block_set_ids[i];
            sum += transformed_child_moments[block_id*term_stride + term_index];
        }

        //now we appropriately scale the sum
        sum *= scale_factor_array[tree_level*term_stride + term_index];

        //now add it to the parent node's moments
        node_moments[parent_node_id*term_stride + term_index] += sum;
    }
}
