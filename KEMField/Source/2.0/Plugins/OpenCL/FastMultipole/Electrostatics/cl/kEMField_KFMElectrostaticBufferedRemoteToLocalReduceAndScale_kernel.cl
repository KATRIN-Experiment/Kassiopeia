#include "kEMField_defines.h"

//defined at compile time
//KFM_DEGREE
//KFM_REAL_STRIDE
//KFM_SPATIAL_STRIDE

__kernel void
ElectrostaticBufferedRemoteToLocalReduceAndScale(const unsigned int n_moment_sets,
                                                 const unsigned int tree_level,
                                                 const unsigned int parent_node_start_index,
                                                 const unsigned int parent_offset,
                                                 __constant const CL_TYPE* scale_factor_array,
                                                 __global unsigned int* node_ids,
                                                 __global unsigned int* block_set_ids,
                                                 __global CL_TYPE2* node_moments,
                                                 __global CL_TYPE2* transformed_child_moments)
{
    unsigned int i_global = get_global_id(0);
    unsigned int item_index = 0;
    unsigned int term_index = 0;
    unsigned int block_id = 0;
    unsigned int node_id = 0;
    unsigned int n_workitems = n_moment_sets*KFM_REAL_STRIDE;
    CL_TYPE2 contrib = 0.0;

    if(i_global < n_workitems)
    {
        item_index = i_global/KFM_REAL_STRIDE;
        term_index = i_global%KFM_REAL_STRIDE;
        block_id = block_set_ids[parent_node_start_index + item_index];
        node_id = node_ids[parent_node_start_index + item_index];

        contrib = transformed_child_moments[parent_offset + term_index*KFM_SPATIAL_STRIDE + block_id];
        contrib *= scale_factor_array[tree_level*KFM_REAL_STRIDE + term_index];

        node_moments[node_id*KFM_REAL_STRIDE + term_index] += contrib;
    }
}
