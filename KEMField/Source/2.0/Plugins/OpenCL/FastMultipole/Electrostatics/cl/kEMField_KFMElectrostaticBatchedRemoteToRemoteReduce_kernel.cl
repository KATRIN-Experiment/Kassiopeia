#include "kEMField_defines.h"

//compiler defines
//KFM_KFM_DEGREE
//KFM_REAL_STRIDE
//KFM_COMPLEX_STRIDE
//KFM_SPATIAL_STRIDE


__kernel void
ElectrostaticBatchedRemoteToRemoteReduce(const unsigned int n_parent_nodes,
                                         __global const unsigned int* parent_node_offset,
                                         __global const unsigned int* n_child_nodes,
                                         __global const unsigned int* parent_node_ids,
                                         __global CL_TYPE2* node_moments,
                                         __global CL_TYPE2* transformed_moments)
{
    //total number of work items is
    unsigned int n_workitems = n_parent_nodes*KFM_REAL_STRIDE;
    unsigned int i_global = get_global_id(0);

    CL_TYPE2 sum = 0.0;

    if(i_global < n_workitems)
    {
        //determine node index and moment index from global id
        unsigned int node_index = i_global/KFM_REAL_STRIDE;
        unsigned int term_index = i_global%KFM_REAL_STRIDE;

        unsigned int offset = parent_node_offset[node_index];
        unsigned int n_children = n_child_nodes[node_index];
        unsigned int parent_id = parent_node_ids[node_index];

        for(unsigned int i=0; i<n_children; i++)
        {
            sum += transformed_moments[(offset + i)*KFM_REAL_STRIDE + term_index];
        }

        //now add it to the parent node's moments
        node_moments[parent_id*KFM_REAL_STRIDE + term_index] += sum;
    }

}
