#ifndef KFMRaggedElementLookup_Defined_H
#define KFMRaggedElementLookup_Defined_H

#include  "kEMField_opencl_defines.h"

//returns the element index in the global element list
//and the corresponding node index in the local node list

uint2
RaggedElementLookup(unsigned int n_nodes,
                    unsigned int ragged_list_index,
                    __global unsigned int* node_list_start_index,
                    __global unsigned int* node_list_size )
{

    uint sub_total = 0;
    uint mod_index;
    uint2 ret_val;

    for(uint i=0; i<n_nodes; i++)
    {
        if(ragged_list_index < sub_total + node_list_size[i])
        {
            mod_index = ragged_list_index - sub_total;
            ret_val.s0 = node_list_start_index[i] + mod_index; //the element index
            ret_val.s1 = i; //the node index
            return ret_val;
        }
        else
        {
            sub_total += node_list_size[i];
        }
    }
}


#endif
