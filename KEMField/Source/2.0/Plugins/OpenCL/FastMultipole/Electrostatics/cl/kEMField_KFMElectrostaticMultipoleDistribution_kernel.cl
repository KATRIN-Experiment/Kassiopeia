#ifndef KFMElectrostaticMultipoleDistribution_Kernel_Defined_H
#define KFMElectrostaticMultipoleDistribution_Kernel_Defined_H

#include "kEMField_defines.h"

//these constants are defined at compile time
//KFM_DEGREE
//KFM_COMPLEX_STRIDE
//KFM_REAL_STRIDE

//NOTE: need to delete the folder ~/.nv/ComputeCache to force recompile on nvidia machines!!

__kernel void
DistributeElectrostaticMultipole( const unsigned int n_unique_nodes, //number of nodes to process
                                  __global unsigned int* node_index, //the indexes the the nodes to process
                                  __global unsigned int* start_index, //element start index for this node
                                  __global unsigned int* element_list_size, //number of elements associated with this node
                                  __global CL_TYPE2* element_moments, //moments of the current list of elements
                                  __global CL_TYPE2* node_moments) //moments of all the relevant nodes
{
    // Get the index of the current element to be processed
    unsigned int i_global = get_global_id(0);
    unsigned int n_workitems = n_unique_nodes*KFM_REAL_STRIDE;
    unsigned int inode = i_global/KFM_REAL_STRIDE;
    unsigned int term_index = i_global%KFM_REAL_STRIDE;

    if(i_global < n_workitems)
    {
        unsigned int node_id = node_index[inode];
        unsigned int start = start_index[inode];
        unsigned int size = element_list_size[inode];

        CL_TYPE2 sum = 0.0;
        for(unsigned int n=0; n<size; n++)
        {
            sum += element_moments[(start+n)*KFM_REAL_STRIDE + term_index];
        }

        node_moments[node_id*KFM_REAL_STRIDE + term_index] += sum;
    }

}


#endif
