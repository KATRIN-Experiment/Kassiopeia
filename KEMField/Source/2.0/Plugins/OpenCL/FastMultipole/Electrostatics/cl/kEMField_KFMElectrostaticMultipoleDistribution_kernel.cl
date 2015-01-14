#ifndef KFMElectrostaticMultipoleDistribution_Kernel_Defined_H
#define KFMElectrostaticMultipoleDistribution_Kernel_Defined_H

#include "kEMField_defines.h"

//these constants are defined at compile time
//KFM_DEGREE
//KFM_COMPLEX_STRIDE
//KFM_REAL_STRIDE

//NOTE: need to delete the folder ~/.nv/ComputeCache to force recompile on nvidia machines!!

//__kernel void
//DistributeElectrostaticMultipole( const unsigned int n_elements, //number of primitives to process
//                                  __global unsigned int* node_index,
//                                  __global CL_TYPE2* element_moments,
//                                  __global CL_TYPE2* node_moments)
//{

//    // Get the index of the current element to be processed
//    unsigned int i_global = get_global_id(0);

//    if(i_global < n_elements)
//    {
//        unsigned int node_id = node_index[i_global];

//        //add the element moments to the node moments
//        for(unsigned int i=0; i<KFM_REAL_STRIDE; i++)
//        {
//            //this may lead to a RACE condition!
//            node_moments[node_id*KFM_REAL_STRIDE + i] = element_moments[i_global*KFM_REAL_STRIDE + i];
//        }

//    }

//}


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

    if(i_global < n_unique_nodes)
    {
        unsigned int node_id = node_index[i_global];
        unsigned int start = start_index[i_global];
        unsigned int size = element_list_size[i_global];

        for(unsigned int n=start; n<start+size; n++)
        {
            //add the element moments to the node moments
            for(unsigned int i=0; i<KFM_REAL_STRIDE; i++)
            {
                node_moments[node_id*KFM_REAL_STRIDE + i] += element_moments[n*KFM_REAL_STRIDE + i];
            }
        }
    }
}


#endif
