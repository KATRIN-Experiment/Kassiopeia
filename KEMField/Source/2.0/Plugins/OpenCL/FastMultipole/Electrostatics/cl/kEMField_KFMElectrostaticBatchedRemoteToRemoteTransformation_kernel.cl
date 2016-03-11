#include "kEMField_defines.h"
#include "kEMField_KFMComplexMultiply.cl"


//compiler defines
//KFM_KFM_DEGREE
//KFM_REAL_STRIDE
//KFM_COMPLEX_STRIDE
//KFM_KFM_SPATIAL_STRIDE

int
DegreeReverseLookUp(int storage_index)
{
    int degree = 0;
    while(storage_index > 0)
    {
        storage_index -= degree+1;
        if(storage_index < 0){return degree;}
        ++degree;
    }
    return degree;
}

int
OrderReverseLookUp(int storage_index, int degree)
{
    return storage_index - ((degree+1)*degree)/2;
}


__kernel void
ElectrostaticBatchedRemoteToRemoteTransform(const unsigned int n_nodes,
                                            const unsigned int tree_level,
                                            __constant const CL_TYPE* source_scale_factor_array,
                                            __constant const CL_TYPE* target_scale_factor_array,
                                            __global CL_TYPE2* response_functions,
                                            __global CL_TYPE2* node_moments,
                                            __global CL_TYPE2* transformed_moments,
                                            __global unsigned int* node_ids,
                                            __global unsigned int* block_set_ids)
{
    //total number of work items is
    unsigned int n_workitems = n_nodes*KFM_REAL_STRIDE;
    unsigned int i_global = get_global_id(0);

    //private workspace to store the child node's moments
    __private CL_TYPE2 child_moments[KFM_REAL_STRIDE];
    CL_TYPE2 conj; conj.s0 = 1.0; conj.s1 = -1.0;

    //determine node index and moment index from global id
    unsigned int node_index = i_global/KFM_REAL_STRIDE;
    unsigned int term_index = i_global%KFM_REAL_STRIDE;

    //loop indices to the relative node data
    unsigned int node_id = node_ids[node_index];
    unsigned int block_id = block_set_ids[node_index];

    if(i_global < n_workitems)
    {
        //copy and scale the child node's moments into private workspace
        for(unsigned int index = 0; index<KFM_REAL_STRIDE; index++)
        {
            CL_TYPE2 s = source_scale_factor_array[tree_level*KFM_REAL_STRIDE + index];
            CL_TYPE2 m = node_moments[node_id*KFM_REAL_STRIDE + index];
            child_moments[index] = s*m;
        }

        //reverse look up of target indices
        int j = DegreeReverseLookUp(term_index);
        int k = OrderReverseLookUp(term_index, j);

        CL_TYPE2 temp, source, response, scale;

        //TODO this memory access pattern maybe slow, look for improvements
        int rssi;
        int ssi, tsi;
        int response_index;

        tsi = j*(j+1) + k;
        temp = 0.0;

        for(int n=0; n <= KFM_DEGREE; n++)
        {
            //this is the m=0 term
            rssi = (n*(n+1))/2;
            ssi = n*(n+1);
            source = child_moments[rssi];

            response_index = (ssi + tsi*KFM_COMPLEX_STRIDE)*KFM_SPATIAL_STRIDE + block_id;
            response = response_functions[response_index];
            temp += ComplexMultiply(response, source);

            for(int m=1; m <= n; m++)
            {
                rssi = (n*(n+1))/2 + m;
                ssi = n*(n+1) + m;

                source = child_moments[rssi];
                response_index = (ssi + tsi*KFM_COMPLEX_STRIDE)*KFM_SPATIAL_STRIDE + block_id;
                response = response_functions[response_index];
                temp += ComplexMultiply(response, source);

                //add contribution from negative m
                source *= conj;
                ssi = n*(n+1) - m;
                response_index = (ssi + tsi*KFM_COMPLEX_STRIDE)*KFM_SPATIAL_STRIDE + block_id;
                response = response_functions[response_index];
                temp += ComplexMultiply(response, source);
            }
        }

        scale = target_scale_factor_array[tree_level*KFM_REAL_STRIDE + term_index];
        transformed_moments[node_index*KFM_REAL_STRIDE + term_index] = temp*scale;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
