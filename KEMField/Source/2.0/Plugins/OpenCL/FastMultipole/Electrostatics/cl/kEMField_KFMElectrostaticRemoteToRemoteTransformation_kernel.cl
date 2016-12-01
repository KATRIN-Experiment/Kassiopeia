#include "kEMField_opencl_defines.h"

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
}

int
OrderReverseLookUp(int storage_index, int degree)
{
    return storage_index - ((degree+1)*degree)/2;
}


__kernel void
ElectrostaticRemoteToRemoteTransform(const unsigned int n_moment_sets,
                                     const unsigned int degree,
                                     const unsigned int spatial_stride,
                                     __global unsigned int* block_set_ids,
                                     __global CL_TYPE2* response_functions,
                                     __global CL_TYPE2* original_moments,
                                     __global CL_TYPE2* transformed_moments)
{
    unsigned int i_global = get_global_id(0);
    int real_term_stride = (degree+1)*(degree+2)/2;
    int complex_term_stride = (degree+1)*(degree+1);

    if( i_global < n_moment_sets*real_term_stride)
    {
        //reverse look up of target indices
        int j = DegreeReverseLookUp(i_global/n_moment_sets);
        int k = OrderReverseLookUp(i_global/n_moment_sets, j);
        int block_id = block_set_ids[i_global%n_moment_sets];

        CL_TYPE2 temp, source, response;

        //TODO this memory access pattern maybe slow, look for improvements
        int rssi, rtsi;
        int ssi, tsi;
        int response_index;

        rtsi = j*(j+1)/2 + k;
        tsi = j*(j+1) + k;
        temp = 0.0;

        for(int n=0; n <= degree; n++)
        {
            for(int m=0; m <= n; m++)
            {
                rssi = (n*(n+1))/2 + m;
                ssi = n*(n+1) + m;

                //add contribution from source(n,m) with appropriate normalized response
                source = original_moments[block_id*real_term_stride + rssi];

                response_index = (ssi + tsi*complex_term_stride)*spatial_stride + block_id;
                response = response_functions[response_index];
                temp.s0 += ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
                temp.s1 += ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );

                if(m != 0)
                {
                    source.s1 *= -1.0; //take complex conj
                    ssi = n*(n+1) - m;
                    response_index = (ssi + tsi*complex_term_stride)*spatial_stride + block_id;
                    response = response_functions[response_index];
                    temp.s0 += ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
                    temp.s1 += ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );
                }
            }
        }

        transformed_moments[block_id*real_term_stride + rtsi] = temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
