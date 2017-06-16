#include "kEMField_opencl_defines.h"
#include "kEMField_KFMComplexMultiply.cl"

//this function applies for non-scale invariant kernels also
__kernel void
BatchedRemoteToLocalTransformation(const unsigned int n_parent_nodes,
                                   const __global CL_TYPE2* restrict remote_moments,
                                   const __global CL_TYPE2* response_functions,
                                   __global CL_TYPE2* restrict local_moments,
                                   const __constant CL_TYPE2* a_coefficient,
                                   const __global unsigned int* restrict reversed_index)
{
    __private CL_TYPE2 s[KFM_COMPLEX_STRIDE]; //source moment workspace
    __private CL_TYPE2 t[KFM_DEGREE+1]; //target moment workspace
    const __global CL_TYPE2* r; //response function workspace

    unsigned int i_global = get_global_id(0);

    unsigned int node_index = i_global/KFM_SPATIAL_STRIDE;
    unsigned int block_index = i_global%KFM_SPATIAL_STRIDE;
    unsigned int reversed_block_index = reversed_index[block_index];
    unsigned int node_offset = node_index*KFM_REAL_STRIDE*KFM_SPATIAL_STRIDE;

////////////////////////////////////////////////////////////////////////////////
//load necessary data

    if(i_global < n_parent_nodes*KFM_SPATIAL_STRIDE )
    {
        //prefetch the response functions
        r = &(response_functions[block_index*KFM_RESPONSE_STRIDE]);
        prefetch(r, KFM_RESPONSE_STRIDE);

        //load the source moments
        CL_TYPE2 conj; conj.s0 = 1.0; conj.s1 = -1.0;
        CL_TYPE sn_fac = 1.0;
        CL_TYPE2 pre_fac, temp;
        for(int n=0; n<=KFM_DEGREE; n++)
        {
            int rssi0 = (n*(n+1))/2;
            int cssi0 = n*(n+1);
            temp = remote_moments[node_offset + rssi0*KFM_SPATIAL_STRIDE + block_index];
            s[cssi0] = ComplexMultiply(sn_fac*a_coefficient[rssi0], temp);

            for(int m=1; m<=n; m++)
            {
                int rssi = rssi0 + m;
                pre_fac = sn_fac*a_coefficient[rssi]; //normalize by a_coeff, and (-1)^n
                temp = remote_moments[node_offset + rssi*KFM_SPATIAL_STRIDE + block_index];
                s[cssi0 + m] = ComplexMultiply(pre_fac, temp);
                temp = conj*remote_moments[node_offset + rssi*KFM_SPATIAL_STRIDE + reversed_block_index];
                s[cssi0 - m] = ComplexMultiply(pre_fac, temp);
            }

            sn_fac *= -1.0;
        }

        //zero out the target moments
        for(int i=0; i<KFM_DEGREE; i++)
        {
            t[i] = 0.0;
        }

    }


////////////////////////////////////////////////////////////////////////////////
//now execute linear transformation

    CL_TYPE2 sm;
    for(int j=0; j<=KFM_DEGREE; j++)
    {
        //zero out workspace
        for(int i=0; i<=j; i++){t[i] = 0.0;};

        int rtsi0 = j*(j+1)/2;
        for(int n=0; n<=KFM_DEGREE; n++)
        {
            int cssi0 = n*(n+1);
            int rsi0 = (j + n)*(j + n + 1);

            for(int m=-n; m<=n; m++)
            {
                sm = s[cssi0+m];
                for(int k=0; k<=j; k++)
                {
                    //multiply source by response add to target moment
                    t[k] += ComplexMultiply(sm, r[rsi0 + m - k] );
                }
            }
        }

        //write out local coefficients for degree j
        if(i_global < n_parent_nodes*KFM_SPATIAL_STRIDE)
        {
            for(int k=0; k<=j; k++)
            {
                local_moments[node_offset + (rtsi0+k)*KFM_SPATIAL_STRIDE + block_index] =  ComplexMultiply(a_coefficient[rtsi0+k], t[k]);
            }
        }

    }


}
