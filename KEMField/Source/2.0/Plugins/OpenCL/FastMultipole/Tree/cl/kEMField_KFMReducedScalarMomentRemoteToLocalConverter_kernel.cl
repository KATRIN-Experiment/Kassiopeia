#include "kEMField_opencl_defines.h"


int
ComplexDegreeReverseLookUp(int storage_index)
{
    int degree = 0;
    while(storage_index > 0)
    {
        storage_index -= 2*degree+1;
        if(storage_index < 0){return degree;}
        ++degree;
    }
    return degree;
}

int ComplexOrderReverseLookUp(int degree, int storage_index)
{
    return storage_index - degree*(degree+1);
}

int
ReducedStorageComplexDegreeReverseLookUp(int storage_index)
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

int ReducedStorageComplexOrderReverseLookUp(int degree, int storage_index)
{
    return storage_index - degree*(degree+1)/2;
}

//this function applies for non-scale invariant kernels also
__kernel void
ReducedScalarMomentRemoteToLocalConverter(const unsigned int total_array_size, //total number of threads
                                          const unsigned int degree, //expansion degree
                                          const unsigned int spatial_stride,
                                          __global CL_TYPE2* remote_moments,
                                          __global CL_TYPE2* response_functions,
                                          __global CL_TYPE2* local_moments,
                                          __global CL_TYPE2* normalization,
                                          __global unsigned int* reversed_index)
{
    int i_global = get_global_id(0);

    if( i_global < total_array_size)
    {
        unsigned int n_terms = (degree+1)*(degree+1);

        unsigned int tsi = i_global/spatial_stride;
        unsigned int spatial_offset = i_global%spatial_stride;

        int j = ReducedStorageComplexDegreeReverseLookUp(tsi);
        int k = ReducedStorageComplexOrderReverseLookUp(j, tsi);
        unsigned int ctsi = j*(j+1) + k;

        CL_TYPE2 temp, target, source, response, norm;
        target = 0.0;

        //TODO this memory access pattern maybe slow, look for improvements
        int rssi0, rssi, cssi0, cssi;
        int rsi0, rsi;
        for(int n=0; n <= degree; n++)
        {
            rssi0 = (n*(n+1))/2;
            cssi0 = n*(n+1);
            rsi0 = (j + n)*(j + n + 1);
            for(int m=0; m <= n; m++)
            {
                rssi = rssi0 + m;
                cssi = cssi0 + m;
                rsi = rsi0 + (m-k);

                //add contribution from source(n,m) with appropriate normalized response
                source = remote_moments[rssi*spatial_stride + spatial_offset];
                norm = normalization[cssi + ctsi*n_terms];
                response = response_functions[rsi*spatial_stride + spatial_offset];
                temp.s0 = ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
                temp.s1 = ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );

                //normalization is a real number, but we store it as complex
                //in case we want to expand it to handle this in future implementations
                target += (norm.s0)*temp;

                if(m > 0)
                {
                    //add contribution from source(n, -m) with appropriate normalized response
                    //use the fact that F((N-n)%N) = conj(F(n)) to get the multipole moment for (n,-m)
                    source = remote_moments[rssi*spatial_stride + reversed_index[spatial_offset]];
                    source.s1 *= -1.0; //take complex conjugate of source moment

                    //update indices used for obtaining response/normalization
                    cssi = cssi0 - m;
                    rsi = rsi0 + (-m-k);
                    norm = normalization[cssi + ctsi*n_terms];
                    response = response_functions[rsi*spatial_stride + spatial_offset];
                    temp.s0 = ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
                    temp.s1 = ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );
                    target += (norm.s0)*temp;
                }
            }
        }

        local_moments[tsi*spatial_stride + spatial_offset] = target;
    }
}
