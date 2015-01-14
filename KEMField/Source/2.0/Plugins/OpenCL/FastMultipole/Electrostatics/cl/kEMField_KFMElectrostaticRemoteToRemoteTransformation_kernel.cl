#include "kEMField_defines.h"

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

}

int ReducedStorageComplexOrderReverseLookUp(int degree, int storage_index)
{
    return storage_index - degree*(degree+1)/2;
}

////this function applies for non-scale invariant kernels also
//__kernel void
//ElectrostaticRemoteToRemoteTransform(const unsigned int n_moment_sets,
//                                     const unsigned int degree,
//                                     const unsigned int spatial_stride,
//                                     __global unsigned int* block_set_ids,
//                                     __global CL_TYPE2* response_functions,
//                                     __global CL_TYPE2* original_moments,
//                                     __global CL_TYPE2* transformed_moments)
//{
//    unsigned int real_term_stride = (degree+1)*(degree+2)/2;
//    unsigned int total_array_size = real_term_stride*n_moment_sets;
//    unsigned int i_global = get_global_id(0);

//    if( i_global < total_array_size)
//    {
//        unsigned int complex_term_stride = (degree+1)*(degree+1);

//        unsigned int tsi = i_global/n_moment_sets;
//        unsigned int block_index = i_global%n_moment_sets;
//        unsigned int block_id = block_set_ids[block_index];

//        int j = ReducedStorageComplexDegreeReverseLookUp(tsi);
//        int k = ReducedStorageComplexOrderReverseLookUp(j, tsi);
//        unsigned int ctsi = j*(j+1) + k;

//        CL_TYPE2 temp, target, source, response, norm;
//        target = 0.0;

//        //TODO this memory access pattern maybe slow, look for improvements
//        int rssi0, rssi, cssi0, cssi;
//        int rsi0, rsi;
//        int n_prime, m_prime;
//        for(int n=0; n <= (degree - j) ; n++)
//        {
//            rssi0 = (n*(n+1))/2;
//            cssi0 = n*(n+1);
//            n_prime = j - n;
//            rsi0 = (n_prime)*(n_prime + 1);
//            for(int m=0; m <= n; m++)
//            {
//                rssi = rssi0 + m;
//                cssi = cssi0 + m;
//                m_prime = k - m;

//                if(abs(m_prime) <= abs(n_prime) )
//                {
//                    rsi = rsi0 + m_prime;

//                    //add contribution from source(n,m) with appropriate normalized response
//                    source = original_moments[block_id*real_term_stride + rssi];
//                    response = response_functions[rsi*spatial_stride + block_id];
//                    temp.s0 = ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
//                    temp.s1 = ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );

//                    //normalization is a real number, but we store it as complex
//                    //in case we want to expand it to handle this in future implementations
//                    target += temp;

//                    if(m > 0)
//                    {
//                        source.s1 *= -1.0; //take complex conjugate of source moment

//                        //update indices used for obtaining response/normalization
//                        cssi = cssi0 - m;
//                        m_prime = k + m;
//                        rsi = rsi0 + m_prime;
//                        response = response_functions[rsi*spatial_stride + block_id];
//                        temp.s0 = ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
//                        temp.s1 = ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );
//                        target += temp;
//                    }
//                }
//            }
//        }

//        transformed_moments[block_id*real_term_stride + j*(j+1)] = target;
//    }
//}


//this function applies for non-scale invariant kernels also
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

    if( i_global < n_moment_sets)
    {
        unsigned int real_term_stride = (degree+1)*(degree+2)/2;
        unsigned int block_id = block_set_ids[i_global];

        CL_TYPE2 temp, source, response;

        //TODO this memory access pattern maybe slow, look for improvements
        int rssi0, rssi, rtsi;
        int rsi0, rsi;
        int n_prime, m_prime;

        temp = 0.0;

        for(int j=0; j <= degree; j++)
        {
            for(int k=0; k<=j; k++)
            {
                rtsi = j*(j+1)/2 + k;

                for(int n=0; n <=j; n++)
                {
                    n_prime = j-n;
                    rssi0 = (n*(n+1))/2;
                    rsi0 = (n_prime)*(n_prime + 1);

                    for(int m=-n; m <= n; m++)
                    {
                        m_prime = k - m;
                        if(abs(m_prime) <= abs(n_prime) )
                        {
                            rssi = rssi0 + abs(m);
                            rsi = rsi0 + m_prime;

                            //add contribution from source(n,m) with appropriate normalized response
                            source = original_moments[block_id*real_term_stride + rssi];

                            if(m < 0)
                            {
                                source.s1 *= -1.0; //take complex conj
                            }

                            response = response_functions[rsi*spatial_stride + block_id];
                            temp.s0 += ( (source.s0)*(response.s0) - (source.s1)*(response.s1) );
                            temp.s1 += ( (source.s0)*(response.s1) + (source.s1)*(response.s0) );
                        }
                    }
                }

                transformed_moments[block_id*real_term_stride + rtsi] += temp;

            }
        }
    }
}
