#include "kEMField_defines.h"

//this function applies for non-scale invariant kernels also
__kernel void
ScalarMomentRemoteToLocalConverter(const unsigned int array_size,
                                   const unsigned int spatial_stride,
                                   const unsigned int n_global_terms,
                                   const unsigned int n_target_offset,
                                   __global CL_TYPE2* remote_moments,
                                   __global CL_TYPE2* response_functions,
                                   __global CL_TYPE2* local_moments)
{
    unsigned int i = get_global_id(0);

    if(i < array_size )
    {
        unsigned int tsi = i/spatial_stride + n_target_offset;
        unsigned int response_tsi = i/spatial_stride;
        unsigned int offset = i%spatial_stride;

        CL_TYPE2 target, source, response;
        target.s0 = 0.0;
        target.s1 = 0.0;

        for(unsigned int ssi=0; ssi<n_global_terms; ssi++)
        {
            source = remote_moments[ssi*spatial_stride + offset];
            response = response_functions[(ssi + response_tsi*n_global_terms)*spatial_stride + offset];
            target.s0 += (source.s0)*(response.s0) - (source.s1)*(response.s1);
            target.s1 += (source.s0)*(response.s1) + (source.s1)*(response.s0);
        }

        //normalize convolution
        target *= 1.0/( (CL_TYPE) spatial_stride);

        local_moments[tsi*spatial_stride + offset] = target;
    }
}
