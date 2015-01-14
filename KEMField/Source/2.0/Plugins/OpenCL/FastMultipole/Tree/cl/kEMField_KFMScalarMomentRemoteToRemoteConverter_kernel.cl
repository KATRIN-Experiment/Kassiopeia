#include "kEMField_defines.h"

__kernel void
ScalarMomentRemoteToRemoteConverter(const unsigned int array_size,
                                    const unsigned int spatial_stride,
                                    const unsigned int n_terms,
                                    __constant CL_TYPE* scale_factor,
                                    __global CL_TYPE2* child_moments,
                                    __global CL_TYPE2* response_functions,
                                    __global CL_TYPE2* child_moment_contrib)
{
    unsigned int i = get_global_id(0);

    if(i < array_size )
    {
        unsigned int tsi = i/spatial_stride;
        unsigned int offset = i%spatial_stride;

        //CL_TYPE scale = scale_factor[tsi];
        CL_TYPE2 target, source, response;
        target = 0.0;

        for(unsigned int ssi=0; ssi<n_terms; ssi++)
        {
            source = scale_factor[ssi]*child_moments[ssi*spatial_stride + offset];
            response = response_functions[(ssi + tsi*n_terms)*spatial_stride + offset];
            target.s0 += (source.s0)*(response.s0) - (source.s1)*(response.s1);
            target.s1 += (source.s0)*(response.s1) + (source.s1)*(response.s0);
        }

        child_moment_contrib[tsi*spatial_stride + offset] = target;
    }
}                                                            
