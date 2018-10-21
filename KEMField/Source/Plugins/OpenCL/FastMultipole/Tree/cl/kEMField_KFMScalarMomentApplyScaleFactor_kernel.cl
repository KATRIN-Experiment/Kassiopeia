#include "kEMField_opencl_defines.h"

__kernel void
ScalarMomentApplyScaleFactor(const unsigned int array_size,
                             const unsigned int spatial_stride,
                             const unsigned int n_terms,
                             const unsigned int tree_level,
                             __constant const CL_TYPE* scale_factor_array,
                             __global CL_TYPE2* moments)
{
    unsigned int i = get_global_id(0);
    if(i < array_size)
    {
        unsigned int storage_index = i/spatial_stride;
        CL_TYPE scale_factor = scale_factor_array[tree_level*n_terms + storage_index];
        moments[i] *= scale_factor;
    }
}                                                            
