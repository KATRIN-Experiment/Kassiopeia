#include "kEMField_opencl_defines.h"

__kernel void
ZeroComplexArray(const unsigned int array_size,
                 __global CL_TYPE2* array)
{
    unsigned int i_global = get_global_id(0);
    if(i_global < array_size)
    {
        array[i_global] = 0.0;
    }
}
