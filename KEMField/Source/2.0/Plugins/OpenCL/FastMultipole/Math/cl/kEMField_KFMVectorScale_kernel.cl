#ifndef KFMVectorScale_Defined_H
#define KFMVectorScale_Defined_H

#include "kEMField_defines.h"

__kernel void
VectorScale(const unsigned int array_size,
            CL_TYPE factor,
            __global CL_TYPE2* data)
{
    int i = get_global_id(0);
    if(i < array_size)
    {
        data[i] *= factor;
    }
}

#endif
