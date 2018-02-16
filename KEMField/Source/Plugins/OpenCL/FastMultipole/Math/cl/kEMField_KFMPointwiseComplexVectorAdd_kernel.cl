#ifndef KFMPointwiseComplexVectorAdd_Defined_H
#define KFMPointwiseComplexVectorAdd_Defined_H



#include  "kEMField_opencl_defines.h"
__kernel void
PointwiseComplexVectorAdd(const unsigned int array_size,
                          __global CL_TYPE2* input1,
                          __global CL_TYPE2* input2,
                          __global CL_TYPE2* output)
{
    int i = get_global_id(0);
    if(i < array_size)
    {
        CL_TYPE2 a = input1[i];
        CL_TYPE2 b = input2[i];
        CL_TYPE2 c = a + b;
        output[i] = c;
    }
}


#endif
