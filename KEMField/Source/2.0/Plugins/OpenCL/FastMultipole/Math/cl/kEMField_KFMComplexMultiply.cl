#include "kEMField_defines.h"

inline CL_TYPE2 ComplexMultiply(CL_TYPE2 x, CL_TYPE2 y)
{
    CL_TYPE2 z;
    z.s0 = x.s0*y.s0 - x.s1*y.s1;
    z.s1 = x.s0*y.s1 + x.s1*y.s0;
    return z;
}
