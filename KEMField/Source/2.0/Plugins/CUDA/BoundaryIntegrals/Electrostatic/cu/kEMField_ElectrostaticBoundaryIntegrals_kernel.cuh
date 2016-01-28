#ifndef KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CUH
#define KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CUH

#include "kEMField_cuda_defines.h"

#include "kEMField_ElectrostaticBoundaryIntegrals.cuh"

//______________________________________________________________________________

__forceinline__ __global__ void PotentialBIKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *data,
        CU_TYPE* phi )
{
    CU_TYPE p[3] = {P[0],P[1],P[2]};
    phi[0] = EBI_Potential(p,shapeType,data);
}

//______________________________________________________________________________

__forceinline__ __global__ void ElectricFieldBIKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *data,
        CU_TYPE4 *eField )
{
    CU_TYPE p[3] = {P[0],P[1],P[2]};
    eField[0] = EBI_EField(p,shapeType,data);
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CUH */
