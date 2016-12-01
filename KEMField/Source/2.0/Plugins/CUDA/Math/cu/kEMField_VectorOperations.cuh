#ifndef KEMFIELD_VECTOROPERATIONS_CUH
#define KEMFIELD_VECTOROPERATIONS_CUH

// Vector operations for three-dimensional arrays
// Author: Daniel Hilk

#include "kEMField_cuda_defines.h"


//______________________________________________________________________________

__forceinline__ __device__
void Compute_UnitVector( const CU_TYPE* A, const CU_TYPE* B, CU_TYPE* unit )
{
	const CU_TYPE mag = 1./SQRT( POW2(B[0]-A[0]) + POW2(B[1]-A[1]) + POW2(B[2]-A[2]) );
	unit[0] = mag * (B[0]-A[0]);
	unit[1] = mag * (B[1]-A[1]);
	unit[2] = mag * (B[2]-A[2]);
}

//______________________________________________________________________________

__forceinline__ __device__
void Compute_CrossProduct( const CU_TYPE* A, const CU_TYPE* B, CU_TYPE* cross )
{
	cross[0] = (A[1]*B[2]) - (A[2]*B[1]);
	cross[1] = (A[2]*B[0]) - (A[0]*B[2]);
	cross[2] = (A[0]*B[1]) - (A[1]*B[0]);
}


#endif /* KEMFIELD_VECTOROPERATIONS_CUH */
