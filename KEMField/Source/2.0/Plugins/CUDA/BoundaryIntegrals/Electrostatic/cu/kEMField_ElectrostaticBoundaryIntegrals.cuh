#ifndef KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CUH
#define KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CUH

#include "kEMField_cuda_defines.h"
#include "kEMField_BoundaryIntegrals.cuh" // includes surface headers

#include "kEMField_ElectrostaticRectangle.cuh"
#include "kEMField_ElectrostaticTriangle.cuh"
#include "kEMField_ElectrostaticLineSegment.cuh"
//#include "kEMField_ElectrostaticConicSection.cuh"

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE EBI_Potential( const CU_TYPE* P,
        const short* shapeType,
        const CU_TYPE* data )
{
    if( shapeType[0] == TRIANGLE )
        return ET_Potential( P, data );
    if( shapeType[0] == RECTANGLE )
        return ER_Potential( P, data );
    if( shapeType[0] == LINESEGMENT )
        return EL_Potential( P, data );
    // conic section code has been deactivated since it leads to
    // long compilation time and very large kernel sizes
    //if( shapeType[0] == CONICSECTION )
        //return EC_Potential( P, data );

  return 0.;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE4 EBI_EField( const CU_TYPE* P,
        const short* shapeType,
        const CU_TYPE* data )
{
    if( shapeType[0] == TRIANGLE )
        return ET_EField( P, data );
    if( shapeType[0] == RECTANGLE )
        return ER_EField( P, data );
    if( shapeType[0] == LINESEGMENT )
        return EL_EField(P,data);
    //if( shapeType[0] == CONICSECTION )
        //return EC_EField( P, data );

    CU_TYPE4 ret;
    ret.x=0.; ret.y=0.; ret.z=0.; ret.w=0.;

    return ret;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE BI_BoundaryIntegral( int iBoundary,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        const short* shapeType_target,
        const short* shapeType_source,
        const CU_TYPE* data_target,
        const CU_TYPE* data_source )
{
    CU_TYPE P_target[3];
    BI_Centroid(&P_target[0],shapeType_target,data_target);

    CU_TYPE4 eField;
    eField.x=0.; eField.y=0.; eField.z=0.; eField.w=0.;
    CU_TYPE P_source[3];
    CU_TYPE N_target[3];
    CU_TYPE val;
    CU_TYPE dist2;


    if( BI_GetBoundaryType(iBoundary,boundaryInfo) == DIRICHLETBOUNDARY ) {
        val = EBI_Potential(P_target,shapeType_source,data_source);
    }
    else {
        eField = EBI_EField(P_target,shapeType_source,data_source);
        BI_Centroid(&P_source[0],shapeType_source,data_source);
        BI_Normal(&N_target[0],shapeType_target,data_target);

        val = eField.x*N_target[0] + eField.y*N_target[1] + eField.z*N_target[2];
        dist2 = ((P_target[0]-P_source[0])*(P_target[0]-P_source[0]) +
                (P_target[1]-P_source[1])*(P_target[1]-P_source[1]) +
                (P_target[2]-P_source[2])*(P_target[2]-P_source[2]));
        if (dist2<1.e-24)
        {
            val = val*( (1. + boundaryData[iBoundary*BOUNDARYSIZE]) / (1. - boundaryData[iBoundary*BOUNDARYSIZE]) );
        }
    }

    return val;
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CUH */
