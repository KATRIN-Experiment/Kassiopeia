#ifndef KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CUH
#define KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CUH

#include "kEMField_cuda_defines.h"

#include "kEMField_ElectrostaticNumericBoundaryIntegrals.cuh"
#include "kEMField_ParallelReduction.cuh"

//______________________________________________________________________________

__global__
void PotentialKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE* phi )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__  CU_TYPE partialPhi[];

    // Do the operation
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};
    partialPhi[local_i] = ( EBI_Potential(p_loc, &shapeType[i], &shapeData[i*SHAPESIZE])
            * basisData[BASISSIZE*i] );

    // now that all of the individual potentials are computed, we sum them using
    // a very basic form of parallel reduction
    Reduce( partialPhi, phi, nWorkgroup, groupID, local_i );
}

//______________________________________________________________________________

__global__
void ElectricFieldKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE4 *eField )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__ CU_TYPE4 partialEField[];

    // Do the operation
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};
    partialEField[local_i] = EBI_EField( p_loc, &shapeType[i], &shapeData[i*SHAPESIZE] );

    CU_TYPE prefactor = basisData[i*BASISSIZE];

    partialEField[local_i].x = partialEField[local_i].x*prefactor;
    partialEField[local_i].y = partialEField[local_i].y*prefactor;
    partialEField[local_i].z = partialEField[local_i].z*prefactor;

    // now that all of the individual potentials are computed, we sum them using
    // a very basic form of parallel reduction
    Reduce4(partialEField, eField, nWorkgroup, groupID, local_i);
}

//______________________________________________________________________________

__global__
void ElectricFieldAndPotentialKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE4 *eFieldAndPhi )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__ CU_TYPE4 partialEFieldAndPhi[];

    // Do the operation
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};
    partialEFieldAndPhi[local_i] = EBI_EFieldAndPotential( p_loc, &shapeType[i], &shapeData[i*SHAPESIZE] );

    CU_TYPE prefactor = basisData[i*BASISSIZE];

    partialEFieldAndPhi[local_i].x = partialEFieldAndPhi[local_i].x*prefactor;
    partialEFieldAndPhi[local_i].y = partialEFieldAndPhi[local_i].y*prefactor;
    partialEFieldAndPhi[local_i].z = partialEFieldAndPhi[local_i].z*prefactor;
    partialEFieldAndPhi[local_i].w = partialEFieldAndPhi[local_i].z*prefactor;

    // now that all of the individual potentials are computed, we sum them using
    // a very basic form of parallel reduction
    Reduce4(partialEFieldAndPhi, eFieldAndPhi, nWorkgroup, groupID, local_i);
}

//______________________________________________________________________________

__global__
void SubsetPotentialKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE* phi,
        unsigned int nElements,
        const unsigned int* elementIDs )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__  CU_TYPE partialPhi[];

    // Do the operation
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};
    if( i < nElements ) {
        //look up the element index
        unsigned int index = elementIDs[i];
        partialPhi[local_i] = ( EBI_Potential(p_loc, &shapeType[index], &shapeData[index*SHAPESIZE]) * basisData[BASISSIZE*index] );
    } 
    else {
        partialPhi[local_i] = 0.0;
    }

    Reduce( partialPhi, phi, nWorkgroup, groupID, local_i );
}

//______________________________________________________________________________

__global__
void SubsetElectricFieldKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE4* eField,
        unsigned int nElements,
        const unsigned int* elementIDs )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__  CU_TYPE4 partialEField[];

    // Get the index of the current element to be processed
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};

    if(i < nElements) {
        //look up the element index
        unsigned int index = elementIDs[i];
        partialEField[local_i] = EBI_EField(p_loc, &shapeType[index], &shapeData[index*SHAPESIZE]);
        CU_TYPE prefactor = basisData[index*BASISSIZE];
        partialEField[local_i].x = partialEField[local_i].x*prefactor;
        partialEField[local_i].y = partialEField[local_i].y*prefactor;
        partialEField[local_i].z = partialEField[local_i].z*prefactor;
    }
    else {
        partialEField[local_i].x = 0.0;
        partialEField[local_i].y = 0.0;
        partialEField[local_i].z = 0.0;
    }

    // now that all of the individual potentials are computed, we sum them using
    // a very basic form of parallel reduction
    Reduce4( partialEField, eField, nWorkgroup, groupID, local_i );
}

//______________________________________________________________________________

__global__
void SubsetElectricFieldAndPotentialKernel( const CU_TYPE *P,
        const short *shapeType,
        const CU_TYPE *shapeData,
        const CU_TYPE *basisData,
        CU_TYPE4* eFieldAndPhi,
        unsigned int nElements,
        const unsigned int* elementIDs )
{
    // Get the index of the current element to be processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__  CU_TYPE4 partialEFieldAndPhi[];

    // Get the index of the current element to be processed
    CU_TYPE p_loc[3] = {P[0],P[1],P[2]};

    if(i < nElements) {
        //look up the element index
        unsigned int index = elementIDs[i];
        partialEFieldAndPhi[local_i] = EBI_EFieldAndPotential(p_loc, &shapeType[index], &shapeData[index*SHAPESIZE]);

        CU_TYPE prefactor = basisData[index*BASISSIZE];

        partialEFieldAndPhi[local_i].x = partialEFieldAndPhi[local_i].x*prefactor;
        partialEFieldAndPhi[local_i].y = partialEFieldAndPhi[local_i].y*prefactor;
        partialEFieldAndPhi[local_i].z = partialEFieldAndPhi[local_i].z*prefactor;
    }
    else {
        partialEFieldAndPhi[local_i].x = 0.0;
        partialEFieldAndPhi[local_i].y = 0.0;
        partialEFieldAndPhi[local_i].z = 0.0;
    }

    // now that all of the individual potentials are computed, we sum them using
    // a very basic form of parallel reduction
    Reduce4( partialEFieldAndPhi, eFieldAndPhi, nWorkgroup, groupID, local_i );
}


#endif /* KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CUH */
