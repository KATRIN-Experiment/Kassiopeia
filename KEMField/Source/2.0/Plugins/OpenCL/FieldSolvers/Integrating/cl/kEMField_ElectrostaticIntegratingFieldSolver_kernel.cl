#ifndef KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CL
#define KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CL

#include "kEMField_defines.h"

#include "kEMField_ParallelReduction.cl"
#include "kEMField_ElectrostaticBoundaryIntegrals.cl"

//______________________________________________________________________________

__kernel void Potential(__global const CL_TYPE *P,
			__global const short *shapeType,
			__global const CL_TYPE *shapeData,
			__global const CL_TYPE *basisData,
			__local  CL_TYPE* partialPhi,
			__global CL_TYPE* phi)
{
  // Get the index of the current element to be processed
  int i = get_global_id(0);

  int local_i = get_local_id(0);
  int nWorkgroup = get_local_size(0);
  int groupID = get_group_id(0);

  // Do the operation

  CL_TYPE p_loc[3] = {P[0],P[1],P[2]};
  partialPhi[local_i] = (EBI_Potential(p_loc,
				       &shapeType[i],
				       &shapeData[i*SHAPESIZE])*
			 basisData[BASISSIZE*i]);

  // now that all of the individual potentials are computed, we sum them using
  // a very basic form of parallel reduction

  Reduce(partialPhi,
  	 phi,
  	 nWorkgroup,
  	 groupID,
  	 local_i);
}

//______________________________________________________________________________

__kernel void ElectricField(__global const CL_TYPE *P,
			    __global const short *shapeType,
			    __global const CL_TYPE *shapeData,
			    __global const CL_TYPE *basisData,
			    __local  CL_TYPE4* partialEField,
			    __global CL_TYPE4 *eField)
{
  // Get the index of the current element to be processed
  int i = get_global_id(0);

  int local_i = get_local_id(0);
  int nWorkgroup = get_local_size(0);
  int groupID = get_group_id(0);

  // Do the operation

  CL_TYPE p_loc[3] = {P[0],P[1],P[2]};
  partialEField[local_i] = EBI_EField(p_loc,
				      &shapeType[i],
				      &shapeData[i*SHAPESIZE]);
  CL_TYPE prefactor = basisData[i*BASISSIZE];
  partialEField[local_i].x = partialEField[local_i].x*prefactor;
  partialEField[local_i].y = partialEField[local_i].y*prefactor;
  partialEField[local_i].z = partialEField[local_i].z*prefactor;

  // now that all of the individual potentials are computed, we sum them using
  // a very basic form of parallel reduction

  Reduce4(partialEField,
	  eField,
	  nWorkgroup,
	  groupID,
	  local_i);
}

//______________________________________________________________________________

__kernel void SubsetPotential(__global const CL_TYPE *P,
			                  __global const short *shapeType,
			                  __global const CL_TYPE *shapeData,
			                  __global const CL_TYPE *basisData,
			                  __local  CL_TYPE* partialPhi,
			                  __global CL_TYPE* phi,
                              unsigned int nElements,
			                  __global const unsigned int* elementIDs)
{
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    int local_i = get_local_id(0);
    int nWorkgroup = get_local_size(0);
    int groupID = get_group_id(0);

    // Do the operation

    CL_TYPE p_loc[3] = {P[0],P[1],P[2]};

    if(i < nElements)
    {
        //look up the element index
        unsigned int index = elementIDs[i];
        partialPhi[local_i] = (EBI_Potential(p_loc, &shapeType[index], &shapeData[index*SHAPESIZE])*basisData[BASISSIZE*index]);
    }
    else
    {
        partialPhi[local_i] = 0.0;
    }

    Reduce(partialPhi, phi, nWorkgroup, groupID, local_i);

}

//______________________________________________________________________________

__kernel void SubsetElectricField(__global const CL_TYPE *P,
			    __global const short *shapeType,
			    __global const CL_TYPE *shapeData,
			    __global const CL_TYPE *basisData,
			    __local  CL_TYPE4* partialEField,
			    __global CL_TYPE4* eField,
                unsigned int nElements,
                __global const unsigned int* elementIDs)
{
    int i = get_global_id(0);
    int local_i = get_local_id(0);
    int nWorkgroup = get_local_size(0);
    int groupID = get_group_id(0);

    // Get the index of the current element to be processed
    CL_TYPE p_loc[3] = {P[0],P[1],P[2]};

    if(i < nElements)
    {
        //look up the element index
        unsigned int index = elementIDs[i];
        partialEField[local_i] = EBI_EField(p_loc, &shapeType[index], &shapeData[index*SHAPESIZE]);
        CL_TYPE prefactor = basisData[index*BASISSIZE];
        partialEField[local_i].x = partialEField[local_i].x*prefactor;
        partialEField[local_i].y = partialEField[local_i].y*prefactor;
        partialEField[local_i].z = partialEField[local_i].z*prefactor;
    }
    else
    {
        partialEField[local_i].x = 0.0;
        partialEField[local_i].y = 0.0;
        partialEField[local_i].z = 0.0;
    }

  // now that all of the individual potentials are computed, we sum them using
  // a very basic form of parallel reduction

  Reduce4(partialEField,
	  eField,
	  nWorkgroup,
	  groupID,
	  local_i);

}



#endif /* KEMFIELD_ELECTROSTATICINTEGRATINGFIELDSOLVER_KERNEL_CL */
