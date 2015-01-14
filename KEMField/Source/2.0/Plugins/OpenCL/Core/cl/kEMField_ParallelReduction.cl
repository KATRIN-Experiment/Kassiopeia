#ifndef KEMFIELD_PARALLELREDUCTION_CL
#define KEMFIELD_PARALLELREDUCTION_CL

#include "kEMField_defines.h"

//______________________________________________________________________________

void Reduce(__local  CL_TYPE* partialVal,
	    __global CL_TYPE* val,
	    int               nWorkgroup,
	    int               groupID,
	    int               local_i)
{
  int tmp = nWorkgroup/2;
  int tmp_last = nWorkgroup;
  int jj = 0;

  while (tmp>0)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i<tmp)
      partialVal[local_i] += partialVal[local_i+tmp];
    if (2*tmp != tmp_last)
    {
      if (local_i==0)
    	for (jj=2*tmp;jj<tmp_last;jj++)
    	  partialVal[local_i] += partialVal[jj];
    }

    tmp_last = tmp;
    tmp/=2;
  }

  if (local_i==0)
    val[groupID] = partialVal[0];
}

//______________________________________________________________________________

void Reduce4(__local  CL_TYPE4* partialVal,
	     __global CL_TYPE4* val,
	     int                nWorkgroup,
	     int                groupID,
	     int                local_i)
{
  int tmp = nWorkgroup/2;
  int tmp_last = nWorkgroup;
  int jj = 0;

  while (tmp>0)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i<tmp)
      partialVal[local_i] = partialVal[local_i] + partialVal[local_i+tmp];
    if (2*tmp != tmp_last)
    {
      if (local_i==0)
    	for (jj=2*tmp;jj<tmp_last;jj++)
    	  partialVal[local_i] = partialVal[local_i] + partialVal[jj];
    }

    tmp_last = tmp;
    tmp/=2;
  }

  if (local_i==0)
    val[groupID] = partialVal[0];
}

#endif /* KEMFIELD_PARALLELREDUCTION_CL */
