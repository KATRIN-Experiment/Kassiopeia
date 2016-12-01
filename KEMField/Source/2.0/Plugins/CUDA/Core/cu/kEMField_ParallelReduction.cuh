#ifndef KEMFIELD_PARALLELREDUCTION_CUH
#define KEMFIELD_PARALLELREDUCTION_CUH

#include "kEMField_cuda_defines.h"


//______________________________________________________________________________

__forceinline__ __device__ void Reduce( CU_TYPE* partialVal,
        CU_TYPE* val,
        int nWorkgroup,
        int groupID,
        int local_i )
{
    int tmp = nWorkgroup/2;
    int tmp_last = nWorkgroup;
    int jj = 0;

    while( tmp>0 ) {
        __syncthreads();

        if( local_i<tmp )
            partialVal[local_i] += partialVal[local_i+tmp];

        if( 2*tmp != tmp_last ) {
            if( local_i==0 )
                for( jj=2*tmp; jj<tmp_last; jj++ )
                    partialVal[local_i] += partialVal[jj];
        }

        tmp_last = tmp;
        tmp/=2;
    }

    if( local_i==0 )
        val[groupID] = partialVal[0];
}

//______________________________________________________________________________

__forceinline__ __device__ void Reduce4( CU_TYPE4* partialVal,
                            CU_TYPE4* val,
                            int nWorkgroup,
                            int groupID,
                            int local_i )
{
    int tmp = nWorkgroup/2;
    int tmp_last = nWorkgroup;
    int jj = 0;

    while( tmp>0 ) {
        __syncthreads();

        if( local_i<tmp ) {
        	partialVal[local_i].x = partialVal[local_i].x + partialVal[local_i+tmp].x;
        	partialVal[local_i].y = partialVal[local_i].y + partialVal[local_i+tmp].y;
        	partialVal[local_i].z = partialVal[local_i].z + partialVal[local_i+tmp].z;
        	partialVal[local_i].w = partialVal[local_i].w + partialVal[local_i+tmp].w;
        }

        if( 2*tmp != tmp_last ) {
            if( local_i==0 ) {
                for( jj=2*tmp; jj<tmp_last; jj++ ) {
                    partialVal[local_i].x = partialVal[local_i].x + partialVal[jj].x;
                    partialVal[local_i].y = partialVal[local_i].y + partialVal[jj].y;
                    partialVal[local_i].z = partialVal[local_i].z + partialVal[jj].z;
                    partialVal[local_i].w = partialVal[local_i].w + partialVal[jj].w;
                }
            }
        }

        tmp_last = tmp;
        tmp/=2;
    }

    if( local_i==0 )
        val[groupID] = partialVal[0];
}

#endif /* KEMFIELD_PARALLELREDUCTION_CUH */
