#include "kEMField_opencl_defines.h"

void ReduceRow( __local CL_TYPE* scratch, __global CL_TYPE* target)
{
    //we assume that the workgroup size is a power of 2
    int workgroup_size = get_local_size(0);
    int id = get_local_id(0);

    int upper_limit = workgroup_size/2;

    while( upper_limit > 0)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(id < upper_limit)
        {
            scratch[id] += scratch[upper_limit+id];
        }
        upper_limit /= 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){target[0] += scratch[0];}
}

void LoadVector
(
    unsigned int size,
    __global unsigned int* index_list,
    __global CL_TYPE* vector,
    __local CL_TYPE* scratch
)
{
    int id = get_local_id(0);
    if(id < size)
    {
        scratch[id] = vector[ index_list[id] ];
    }
    else
    {
        scratch[id] = 0.0;
    }
}


__kernel void
DenseBlockSparseMatrixVectorProduct
(
    const unsigned int NRows,
    const unsigned int NCols,
    __global unsigned int* Rows,
    __global unsigned int* Columns,
    __global CL_TYPE* Elements,
    __global CL_TYPE* in_vector,
    __global CL_TYPE* out_vector,
    __local CL_TYPE* scratch1,
    __local CL_TYPE* scratch2
)
{
    //The number of columns per row must always be less than the workgroup size
    //If there is a row with more columns than this, then it must be broken up
    //into smaller pieces for processing

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    LoadVector(NCols, Columns, in_vector, scratch1);

    barrier(CLK_LOCAL_MEM_FENCE);

    //now we perform the element wise multiplication
    if(local_id < NCols)
    {
        scratch2[local_id] = scratch1[local_id]*Elements[group_id*NCols + local_id];
    }
    else
    {
        scratch2[local_id] = 0.0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    ReduceRow(scratch2, &(out_vector[ Rows[group_id] ] ) );
}
