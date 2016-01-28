#include "kEMField_defines.h"

#include "kEMField_LinearAlgebra.cl"

__kernel void
ElectrostaticSparseShellMatrixVectorProduct_ComputeBlock
(
    __global const short* shapeInfo, //fixed argument
    __global const CL_TYPE* shapeData, //fixed argument
    __global const int* boundaryInfo, //fixed argument
    __global const CL_TYPE* boundaryData, //fixed argument
    const unsigned int TotalNElements,
    const unsigned int TotalNBlocks,
    __global unsigned int* RowSizes,
    __global unsigned int* ColSizes,
    __global unsigned int* Rows,
    __global unsigned int* Columns,
    __global CL_TYPE* in_vector,
    __global CL_TYPE* block_data
)
{
    unsigned int global_id = get_global_id(0);

    //work item must be less than total number of elements in the block
    if(global_id < TotalNElements)
    {
        unsigned int element_total = 0;
        unsigned int row_total = 0;
        unsigned int col_total = 0;

        CL_TYPE mx_element = 0;

        for(unsigned int i=0; i<TotalNBlocks; i++)
        {
            unsigned int current_row_size = RowSizes[i];
            unsigned int current_col_size = ColSizes[i];
            unsigned int current_block_size = current_row_size*current_col_size;

            if( (element_total <= global_id) && (global_id < (element_total + current_block_size) ) )
            {
                unsigned int row_offset = (global_id - element_total)/current_col_size;
                unsigned int col_offset = (global_id - element_total) - row_offset*current_col_size;
                unsigned int row = Rows[row_total + row_offset ];
                unsigned int col = Columns[col_total + col_offset ];
                mx_element = LA_GetMatrixElement(row,col,boundaryInfo,boundaryData,shapeInfo,shapeData);
                mx_element *= in_vector[col];
                break;
            }
            else
            {
                row_total += current_row_size;
                col_total += current_col_size;
                element_total += current_block_size;
            }
        }

        block_data[global_id] = mx_element;
    }
}

__kernel void
ElectrostaticSparseShellMatrixVectorProduct_ReduceBlock
(
    const unsigned int TotalNBlocks,
    __global unsigned int* RowSizes,
    __global unsigned int* ColSizes,
    __global unsigned int* Rows,
    __global CL_TYPE* block_data,
    __global CL_TYPE* out_vector
)
{
    unsigned int global_id = get_global_id(0);

    //each work item processes a block
    //(this could be more efficient if each work item processed a row instead)
    if(global_id < TotalNBlocks)
    {
        //calculate row, column, and data indices of the beginning of the block
        unsigned int row_start = 0;
        unsigned int col_start = 0;
        unsigned int data_start = 0;
        for(unsigned int i=0; i<global_id; i++)
        {
            unsigned int n_row = RowSizes[i];
            unsigned int n_col = ColSizes[i];
            row_start += n_row;
            col_start += n_col;
            data_start += n_row*n_col;
        }

        unsigned int n_row = RowSizes[global_id];
        unsigned int n_col = ColSizes[global_id];
        unsigned int data_offset = 0;

        for(unsigned int i=0; i<n_row; i++)
        {
            CL_TYPE val = 0;
            for(unsigned int j=0; j<n_col; j++)
            {
                val += block_data[data_start + data_offset];
                data_offset++;
            }
            out_vector[ Rows[row_start + i] ] += val;
        }
    }
}
