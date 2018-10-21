#include "kEMField_opencl_defines.h"
#include "kEMField_KFMArrayMath.cl"

//__kernel void
//ElectrostaticBufferedRemoteToLocalCopyAndScale(const unsigned int n_moment_sets,
//                                               const unsigned int term_stride,
//                                               const unsigned int spatial_stride,
//                                               const unsigned int tree_level,
//                                               const unsigned int parent_node_start_index,
//                                               const unsigned int parent_offset,
//                                               __constant const CL_TYPE* scale_factor_array,
//                                               __global unsigned int* node_ids,
//                                               __global unsigned int* block_set_ids,
//                                               __global CL_TYPE2* node_moments,
//                                               __global CL_TYPE2* block_moments)
//{
//    unsigned int i_global = get_global_id(0);
//    if(i_global < n_moment_sets*term_stride)
//    {
//        unsigned int block_id = block_set_ids[parent_node_start_index + i_global/term_stride];
//        unsigned int node_id = node_ids[parent_node_start_index + i_global/term_stride];
//        unsigned int term_index = i_global%term_stride;

//        CL_TYPE scale_factor = scale_factor_array[tree_level*term_stride + term_index];
//        block_moments[parent_offset*term_stride*spatial_stride + term_index*spatial_stride + block_id] =  scale_factor*node_moments[node_id*term_stride + term_index];
//    }
//}



__kernel void
ElectrostaticBufferedRemoteToLocalCopyAndScale(const unsigned int n_parent_nodes,
                                               const unsigned int n_children,
                                               const unsigned int term_stride,
                                               const unsigned int spatial_stride,
                                               const unsigned int tree_level,
                                               __constant const CL_TYPE* scale_factor_array,
                                               __global int* child_node_ids,
                                               __global unsigned int* child_block_set_ids,
                                               __global CL_TYPE2* node_moments,
                                               __global CL_TYPE2* block_moments)
{
    unsigned int i_global = get_global_id(0);
    unsigned int dim_size[3];
    unsigned int ind[3];
    unsigned int div[3];

    if(i_global < n_parent_nodes*term_stride*n_children)
    {
        //size of each dimension of the work-items
        dim_size[0] = n_parent_nodes;
        dim_size[1] = term_stride;
        dim_size[2] = n_children;

        ind[0] = 0; //node index
        ind[1] = 0; //multipole term index
        ind[2] = 0; //spatial block index
        div[0] = 0; div[1] = 0; div[2] = 0;

        RowMajorIndexFromOffset(3, i_global, dim_size, ind, div);

        //parent_id*n_children + child_index
        int node_id = child_node_ids[ind[0]*n_children + ind[2]];

        if(node_id != -1)
        {
            unsigned int block_id = child_block_set_ids[ind[0]*n_children + ind[2]];

            //look up scale factor for this tree level and term
            CL_TYPE scale_factor = scale_factor_array[tree_level*term_stride + ind[1]];

            //compute offset for block moment array
            dim_size[2] = spatial_stride; //change the size of the last index, to match the full spatial stride
            ind[2] = block_id;
            unsigned int offset = OffsetFromRowMajorIndex(3, dim_size, ind);

            //set the appropriate block moment, scaled by the appropriate factor
            block_moments[offset] = scale_factor*node_moments[node_id*term_stride + ind[1]];
        }
    }
}
