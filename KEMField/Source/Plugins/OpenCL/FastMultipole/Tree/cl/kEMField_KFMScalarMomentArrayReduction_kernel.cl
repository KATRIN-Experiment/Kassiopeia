#include "kEMField_opencl_defines.h"

__kernel void
ScalarMomentArrayReduction(const unsigned int array_size, //total size of array
                           const unsigned int spatial_stride, //size of spatial block
                           const unsigned int n_terms, //number of terms in series
                           __global CL_TYPE2* child_moments, //the child moments over which we sum
                           __global CL_TYPE2* parent_moments) //parents moments, output
{
    //this is not the most efficient method, but it is easy to implement

    unsigned int i = get_global_id(0); //sums the i'th moment
    if(i < n_terms)
    {
        CL_TYPE2 accumulator = 0.0;

        unsigned int base_index = i*spatial_stride;
        for(unsigned int n = 0; n < spatial_stride; n++)
        {
            accumulator += child_moments[base_index + n];
        }

        //write out to parent's moments
        parent_moments[i] = accumulator;
    }
}                                                            
