#ifndef KFMMultidimensionalFastFourierTransform_Defined_H
#define KFMMultidimensionalFastFourierTransform_Defined_H

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMArrayMath.cl"
#include "kEMField_KFMFastFourierTransformUtilities.cl"

//compile time constants
//FFT_NDIM
//FFT_BUFFERSIZE
//the buffer size must be at least as big as max(BluesteinArraySize(dim))
//or there will not be enough space to execute all the needed transforms

//This kernel executes a single stage of multiple multidimensional FFT's on a block of data
//that consists of S N-dimensional arrays stored in row major order.
//The the dimensions of the array can be specified at run time but the maximum size of any dimension must be
//specified at compile time, in order to allocate private workspace.
//The output of the FFT is returned in the input array.
//In order for the complete multidimensional FFT to be performed, the following function
//must be repeatedly called on the same data with D = 0, 1, 2, ...FFT_NDIM
//The permutation array, twiddle factors, scale, and circulant data must correspond
//correctly to the size of the current dimension (D) being transformed
//their calculation must be handled on the host side.

//To specify usage of the GPU's constant memory, add the flag FFT_USE_CONST_MEM
//at compile time

#define USE_GPU

#ifdef USE_GPU


__kernel void
#ifdef FFT_USE_CONST_MEM
MultidimensionalFastFourierTransform_Stage(unsigned int S, //number of transforms
                                           unsigned int D, //d = 0, 1, ...FFT_NDIM-1 specifies the dimension being transformed
                                           int isForward, //1 = forward FFT, 0 = backward FFT
                                           __constant const unsigned int* array_dimensions,
                                           __constant CL_TYPE2* twiddle, //fft twiddle factors
                                           __constant CL_TYPE2* conj_twiddle, //conjugate fft twiddle factors
                                           __constant CL_TYPE2* scale, //bluestein scale factors
                                           __constant CL_TYPE2* circulant, //bluestein circulant vector
                                           __constant const unsigned int* permutation_array, //bit reversal permutation indices
                                           __global CL_TYPE2* data,
                                           __global CL_TYPE2* workspace)
#else
MultidimensionalFastFourierTransform_Stage(unsigned int S, //number of transforms
                                           unsigned int D, //d = 0, 1, ...FFT_NDIM-1 specifies the dimension being transformed
                                           int isForward, //1 = forward FFT, 0 = backward FFT
                                           __global const unsigned int* array_dimensions,
                                           __global CL_TYPE2* twiddle, //fft twiddle factors
                                           __global CL_TYPE2* conj_twiddle, //conjugate fft twiddle factors
                                           __global CL_TYPE2* scale, //bluestein scale factors
                                           __global CL_TYPE2* circulant, //bluestein circulant vector
                                           __global const unsigned int* permutation_array, //bit reversal permutation indices
                                           __global CL_TYPE2* data,
                                           __global CL_TYPE2* workspace)
#endif
{
    //get the index of the current thread
    unsigned int i_global = get_global_id(0);

    //ptr to private workspace to perform the selected 1d fft
    CL_TYPE2 buffer[FFT_BUFFERSIZE];

    //assign private variable the array dimensions
    unsigned int dim[FFT_NDIM];
    for(unsigned int i=0; i<FFT_NDIM; i++)
    {
        dim[i] = array_dimensions[i];
    }

    //compute single array block size as data access stride
    unsigned int major_stride = TotalArraySize(FFT_NDIM, dim);

    //compute number of one dimensional fft's that must be performed per block of data
    //as well as the index of the appropriate data for this thread's fft
    unsigned int index[FFT_NDIM];
    unsigned int div_scratch[FFT_NDIM];
    unsigned int non_active_dimension_size[FFT_NDIM-1];
    unsigned int non_active_dimension_value[FFT_NDIM-1];
    unsigned int non_active_dimension_index[FFT_NDIM-1];
    unsigned int n_local_1d_transforms = 1;
    unsigned int count = 0;
    for(unsigned int i = 0; i < FFT_NDIM; i++)
    {
        if(i != D)
        {
            n_local_1d_transforms *= dim[i];
            non_active_dimension_index[count] = i;
            non_active_dimension_size[count] = dim[i];
            count++;
        }
    }

    unsigned int block_id = i_global/n_local_1d_transforms;
    unsigned int fft_local_id = i_global % n_local_1d_transforms;

    //compute ptr to the n-dimensional block of data relevant to this transform
    __global CL_TYPE2* data_proxy;
    if(i_global < S*n_local_1d_transforms) //thread id must be less than total number of 1d fft's
    {
        data_proxy = &(data[major_stride*block_id]);
    }

    //now invert the local index to obtain indices of the needed row in the local block
    RowMajorIndexFromOffset(FFT_NDIM-1, fft_local_id, non_active_dimension_size, non_active_dimension_value, div_scratch);

    //copy the value of the non-active dimensions in to index
    for(unsigned int i=0; i<FFT_NDIM-1; i++)
    {
        index[ non_active_dimension_index[i] ] = non_active_dimension_value[i];
    }
    index[D] = 0;

    //compute the minor stride of the data
    unsigned int minor_stride = StrideFromRowMajorIndex(FFT_NDIM, D, dim);
    unsigned int minor_offset = OffsetFromRowMajorIndex(FFT_NDIM, dim, index);

    if(i_global < S*n_local_1d_transforms) //thread id must be less than total number of 1d fft's
    {
        //now copy the row selected by the other dimensions into the private buffer
        for(unsigned int i=0; i<dim[D]; i++)
        {
            buffer[i] = data_proxy[minor_offset + i*minor_stride];
        }

        if(isForward == 0)
        {
            //conjugate FFT input data if we have a backwards transform
            for(unsigned int i=0; i<dim[D]; i++)
            {
                buffer[i].s1 *= -1.0;
            }
        }

        //if the D-dimension's size is a power of 2, use radix-2 algorithm
        if( IsPowerOfBase(dim[D], 2) )
        {
            //permute array and execute FFT using radix-2
            PermuteArray(dim[D], permutation_array, buffer);
            FFTRadixTwo_DIT(dim[D], buffer, twiddle);
        }
        else
        {
            //not a power of 2 length
            //execute FFT using the bluestein algorithm for an arbitrary length array
            FFTBluestein(dim[D], BluesteinArraySize(dim[D]), twiddle, conj_twiddle, scale, circulant, buffer);
        }

        if(isForward == 0)
        {
            //conjugate FFT output data if we have a backwards transform
            for(unsigned int i=0; i<dim[D]; i++)
            {
                buffer[i].s1 *= -1.0;
            }
        }

        //copy the buffer back to selected row
        for(unsigned int i=0; i<dim[D]; i++)
        {
           data_proxy[minor_offset + i*minor_stride] = buffer[i];
        }

    }

}

#else //end of USE_GPU, otherwise we assume a CPU or accelerator device XXXXXXXX


__kernel void
#ifdef FFT_USE_CONST_MEM
MultidimensionalFastFourierTransform_Stage(unsigned int S, //number of transforms
                                           unsigned int D, //d = 0, 1, ...FFT_NDIM-1 specifies the dimension being transformed
                                           int isForward, //1 = forward FFT, 0 = backward FFT
                                           __constant const unsigned int* array_dimensions,
                                           __constant CL_TYPE2* twiddle, //fft twiddle factors
                                           __constant CL_TYPE2* conj_twiddle, //conjugate fft twiddle factors
                                           __constant CL_TYPE2* scale, //bluestein scale factors
                                           __constant CL_TYPE2* circulant, //bluestein circulant vector
                                           __constant const unsigned int* permutation_array, //bit reversal permutation indices
                                           __global CL_TYPE2* data,
                                           __global CL_TYPE2* workspace)
#else
MultidimensionalFastFourierTransform_Stage(unsigned int S, //number of transforms
                                           unsigned int D, //d = 0, 1, ...FFT_NDIM-1 specifies the dimension being transformed
                                           int isForward, //1 = forward FFT, 0 = backward FFT
                                           __global const unsigned int* array_dimensions,
                                           __global CL_TYPE2* twiddle, //fft twiddle factors
                                           __global CL_TYPE2* conj_twiddle, //conjugate fft twiddle factors
                                           __global CL_TYPE2* scale, //bluestein scale factors
                                           __global CL_TYPE2* circulant, //bluestein circulant vector
                                           __global const unsigned int* permutation_array, //bit reversal permutation indices
                                           __global CL_TYPE2* data,
                                           __global CL_TYPE2* workspace)
#endif
{
    //get the index of the current thread
    unsigned int i_global = get_global_id(0);

    //ptr to private workspace to perform the selected 1d fft
    __global CL_TYPE2* buffer = &(workspace[i_global*FFT_BUFFERSIZE]);
    //CL_TYPE2 buffer[FFT_BUFFERSIZE];

    //assign private variable the array dimensions
    unsigned int dim[FFT_NDIM];
    for(unsigned int i=0; i<FFT_NDIM; i++)
    {
        dim[i] = array_dimensions[i];
    }

    //compute single array block size as data access stride
    unsigned int major_stride = TotalArraySize(FFT_NDIM, dim);

    //compute number of one dimensional fft's that must be performed per block of data
    //as well as the index of the appropriate data for this thread's fft
    unsigned int index[FFT_NDIM];
    unsigned int div_scratch[FFT_NDIM];
    unsigned int non_active_dimension_size[FFT_NDIM-1];
    unsigned int non_active_dimension_value[FFT_NDIM-1];
    unsigned int non_active_dimension_index[FFT_NDIM-1];
    unsigned int n_local_1d_transforms = 1;
    unsigned int count = 0;
    for(unsigned int i = 0; i < FFT_NDIM; i++)
    {
        if(i != D)
        {
            n_local_1d_transforms *= dim[i];
            non_active_dimension_index[count] = i;
            non_active_dimension_size[count] = dim[i];
            count++;
        }
    }

    unsigned int block_id = i_global/n_local_1d_transforms;
    unsigned int fft_local_id = i_global % n_local_1d_transforms;

    //compute ptr to the n-dimensional block of data relevant to this transform
    __global CL_TYPE2* data_proxy;
    if(i_global < S*n_local_1d_transforms) //thread id must be less than total number of 1d fft's
    {
        data_proxy = &(data[major_stride*block_id]);
    }

    //now invert the local index to obtain indices of the needed row in the local block
    RowMajorIndexFromOffset(FFT_NDIM-1, fft_local_id, non_active_dimension_size, non_active_dimension_value, div_scratch);

    //copy the value of the non-active dimensions in to index
    for(unsigned int i=0; i<FFT_NDIM-1; i++)
    {
        index[ non_active_dimension_index[i] ] = non_active_dimension_value[i];
    }
    index[D] = 0;

    //compute the minor stride of the data
    unsigned int minor_stride = StrideFromRowMajorIndex(FFT_NDIM, D, dim);
    unsigned int minor_offset = OffsetFromRowMajorIndex(FFT_NDIM, dim, index);

    if(i_global < S*n_local_1d_transforms) //thread id must be less than total number of 1d fft's
    {
        //now copy the row selected by the other dimensions into the private buffer
        for(unsigned int i=0; i<dim[D]; i++)
        {
            buffer[i] = data_proxy[minor_offset + i*minor_stride];
        }

        if(isForward == 0)
        {
            //conjugate FFT input data if we have a backwards transform
            for(unsigned int i=0; i<dim[D]; i++)
            {
                buffer[i].s1 *= -1.0;
            }
        }

        //if the D-dimension's size is a power of 2, use radix-2 algorithm
        if( IsPowerOfBase(dim[D], 2) )
        {
            //permute array and execute FFT using radix-2
            PermuteArray(dim[D], permutation_array, buffer);
            FFTRadixTwo_DIT(dim[D], buffer, twiddle);
        }
        else
        {
            //not a power of 2 length
            //execute FFT using the bluestein algorithm for an arbitrary length array
            FFTBluestein(dim[D], BluesteinArraySize(dim[D]), twiddle, conj_twiddle, scale, circulant, buffer);
        }

        if(isForward == 0)
        {
            //conjugate FFT output data if we have a backwards transform
            for(unsigned int i=0; i<dim[D]; i++)
            {
                buffer[i].s1 *= -1.0;
            }
        }

        //copy the buffer back to selected row
        for(unsigned int i=0; i<dim[D]; i++)
        {
           data_proxy[minor_offset + i*minor_stride] = buffer[i];
        }

    }

}


#endif //end of use other type of device


#endif /* KFMMultidimensionalFastFourierTransform_Defined_H */
