#ifndef KEMFIELD_GAUSSIANCUBATURE_CL
#define KEMFIELD_GAUSSIANCUBATURE_CL

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMArrayMath.cl"

//define gaussian cubature macro

#define GaussianCubature(FUNCTION, DDIM, RDIM)                                 \
void GaussianCubature_##FUNCTION##_##DDIM##_##RDIM(int N,                      \
                                           __constant const CL_TYPE* abscissa, \
                                           __constant const CL_TYPE* weights,  \
                                           CL_TYPE* lower_limits,              \
                                           CL_TYPE* upper_limits,              \
                                           CL_TYPE* par,                       \
                                           CL_TYPE* result)                    \
{                                                                              \
                                                                               \
    unsigned int n_eval = pown(N, DDIM);                                       \
    unsigned int dim_size[DDIM];                                               \
    unsigned int index[DDIM];                                                  \
    unsigned int div[DDIM];                                                    \
                                                                               \
    CL_TYPE point[DDIM];                                                       \
    CL_TYPE weight;                                                            \
    CL_TYPE function_result[RDIM];                                             \
                                                                               \
    for(unsigned int i=0; i<DDIM; i++){ dim_size[i] = N;};                     \
    for(unsigned int i=0; i<RDIM; i++){ result[i] = 0.0;};                     \
                                                                               \
    for(unsigned int i=0; i<n_eval; i++)                                       \
    {                                                                          \
        RowMajorIndexFromOffset(DDIM, i, dim_size, index, div);                \
        weight = 1.0;                                                          \
        for(unsigned int j=0; j<DDIM; j++)                                     \
        {                                                                      \
            point[j] = 0.0;                                                    \
            point[j] += 0.5*(upper_limits[j] - lower_limits[j]);               \
            point[j] *= abscissa[index[j]];                                    \
            point[j] += 0.5*(upper_limits[j] + lower_limits[j]);               \
            weight *= weights[index[j]];                                       \
        }                                                                      \
                                                                               \
        FUNCTION(par, point, function_result);                                 \
                                                                               \
        for(unsigned int j=0; j<RDIM; j++)                                     \
        {                                                                      \
            result[j] += weight*function_result[j];                            \
        }                                                                      \
    }                                                                          \
                                                                               \
    CL_TYPE prefactor = 1.0;                                                   \
    for(unsigned int i=0; i<DDIM; i++)                                         \
    {                                                                          \
        prefactor *= 0.5*(upper_limits[j] - lower_limits[j])                   \
    }                                                                          \
                                                                               \
    for(unsigned int j=0; j<RDIM; j++)                                         \
    {                                                                          \
        result[j] *= prefactor;                                                \
    }                                                                          \
                                                                               \
};

//end of macro

//macro for complex functions

#define GaussianCubatureComplex(FUNCTION, DDIM, RDIM)                          \
void GaussianCubatureComplex_##FUNCTION##_##DDIM##_##RDIM(int N,               \
                                           __constant const CL_TYPE* abscissa, \
                                           __constant const CL_TYPE* weights,  \
                                           CL_TYPE* lower_limits,              \
                                           CL_TYPE* upper_limits,              \
                                           CL_TYPE* par,                       \
                                           CL_TYPE2* result)                   \
{                                                                              \
                                                                               \
    unsigned int n_eval = 1;                                                   \
    for(unsigned int i=0; i<DDIM; i++){ n_eval *= N;};                         \
    unsigned int dim_size[DDIM];                                               \
    unsigned int index[DDIM];                                                  \
    unsigned int div[DDIM];                                                    \
                                                                               \
    CL_TYPE point[DDIM];                                                       \
    CL_TYPE weight;                                                            \
    CL_TYPE2 function_result[RDIM];                                            \
                                                                               \
    for(unsigned int i=0; i<DDIM; i++){ dim_size[i] = N;};                     \
    for(unsigned int i=0; i<RDIM; i++){ result[i] = 0.0;};                     \
                                                                               \
    for(unsigned int i=0; i<n_eval; i++)                                       \
    {                                                                          \
        RowMajorIndexFromOffset(DDIM, i, dim_size, index, div);                \
        weight = 1.0;                                                          \
        for(unsigned int j=0; j<DDIM; j++)                                     \
        {                                                                      \
            point[j] = 0.0;                                                    \
            point[j] += 0.5*(upper_limits[j] - lower_limits[j]);               \
            point[j] *= abscissa[index[j]];                                    \
            point[j] += 0.5*(upper_limits[j] + lower_limits[j]);               \
            weight *= weights[index[j]];                                       \
        }                                                                      \
                                                                               \
        FUNCTION(par, point, function_result);                                 \
                                                                               \
        for(unsigned int j=0; j<RDIM; j++)                                     \
        {                                                                      \
            result[j] += weight*function_result[j];                            \
        }                                                                      \
    }                                                                          \
                                                                               \
    CL_TYPE prefactor = 1.0;                                                   \
    for(unsigned int i=0; i<DDIM; i++)                                         \
    {                                                                          \
        prefactor *= 0.5*(upper_limits[i] - lower_limits[i]);                  \
    }                                                                          \
                                                                               \
    for(unsigned int j=0; j<RDIM; j++)                                         \
    {                                                                          \
        result[j] *= prefactor;                                                \
    }                                                                          \
                                                                               \
};

//end of macro



#endif
