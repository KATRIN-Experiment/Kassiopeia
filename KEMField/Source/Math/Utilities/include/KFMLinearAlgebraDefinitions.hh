#ifndef KFMLinearAlgebraDefinitions_HH__
#define KFMLinearAlgebraDefinitions_HH__

#ifdef KEMFIELD_USE_GSL
//use GSL's faster linear algebra libraries

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#endif

#include "KFMNumericalConstants.hh"

#include <cmath>
#include <cstddef>

namespace KEMField
{

/*
*
*@file KFMLinearAlgebraDefinitions.hh
*@class KFMLinearAlgebraDefinitions
*@brief simple collection of functions and structs meant to mirror an extremely minimal set of
* GSL's linear algebra functionality so that we make GSL and optional dependency
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Nov 12 12:38:31 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#ifdef KEMFIELD_USE_GSL
typedef gsl_vector kfm_vector;
typedef gsl_matrix kfm_matrix;
#else

struct kfm_matrix
{
    unsigned int size1;
    unsigned int size2;
    double* data;
};

struct kfm_vector
{
    unsigned int size;
    double* data;
};

#endif

//not a real sparse matrix format
//its only purpose is to accerlate matrix vector multiplication
//element look up is VERY slow!!
struct kfm_sparse_matrix
{
    unsigned int n_elements;
    unsigned int size1;
    unsigned int size2;
    double* data;
    unsigned int* row;
    unsigned int* column;
};


}  // namespace KEMField

#endif /* KFMLinearAlgebraDefinitions_H__ */
