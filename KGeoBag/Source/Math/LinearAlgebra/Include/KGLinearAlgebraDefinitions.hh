#ifndef KGLinearAlgebraDefinitions_HH__
#define KGLinearAlgebraDefinitions_HH__

#ifdef KGEOBAG_MATH_USE_GSL
//use GSL's faster linear algebra libraries

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>

#endif

#include <cmath>
#include <cstddef>
#include "KGNumericalConstants.hh"

namespace KGeoBag
{

/*
*
*@file KGLinearAlgebraDefinitions.hh
*@class KGLinearAlgebraDefinitions
*@brief simple collection of functions and structs meant to mirror an extremely minimal set of
* GSL's linear algebra functionality so that we can make GSL an optional dependency
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Nov 12 12:38:31 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#ifdef KGEOBAG_MATH_USE_GSL
    typedef gsl_vector kg_vector;
    typedef gsl_matrix kg_matrix;
#else

    struct kg_matrix
    {
        unsigned int size1;
        unsigned int size2;
        double* data;
    };

    struct kg_vector
    {
        unsigned int size;
        double* data;
    };

#endif

    //not a real sparse matrix format
    //its only purpose is to accerlate matrix vector multiplication
    //element look up is VERY slow!!
    struct kg_sparse_matrix
    {
        unsigned int n_elements;
        unsigned int size1;
        unsigned int size2;
        double* data;
        unsigned int* row;
        unsigned int* column;
    };


}//end KGeoBag

#endif /* KGLinearAlgebraDefinitions_H__ */
