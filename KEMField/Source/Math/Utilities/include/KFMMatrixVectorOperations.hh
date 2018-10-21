#ifndef KFMMatrixVectorOperations_HH__
#define KFMMatrixVectorOperations_HH__



/*
*
*@file KFMMatrixVectorOperations.hh
*@class KFMMatrixVectorOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:42 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KFMLinearAlgebraDefinitions.hh"

namespace KEMField
{


void kfm_matrix_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out);

void kfm_matrix_transpose_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out);

void kfm_vector_outer_product(const kfm_vector* a, const kfm_vector* b, kfm_matrix* p);

void kfm_sparse_matrix_vector_product(const kfm_sparse_matrix* m, const kfm_vector* in, kfm_vector* out);

}


#endif /* KFMMatrixVectorOperations_H__ */
