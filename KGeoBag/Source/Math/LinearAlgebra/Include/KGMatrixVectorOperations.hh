#ifndef KGMatrixVectorOperations_HH__
#define KGMatrixVectorOperations_HH__



/*
*
*@file KGMatrixVectorOperations.hh
*@class KGMatrixVectorOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:42 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KGLinearAlgebraDefinitions.hh"

namespace KGeoBag
{


void kg_matrix_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out);

void kg_matrix_transpose_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out);

void kg_vector_outer_product(const kg_vector* a, const kg_vector* b, kg_matrix* p);

void kg_sparse_matrix_vector_product(const kg_sparse_matrix* m, const kg_vector* in, kg_vector* out);

}


#endif /* KGMatrixVectorOperations_H__ */
