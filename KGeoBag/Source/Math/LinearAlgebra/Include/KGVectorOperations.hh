#ifndef KGVectorOperations_HH__
#define KGVectorOperations_HH__

/*
*
*@file KGVectorOperations.hh
*@class KGVectorOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:03 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KGLinearAlgebraDefinitions.hh"

namespace KGeoBag
{

    //allocation/deallocation
    kg_vector* kg_vector_alloc(size_t n);
    kg_vector* kg_vector_calloc(size_t n);
    void kg_vector_free(kg_vector* v);

    //access
    double kg_vector_get(const kg_vector* v, size_t i);
    void kg_vector_set(kg_vector* v, size_t i, double x);
    void kg_vector_set_zero(kg_vector* v);
    void kg_vector_set(const kg_vector* src, kg_vector* dest);

    //operations
    void kg_vector_scale(kg_vector* a, double scale_factor);

    double kg_vector_inner_product(const kg_vector* a, const kg_vector* b);

    //only applicable to vectors of length 3, but no check is performed
    void kg_vector_cross_product(const kg_vector* a, const kg_vector* b, kg_vector* c);

    double kg_vector_norm(const kg_vector* a);
    void kg_vector_normalize(kg_vector* a);

    void kg_vector_sub(const kg_vector* a, const kg_vector* b, kg_vector* c);
    void kg_vector_add(const kg_vector* a, const kg_vector* b, kg_vector* c);

    void kg_vector_sub(kg_vector* a, const kg_vector* b);
    void kg_vector_add(kg_vector* a, const kg_vector* b);

}


#endif /* KGVectorOperations_H__ */
