#ifndef KFMVectorOperations_HH__
#define KFMVectorOperations_HH__

/*
*
*@file KFMVectorOperations.hh
*@class KFMVectorOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:03 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KFMLinearAlgebraDefinitions.hh"

namespace KEMField
{

    //allocation/deallocation
    kfm_vector* kfm_vector_alloc(unsigned int n);
    kfm_vector* kfm_vector_calloc(unsigned int n);
    void kfm_vector_free(kfm_vector* v);

    //access
    double kfm_vector_get(const kfm_vector* v, unsigned int i);
    void kfm_vector_set(kfm_vector* v, unsigned int i, double x);
    void kfm_vector_set_zero(kfm_vector* v);
    void kfm_vector_set(const kfm_vector* src, kfm_vector* dest);

    //operations
    void kfm_vector_scale(kfm_vector* a, double scale_factor);

    double kfm_vector_inner_product(const kfm_vector* a, const kfm_vector* b);

    //only applicable to vectors of length 3, but no check is performed
    void kfm_vector_cross_product(const kfm_vector* a, const kfm_vector* b, kfm_vector* c);

    double kfm_vector_norm(const kfm_vector* a);
    void kfm_vector_normalize(kfm_vector* a);

    void kfm_vector_sub(const kfm_vector* a, const kfm_vector* b, kfm_vector* c);
    void kfm_vector_add(const kfm_vector* a, const kfm_vector* b, kfm_vector* c);

    void kfm_vector_sub(kfm_vector* a, const kfm_vector* b);
    void kfm_vector_add(kfm_vector* a, const kfm_vector* b);

}


#endif /* KFMVectorOperations_H__ */
