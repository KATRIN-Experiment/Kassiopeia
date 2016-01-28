#include "KGVectorOperations.hh"

#include <sstream>
#include "KGMathMessage.hh"

namespace KGeoBag
{


#ifdef KGEOBAG_MATH_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use fast BLAS based implementation

//allocation/deallocation

kg_vector*
kg_vector_alloc(unsigned int n)
{
    return gsl_vector_alloc(n);
}


kg_vector*
kg_vector_calloc(unsigned int n)
{
    return gsl_vector_calloc(n);
}


void
kg_vector_free(kg_vector* v)
{
    gsl_vector_free(v);
}

//access

double kg_vector_get(const kg_vector* v, unsigned int i)
{
    return gsl_vector_get(v, i);
}

void kg_vector_set(kg_vector* v, unsigned int i, double x)
{
    gsl_vector_set(v, i, x);
}

void kg_vector_set_zero(kg_vector* v)
{
    gsl_vector_set_zero(v);
}

void
kg_vector_set(const kg_vector* src, kg_vector* dest)
{
    gsl_vector_memcpy(dest, src);
}

//operations
void
kg_vector_scale(kg_vector* a, double scale_factor)
{
    gsl_vector_scale(a,scale_factor);
}

void
kg_vector_sub(kg_vector* a, const kg_vector* b)
{
    gsl_vector_sub(a,b);
}

void
kg_vector_add(kg_vector* a, const kg_vector* b)
{
    gsl_vector_add(a,b);
}


double
kg_vector_inner_product(const kg_vector* a, const kg_vector* b)
{
    double val = 0.0;
    gsl_blas_ddot(a, b, &val);
    return val;
}

double
kg_vector_norm(const kg_vector* a)
{
    return gsl_blas_dnrm2(a);
}


#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

//allocation/deallocation

kg_vector*
kg_vector_alloc(unsigned int n)
{
    kg_vector* v = new kg_vector();
    v->size = n;
    v->data = new double[n];
    return v;
}


kg_vector*
kg_vector_calloc(unsigned int n)
{
    kg_vector* v = new kg_vector();
    v->size = n;
    double* d = new double[n];
    for(unsigned int i=0; i<n; i++)
    {
        d[i] = 0.;
    }
    v->data = d;
    return v;
}


void
kg_vector_free(kg_vector* v)
{
    delete[] v->data;
    delete v;
}

//access

double
kg_vector_get(const kg_vector* v, unsigned int i)
{
    return v->data[i];
}

void
kg_vector_set(kg_vector* v, unsigned int i, double x)
{
    v->data[i] = x;
}

void
kg_vector_set_zero(kg_vector* v)
{
    for(unsigned int i=0; i<v->size; i++)
    {
        v->data[i] = 0.;
    }
}

void
kg_vector_set(const kg_vector* src, kg_vector* dest)
{
    if(src->size == dest->size)
    {
        for(unsigned int i=0; i<src->size; i++)
        {
            dest->data[i] = src->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_vector_set: error, vectors have difference sizes: "<<src->size<<" != "<<dest->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


//operations
void
kg_vector_scale(kg_vector* a, double scale_factor)
{
    for(unsigned int i=0; i<a->size; i++)
    {
        a->data[i] *= scale_factor;
    }
}

void
kg_vector_sub(kg_vector* a, const kg_vector* b)
{
    if(a->size == b->size )
    {
        for(unsigned int i=0; i<a->size; i++)
        {
            a->data[i] -= b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_vector_sub: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

void
kg_vector_add(kg_vector* a, const kg_vector* b)
{
    if(a->size == b->size)
    {
        for(unsigned int i=0; i<a->size; i++)
        {
            a->data[i] += b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_vector_add: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


double
kg_vector_inner_product(const kg_vector* a, const kg_vector* b)
{
    double val = 0;

    if(a->size == b->size)
    {

        for(unsigned int i=0; i<a->size; i++)
        {
            val += (a->data[i])*(b->data[i]);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_inner_product: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }

    return val;
}

double
kg_vector_norm(const kg_vector* a)
{
    double val = kg_vector_inner_product(a,a);
    return std::sqrt(val);
}

#endif

////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

void
kg_vector_normalize(kg_vector* a)
{
    double scale = kg_vector_norm(a);
    kg_vector_scale(a, 1.0/scale);
}

void
kg_vector_sub(const kg_vector* a, const kg_vector* b, kg_vector* c)
{

    if( (a->size == b->size) && (b->size == c->size) )
    {
        for(unsigned int i=0; i<a->size; i++)
        {
            c->data[i] = a->data[i] - b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_inner_sub: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<" != "<<c->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

void
kg_vector_add(const kg_vector* a, const kg_vector* b, kg_vector* c)
{
    if( (a->size == b->size) && (b->size == c->size) )
    {
        for(unsigned int i=0; i<a->size; i++)
        {
            c->data[i] = a->data[i] + b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_inner_add: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<" != "<<c->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


void
kg_vector_cross_product(const kg_vector* a, const kg_vector* b, kg_vector* c)
{
    kg_vector_set(c, 0, kg_vector_get(a,1)*kg_vector_get(b,2) - kg_vector_get(a,2)*kg_vector_get(b,1));
    kg_vector_set(c, 1, kg_vector_get(a,2)*kg_vector_get(b,0) - kg_vector_get(a,0)*kg_vector_get(b,2));
    kg_vector_set(c, 2, kg_vector_get(a,0)*kg_vector_get(b,1) - kg_vector_get(a,1)*kg_vector_get(b,0));
}

////////////////////////////////////////////////////////////////////////////////


}
