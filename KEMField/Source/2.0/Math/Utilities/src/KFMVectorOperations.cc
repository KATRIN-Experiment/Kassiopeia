#include "KFMVectorOperations.hh"
#include "KFMMessaging.hh"

namespace KEMField
{


#ifdef KEMFIELD_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use fast BLAS based implementation

//allocation/deallocation

kfm_vector*
kfm_vector_alloc(unsigned int n)
{
    return gsl_vector_alloc(n);
}


kfm_vector*
kfm_vector_calloc(unsigned int n)
{
    return gsl_vector_calloc(n);
}


void
kfm_vector_free(kfm_vector* v)
{
    gsl_vector_free(v);
}

//access

double kfm_vector_get(const kfm_vector* v, unsigned int i)
{
    return gsl_vector_get(v, i);
}

void kfm_vector_set(kfm_vector* v, unsigned int i, double x)
{
    gsl_vector_set(v, i, x);
}

void kfm_vector_set_zero(kfm_vector* v)
{
    gsl_vector_set_zero(v);
}

void
kfm_vector_set(const kfm_vector* src, kfm_vector* dest)
{
    gsl_vector_memcpy(dest, src);
}

//operations
void
kfm_vector_scale(kfm_vector* a, double scale_factor)
{
    gsl_vector_scale(a,scale_factor);
}

void
kfm_vector_sub(kfm_vector* a, const kfm_vector* b)
{
    gsl_vector_sub(a,b);
}

void
kfm_vector_add(kfm_vector* a, const kfm_vector* b)
{
    gsl_vector_add(a,b);
}


double
kfm_vector_inner_product(const kfm_vector* a, const kfm_vector* b)
{
    double val = 0.0;
    gsl_blas_ddot(a, b, &val);
    return val;
}

double
kfm_vector_norm(const kfm_vector* a)
{
    return gsl_blas_dnrm2(a);
}


#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

//allocation/deallocation

kfm_vector*
kfm_vector_alloc(unsigned int n)
{
    kfm_vector* v = new kfm_vector();
    v->size = n;
    v->data = new double[n];
    return v;
}


kfm_vector*
kfm_vector_calloc(unsigned int n)
{
    kfm_vector* v = new kfm_vector();
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
kfm_vector_free(kfm_vector* v)
{
    delete[] v->data;
    delete v;
}

//access

double
kfm_vector_get(const kfm_vector* v, unsigned int i)
{
    return v->data[i];
}

void
kfm_vector_set(kfm_vector* v, unsigned int i, double x)
{
    v->data[i] = x;
}

void
kfm_vector_set_zero(kfm_vector* v)
{
    for(unsigned int i=0; i<v->size; i++)
    {
        v->data[i] = 0.;
    }
}

void
kfm_vector_set(const kfm_vector* src, kfm_vector* dest)
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
        kfmout << "kfm_vector_set: error, vectors have difference sizes: "<<src->size<<" != "<<dest->size<<"."<<kfmendl;
        kfmexit(1);
    }
}


//operations
void
kfm_vector_scale(kfm_vector* a, double scale_factor)
{
    for(unsigned int i=0; i<a->size; i++)
    {
        a->data[i] *= scale_factor;
    }
}

void
kfm_vector_sub(kfm_vector* a, const kfm_vector* b)
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
        kfmout << "kfm_vector_sub: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<"."<< kfmendl;
        kfmexit(1);
    }
}

void
kfm_vector_add(kfm_vector* a, const kfm_vector* b)
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
        kfmout << "kfm_vector_add: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<"."<< kfmendl;
        kfmexit(1);
    }
}


double
kfm_vector_inner_product(const kfm_vector* a, const kfm_vector* b)
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
        kfmout << "kfm_inner_product: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<"."<< kfmendl;
        kfmexit(1);
    }

    return val;
}

double
kfm_vector_norm(const kfm_vector* a)
{
    double val = kfm_vector_inner_product(a,a);
    return std::sqrt(val);
}

#endif

////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

void
kfm_vector_normalize(kfm_vector* a)
{
    double scale = kfm_vector_norm(a);
    kfm_vector_scale(a, 1.0/scale);
}

void
kfm_vector_sub(const kfm_vector* a, const kfm_vector* b, kfm_vector* c)
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
        kfmout << "kfm_inner_sub: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<" != "<<c->size<<"."<< kfmendl;
        kfmexit(1);
    }
}

void
kfm_vector_add(const kfm_vector* a, const kfm_vector* b, kfm_vector* c)
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
        kfmout << "kfm_inner_add: error, vectors have difference sizes: "<<a->size<<" != "<<b->size<<" != "<<c->size<<"."<< kfmendl;
        kfmexit(1);
    }
}


void
kfm_vector_cross_product(const kfm_vector* a, const kfm_vector* b, kfm_vector* c)
{
    kfm_vector_set(c, 0, kfm_vector_get(a,1)*kfm_vector_get(b,2) - kfm_vector_get(a,2)*kfm_vector_get(b,1));
    kfm_vector_set(c, 1, kfm_vector_get(a,2)*kfm_vector_get(b,0) - kfm_vector_get(a,0)*kfm_vector_get(b,2));
    kfm_vector_set(c, 2, kfm_vector_get(a,0)*kfm_vector_get(b,1) - kfm_vector_get(a,1)*kfm_vector_get(b,0));
}

////////////////////////////////////////////////////////////////////////////////


}
