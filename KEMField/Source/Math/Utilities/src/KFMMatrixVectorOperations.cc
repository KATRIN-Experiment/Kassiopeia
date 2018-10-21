#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMMessaging.hh"

namespace KEMField
{

#ifdef KEMFIELD_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use fast BLAS based implementation

void kfm_matrix_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out)
{
    gsl_blas_dgemv(CblasNoTrans, 1.0, m, in, 0.0, out);
}

void kfm_matrix_transpose_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out)
{
    gsl_blas_dgemv(CblasTrans, 1.0, m, in, 0.0, out);
}

#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

void kfm_matrix_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out)
{
    //check sizes
    if( (in->size == m->size2) && (out->size == m->size1) )
    {

        double elem;
        for(unsigned int i=0; i<m->size1; i++)
        {
            elem = 0.0;

            for(unsigned int j=0; j<m->size2; j++)
            {
                elem += kfm_matrix_get(m,i,j) * kfm_vector_get(in, j);
            }
            kfm_vector_set(out, i, elem);
        }
    }
    else
    {
        kfmout << "kfm_matrix_vector_product: error, matrix/vector sizes are mismatched."<<kfmendl;
        kfmout << "matrix m is "<<m->size1<<" by "<<m->size2<<"."<<kfmendl;
        kfmout << "input vector has size "<<in->size<<"."<<kfmendl;
        kfmout << "output vector has size "<<out->size<<"."<<kfmendl;
        kfmexit(1);
    }
}


void kfm_matrix_transpose_vector_product(const kfm_matrix* m, const kfm_vector* in, kfm_vector* out)
{
    //check sizes
    if( (in->size == m->size1) && (out->size == m->size2) )
    {

        double elem;
        for(unsigned int i=0; i<m->size2; i++)
        {
            elem = 0.0;

            for(unsigned int j=0; j<m->size1; j++)
            {
                elem += kfm_matrix_get(m,j,i) * kfm_vector_get(in, j);
            }

            kfm_vector_set(out, i, elem);
        }
    }
    else
    {
        kfmout << "kfm_matrix_transpose_vector_product: error, matrix/vector sizes are mismatched."<<kfmendl;
        kfmout << "transpose of matrix m is "<<m->size2<<" by "<<m->size1<<"."<<kfmendl;
        kfmout << "input vector has size "<<in->size<<"."<<kfmendl;
        kfmout << "output vector has size "<<out->size<<"."<<kfmendl;
        kfmexit(1);
    }
}

#endif

////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

void kfm_sparse_matrix_vector_product(const kfm_sparse_matrix* m, const kfm_vector* in, kfm_vector* out)
{
    //check sizes
    if( (in->size == m->size2) && (out->size == m->size1) )
    {
        kfm_vector_set_zero(out);

        double in_val;
        double mx_val;
        unsigned int row;
        unsigned int col;

        for(unsigned int i=0; i < m->n_elements; i++)
        {
            row = (m->row)[i];
            col = (m->column)[i];
            in_val = (in->data)[col];
            mx_val = (m->data)[i];
            (out->data)[row] += in_val*mx_val;
        }
    }
    else
    {
        kfmout << "kfm_sparse_matrix_vector_product: error, matrix/vector sizes are mismatched."<<kfmendl;
        kfmout << "matrix m is "<<m->size1<<" by "<<m->size2<<"."<<kfmendl;
        kfmout << "input vector has size "<<in->size<<"."<<kfmendl;
        kfmout << "output vector has size "<<out->size<<"."<<kfmendl;
        kfmexit(1);
    }
}



void kfm_vector_outer_product(const kfm_vector* a, const kfm_vector* b, kfm_matrix* p)
{
    //check sizes
    if( (a->size == p->size1) && ( b->size == p->size2) &&  (a->size == b->size) )
    {
        double elem;
        for(unsigned int i=0; i<p->size1; i++)
        {
            for(unsigned int j=0; j<p->size2; j++)
            {
                elem = ( kfm_vector_get(a,i) )*( kfm_vector_get(b,j) );
                kfm_matrix_set(p, i, j, elem);
            }
        }
    }
    else
    {
        kfmout << "kfm_vector_outer_product: error, matrix/vector sizes are mismatched."<<kfmendl;
        kfmout << "output matrix p is "<<p->size1<<" by "<<p->size2<<"."<<kfmendl;
        kfmout << "input vector a has size "<<a->size<<"."<<kfmendl;
        kfmout << "input vector b has size "<<b->size<<"."<<kfmendl;
        kfmexit(1);
    }
}


}
