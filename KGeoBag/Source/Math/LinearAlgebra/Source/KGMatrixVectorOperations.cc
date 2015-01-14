#include "KGVectorOperations.hh"
#include "KGMatrixOperations.hh"
#include "KGMatrixVectorOperations.hh"

#include <sstream>
#include "KGMathMessage.hh"

namespace KGeoBag
{

#ifdef KGEOBAG_MATH_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use fast BLAS based implementation

void kg_matrix_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out)
{
    gsl_blas_dgemv(CblasNoTrans, 1.0, m, in, 0.0, out);
}

void kg_matrix_transpose_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out)
{
    gsl_blas_dgemv(CblasTrans, 1.0, m, in, 0.0, out);
}

#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

void kg_matrix_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out)
{
    //check sizes
    if( (in->size == m->size2) && (out->size == m->size1) )
    {

        double elem;
        for(size_t i=0; i<m->size1; i++)
        {
            elem = 0.0;

            for(size_t j=0; j<m->size2; j++)
            {
                elem += kg_matrix_get(m,i,j) * kg_vector_get(in, j);
            }

            kg_vector_set(out, i, elem);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_vector_product: error, matrix/vector sizes are mismatched. \n";
        ss << "matrix m is "<<m->size1<<" by "<<m->size2<<".\n";
        ss << "input vector has size "<<in->size<<".\n";
        ss << "output vector has size "<<out->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


void kg_matrix_transpose_vector_product(const kg_matrix* m, const kg_vector* in, kg_vector* out)
{
    //check sizes
    if( (in->size == m->size1) && (out->size == m->size2) )
    {

        double elem;
        for(size_t i=0; i<m->size2; i++)
        {
            elem = 0.0;

            for(size_t j=0; j<m->size1; j++)
            {
                elem += kg_matrix_get(m,j,i) * kg_vector_get(in, j);
            }

            kg_vector_set(out, i, elem);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_transpose_vector_product: error, matrix/vector sizes are mismatched.\n";
        ss << "transpose of matrix m is "<<m->size2<<" by "<<m->size1<<".\n";
        ss << "input vector has size "<<in->size<<".\n";
        ss << "output vector has size "<<out->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

#endif

////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

void kg_sparse_matrix_vector_product(const kg_sparse_matrix* m, const kg_vector* in, kg_vector* out)
{
    //check sizes
    if( (in->size == m->size2) && (out->size == m->size1) )
    {
        kg_vector_set_zero(out);

        double in_val;
        double mx_val;
        size_t row;
        size_t col;

        for(size_t i=0; i < m->n_elements; i++)
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
        std::stringstream ss;
        ss << "kg_sparse_matrix_vector_product: error, matrix/vector sizes are mismatched.\n";
        ss << "matrix m is "<<m->size1<<" by "<<m->size2<<".\n";
        ss << "input vector has size "<<in->size<<".\n";
        ss << "output vector has size "<<out->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}



void kg_vector_outer_product(const kg_vector* a, const kg_vector* b, kg_matrix* p)
{
    //check sizes
    if( (a->size == p->size1) && ( b->size == p->size2) &&  (a->size == b->size) )
    {
        double elem;
        for(size_t i=0; i<p->size1; i++)
        {
            for(size_t j=0; j<p->size2; j++)
            {
                elem = ( kg_vector_get(a,i) )*( kg_vector_get(b,j) );
                kg_matrix_set(p, i, j, elem);
            }
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_vector_outer_product: error, matrix/vector sizes are mismatched.\n";
        ss << "output matrix p is "<<p->size1<<" by "<<p->size2<<".\n";
        ss << "input vector a has size "<<a->size<<".\n";
        ss << "input vector b has size "<<b->size<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


}
