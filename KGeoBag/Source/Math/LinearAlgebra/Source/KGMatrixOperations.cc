#include "KGVectorOperations.hh"
#include "KGMatrixOperations.hh"
#include "KGMatrixVectorOperations.hh"

#include <sstream>

#include "KGMathMessage.hh"

namespace KGeoBag
{

#ifdef KGEOBAG_MATH_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use the fast BLAS based implementation

kg_matrix*
kg_matrix_alloc(unsigned int nrows, unsigned int ncolumns)
{
    return gsl_matrix_alloc(nrows, ncolumns);
}

kg_matrix* kg_matrix_calloc(unsigned int nrows, unsigned int ncolumns)
{
    return gsl_matrix_calloc(nrows, ncolumns);
}

void kg_matrix_free(kg_matrix* m)
{
    gsl_matrix_free(m);
}

double kg_matrix_get(const kg_matrix* m, unsigned int i, unsigned int j)
{
    return gsl_matrix_get(m, i, j);
}

void kg_matrix_set(kg_matrix* m, unsigned int i, unsigned int j, double x)
{
    gsl_matrix_set(m, i, j,x);
}

void kg_matrix_set_zero(kg_matrix* m)
{
    gsl_matrix_set_zero(m);
}

void
kg_matrix_set_identity(kg_matrix* m)
{
    gsl_matrix_set_identity(m);
}

void
kg_matrix_set(const gsl_matrix* src, gsl_matrix* dest)
{
    gsl_matrix_memcpy(dest, src);
}

void kg_matrix_sub(kg_matrix* a, const kg_matrix* b)
{
    gsl_matrix_sub(a,b);
}

void kg_matrix_add(kg_matrix* a, const kg_matrix* b)
{
    gsl_matrix_add(a,b);
}

void kg_matrix_scale(kg_matrix* a, double scale_factor)
{
    gsl_matrix_scale(a, scale_factor);
}


void kg_matrix_multiply(const kg_matrix* A, const kg_matrix* B, kg_matrix* C)
{
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


void
kg_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kg_matrix* A, const kg_matrix* B, kg_matrix* C)
{
    if(!transposeA && !transposeB)
    {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
    }

    if(transposeA && !transposeB)
    {
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
    }

    if(!transposeA && transposeB)
    {
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, A, B, 0.0, C);
    }

    if(transposeA && transposeB)
    {
        gsl_blas_dgemm(CblasTrans, CblasTrans, 1.0, A, B, 0.0, C);
    }
}

//computes the singular value decomposition of the square matrix A = U*diag(S)*V^T
void
kg_matrix_svd(const kg_matrix* A, kg_matrix* U, kg_vector* S, kg_matrix* V)
{
    kg_vector* work = kg_vector_alloc(A->size1);
    kg_matrix_set(A, U); //copy A into U
    gsl_linalg_SV_decomp(U, V, S, work);
    kg_vector_free(work);
}

//given the singular value decomposition of the matrix A = U*diag(S)*V^T, this function solves the equation Ax = b
void
kg_matrix_svd_solve(const kg_matrix* U, const kg_vector* S, const kg_matrix* V, const kg_vector* b, kg_vector* x)
{
    gsl_linalg_SV_solve(U, V, S, b, x);
}

#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

kg_matrix*
kg_matrix_alloc(unsigned int nrows, unsigned int ncolumns)
{
    kg_matrix* m = new kg_matrix();
    m->size1 = nrows;
    m->size2 = ncolumns;
    m->data = new double[nrows*ncolumns];
    return m;
}

kg_matrix* kg_matrix_calloc(unsigned int nrows, unsigned int ncolumns)
{
    kg_matrix* m = new kg_matrix();
    m->size1 = nrows;
    m->size2 = ncolumns;
    unsigned int total_size = nrows*ncolumns;
    double* d = new double[total_size];
    for(unsigned int i=0; i<total_size; i++)
    {
        d[i] = 0.;
    }
    m->data = d;
    return m;
}

void kg_matrix_free(kg_matrix* m)
{
    delete[] m->data;
    delete m;
}

double kg_matrix_get(const kg_matrix* m, unsigned int i, unsigned int j)
{
    unsigned int index = i*(m->size2) + j;
    return m->data[index];
}

void kg_matrix_set(kg_matrix* m, unsigned int i, unsigned int j, double x)
{
    unsigned int index = i*(m->size2) + j;
    m->data[index] = x;
}

void kg_matrix_set_zero(kg_matrix* m)
{
    unsigned int total_size = (m->size1)*(m->size2);
    double* d = m->data;
    for(unsigned int i=0; i<total_size; i++)
    {
        d[i] = 0.;
    }
}

void
kg_matrix_set_identity(kg_matrix* m)
{
    kg_matrix_set_zero(m);
    unsigned int min;
    if(m->size1 < m->size2)
    {
        min = m->size1;
    }
    else
    {
        min = m->size2;
    }
    for(unsigned int i=0; i<min; i++)
    {
        kg_matrix_set(m,i,i,1.0);
    }
}

void
kg_matrix_set(const kg_matrix* src, kg_matrix* dest)
{
    if( (src->size1 == dest->size1) && (src->size2 == dest->size2) )
    {
        unsigned int total_size = (src->size1)*(src->size2);
        for(unsigned int i=0; i<total_size; i++)
        {
            dest->data[i] = src->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_set: error, matrices have difference sizes. \n";
        ss << "source matrix is "<<src->size1<<" by "<<src->size2<<". \n";
        ss << "destination matrix is "<<dest->size1<<" by "<<dest->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

void kg_matrix_sub(kg_matrix* a, const kg_matrix* b)
{
    if(a->size1 == b->size1 && a->size2 == b->size2)
    {
        unsigned int total_size = (a->size1)*(a->size2);
        for(unsigned int i=0; i<total_size; i++)
        {
            a->data[i] -= b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_sub: error, matrices have difference sizes.\n";
        ss << "matrix a is "<<a->size1<<" by "<<a->size2<<".\n";
        ss << "matrix b is "<<b->size1<<" by "<<b->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

void kg_matrix_add(kg_matrix* a, const kg_matrix* b)
{
    if(a->size1 == b->size1 && a->size2 == b->size2)
    {
        unsigned int total_size = (a->size1)*(a->size2);
        for(unsigned int i=0; i<total_size; i++)
        {
            a->data[i] += b->data[i];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_add: error, matrices have difference sizes.\n";
        ss << "matrix a is "<<a->size1<<" by "<<a->size2<<".\n";
        ss << "matrix b is "<<b->size1<<" by "<<b->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}

void kg_matrix_scale(kg_matrix* a, double scale_factor)
{
    unsigned int total_size = (a->size1)*(a->size2);
    for(unsigned int i=0; i<total_size; i++)
    {
        a->data[i] *= scale_factor;
    }
}

void kg_matrix_multiply(const kg_matrix* A, const kg_matrix* B, kg_matrix* C)
{
    //check that sizes are valid
    unsigned int a_row = A->size1;
    unsigned int a_col = A->size2;

    unsigned int b_row = B->size1;
    unsigned int b_col = B->size2;

    unsigned int c_row = C->size1;
    unsigned int c_col = C->size2;

    if( (a_col == b_row) && (c_row == a_row) && (c_col == b_col) )
    {
        //perform super slow naive direct O(N^3) matrix multiplication
        //this will likely be much slower than the GSL/BLAS implemention
        for(unsigned int i=0; i<c_row; i++)
        {
            for(unsigned int j=0; j<c_col; j++)
            {
                double elem = 0.0;

                for(unsigned int offset=0; offset<b_row; offset++)
                {
                    elem += ( kg_matrix_get(A, i, offset) )*( kg_matrix_get(B, offset, j) );
                }

                kg_matrix_set(C, i, j, elem);
            }
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_multiply: error, matrices have difference sizes.\n";
        ss << "matrix a is "<<A->size1<<" by "<<A->size2<<".\n";
        ss << "matrix b is "<<B->size1<<" by "<<B->size2<<".\n";
        ss << "matrix c is "<<C->size1<<" by "<<C->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


void
kg_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kg_matrix* A, const kg_matrix* B, kg_matrix* C)
{
    //check that sizes are valid
    unsigned int a_row = A->size1;
    unsigned int a_col = A->size2;

    if(transposeA)
    {
        unsigned int swap = a_col;
        a_col = a_row;
        a_row = swap;
    }

    unsigned int b_row = B->size1;
    unsigned int b_col = B->size2;

    if(transposeB)
    {
        unsigned int swap = b_col;
        b_col = b_row;
        b_row = swap;
    }

    unsigned int c_row = C->size1;
    unsigned int c_col = C->size2;


    unsigned int ai, aj, bi, bj;

    if( (a_col == b_row) && (c_row == a_row) && (c_col == b_col) )
    {
        //perform super slow naive direct O(N^3) matrix multiplication
        //this will likely be at least 100x slower than the GSL/BLAS implemention
        for(unsigned int i=0; i<c_row; i++)
        {
            for(unsigned int j=0; j<c_col; j++)
            {
                double elem = 0.0;

                for(unsigned int offset=0; offset<b_row; offset++)
                {
                    if(transposeA)
                    {
                        ai = offset; aj = i;
                    }
                    else
                    {
                        ai = i; aj = offset;
                    }


                    if(transposeB)
                    {
                        bi = j; bj = offset;
                    }
                    else
                    {
                        bi = offset; bj = j;
                    }

                    elem += ( kg_matrix_get(A, ai, aj) )*( kg_matrix_get(B, bi, bj) );
                }

                kg_matrix_set(C, i, j, elem);
            }
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_multiply_with_transpose: error, matrices have difference sizes.\n";
        ss << "matrix a is "<<A->size1<<" by "<<A->size2<<".\n";
        ss << "matrix b is "<<B->size1<<" by "<<B->size2<<".\n";
        ss << "matrix c is "<<C->size1<<" by "<<C->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


void
kg_matrix_svd(const kg_matrix* A, kg_matrix* U, kg_vector* S, kg_matrix* V)
{
    //this function uses the slower but more accurate one-sided jacobi svd
    //as defined in the paper:
    //Jacobi's method is more accurate than QR by J. Demmel and K. Veselic
    //SIAM. J. Matrix Anal. & Appl., 13(4), 1204-1245.
    //www.netlib.org/lapack/lawnspdf/lawn15.pdf

    //assume that A is m x n
    //then U is an m x n
    //V is n x n
    //S is length n


    unsigned int n = A->size1;
    unsigned int m = A->size2;

    if( U->size1 != n || U->size2 != m )
    {
        std::stringstream ss;
        ss << "kg_matrix_svd: error, matrices A and U have different sizes.\n";
        ss << "matrix A is "<<A->size1<<" by "<<A->size2<<".\n";
        ss << "matrix U is "<<U->size1<<" by "<<U->size2<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }

    if( n != V->size1 || n != V->size1 )
    {
        std::stringstream ss;
        ss << "kg_matrix_svd: error, matrix V has wrong size.\n";
        ss << "matrix V is "<<V->size1<<" by "<<V->size2<<".\n";
        ss << "matrix V should be "<<n<<" by "<<n<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }

    if(S->size != n)
    {
        std::stringstream ss;
        ss << "kg_matrix_svd: error, vector S has wrong size.\n";
        ss << "vector S is length "<<S->size<<".\n";
        ss << "vector S should be length "<<n<<".\n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }

    //copy A into U
    kg_matrix_set(A,U);
    //set V to the identity
    kg_matrix_set_identity(V);

    //put some limits on the number of iterations, this is completely arbitrary
    int n_iter = 0;
    int n_max_iter = 10*n;

    //scratch space
    double a, b, c, g1, g2, cs, sn, t, psi, sign;

    //make a very rough empirical estimate of the appropriate tolerance
    double tol = 0;
    for(unsigned int i=0; i<n; i++)
    {
        for(unsigned int j=0; j<n; j++)
        {
            g1 = kg_matrix_get(U,i,j);
            tol += g1*g1;
        }
    }
    tol = tol*n*m*KG_EPSILON*KG_EPSILON;

    //convergence count
    int count = 0;
    do
    {
        count = 0;

        //for all column pairs i < j < n
        for(unsigned int i=0; i<n-1; i++)
        {
            for(unsigned int j=i+1; j<n; j++)
            {
                //add to the convergence count
                count++;

                //compute the a,b,c submatrix
                a = 0;
                b = 0;
                c = 0;

                for(unsigned int k=0; k<m; k++)
                {
                    g1 = kg_matrix_get(U, k, i);
                    g2 = kg_matrix_get(U, k, j);
                    a += g1*g1;
                    b += g2*g2;
                    c += g1*g2;
                }

                if( (c*c)/(a*b) > tol )
                {
                    //compute the sine/cosine of the jacobi rotation
                    psi = (b-a)/(2.0*c);

                    sign = 1.0;
                    if( psi < 0 )
                    {
                        sign = -1.0;
                    }
                    else
                    {
                        sign = 1.0;
                    }

                    t = ( sign )/( std::fabs(psi) + std::sqrt(1.0 + psi*psi) );
                    cs = 1.0/std::sqrt(1.0 + t*t);
                    sn = cs*t;

                    //apply the rotation to U and V
                    for(unsigned int k=0; k<m; k++)
                    {
                        //apply to U
                        g1 = kg_matrix_get(U, k, i);
                        g2 = kg_matrix_get(U, k, j);
                        kg_matrix_set(U, k, i, cs*g1 - sn*g2);
                        kg_matrix_set(U, k, j, sn*g1 + cs*g2);

                        //apply to V
                        g1 = kg_matrix_get(V, k, i);
                        g2 = kg_matrix_get(V, k, j);
                        kg_matrix_set(V, k, i, cs*g1 - sn*g2);
                        kg_matrix_set(V, k, j, sn*g1 + cs*g2);
                    }
                }
                else
                {
                    //subtract from convergence count
                    count--;
                }

            }
        }

        n_iter++;

        if(n_iter >= n_max_iter)
        {
            std::stringstream ss;
            ss << "kg_matrix_svd: warning, singular value decomposition failed to converge within "<<n_max_iter<<" iterations. \n";
            mathmsg( eDebug ) << ss.str().c_str() << eom;
            break;
        }

    }
    while( count > 0 );

    //now we compute the singluar values, they are the norms of the columns of U
    //we also compute the norm of all the singular values
    double norm_s = 0.0;
    for(unsigned int i=0; i<n; i++)
    {
        a = 0;
        for(unsigned int j=0; j<m; j++)
        {
            g1 = kg_matrix_get(U,j,i);
            a += g1*g1;
        }
        norm_s += a;
        a = std::sqrt(a);
        kg_vector_set(S,i,a);
    }

    norm_s = std::sqrt(norm_s);

    tol = KG_EPSILON*norm_s;

    //eliminate all those singular values which are below the tolerance
    for(unsigned int i=0; i<n; i++)
    {
        if(kg_vector_get(S,i) < tol)
        {
            kg_vector_set(S,i,0.0);
        }
    }

    //now we fix U by post multiplying with the inverse of diag(S)
    for(unsigned int i=0; i<m; i++) //rows
    {
        for(unsigned int j=0; j<n; j++) //col
        {
            g1 = kg_matrix_get(U,i,j);
            g2 = kg_vector_get(S,j);
            if(g2 == 0.0)
            {
                kg_matrix_set(U,i,j, 0.0);//set to zero
            }
            else
            {
                kg_matrix_set(U,i,j, g1/g2 );
            }
        }
    }
}



void
kg_matrix_svd_solve(const kg_matrix* U, const kg_vector* S, const kg_matrix* V, const kg_vector* b, kg_vector* x)
{
    //the solution is given by:
    //x = [V*diag(S)^{-1}*U^{T}]b

    //workspace
    kg_vector* work = kg_vector_alloc(x->size);

    //first we copy b into x and apply U^T
    kg_vector_set(b, x);
    kg_matrix_transpose_vector_product(U, x, work);
    kg_vector_set(work, x);

    //now we apply the inverse of diag(S) to x
    //with the exception that if a singular value is zero then we apply zero
    //we assume anything less than KG_EPSILON*norm(S) to be essentially zero (singular values should all be positive)
    double s, elem;
    double norm_s = kg_vector_norm(S);
    for(unsigned int i=0; i<S->size; i++)
    {
        s = kg_vector_get(S,i);
        if(s > KG_EPSILON*norm_s)
        {
            //multiply 1/s against the i'th element of x
            elem = (1.0/s)*kg_vector_get(x,i);
            kg_vector_set(x,i,elem);
        }
        else
        {
            kg_vector_set(x,i,0.0);
        }
    }

    //finally we apply the matrix V to the vector x
    kg_matrix_vector_product(V, x, work);
    kg_vector_set(work, x);


    //free workspace
    kg_vector_free(work);

}

#endif


////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

kg_sparse_matrix*
kg_sparse_matrix_alloc(unsigned int nrows, unsigned int ncolumns, unsigned int n_elements)
{
    kg_sparse_matrix* m = new kg_sparse_matrix();
    m->size1 = nrows;
    m->size2 = ncolumns;
    m->data = new double[n_elements];
    m->row = new unsigned int[n_elements];
    m->column = new unsigned int[n_elements];
    return m;

}

void
kg_sparse_matrix_free(kg_sparse_matrix* m)
{
    delete[] m->data;
    delete[] m->row;
    delete[] m->column;
}


double
kg_sparse_matrix_get(const kg_sparse_matrix* m, unsigned int i, unsigned int j)
{
    for(unsigned int n=0; n<m->n_elements; n++)
    {
        if( (i == (m->row)[n]) && (j == (m->column)[n]) )
        {
            return (m->data)[n];
        }
    }

    return 0;

}

void
kg_sparse_matrix_set(kg_sparse_matrix* m, unsigned int i, unsigned int j, unsigned int element_index, double x)
{
    (m->data)[element_index] = x;
    (m->row)[element_index] = i;
    (m->column)[element_index] = j;
}


void kg_matrix_transpose(const kg_matrix* in, kg_matrix* out)
{
    for(unsigned int row=0; row<in->size1; row++)
    {
        for(unsigned int col=0; col<in->size2; col++)
        {
            kg_matrix_set(out, col, row, kg_matrix_get(in, row, col) );
        }
    }
}


void kg_matrix_euler_angles_ZYZ(const kg_matrix* R, double& alpha, double& beta, double& gamma, double& tol)
{
    //check that R is 3x3
    if(R->size1 == 3 && R->size2 == 3)
    {

        //calloc temporary matrix C
        kg_matrix* C = kg_matrix_alloc(3,3);

        //multiply the matrix by its inverse (transpose)
        kg_matrix_multiply_with_transpose(false, true, R, R, C);

        //compute the L2 norm of the difference from the identity
        //and return in tol
        tol = 0.0;
        double temp;
        for(unsigned int i=0; i<3; i++)
        {
            for(unsigned int j=0; j<3; j++)
            {
                if(i != j)
                {
                    temp = kg_matrix_get(C,i,j);
                }
                else
                {
                    temp =  kg_matrix_get(C,i,j) - 1.0;
                }
                tol += temp*temp;
            }
        }
        tol = std::sqrt(tol + KG_EPSILON*KG_EPSILON);


        //the angles that are computed are not unique but represent one possible
        //set of angles that construct the rotation matrix
        bool isDegenerate = false;

        //if  |1-|cos(beta)| | < tol we are in the degenerate case
        if( std::fabs( 1. - std::fabs( kg_matrix_get(R,2,2) ) ) <= tol)
        {
            isDegenerate = true;
        }

        if(!isDegenerate)
        {
            beta = std::acos( kg_matrix_get(R,2,2) );
            alpha = std::atan2( (-1.0* kg_matrix_get(R,2,1) )/std::sin(beta), ( kg_matrix_get(R,2,0))/std::sin(beta) );
            gamma = std::atan2( (-1.0* kg_matrix_get(R,1,2) )/std::sin(beta), -1*( kg_matrix_get(R,0,2))/std::sin(beta)  ) ;
        }
        else
        {
            if( std::fabs(1. -  kg_matrix_get(R,2,2) ) <= tol)
            {
                alpha =  std::atan2(  kg_matrix_get(R,1,0),  kg_matrix_get(R,0,0) );
                beta = 0;
                gamma = 0;
            }
            else if( std::fabs(1. +  kg_matrix_get(R,2,2) ) <= tol)
            {
                alpha = std::atan2(  kg_matrix_get(R,0,1),  kg_matrix_get(R,1,1) );
                beta = M_PI;
                gamma = 0;
            }
            else
            {
                //either no solution found, or R is the identity!
                alpha = 0;
                beta = 0;
                gamma = 0;
            }
        }

        kg_matrix_free(C);
    }

}



void kg_matrix_from_euler_angles_ZYZ(kg_matrix* R, double alpha, double beta, double gamma)
{
    double sin_a = std::sin(alpha);
    double cos_a = std::cos(alpha);

    double sin_b = std::sin(beta);
    double cos_b = std::cos(beta);

    double sin_c = std::sin(gamma);
    double cos_c = std::cos(gamma);

    kg_matrix_set_identity(R);

    kg_matrix* A = kg_matrix_alloc(3,3);
    kg_matrix* B = kg_matrix_alloc(3,3);
    kg_matrix* C = kg_matrix_alloc(3,3);

    kg_matrix_set_identity(A);
    kg_matrix_set_identity(B);
    kg_matrix_set_identity(C);

    kg_matrix_set(A, 0, 0, cos_a);
    kg_matrix_set(A, 1, 1, cos_a);
    kg_matrix_set(A, 0, 1, -sin_a);
    kg_matrix_set(A, 1, 0, sin_a);

    kg_matrix_set(B, 0, 0, cos_b);
    kg_matrix_set(B, 2, 2, cos_b);
    kg_matrix_set(B, 0, 2, -sin_b);
    kg_matrix_set(B, 2, 0, sin_b);

    kg_matrix_set(C, 0, 0, cos_c);
    kg_matrix_set(C, 1, 1, cos_c);
    kg_matrix_set(C, 0, 1, -sin_c);
    kg_matrix_set(C, 1, 0, sin_c);

    kg_matrix_set(A,R);

    kg_matrix_multiply(B,R,A);
    kg_matrix_multiply(C,A,R);

    kg_matrix_free(A);
    kg_matrix_free(B);
    kg_matrix_free(C);
}


void kg_matrix_from_axis_angle(kg_matrix* R, double cos_angle, double sin_angle, const kg_vector* axis)
{
    kg_vector_outer_product(axis, axis, R);

    double c = cos_angle;
    double s = sin_angle;
    double t = 1.0 - c;

    for(unsigned int i=0; i<3; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            kg_matrix_set( R, i, j, t*kg_matrix_get(R, i, j) );
        }
    }

    for(unsigned int i=0; i<3; i++)
    {
        kg_matrix_set( R, i, i, kg_matrix_get(R, i, i) + c);
    }

    kg_matrix_set( R, 0, 1, kg_matrix_get(R, 0, 1) - s*kg_vector_get(axis, 2));
    kg_matrix_set( R, 0, 2, kg_matrix_get(R, 0, 2) + s*kg_vector_get(axis, 1));
    kg_matrix_set( R, 1, 2, kg_matrix_get(R, 1, 2) - s*kg_vector_get(axis, 0));

    kg_matrix_set( R, 1, 0, kg_matrix_get(R, 1, 0) + s*kg_vector_get(axis, 2));
    kg_matrix_set( R, 2, 0, kg_matrix_get(R, 2, 0) - s*kg_vector_get(axis, 1));
    kg_matrix_set( R, 2, 1, kg_matrix_get(R, 2, 1) + s*kg_vector_get(axis, 0));


}

void kg_matrix_from_axis_angle(kg_matrix* R, double angle, const kg_vector* axis)
{
    kg_vector_outer_product(axis, axis, R);

    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;

    for(unsigned int i=0; i<3; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            kg_matrix_set( R, i, j, t*kg_matrix_get(R, i, j) );
        }
    }

    for(unsigned int i=0; i<3; i++)
    {
        kg_matrix_set( R, i, i, kg_matrix_get(R, i, i) + c);
    }

    kg_matrix_set( R, 0, 1, kg_matrix_get(R, 0, 1) - s*kg_vector_get(axis, 2));
    kg_matrix_set( R, 0, 2, kg_matrix_get(R, 0, 2) + s*kg_vector_get(axis, 1));
    kg_matrix_set( R, 1, 2, kg_matrix_get(R, 1, 2) - s*kg_vector_get(axis, 0));

    kg_matrix_set( R, 1, 0, kg_matrix_get(R, 1, 0) + s*kg_vector_get(axis, 2));
    kg_matrix_set( R, 2, 0, kg_matrix_get(R, 2, 0) - s*kg_vector_get(axis, 1));
    kg_matrix_set( R, 2, 1, kg_matrix_get(R, 2, 1) + s*kg_vector_get(axis, 0));
}



void
kg_matrix_householder_bidiagonalize(const kg_matrix* A, kg_matrix* P, kg_matrix* J, kg_matrix* Q)
{
    //take the matrix A and bidiagonalize it A = PJQ^T, where J is the upper bidiagonal matrix
    //uses householder transformations as stated by Golub-Kahan in
    //Calculating the singular values and pseudo-inverse of a matrix. J. SIAM Numer. Anal. Ser. B, Vol 2, No. 2 1965

    unsigned int n = A->size1;

    if(A->size1 == A->size2)
    {
        if(P->size1 != n || P->size2 != n || J->size1 != n || J->size2 != n || Q->size1 != n || Q->size2 != n )
        {
            std::stringstream ss;
            ss << "kg_matrix_householder_bidiagonalize: error, matrices A, P, J, or Q have different sizes. \n";
            ss << "matrix A is "<<A->size1<<" by "<<A->size2<<".\n";
            ss << "matrix P is "<<P->size1<<" by "<<P->size2<<".\n";
            ss << "matrix J is "<<J->size1<<" by "<<J->size2<<".\n";
            ss << "matrix Q is "<<Q->size1<<" by "<<Q->size2<<".\n";
            mathmsg( eDebug ) << ss.str().c_str() << eom;
        }

        //copy A into J
        kg_matrix_set(A,J);
        //set P and Q to identity
        kg_matrix_set_identity(P);
        kg_matrix_set_identity(Q);

        //allocate workspace
        kg_vector* x = kg_vector_alloc(n);
        kg_matrix* H = kg_matrix_alloc(n,n);
        kg_matrix* W = kg_matrix_alloc(n,n);
        double sk, tk, ck, dk, ak, signk;

        for(unsigned int k=0; k < n; k++)
        {
            //take care of the column first/////////////////////////////////////
            //copy the k-th column into x while zero-ing out all components where j<k
            for(unsigned int j=0; j<n; j++)
            {
                if(j < k)
                {
                    kg_vector_set(x, j, 0.0);
                }
                else
                {
                    kg_vector_set(x, j, kg_matrix_get(J, j, k) );
                }
            }

            //get a_k,k
            ak = kg_matrix_get(J, k, k);
            //compute signum of a_k,k
            signk = ak/(std::fabs(ak));
            //compute norm of x to get 's_k' value
            sk = kg_vector_norm(x);
            //compute the 'c_k' value
            ck = 1.0/(2.0*sk*signk*std::sqrt( 0.5*(1.0 + std::fabs(ak)/sk) ) );

            //now we compute the values of x
            for(unsigned int j=k; j<n; j++)
            {
                if(j == k)
                {
                    kg_vector_set(x, j, std::sqrt( 0.5*(1.0 + std::fabs(ak)/sk) ) );
                }
                else
                {
                    kg_vector_set(x, j, ck*kg_vector_get(x,j) );
                }
            }

            //compute the householder matrix H
            kg_matrix_householder(H,x);

            //pre-apply H to J, and post-apply H to P
            kg_matrix_multiply(H, J, W);
            kg_matrix_set(W, J);

            kg_matrix_multiply(P, H, W);
            kg_matrix_set(W, P);

            //next take care of the row/////////////////////////////////////////
            //copy the k-th row into x while zero-ing out all components where j<k
            if(k < n-1) //only need to do this if it is not the last column
            {

                for(unsigned int j=0; j<n; j++)
                {
                    if(j <= k)
                    {
                        kg_vector_set(x, j, 0.0);
                    }
                    else
                    {
                        kg_vector_set(x, j, kg_matrix_get(J, k, j) );
                    }
                }

                //get a_k,k+1
                ak = kg_matrix_get(J, k, k+1);
                //compute signum of a_k,k+1
                signk = ak/(std::fabs(ak));
                //compute norm of x to get 't_k' value
                tk = kg_vector_norm(x);
                //compute the 'd_k' value
                dk = 1.0/(2.0*tk*signk*std::sqrt( 0.5*(1.0 + std::fabs(ak)/tk) ) );

                //now we compute the values of x
                for(unsigned int j=k+1; j<n; j++)
                {
                    if(j == k+1)
                    {
                        kg_vector_set(x, j, std::sqrt( 0.5*(1.0 + std::fabs(ak)/tk) ) );
                    }
                    else
                    {
                        kg_vector_set(x, j, dk*kg_vector_get(x,j) );
                    }
                }

                //compute the householder matrix H
                kg_matrix_householder(H,x);

                //post-apply H to J, and pre-apply H to Q
                kg_matrix_multiply(J, H, W);
                kg_matrix_set(W, J);

                kg_matrix_multiply(H, Q, W);
                kg_matrix_set(W, Q);
            }
        }
    }
    else
    {
        //because we are lazy and don't need the functionality this is only implemented for square matrices
        std::stringstream ss;
        ss << "kg_matrix_householder_bidiagonalize: error, matrix A is not square. \n";
        ss << "matrix A is "<<A->size1<<" by "<<A->size2<<". \n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}




void
kg_matrix_householder(kg_matrix* H, const kg_vector* w)
{
    //check that H has dimensions n x n
    if(H->size1 == w->size && H->size2 == w->size)
    {
        //place the outer product of w with itself into H
        kg_vector_outer_product(w,w,H);

        //scale by -2
        kg_matrix_scale(H, -2.0);

        //add the identity to H
        for(unsigned int i=0; i<H->size1; i++)
        {
            kg_matrix_set( H, i, i, kg_matrix_get(H,i,i) + 1.0);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "kg_matrix_householder: error, matrix H is not square. \n";
        ss << "matrix H is "<<H->size1<<" by "<<H->size2<<". \n";
        mathmsg( eDebug ) << ss.str().c_str() << eom;
    }
}


//void
//kg_matrix_print(const kg_matrix* m)
//{
//    for(unsigned int i=0; i<m->size1; i++) //rows
//    {
//        for(unsigned int j=0; j < (m->size2-1); j++) //col
//        {
//            ss<<kg_matrix_get(m,i,j)<<", ";
//        }

//        ss<<kg_matrix_get(m,i,m->size2-1);

//        ss<<kgendl;
//    }
//}



}
