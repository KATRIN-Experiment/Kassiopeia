#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"

#include "KFMMessaging.hh"

namespace KEMField
{

#ifdef KEMFIELD_USE_GSL
////////////////////////////////////////////////////////////////////////////////
//we have GSL so use the fast BLAS based implementation

kfm_matrix*
kfm_matrix_alloc(unsigned int nrows, unsigned int ncolumns)
{
    return gsl_matrix_alloc(nrows, ncolumns);
}

kfm_matrix* kfm_matrix_calloc(unsigned int nrows, unsigned int ncolumns)
{
    return gsl_matrix_calloc(nrows, ncolumns);
}

void kfm_matrix_free(kfm_matrix* m)
{
    gsl_matrix_free(m);
}

double kfm_matrix_get(const kfm_matrix* m, unsigned int i, unsigned int j)
{
    return gsl_matrix_get(m, i, j);
}

void kfm_matrix_set(kfm_matrix* m, unsigned int i, unsigned int j, double x)
{
    gsl_matrix_set(m, i, j,x);
}

void kfm_matrix_set_zero(kfm_matrix* m)
{
    gsl_matrix_set_zero(m);
}

void
kfm_matrix_set_identity(kfm_matrix* m)
{
    gsl_matrix_set_identity(m);
}

void
kfm_matrix_set(const gsl_matrix* src, gsl_matrix* dest)
{
    gsl_matrix_memcpy(dest, src);
}

void kfm_matrix_sub(kfm_matrix* a, const kfm_matrix* b)
{
    gsl_matrix_sub(a,b);
}

void kfm_matrix_add(kfm_matrix* a, const kfm_matrix* b)
{
    gsl_matrix_add(a,b);
}

void kfm_matrix_scale(kfm_matrix* a, double scale_factor)
{
    gsl_matrix_scale(a, scale_factor);
}


void kfm_matrix_multiply(const kfm_matrix* A, const kfm_matrix* B, kfm_matrix* C)
{
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


void
kfm_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kfm_matrix* A, const kfm_matrix* B, kfm_matrix* C)
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
kfm_matrix_svd(const kfm_matrix* A, kfm_matrix* U, kfm_vector* S, kfm_matrix* V)
{
    kfm_vector* work = kfm_vector_alloc(A->size1);
    kfm_matrix_set(A, U); //copy A into U
    gsl_linalg_SV_decomp(U, V, S, work);
    kfm_vector_free(work);
}

//given the singular value decomposition of the matrix A = U*diag(S)*V^T, this function solves the equation Ax = b
void
kfm_matrix_svd_solve(const kfm_matrix* U, const kfm_vector* S, const kfm_matrix* V, const kfm_vector* b, kfm_vector* x)
{
    gsl_linalg_SV_solve(U, V, S, b, x);
}

#else
////////////////////////////////////////////////////////////////////////////////
//no GSL available

kfm_matrix*
kfm_matrix_alloc(unsigned int nrows, unsigned int ncolumns)
{
    kfm_matrix* m = new kfm_matrix();
    m->size1 = nrows;
    m->size2 = ncolumns;
    m->data = new double[nrows*ncolumns];
    return m;
}

kfm_matrix* kfm_matrix_calloc(unsigned int nrows, unsigned int ncolumns)
{
    kfm_matrix* m = new kfm_matrix();
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

void kfm_matrix_free(kfm_matrix* m)
{
    delete[] m->data;
    delete m;
}

double kfm_matrix_get(const kfm_matrix* m, unsigned int i, unsigned int j)
{
    unsigned int index = i*(m->size2) + j;
    return m->data[index];
}

void kfm_matrix_set(kfm_matrix* m, unsigned int i, unsigned int j, double x)
{
    unsigned int index = i*(m->size2) + j;
    m->data[index] = x;
}

void kfm_matrix_set_zero(kfm_matrix* m)
{
    unsigned int total_size = (m->size1)*(m->size2);
    double* d = m->data;
    for(unsigned int i=0; i<total_size; i++)
    {
        d[i] = 0.;
    }
}

void
kfm_matrix_set_identity(kfm_matrix* m)
{
    kfm_matrix_set_zero(m);
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
        kfm_matrix_set(m,i,i,1.0);
    }
}

void
kfm_matrix_set(const kfm_matrix* src, kfm_matrix* dest)
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
        kfmout << "kfm_matrix_set: error, matrices have difference sizes."<<kfmendl;
        kfmout << "source matrix is "<<src->size1<<" by "<<src->size2<<"."<<kfmendl;
        kfmout << "destination matrix is "<<dest->size1<<" by "<<dest->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}

void kfm_matrix_sub(kfm_matrix* a, const kfm_matrix* b)
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
        kfmout << "kfm_matrix_sub: error, matrices have difference sizes."<<kfmendl;
        kfmout << "matrix a is "<<a->size1<<" by "<<a->size2<<"."<<kfmendl;
        kfmout << "matrix b is "<<b->size1<<" by "<<b->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}

void kfm_matrix_add(kfm_matrix* a, const kfm_matrix* b)
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
        kfmout << "kfm_matrix_add: error, matrices have difference sizes."<<kfmendl;
        kfmout << "matrix a is "<<a->size1<<" by "<<a->size2<<"."<<kfmendl;
        kfmout << "matrix b is "<<b->size1<<" by "<<b->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}

void kfm_matrix_scale(kfm_matrix* a, double scale_factor)
{
    unsigned int total_size = (a->size1)*(a->size2);
    for(unsigned int i=0; i<total_size; i++)
    {
        a->data[i] *= scale_factor;
    }
}

void kfm_matrix_multiply(const kfm_matrix* A, const kfm_matrix* B, kfm_matrix* C)
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
        //this will likely be at least 100x slower than the GSL/BLAS implemention
        for(unsigned int i=0; i<c_row; i++)
        {
            for(unsigned int j=0; j<c_col; j++)
            {
                double elem = 0.0;

                for(unsigned int offset=0; offset<b_row; offset++)
                {
                    elem += ( kfm_matrix_get(A, i, offset) )*( kfm_matrix_get(B, offset, j) );
                }

                kfm_matrix_set(C, i, j, elem);
            }
        }
    }
    else
    {
        kfmout << "kfm_matrix_multiply: error, matrices have difference sizes."<<kfmendl;
        kfmout << "matrix a is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
        kfmout << "matrix b is "<<B->size1<<" by "<<B->size2<<"."<<kfmendl;
        kfmout << "matrix c is "<<C->size1<<" by "<<C->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}


void
kfm_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kfm_matrix* A, const kfm_matrix* B, kfm_matrix* C)
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

                    elem += ( kfm_matrix_get(A, ai, aj) )*( kfm_matrix_get(B, bi, bj) );
                }

                kfm_matrix_set(C, i, j, elem);
            }
        }
    }
    else
    {
        kfmout << "kfm_matrix_multiply_with_transpose: error, matrices have difference sizes."<<kfmendl;
        kfmout << "matrix a is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
        kfmout << "matrix b is "<<B->size1<<" by "<<B->size2<<"."<<kfmendl;
        kfmout << "matrix c is "<<C->size1<<" by "<<C->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}


void
kfm_matrix_svd(const kfm_matrix* A, kfm_matrix* U, kfm_vector* S, kfm_matrix* V)
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
        kfmout << "kfm_matrix_svd: error, matrices A and U have different sizes."<<kfmendl;
        kfmout << "matrix A is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
        kfmout << "matrix U is "<<U->size1<<" by "<<U->size2<<"."<<kfmendl;
        kfmexit(1);
    }

    if( n != V->size1 || n != V->size1 )
    {
        kfmout << "kfm_matrix_svd: error, matrix V has wrong size."<<kfmendl;
        kfmout << "matrix V is "<<V->size1<<" by "<<V->size2<<"."<<kfmendl;
        kfmout << "matrix V should be "<<n<<" by "<<n<<"."<<kfmendl;
        kfmexit(1);
    }

    if(S->size != n)
    {
        kfmout << "kfm_matrix_svd: error, vector S has wrong size."<<kfmendl;
        kfmout << "vector S is length "<<S->size<<kfmendl;
        kfmout << "vector S should be length "<<n<<kfmendl;
        kfmexit(1);
    }

    //copy A into U
    kfm_matrix_set(A,U);
    //set V to the identity
    kfm_matrix_set_identity(V);

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
            g1 = kfm_matrix_get(U,i,j);
            tol += g1*g1;
        }
    }
    tol = tol*n*m*KFM_EPSILON*KFM_EPSILON;

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
                    g1 = kfm_matrix_get(U, k, i);
                    g2 = kfm_matrix_get(U, k, j);
                    a += g1*g1;
                    b += g2*g2;
                    c += g1*g2;
                }

                if( (c*c)/(a*b) > tol )
                {
                    //compute the sine/cosine of the Given's rotation
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
                        g1 = kfm_matrix_get(U, k, i);
                        g2 = kfm_matrix_get(U, k, j);
                        kfm_matrix_set(U, k, i, cs*g1 - sn*g2);
                        kfm_matrix_set(U, k, j, sn*g1 + cs*g2);

                        //apply to V
                        g1 = kfm_matrix_get(V, k, i);
                        g2 = kfm_matrix_get(V, k, j);
                        kfm_matrix_set(V, k, i, cs*g1 - sn*g2);
                        kfm_matrix_set(V, k, j, sn*g1 + cs*g2);
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
            #ifdef KFMMATH_DEBUG
            kfmout << "kfm_matrix_svd: warning, SVD failed to converge within "<<n_max_iter<<" iterations, "<<kfmendl;
            #endif
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
            g1 = kfm_matrix_get(U,j,i);
            a += g1*g1;
        }
        norm_s += a;
        a = std::sqrt(a);
        kfm_vector_set(S,i,a);
    }

    norm_s = std::sqrt(norm_s);

    tol = KFM_EPSILON*norm_s;

    //eliminate all those singular values which are below the tolerance
    for(unsigned int i=0; i<n; i++)
    {
        if(kfm_vector_get(S,i) < tol)
        {
            kfm_vector_set(S,i,0.0);
        }
    }

    //now we fix U by post multiplying with the inverse of diag(S)
    for(unsigned int i=0; i<m; i++) //rows
    {
        for(unsigned int j=0; j<n; j++) //col
        {
            g1 = kfm_matrix_get(U,i,j);
            g2 = kfm_vector_get(S,j);
            if(g2 == 0.0)
            {
                kfm_matrix_set(U,i,j, 0.0);//set to zero
            }
            else
            {
                kfm_matrix_set(U,i,j, g1/g2 );
            }
        }
    }
}



void
kfm_matrix_svd_solve(const kfm_matrix* U, const kfm_vector* S, const kfm_matrix* V, const kfm_vector* b, kfm_vector* x)
{
    //the solution is given by:
    //x = [V*diag(S)^{-1}*U^{T}]b

    //workspace
    kfm_vector* work = kfm_vector_alloc(x->size);

    //first we copy b into x and apply U^T
    kfm_vector_set(b, x);
    kfm_matrix_transpose_vector_product(U, x, work);
    kfm_vector_set(work, x);

    //now we apply the inverse of diag(S) to x
    //with the exception that if a singular value is zero then we apply zero
    //we assume anything less than KFM_EPSILON*norm(S) to be essentially zero (singular values should all be positive)
    double s, elem;
    double norm_s = kfm_vector_norm(S);
    for(unsigned int i=0; i<S->size; i++)
    {
        s = kfm_vector_get(S,i);
        if(s > KFM_EPSILON*norm_s)
        {
            //multiply 1/s against the i'th element of x
            elem = (1.0/s)*kfm_vector_get(x,i);
            kfm_vector_set(x,i,elem);
        }
        else
        {
            kfm_vector_set(x,i,0.0);
        }
    }

    //finally we apply the matrix V to the vector x
    kfm_matrix_vector_product(V, x, work);
    kfm_vector_set(work, x);


    //free workspace
    kfm_vector_free(work);

}

#endif


////////////////////////////////////////////////////////////////////////////////
//functions defined for convenience whether we have GSL or not

kfm_sparse_matrix*
kfm_sparse_matrix_alloc(unsigned int nrows, unsigned int ncolumns, unsigned int n_elements)
{
    kfm_sparse_matrix* m = new kfm_sparse_matrix();
    m->size1 = nrows;
    m->size2 = ncolumns;
    m->data = new double[n_elements];
    m->row = new unsigned int[n_elements];
    m->column = new unsigned int[n_elements];
    return m;

}

void
kfm_sparse_matrix_free(kfm_sparse_matrix* m)
{
    delete[] m->data;
    delete[] m->row;
    delete[] m->column;
}


double
kfm_sparse_matrix_get(const kfm_sparse_matrix* m, unsigned int i, unsigned int j)
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
kfm_sparse_matrix_set(kfm_sparse_matrix* m, unsigned int i, unsigned int j, unsigned int element_index, double x)
{
    (m->data)[element_index] = x;
    (m->row)[element_index] = i;
    (m->column)[element_index] = j;
}


void kfm_matrix_transpose(const kfm_matrix* in, kfm_matrix* out)
{
    for(unsigned int row=0; row<in->size1; row++)
    {
        for(unsigned int col=0; col<in->size2; col++)
        {
            kfm_matrix_set(out, col, row, kfm_matrix_get(in, row, col) );
        }
    }
}


void kfm_matrix_euler_angles(const kfm_matrix* R, double& alpha, double& beta, double& gamma, double& tol)
{
    //check that R is 3x3
    if(R->size1 == 3 && R->size2 == 3)
    {

        //calloc temporary matrix C
        kfm_matrix* C = kfm_matrix_alloc(3,3);

        //multiply the matrix by its inverse (transpose)
        kfm_matrix_multiply_with_transpose(false, true, R, R, C);

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
                    temp = kfm_matrix_get(C,i,j);
                }
                else
                {
                    temp =  kfm_matrix_get(C,i,j) - 1.0;
                }
                tol += temp*temp;
            }
        }
        tol = std::sqrt(tol + KFM_EPSILON*KFM_EPSILON);


        //the angles that are computed are not unique but represent one possible
        //set of angles that construct the rotation matrix
        bool isDegenerate = false;

        //if  |1-|cos(beta)| | < tol we are in the degenerate case
        if( std::fabs( 1. - std::fabs( kfm_matrix_get(R,2,2) ) ) <= tol)
        {
            isDegenerate = true;
        }

        if(!isDegenerate)
        {
            beta = std::acos( kfm_matrix_get(R,2,2) );
            alpha = std::atan2( (-1.0* kfm_matrix_get(R,2,1) )/std::sin(beta), ( kfm_matrix_get(R,2,0))/std::sin(beta) );
            gamma = std::atan2( (-1.0* kfm_matrix_get(R,1,2) )/std::sin(beta), -1*( kfm_matrix_get(R,0,2))/std::sin(beta)  ) ;
        }
        else
        {
            if( std::fabs(1. -  kfm_matrix_get(R,2,2) ) <= tol)
            {
                alpha =  std::atan2(  kfm_matrix_get(R,1,0),  kfm_matrix_get(R,0,0) );
                beta = 0;
                gamma = 0;
            }
            else if( std::fabs(1. +  kfm_matrix_get(R,2,2) ) <= tol)
            {
                alpha = std::atan2(  kfm_matrix_get(R,0,1),  kfm_matrix_get(R,1,1) );
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

        kfm_matrix_free(C);
    }

}



void kfm_matrix_from_euler_angles_ZYZ(kfm_matrix* R, double alpha, double beta, double gamma)
{
    double sin_a = std::sin(alpha);
    double cos_a = std::cos(alpha);

    double sin_b = std::sin(beta);
    double cos_b = std::cos(beta);

    double sin_c = std::sin(gamma);
    double cos_c = std::cos(gamma);

    kfm_matrix_set_identity(R);

    kfm_matrix* A = kfm_matrix_alloc(3,3);
    kfm_matrix* B = kfm_matrix_alloc(3,3);
    kfm_matrix* C = kfm_matrix_alloc(3,3);

    kfm_matrix_set_identity(A);
    kfm_matrix_set_identity(B);
    kfm_matrix_set_identity(C);

    kfm_matrix_set(A, 0, 0, cos_a);
    kfm_matrix_set(A, 1, 1, cos_a);
    kfm_matrix_set(A, 0, 1, -sin_a);
    kfm_matrix_set(A, 1, 0, sin_a);

    kfm_matrix_set(B, 0, 0, cos_b);
    kfm_matrix_set(B, 2, 2, cos_b);
    kfm_matrix_set(B, 0, 2, -sin_b);
    kfm_matrix_set(B, 2, 0, sin_b);

    kfm_matrix_set(C, 0, 0, cos_c);
    kfm_matrix_set(C, 1, 1, cos_c);
    kfm_matrix_set(C, 0, 1, -sin_c);
    kfm_matrix_set(C, 1, 0, sin_c);

    kfm_matrix_set(A,R);

    kfm_matrix_multiply(B,R,A);
    kfm_matrix_multiply(C,A,R);

    kfm_matrix_free(A);
    kfm_matrix_free(B);
    kfm_matrix_free(C);
}


void kfm_matrix_from_axis_angle(kfm_matrix* R, double cos_angle, double sin_angle, const kfm_vector* axis)
{
    kfm_vector_outer_product(axis, axis, R);

    double c = cos_angle;
    double s = sin_angle;
    double t = 1.0 - c;

    for(unsigned int i=0; i<3; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            kfm_matrix_set( R, i, j, t*kfm_matrix_get(R, i, j) );
        }
    }

    for(unsigned int i=0; i<3; i++)
    {
        kfm_matrix_set( R, i, i, kfm_matrix_get(R, i, i) + c);
    }

    kfm_matrix_set( R, 0, 1, kfm_matrix_get(R, 0, 1) - s*kfm_vector_get(axis, 2));
    kfm_matrix_set( R, 0, 2, kfm_matrix_get(R, 0, 2) + s*kfm_vector_get(axis, 1));
    kfm_matrix_set( R, 1, 2, kfm_matrix_get(R, 1, 2) - s*kfm_vector_get(axis, 0));

    kfm_matrix_set( R, 1, 0, kfm_matrix_get(R, 1, 0) + s*kfm_vector_get(axis, 2));
    kfm_matrix_set( R, 2, 0, kfm_matrix_get(R, 2, 0) - s*kfm_vector_get(axis, 1));
    kfm_matrix_set( R, 2, 1, kfm_matrix_get(R, 2, 1) + s*kfm_vector_get(axis, 0));


}

void kfm_matrix_from_axis_angle(kfm_matrix* R, double angle, const kfm_vector* axis)
{
    kfm_vector_outer_product(axis, axis, R);

    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;

    for(unsigned int i=0; i<3; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            kfm_matrix_set( R, i, j, t*kfm_matrix_get(R, i, j) );
        }
    }

    for(unsigned int i=0; i<3; i++)
    {
        kfm_matrix_set( R, i, i, kfm_matrix_get(R, i, i) + c);
    }

    kfm_matrix_set( R, 0, 1, kfm_matrix_get(R, 0, 1) - s*kfm_vector_get(axis, 2));
    kfm_matrix_set( R, 0, 2, kfm_matrix_get(R, 0, 2) + s*kfm_vector_get(axis, 1));
    kfm_matrix_set( R, 1, 2, kfm_matrix_get(R, 1, 2) - s*kfm_vector_get(axis, 0));

    kfm_matrix_set( R, 1, 0, kfm_matrix_get(R, 1, 0) + s*kfm_vector_get(axis, 2));
    kfm_matrix_set( R, 2, 0, kfm_matrix_get(R, 2, 0) - s*kfm_vector_get(axis, 1));
    kfm_matrix_set( R, 2, 1, kfm_matrix_get(R, 2, 1) + s*kfm_vector_get(axis, 0));
}



void
kfm_matrix_householder_bidiagonalize(const kfm_matrix* A, kfm_matrix* P, kfm_matrix* J, kfm_matrix* Q)
{
    //take the matrix A and bidiagonalize it A = PJQ^T, where J is the upper bidiagonal matrix
    //uses householder transformations as stated by Golub-Kahan in
    //Calculating the singular values and pseudo-inverse of a matrix. J. SIAM Numer. Anal. Ser. B, Vol 2, No. 2 1965

    unsigned int n = A->size1;

    if(A->size1 == A->size2)
    {
        if(P->size1 != n || P->size2 != n || J->size1 != n || J->size2 != n || Q->size1 != n || Q->size2 != n )
        {
            kfmout << "kfm_matrix_householder_bidiagonalize: error, matrices A, P, J, or Q have different sizes."<<kfmendl;
            kfmout << "matrix A is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
            kfmout << "matrix P is "<<P->size1<<" by "<<P->size2<<"."<<kfmendl;
            kfmout << "matrix J is "<<J->size1<<" by "<<J->size2<<"."<<kfmendl;
            kfmout << "matrix Q is "<<Q->size1<<" by "<<Q->size2<<"."<<kfmendl;
            kfmexit(1);
        }

        //copy A into J
        kfm_matrix_set(A,J);
        //set P and Q to identity
        kfm_matrix_set_identity(P);
        kfm_matrix_set_identity(Q);

        //allocate workspace
        kfm_vector* x = kfm_vector_alloc(n);
        kfm_matrix* H = kfm_matrix_alloc(n,n);
        kfm_matrix* W = kfm_matrix_alloc(n,n);
        double sk, tk, ck, dk, ak, signk;

        for(unsigned int k=0; k < n; k++)
        {
            //take care of the column first/////////////////////////////////////
            //copy the k-th column into x while zero-ing out all components where j<k
            for(unsigned int j=0; j<n; j++)
            {
                if(j < k)
                {
                    kfm_vector_set(x, j, 0.0);
                }
                else
                {
                    kfm_vector_set(x, j, kfm_matrix_get(J, j, k) );
                }
            }

            //get a_k,k
            ak = kfm_matrix_get(J, k, k);
            //compute signum of a_k,k
            signk = ak/(std::fabs(ak));
            //compute norm of x to get 's_k' value
            sk = kfm_vector_norm(x);
            //compute the 'c_k' value
            ck = 1.0/(2.0*sk*signk*std::sqrt( 0.5*(1.0 + std::fabs(ak)/sk) ) );

            //now we compute the values of x
            for(unsigned int j=k; j<n; j++)
            {
                if(j == k)
                {
                    kfm_vector_set(x, j, std::sqrt( 0.5*(1.0 + std::fabs(ak)/sk) ) );
                }
                else
                {
                    kfm_vector_set(x, j, ck*kfm_vector_get(x,j) );
                }
            }

            //compute the householder matrix H
            kfm_matrix_householder(H,x);

            //pre-apply H to J, and post-apply H to P
            kfm_matrix_multiply(H, J, W);
            kfm_matrix_set(W, J);

            kfm_matrix_multiply(P, H, W);
            kfm_matrix_set(W, P);

            //next take care of the row/////////////////////////////////////////
            //copy the k-th row into x while zero-ing out all components where j<k
            if(k < n-1) //only need to do this if it is not the last column
            {

                for(unsigned int j=0; j<n; j++)
                {
                    if(j <= k)
                    {
                        kfm_vector_set(x, j, 0.0);
                    }
                    else
                    {
                        kfm_vector_set(x, j, kfm_matrix_get(J, k, j) );
                    }
                }

                //get a_k,k+1
                ak = kfm_matrix_get(J, k, k+1);
                //compute signum of a_k,k+1
                signk = ak/(std::fabs(ak));
                //compute norm of x to get 't_k' value
                tk = kfm_vector_norm(x);
                //compute the 'd_k' value
                dk = 1.0/(2.0*tk*signk*std::sqrt( 0.5*(1.0 + std::fabs(ak)/tk) ) );

                //now we compute the values of x
                for(unsigned int j=k+1; j<n; j++)
                {
                    if(j == k+1)
                    {
                        kfm_vector_set(x, j, std::sqrt( 0.5*(1.0 + std::fabs(ak)/tk) ) );
                    }
                    else
                    {
                        kfm_vector_set(x, j, dk*kfm_vector_get(x,j) );
                    }
                }

                //compute the householder matrix H
                kfm_matrix_householder(H,x);

                //post-apply H to J, and pre-apply H to Q
                kfm_matrix_multiply(J, H, W);
                kfm_matrix_set(W, J);

                kfm_matrix_multiply(H, Q, W);
                kfm_matrix_set(W, Q);
            }
        }
    }
    else
    {
        //because we are lazy and don't need the functionality this is only implemented for square matrices
        kfmout << "kfm_matrix_householder_bidiagonalize: error, matrix A is not square."<<kfmendl;
        kfmout << "matrix A is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}




void
kfm_matrix_householder(kfm_matrix* H, const kfm_vector* w)
{
    //check that H has dimensions n x n
    if(H->size1 == w->size && H->size2 == w->size)
    {
        //place the outer product of w with itself into H
        kfm_vector_outer_product(w,w,H);

        //scale by -2
        kfm_matrix_scale(H, -2.0);

        //add the identity to H
        for(unsigned int i=0; i<H->size1; i++)
        {
            kfm_matrix_set( H, i, i, kfm_matrix_get(H,i,i) + 1.0);
        }
    }
    else
    {
        kfmout << "kfm_matrix_householder: error, matrix H is not square."<<kfmendl;
        kfmout << "matrix H is "<<H->size1<<" by "<<H->size2<<"."<<kfmendl;
        kfmexit(1);
    }
}


void kfm_matrix_upper_triangular_solve(const kfm_matrix* A, const kfm_vector* b, kfm_vector* x)
{
    if(A->size1 == A->size2 && A->size1 == b->size && b->size == x->size)
    {
        int n = A->size1;

        double x_temp;

        for(int i= (n-1); i >= 0; i--)
        {
            x_temp = kfm_vector_get(b,i);
            for(int j= i+1; j < n; j++ )
            {
                double x_j = kfm_vector_get(x,j);
                double a_ij = kfm_matrix_get(A, i, j);
                x_temp -= a_ij*x_j;
            }
            double a_ii =  kfm_matrix_get(A, i, i);
            x_temp /= a_ii;
            kfm_vector_set(x, i, x_temp);
        }

    }
    else
    {
        kfmout << "kfm_matrix_upper_triangular_solve: error, matrix and vector dimensions are miss-matched."<<kfmendl;
        kfmout << "matrix A is "<<A->size1<<" by "<<A->size2<<"."<<kfmendl;
        kfmout << "vector b is "<<b->size<<std::endl;
        kfmout << "vector x is "<<x->size<<std::endl;
        kfmexit(1);
    }



}

void
kfm_matrix_print(const kfm_matrix* m)
{
    for(unsigned int i=0; i<m->size1; i++) //rows
    {
        for(unsigned int j=0; j < (m->size2-1); j++) //col
        {
            kfmout<<kfm_matrix_get(m,i,j)<<", ";
        }

        kfmout<<kfm_matrix_get(m,i,m->size2-1);

        kfmout<<kfmendl;
    }
}



}
