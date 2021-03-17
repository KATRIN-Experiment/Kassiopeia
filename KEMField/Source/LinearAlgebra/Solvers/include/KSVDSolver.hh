#ifndef KSVDSOLVER_DEF
#define KSVDSOLVER_DEF

#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMMatrixOperations.hh"
#include "KFMVectorOperations.hh"
#include "KMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
template<typename ValueType> class KSVDSolver
{
  public:
    using Matrix = KMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KSVDSolver() : fTolerance(1.e-14) {}
    virtual ~KSVDSolver() = default;

    bool Solve(const Matrix& A, Vector& x, const Vector& b) const;
    void SetTolerance(double tol)
    {
        fTolerance = tol;
    }

  private:
    double fTolerance;
};

template<typename ValueType> bool KSVDSolver<ValueType>::Solve(const Matrix& A, Vector& x, const Vector& b) const
{
    unsigned int M = A.Dimension(0);
    unsigned int N = A.Dimension(1);

    //need to pretend the matrix is square
    //to help out the svd decomposition
    if (M < N) {
        M = N;
    };
    if (N < M) {
        N = M;
    };

    kfm_matrix* A_ = kfm_matrix_calloc(M, N);
    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            if (i < A.Dimension(0) && j < A.Dimension(1)) {
                kfm_matrix_set(A_, i, j, A(i, j));
            }
            else {
                kfm_matrix_set(A_, i, j, 0.);
            }
        }
    }

    kfm_vector* x_ = kfm_vector_calloc(N);
    kfm_vector* b_ = kfm_vector_calloc(M);
    for (unsigned int i = 0; i < M; i++) {
        if (i < b.Dimension()) {
            kfm_vector_set(b_, i, b(i));
        }
        else {
            kfm_vector_set(b_, i, 0.);
        }
    }

    //now we are going to construct the SVD, and compute the psuedo inverse
    kfm_matrix* U = kfm_matrix_calloc(M, N);
    kfm_matrix* V = kfm_matrix_calloc(N, N);
    kfm_vector* s = kfm_vector_calloc(N);

    //computes the singular value decomposition of the matrix A = U*diag(S)*V^T
    kfm_matrix_svd(A_, U, s, V);

    //given the singular value decomposition of the matrix A = U*diag(S)*V^T
    //this function solves the equation Ax = b in the least squares sense
    kfm_matrix_svd_solve(U, s, V, b_, x_);

    for (unsigned int i = 0; i < x.Dimension(); i++) {
        x[i] = kfm_vector_get(x_, i);
    }

    KSimpleVector<double> b_comp(b.Dimension());
    A.Multiply(x, b_comp);
    b_comp -= b;


    kfm_matrix_free(A_);
    kfm_matrix_free(U);
    kfm_matrix_free(V);
    kfm_vector_free(s);
    kfm_vector_free(b_);
    kfm_vector_free(x_);

    return b_comp.InfinityNorm() < fTolerance;
}

}  // namespace KEMField

#endif /* KSVDSOLVER_DEF */
