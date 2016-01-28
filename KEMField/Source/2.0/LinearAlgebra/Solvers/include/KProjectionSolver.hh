#ifndef KProjectionSolver_DEF
#define KProjectionSolver_DEF

#include "KSimpleVector.hh"
#include "KSimpleMatrix.hh"
#include "KEMCout.hh"

#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMVectorOperations.hh"


namespace KEMField
{

/*
*
*@file KProjectionSolver.hh
*@class KProjectionSolver
*@brief
* very simple solver which given a collection of solutions
* and their right hand sides to the same matrix equation attempts
* to form a new solution out of the best linear combination of
* pre-existing solutions, similar to the KSuperpositionSolver.
* Helpful for giving Krylov solvers an inital guess state
* (instead of starting them off at the zero vector). Similar
* to a gram-schmidt process, except the basis vectors (pre-existing solutions)
* do not necessarily need to be orthonormal.
*
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Mar 28 15:27:04 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


template <typename ValueType>
class KProjectionSolver
{
    public:
        typedef KVector<ValueType> Vector;

        KProjectionSolver(): fTolerance(1.e-14) {}
        virtual ~KProjectionSolver() {}

        void SetTolerance(double d) { fTolerance = d; }

        //must all be solutions to the same linear equation w/ matrix A
        void AddSolvedSystem(const Vector& x, const Vector& b);

        //compute the 'solution' x which minimizes the component
        //of b which is orthogonal to the previously solved system right hand sides
        void Solve(Vector& x, const Vector& b);

        //get the l2 norm diference between the right hand side given right hand side
        //and the best summed projection over all of the existing solution right hand sides
        double GetL2NormDifference(){return fResidualNorm;};

        double InnerProduct(const Vector& a, const Vector& b);

    private:

        //pre-exising solution vectors and their associated right hand sides
        std::vector<const Vector*> fX;
        std::vector<const Vector*> fB;

        double fResidualNorm;
        double fTolerance;

};

template< typename ValueType >
void
KProjectionSolver<ValueType>::Solve(Vector& x, const Vector& b)
{
    if(fB.size() != 0)
    {
        //we are looking for the vector projection p, which is closest to our given b
        //in the space spanned by the previously solved solutions

        //allocate space to solve the minimization problem
        unsigned int dim = fB.size();

        kfm_matrix* M = kfm_matrix_calloc(dim,dim);
        for(unsigned int i=0;i<dim;i++)
        {
            for(unsigned int j=0; j<=i ;j++)
            {
                //compute the dot product, and set matrix element
                double d = InnerProduct(*fB[i], *fB[j]);
                kfm_matrix_set(M,i,j,d);
                kfm_matrix_set(M,j,i,d); //symmetric matrix
            }
        }

        kfm_matrix_print(M);

        //compute the right hand side of the matrix equation
        kfm_vector* v = kfm_vector_calloc(dim);
        for(unsigned int i=0;i<dim;i++)
        {
            double d = InnerProduct(b,*fB[i]);
            kfm_vector_set(v,i,d);
        }

        kfm_vector* p = kfm_vector_calloc(dim);
        //now we are going to construct the SVD, and compute the psuedo inverse
        kfm_matrix* U = kfm_matrix_calloc(dim, dim);
        kfm_matrix* V = kfm_matrix_calloc(dim, dim);
        kfm_vector* s = kfm_vector_calloc(dim);

        //computes the singular value decomposition of the matrix M = U*diag(S)*V^T
        kfm_matrix_svd(M, U, s, V);

        //solve the linear eq
        kfm_matrix_svd_solve(U, s, V, v, p);

        //compute norm of b
        double b_norm = std::sqrt(InnerProduct(b,b));

        //compute the projection vector
        KSimpleVector<ValueType> b_proxy(b.Dimension(), 0);
        KSimpleVector<ValueType> diff(b.Dimension(), 0);
        for(unsigned int i=0; i<b.Dimension(); i++){b_proxy[i] = 0.0;};

        for(unsigned int j=0; j<dim; j++)
        {
            double weight = kfm_vector_get(p,j);
            for(unsigned int i=0; i<b_proxy.Dimension(); i++)
            {
                b_proxy[i] += weight*( (*fB[j])(i) );
            }
        }

        //compute the difference
        for(unsigned int i=0; i<x.Dimension(); i++)
        {
            diff[i] = b_proxy(i) - b(i);
        }

        //compute relative norm difference of b_proxy to determine how close we were able to come to a
        //true solution, if the newly desired b is in the space spanned by the previous
        //solutions, then the norm of b_proxy should be zero
        fResidualNorm = std::sqrt( InnerProduct(diff, diff) )/b_norm;

        //zero out x
        for(unsigned int i=0; i<x.Dimension(); i++){x[i] = 0.0;}

        //now that we have the weights for all of the solutions we can compute
        //the nearest solution
        for(unsigned int j=0; j<dim; j++)
        {
            double weight = kfm_vector_get(p,j);
            for(unsigned int i=0; i<x.Dimension(); i++)
            {
                x[i] += weight*( (*fX[j])(i) );
            }
        }

        kfm_matrix_free(M);
        kfm_matrix_free(U);
        kfm_matrix_free(V);
        kfm_vector_free(s);
        kfm_vector_free(p);
        kfm_vector_free(v);
    }

};


template< typename ValueType >
void
KProjectionSolver<ValueType>::AddSolvedSystem(const Vector& x, const Vector& b)
{
    fX.push_back(&x);
    fB.push_back(&b);
}

template< typename ValueType >
double
KProjectionSolver<ValueType>::InnerProduct(const Vector& a, const Vector& b)
{
    unsigned dim = a.Dimension();
    double result = 0.;
    for(unsigned int i=0; i<dim; i++)
    {
        result += a(i)*b(i);
    }
    return result;
};


}//end of namespace

#endif /* KProjectionSolver_DEF */
