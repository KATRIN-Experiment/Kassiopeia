#ifndef KBiconjugateGradientStabilized_HH__
#define KBiconjugateGradientStabilized_HH__

#include <cmath>
#include <iostream>

#include "KMatrix.hh"
#include "KVector.hh"

#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

namespace KEMField
{

/*
*
*@file KBiconjugateGradientStabilized.hh
*@class KBiconjugateGradientStabilized
*@brief
*@details: note only valid for real types, do not use with ValueType = std::complex<...>
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Feb  1 10:20:58 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ValueType >
class KBiconjugateGradientStabilized
{
    public:
        typedef KSquareMatrix<ValueType> Matrix;
        typedef KVector<ValueType> Vector;

        KBiconjugateGradientStabilized(const Matrix& A, Vector& x, const Vector& b):
        fA(A),
        fX(x),
        fB(b)
        {
            fDim = fB.Dimension();
            fR.resize(fDim, 0.);
            fR_hat.resize(fDim, 0.);
            fP.resize(fDim, 0.);
            fV.resize(fDim, 0.);
            fS.resize(fDim, 0.);
            fT.resize(fDim, 0.);
        };

        virtual ~KBiconjugateGradientStabilized(){};

        void Initialize();
        void AugmentKrylovSubspace();
        void UpdateSolution(){}; //solution is updated in krylov subspace step

        void GetResidualNorm(double& norm);
        void CoalesceData(){};
        void Finalize(){};

        unsigned int Dimension() const {return fDim;};

        void SetResidualVector(const Vector&);
        void GetResidualVector(Vector&) const;

    private:

        double InnerProduct(const Vector& a, const Vector& b);

        unsigned int fDim;
        const Matrix& fA;
        Vector& fX;
        const Vector& fB;

        double rho;
        double beta;
        double alpha;
        double omega;
        double rho_prev;

        KSimpleVector<ValueType> fR;
        KSimpleVector<ValueType> fR_hat;
        KSimpleVector<ValueType> fP;
        KSimpleVector<ValueType> fV;

        KSimpleVector<ValueType> fS;
        KSimpleVector<ValueType> fT;
};

template< typename ValueType >
void
KBiconjugateGradientStabilized<ValueType>::Initialize()
{
    //first we compute the initial residual vector: r = b - Ax
    fA.Multiply(fX, fV);
    for(unsigned int i=0; i<fDim; i++)
    {
        fR[i] = fB(i) - fV[i];
        fR_hat[i] = fR[i];
    }

    rho = 1.;
    rho_prev = 1.;
    alpha = 1.;
    omega = 1.;

    //we take the first conjugate vector to be the residual
    for(unsigned int i=0; i<fDim; i++)
    {
        fV[i] = 0.;
        fP[i] = 0.;
    }
}


template< typename ValueType >
void
KBiconjugateGradientStabilized<ValueType>::AugmentKrylovSubspace()
{
    rho = InnerProduct(fR_hat, fR);
    beta = (rho/rho_prev)*(alpha/omega);

    for(unsigned int i=0; i<fDim; i++)
    {
        fP[i] = fR[i] + beta*(fP[i] - omega*fV[i]);
    }

    fA.Multiply(fP, fV);

    alpha = rho/InnerProduct(fR_hat, fV);

    for(unsigned int i=0; i<fDim; i++)
    {
        fS[i] = fR[i] - alpha*fV[i];
    }

    fA.Multiply(fS, fT);

    omega = InnerProduct(fT,fS)/InnerProduct(fT,fT);

    for(unsigned int i=0; i<fDim; i++)
    {
        fX[i] = fX[i] + alpha*fP[i] + omega*fS[i];
        fR[i] = fS[i] - omega*fT[i];
    }

    //update the previous values of rho
    rho_prev = rho;
}

template< typename ValueType >
void
KBiconjugateGradientStabilized<ValueType>::GetResidualNorm(double& norm)
{
    norm = std::sqrt(InnerProduct(fR,fR));
}

template< typename ValueType >
double
KBiconjugateGradientStabilized<ValueType>::InnerProduct(const Vector& a, const Vector& b)
{
    double result = 0.;

    for(unsigned int i=0; i<fDim; i++)
    {
        result += a(i)*b(i);
    }

    return result;
}

template <typename ValueType>
void
KBiconjugateGradientStabilized<ValueType>::SetResidualVector(const Vector& v)
{
  fR.resize(v.Dimension());

  for (unsigned int i = 0;i<v.Dimension();i++)
    fR[i] = v(i);
}

template <typename ValueType>
void
KBiconjugateGradientStabilized<ValueType>::GetResidualVector(Vector& v) const
{
  for (unsigned int i = 0;i<fR.Dimension();i++)
    v[i] = fR(i);
}

}

#endif /* KBiconjugateGradientStabilized_H__ */
