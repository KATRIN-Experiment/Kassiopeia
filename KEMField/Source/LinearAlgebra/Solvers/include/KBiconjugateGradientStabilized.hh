#ifndef KBiconjugateGradientStabilized_HH__
#define KBiconjugateGradientStabilized_HH__

#include <cmath>
#include <iostream>

#include "KMatrix.hh"
#include "KVector.hh"

#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

#include "KBiconjugateGradientStabilizedState.hh"

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

        typedef KSimpleMatrix<ValueType> KSimpleMatrixType;
        typedef KSimpleVector<ValueType> KSimpleVectorType;

        KBiconjugateGradientStabilized(const Matrix& A, Vector& x, const Vector& b):
        fA(A),
        fX(x),
        fB(b)
        {
            fDim = fB.Dimension();
            fExternalStateSet = false;
        };

        virtual ~KBiconjugateGradientStabilized(){};

        static std::string Name() { return std::string("bicgstab"); }
        std::string NameLabel() { return std::string("bicgstab"); }

        const KBiconjugateGradientStabilizedState<ValueType>& GetState() const;
        void SetState(const KBiconjugateGradientStabilizedState<ValueType>& state);

        void Initialize();
        void ResetAndInitialize();
        void AugmentKrylovSubspace();
        void UpdateSolution(){}; //solution is updated in krylov subspace step

        void GetResidualNorm(ValueType& norm);
        void CoalesceData(){};
        void Finalize(){};

        unsigned int Dimension() const {return fDim;};

        void SetResidualVector(const Vector&){};
        void GetResidualVector(Vector&) const;

    private:

        ValueType InnerProduct(const Vector& a, const Vector& b);
        void ReconstructState();

        unsigned int fDim;
        const Matrix& fA;
        Vector& fX;
        const Vector& fB;

        ValueType rho;
        ValueType beta;
        ValueType alpha;
        ValueType omega;
        ValueType rho_prev;

        KSimpleVector<ValueType> fR;
        KSimpleVector<ValueType> fR_hat;
        KSimpleVector<ValueType> fP;
        KSimpleVector<ValueType> fV;

        KSimpleVector<ValueType> fS;
        KSimpleVector<ValueType> fT;

        mutable KBiconjugateGradientStabilizedState<ValueType> fState;
        bool fExternalStateSet;

};

template< typename ValueType >
void
KBiconjugateGradientStabilized<ValueType>::Initialize()
{
    if(fExternalStateSet) //we have data from a previous run of the same process
    {
        ReconstructState();
    }
    else
    {
        //no previous state to load, go ahead
        ResetAndInitialize();
    }
}


template< typename ValueType >
void
KBiconjugateGradientStabilized<ValueType>::ResetAndInitialize()
{
    fR.resize(fDim, 0.);
    fR_hat.resize(fDim, 0.);
    fP.resize(fDim, 0.);
    fV.resize(fDim, 0.);
    fS.resize(fDim, 0.);
    fT.resize(fDim, 0.);

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
KBiconjugateGradientStabilized<ValueType>::GetResidualNorm(ValueType& norm)
{
    norm = std::sqrt(InnerProduct(fR,fR));
}

template< typename ValueType >
ValueType
KBiconjugateGradientStabilized<ValueType>::InnerProduct(const Vector& a, const Vector& b)
{
    ValueType result = 0.;

    for(unsigned int i=0; i<fDim; i++)
    {
        result += a(i)*b(i);
    }

    return result;
}

template <typename ValueType>
void
KBiconjugateGradientStabilized<ValueType>::GetResidualVector(Vector& v) const
{
  for (unsigned int i = 0;i<fR.Dimension();i++)
    v[i] = fR(i);
}


template <typename ValueType>
const KBiconjugateGradientStabilizedState<ValueType>&
KBiconjugateGradientStabilized<ValueType>::GetState() const
{
    fState.SetDimension(fDim);

    //have to handle x and b specially
    //fill temp vector with x fState
    KSimpleVectorType temp; temp.resize(fDim);
    for(unsigned int i=0; i<fDim; i++)
    {
        temp[i] = fX(i);
    }
    fState.SetSolutionVector(&temp);

    for(unsigned int i=0; i<fDim; i++)
    {
        temp[i] = fB(i);
    }
    fState.SetRightHandSide(&temp);

    return fState;
}

template <typename ValueType>
void
KBiconjugateGradientStabilized<ValueType>::SetState(const KBiconjugateGradientStabilizedState<ValueType>& state)
{
    fState.SetDimension(state.GetDimension());
    const KSimpleVector<ValueType>* temp;

    temp = state.GetSolutionVector();
    fState.SetSolutionVector(temp);

    temp = state.GetRightHandSide();
    fState.SetRightHandSide(temp);

    fExternalStateSet = true;
}



template <typename ValueType>
void
KBiconjugateGradientStabilized<ValueType>::ReconstructState()
{
    if(fExternalStateSet)
    {
        fDim = fState.GetDimension();

        fR.resize(fDim, 0.);
        fR_hat.resize(fDim, 0.);
        fP.resize(fDim, 0.);
        fV.resize(fDim, 0.);
        fS.resize(fDim, 0.);
        fT.resize(fDim, 0.);

        const KSimpleVectorType* temp;
        temp = fState.GetSolutionVector();
        for(unsigned int i=0; i<temp->size(); i++){fX[i] = (*temp)(i); };

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
}


template <typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KBiconjugateGradientStabilized<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    KBiconjugateGradientStabilizedState<ValueType> state;
    s >> state;
    aData.SetState(state);

    s.PostStreamInAction(aData);
    return s;
}


template <typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KBiconjugateGradientStabilized<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetState();

    s.PostStreamOutAction(aData);

    return s;
}


}

#endif /* KBiconjugateGradientStabilized_H__ */
