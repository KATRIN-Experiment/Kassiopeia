#ifndef KPreconditionedBiconjugateGradientStabilized_HH__
#define KPreconditionedBiconjugateGradientStabilized_HH__

#include "KBiconjugateGradientStabilizedState.hh"
#include "KMatrix.hh"
#include "KPreconditioner.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

#include <cmath>
#include <iostream>

namespace KEMField
{

/*
*
*@file KPreconditionedBiconjugateGradientStabilized.hh
*@class KPreconditionedBiconjugateGradientStabilized
*@brief
*@details: note only valid for real types, do not use with ValueType = std::complex<...>
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Feb  1 10:20:58 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType> class KPreconditionedBiconjugateGradientStabilized
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;
    using Preconditioner = KPreconditioner<ValueType>;

    using KSimpleMatrixType = KSimpleMatrix<ValueType>;
    using KSimpleVectorType = KSimpleVector<ValueType>;

    KPreconditionedBiconjugateGradientStabilized(const Matrix& A, Preconditioner& P, Vector& x, const Vector& b) :
        fDim(A.Dimension()),
        fA(A),
        fPreconditioner(P),
        fX(x),
        fB(b)
    {
        fExternalStateSet = false;
    };

    virtual ~KPreconditionedBiconjugateGradientStabilized() = default;
    ;


    static std::string Name()
    {
        return std::string("pbicgstab");
    }
    std::string NameLabel()
    {
        return std::string("pbicgstab");
    }

    const KBiconjugateGradientStabilizedState<ValueType>& GetState() const;
    void SetState(const KBiconjugateGradientStabilizedState<ValueType>& state);

    void Initialize();
    void ResetAndInitialize();
    void AugmentKrylovSubspace();
    void UpdateSolution(){};  //performed in the krylov step
    void GetResidualNorm(double& norm);
    void CoalesceData(){};
    void Finalize(){};

    unsigned int Dimension() const
    {
        return fDim;
    };

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    double InnerProduct(const Vector& a, const Vector& b);
    void ReconstructState();

    unsigned int fDim;
    const Matrix& fA;
    Preconditioner& fPreconditioner;
    Vector& fX;
    const Vector& fB;

    double rho;
    double beta;
    double alpha;
    double omega;
    double rho_prev;

    KSimpleVectorType fR;
    KSimpleVectorType fR_hat;
    KSimpleVectorType fP;
    KSimpleVectorType fV;

    KSimpleVectorType fS;
    KSimpleVectorType fT;

    KSimpleVectorType fY;
    KSimpleVectorType fZ;
    KSimpleVectorType fPT;


    KSimpleVectorType fTempA;
    KSimpleVectorType fTempB;

    mutable KBiconjugateGradientStabilizedState<ValueType> fState;
    bool fExternalStateSet;
};

template<typename ValueType> void KPreconditionedBiconjugateGradientStabilized<ValueType>::Initialize()
{
    if (fExternalStateSet)  //we have data from a previous run of the same process
    {
        ReconstructState();
    }
    else {
        //no previous state to load, go ahead
        ResetAndInitialize();
    }
}

template<typename ValueType> void KPreconditionedBiconjugateGradientStabilized<ValueType>::ResetAndInitialize()
{
    fR.resize(fDim, 0.);
    fR_hat.resize(fDim, 0.);
    fP.resize(fDim, 0.);
    fV.resize(fDim, 0.);
    fS.resize(fDim, 0.);
    fT.resize(fDim, 0.);
    fY.resize(fDim, 0.);
    fZ.resize(fDim, 0.);
    fPT.resize(fDim, 0.);
    fTempA.resize(fDim, 0.);
    fTempB.resize(fDim, 0.);

    //first we compute the initial residual vector: r = b - Ax
    fA.Multiply(fX, fV);
    for (unsigned int i = 0; i < fDim; i++) {
        fR[i] = fB(i) - fV[i];
        fR_hat[i] = fR[i];
    }

    rho = 1.;
    rho_prev = 1.;
    alpha = 1.;
    omega = 1.;

    //we take the first conjugate vector to be the residual
    for (unsigned int i = 0; i < fDim; i++) {
        fV[i] = 0.;
        fP[i] = 0.;
    }
}


template<typename ValueType> void KPreconditionedBiconjugateGradientStabilized<ValueType>::AugmentKrylovSubspace()
{
    rho = InnerProduct(fR_hat, fR);
    beta = (rho / rho_prev) * (alpha / omega);

    for (unsigned int i = 0; i < fDim; i++) {
        fP[i] = fR[i] + beta * (fP[i] - omega * fV[i]);
    }

    //apply the preconditioner
    fPreconditioner.Multiply(fP, fY);

    //apply the system matrix V = Ay
    fA.Multiply(fY, fV);

    alpha = rho / InnerProduct(fR_hat, fV);

    for (unsigned int i = 0; i < fDim; i++) {
        fS[i] = fR[i] - alpha * fV[i];
    }

    //apply the preconditioner
    fPreconditioner.Multiply(fS, fZ);

    //apply the system matrix t = Az
    fA.Multiply(fZ, fT);

    //apply the preconditioner to T
    fPreconditioner.Multiply(fT, fPT);

    //omega = < K^{-1}t, K^{-1}s >/< K^{-1}t, K^{-1}t >
    omega = InnerProduct(fPT, fZ) / InnerProduct(fPT, fPT);

    for (unsigned int i = 0; i < fDim; i++) {
        fX[i] = fX[i] + alpha * fY[i] + omega * fZ[i];
        fR[i] = fS[i] - omega * fT[i];
    }

    //update the previous values of rho
    rho_prev = rho;
}


template<typename ValueType> void KPreconditionedBiconjugateGradientStabilized<ValueType>::GetResidualNorm(double& norm)
{
    norm = std::sqrt(InnerProduct(fR, fR));
}

template<typename ValueType>
double KPreconditionedBiconjugateGradientStabilized<ValueType>::InnerProduct(const Vector& a, const Vector& b)
{
    double result = 0.;

    for (unsigned int i = 0; i < fDim; i++) {
        result += a(i) * b(i);
    }

    return result;
}

template<typename ValueType>
void KPreconditionedBiconjugateGradientStabilized<ValueType>::SetResidualVector(const Vector& v)
{
    fR.resize(v.Dimension());

    for (unsigned int i = 0; i < v.Dimension(); i++)
        fR[i] = v(i);
}

template<typename ValueType>
void KPreconditionedBiconjugateGradientStabilized<ValueType>::GetResidualVector(Vector& v) const
{
    for (unsigned int i = 0; i < fR.Dimension(); i++)
        v[i] = fR(i);
}


template<typename ValueType>
const KBiconjugateGradientStabilizedState<ValueType>&
KPreconditionedBiconjugateGradientStabilized<ValueType>::GetState() const
{
    fState.SetDimension(fDim);

    //have to handle x and b specially
    //fill temp vector with x fState
    KSimpleVectorType temp;
    temp.resize(fDim);
    for (unsigned int i = 0; i < fDim; i++) {
        temp[i] = fX(i);
    }
    fState.SetSolutionVector(&temp);

    for (unsigned int i = 0; i < fDim; i++) {
        temp[i] = fB(i);
    }
    fState.SetRightHandSide(&temp);

    return fState;
}

template<typename ValueType>
void KPreconditionedBiconjugateGradientStabilized<ValueType>::SetState(
    const KBiconjugateGradientStabilizedState<ValueType>& state)
{
    fState.SetDimension(state.GetDimension());
    const KSimpleVector<ValueType>* temp;

    temp = state.GetSolutionVector();
    fState.SetSolutionVector(temp);

    temp = state.GetRightHandSide();
    fState.SetRightHandSide(temp);

    fExternalStateSet = true;
}


template<typename ValueType> void KPreconditionedBiconjugateGradientStabilized<ValueType>::ReconstructState()
{
    if (fExternalStateSet) {
        fDim = fState.GetDimension();

        fR.resize(fDim, 0.);
        fR_hat.resize(fDim, 0.);
        fP.resize(fDim, 0.);
        fV.resize(fDim, 0.);
        fS.resize(fDim, 0.);
        fT.resize(fDim, 0.);
        fY.resize(fDim, 0.);
        fZ.resize(fDim, 0.);
        fPT.resize(fDim, 0.);
        fTempA.resize(fDim, 0.);
        fTempB.resize(fDim, 0.);

        const KSimpleVectorType* temp;
        temp = fState.GetSolutionVector();
        for (unsigned int i = 0; i < temp->size(); i++) {
            fX[i] = (*temp)(i);
        };

        //first we compute the initial residual vector: r = b - Ax
        fA.Multiply(fX, fV);
        for (unsigned int i = 0; i < fDim; i++) {
            fR[i] = fB(i) - fV[i];
            fR_hat[i] = fR[i];
        }

        rho = 1.;
        rho_prev = 1.;
        alpha = 1.;
        omega = 1.;

        //we take the first conjugate vector to be the residual
        for (unsigned int i = 0; i < fDim; i++) {
            fV[i] = 0.;
            fP[i] = 0.;
        }
    }
}


template<typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KPreconditionedBiconjugateGradientStabilized<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    KBiconjugateGradientStabilizedState<ValueType> state;
    s >> state;
    aData.SetState(state);

    s.PostStreamInAction(aData);
    return s;
}


template<typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KPreconditionedBiconjugateGradientStabilized<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetState();

    s.PostStreamOutAction(aData);

    return s;
}


}  // namespace KEMField

#endif /* KPreconditionedBiconjugateGradientStabilized_H__ */
