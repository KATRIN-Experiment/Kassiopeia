#ifndef KROBINHOOD_SINGLETHREAD_DEF
#define KROBINHOOD_SINGLETHREAD_DEF

#include "KSimpleVector.hh"
#include "KSquareMatrix.hh"
#include "KEMCoreMessage.hh"

namespace KEMField
{
template<typename ValueType> class KRobinHood_SingleThread
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KRobinHood_SingleThread(const Matrix& A, Vector& x, const Vector& b);
    ~KRobinHood_SingleThread() {}

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void IdentifyLargestResidualElement();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData() {}
    void Finalize() {}

    unsigned int Dimension() const
    {
        return fB.Dimension();
    }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    KSimpleVector<ValueType> fB_iterative;
    KSimpleVector<ValueType> fResidual;

    double fBInfinityNorm;

    unsigned int fMaxResidualIndex;

    ValueType fCorrection;
};

template<typename ValueType>
KRobinHood_SingleThread<ValueType>::KRobinHood_SingleThread(const Matrix& A, Vector& x, const Vector& b) :
    fA(A),
    fX(x),
    fB(b)
{}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::Initialize()
{
    if (fResidual.Dimension() == 0) {
        fB_iterative.resize(fB.Dimension(), 0.);
        fResidual.resize(fB.Dimension(), 0.);

        if (fX.InfinityNorm() > 1.e-16)
            fA.Multiply(fX, fB_iterative);
    }

    fBInfinityNorm = fB.InfinityNorm();
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::FindResidual()
{
    for (unsigned int i = 0; i < fB.Dimension(); i++)
        fResidual[i] = fB(i) - fB_iterative(i);
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::FindResidualNorm(double& residualNorm)
{
    residualNorm = fResidual.InfinityNorm() / fBInfinityNorm;
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::IdentifyLargestResidualElement()
{
    fMaxResidualIndex = 0;
    ValueType maxResidual = fabs(fResidual(0));
    for (unsigned int i = 0; i < fResidual.Dimension(); i++) {
        if (fabs(fResidual(i)) > maxResidual) {
            maxResidual = fabs(fResidual(i));
            fMaxResidualIndex = i;
        }
    }
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::ComputeCorrection()
{
    fCorrection = (fB(fMaxResidualIndex) - fB_iterative(fMaxResidualIndex)) / fA(fMaxResidualIndex, fMaxResidualIndex);
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::UpdateSolutionApproximation()
{
    fX[fMaxResidualIndex] += fCorrection;
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::UpdateVectorApproximation()
{
    for (unsigned int i = 0; i < fB_iterative.Dimension(); i++)
        fB_iterative[i] += fA(i, fMaxResidualIndex) * fCorrection;
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::SetResidualVector(const Vector& v)
{
    fResidual.resize(v.Dimension());
    fB_iterative.resize(v.Dimension());

    for (unsigned int i = 0; i < v.Dimension(); i++) {
        fResidual[i] = v(i);
        fB_iterative[i] = fB(i) - fResidual(i);
    }
}

template<typename ValueType> void KRobinHood_SingleThread<ValueType>::GetResidualVector(Vector& v) const
{
    for (unsigned int i = 0; i < fResidual.Dimension(); i++)
        v[i] = fResidual(i);
}

}  // namespace KEMField

#endif /* KROBINHOOD_SINGLETHREAD_DEF */
