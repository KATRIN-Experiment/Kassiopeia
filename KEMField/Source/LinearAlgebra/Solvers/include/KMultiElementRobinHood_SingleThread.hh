#ifndef KMULTIELEMENTROBINHOOD_SINGLETHREAD_DEF
#define KMULTIELEMENTROBINHOOD_SINGLETHREAD_DEF

#include "KGaussianElimination.hh"
#include "KSimpleVector.hh"

namespace KEMField
{
template<typename ValueType> class KMultiElementRobinHood_SingleThread
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KMultiElementRobinHood_SingleThread(const Matrix& A, Vector& x, const Vector& b);
    ~KMultiElementRobinHood_SingleThread() = default;

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void IdentifyLargestResidualElements();
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

    void SetSubspaceDimension(unsigned int i)
    {
        fSubspaceDimension = i;
    }

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    KSimpleVector<ValueType> fB_iterative;
    KSimpleVector<ValueType> fResidual;

    double fBMax2;

    KSimpleVector<unsigned int> fMaxResidualIndex;

    class SubMatrix : public KSquareMatrix<ValueType>
    {
      public:
        SubMatrix(const KSquareMatrix<ValueType>& parentMatrix, const KSimpleVector<unsigned int>& subspace) :
            fSubspace(subspace),
            fParentMatrix(parentMatrix)
        {}
        ~SubMatrix() override = default;

        unsigned int Dimension() const override
        {
            return fSubspace.Dimension();
        }
        const ValueType& operator()(unsigned int, unsigned int) const override;

      private:
        const KSimpleVector<unsigned int>& fSubspace;
        const KSquareMatrix<ValueType>& fParentMatrix;
    };

    class SubVector : public KVector<ValueType>
    {
      public:
        SubVector(KVector<ValueType>& parentVector, const KSimpleVector<unsigned int>& subspace) :
            fSubspace(subspace),
            fParentVector(parentVector)
        {}
        ~SubVector() override = default;

        unsigned int Dimension() const override
        {
            return fSubspace.Dimension();
        }

        const ValueType& operator()(unsigned int) const override;
        ValueType& operator[](unsigned int) override;

      private:
        const KSimpleVector<unsigned int>& fSubspace;
        KVector<ValueType>& fParentVector;
    };


    unsigned int fSubspaceDimension;
    KGaussianElimination<ValueType> fSubSolver;
    const SubMatrix fSubA;
    const SubVector fSubResidual;
    KSimpleVector<ValueType> fCorrection;
};

template<typename ValueType>
const ValueType& KMultiElementRobinHood_SingleThread<ValueType>::SubMatrix::operator()(unsigned int i,
                                                                                       unsigned int j) const
{
    return fParentMatrix(fSubspace(i), fSubspace(j));
}

template<typename ValueType>
const ValueType& KMultiElementRobinHood_SingleThread<ValueType>::SubVector::operator()(unsigned int i) const
{
    return fParentVector(fSubspace(i));
}

template<typename ValueType>
ValueType& KMultiElementRobinHood_SingleThread<ValueType>::SubVector::operator[](unsigned int i)
{
    return fParentVector[fSubspace(i)];
}

template<typename ValueType>
KMultiElementRobinHood_SingleThread<ValueType>::KMultiElementRobinHood_SingleThread(const Matrix& A, Vector& x,
                                                                                    const Vector& b) :
    fA(A),
    fX(x),
    fB(b),
    fSubspaceDimension(2),
    fSubA(A, fMaxResidualIndex),
    fSubResidual(fResidual, fMaxResidualIndex)
{}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::Initialize()
{
    if (fResidual.Dimension() == 0) {
        fB_iterative.resize(fB.Dimension(), 0.);
        fResidual.resize(fB.Dimension(), 0.);

        if (fX.InfinityNorm() > 1.e-16)
            fA.Multiply(fX, fB_iterative);
    }

    fBMax2 = 0.;
    for (unsigned int i = 0; i < fB.Dimension(); i++) {
        double b2 = fB(i) * fB(i);
        if (b2 > fBMax2)
            fBMax2 = b2;
    }

    fCorrection.resize(fSubspaceDimension);
    fMaxResidualIndex.resize(fSubspaceDimension);
    for (unsigned int i = 0; i < fSubspaceDimension; i++) {
        fCorrection[i] = 0.;
        fMaxResidualIndex[i] = i;
    }
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::FindResidual()
{
    for (unsigned int i = 0; i < fB.Dimension(); i++)
        fResidual[i] = fB(i) - fB_iterative(i);
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::FindResidualNorm(double& residualNorm)
{
    residualNorm = 0.;
    for (unsigned int i = 0; i < fResidual.Dimension(); i++)
        residualNorm += fResidual(i) * fResidual(i);

    residualNorm = sqrt(residualNorm / fResidual.Dimension()) / fBMax2;
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::IdentifyLargestResidualElements()
{
    for (unsigned int i = 0; i < fSubspaceDimension; i++)
        fMaxResidualIndex[i] = i;

    for (unsigned int i = fSubspaceDimension; i < fResidual.Dimension(); i++) {
        for (unsigned int j = 0; j < fSubspaceDimension; j++) {
            if (fabs(fResidual(i)) > fabs(fSubResidual(j))) {
                for (unsigned int k = j + 1; k < fSubspaceDimension; k++) {
                    if (fabs(fSubResidual(j)) > fabs(fSubResidual(k))) {
                        fMaxResidualIndex[k] = fMaxResidualIndex[j];
                        break;
                    }
                }
                fMaxResidualIndex[j] = i;
                break;
            }
        }
    }
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::ComputeCorrection()
{
    fSubSolver.Solve(fSubA, fCorrection, fSubResidual);
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::UpdateSolutionApproximation()
{
    for (unsigned int i = 0; i < fSubspaceDimension; i++)
        fX[fMaxResidualIndex(i)] += fCorrection(i);
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::UpdateVectorApproximation()
{
    for (unsigned int i = 0; i < fB_iterative.Dimension(); i++) {
        for (unsigned int j = 0; j < fSubspaceDimension; j++) {
            fB_iterative[i] += fA(i, fMaxResidualIndex(j)) * fCorrection(j);
        }
    }
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::SetResidualVector(const Vector& v)
{
    fResidual.resize(v.Dimension());
    fB_iterative.resize(v.Dimension());

    for (unsigned int i = 0; i < v.Dimension(); i++) {
        fResidual[i] = v(i);
        fB_iterative[i] = fB(i) - fResidual(i);
    }
}

template<typename ValueType> void KMultiElementRobinHood_SingleThread<ValueType>::GetResidualVector(Vector& v) const
{
    for (unsigned int i = 0; i < fResidual.Dimension(); i++)
        v[i] = fResidual(i);
}
}  // namespace KEMField

#endif /* KMULTIELEMENTROBINHOOD_SINGLETHREAD_DEF */
