#ifndef KROBINHOOD_DEF
#define KROBINHOOD_DEF

#include "KIterativeSolver.hh"
#include "KRobinHood_SingleThread.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
template<typename ValueType, template<typename> class ParallelTrait = KRobinHood_SingleThread>
class KRobinHood : public KIterativeSolver<ValueType>
{
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KRobinHood();
    ~KRobinHood() override;

    void Solve(const Matrix& A, Vector& x, const Vector& b);

    void SetResidualCheckInterval(unsigned int i)
    {
        fResidualCheckInterval = i;
    }

    void CoalesceData() override
    {
        if (fTrait)
            fTrait->CoalesceData();
    }

  private:
    unsigned int Dimension() const override
    {
        return (fTrait ? fTrait->Dimension() : 0);
    }
    void SetResidualVector(const Vector& v) override
    {
        if (fTrait)
            fTrait->SetResidualVector(v);
    }
    void GetResidualVector(Vector& v) override
    {
        if (fTrait)
            fTrait->GetResidualVector(v);
    }

    unsigned int fResidualCheckInterval;
    ParallelTrait<ValueType>* fTrait;
};

template<typename ValueType, template<typename> class ParallelTrait>
KRobinHood<ValueType, ParallelTrait>::KRobinHood() : fResidualCheckInterval(0), fTrait(nullptr)
{}

template<typename ValueType, template<typename> class ParallelTrait> KRobinHood<ValueType, ParallelTrait>::~KRobinHood()
{}

template<typename ValueType, template<typename> class ParallelTrait>
void KRobinHood<ValueType, ParallelTrait>::Solve(const Matrix& A, Vector& x, const Vector& b)
{
    if (fResidualCheckInterval == 0)
        fResidualCheckInterval = b.Dimension();

    ParallelTrait<ValueType> trait(A, x, b);
    fTrait = &trait;

    this->InitializeVisitors();

    trait.Initialize();

    unsigned int subIteration = 0;

    while (this->fResidualNorm > this->fTolerance && !(this->Terminate())) {
        subIteration++;
        trait.FindResidual();
        if (subIteration == fResidualCheckInterval) {
            subIteration = 0;
            this->fIteration++;
            trait.FindResidualNorm(this->fResidualNorm);

            this->AcceptVisitors();
        }
        trait.IdentifyLargestResidualElement();
        trait.ComputeCorrection();
        trait.UpdateSolutionApproximation();
        trait.UpdateVectorApproximation();
    }

    trait.Finalize();

    this->FinalizeVisitors();

    fTrait = nullptr;
}

}  // namespace KEMField

#endif /* KROBINHOOD_DEF */
