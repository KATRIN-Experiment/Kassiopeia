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
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

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

        if (subIteration % fResidualCheckInterval == 0) {
            subIteration = 0;
            double residualNorm;
            trait.FindResidualNorm(residualNorm);
            if (residualNorm > 1e100)
                kem_cout(eError) << "Iterative solve failed to converge, current |Residual|: " << residualNorm << eom;
            else if (this->fIteration > 0 && residualNorm > this->fResidualNorm*2.)
                kem_cout(eWarning) << "Convergence problem, |Residual| increased by " << (100*residualNorm/this->fResidualNorm) << "%" << eom;
            this->fResidualNorm = residualNorm;
            this->fIteration++;
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
