#ifndef KGAUSSSEIDEL_DEF
#define KGAUSSSEIDEL_DEF

#include "KGaussSeidel_SingleThread.hh"
#include "KIterativeSolver.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
template<typename ValueType, template<typename> class ParallelTrait = KGaussSeidel_SingleThread>
class KGaussSeidel : public KIterativeSolver<ValueType>
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector =  KVector<ValueType>;

    KGaussSeidel();
    ~KGaussSeidel() override;

    void Solve(const Matrix& A, Vector& x, const Vector& b);

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

    ParallelTrait<ValueType>* fTrait;
};

template<typename ValueType, template<typename> class ParallelTrait>
KGaussSeidel<ValueType, ParallelTrait>::KGaussSeidel() : fTrait(nullptr)
{}

template<typename ValueType, template<typename> class ParallelTrait>
KGaussSeidel<ValueType, ParallelTrait>::~KGaussSeidel()
{}

template<typename ValueType, template<typename> class ParallelTrait>
void KGaussSeidel<ValueType, ParallelTrait>::Solve(const Matrix& A, Vector& x, const Vector& b)
{
    ParallelTrait<ValueType> trait(A, x, b);
    fTrait = &trait;

    this->InitializeVisitors();

    trait.Initialize();

    unsigned int subIteration = 0;

    while (this->fResidualNorm > this->fTolerance && !(this->Terminate())) {
        subIteration++;
        trait.FindResidual();
        if (subIteration == Dimension()) {
            subIteration = 0;
            this->fIteration++;
            trait.FindResidualNorm(this->fResidualNorm);

            this->AcceptVisitors();
        }
        trait.IncrementIndex();
        trait.ComputeCorrection();
        trait.UpdateSolutionApproximation();
        trait.UpdateVectorApproximation();
    }

    trait.Finalize();

    this->FinalizeVisitors();

    fTrait = nullptr;
}

}  // namespace KEMField

#endif /* KGAUSSSEIDEL_DEF */
