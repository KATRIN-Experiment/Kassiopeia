#ifndef KROBINHOOD_DEF
#define KROBINHOOD_DEF

#include "KSquareMatrix.hh"
#include "KVector.hh"

#include "KIterativeSolver.hh"

#include "KRobinHood_SingleThread.hh"

namespace KEMField
{
  template <typename ValueType, template <typename> class ParallelTrait = KRobinHood_SingleThread>
  class KRobinHood : public KIterativeSolver<ValueType>
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KRobinHood();
    ~KRobinHood();

    void Solve(const Matrix& A,Vector& x,const Vector& b);

    void SetResidualCheckInterval(unsigned int i) { fResidualCheckInterval = i; }

    void CoalesceData() { if (fTrait) fTrait->CoalesceData(); }

  private:
    unsigned int Dimension() const { return (fTrait ? fTrait->Dimension() : 0); }
    void SetResidualVector(const Vector& v) { if (fTrait) fTrait->SetResidualVector(v); }
    void GetResidualVector(Vector& v) { if (fTrait) fTrait->GetResidualVector(v); }

    unsigned int fResidualCheckInterval;
    ParallelTrait<ValueType>* fTrait;
  };

  template <typename ValueType,template <typename> class ParallelTrait>
  KRobinHood<ValueType,ParallelTrait>::KRobinHood()
    : fResidualCheckInterval(0), fTrait(NULL)
  {
  }

  template <typename ValueType,template <typename> class ParallelTrait>
  KRobinHood<ValueType,ParallelTrait>::~KRobinHood()
  {
  }

  template <typename ValueType,template <typename> class ParallelTrait>
  void KRobinHood<ValueType,ParallelTrait>::Solve(const Matrix& A,
						  Vector& x,
						  const Vector& b)
  {
    if (fResidualCheckInterval == 0)
      fResidualCheckInterval = b.Dimension();

    ParallelTrait<ValueType> trait(A,x,b);
    fTrait = &trait;

    this->InitializeVisitors();

    trait.Initialize();

    unsigned int subIteration = 0;

    while (this->fResidualNorm > this->fTolerance && !(this->Terminate()) )
    {
      subIteration++;
      trait.FindResidual();
      if (subIteration == fResidualCheckInterval)
      {
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

    fTrait = NULL;
  }

}

#endif /* KROBINHOOD_DEF */
