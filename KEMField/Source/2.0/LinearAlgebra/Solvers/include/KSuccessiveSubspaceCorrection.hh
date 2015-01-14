#ifndef KSUCCESSIVESUBSPACECORRECTION_DEF
#define KSUCCESSIVESUBSPACECORRECTION_DEF

#include "KSquareMatrix.hh"
#include "KVector.hh"

#include "KIterativeSolver.hh"

#include "KSuccessiveSubspaceCorrection_SingleThread.hh"

namespace KEMField
{
  template <typename ValueType, template <typename> class ParallelTrait = KSuccessiveSubspaceCorrection_SingleThread>
  class KSuccessiveSubspaceCorrection : public KIterativeSolver<ValueType>
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KSuccessiveSubspaceCorrection();
    virtual ~KSuccessiveSubspaceCorrection();

    void SetSubspaceDimension(unsigned int i) { fSubspaceDimension = i; }

    void Solve(const Matrix& A,Vector& x,const Vector& b);

    void SetResidualCheckInterval(unsigned int i) { fResidualCheckInterval = i; }

    void CoalesceData() { if (fTrait) fTrait->CoalesceData(); }

  private:
    unsigned int Dimension() const { return (fTrait ? fTrait->Dimension() : 0); }
    void SetResidualVector(const Vector& v) { if (fTrait) fTrait->SetResidualVector(v); }
    void GetResidualVector(Vector& v) { if (fTrait) fTrait->GetResidualVector(v); }

    unsigned int fResidualCheckInterval;
    unsigned int fSubspaceDimension;
    ParallelTrait<ValueType>* fTrait;
  };

  template <typename ValueType,template <typename> class ParallelTrait>
  KSuccessiveSubspaceCorrection<ValueType,ParallelTrait>::KSuccessiveSubspaceCorrection()
    : fResidualCheckInterval(0), fSubspaceDimension(2), fTrait(NULL)
  {
  }

  template <typename ValueType,template <typename> class ParallelTrait>
  KSuccessiveSubspaceCorrection<ValueType,ParallelTrait>::~KSuccessiveSubspaceCorrection()
  {
  }

  template <typename ValueType,template <typename> class ParallelTrait>
  void KSuccessiveSubspaceCorrection<ValueType,ParallelTrait>::Solve(const Matrix& A,
							      Vector& x,
							      const Vector& b)
  {
    if (fResidualCheckInterval == 0)
      fResidualCheckInterval = b.Dimension();

    ParallelTrait<ValueType> trait(A,x,b);
    fTrait = &trait;

    trait.SetSubspaceDimension(fSubspaceDimension);

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
      trait.IdentifyResidualElements();
      trait.ComputeCorrection();
      trait.UpdateSolutionApproximation();
      trait.UpdateVectorApproximation();
    }

    trait.Finalize();

    this->FinalizeVisitors();

    fTrait = NULL;
  }

}

#endif /* KSUCCESSIVESUBSPACECORRECTION_DEF */
