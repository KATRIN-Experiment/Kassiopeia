#ifndef KPreconditionedIterativeKrylovSolver_HH__
#define KPreconditionedIterativeKrylovSolver_HH__

#include "KIterativeKrylovRestartCondition.hh"
#include "KIterativeKrylovSolver.hh"
#include "KIterativeSolver.hh"
#include "KPreconditioner.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

namespace KEMField
{

/*
*
*@file KPreconditionedIterativeKrylovSolver.hh
*@class KPreconditionedIterativeKrylovSolver
*@brief controller class for right-preconditioned krylov solvers
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 31 15:27:04 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType, template<typename> class ParallelTrait>
class KPreconditionedIterativeKrylovSolver : public KIterativeKrylovSolver<ValueType>
{
  public:
    using typename KIterativeKrylovSolver<ValueType>::Matrix;
    using typename KIterativeKrylovSolver<ValueType>::Vector;
    typedef KPreconditioner<ValueType> Preconditioner;

    KPreconditionedIterativeKrylovSolver();
    ~KPreconditionedIterativeKrylovSolver() override;

    void Solve(const Matrix& A, Preconditioner& P, Vector& x, const Vector& b);

    void SetPreconditioner(KSmartPointer<Preconditioner> preconditioner)
    {
        fPreconditioner = preconditioner;
    }

    //set tolerance (default is absolute tolerance on l2 residual norm)
    void SetTolerance(double d) override
    {
        SetRelativeTolerance(d);
    };
    virtual void SetAbsoluteTolerance(double d)
    {
        this->fTolerance = d;
        fUseRelativeTolerance = false;
    };
    virtual void SetRelativeTolerance(double d)
    {
        this->fTolerance = d;
        fUseRelativeTolerance = true;
    };

    double ResidualNorm() const override
    {
        if (fUseRelativeTolerance) {
            return (this->fResidualNorm) / fInitialResidual;
        }
        else {
            return this->fResidualNorm;
        }
    }

    ParallelTrait<ValueType>* GetTrait()
    {
        return fTrait;
    };

  private:
    void SolveCore(Vector& x, const Vector& b) override;

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

    virtual bool HasConverged() const;

    double InnerProduct(const Vector& a, const Vector& b);

    ParallelTrait<ValueType>* fTrait;
    bool fUseRelativeTolerance;
    double fInitialResidual;

    KSmartPointer<Preconditioner> fPreconditioner;
};

template<typename ValueType, template<typename> class ParallelTrait>
KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::KPreconditionedIterativeKrylovSolver() : fTrait(nullptr)
{
    fUseRelativeTolerance = false;
    fInitialResidual = 1.0;
};

template<typename ValueType, template<typename> class ParallelTrait>
KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::~KPreconditionedIterativeKrylovSolver() = default;
;

template<typename ValueType, template<typename> class ParallelTrait>
void KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::Solve(const Matrix& A, Preconditioner& P,
                                                                           Vector& x, const Vector& b)
{
    ParallelTrait<ValueType> trait(A, P, x, b);
    fTrait = &trait;

    this->InitializeVisitors();
    this->fIteration = 0;

    trait.Initialize();

    //needed if we are using relative tolerance as convergence condition
    fInitialResidual = std::sqrt(InnerProduct(b, b));
    bool solutionUpdated = false;

    do {
        solutionUpdated = false;
        trait.AugmentKrylovSubspace();
        double residualNorm;
        trait.GetResidualNorm(residualNorm);
        if (this->fIteration > 0 && residualNorm > this->fResidualNorm*2.)
            kem_cout(eWarning) << "Convergence problem, |Residual| increased by " << (100*residualNorm/this->fResidualNorm) << "%" << eom;
        this->fResidualNorm = residualNorm;
        this->GetRestartCondition()->UpdateProgress(this->fResidualNorm);
        this->fIteration++;

        if (this->GetRestartCondition()->PerformRestart()) {
            trait.UpdateSolution();
            trait.ResetAndInitialize();  //clears krylov subspace vectors, restarts from current solution
        }
        else if (HasConverged() || (this->fIteration >= this->GetMaximumIterations())) {
            trait.UpdateSolution();
            solutionUpdated = true;
            break;
        }
        else if (this->Terminate()) {
            trait.UpdateSolution();
            solutionUpdated = true;
            break;
        }

        this->AcceptVisitors();
    } while (!(HasConverged()) && (this->fIteration < this->GetMaximumIterations()) && !(this->Terminate()));

    if (!solutionUpdated) {
        trait.UpdateSolution();
    };

    trait.Finalize();
    this->FinalizeVisitors();

    fTrait = nullptr;
}

template<typename ValueType, template<typename> class ParallelTrait>
void KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::SolveCore(Vector& x, const Vector& b)
{
    Solve(*(this->GetMatrix()), *fPreconditioner, x, b);
}

template<typename ValueType, template<typename> class ParallelTrait>
inline bool KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::HasConverged() const
{
    if (fUseRelativeTolerance) {
        double relative_residual_norm = (this->fResidualNorm) / (fInitialResidual);
        if (relative_residual_norm > this->fTolerance) {
            return false;
        }
        else {
            return true;
        }
    }
    else {
        if (this->fResidualNorm > this->fTolerance) {
            return false;
        }
        else {
            return true;
        }
    }
}

template<typename ValueType, template<typename> class ParallelTrait>
double KPreconditionedIterativeKrylovSolver<ValueType, ParallelTrait>::InnerProduct(const Vector& a, const Vector& b)
{
    double result = 0.;

    unsigned int dim = a.Dimension();
    for (unsigned int i = 0; i < dim; i++) {
        result += a(i) * b(i);
    }

    return result;
}

}  // namespace KEMField

#endif /* KPreconditionedIterativeKrylovSolver_H__ */
