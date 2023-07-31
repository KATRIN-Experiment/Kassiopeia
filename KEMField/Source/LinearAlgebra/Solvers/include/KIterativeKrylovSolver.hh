#ifndef KIterativeKrylovSolver_HH__
#define KIterativeKrylovSolver_HH__

#include "KIterativeKrylovRestartCondition.hh"
#include "KIterativeSolver.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

#include <memory>

namespace KEMField
{


/*
*
*@file KIterativeKrylovSolver.hh
*@class KIterativeKrylovSolver
*@brief controller class for iterative krylov solvers
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 31 15:27:04 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/
template<typename ValueType> class KIterativeKrylovSolver : public KIterativeSolver<ValueType>
{
  public:
    KIterativeKrylovSolver() : fMaxIterations(UINT_MAX)
    {
        //create a default restart condition
        fRestartCondition = std::make_shared<KIterativeKrylovRestartCondition>();
    }
    ~KIterativeKrylovSolver() override = default;

    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    void Solve(Vector& x, const Vector& b)
    {
        SolveCore(x, b);
    }

    void SetMatrix(std::shared_ptr<const Matrix> A)
    {
        fMatrix = A;
    }

    void SetMaximumIterations(unsigned int i)
    {
        fMaxIterations = i;
    }
    void SetRestartCondition(const std::shared_ptr<KIterativeKrylovRestartCondition>& restart)
    {
        fRestartCondition = restart;
    }

  protected:
    std::shared_ptr<const Matrix> GetMatrix() const
    {
        return fMatrix;
    }

    unsigned int GetMaximumIterations()
    {
        return fMaxIterations;
    }
    std::shared_ptr<KIterativeKrylovRestartCondition> GetRestartCondition()
    {
        return fRestartCondition;
    }

  private:
    virtual void SolveCore(Vector& x, const Vector& b) = 0;

    unsigned int fMaxIterations;
    std::shared_ptr<KIterativeKrylovRestartCondition> fRestartCondition;
    std::shared_ptr<const Matrix> fMatrix;
};


}  // namespace KEMField

#endif /* KIterativeKrylovSolver_H__ */
