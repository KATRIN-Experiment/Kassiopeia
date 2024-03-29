#ifndef __KImplicitKrylovPreconditioner_H__
#define __KImplicitKrylovPreconditioner_H__

#include "KFMMessaging.hh"
#include "KIterativeKrylovSolver.hh"
#include "KKrylovSolverFactory.hh"
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

/**
 *
 *@file KImplicitKrylovPreconditioner.hh
 *@class KImplicitKrylovPreconditioner
 *@brief
 *@details
 *<b>Revision History:<b>
 *Date Name Brief Description
 *Tue Sep  2 10:56:25 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
 *
 */

template<typename ValueType> class KImplicitKrylovPreconditioner : public KPreconditioner<ValueType>
{
  public:
    KImplicitKrylovPreconditioner(std::shared_ptr<KIterativeKrylovSolver<ValueType>> solver) : fZero(0.), fSolver(solver)
    {}

    ~KImplicitKrylovPreconditioner() override = default;

    std::string Name() override
    {
        return std::string("implicit_krylov");
    }

    std::shared_ptr<KIterativeKrylovSolver<ValueType>> GetSolver()
    {
        return fSolver;
    }

    void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        y.Fill(0.);
        fSolver->Solve(y, x);
    }

    void MultiplyTranspose(const KVector<ValueType>& /*x*/, KVector<ValueType>& /*y*/) const override
    {
        //this function must be present, but its not defined
        kfmout << "KImplicitKrylovPreconditioner::MultiplyTranspose: Fatal error, this function is not implemented."
               << kfmendl;
        kfmexit(1);
    }

    //while the matrix defining the preconditioner is fixed
    //because we must implicity solve for the inverse action of this matrix
    //we consider this to be a non-stationary preconditioner
    bool IsStationary() override
    {
        return false;
    };

    unsigned int Dimension() const override
    {
        return fSolver->Dimension();
    };

    const ValueType& operator()(unsigned int /*i*/, unsigned int /*j*/) const override
    {
        //This function always returns zero as an implicit (non-stationary)
        //preconditioner cannot easily compute fixed matrix elements of the inverse action
        return fZero;
    }

  protected:
    ValueType fZero;

    std::shared_ptr<KIterativeKrylovSolver<ValueType>> fSolver;
};

template<typename ValueType>
std::shared_ptr<KPreconditioner<ValueType>>
KBuildKrylovPreconditioner(std::shared_ptr<KIterativeKrylovSolver<ValueType>> solver)
{
    return std::make_shared<KImplicitKrylovPreconditioner<ValueType>>(solver);
}

template<typename ValueType>
std::shared_ptr<KPreconditioner<ValueType>>
KBuildKrylovPreconditioner(const KKrylovSolverConfiguration& config,
                           std::shared_ptr<const KSquareMatrix<ValueType>> matrix,
                           std::shared_ptr<const KSquareMatrix<ValueType>> preconditioner = nullptr)
{
    auto solver = KBuildKrylovSolver(config, matrix, preconditioner);
    return KBuildKrylovPreconditioner<ValueType>(solver);
}

}  // namespace KEMField

#endif /* __KImplicitKrylovPreconditioner_H__ */
