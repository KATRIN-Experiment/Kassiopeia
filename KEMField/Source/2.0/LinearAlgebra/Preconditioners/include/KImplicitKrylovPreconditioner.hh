#ifndef __KImplicitKrylovPreconditioner_H__
#define __KImplicitKrylovPreconditioner_H__

#include <cmath>
#include <iostream>
#include "KFMMessaging.hh"

#include "KMatrix.hh"
#include "KVector.hh"

#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

#include "KPreconditioner.hh"
#include "KIterativeKrylovSolver.hh"

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

template <typename ValueType, template <typename> class ParallelTrait>
class KImplicitKrylovPreconditioner: public KPreconditioner< ValueType >
{
    public:

       KImplicitKrylovPreconditioner(const KSquareMatrix<ValueType>& A):
            fA(A),
            fDimension(A.Dimension()),
            fZero(0.)
            {
                fSolver = new KIterativeKrylovSolver<ValueType, ParallelTrait>();
            };

        virtual ~KImplicitKrylovPreconditioner()
        {
            delete fSolver;
        };

        virtual std::string Name(){ return std::string("ik") + fSolver->GetTrait()->Name(); };

        KIterativeKrylovSolver<ValueType, ParallelTrait>* GetSolver(){return fSolver;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            for(unsigned int i=0; i<fDimension; i++){y[i] = 0.0;};
            fSolver->Solve(fA, y, x);
        }

        virtual void MultiplyTranspose(const KVector<ValueType>& /*x*/, KVector<ValueType>& /*y*/) const
        {
            //this function must be present, but its not defined
            kfmout<<"KImplicitKrylovPreconditioner::MultiplyTranspose: Fatal error, this function is not implemented."<<kfmendl;
            kfmexit(1);
        }

        //while the matrix defining the preconditioner is fixed
        //because we must implicity solve for the inverse action of this matrix
        //we consider this to be a non-stationary preconditioner
        virtual bool IsStationary(){return false;};

        virtual unsigned int Dimension() const {return fDimension;} ;

        virtual const ValueType& operator()(unsigned int /*i*/, unsigned int /*j*/) const
        {
            //This function always returns zero as an implicit (non-stationary)
            //preconditioner cannot easily compute fixed matrix elements of the inverse action
            return fZero;
        }

    protected:

        const KSquareMatrix<ValueType>& fA; //system matrix to be solved on each preconditioning action
        unsigned int fDimension;

        ValueType fZero;

        mutable KIterativeKrylovSolver<ValueType, ParallelTrait>* fSolver;

};


}

#endif /* __KImplicitKrylovPreconditioner_H__ */
