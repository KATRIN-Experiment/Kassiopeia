#ifndef __KImplicitKrylovPreconditioner_H__
#define __KImplicitKrylovPreconditioner_H__

#include <cmath>
#include <iostream>
#include "KFMMessaging.hh"

#include "KSmartPointer.hh"

#include "KMatrix.hh"
#include "KVector.hh"

#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

#include "KPreconditioner.hh"
#include "KIterativeKrylovSolver.hh"

#include "KKrylovSolverFactory.hh"

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

template <typename ValueType>
class KImplicitKrylovPreconditioner: public KPreconditioner< ValueType >
{
public:

	KImplicitKrylovPreconditioner( KSmartPointer<KIterativeKrylovSolver<ValueType> > solver) :
		fZero(0.),
		fSolver(solver)
	{
	}

	virtual ~KImplicitKrylovPreconditioner() {}

	virtual std::string Name(){ return std::string("implicit_krylov"); }

	KSmartPointer< KIterativeKrylovSolver<ValueType> > GetSolver(){return fSolver;}

	virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
	{
		y.Fill(0.);
		fSolver->Solve(y, x);
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

	virtual unsigned int Dimension() const {return fSolver->Dimension();} ;

	virtual const ValueType& operator()(unsigned int /*i*/, unsigned int /*j*/) const
	{
		//This function always returns zero as an implicit (non-stationary)
		//preconditioner cannot easily compute fixed matrix elements of the inverse action
		return fZero;
	}

protected:

	ValueType fZero;

	KSmartPointer<KIterativeKrylovSolver<ValueType> > fSolver;

};

template<typename ValueType>
KSmartPointer< KPreconditioner<ValueType> >
KBuildKrylovPreconditioner(KSmartPointer<KIterativeKrylovSolver<ValueType> > solver){
	return new KImplicitKrylovPreconditioner<ValueType>(solver);
}

template<typename ValueType>
KSmartPointer< KPreconditioner<ValueType> >
KBuildKrylovPreconditioner(
		const KKrylovSolverConfiguration& config,
		KSmartPointer<const KSquareMatrix<ValueType> > matrix,
		KSmartPointer<const KSquareMatrix<ValueType> > preconditioner = NULL)
{
	KSmartPointer< KIterativeKrylovSolver<ValueType > > solver =
			KBuildKrylovSolver(config,matrix,preconditioner);
	return KBuildKrylovPreconditioner<ValueType>(solver);
}

}

#endif /* __KImplicitKrylovPreconditioner_H__ */
