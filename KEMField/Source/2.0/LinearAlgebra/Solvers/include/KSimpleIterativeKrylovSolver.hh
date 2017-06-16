/*
 * KSimpleIterativeKrylovSolver.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_LINEARALGEBRA_SOLVERS_INCLUDE_KSIMPLEITERATIVEKRYLOVSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_LINEARALGEBRA_SOLVERS_INCLUDE_KSIMPLEITERATIVEKRYLOVSOLVER_HH_

#include "KIterativeKrylovSolver.hh"

namespace KEMField {

template <typename ValueType, template <typename> class ParallelTrait>
class KSimpleIterativeKrylovSolver: public KIterativeKrylovSolver<ValueType>
{
public:
	using typename KIterativeKrylovSolver<ValueType>::Matrix;
	using typename KIterativeKrylovSolver<ValueType>::Vector;


	KSimpleIterativeKrylovSolver();
	virtual ~KSimpleIterativeKrylovSolver();

	virtual void Solve(const Matrix& A,Vector& x,const Vector& b);

	//set tolerancing (default is relative tolerance on l2 residual norm)
	virtual void SetTolerance(double d) { SetRelativeTolerance(d); };
	virtual void SetAbsoluteTolerance(double d){ this->fTolerance = d; fUseRelativeTolerance = false;};
	virtual void SetRelativeTolerance(double d){ this->fTolerance = d; fUseRelativeTolerance = true;};

	virtual double ResidualNorm() const
	{
		if(fUseRelativeTolerance)
		{
			return (this->fResidualNorm)/fInitialResidual;
		}
		else
		{
			return this->fResidualNorm;
		}
	}

	void CoalesceData() { if (fTrait) fTrait->CoalesceData(); }

	ParallelTrait<ValueType>* GetTrait() {return fTrait;};

private:
	virtual void SolveCore(Vector& x, const Vector& b);

	unsigned int Dimension() const { return (fTrait ? fTrait->Dimension() : 0); }
	void SetResidualVector(const Vector& v) { if (fTrait) fTrait->SetResidualVector(v); }
	void GetResidualVector(Vector& v) { if (fTrait) fTrait->GetResidualVector(v); }

	virtual bool HasConverged() const
	{
		if(fUseRelativeTolerance)
		{
			double relative_residual_norm = (this->fResidualNorm)/(fInitialResidual);
			if( relative_residual_norm > this->fTolerance ){return false;}
			else{return true;}
		}
		else
		{
			if( this->fResidualNorm > this->fTolerance ){ return false; }
			else{ return true; }
		}
	}

	double InnerProduct(const Vector& a, const Vector& b) const
	{
		double result = 0.;

		unsigned int dim = a.Dimension();
		for(unsigned int i=0; i<dim; i++)
		{
			result += a(i)*b(i);
		}

		return result;
	}

	ParallelTrait<ValueType>* fTrait;
	bool fUseRelativeTolerance;
	double fInitialResidual;
};


template <typename ValueType,template <typename> class ParallelTrait>
KSimpleIterativeKrylovSolver<ValueType,ParallelTrait>::KSimpleIterativeKrylovSolver():
fTrait(NULL)
{
	fUseRelativeTolerance = false;
	fInitialResidual = 1.0;
}

template <typename ValueType,template <typename> class ParallelTrait>
KSimpleIterativeKrylovSolver<ValueType,ParallelTrait>::~KSimpleIterativeKrylovSolver()
{
}

template< typename ValueType,template <typename> class ParallelTrait>
void KSimpleIterativeKrylovSolver<ValueType,ParallelTrait>::SolveCore(Vector& x,const Vector& b)
{
	Solve(*(this->GetMatrix()),x,b);
}

template <typename ValueType,template <typename> class ParallelTrait>
void KSimpleIterativeKrylovSolver<ValueType,ParallelTrait>::Solve(const Matrix& A, Vector& x, const Vector& b)
{
	ParallelTrait<ValueType> trait(A,x,b);
	fTrait = &trait;

	this->InitializeVisitors();
	this->fIteration = 0;

	trait.Initialize();

	//needed if we are using relative tolerance as convergence condition
			fInitialResidual = std::sqrt(InnerProduct(b,b));

	bool solutionUpdated = false;

	do
	{
		solutionUpdated = false;
		trait.AugmentKrylovSubspace();
		trait.GetResidualNorm(this->fResidualNorm);
		this->GetRestartCondition()->UpdateProgress(this->fResidualNorm);
		this->fIteration++;

		if( this->GetRestartCondition()->PerformRestart() )
		{
			trait.UpdateSolution();
			trait.ResetAndInitialize(); //clears krylov subspace vectors, restarts from current solution
		}
		else if( HasConverged() || (this->fIteration >= this->GetMaximumIterations()) )
		{
			trait.UpdateSolution();
			solutionUpdated = true;
			break;
		}
		else if( this->Terminate() )
		{
			trait.UpdateSolution();
			solutionUpdated = true;
			break;
		}

		this->AcceptVisitors();
	}
	while( !( HasConverged() ) && (this->fIteration < this->GetMaximumIterations()) && !(this->Terminate()) );

	if(!solutionUpdated){trait.UpdateSolution();};
	trait.Finalize();

	this->FinalizeVisitors();

	fTrait = NULL;
}

} /* namespace KEMField */



#endif /* KEMFIELD_SOURCE_2_0_LINEARALGEBRA_SOLVERS_INCLUDE_KSIMPLEITERATIVEKRYLOVSOLVER_HH_ */
