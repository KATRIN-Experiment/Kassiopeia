/*
 * KKrylovChargeDensitySolver.cc
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovChargeDensitySolver.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryMatrixGenerator.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KKrylovSolverFactory.hh"

namespace KEMField {

KKrylovChargeDensitySolver::KKrylovChargeDensitySolver() {
	fKrylovConfig.SetDisplayName("Solver: ");
}

KKrylovChargeDensitySolver::~KKrylovChargeDensitySolver() { }

void KKrylovChargeDensitySolver::SetMatrixGenerator(
		KSmartPointer<MatrixGenerator> matrixGen)
{
	fMatrixGenerator = matrixGen;
}

KSmartPointer<const KKrylovChargeDensitySolver::MatrixGenerator>
KKrylovChargeDensitySolver::GetMatrixGenerator() const
{
	return fMatrixGenerator;
}

void KKrylovChargeDensitySolver::SetPreconditionerGenerator(
		KSmartPointer<MatrixGenerator> preconGen)
{
	fPreconditionerGenerator = preconGen;
}

void KKrylovChargeDensitySolver::ComputeSolution(KSurfaceContainer& container)
{
    /* Here I assume that the electrostatic vector space basis consists of one
     * ValueType per surface element and that these are arranged in the same order
     * as in the KSurface container. */

	// The integration method should not matter, as we are only using the vectors
	// Therefore, using simply default.
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};

        KSmartPointer<KSquareMatrix<ValueType> > A = fMatrixGenerator->Build(container);
        KSmartPointer<KSquareMatrix<ValueType> > P;
        if(fPreconditionerGenerator.Is())
            P = fPreconditionerGenerator->Build(container);

        KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator > x( container, integrator );
        KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator > b( container, integrator );

        KSmartPointer< KIterativeKrylovSolver<ValueType> > solver =
                KBuildKrylovSolver<ValueType>(fKrylovConfig,A,P);
        solver->Solve(x,b);
        SaveSolution(solver->ResidualNorm(), container);
}

void KKrylovChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
	if(!FindSolution(fKrylovConfig.GetTolerance(),container))
	    ComputeSolution(container);

}

} /* namespace KEMField */
