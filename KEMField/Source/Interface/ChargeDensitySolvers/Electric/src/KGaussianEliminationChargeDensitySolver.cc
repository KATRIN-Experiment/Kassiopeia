/*
 * KGaussianEliminationChargeDensitySolver.cc
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#include "KGaussianEliminationChargeDensitySolver.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KSquareMatrix.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KGaussianElimination.hh"

namespace KEMField {

KGaussianEliminationChargeDensitySolver::KGaussianEliminationChargeDensitySolver()
{
}

KGaussianEliminationChargeDensitySolver::~KGaussianEliminationChargeDensitySolver()
{
}

void KGaussianEliminationChargeDensitySolver::SetIntegratorPolicy(
		const KEBIPolicy& policy)
{
	fIntegratorPolicy = policy;
}

void KGaussianEliminationChargeDensitySolver::InitializeCore( KSurfaceContainer& container )
{
	if( FindSolution( 0., container ) == false )
	{
		KElectrostaticBoundaryIntegrator integrator {fIntegratorPolicy.CreateIntegrator()};
		KBoundaryIntegralMatrix< KElectrostaticBoundaryIntegrator > A( container, integrator );
		KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator > x( container, integrator );
		KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator > b( container, integrator );

		KGaussianElimination< KElectrostaticBoundaryIntegrator::ValueType > gaussianElimination;
		gaussianElimination.Solve( A, x, b );

		SaveSolution( 0., container );
	}
	return;
}

} // KEMField


