/*
 * KGaussianEliminationChargeDensitySolver.cc
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#include "KGaussianEliminationChargeDensitySolver.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KGaussianElimination.hh"
#include "KSquareMatrix.hh"

namespace KEMField
{

KGaussianEliminationChargeDensitySolver::KGaussianEliminationChargeDensitySolver() = default;

KGaussianEliminationChargeDensitySolver::~KGaussianEliminationChargeDensitySolver() = default;

void KGaussianEliminationChargeDensitySolver::SetIntegratorPolicy(const KEBIPolicy& policy)
{
    fIntegratorPolicy = policy;
}

void KGaussianEliminationChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    if (FindSolution(0., container) == false) {
        KElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateIntegrator()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(container, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(container, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(container, integrator);

        KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
        gaussianElimination.Solve(A, x, b);

        SaveSolution(0., container);
    }
    return;
}

}  // namespace KEMField
