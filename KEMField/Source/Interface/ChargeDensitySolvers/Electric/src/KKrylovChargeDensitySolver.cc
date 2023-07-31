/*
 * KKrylovChargeDensitySolver.cc
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovChargeDensitySolver.hh"

#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryMatrixGenerator.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KKrylovSolverFactory.hh"
#include "KEMCoreMessage.hh"

namespace KEMField
{

KKrylovChargeDensitySolver::KKrylovChargeDensitySolver()
{
    fKrylovConfig.SetDisplayName("Solver: ");
}

KKrylovChargeDensitySolver::~KKrylovChargeDensitySolver() = default;

void KKrylovChargeDensitySolver::SetMatrixGenerator(const std::shared_ptr<MatrixGenerator>& matrixGen)
{
    fMatrixGenerator = matrixGen;
}

std::shared_ptr<const KKrylovChargeDensitySolver::MatrixGenerator> KKrylovChargeDensitySolver::GetMatrixGenerator() const
{
    return fMatrixGenerator;
}

void KKrylovChargeDensitySolver::SetPreconditionerGenerator(const std::shared_ptr<MatrixGenerator>& preconGen)
{
    fPreconditionerGenerator = preconGen;
}

void KKrylovChargeDensitySolver::ComputeSolution(KSurfaceContainer& container)
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: Krylov solver got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    /* Here I assume that the electrostatic vector space basis consists of one
     * ValueType per surface element and that these are arranged in the same order
     * as in the KSurface container. */

    // The integration method should not matter, as we are only using the vectors
    // Therefore, using simply default.
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};

    std::shared_ptr<KSquareMatrix<ValueType>> A = fMatrixGenerator->Build(container);
    std::shared_ptr<KSquareMatrix<ValueType>> P;
    if (fPreconditionerGenerator)
        P = fPreconditionerGenerator->Build(container);

    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(container, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(container, integrator);

    auto solver = KBuildKrylovSolver<ValueType>(fKrylovConfig, A, P);
    solver->Solve(x, b);
    SaveSolution(solver->ResidualNorm(), container);
}

void KKrylovChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: Krylov solver got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    if (!FindSolution(fKrylovConfig.GetTolerance(), container))
        ComputeSolution(container);
}

} /* namespace KEMField */
