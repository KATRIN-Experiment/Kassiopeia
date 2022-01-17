/*
 * KGaussSeidelChargeDensitySolver.cc
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#include "KGaussSeidelChargeDensitySolver.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KElectrostaticBoundaryIntegrator.hh"
#include "KGaussSeidel.hh"
#include "KSquareMatrix.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
using KEMField::KMPIInterface;
#include "KGaussSeidel_MPI.hh"
using KEMField::KGaussSeidel_MPI;
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KGaussSeidel_OpenCL.hh"
using KEMField::KGaussSeidel_OpenCL;
#endif

#ifdef KEMFIELD_USE_MPI
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (true)
#endif
#endif

namespace KEMField
{

KGaussSeidelChargeDensitySolver::KGaussSeidelChargeDensitySolver() = default;

KGaussSeidelChargeDensitySolver::~KGaussSeidelChargeDensitySolver()
{
#ifdef KEMFIELD_USE_OPENCL
    if (fUseOpenCL) {
        KOpenCLSurfaceContainer* oclContainer =
            dynamic_cast<KOpenCLSurfaceContainer*>(KOpenCLInterface::GetInstance()->GetActiveData());
        if (oclContainer)
            delete oclContainer;
        oclContainer = NULL;
        KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    }
#endif
}

void KGaussSeidelChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    if (FindSolution(0., container) == false) {
        if (fUseOpenCL) {
#ifdef KEMFIELD_USE_OPENCL
            KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(container);
            KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
            KOpenCLElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateOpenCLIntegrator(*oclContainer)};
            KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(*oclContainer, integrator);
            KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(*oclContainer, integrator);
            KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(*oclContainer, integrator);

            // NOTE: Running with OpenCL+MPI is not supported here!
            KGaussSeidel<KElectrostaticBoundaryIntegrator::ValueType, KGaussSeidel_OpenCL> gaussSeidel;

            gaussSeidel.Solve(A, x, b);

            MPI_SINGLE_PROCESS
            {
                SaveSolution(0., container);
            }
            return;
#endif
        }

        KElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateIntegrator()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(container, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(container, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(container, integrator);

#ifdef KEMFIELD_USE_MPI
        KGaussSeidel<KElectrostaticBoundaryIntegrator::ValueType, KGaussSeidel_MPI> gaussSeidel;
#else
        KGaussSeidel<KElectrostaticBoundaryIntegrator::ValueType> gaussSeidel;
#endif

        gaussSeidel.Solve(A, x, b);

        MPI_SINGLE_PROCESS
        {
            SaveSolution(0., container);
        }
        return;
    }
    return;
}

}  // namespace KEMField
