/*
 * KKrylovChargeDensitySolverOld.cc
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovChargeDensitySolverOld.hh"

#include "KFMElectrostaticFastMultipoleBoundaryValueSolver.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
#include "KFMMessaging.hh"
#include "KEMCoreMessage.hh"

namespace KEMField
{

KKrylovChargeDensitySolverOld::KKrylovChargeDensitySolverOld() :
    fKrylovConfig(new KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration()),
    fSolver(new KFMElectrostaticFastMultipoleBoundaryValueSolver()){};

KKrylovChargeDensitySolverOld::~KKrylovChargeDensitySolverOld()
{
    delete fSolver;
    delete fKrylovConfig;
}

void KKrylovChargeDensitySolverOld::InitializeCore(KSurfaceContainer& surfaceContainer)
{
    if (surfaceContainer.empty()) {
        kem_cout(eWarning) << "Krylov solver got no elctrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    if (fKrylovConfig->GetFFTMParams() == nullptr) {
        kfmout << "ABORTING no multiplication method set for"
                  " krylov bem solver"
               << kfmendl;
        kfmexit(1);
    }
    fSolver->SetSolverElectrostaticParameters(*fKrylovConfig->GetFFTMParams());
    fSolver->SetConfigurationObject(fKrylovConfig);
    if (fKrylovConfig->GetPreconditionerFFTMParams() != nullptr)
        fSolver->SetPreconditionerElectrostaticParameters(*fKrylovConfig->GetPreconditionerFFTMParams());

    kfmout << fSolver->GetParameterInformation() << kfmendl;

    if (!FindSolution(fSolver->GetTolerance(), surfaceContainer)) {
        //solve the boundary value problem
        fSolver->Solve(surfaceContainer);
        SaveSolution(fSolver->GetResidualNorm(), surfaceContainer);
    }
}


} /* namespace KEMField */
