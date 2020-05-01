/*
 * KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.cc
 *
 *  Created on: 27.04.2015
 *      Author: gosda
 */
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"

#include "KFMMessaging.hh"

namespace KEMField
{
KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::
    ~KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration()
{
    if (fFFTMParams != nullptr)
        delete fFFTMParams;
    if (fPreconditionerFFTMParams != nullptr)
        delete fPreconditionerFFTMParams;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetFFTMParams(KFMElectrostaticParameters* config)
{
    if (fFFTMParams != nullptr) {
        kfmout << "ABORTING tried to assign more than"
                  " one set of fftm configurations to the"
                  " FastMultipoleBEMSolver "
               << kfmendl;
        kfmexit(1);
    }
    fFFTMParams = config;
}

void KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetPreconditionerFFTMParams(
    KFMElectrostaticParameters* config)
{
    if (fPreconditionerFFTMParams != nullptr) {
        kfmout << "ABORTING tried to assign more than"
                  " one set of fftm configurations to the"
                  " FastMultipoleBEMSolver "
               << kfmendl;
        kfmexit(1);
    }
    fPreconditionerFFTMParams = config;
}

}  // namespace KEMField
