/*
 * KKrylovChargeDensitySolverOldBuilder.hh
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_
#define KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_

#include "KComplexElement.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
#include "KFMElectrostaticParametersBuilder.hh"
#include "KKrylovChargeDensitySolverOld.hh"

namespace katrin
{


typedef KComplexElement<KEMField::KKrylovChargeDensitySolverOld> KKrylovChargeDensitySolverOldBuilder;

template<> inline bool KKrylovChargeDensitySolverOldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "solver_name") {
        std::string name = "";
        aContainer->CopyTo(name);
        fObject->GetSolverConfig()->SetSolverName(name);
        return true;
    }
    if (aContainer->GetName() == "preconditioner") {
        std::string name = "";
        aContainer->CopyTo(name);
        fObject->GetSolverConfig()->SetPreconditionerName(name);
        return true;
    }
    if (aContainer->GetName() == "tolerance") {
        double tolerance = 0.0;
        aContainer->CopyTo(tolerance);
        fObject->GetSolverConfig()->SetSolverTolerance(tolerance);
        return true;
    }
    if (aContainer->GetName() == "max_iterations") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetMaxSolverIterations(value);
        return true;
    }
    if (aContainer->GetName() == "iterations_between_restarts") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetIterationsBetweenRestart(value);
        return true;
    }
    if (aContainer->GetName() == "preconditioner_tolerance") {
        double value = 0.;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetPreconditionerTolerance(value);
        return true;
    }
    if (aContainer->GetName() == "max_preconditioner_iterations") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetMaxPreconditionerIterations(value);
        return true;
    }
    if (aContainer->GetName() == "preconditioner_degree") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetPreconditionerDegree(value);
        return true;
    }
    if (aContainer->GetName() == "intermediate_save_interval") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseCheckpoints(value);
        fObject->GetSolverConfig()->SetCheckpointFrequency(value);
        return true;
    }
    if (aContainer->GetName() == "use_display") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseDisplay(value);
        return true;
    }
    if (aContainer->GetName() == "show_plot") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUsePlot(value);
        return true;
    }
    if (aContainer->GetName() == "use_timer") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseTimer(value);
        return true;
    }
    if (aContainer->GetName() == "time_limit_in_seconds") {
        double value = 0.;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetTimeLimitSeconds(value);
        return true;
    }
    if (aContainer->GetName() == "time_check_interval") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetTimeCheckFrequency(value);
        return true;
    }
    return false;
}

template<> inline bool KKrylovChargeDensitySolverOldBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "fftm_multiplication") {
        anElement->ReleaseTo(fObject->GetSolverConfig(),
                             &KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetFFTMParams);
        return true;
    }
    if (anElement->GetName() == "preconditioner_electrostatic_parameters") {
        anElement->ReleaseTo(
            fObject->GetSolverConfig(),
            &KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetPreconditionerFFTMParams);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_ */
