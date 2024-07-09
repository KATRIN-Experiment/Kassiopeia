/*
 * KKrylovSolverConfigurationReader.hh
 *
 *  Created on: 20 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVSOLVERCONFIGURATIONREADER_HH_
#define KKRYLOVSOLVERCONFIGURATIONREADER_HH_

#include "KEMStringUtils.hh"
#include "KKrylovChargeDensitySolver.hh"

namespace katrin
{

template<class ObjectType> bool SetKrylovSolverMatrixAndPrecon(KContainer& anElement, ObjectType& fObject)
{
    if (anElement.Is<KEMField::KKrylovChargeDensitySolver::MatrixGenerator>()) {
        std::shared_ptr<KEMField::KKrylovChargeDensitySolver::MatrixGenerator> matrixGenerator;
        anElement.ReleaseTo(matrixGenerator);
        if (KEMField::endsWith(anElement.GetName(), "preconditioner"))
            fObject.SetPreconditionerGenerator(matrixGenerator);
        else
            fObject.SetMatrixGenerator(matrixGenerator);
        return true;
    }
    return false;
}

template<class ObjectType> bool SetKrylovSolverConfiguration(KContainer& aContainer, ObjectType& fObject)
{
    if (aContainer.GetName() == "solver_name") {
        std::string name = "";
        aContainer.CopyTo(name);
        fObject.SetSolverName(name);
        return true;
    }
    if (aContainer.GetName() == "tolerance") {
        double tolerance = 0.0;
        aContainer.CopyTo(tolerance);
        fObject.SetTolerance(tolerance);
        return true;
    }
    if (aContainer.GetName() == "max_iterations") {
        unsigned int value = 0;
        aContainer.CopyTo(value);
        fObject.SetMaxIterations(value);
        return true;
    }
    if (aContainer.GetName() == "iterations_between_restarts") {
        unsigned int value = 0;
        aContainer.CopyTo(value);
        fObject.SetIterationsBetweenRestart(value);
        return true;
    }
    if (aContainer.GetName() == "intermediate_save_interval") {
        unsigned int value = 0;
        aContainer.CopyTo(value);
        fObject.SetUseCheckpoints(value);
        fObject.SetStepsBetweenCheckpoints(value);
        return true;
    }
    if (aContainer.GetName() == "use_display") {
        bool value = false;
        aContainer.CopyTo(value);
        fObject.SetUseDisplay(value);
        return true;
    }
    if (aContainer.GetName() == "show_plot") {
        bool value = false;
        aContainer.CopyTo(value);
        fObject.SetUsePlot(value);
        return true;
    }
    if (aContainer.GetName() == "use_timer") {
        bool value = false;
        aContainer.CopyTo(value);
        fObject.SetUseTimer(value);
        return true;
    }
    if (aContainer.GetName() == "time_limit_in_seconds") {
        double value = 0.0;
        aContainer.CopyTo(value);
        fObject.SetTimeLimitSeconds(value);
        return true;
    }
    if (aContainer.GetName() == "time_check_interval") {
        unsigned int value = 0;
        aContainer.CopyTo(value);
        fObject.SetStepsBetweenTimeChecks(value);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVSOLVERCONFIGURATIONREADER_HH_ */
